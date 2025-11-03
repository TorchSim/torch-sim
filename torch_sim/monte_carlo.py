"""Propagators for Monte Carlo simulations.

This module provides functionality for performing Monte Carlo simulations,
particularly focused on swap Monte Carlo for atomic systems. It includes
implementations of the Metropolis criterion, swap generation, and utility
functions for handling permutations in batched systems.

The `swap_mc_init` and `swap_mc_step` functions can be used
with `integrate` but if a trajectory is being reported, the
`TorchSimTrajectory.write_state` method must be called with `variable_masses=True`.

Examples:
    >>> import torch_sim as ts
    >>> mc_state = ts.swap_mc_init(model, initial_state, seed=42)
    >>> for _ in range(1000):
    ...     mc_state = ts.swap_mc_step(model, mc_state, kT=0.1 * units.energy)
"""

from dataclasses import dataclass

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


@dataclass(kw_only=True)
class SwapMCState(SimState):
    """State for Monte Carlo simulations with swap moves.

    This class extends the SimState to include properties specific to Monte Carlo
    simulations, such as the system energy and records of permutations applied
    during the simulation.

    Attributes:
        energy (torch.Tensor): Energy of the system with shape [batch_size]
        last_permutation (torch.Tensor): Last permutation applied to the system,
            with shape [n_atoms], tracking the moves made for analysis or reversal
    """

    energy: torch.Tensor
    last_permutation: torch.Tensor

    _atom_attributes = SimState._atom_attributes | {"last_permutation"}  # noqa: SLF001
    _system_attributes = SimState._system_attributes | {"energy"}  # noqa: SLF001


def generate_swaps(state: SimState, rng: torch.Generator | None = None) -> torch.Tensor:
    """Generate atom swaps for a given batched system.

    Generates proposed swaps between atoms of different types within the same system.
    The function ensures that swaps only occur between atoms with different atomic
    numbers.

    Args:
        state (SimState): The simulation state
        rng (torch.Generator | None, optional): Random number generator for
            reproducibility. Defaults to None.

    Returns:
        torch.Tensor: A tensor of proposed swaps with shape [n_systems, 2],
            where each row contains indices of atoms to be swapped
    """
    system = state.system_idx
    atomic_numbers = state.atomic_numbers

    system_lengths = system.bincount()

    # change system_lengths to system
    system = torch.repeat_interleave(
        torch.arange(len(system_lengths), device=system.device), system_lengths
    )

    # Create ragged weights tensor without loops
    max_length = torch.max(system_lengths).item()
    n_systems = len(system_lengths)

    # Create a range tensor for each system
    range_tensor = torch.arange(max_length, device=system.device).expand(
        n_systems, max_length
    )

    # Create a mask where values are less than the max system length
    system_lengths_expanded = system_lengths.unsqueeze(1).expand(n_systems, max_length)
    weights = (range_tensor < system_lengths_expanded).float()

    first_index = torch.multinomial(weights, 1, replacement=False, generator=rng)

    # Process each system - we need this loop because of ragged systems
    system_starts = system_lengths.cumsum(dim=0) - system_lengths[0]

    for sys_idx in range(n_systems):
        # Get global index of selected atom
        first_idx = first_index[sys_idx, 0].item() + system_starts[sys_idx].item()
        first_type = atomic_numbers[first_idx]

        # Get indices of atoms in this system
        system_start = system_starts[sys_idx].item()
        system_end = system_start + system_lengths[sys_idx].item()

        # Create mask for same-type atoms
        same_type = atomic_numbers[system_start:system_end] == first_type

        # Zero out weights for same-type atoms (accounting for padding)
        weights[sys_idx, : len(same_type)][same_type] = 0.0

    second_index = torch.multinomial(weights, 1, replacement=False, generator=rng)
    zeroed_swaps = torch.concatenate([first_index, second_index], dim=1)

    return zeroed_swaps + (system_lengths.cumsum(dim=0) - system_lengths[0]).unsqueeze(1)


def swaps_to_permutation(swaps: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Convert atom swap pairs to a full permutation tensor.

    Creates a permutation tensor that represents the result of applying the specified
    swaps to the system.

    Args:
        swaps (torch.Tensor): Tensor of shape [n_swaps, 2] containing pairs of indices
            to swap
        n_atoms (int): Total number of atoms in the system

    Returns:
        torch.Tensor: Permutation tensor of shape [n_atoms] where permutation[i]
            contains the index of the atom that should be moved to position i
    """
    permutation = torch.arange(n_atoms, device=swaps.device)

    for swap in swaps:
        idx1, idx2 = swap
        temp = permutation[idx1].clone()
        permutation[idx1] = permutation[idx2]
        permutation[idx2] = temp

    return permutation


def metropolis_criterion(
    energy_new: torch.Tensor,
    energy_old: torch.Tensor,
    kT: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the Metropolis acceptance criterion for Monte Carlo moves.

    Determines whether proposed moves should be accepted or rejected based on
    the energy difference and system temperature, following the Boltzmann distribution.

    Args:
        energy_new (torch.Tensor): New energy after proposed move of shape [batch_size]
        energy_old (torch.Tensor): Old energy before proposed move of shape [batch_size]
        kT (float): Temperature of the system in energy units
        rng (torch.Generator | None, optional): Random number generator for
            reproducibility. Defaults to None.

    Returns:
        torch.Tensor: Boolean tensor of shape [batch_size] indicating acceptance (True)
            or rejection (False) for each move

    Notes:
        The acceptance probability follows min(1, exp(-ΔE/kT)) according to the
        standard Metropolis algorithm.
    """
    delta_e = energy_new - energy_old

    # Calculate acceptance probability: min(1, exp(-ΔE/kT))
    p_acceptance = torch.clamp(torch.exp(-delta_e / kT), max=1.0)

    # Generate random numbers between 0 and 1 using the generator
    random_values = torch.rand(
        p_acceptance.shape, generator=rng, device=p_acceptance.device
    )

    # Accept if random value < acceptance probability
    return random_values < p_acceptance


def swap_mc_init(
    state: SimState,
    model: ModelInterface,
) -> SwapMCState:
    """Initialize a swap Monte Carlo state from input data.

    Creates an initial state for swap Monte Carlo simulations by computing initial
    energy and setting up the permutation tracking. The simulation uses the Metropolis
    criterion to accept or reject proposed swaps based on energy differences.

    Make sure that if the trajectory is being reported, the
    `TorchSimTrajectory.write_state` method is called with `variable_masses=True`.

    Args:
        model: Energy model that takes a SimState and returns a dict containing
            'energy' as a key
        state: The simulation state to initialize from

    Returns:
        SwapMCState: Initialized state for swap Monte Carlo simulation containing
            positions, energy, and permutation tracking

    Examples:
        >>> mc_state = swap_monte_carlo_init(model=energy_model, state=initial_state)
        >>> for _ in range(100):
        >>>     mc_state = swap_monte_carlo_step(model, mc_state, kT=0.1)
    """
    model_output = model(state)

    return SwapMCState(
        positions=state.positions,
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers,
        system_idx=state.system_idx,
        energy=model_output["energy"],
        last_permutation=torch.arange(state.n_atoms, device=state.device),
    )


def swap_mc_step(
    state: SwapMCState,
    model: ModelInterface,
    *,
    kT: float,
    seed: int | None = None,
    rng: torch.Generator | None = None,
) -> SwapMCState:
    """Perform a single swap Monte Carlo step.

    Proposes atom swaps, evaluates the energy change, and uses the Metropolis
    criterion to determine whether to accept the move. Rejected moves are reversed.

    Args:
        model: Energy model that takes a SimState and returns a dict containing
            'energy' as a key
        state: The current Monte Carlo state
        kT: Temperature parameter in energy units
        seed: (Deprecated) Seed for the random number generator. If provided and
            `generator` is None, a temporary generator seeded with this value will
            be used.
        rng: Optional torch.Generator to drive all randomness for this step.
            Prefer passing a persistent generator across steps for reproducibility.

    Returns:
        SwapMCState: Updated Monte Carlo state after applying the step

    Notes:
        The function handles batched systems and ensures that swaps only occur
        within the same system.
    """
    # Prefer explicit generator if provided; otherwise build one from seed
    _rng = rng
    if _rng is None and seed is not None:
        _rng = torch.Generator(device=model.device)
        _rng.manual_seed(seed)

    swaps = generate_swaps(state, rng=_rng)

    permutation = swaps_to_permutation(swaps, state.n_atoms)

    if not torch.all(state.system_idx == state.system_idx[permutation]):
        raise ValueError("Swaps must be between atoms in the same system")

    energies_old = state.energy.clone()
    state.positions = state.positions[permutation].clone()

    model_output = model(state)
    energies_new = model_output["energy"]

    accepted = metropolis_criterion(energies_new, energies_old, kT, rng=_rng)
    rejected_swaps = swaps[~accepted]
    reverse_rejected_swaps = swaps_to_permutation(rejected_swaps, state.n_atoms)
    state.positions = state.positions[reverse_rejected_swaps]

    state.energy = torch.where(accepted, energies_new, energies_old)
    state.last_permutation = permutation[reverse_rejected_swaps].clone()

    return state
