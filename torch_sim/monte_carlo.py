"""Propagators for Monte Carlo simulations.

This module provides functionality for performing Monte Carlo simulations,
particularly focused on swap Monte Carlo for atomic systems. It includes
implementations of the Metropolis criterion, swap generation, and utility
functions for handling permutations in batched systems.

The `swap_monte_carlo` function can be used with `integrate` but if
a trajectory is being reported, the `TorchSimTrajectory.write_state` method
must be called with `variable_masses=True`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


@dataclass
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


def generate_swaps(
    state: SimState, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Generate atom swaps for a given batched system.

    Generates proposed swaps between atoms of different types within the same system.
    The function ensures that swaps only occur between atoms with different atomic
    numbers.

    Args:
        state (SimState): The simulation state
        generator (torch.Generator | None, optional): Random number generator for
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

    first_index = torch.multinomial(weights, 1, replacement=False, generator=generator)

    # Process each system - we need this loop because of ragged systems
    system_starts = system_lengths.cumsum(dim=0) - system_lengths[0]

    for b in range(n_systems):
        # Get global index of selected atom
        first_idx = first_index[b, 0].item() + system_starts[b].item()
        first_type = atomic_numbers[first_idx]

        # Get indices of atoms in this system
        system_start = system_starts[b].item()
        system_end = system_start + system_lengths[b].item()

        # Create mask for same-type atoms
        same_type = atomic_numbers[system_start:system_end] == first_type

        # Zero out weights for same-type atoms (accounting for padding)
        weights[b, : len(same_type)][same_type] = 0.0

    second_index = torch.multinomial(weights, 1, replacement=False, generator=generator)
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
    permutation[swaps[:, 0]] = swaps[:, 1]
    permutation[swaps[:, 1]] = swaps[:, 0]
    return permutation


def validate_permutation(permutation: torch.Tensor, system_idx: torch.Tensor) -> None:
    """Validate that permutations only swap atoms within the same system.

    Confirms that no swaps are attempted between atoms in different systems,
    which would lead to physically invalid configurations.

    Args:
        permutation (torch.Tensor): Permutation tensor of shape [n_atoms]
        system_idx (torch.Tensor): system_idx for each atom of shape [n_atoms]

    Raises:
        ValueError: If any swaps are between atoms in different systems
    """
    if not torch.all(system_idx == system_idx[permutation]):
        raise ValueError("Swaps must be between atoms in the same system")


def metropolis_criterion(
    energy_new: torch.Tensor,
    energy_old: torch.Tensor,
    kT: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the Metropolis acceptance criterion for Monte Carlo moves.

    Determines whether proposed moves should be accepted or rejected based on
    the energy difference and system temperature, following the Boltzmann distribution.

    Args:
        energy_new (torch.Tensor): New energy after proposed move of shape [batch_size]
        energy_old (torch.Tensor): Old energy before proposed move of shape [batch_size]
        kT (float): Temperature of the system in energy units
        generator (torch.Generator | None, optional): Random number generator for
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
    p_acceptance = torch.exp(-delta_e / kT)

    # Generate random numbers between 0 and 1 using the generator
    random_values = torch.rand(
        p_acceptance.shape, generator=generator, device=p_acceptance.device
    )

    # Accept if random value < acceptance probability
    return random_values < p_acceptance


def swap_monte_carlo(
    *,
    model: ModelInterface,
    kT: float,
    seed: int | None = None,
) -> tuple[
    Callable[[SimState], SwapMCState],
    Callable[[SwapMCState, float, torch.Generator | None], SwapMCState],
]:
    """Initialize a swap Monte Carlo simulation for atomic structure optimization.

    Creates and returns functions for initializing the Monte Carlo state and performing
    Monte Carlo steps. The simulation uses the Metropolis criterion to accept or reject
    proposed swaps based on energy differences.

    Make sure that if the trajectory is being reported, the
    `TorchSimTrajectory.write_state` method is called with `variable_masses=True`.

    Args:
        model (torch.nn.Module): Energy model that takes a SimState and returns a dict
            containing 'energy' as a key
        kT (float): Temperature of the system in energy units
        seed (int | None, optional): Seed for the random number generator.
            Defaults to None.

    Returns:
        tuple: A tuple containing:
            - init_function (Callable): Function to initialize a SwapMCState from a
              SimState
            - step_function (Callable): Function to perform a single Monte Carlo step

    Examples:
        >>> init_fn, step_fn = swap_monte_carlo(model=energy_model, kT=0.1, seed=42)
        >>> mc_state = init_fn(initial_state)
        >>> for _ in range(100):
        >>>     mc_state = step_fn(mc_state)
    """
    if seed is not None:
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed)
    else:
        generator = None

    def init_swap_mc_state(state: SimState) -> SwapMCState:
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

    def swap_monte_carlo_step(
        state: SwapMCState,
        kT: float = kT,
        generator: torch.Generator | None = generator,
    ) -> SwapMCState:
        """Perform a single swap Monte Carlo step.

        Proposes atom swaps, evaluates the energy change, and uses the Metropolis
        criterion to determine whether to accept the move. Rejected moves are reversed.

        Args:
            state (SwapMCState): The current Monte Carlo state
            kT (float, optional): Temperature parameter in energy units. Defaults to the
                value specified in the outer function.
            generator (torch.Generator | None, optional): Random number generator.
                Defaults to None.

        Returns:
            SwapMCState: Updated Monte Carlo state after applying the step

        Notes:
            The function handles batched systems and ensures that swaps only occur
            within the same system.
        """
        swaps = generate_swaps(state, generator=generator)

        permutation = swaps_to_permutation(swaps, state.n_atoms)
        validate_permutation(permutation, state.system_idx)

        energies_old = state.energy.clone()
        state.positions = state.positions[permutation].clone()

        model_output = model(state)
        energies_new = model_output["energy"]

        accepted = metropolis_criterion(
            energies_new, energies_old, kT, generator=generator
        )
        rejected_swaps = swaps[~accepted]
        reverse_rejected_swaps = swaps_to_permutation(rejected_swaps, state.n_atoms)
        state.positions = state.positions[reverse_rejected_swaps]

        state.energy = torch.where(accepted, energies_new, energies_old)
        state.last_permutation = permutation[reverse_rejected_swaps].clone()

        return state

    return init_swap_mc_state, swap_monte_carlo_step
