"""Monte Carlo simulations."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_sim.state import SimState


@dataclass
class SwapMCState(SimState):
    """State for Monte Carlo simulations.

    Attributes:
        energy: Energy of the system
        last_permutation: Last permutation applied, we track this so
            we can have a record of the moves made
    """

    energy: torch.Tensor
    # TODO: change back to last_swap? clearer + more efficient
    # TODO: a little less general though because only one move is tracked
    last_permutation: torch.Tensor


def generate_swaps(
    state: SimState, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Generate swaps for a given batch."""
    # TODO: add atomic numbers and a way to guarantee swaps
    # between atoms of different atomic numbers
    batch = state.batch
    atomic_numbers = state.atomic_numbers

    batch_lengths = batch.bincount()

    # change batch_lengths to batch
    batch = torch.repeat_interleave(
        torch.arange(len(batch_lengths), device=batch.device), batch_lengths
    )

    # Create ragged weights tensor without loops
    max_length = torch.max(batch_lengths).item()
    n_batches = len(batch_lengths)

    # Create a range tensor for each batch
    range_tensor = torch.arange(max_length, device=batch.device).expand(
        n_batches, max_length
    )

    # Create a mask where values are less than the batch length
    batch_lengths_expanded = batch_lengths.unsqueeze(1).expand(n_batches, max_length)
    weights = (range_tensor < batch_lengths_expanded).float()

    first_index = torch.multinomial(weights, 1, replacement=False, generator=generator)

    # Process each batch - we need this loop because of ragged batches
    batch_starts = batch_lengths.cumsum(dim=0) - batch_lengths[0]

    for b in range(n_batches):
        # Get global index of selected atom
        first_idx = first_index[b, 0].item() + batch_starts[b].item()
        first_type = atomic_numbers[first_idx]

        # Get indices of atoms in this batch
        batch_start = batch_starts[b].item()
        batch_end = batch_start + batch_lengths[b].item()

        # Create mask for same-type atoms
        same_type = atomic_numbers[batch_start:batch_end] == first_type

        # Zero out weights for same-type atoms (accounting for padding)
        weights[b, : len(same_type)][same_type] = 0.0

    second_index = torch.multinomial(weights, 1, replacement=False, generator=generator)
    zeroed_swaps = torch.concatenate([first_index, second_index], dim=1)

    return zeroed_swaps + (batch_lengths.cumsum(dim=0) - batch_lengths[0]).unsqueeze(1)


def swaps_to_permutation(swaps: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Convert swaps to a permutation."""
    permutation = torch.arange(n_atoms, device=swaps.device)
    permutation[swaps[:, 0]] = swaps[:, 1]
    permutation[swaps[:, 1]] = swaps[:, 0]
    return permutation


def validate_permutation(permutation: torch.Tensor, batch: torch.Tensor) -> None:
    """Validate that no swaps are attempted between atoms in different batches.

    Args:
        permutation: A tensor of shape (n_atoms,) with indices swapped
        batch: The batch of the atoms
    """
    if not torch.all(batch == batch[permutation]):
        raise ValueError("Swaps must be between atoms in the same batch")


def metropolis_criterion(
    energy_new: torch.Tensor,
    energy_old: torch.Tensor,
    kT: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perform a Metropolis criterion using PyTorch.

    Args:
        energy_new: New energy after proposed move
        energy_old: Old energy before proposed move
        kT: Temperature of the system (energy units)
        generator: Generator for random numbers

    Returns:
        Boolean tensor indicating acceptance (True) or rejection (False) for each move
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
    model: torch.nn.Module,
    kT: float,
    seed: int | None = None,
) -> tuple[SwapMCState, Callable[[SwapMCState, torch.Tensor], SwapMCState]]:
    """Swap Monte Carlo simulation.

    Args:
        state: The state to perform the Monte Carlo simulation on
        model: The model to use for the Monte Carlo simulation
        kT: The temperature of the system
        seed: The seed for the random number generator

    Returns:
        A tuple containing the initial state and the step function
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
            batch=state.batch,
            energy=model_output["energy"],
            last_permutation=torch.arange(state.n_atoms, device=state.device),
        )

    def swap_monte_carlo_step(
        state: SwapMCState,
        kT: float = kT,
        generator: torch.Generator | None = None,
    ) -> SwapMCState:
        """Perform a single swap step.

        Args:
            state: The state to perform the swap step on
            kT: The temperature of the system
            generator: The generator for the random number generator

        Returns:
            The state after the swap step
        """
        swaps = generate_swaps(state, generator=generator)

        permutation = swaps_to_permutation(swaps, state.n_atoms)
        validate_permutation(permutation, state.batch)

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
