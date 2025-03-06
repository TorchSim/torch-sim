import pytest
import torch
from pymatgen.core import Structure

from torchsim.monte_carlo import (
    SwapMCState,
    generate_swaps,
    swap_monte_carlo,
    swaps_to_permutation,
    validate_permutation,
)
from torchsim.runners import structures_to_state
from torchsim.state import BaseState


@pytest.fixture
def diverse_structure() -> Structure:
    lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
    species = ["H", "He", "Li", "Be", "B", "C", "N", "O"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def generator(device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    return generator


@pytest.fixture
def batched_diverse_state(
    diverse_structure: Structure, device: torch.device
) -> BaseState:
    return structures_to_state(
        [diverse_structure] * 2, device=device, dtype=torch.float64
    )


def test_generate_permutation(
    batched_diverse_state: BaseState, generator: torch.Generator
):
    swaps = generate_swaps(batched_diverse_state, generator=generator)
    permutation = swaps_to_permutation(swaps, batched_diverse_state.n_atoms)
    validate_permutation(permutation, batched_diverse_state.batch)


def test_generate_swaps(batched_diverse_state: BaseState, generator: torch.Generator):
    swaps = generate_swaps(batched_diverse_state, generator=generator)

    # Check shape and type
    assert isinstance(swaps, torch.Tensor)
    assert swaps.shape[1] == 2

    # Check swaps are within valid range
    assert torch.all(swaps >= 0)
    assert torch.all(swaps < batched_diverse_state.n_atoms)

    # Check swaps are within same batch
    batch = batched_diverse_state.batch
    assert torch.all(batch[swaps[:, 0]] == batch[swaps[:, 1]])


def test_swaps_to_permutation(
    batched_diverse_state: BaseState, generator: torch.Generator
):
    swaps = generate_swaps(batched_diverse_state, generator=generator)
    n_atoms = batched_diverse_state.n_atoms
    permutation = swaps_to_permutation(swaps, n_atoms)

    # Check shape and type
    assert isinstance(permutation, torch.Tensor)
    assert permutation.shape == (n_atoms,)

    # Check permutation contains all indices
    assert torch.sort(permutation)[0].equal(
        torch.arange(n_atoms, device=permutation.device)
    )

    # Check swapped pairs
    for i, j in swaps:
        assert permutation[i] == j
        assert permutation[j] == i


def test_validate_permutation(batched_diverse_state: BaseState):
    # Valid permutation
    swaps = generate_swaps(batched_diverse_state)
    permutation = swaps_to_permutation(swaps, batched_diverse_state.n_atoms)
    validate_permutation(permutation, batched_diverse_state.batch)  # Should not raise

    # Invalid permutation (swap between batches)
    invalid_perm = permutation.clone()
    if batched_diverse_state.n_atoms > 2:
        # Swap first atom with last atom (different batches)
        invalid_perm[0] = batched_diverse_state.n_atoms - 1
        invalid_perm[batched_diverse_state.n_atoms - 1] = 0

        with pytest.raises(ValueError, match="Swaps must be between"):
            validate_permutation(invalid_perm, batched_diverse_state.batch)


def test_monte_carlo(
    batched_diverse_state: BaseState,
    lj_calculator: torch.nn.Module,
):
    """Test the monte_carlo function that returns a step function and initial state."""

    # Call monte_carlo to get the initial state and step function
    init_state_fn, monte_carlo_step_fn = swap_monte_carlo(
        model=lj_calculator, kT=1.0, seed=42
    )
    initial_state = init_state_fn(batched_diverse_state)

    # Verify the returned values
    assert isinstance(initial_state, SwapMCState)
    assert callable(monte_carlo_step_fn)

    # Verify the initial state has the expected attributes
    assert hasattr(initial_state, "energy")
    assert hasattr(initial_state, "last_permutation")

    # Make a copy of the initial state for comparison
    initial_positions = initial_state.positions.clone()

    # Get the current state
    current_state = initial_state

    # Run multiple Monte Carlo steps
    n_steps = 5
    for i in range(n_steps):
        # Create a new generator for each step
        step_generator = torch.Generator(device=batched_diverse_state.device)
        step_generator.manual_seed(42 + i + 1)  # Different seed for each step

        # Run a Monte Carlo step
        current_state = monte_carlo_step_fn(current_state, generator=step_generator)

        # Verify the state is an MCState
        assert isinstance(current_state, SwapMCState)

    # Verify the state has changed after multiple steps
    assert not torch.allclose(current_state.positions, initial_positions)

    # Verify batch assignments remain unchanged
    assert torch.all(current_state.batch == batched_diverse_state.batch)

    # Verify atomic numbers distribution remains the same per batch
    for batch_idx in torch.unique(current_state.batch):
        batch_mask_orig = batched_diverse_state.batch == batch_idx
        batch_mask_result = current_state.batch == batch_idx

        orig_counts = torch.bincount(
            batched_diverse_state.atomic_numbers[batch_mask_orig]
        )
        result_counts = torch.bincount(current_state.atomic_numbers[batch_mask_result])

        assert torch.all(orig_counts == result_counts)
