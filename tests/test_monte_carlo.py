import pytest
import torch
from pymatgen.core import Structure

import torch_sim as ts
from tests.conftest import DEVICE
from torch_sim.models.interface import ModelInterface
from torch_sim.monte_carlo import (
    SwapMCState,
    generate_swaps,
    metropolis_criterion,
    swap_mc_init,
    swap_mc_step,
    swaps_to_permutation,
)


@pytest.fixture
def batched_diverse_state() -> ts.SimState:
    """Create a batched state with diverse atomic species for testing."""
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
    structure = Structure(lattice, species, coords)
    return ts.io.structures_to_state([structure] * 2, device=DEVICE, dtype=torch.float64)


@pytest.mark.parametrize("use_generator", [True, False])
def test_generate_swaps(batched_diverse_state: ts.SimState, *, use_generator: bool):
    """Test swap generation with and without generator."""
    rng = torch.Generator(device=DEVICE) if use_generator else None
    if rng:
        rng.manual_seed(42)

    swaps = generate_swaps(batched_diverse_state, rng=rng)

    # Basic validation
    assert isinstance(swaps, torch.Tensor)
    assert swaps.shape[1] == 2
    assert torch.all(swaps >= 0)
    assert torch.all(swaps < batched_diverse_state.n_atoms)

    # System consistency
    system_idx = batched_diverse_state.system_idx
    assert torch.all(system_idx[swaps[:, 0]] == system_idx[swaps[:, 1]])

    # Different atomic numbers
    atomic_numbers = batched_diverse_state.atomic_numbers
    for swap in swaps:
        assert atomic_numbers[swap[0]] != atomic_numbers[swap[1]]

    # Test reproducibility with generator
    if use_generator and rng is not None:
        rng.manual_seed(42)
        swaps2 = generate_swaps(batched_diverse_state, rng=rng)
        assert torch.equal(swaps, swaps2)


@pytest.mark.parametrize("n_swaps", [0, 1, 3])
def test_swaps_to_permutation(batched_diverse_state: ts.SimState, *, n_swaps: int):
    """Test permutation generation with different numbers of swaps."""
    n_atoms = batched_diverse_state.n_atoms
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(42)

    if n_swaps == 0:
        combined_swaps = torch.empty((0, 2), dtype=torch.long, device=DEVICE)
    else:
        all_swaps = [
            generate_swaps(batched_diverse_state, rng=rng) for _ in range(n_swaps)
        ]
        combined_swaps = torch.cat(all_swaps, dim=0)

    permutation = swaps_to_permutation(combined_swaps, n_atoms)

    # Validation
    assert isinstance(permutation, torch.Tensor)
    assert permutation.shape == (n_atoms,)
    expected_range = torch.arange(n_atoms, device=permutation.device)
    assert torch.sort(permutation)[0].equal(expected_range)

    # Test permutation preserves system assignments
    original_system = batched_diverse_state.system_idx
    assert torch.all(original_system == original_system[permutation])


@pytest.mark.parametrize(
    ("energy_old", "energy_new", "kT", "expected_rate"),
    [
        ([10.0, 20.0], [5.0, 15.0], 1.0, 1.0),  # Energy decreases
        ([5.0, 15.0], [25.0, 35.0], 0.1, 0.0),  # Energy increases significantly
        ([10.0, 20.0], [10.0, 20.0], 1.0, 1.0),  # Energy stays same
        ([10.0, 20.0], [15.0, 25.0], 1000.0, 1.0),  # Very high temperature
        ([10.0, 20.0], [15.0, 25.0], 0.001, 0.0),  # Very low temperature
    ],
)
def test_metropolis_criterion(
    *,
    energy_old: list[float],
    energy_new: list[float],
    kT: float,
    expected_rate: float,
):
    """Test metropolis criterion with different energy scenarios."""
    energy_old_tensor = torch.tensor(energy_old, device=DEVICE)
    energy_new_tensor = torch.tensor(energy_new, device=DEVICE)

    if expected_rate in [0.0, 1.0]:
        # Deterministic cases
        accepted = metropolis_criterion(energy_new_tensor, energy_old_tensor, kT)
        actual_rate = accepted.float().mean().item()
        assert abs(actual_rate - expected_rate) < 0.1
    else:
        # Statistical test
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(42)
        total_accepted = sum(
            metropolis_criterion(energy_new_tensor, energy_old_tensor, kT, rng=rng)
            .sum()
            .item()
            for _ in range(1000)
        )
        actual_rate = total_accepted / (1000 * len(energy_old))
        assert abs(actual_rate - expected_rate) < 0.15


def test_metropolis_criterion_randomness():
    """Test that different generators produce different results."""
    energy_old = torch.tensor([10.0, 20.0], device=DEVICE)
    energy_new = torch.tensor([11.0, 21.0], device=DEVICE)  # ~37% acceptance

    rng1 = torch.Generator(device=DEVICE)
    rng1.manual_seed(42)
    rng2 = torch.Generator(device=DEVICE)
    rng2.manual_seed(43)

    accepted1 = metropolis_criterion(energy_new, energy_old, kT=1.0, rng=rng1)
    accepted2 = metropolis_criterion(energy_new, energy_old, kT=1.0, rng=rng2)
    accepted3 = metropolis_criterion(energy_new, energy_old, kT=1.0, rng=None)

    different_results = not torch.equal(accepted1, accepted2) or not torch.equal(
        accepted1, accepted3
    )
    assert different_results


@pytest.mark.parametrize(("kT", "n_steps"), [(0.1, 3), (1.0, 5), (10.0, 2)])
def test_monte_carlo_integration(
    batched_diverse_state: ts.SimState,
    lj_model: ModelInterface,
    *,
    kT: float,
    n_steps: int,
):
    """Test the complete Monte Carlo workflow."""
    # Initialize
    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(42)
    mc_state = swap_mc_init(state=batched_diverse_state, model=lj_model)
    assert isinstance(mc_state, SwapMCState)
    assert mc_state.energy.shape == (batched_diverse_state.n_systems,)
    assert mc_state.last_permutation.shape == (batched_diverse_state.n_atoms,)
    expected_identity = torch.arange(batched_diverse_state.n_atoms, device=DEVICE)
    assert torch.equal(mc_state.last_permutation, expected_identity)

    # Run steps
    for _step in range(n_steps):
        mc_state = swap_mc_step(state=mc_state, model=lj_model, kT=kT, rng=rng)
        assert isinstance(mc_state, SwapMCState)

    # Verify conservation properties
    assert torch.all(mc_state.system_idx == batched_diverse_state.system_idx)
    for sys_idx in torch.unique(mc_state.system_idx):
        orig_mask = batched_diverse_state.system_idx == sys_idx
        result_mask = mc_state.system_idx == sys_idx
        orig_counts = torch.bincount(batched_diverse_state.atomic_numbers[orig_mask])
        result_counts = torch.bincount(mc_state.atomic_numbers[result_mask])
        assert torch.all(orig_counts == result_counts)


def test_swap_mc_state_attributes():
    """Test SwapMCState class structure and inheritance."""
    from torch_sim.state import SimState

    assert issubclass(SwapMCState, SimState)
    assert "last_permutation" in SwapMCState._atom_attributes  # noqa: SLF001
    assert "energy" in SwapMCState._system_attributes  # noqa: SLF001
    atom_attrs = SwapMCState._atom_attributes  # noqa: SLF001
    system_attrs = SwapMCState._system_attributes  # noqa: SLF001
    parent_atom_attrs = SimState._atom_attributes  # noqa: SLF001
    parent_system_attrs = SimState._system_attributes  # noqa: SLF001
    assert atom_attrs >= parent_atom_attrs
    assert system_attrs >= parent_system_attrs


def test_generate_swaps_ragged_systems():
    """
    Test that generate_swaps works with multiple systems with different atom counts.

    This ensures that we are properly calculating the system_starts for each system.
    """
    s1 = Structure(torch.eye(3), ["H", "He"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s2 = Structure(
        torch.eye(3),
        ["Li", "Be", "B"],
        [[0, 0, 0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]],
    )

    # Combine into a single batched state
    ragged_state = ts.io.structures_to_state([s1, s2], device=DEVICE, dtype=torch.float64)

    # Run multiple times to ensure the RNG hits the out-of-bounds indices
    rng = torch.Generator(device=DEVICE)
    _ = rng.manual_seed(42)

    for _ in range(10):
        swaps = generate_swaps(ragged_state, rng=rng)

        # Check that indices are within bounds
        assert torch.all(swaps < ragged_state.n_atoms), (
            f"Swap indices {swaps.max()} exceed total n_atoms {ragged_state.n_atoms}"
        )

        # Check that swapped atoms belong to the same system
        sys_idx = ragged_state.system_idx
        sys_0 = sys_idx[swaps[:, 0]]
        sys_1 = sys_idx[swaps[:, 1]]

        assert torch.all(sys_0 == sys_1), "Proposed swap crosses system boundaries!"
