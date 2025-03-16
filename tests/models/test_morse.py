"""Tests for Morse potential calculator using copper parameters."""

import pytest
import torch

from torch_sim.unbatched.models.morse import (
    UnbatchedMorseModel,
    morse_pair,
    morse_pair_force,
)


def test_morse_pair_minimum() -> None:
    """Test that the potential has its minimum at r=sigma."""
    dr = torch.linspace(0.8, 1.2, 100)
    dr = dr.reshape(-1, 1)
    energy = morse_pair(dr)
    min_idx = torch.argmin(energy)
    torch.testing.assert_close(dr[min_idx], torch.tensor([1.0]), rtol=0.01, atol=1e-5)


def test_morse_pair_scaling() -> None:
    """Test that the potential scales correctly with epsilon."""
    dr = torch.ones(5, 5) * 1.5
    e1 = morse_pair(dr, epsilon=1.0)
    e2 = morse_pair(dr, epsilon=2.0)
    torch.testing.assert_close(e2, 2 * e1, rtol=1e-5, atol=1e-5)


def test_morse_pair_asymptotic() -> None:
    """Test that the potential approaches -epsilon at large distances."""
    dr = torch.tensor([[1.0]])  # Large distance
    epsilon = 5.0
    energy = morse_pair(dr, epsilon=epsilon)
    print(energy, -epsilon * torch.ones_like(energy))
    torch.testing.assert_close(
        energy, -epsilon * torch.ones_like(energy), rtol=1e-2, atol=1e-5
    )


def test_morse_pair_force_scaling() -> None:
    """Test that the force scales correctly with epsilon."""
    dr = torch.ones(5, 5) * 1.5
    f1 = morse_pair_force(dr, epsilon=1.0)
    f2 = morse_pair_force(dr, epsilon=2.0)
    assert torch.allclose(f2, 2 * f1)


def test_morse_force_energy_consistency() -> None:
    """Test that the force is consistent with the energy gradient."""
    dr = torch.linspace(0.8, 2.0, 100, requires_grad=True)
    dr = dr.reshape(-1, 1)

    # Calculate force directly
    force_direct = morse_pair_force(dr)

    # Calculate force from energy gradient
    energy = morse_pair(dr)
    force_from_grad = -torch.autograd.grad(energy.sum(), dr, create_graph=True)[0]

    # Compare forces
    print(force_direct)
    print(force_from_grad)
    assert torch.allclose(force_direct, force_from_grad, rtol=1e-4, atol=1e-4)


def test_morse_alpha_effect() -> None:
    """Test that larger alpha values make the potential well narrower."""
    dr = torch.linspace(0.8, 1.2, 100)
    dr = dr.reshape(-1, 1)

    energy1 = morse_pair(dr, alpha=5.0)
    energy2 = morse_pair(dr, alpha=10.0)

    # Calculate width at half minimum
    def get_well_width(energy: torch.Tensor) -> torch.Tensor:
        min_e = torch.min(energy)
        half_e = min_e / 2
        mask = energy < half_e
        return dr[mask].max() - dr[mask].min()

    width1 = get_well_width(energy1)
    width2 = get_well_width(energy2)
    assert width2 < width1  # Higher alpha should give narrower well


@pytest.fixture
def cu_fcc_system() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a NxNxN FCC crystal system for Copper with random displacements."""
    # FCC conventional cell basis positions (in fractional coordinates)
    basis = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # corner
            [0.0, 0.5, 0.5],  # face center
            [0.5, 0.0, 0.5],  # face center
            [0.5, 0.5, 0.0],  # face center
        ],
        dtype=torch.float64,
    )

    # Lattice constant for Copper
    a_len = 3.61  # Å
    N = 4
    # Create NxNxN supercell
    positions = []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                offset = torch.tensor([x, y, z], dtype=torch.float64)
                for b in basis:
                    # Convert fractional to cartesian coordinates
                    pos = (b + offset) * a_len
                    positions.append(pos)

    # Stack all positions into a single tensor
    positions = torch.stack(positions)

    # Create the cell tensor (cubic cell)
    cell = torch.eye(3, dtype=torch.float64) * (N * a_len)

    # Add random displacements (0.1 Å)
    torch.manual_seed(42)
    positions += 0.1 * torch.randn_like(positions)

    return positions, cell


@pytest.fixture
def calculators(
    cu_fcc_system: tuple[torch.Tensor, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create both neighbor list and direct calculators with Copper parameters."""
    # Parameters for Copper (Cu) using Morse potential
    # Values from: https://doi.org/10.1016/j.commatsci.2004.12.069
    calc_params = {
        "sigma": 2.55,  # Å, equilibrium distance
        "epsilon": 0.436,  # eV, dissociation energy
        "alpha": 1.359,  # Å^-1, controls potential well width
        "dtype": torch.float64,
        "periodic": True,
        "compute_force": True,
        "compute_stress": True,
    }

    cutoff = 2.5 * 2.55  # Similar scaling as LJ cutoff
    calc_nl = UnbatchedMorseModel(use_neighbor_list=True, cutoff=cutoff, **calc_params)
    calc_direct = UnbatchedMorseModel(
        use_neighbor_list=False, cutoff=cutoff, **calc_params
    )

    state = dict(
        positions=cu_fcc_system[0],
        cell=cu_fcc_system[1],
        atomic_numbers=torch.ones(len(cu_fcc_system[0]), dtype=torch.int32),
    )
    return calc_nl(state), calc_direct(state)


def test_energy_match(
    calculators: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that total energy matches between neighbor list and direct calculations."""
    results_nl, results_direct = calculators
    assert torch.allclose(results_nl["energy"], results_direct["energy"], rtol=1e-10)


def test_forces_match(
    calculators: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces match between neighbor list and direct calculations."""
    results_nl, results_direct = calculators
    assert torch.allclose(results_nl["forces"], results_direct["forces"], rtol=1e-10)


def test_stress_match(
    calculators: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensors match between neighbor list and direct calculations."""
    results_nl, results_direct = calculators
    assert torch.allclose(results_nl["stress"], results_direct["stress"], rtol=1e-10)


def test_force_conservation(
    calculators: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces sum to zero (Newton's third law)."""
    results_nl, _ = calculators
    assert torch.allclose(
        results_nl["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10
    )


def test_stress_tensor_symmetry(
    calculators: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensor is symmetric."""
    results_nl, _ = calculators
    assert torch.allclose(results_nl["stress"], results_nl["stress"].T, atol=1e-10)
