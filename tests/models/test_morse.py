"""Tests for Morse potential calculator using copper parameters."""

import pytest
import torch

import torch_sim as ts
from torch_sim.models.morse import MorseModel, morse_pair, morse_pair_force


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
def models(
    cu_supercell_sim_state: ts.SimState,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create both neighbor list and direct calculators with Copper parameters."""
    # Parameters for Copper (Cu) using Morse potential
    # Values from: https://doi.org/10.1016/j.commatsci.2004.12.069
    model_kwargs: dict[str, float | bool | torch.dtype] = {
        "sigma": 2.55,  # Å, equilibrium distance
        "epsilon": 0.436,  # eV, dissociation energy
        "alpha": 1.359,  # Å^-1, controls potential well width
        "dtype": torch.float64,
        "compute_forces": True,
        "compute_stress": True,
    }
    cutoff = 2.5 * 2.55  # Similar scaling as LJ cutoff
    model_nl = MorseModel(use_neighbor_list=True, cutoff=cutoff, **model_kwargs)
    model_direct = MorseModel(use_neighbor_list=False, cutoff=cutoff, **model_kwargs)

    return model_nl(cu_supercell_sim_state), model_direct(cu_supercell_sim_state)


def test_energy_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that total energy matches between neighbor list and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl["energy"], results_direct["energy"], rtol=1e-10)


def test_forces_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces match between neighbor list and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl["forces"], results_direct["forces"], rtol=1e-10)


def test_stress_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensors match between neighbor list and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl["stress"], results_direct["stress"], rtol=1e-10)


def test_force_conservation(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces sum to zero (Newton's third law)."""
    results_nl, _ = models
    assert torch.allclose(
        results_nl["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10
    )


def test_stress_tensor_symmetry(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensor is symmetric."""
    results_nl, _ = models
    assert torch.allclose(
        results_nl["stress"].squeeze(), results_nl["stress"].squeeze().T, atol=1e-10
    )
