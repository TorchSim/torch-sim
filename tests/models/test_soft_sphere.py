"""Tests for soft sphere models ensuring different parts of torchsim work together."""

import itertools

import pytest
import torch

from torch_sim.models.interface import validate_model_outputs
from torch_sim.models.soft_sphere import SoftSphereModel, SoftSphereMultiModel
from torch_sim.state import SimState


@pytest.fixture
def calculator_outputs(
    fe_fcc_sim_state: SimState,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create both neighbor list and direct calculators."""
    calc_params = {
        "sigma": 3.405,  # Å, typical for Ar
        "epsilon": 0.0104,  # eV, typical for Ar
        "alpha": 2.0,
        "dtype": torch.float64,
        "periodic": True,
        "compute_force": True,
        "compute_stress": True,
    }

    calc_nl = SoftSphereModel(use_neighbor_list=True, **calc_params)
    calc_direct = SoftSphereModel(use_neighbor_list=False, **calc_params)

    return calc_nl(fe_fcc_sim_state), calc_direct(fe_fcc_sim_state)


@pytest.fixture
def multi_species_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a binary system with two different species
    arranged in alternating layers.
    """
    N = 4
    a_len = 5.26  # Å
    positions = []
    species = []

    # Create simple cubic lattice with alternating species layers
    for x, y, z in itertools.product(range(N), range(N), range(N)):
        pos = torch.tensor([x, y, z], dtype=torch.float64) * a_len
        positions.append(pos)
        # Alternate species by z-layer
        species.append(0 if z % 2 == 0 else 1)  # Using integer indices instead of strings

    positions = torch.stack(positions)
    species = torch.tensor(species, dtype=torch.long)  # Convert to tensor
    cell = torch.eye(3, dtype=torch.float64) * (N * a_len)

    # Add small random displacements
    torch.manual_seed(42)
    positions += 0.2 * torch.randn_like(positions)

    return positions, cell, species


@pytest.fixture
def multi_calculators(
    multi_species_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create both neighbor list and direct multi-species calculators."""
    # Define interaction parameters for binary system
    sigma_matrix = torch.tensor([[3.405, 3.0], [3.0, 2.5]], dtype=torch.float64)  # Å
    epsilon_matrix = torch.tensor(
        [[0.0104, 0.008], [0.008, 0.006]], dtype=torch.float64
    )  # eV
    alpha_matrix = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float64)

    calc_params = {
        "species": torch.tensor([0, 1], dtype=torch.long),  # Species indices
        "sigma_matrix": sigma_matrix,
        "epsilon_matrix": epsilon_matrix,
        "alpha_matrix": alpha_matrix,
        "dtype": torch.float64,
        "periodic": True,
        "compute_force": True,
        "compute_stress": True,
        "per_atom_energies": True,
        "per_atom_stresses": True,
    }

    calc_nl = SoftSphereMultiModel(use_neighbor_list=True, **calc_params)
    calc_direct = SoftSphereMultiModel(use_neighbor_list=False, **calc_params)

    positions, cell, species = multi_species_system
    multi_species_system_dict = {
        "positions": positions,
        "cell": cell,
        "species": species,
    }
    return calc_nl(multi_species_system_dict), calc_direct(multi_species_system_dict)


def test_energy_match(
    calculator_outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that total energy matches between neighbor list and direct calculations."""
    results_nl, results_direct = calculator_outputs
    assert torch.allclose(results_nl["energy"], results_direct["energy"], rtol=1e-10)


def test_forces_match(
    calculator_outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces match between neighbor list and direct calculations."""
    results_nl, results_direct = calculator_outputs
    assert torch.allclose(results_nl["forces"], results_direct["forces"], rtol=1e-10)


def test_stress_match(
    calculator_outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensors match between neighbor list and direct calculations."""
    results_nl, results_direct = calculator_outputs
    assert torch.allclose(results_nl["stress"], results_direct["stress"], rtol=1e-10)


def test_force_conservation(
    calculator_outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces sum to zero."""
    results_nl, _ = calculator_outputs
    assert torch.allclose(
        results_nl["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10
    )


def test_stress_tensor_symmetry(
    calculator_outputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensor is symmetric."""
    results_nl, _ = calculator_outputs
    assert torch.allclose(results_nl["stress"], results_nl["stress"].T, atol=1e-10)


def test_validate_model_outputs(device: torch.device) -> None:
    """Test that the model outputs are valid."""
    calc_params = {
        "sigma": 3.405,  # Å, typical for Ar
        "epsilon": 0.0104,  # eV, typical for Ar
        "alpha": 2.0,
        "dtype": torch.float64,
        "periodic": True,
        "compute_force": True,
        "compute_stress": True,
    }

    calc_nl = SoftSphereModel(use_neighbor_list=True, **calc_params)
    calc_direct = SoftSphereModel(use_neighbor_list=False, **calc_params)
    for out in [calc_nl, calc_direct]:
        validate_model_outputs(out, device, torch.float64)
