"""Cheap integration tests ensuring different parts of torchsim work together."""

import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.interface import validate_model_outputs
from torch_sim.state import SimState
from torch_sim.unbatched.models.lennard_jones import (
    UnbatchedLennardJonesModel,
    lennard_jones_pair,
    lennard_jones_pair_force,
)


def test_lennard_jones_pair_minimum() -> None:
    """Test that the potential has its minimum at r=sigma."""
    dr = torch.linspace(0.8, 1.2, 100)
    dr = dr.reshape(-1, 1)
    energy = lennard_jones_pair(dr, sigma=1.0, epsilon=1.0)
    min_idx = torch.argmin(energy)

    torch.testing.assert_close(
        dr[min_idx], torch.tensor([2 ** (1 / 6)]), rtol=1e-2, atol=1e-2
    )


def test_lennard_jones_pair_scaling() -> None:
    """Test that the potential scales correctly with epsilon."""
    dr = torch.ones(5, 5) * 1.5
    e1 = lennard_jones_pair(dr, sigma=1.0, epsilon=1.0)
    e2 = lennard_jones_pair(dr, sigma=1.0, epsilon=2.0)
    torch.testing.assert_close(e2, 2 * e1)


def test_lennard_jones_pair_repulsive_core() -> None:
    """Test that the potential is strongly repulsive at short distances."""
    dr_close = torch.tensor([[0.5]])  # Less than sigma
    dr_far = torch.tensor([[2.0]])  # Greater than sigma
    e_close = lennard_jones_pair(dr_close)
    e_far = lennard_jones_pair(dr_far)
    assert e_close > e_far
    assert e_close > 0  # Repulsive
    assert e_far < 0  # Attractive


def test_lennard_jones_pair_tensor_params() -> None:
    """Test that the function works with tensor parameters."""
    dr = torch.ones(3, 3) * 1.5
    sigma = torch.ones(3, 3)
    epsilon = torch.ones(3, 3) * 2.0
    energy = lennard_jones_pair(dr, sigma=sigma, epsilon=epsilon)
    assert energy.shape == (3, 3)


def test_lennard_jones_pair_zero_distance() -> None:
    """Test that the function handles zero distances gracefully."""
    dr = torch.zeros(2, 2)
    energy = lennard_jones_pair(dr)
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()


def test_lennard_jones_pair_batch() -> None:
    """Test that the function works with batched inputs."""
    batch_size = 10
    n_particles = 5
    dr = torch.rand(batch_size, n_particles, n_particles) + 0.5
    energy = lennard_jones_pair(dr)
    assert energy.shape == (batch_size, n_particles, n_particles)


def test_lennard_jones_pair_force_scaling() -> None:
    """Test that the force scales correctly with epsilon."""
    dr = torch.ones(5, 5) * 1.5
    f1 = lennard_jones_pair_force(dr, sigma=1.0, epsilon=1.0)
    f2 = lennard_jones_pair_force(dr, sigma=1.0, epsilon=2.0)
    assert torch.allclose(f2, 2 * f1)


def test_lennard_jones_pair_force_repulsive_core() -> None:
    """Test that the force is strongly repulsive at short distances."""
    dr_close = torch.tensor([[0.5]])  # Less than sigma
    dr_far = torch.tensor([[2.0]])  # Greater than sigma
    f_close = lennard_jones_pair_force(dr_close)
    f_far = lennard_jones_pair_force(dr_far)
    assert f_close > 0  # Repulsive
    assert f_far < 0  # Attractive
    assert abs(f_close) > abs(f_far)  # Stronger at short range


def test_lennard_jones_pair_force_tensor_params() -> None:
    """Test that the function works with tensor parameters."""
    dr = torch.ones(3, 3) * 1.5
    sigma = torch.ones(3, 3)
    epsilon = torch.ones(3, 3) * 2.0
    force = lennard_jones_pair_force(dr, sigma=sigma, epsilon=epsilon)
    assert force.shape == (3, 3)


def test_lennard_jones_pair_force_zero_distance() -> None:
    """Test that the function handles zero distances gracefully."""
    dr = torch.zeros(2, 2)
    force = lennard_jones_pair_force(dr)
    assert not torch.isnan(force).any()
    assert not torch.isinf(force).any()


def test_lennard_jones_pair_force_batch() -> None:
    """Test that the function works with batched inputs."""
    batch_size = 10
    n_particles = 5
    dr = torch.rand(batch_size, n_particles, n_particles) + 0.5
    force = lennard_jones_pair_force(dr)
    assert force.shape == (batch_size, n_particles, n_particles)


def test_lennard_jones_force_energy_consistency() -> None:
    """Test that the force is consistent with the energy gradient."""
    dr = torch.linspace(0.8, 2.0, 100, requires_grad=True)
    dr = dr.reshape(-1, 1)

    # Calculate force directly
    force_direct = lennard_jones_pair_force(dr)

    # Calculate force from energy gradient
    energy = lennard_jones_pair(dr)
    force_from_grad = -torch.autograd.grad(energy.sum(), dr, create_graph=True)[0]

    # Compare forces (allowing for some numerical differences)
    assert torch.allclose(force_direct, force_from_grad, rtol=1e-4, atol=1e-4)


# NOTE: This is a large system to test the neighbor list and direct calculation
#       are consistent. Direct calculation uses minimal image convention, which
#       is not used in the neighbor list calculation. So to get correct results,
#       we need a system that is large enough (2*cutoff).
@pytest.fixture
def ar_supercell_sim_state_large(device: torch.device) -> SimState:
    """Create a face-centered cubic (FCC) Argon structure."""
    # Create FCC Ar using ASE, with 4x4x4 supercell
    ar_atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([4, 4, 4])
    return ts.io.atoms_to_state(ar_atoms, device, torch.float64)


@pytest.fixture
def models(
    ar_supercell_sim_state_large: SimState,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create both neighbor list and direct models with Argon parameters."""
    calc_params = {
        "sigma": 3.405,  # Å, typical for Ar
        "epsilon": 0.0104,  # eV, typical for Ar
        "dtype": torch.float64,
        "compute_forces": True,
        "compute_stress": True,
        "per_atom_energies": True,
        "per_atom_stresses": True,
    }

    cutoff = 2.5 * 3.405  # Standard LJ cutoff * sigma
    model_nl = UnbatchedLennardJonesModel(
        use_neighbor_list=True, cutoff=cutoff, **calc_params
    )
    model_direct = UnbatchedLennardJonesModel(
        use_neighbor_list=False, cutoff=cutoff, **calc_params
    )

    return model_nl(ar_supercell_sim_state_large), model_direct(
        ar_supercell_sim_state_large
    )


def test_energy_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that total energy matches between neighbor list and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl["energy"], results_direct["energy"], rtol=1e-10)


def test_per_atom_energy_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that per-atom energy matches between neighbor list and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl["energies"], results_direct["energies"], rtol=1e-10)


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


def test_per_atom_stress_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that per-atom stress tensors match between neighbor list
    and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl["stresses"], results_direct["stresses"], rtol=1e-10)


def test_force_conservation(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that forces sum to zero."""
    results_nl, _ = models
    assert torch.allclose(
        results_nl["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10
    )


def test_stress_tensor_symmetry(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that stress tensor is symmetric."""
    results_nl, _ = models
    # select trailing two dimensions
    stress_tensor = results_nl["stress"][0]
    assert torch.allclose(stress_tensor, stress_tensor.T, atol=1e-10)


def test_validate_model_outputs(
    lj_model: UnbatchedLennardJonesModel,
    device: torch.device,
) -> None:
    """Test that the model outputs are valid."""
    validate_model_outputs(lj_model, device, torch.float64)
