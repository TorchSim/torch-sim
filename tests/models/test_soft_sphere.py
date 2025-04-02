"""Tests for soft sphere models ensuring different parts of torchsim work together."""

import pytest
import torch

from torch_sim.models.interface import validate_model_outputs
from torch_sim.models.soft_sphere import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_SIGMA,
    SoftSphereModel,
    SoftSphereMultiModel,
    soft_sphere_pair,
    soft_sphere_pair_force,
)
from torch_sim.state import SimState, concatenate_states


@pytest.fixture
def models(
    fe_fcc_sim_state: SimState,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create both neighbor list and direct calculators."""
    calc_params = {
        "sigma": 3.405,  # Å, typical for Ar
        "epsilon": 0.0104,  # eV, typical for Ar
        "alpha": 2.0,
        "dtype": torch.float64,
        "compute_forces": True,
        "compute_stress": True,
    }

    model_nl = SoftSphereModel(use_neighbor_list=True, **calc_params)
    model_direct = SoftSphereModel(use_neighbor_list=False, **calc_params)

    return model_nl(fe_fcc_sim_state), model_direct(fe_fcc_sim_state)


@pytest.fixture
def models_with_per_atom(
    fe_fcc_sim_state: SimState,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create calculators with per-atom properties enabled."""
    calc_params = {
        "sigma": 3.405,  # Å, typical for Ar
        "epsilon": 0.0104,  # eV, typical for Ar
        "alpha": 2.0,
        "dtype": torch.float64,
        "compute_forces": True,
        "compute_stress": True,
        "per_atom_energies": True,
        "per_atom_stresses": True,
    }

    model_nl = SoftSphereModel(use_neighbor_list=True, **calc_params)
    model_direct = SoftSphereModel(use_neighbor_list=False, **calc_params)

    return model_nl(fe_fcc_sim_state), model_direct(fe_fcc_sim_state)


@pytest.fixture
def small_system() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small simple cubic system for testing."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    return positions, cell


@pytest.fixture
def small_sim_state(small_system: tuple[torch.Tensor, torch.Tensor]) -> SimState:
    """Create a small SimState for testing."""
    positions, cell = small_system
    return SimState(
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
        masses=torch.ones(positions.shape[0], dtype=torch.float64),
        atomic_numbers=torch.ones(
            positions.shape[0], dtype=torch.long
        ),  # Add atomic numbers
    )


@pytest.fixture
def small_batched_sim_state(small_sim_state: SimState) -> SimState:
    """Create a batched state from the small system."""
    return concatenate_states(
        [small_sim_state, small_sim_state], device=small_sim_state.device
    )


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
    assert torch.allclose(results_nl["stress"], results_nl["stress"].T, atol=1e-10)


def test_validate_model_outputs(device: torch.device) -> None:
    """Test that the model outputs are valid."""
    model_params = {
        "sigma": 3.405,  # Å, typical for Ar
        "epsilon": 0.0104,  # eV, typical for Ar
        "alpha": 2.0,
        "dtype": torch.float64,
        "compute_forces": True,
        "compute_stress": True,
    }

    model_nl = SoftSphereModel(use_neighbor_list=True, **model_params)
    model_direct = SoftSphereModel(use_neighbor_list=False, **model_params)
    for out in [model_nl, model_direct]:
        validate_model_outputs(out, device, torch.float64)


def test_per_atom_energies(
    models_with_per_atom: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that per-atom energies are calculated correctly."""
    results_nl, results_direct = models_with_per_atom

    # Check per-atom energies are calculated and match
    assert torch.allclose(results_nl["energies"], results_direct["energies"], rtol=1e-10)

    # Check sum of per-atom energies matches total energy
    assert torch.allclose(results_nl["energies"].sum(), results_nl["energy"], rtol=1e-10)


def test_per_atom_stresses(
    models_with_per_atom: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    """Test that per-atom stresses are calculated correctly."""
    results_nl, results_direct = models_with_per_atom

    # Check per-atom stresses are calculated and match
    assert torch.allclose(results_nl["stresses"], results_direct["stresses"], rtol=1e-10)

    # Check sum of per-atom stresses matches total stress
    total_stress_from_atoms = results_nl["stresses"].sum(dim=0)
    assert torch.allclose(total_stress_from_atoms, results_nl["stress"], rtol=1e-10)


def test_model_initialization_defaults() -> None:
    """Test initialization with default parameters."""
    model = SoftSphereModel()

    # Check default parameters are used
    assert torch.allclose(model.sigma, DEFAULT_SIGMA)
    assert torch.allclose(model.epsilon, DEFAULT_EPSILON)
    assert torch.allclose(model.alpha, DEFAULT_ALPHA)
    assert torch.allclose(model.cutoff, DEFAULT_SIGMA)  # Default cutoff is sigma

    # Check computation flags
    assert model.per_atom_energies is False
    assert model.per_atom_stresses is False
    assert model.use_neighbor_list is True


def test_model_initialization_custom() -> None:
    """Test initialization with custom parameters."""
    model = SoftSphereModel(
        sigma=2.0,
        epsilon=3.0,
        alpha=4.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=False,
        compute_stress=True,
        per_atom_energies=True,
        per_atom_stresses=True,
        use_neighbor_list=False,
        cutoff=5.0,
    )

    # Check custom parameters are used
    assert torch.allclose(model.sigma, torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(model.epsilon, torch.tensor(3.0, dtype=torch.float64))
    assert torch.allclose(model.alpha, torch.tensor(4.0, dtype=torch.float64))
    assert torch.allclose(model.cutoff, torch.tensor(5.0, dtype=torch.float64))

    # Check computation flags
    assert model.per_atom_energies is True
    assert model.per_atom_stresses is True
    assert model.use_neighbor_list is False


def test_soft_sphere_pair() -> None:
    """Test the soft sphere pair calculation function."""
    # Test single distance less than sigma
    distance = torch.tensor(0.5)
    sigma = torch.tensor(1.0)
    epsilon = torch.tensor(1.0)
    alpha = torch.tensor(2.0)

    energy = soft_sphere_pair(distance, sigma, epsilon, alpha)
    # epsilon/alpha * (1 - dr/sigma)^alpha = 1/2 * (1 - 0.5/1)^2 = 1/2 * 0.5^2
    # = 1/2 * 0.25 = 0.125
    expected = torch.tensor(0.125)
    assert torch.allclose(energy, expected)

    # Test single distance equal to sigma (should return 0)
    distance = torch.tensor(1.0)
    energy = soft_sphere_pair(distance, sigma, epsilon, alpha)
    expected = torch.tensor(0.0)
    assert torch.allclose(energy, expected)

    # Test single distance greater than sigma (should return 0)
    distance = torch.tensor(1.5)
    energy = soft_sphere_pair(distance, sigma, epsilon, alpha)
    expected = torch.tensor(0.0)
    assert torch.allclose(energy, expected)

    # Test multiple distances
    distances = torch.tensor([0.5, 1.0, 1.5])
    energies = soft_sphere_pair(distances, sigma, epsilon, alpha)
    expected = torch.tensor([0.125, 0.0, 0.0])
    assert torch.allclose(energies, expected)

    # Test with different parameters
    distances = torch.tensor([0.5, 0.5, 0.5])
    sigmas = torch.tensor([1.0, 2.0, 0.4])
    epsilons = torch.tensor([1.0, 2.0, 0.5])
    alphas = torch.tensor([2.0, 3.0, 1.0])

    energies = soft_sphere_pair(distances, sigmas, epsilons, alphas)
    # Case 1: 1/2 * (1 - 0.5/1)^2 = 0.125
    # Case 2: 2/3 * (1 - 0.5/2)^3 = 2/3 * (0.75)^3 = 2/3 * 0.422 = 0.281
    # Case 3: distance > sigma, so result should be 0
    expected = torch.tensor([0.125, 0.281, 0.0])
    assert torch.allclose(energies, expected, atol=1e-3)


def test_soft_sphere_pair_force() -> None:
    """Test the soft sphere pair force calculation function."""
    # Test single distance less than sigma
    distance = torch.tensor(0.5)
    sigma = torch.tensor(1.0)
    epsilon = torch.tensor(1.0)
    alpha = torch.tensor(2.0)

    force = soft_sphere_pair_force(distance, sigma, epsilon, alpha)
    # (-epsilon/sigma) * (1 - dr/sigma)^(alpha-1) = -1/1 * (1 - 0.5/1)^(2-1)
    # = -1 * 0.5^1 = -0.5
    expected = torch.tensor(-0.5)
    assert torch.allclose(force, expected)

    # Test single distance equal to sigma (should return 0)
    distance = torch.tensor(1.0)
    force = soft_sphere_pair_force(distance, sigma, epsilon, alpha)
    expected = torch.tensor(0.0)
    assert torch.allclose(force, expected)

    # Test single distance greater than sigma (should return 0)
    distance = torch.tensor(1.5)
    force = soft_sphere_pair_force(distance, sigma, epsilon, alpha)
    expected = torch.tensor(0.0)
    assert torch.allclose(force, expected)

    # Test multiple distances
    distances = torch.tensor([0.5, 1.0, 1.5])
    forces = soft_sphere_pair_force(distances, sigma, epsilon, alpha)
    expected = torch.tensor([-0.5, 0.0, 0.0])
    assert torch.allclose(forces, expected)

    # Test with different parameters
    distances = torch.tensor([0.5, 0.5, 0.5])
    sigmas = torch.tensor([1.0, 2.0, 0.4])
    epsilons = torch.tensor([1.0, 2.0, 0.5])
    alphas = torch.tensor([2.0, 3.0, 1.0])

    forces = soft_sphere_pair_force(distances, sigmas, epsilons, alphas)
    # Case 1: -1/1 * (1 - 0.5/1)^(2-1) = -0.5
    # Case 2: -2/2 * (1 - 0.5/2)^(3-1) = -1 * (0.75)^2 = -0.563
    # Case 3: distance > sigma, so result should be 0
    expected = torch.tensor([-0.5, -0.563, 0.0])
    assert torch.allclose(forces, expected, atol=1e-3)


def test_multispecies_initialization_defaults() -> None:
    """Test initialization of multi-species model with defaults."""
    # Create with minimal parameters
    species = torch.tensor([0, 1], dtype=torch.long)
    model = SoftSphereMultiModel(species=species)

    # Check matrices are created with defaults
    assert model.sigma_matrix.shape == (2, 2)
    assert model.epsilon_matrix.shape == (2, 2)
    assert model.alpha_matrix.shape == (2, 2)

    # Check default values
    assert torch.allclose(model.sigma_matrix, DEFAULT_SIGMA * torch.ones(2, 2))
    assert torch.allclose(model.epsilon_matrix, DEFAULT_EPSILON * torch.ones(2, 2))
    assert torch.allclose(model.alpha_matrix, DEFAULT_ALPHA * torch.ones(2, 2))

    # Check cutoff is max sigma
    assert model.cutoff.item() == DEFAULT_SIGMA.item()


def test_multispecies_initialization_custom() -> None:
    """Test initialization of multi-species model with custom parameters."""
    species = torch.tensor([0, 1], dtype=torch.long)
    sigma_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]], dtype=torch.float64)
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
    alpha_matrix = torch.tensor([[2.0, 3.0], [3.0, 4.0]], dtype=torch.float64)

    model = SoftSphereMultiModel(
        species=species,
        sigma_matrix=sigma_matrix,
        epsilon_matrix=epsilon_matrix,
        alpha_matrix=alpha_matrix,
        cutoff=3.0,
        dtype=torch.float64,
    )

    # Check matrices are stored correctly
    assert torch.allclose(model.sigma_matrix, sigma_matrix)
    assert torch.allclose(model.epsilon_matrix, epsilon_matrix)
    assert torch.allclose(model.alpha_matrix, alpha_matrix)

    # Check cutoff is set explicitly
    assert model.cutoff.item() == 3.0


def test_multispecies_matrix_validation() -> None:
    """Test validation of parameter matrices."""
    species = torch.tensor([0, 1, 2], dtype=torch.long)  # 3 unique species

    # Create incorrect-sized matrices (2x2 instead of 3x3)
    sigma_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]])
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]])

    # Should raise ValueError due to matrix size mismatch
    with pytest.raises(ValueError, match="sigma_matrix must have shape"):
        SoftSphereMultiModel(
            species=species,
            sigma_matrix=sigma_matrix,
            epsilon_matrix=epsilon_matrix,
        )


def test_soft_sphere_model_attributes() -> None:
    """Test additional attributes and methods of the SoftSphereModel."""
    # Test setting custom device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = SoftSphereModel(device=device)
        assert model.sigma.device == device
        assert model.epsilon.device == device
        assert model.alpha.device == device
        assert model.cutoff.device == device

    # Test with different dtypes
    for dtype in [torch.float32, torch.float64]:
        model = SoftSphereModel(dtype=dtype)
        assert model.sigma.dtype == dtype
        assert model.epsilon.dtype == dtype
        assert model.alpha.dtype == dtype
        assert model.cutoff.dtype == dtype


def test_multispecies_cutoff_default() -> None:
    """Test that the default cutoff is the maximum sigma value."""
    # Create model with varying sigma values
    species = torch.tensor([0, 1, 2], dtype=torch.long)
    sigma_matrix = torch.tensor([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5], [2.0, 2.5, 3.0]])

    model = SoftSphereMultiModel(species=species, sigma_matrix=sigma_matrix)

    # Cutoff should default to max value in sigma_matrix
    assert model.cutoff.item() == 3.0


def test_matrix_symmetry_validation() -> None:
    """Test that parameter matrices are validated for symmetry."""
    species = torch.tensor([0, 1], dtype=torch.long)

    # Create non-symmetric matrices
    sigma_matrix = torch.tensor([[1.0, 1.5], [2.0, 2.0]])  # Not symmetric
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]])  # Symmetric
    alpha_matrix = torch.tensor([[2.0, 3.0], [3.0, 4.0]])  # Symmetric

    # Should raise AssertionError due to asymmetric sigma matrix
    with pytest.raises(AssertionError):
        SoftSphereMultiModel(
            species=species,
            sigma_matrix=sigma_matrix,
            epsilon_matrix=epsilon_matrix,
            alpha_matrix=alpha_matrix,
        )

    # Test with non-symmetric epsilon matrix
    sigma_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]])  # Symmetric
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.7, 1.5]])  # Not symmetric

    with pytest.raises(AssertionError):
        SoftSphereMultiModel(
            species=species,
            sigma_matrix=sigma_matrix,
            epsilon_matrix=epsilon_matrix,
            alpha_matrix=alpha_matrix,
        )


def test_multispecies_model_attributes() -> None:
    """Test additional attributes and methods of the SoftSphereMultiModel."""
    species = torch.tensor([0, 1], dtype=torch.long)

    # Test setting PBC flag
    model_pbc_true = SoftSphereMultiModel(species=species, pbc=True)
    assert model_pbc_true.pbc is True

    model_pbc_false = SoftSphereMultiModel(species=species, pbc=False)
    assert model_pbc_false.pbc is False

    # Test computation flags
    model = SoftSphereMultiModel(
        species=species,
        compute_forces=False,
        compute_stress=True,
        per_atom_energies=True,
        per_atom_stresses=False,
    )

    assert model.compute_forces is False
    assert model.compute_stress is True
    assert model.per_atom_energies is True
    assert model.per_atom_stresses is False

    # Test setting neighbor list flag
    model_nl = SoftSphereMultiModel(species=species, use_neighbor_list=True)
    assert model_nl.use_neighbor_list is True

    model_no_nl = SoftSphereMultiModel(species=species, use_neighbor_list=False)
    assert model_no_nl.use_neighbor_list is False
