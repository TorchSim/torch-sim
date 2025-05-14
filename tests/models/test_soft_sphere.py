"""Tests for soft sphere models ensuring different parts of torchsim work together."""

import pytest
import torch

import torch_sim as ts
import torch_sim.models.soft_sphere as ss
from torch_sim.models.interface import validate_model_outputs


@pytest.fixture
def models(
    fe_supercell_sim_state: ts.SimState,
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

    model_nl = ss.SoftSphereModel(use_neighbor_list=True, **calc_params)
    model_direct = ss.SoftSphereModel(use_neighbor_list=False, **calc_params)

    return model_nl(fe_supercell_sim_state), model_direct(fe_supercell_sim_state)


@pytest.fixture
def models_with_per_atom(
    fe_supercell_sim_state: ts.SimState,
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

    model_nl = ss.SoftSphereModel(use_neighbor_list=True, **calc_params)
    model_direct = ss.SoftSphereModel(use_neighbor_list=False, **calc_params)

    return model_nl(fe_supercell_sim_state), model_direct(fe_supercell_sim_state)


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
def small_sim_state(small_system: tuple[torch.Tensor, torch.Tensor]) -> ts.SimState:
    """Create a small SimState for testing."""
    positions, cell = small_system
    return ts.SimState(
        positions=positions,
        cell=cell,
        pbc=torch.tensor([True, True, True]),
        masses=torch.ones(positions.shape[0], dtype=torch.float64),
        atomic_numbers=torch.ones(positions.shape[0], dtype=torch.long),
    )


@pytest.fixture
def small_batched_sim_state(small_sim_state: ts.SimState) -> ts.SimState:
    """Create a batched state from the small system."""
    return ts.concatenate_states(
        [small_sim_state, small_sim_state], device=small_sim_state.device
    )


@pytest.mark.parametrize("output_key", ["energy", "forces", "stress"])
def test_outputs_match(
    models: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    output_key: str,
) -> None:
    """Test that outputs match between neighbor list and direct calculations."""
    results_nl, results_direct = models
    assert torch.allclose(results_nl[output_key], results_direct[output_key], rtol=1e-10)


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

    model_nl = ss.SoftSphereModel(use_neighbor_list=True, **model_params)
    model_direct = ss.SoftSphereModel(use_neighbor_list=False, **model_params)
    for out in [model_nl, model_direct]:
        validate_model_outputs(out, device, torch.float64)


@pytest.mark.parametrize(
    ("per_atom_key", "total_key"), [("energies", "energy"), ("stresses", "stress")]
)
def test_per_atom_properties(
    models_with_per_atom: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    per_atom_key: str,
    total_key: str,
) -> None:
    """Test that per-atom properties are calculated correctly."""
    results_nl, results_direct = models_with_per_atom

    # Check per-atom properties are calculated and match
    assert torch.allclose(
        results_nl[per_atom_key], results_direct[per_atom_key], rtol=1e-10
    )

    # Check sum of per-atom properties matches total property
    if per_atom_key == "energies":
        assert torch.allclose(
            results_nl[per_atom_key].sum(), results_nl[total_key], rtol=1e-10
        )
    else:  # stresses
        total_from_atoms = results_nl[per_atom_key].sum(dim=0)
        assert torch.allclose(total_from_atoms, results_nl[total_key], rtol=1e-10)


@pytest.mark.parametrize(
    ("distance", "sigma", "epsilon", "alpha", "expected"),
    [
        (0.5, 1.0, 1.0, 2.0, 0.125),  # distance < sigma
        (1.0, 1.0, 1.0, 2.0, 0.0),  # distance = sigma
        (1.5, 1.0, 1.0, 2.0, 0.0),  # distance > sigma
    ],
)
def test_soft_sphere_pair_single(
    distance: float, sigma: float, epsilon: float, alpha: float, expected: float
) -> None:
    """Test the soft sphere pair calculation for single values."""
    energy = ss.soft_sphere_pair(
        torch.tensor(distance),
        torch.tensor(sigma),
        torch.tensor(epsilon),
        torch.tensor(alpha),
    )
    assert torch.allclose(energy, torch.tensor(expected))


def test_model_initialization_defaults() -> None:
    """Test initialization with default parameters."""
    model = ss.SoftSphereModel()

    # Check default parameters are used
    assert torch.allclose(model.sigma, ss.DEFAULT_SIGMA)
    assert torch.allclose(model.epsilon, ss.DEFAULT_EPSILON)
    assert torch.allclose(model.alpha, ss.DEFAULT_ALPHA)
    assert torch.allclose(model.cutoff, ss.DEFAULT_SIGMA)  # Default cutoff is sigma


@pytest.mark.parametrize(
    ("param_name", "param_value", "expected_dtype"),
    [
        ("sigma", 2.0, torch.float64),
        ("epsilon", 3.0, torch.float64),
        ("alpha", 4.0, torch.float64),
        ("cutoff", 5.0, torch.float64),
    ],
)
def test_model_initialization_custom_params(
    param_name: str, param_value: float, expected_dtype: torch.dtype
) -> None:
    """Test initialization with custom parameters."""
    params = {param_name: param_value, "dtype": expected_dtype}
    model = ss.SoftSphereModel(**params)

    param_tensor = getattr(model, param_name)
    assert torch.allclose(param_tensor, torch.tensor(param_value, dtype=expected_dtype))
    assert param_tensor.dtype == expected_dtype


@pytest.mark.parametrize(
    ("flag_name", "flag_value"),
    [
        ("compute_forces", False),
        ("compute_stress", True),
        ("per_atom_energies", True),
        ("per_atom_stresses", True),
        ("use_neighbor_list", False),
    ],
)
def test_model_initialization_custom_flags(*, flag_name: str, flag_value: bool) -> None:
    """Test initialization with custom flags."""
    model = ss.SoftSphereModel(**{flag_name: flag_value})

    # For compute_forces and compute_stress, we need to check the private attributes
    if flag_name == "compute_forces":
        flag_name = "_compute_forces"
    elif flag_name == "compute_stress":
        flag_name = "_compute_stress"

    assert getattr(model, flag_name) is flag_value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_dtype(dtype: torch.dtype) -> None:
    """Test model with different dtypes."""
    model = ss.SoftSphereModel(dtype=dtype)

    assert model.sigma.dtype == dtype
    assert model.epsilon.dtype == dtype
    assert model.alpha.dtype == dtype
    assert model.cutoff.dtype == dtype


def test_multispecies_initialization_defaults() -> None:
    """Test initialization of multi-species model with defaults."""
    # Create with minimal parameters
    species = torch.tensor([0, 1], dtype=torch.long)
    model = ss.SoftSphereMultiModel(species=species)

    # Check matrices are created with defaults
    assert model.sigma_matrix.shape == (2, 2)
    assert model.epsilon_matrix.shape == (2, 2)
    assert model.alpha_matrix.shape == (2, 2)

    # Check default values
    assert torch.allclose(model.sigma_matrix, ss.DEFAULT_SIGMA * torch.ones(2, 2))
    assert torch.allclose(model.epsilon_matrix, ss.DEFAULT_EPSILON * torch.ones(2, 2))
    assert torch.allclose(model.alpha_matrix, ss.DEFAULT_ALPHA * torch.ones(2, 2))

    # Check cutoff is max sigma
    assert model.cutoff.item() == ss.DEFAULT_SIGMA.item()


def test_multispecies_initialization_custom() -> None:
    """Test initialization of multi-species model with custom parameters."""
    species = torch.tensor([0, 1], dtype=torch.long)
    sigma_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]], dtype=torch.float64)
    epsilon_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
    alpha_matrix = torch.tensor([[2.0, 3.0], [3.0, 4.0]], dtype=torch.float64)

    model = ss.SoftSphereMultiModel(
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
        ss.SoftSphereMultiModel(
            species=species,
            sigma_matrix=sigma_matrix,
            epsilon_matrix=epsilon_matrix,
        )


@pytest.mark.parametrize(
    ("matrix_name", "matrix"),
    [
        ("sigma_matrix", torch.tensor([[1.0, 1.5], [2.0, 2.0]])),
        ("epsilon_matrix", torch.tensor([[1.0, 0.5], [0.7, 1.5]])),
        ("alpha_matrix", torch.tensor([[2.0, 3.0], [4.0, 4.0]])),
    ],
)
def test_matrix_symmetry_validation(matrix_name: str, matrix: torch.Tensor) -> None:
    """Test that parameter matrices are validated for symmetry."""
    species = torch.tensor([0, 1], dtype=torch.long)

    # Create symmetric matrices for the other parameters
    symmetric_matrix = torch.tensor([[1.0, 1.5], [1.5, 2.0]])

    params = {
        "species": species,
        "sigma_matrix": symmetric_matrix,
        "epsilon_matrix": symmetric_matrix,
        "alpha_matrix": symmetric_matrix,
    }

    # Replace one matrix with the non-symmetric version
    params[matrix_name] = matrix

    # Should raise AssertionError due to asymmetric matrix
    with pytest.raises(AssertionError):
        ss.SoftSphereMultiModel(**params)


def test_multispecies_cutoff_default() -> None:
    """Test that the default cutoff is the maximum sigma value."""
    # Create model with varying sigma values
    species = torch.tensor([0, 1, 2], dtype=torch.long)
    sigma_matrix = torch.tensor([[1.0, 1.5, 2.0], [1.5, 2.0, 2.5], [2.0, 2.5, 3.0]])

    model = ss.SoftSphereMultiModel(species=species, sigma_matrix=sigma_matrix)

    # Cutoff should default to max value in sigma_matrix
    assert model.cutoff.item() == 3.0


@pytest.mark.parametrize(
    ("flag_name", "flag_value"),
    [
        ("pbc", True),
        ("pbc", False),
        ("compute_forces", False),
        ("compute_stress", True),
        ("per_atom_energies", True),
        ("per_atom_stresses", False),
        ("use_neighbor_list", True),
        ("use_neighbor_list", False),
    ],
)
def test_multispecies_model_flags(*, flag_name: str, flag_value: bool) -> None:
    """Test flags of the SoftSphereMultiModel."""
    species = torch.tensor([0, 1], dtype=torch.long)

    model = ss.SoftSphereMultiModel(species=species, **{flag_name: flag_value})

    # For SoftSphereMultiModel, we don't need to convert attribute names
    # as it uses public attribute names for all flags
    assert getattr(model, flag_name) is flag_value
