import pytest
import torch
from ase.build import bulk

from torch_sim.io import atoms_to_state
from torch_sim.models.interface import validate_model_outputs
from torch_sim.state import SimState


try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.atomic_system import SystemConfig
    from orb_models.forcefield.calculator import ORBCalculator

    from torch_sim.models.orb import OrbModel
except ImportError:
    pytest.skip("ORB not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def pretrained_orb_model(device: torch.device):
    """Load a pretrained ORB model for testing."""
    return pretrained.orb_v2(device=device)


@pytest.fixture
def cu_system(dtype: torch.dtype, device: torch.device) -> SimState:
    # Create FCC Copper
    cu_fcc = bulk("Cu", "fcc", a=3.58, cubic=True)
    return atoms_to_state([cu_fcc], device, dtype)


@pytest.fixture
def orb_model(pretrained_orb_model: torch.nn.Module, device: torch.device) -> OrbModel:
    """Create an OrbModel wrapper for the pretrained model."""
    return OrbModel(
        model=pretrained_orb_model,
        device=device,
        system_config=SystemConfig(radius=6.0, max_num_neighbors=20),
    )


@pytest.fixture
def orb_calculator(
    pretrained_orb_model: torch.nn.Module, device: torch.device
) -> ORBCalculator:
    """Create an ORBCalculator for the pretrained model."""
    return ORBCalculator(
        model=pretrained_orb_model,
        system_config=SystemConfig(radius=6.0, max_num_neighbors=20),
        device=device,
    )


def test_orb_initialization(
    pretrained_orb_model: torch.nn.Module, device: torch.device
) -> None:
    """Test that the ORB model initializes correctly."""
    model = OrbModel(
        model=pretrained_orb_model,
        device=device,
    )
    # Check that properties were set correctly
    assert "energy" in model.implemented_properties
    assert "forces" in model.implemented_properties
    assert model._device == device  # noqa: SLF001


def test_orb_calculator_consistency(
    orb_model: OrbModel,
    orb_calculator: ORBCalculator,
    cu_system: SimState,
    device: torch.device,
) -> None:
    """Test consistency between OrbModel and ORBCalculator."""
    # Get OrbModel results
    orb_results = orb_model(cu_system)

    # Set up ASE calculator
    cu_fcc = bulk("Cu", "fcc", a=3.58, cubic=True)
    cu_fcc.calc = orb_calculator

    # Get calculator results
    calc_energy = cu_fcc.get_potential_energy()
    calc_forces = torch.tensor(
        cu_fcc.get_forces(),
        device=device,
        dtype=orb_results["forces"].dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        orb_results["energy"].item(),
        calc_energy,
        atol=1e-4,
        rtol=0.5,
    )
    torch.testing.assert_close(
        orb_results["forces"],
        calc_forces,
        atol=1e-4,
        rtol=0.5,
    )


def test_validate_model_outputs(orb_model: OrbModel, device: torch.device) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(orb_model, device, torch.float32)
