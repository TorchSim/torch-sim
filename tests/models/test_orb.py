import pytest
import torch

from tests.conftest import make_model_calculator_consistency_test
from torch_sim.models.interface import validate_model_outputs


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


test_orb_consistency = make_model_calculator_consistency_test(
    test_name="orb",
    model_fixture_name="orb_model",
    calculator_fixture_name="orb_calculator",
    sim_state_names=[
        "cu_sim_state",
        "mg_sim_state",
        "sb_sim_state",
        "tio2_sim_state",
        "ga_sim_state",
        "niti_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        "benzene_sim_state",
    ],
    rtol=1e-5,
    atol=1e-5,
)


def test_validate_model_outputs(orb_model: OrbModel, device: torch.device) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(orb_model, device, torch.float32)
