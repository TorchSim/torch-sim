import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


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
    sim_state_names=consistency_test_simstate_fixtures,
    atol=5e-4,
    rtol=1,
)


test_validate_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="orb_model",
)
