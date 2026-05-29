from __future__ import annotations

import traceback

import pytest

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_GENERATORS, ModelTolerance


try:
    from mattersim.forcefield import MatterSimCalculator, Potential

    from torch_sim.models.mattersim import MatterSimModel

    _IMPORT_ERROR: str | None = None
except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"mattersim not installed: {_IMPORT_ERROR}"
)


model_name = "mattersim-v1.0.0-1m.pth"


@pytest.fixture
def pretrained_mattersim_model():
    """Load a pretrained MatterSim model for testing."""
    return Potential.from_checkpoint(
        load_path=model_name,
        model_name="m3gnet",
        device=DEVICE,
        load_training_state=False,
    )


@pytest.fixture
def mattersim_model(pretrained_mattersim_model: Potential) -> MatterSimModel:
    """Create an MatterSimModel wrapper for the pretrained model."""
    return MatterSimModel(model=pretrained_mattersim_model, device=DEVICE)


@pytest.fixture
def mattersim_calculator(pretrained_mattersim_model: Potential) -> MatterSimCalculator:
    """Create an MatterSimCalculator for the pretrained model."""
    return MatterSimCalculator(pretrained_mattersim_model, device=DEVICE)


def test_mattersim_initialization(pretrained_mattersim_model: Potential) -> None:
    """Test that the MatterSim model initializes correctly."""
    model = MatterSimModel(model=pretrained_mattersim_model, device=DEVICE)
    assert model.device == DEVICE


test_mattersim_consistency = make_model_calculator_consistency_test(
    test_name="mattersim",
    model_fixture_name="mattersim_model",
    calculator_fixture_name="mattersim_calculator",
    sim_state_names=tuple(SIMSTATE_GENERATORS.keys()),
    energy_rtol=ModelTolerance.LOOSE,
    energy_atol=ModelTolerance.LOOSE,
    force_rtol=ModelTolerance.LOOSE,
    force_atol=ModelTolerance.STANDARD,
)

test_mattersim_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="mattersim_model",
)
