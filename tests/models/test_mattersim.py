# codespell-ignore: convertor

import traceback

import ase.spacegroup
import ase.units
import pytest

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_GENERATORS


try:
    from mattersim.forcefield import MatterSimCalculator, Potential

    from torch_sim.models.mattersim import MatterSimModel

except ImportError:
    pytest.skip(
        f"mattersim not installed: {traceback.format_exc()}", allow_module_level=True
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
    assert model.stress_weight == ase.units.GPa


test_mattersim_consistency = make_model_calculator_consistency_test(
    test_name="mattersim",
    model_fixture_name="mattersim_model",
    calculator_fixture_name="mattersim_calculator",
    sim_state_names=tuple(SIMSTATE_GENERATORS.keys()),
)

test_mattersim_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="mattersim_model",
)
