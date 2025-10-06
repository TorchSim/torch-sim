# codespell-ignore: convertor

import ase.spacegroup
import ase.units
import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
    make_validate_single_system_model_outputs_test,
)


try:
    from mattersim.forcefield import MatterSimCalculator, Potential

    from torch_sim.models.mattersim import MatterSimModel

except ImportError:
    pytest.skip("mattersim not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def model_name() -> str:
    """Fixture to provide the model name for testing. Load smaller 1M model
    for testing purposes.
    """
    return "mattersim-v1.0.0-1m.pth"


@pytest.fixture
def pretrained_mattersim_model(device: torch.device, model_name: str):
    """Load a pretrained MatterSim model for testing."""
    return Potential.from_checkpoint(
        load_path=model_name,
        model_name="m3gnet",
        device=device,
        load_training_state=False,
    )


@pytest.fixture
def mattersim_model(
    pretrained_mattersim_model: Potential, device: torch.device
) -> MatterSimModel:
    """Create an MatterSimModel wrapper for the pretrained model."""
    return MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )


@pytest.fixture
def mattersim_calculator(
    pretrained_mattersim_model: Potential, device: torch.device
) -> MatterSimCalculator:
    """Create an MatterSimCalculator for the pretrained model."""
    return MatterSimCalculator(pretrained_mattersim_model, device=device)


def test_mattersim_initialization(
    pretrained_mattersim_model: Potential, device: torch.device
) -> None:
    """Test that the MatterSim model initializes correctly."""
    model = MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )
    assert model._device == device  # noqa: SLF001
    assert model.stress_weight == ase.units.GPa


test_mattersim_consistency = make_model_calculator_consistency_test(
    test_name="mattersim",
    model_fixture_name="mattersim_model",
    calculator_fixture_name="mattersim_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
)

test_mattersim_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="mattersim_model",
)

test_mace_off_model_single_output = make_validate_single_system_model_outputs_test(
    model_fixture_name="mattersim_model"
)
