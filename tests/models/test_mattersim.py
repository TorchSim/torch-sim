# codespell-ignore: convertor

import ase.spacegroup
import ase.units
import pytest
import torch

from tests.conftest import make_model_calculator_consistency_test
from torch_sim.models.interface import validate_model_outputs


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
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
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
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
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
    sim_state_names=[
        "cu_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        "benzene_sim_state",
    ],
    rtol=1e-5,
    atol=1e-5,
)


def test_validate_model_outputs(
    mattersim_model: MatterSimModel, device: torch.device
) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(mattersim_model, device, torch.float32)
