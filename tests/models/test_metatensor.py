import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    from metatensor.torch.atomistic import ase_calculator
    from metatrain.utils.io import load_model

    from torch_sim.models.metatensor import MetatensorModel
except ImportError:
    pytest.skip("Metatensor not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def metatensor_calculator(device: torch.device):
    """Load a pretrained metatensor model for testing."""
    return ase_calculator.MetatensorCalculator(
        model=load_model(
            "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"
        ).export(),
        device=device,
    )


@pytest.fixture
def metatensor_model(device: torch.device) -> MetatensorModel:
    """Create an MetatensorModel wrapper for the pretrained model."""
    return MetatensorModel(
        model="pet-mad",
        device=device,
    )


def test_metatensor_initialization(device: torch.device) -> None:
    """Test that the metatensor model initializes correctly."""
    model = MetatensorModel(
        model="pet-mad",
        device=device,
    )
    assert model.device == device
    assert model.dtype == torch.float32


test_mattersim_consistency = make_model_calculator_consistency_test(
    test_name="metatensor",
    model_fixture_name="metatensor_model",
    calculator_fixture_name="metatensor_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
)

test_mattersim_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="metatensor_model",
)
