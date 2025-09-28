import pytest
import torch

from tests.conftest import DEVICE
from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    from metatomic.torch import ase_calculator
    from metatrain.utils.io import load_model

    from torch_sim.models.metatomic import MetatomicModel
except ImportError:
    pytest.skip("metatomic not installed", allow_module_level=True)


@pytest.fixture
def metatomic_calculator():
    """Load a pretrained metatomic model for testing."""
    model_url = "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
    return ase_calculator.MetatomicCalculator(
        model=load_model(model_url).export(), device=DEVICE
    )


@pytest.fixture
def metatomic_model() -> MetatomicModel:
    """Create an MetatomicModel wrapper for the pretrained model."""
    return MetatomicModel(model="pet-mad", device=DEVICE)


def test_metatomic_initialization() -> None:
    """Test that the metatomic model initializes correctly."""
    model = MetatomicModel(
        model="pet-mad",
        device=DEVICE,
    )
    assert model.device == DEVICE
    assert model.dtype == torch.float32


test_metatomic_consistency = make_model_calculator_consistency_test(
    test_name="metatomic",
    model_fixture_name="metatomic_model",
    calculator_fixture_name="metatomic_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
    energy_atol=5e-5,
)

test_metatomic_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="metatomic_model",
    dtype=torch.float32,
    device=DEVICE,
)
