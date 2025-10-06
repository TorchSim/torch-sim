import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
    make_validate_single_system_model_outputs_test,
)


try:
    from metatomic.torch import ase_calculator
    from metatrain.utils.io import load_model

    from torch_sim.models.metatomic import MetatomicModel
except ImportError:
    pytest.skip("metatomic not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def metatomic_calculator(device: torch.device):
    """Load a pretrained metatomic model for testing."""
    return ase_calculator.MetatomicCalculator(
        model=load_model(
            "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
        ).export(),
        device=device,
    )


@pytest.fixture
def metatomic_model(device: torch.device) -> MetatomicModel:
    """Create an MetatomicModel wrapper for the pretrained model."""
    return MetatomicModel(
        model="pet-mad",
        device=device,
    )


def test_metatomic_initialization(device: torch.device) -> None:
    """Test that the metatomic model initializes correctly."""
    model = MetatomicModel(
        model="pet-mad",
        device=device,
    )
    assert model.device == device
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
)

test_metatomic_model_single_output = make_validate_single_system_model_outputs_test(
    model_fixture_name="metatomic_model",
)