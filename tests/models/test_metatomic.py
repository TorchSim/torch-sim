import traceback

import pytest
import torch

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_GENERATORS


try:
    import requests
    from metatomic.torch import AtomisticModel, load_atomistic_model
    from metatomic_ase import MetatomicCalculator

    from torch_sim.models.metatomic import MetatomicModel
except ImportError:
    pytest.skip(
        f"metatomic not installed: {traceback.format_exc()}",
        allow_module_level=True,
    )


@pytest.fixture
def metatomic_module(tmp_path):
    model_url = "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt"
    model_path = tmp_path / "pet-mad-v1.1.0.ckpt"
    response = requests.get(model_url)
    response.raise_for_status()
    model_path.write_bytes(response.content)
    return load_atomistic_model(model_path)


@pytest.fixture
def metatomic_calculator(metatomic_module: AtomisticModel) -> MetatomicCalculator:
    """Load a pretrained metatomic model for testing."""
    return MetatomicCalculator(model=metatomic_module, device=DEVICE)


@pytest.fixture
def metatomic_model(metatomic_module: AtomisticModel) -> MetatomicModel:
    """Create an MetatomicModel wrapper for the pretrained model."""
    return MetatomicModel(model=metatomic_module, device=DEVICE)


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
    sim_state_names=tuple(SIMSTATE_GENERATORS.keys()),
    energy_atol=5e-5,
    dtype=torch.float32,
    device=DEVICE,
)

test_metatomic_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="metatomic_model",
    dtype=torch.float32,
    device=DEVICE,
)
