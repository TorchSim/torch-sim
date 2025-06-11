import pytest
import torch

from tests.models.conftest import make_validate_model_outputs_test


try:
    from huggingface_hub.utils._auth import get_token

    from torch_sim.models.fairchem import FairChemModel

except ImportError:
    pytest.skip("FairChem not installed", allow_module_level=True)


@pytest.fixture
def eqv2_uma_model_pbc(device: torch.device) -> FairChemModel:
    """Use the UMA model which is available in fairchem-core-2.2.0+."""
    cpu = device.type == "cpu"
    return FairChemModel(model=None, model_name="uma-s-1", task_name="omat", cpu=cpu)


@pytest.fixture
def eqv2_uma_model_non_pbc(device: torch.device) -> FairChemModel:
    """Use the UMA model for non-PBC systems."""
    cpu = device.type == "cpu"
    return FairChemModel(model=None, model_name="uma-s-1", task_name="omat", cpu=cpu)


# Removed calculator consistency tests since we're using predictor interface only


test_fairchem_uma_model_outputs = pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)(make_validate_model_outputs_test(model_fixture_name="eqv2_uma_model_pbc"))
