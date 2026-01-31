import traceback

import pytest
import torch

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS


try:
    import sevenn.util
    from sevenn.calculator import SevenNetCalculator
    from sevenn.nn.sequential import AtomGraphSequential

    from torch_sim.models.sevennet import SevenNetModel

except ImportError:
    pytest.skip(
        f"sevenn not installed: {traceback.format_exc()}", allow_module_level=True
    )


model_name = "sevennet-mf-ompa"
modal_name = "mpa"
DTYPE = torch.float32


@pytest.fixture
def pretrained_sevenn_model():
    """Load a pretrained SevenNet model for testing."""
    cp = sevenn.util.load_checkpoint(model_name)

    model_loaded = cp.build_model()
    model_loaded.set_is_batch_data(True)

    return model_loaded.to(DEVICE)


@pytest.fixture
def sevenn_model(pretrained_sevenn_model: AtomGraphSequential) -> SevenNetModel:
    """Create an SevenNetModel wrapper for the pretrained model."""
    return SevenNetModel(model=pretrained_sevenn_model, modal=modal_name, device=DEVICE)


@pytest.fixture
def sevenn_calculator() -> SevenNetCalculator:
    """Create an SevenNetCalculator for the pretrained model."""
    return SevenNetCalculator(model_name, modal=modal_name, device=DEVICE)


def test_sevennet_initialization(pretrained_sevenn_model: AtomGraphSequential) -> None:
    """Test that the SevenNet model initializes correctly."""
    model = SevenNetModel(model=pretrained_sevenn_model, modal="omat24", device=DEVICE)
    # Check that properties were set correctly
    assert model.modal == "omat24"
    assert model.device == DEVICE


# NOTE: we do not test on the molecule sim states due to eps volume giving numerically
# unstable stress off diagonal in xy. See: https://github.com/MDIL-SNU/SevenNet/issues/212
test_sevennet_consistency = make_model_calculator_consistency_test(
    test_name="sevennet",
    model_fixture_name="sevenn_model",
    calculator_fixture_name="sevenn_calculator",
    sim_state_names=tuple(SIMSTATE_BULK_GENERATORS.keys()),
    dtype=DTYPE,
)


test_sevennet_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="sevenn_model", dtype=DTYPE
)
