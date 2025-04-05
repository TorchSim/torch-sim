import pytest
import torch

from tests.conftest import make_model_calculator_consistency_test
from torch_sim.models.interface import validate_model_outputs


try:
    import sevenn.util
    from sevenn.calculator import SevenNetCalculator

    from torch_sim.models.sevennet import SevenNetModel

except ImportError:
    pytest.skip("sevenn not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def model_name() -> str:
    """Fixture to provide the model name for testing."""
    return "sevennet-mf-ompa"


@pytest.fixture
def pretrained_sevenn_model(device: torch.device, model_name: str):
    """Load a pretrained SevenNet model for testing."""
    cp = sevenn.util.load_checkpoint(model_name)

    backend = "e3nn"
    model_loaded = cp.build_model(backend)
    model_loaded.set_is_batch_data(True)

    return model_loaded.to(device)


@pytest.fixture
def sevenn_model(
    pretrained_sevenn_model: torch.nn.Module, device: torch.device
) -> SevenNetModel:
    """Create an SevenNetModel wrapper for the pretrained model."""
    return SevenNetModel(
        model=pretrained_sevenn_model,
        modal="mpa",
        device=device,
    )


@pytest.fixture
def sevenn_calculator(device: torch.device, model_name: str) -> SevenNetCalculator:
    """Create an SevenNetCalculator for the pretrained model."""
    return SevenNetCalculator(model_name, modal="mpa", device=device)


def test_sevennet_initialization(
    pretrained_sevenn_model: torch.nn.Module, device: torch.device
) -> None:
    """Test that the SevenNet model initializes correctly."""
    model = SevenNetModel(
        model=pretrained_sevenn_model,
        modal="omat24",
        device=device,
    )
    # Check that properties were set correctly
    assert model.modal == "omat24"
    assert model._device == device  # noqa: SLF001


test_sevennet_consistency = make_model_calculator_consistency_test(
    test_name="sevennet",
    model_fixture_name="sevenn_model",
    calculator_fixture_name="sevenn_calculator",
    sim_state_names=[
        "cu_sim_state",
        "mg_sim_state",
        "sb_sim_state",
        "tio2_sim_state",
        "ga_sim_state",
        "niti_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        "benzene_sim_state",
    ],
    rtol=1e-5,
    atol=1e-5,
)


def test_validate_model_outputs(
    sevenn_model: SevenNetModel, device: torch.device, dtype: torch.dtype
) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(sevenn_model, device, dtype)
