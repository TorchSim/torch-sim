import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    import sevenn.util
    from sevenn.calculator import SevenNetCalculator
    from sevenn.nn.sequential import AtomGraphSequential

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
def modal_name() -> str:
    """Fixture to provide the modal name for testing."""
    return "mpa"


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
    pretrained_sevenn_model: AtomGraphSequential, device: torch.device, modal_name: str
) -> SevenNetModel:
    """Create an SevenNetModel wrapper for the pretrained model."""
    return SevenNetModel(
        model=pretrained_sevenn_model,
        modal=modal_name,
        device=device,
    )


@pytest.fixture
def sevenn_calculator(
    device: torch.device, model_name: str, modal_name: str
) -> SevenNetCalculator:
    """Create an SevenNetCalculator for the pretrained model."""
    return SevenNetCalculator(model_name, modal=modal_name, device=device)


def test_sevennet_initialization(
    pretrained_sevenn_model: AtomGraphSequential, device: torch.device
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


# NOTE: we take [:-1] to skipbenzene due to eps volume giving numerically
# unstable stress off diagonal in xy. See: https://github.com/MDIL-SNU/SevenNet/issues/212
test_sevennet_consistency = make_model_calculator_consistency_test(
    test_name="sevennet",
    model_fixture_name="sevenn_model",
    calculator_fixture_name="sevenn_calculator",
    sim_state_names=consistency_test_simstate_fixtures[:-1],
)


test_sevennet_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="sevenn_model",
)
