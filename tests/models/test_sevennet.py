import pytest
import torch
from ase.build import bulk

from torch_sim.io import atoms_to_state
from torch_sim.models.interface import validate_model_outputs
from torch_sim.state import SimState


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
def cu_system(dtype: torch.dtype, device: torch.device) -> SimState:
    # Create FCC Copper
    cu_fcc = bulk("Cu", "fcc", a=3.58, cubic=True)
    return atoms_to_state([cu_fcc], device, dtype)


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


def test_sevennet_calculator_consistency(
    sevenn_model: SevenNetModel,
    sevenn_calculator: SevenNetCalculator,
    cu_system: SimState,
    device: torch.device,
) -> None:
    """Test consistency between SevenNetModel and SevenNetCalculator."""
    # Get SevenNetModel results
    sevenn_results = sevenn_model(cu_system)

    # Set up ASE calculator
    cu_fcc = bulk("Cu", "fcc", a=3.58, cubic=True)
    cu_fcc.calc = sevenn_calculator

    # Get calculator results
    calc_energy = cu_fcc.get_potential_energy()
    calc_forces = torch.tensor(
        cu_fcc.get_forces(),
        device=device,
        dtype=sevenn_results["forces"].dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        sevenn_results["energy"].item(),
        calc_energy,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        sevenn_results["forces"],
        calc_forces,
        rtol=1e-5,
        atol=1e-5,
    )


def test_validate_model_outputs(
    sevenn_model: SevenNetModel, device: torch.device
) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(sevenn_model, device, torch.float32)
