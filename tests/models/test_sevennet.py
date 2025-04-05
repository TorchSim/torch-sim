import pytest
import torch

from torch_sim.io import state_to_atoms
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


@pytest.mark.parametrize(
    "sim_state_name",
    [
        "cu_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        "benzene_sim_state",
    ],
)
def test_sevennet_calculator_consistency(
    sim_state_name: str,
    sevenn_model: SevenNetModel,
    sevenn_calculator: SevenNetCalculator,
    request: pytest.FixtureRequest,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test consistency between SevenNetModel and SevenNetCalculator for all sim states.

    Args:
        sim_state_name: Name of the sim_state fixture to test
        sevenn_model: The SevenNet model to test
        sevenn_calculator: The SevenNet calculator to test
        request: Pytest fixture request object to get dynamic fixtures
        device: Device to run tests on
        dtype: Data type to use
    """
    # Get the sim_state fixture dynamically using the name
    sim_state = request.getfixturevalue(sim_state_name).to(device, dtype)

    # Set up ASE calculator
    atoms = state_to_atoms(sim_state)[0]
    atoms.calc = sevenn_calculator

    # Get SevenNetModel results
    sevenn_results = sevenn_model(sim_state)

    # Get calculator results
    calc_energy = atoms.get_potential_energy()
    calc_forces = torch.tensor(
        atoms.get_forces(),
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
    sevenn_model: SevenNetModel, device: torch.device, dtype: torch.dtype
) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(sevenn_model, device, dtype)
