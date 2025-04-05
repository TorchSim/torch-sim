import pytest
import torch

from torch_sim.io import state_to_atoms
from torch_sim.models.interface import validate_model_outputs


try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.atomic_system import SystemConfig
    from orb_models.forcefield.calculator import ORBCalculator

    from torch_sim.models.orb import OrbModel
except ImportError:
    pytest.skip("ORB not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def pretrained_orb_model(device: torch.device):
    """Load a pretrained ORB model for testing."""
    return pretrained.orb_v2(device=device)


@pytest.fixture
def orb_model(pretrained_orb_model: torch.nn.Module, device: torch.device) -> OrbModel:
    """Create an OrbModel wrapper for the pretrained model."""
    return OrbModel(
        model=pretrained_orb_model,
        device=device,
        system_config=SystemConfig(radius=6.0, max_num_neighbors=20),
    )


@pytest.fixture
def orb_calculator(
    pretrained_orb_model: torch.nn.Module, device: torch.device
) -> ORBCalculator:
    """Create an ORBCalculator for the pretrained model."""
    return ORBCalculator(
        model=pretrained_orb_model,
        system_config=SystemConfig(radius=6.0, max_num_neighbors=20),
        device=device,
    )


def test_orb_initialization(
    pretrained_orb_model: torch.nn.Module, device: torch.device
) -> None:
    """Test that the ORB model initializes correctly."""
    model = OrbModel(
        model=pretrained_orb_model,
        device=device,
    )
    # Check that properties were set correctly
    assert "energy" in model.implemented_properties
    assert "forces" in model.implemented_properties
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
def test_orb_calculator_consistency(
    sim_state_name: str,
    orb_model: OrbModel,
    orb_calculator: ORBCalculator,
    request: pytest.FixtureRequest,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test consistency between OrbModel and ORBCalculator for all sim states.

    Args:
        sim_state_name: Name of the sim_state fixture to test
        orb_model: The ORB model to test
        orb_calculator: The ORB calculator to test
        request: Pytest fixture request object to get dynamic fixtures
        device: Device to run tests on
        dtype: Data type to use
    """
    # Get the sim_state fixture dynamically using the name
    sim_state = request.getfixturevalue(sim_state_name).to(device, dtype)

    # Set up ASE calculator
    atoms = state_to_atoms(sim_state)[0]
    atoms.calc = orb_calculator

    # Get OrbModel results
    orb_results = orb_model(sim_state)

    # Get calculator results
    calc_energy = atoms.get_potential_energy()
    calc_forces = torch.tensor(
        atoms.get_forces(),
        device=device,
        dtype=orb_results["forces"].dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        orb_results["energy"].item(),
        calc_energy,
        atol=1e-4,
        rtol=0.5,
    )
    torch.testing.assert_close(
        orb_results["forces"],
        calc_forces,
        atol=1e-4,
        rtol=0.5,
    )


def test_validate_model_outputs(orb_model: OrbModel, device: torch.device) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(orb_model, device, torch.float32)
