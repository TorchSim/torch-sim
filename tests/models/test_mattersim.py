# codespell-ignore: convertor

import ase.spacegroup
import ase.units
import pytest
import torch

from torch_sim.io import state_to_atoms
from torch_sim.models.interface import validate_model_outputs


try:
    from mattersim.forcefield import MatterSimCalculator, Potential

    from torch_sim.models.mattersim import MatterSimModel

except ImportError:
    pytest.skip("mattersim not installed", allow_module_level=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def model_name() -> str:
    """Fixture to provide the model name for testing. Load smaller 1M model
    for testing purposes.
    """
    return "mattersim-v1.0.0-1m.pth"


@pytest.fixture
def pretrained_mattersim_model(device: torch.device, model_name: str):
    """Load a pretrained MatterSim model for testing."""
    return Potential.from_checkpoint(
        load_path=model_name,
        model_name="m3gnet",
        device=device,
        load_training_state=False,
    )


@pytest.fixture
def mattersim_model(
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
) -> MatterSimModel:
    """Create an MatterSimModel wrapper for the pretrained model."""
    return MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )


@pytest.fixture
def mattersim_calculator(
    pretrained_mattersim_model: Potential, device: torch.device
) -> MatterSimCalculator:
    """Create an MatterSimCalculator for the pretrained model."""
    return MatterSimCalculator(pretrained_mattersim_model, device=device)


def test_mattersim_initialization(
    pretrained_mattersim_model: torch.nn.Module, device: torch.device
) -> None:
    """Test that the MatterSim model initializes correctly."""
    model = MatterSimModel(
        model=pretrained_mattersim_model,
        device=device,
    )
    assert model._device == device  # noqa: SLF001
    assert model.stress_weight == ase.units.GPa


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
def test_mattersim_calculator_consistency(
    sim_state_name: str,
    mattersim_model: MatterSimModel,
    mattersim_calculator: MatterSimCalculator,
    request: pytest.FixtureRequest,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test consistency between MatterSimModel and MatterSimCalculator for all sim states.

    Args:
        sim_state_name: Name of the sim_state fixture to test
        mattersim_model: The MatterSim model to test
        mattersim_calculator: The MatterSim calculator to test
        request: Pytest fixture request object to get dynamic fixtures
        device: Device to run tests on
    """
    # Get the sim_state fixture dynamically using the name
    sim_state = request.getfixturevalue(sim_state_name).to(device, dtype)

    # Set up ASE calculator
    atoms = state_to_atoms(sim_state)[0]
    atoms.calc = mattersim_calculator

    # Get MatterSimModel results
    mattersim_results = mattersim_model(sim_state)

    # Get calculator results
    calc_energy = atoms.get_potential_energy()
    calc_forces = torch.tensor(
        atoms.get_forces(),
        device=device,
        dtype=mattersim_results["forces"].dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        mattersim_results["energy"].item(),
        calc_energy,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        mattersim_results["forces"],
        calc_forces,
        rtol=1e-5,
        atol=1e-5,
    )


def test_validate_model_outputs(
    mattersim_model: MatterSimModel, device: torch.device
) -> None:
    """Test that the model passes the standard validation."""
    validate_model_outputs(mattersim_model, device, torch.float32)
