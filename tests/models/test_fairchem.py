import pytest
import torch

from torch_sim.io import state_to_atoms
from torch_sim.models.interface import validate_model_outputs


try:
    from fairchem.core import OCPCalculator
    from fairchem.core.models.model_registry import model_name_to_local_file

    from torch_sim.models.fairchem import FairChemModel

except ImportError:
    pytest.skip("FairChem not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp_path = tmp_path_factory.mktemp("fairchem_checkpoints")
    return model_name_to_local_file(
        "EquiformerV2-31M-S2EF-OC20-All+MD", local_cache=str(tmp_path)
    )


@pytest.fixture
def fairchem_model(model_path: str, device: torch.device) -> FairChemModel:
    cpu = device.type == "cpu"
    return FairChemModel(
        model=model_path,
        cpu=cpu,
        seed=0,
    )


@pytest.fixture
def ocp_calculator(model_path: str) -> OCPCalculator:
    return OCPCalculator(checkpoint_path=model_path, cpu=False, seed=0)


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
def test_fairchem_ocp_consistency(
    sim_state_name: str,
    fairchem_model: FairChemModel,
    ocp_calculator: OCPCalculator,
    request: pytest.FixtureRequest,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test consistency between FairChemModel and OCPCalculator for all sim states.

    Args:
        sim_state_name: Name of the sim_state fixture to test
        fairchem_model: The FairChem model to test
        ocp_calculator: The OCP calculator to test
        request: Pytest fixture request object to get dynamic fixtures
        device: Device to run tests on
        dtype: Data type to use
    """
    # Get the sim_state fixture dynamically using the name
    sim_state = request.getfixturevalue(sim_state_name).to(device, dtype)

    # Set up ASE calculator
    atoms = state_to_atoms(sim_state)[0]
    atoms.calc = ocp_calculator

    # Get FairChem results
    fairchem_results = fairchem_model(sim_state)

    # Get OCP results
    ocp_forces = torch.tensor(
        atoms.get_forces(),
        device=device,
        dtype=fairchem_results["forces"].dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        fairchem_results["energy"].item(),
        atoms.get_potential_energy(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"Energy mismatch for {sim_state_name}",
    )
    torch.testing.assert_close(
        fairchem_results["forces"],
        ocp_forces,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Forces mismatch for {sim_state_name}",
    )


# fairchem batching is broken on CPU, do not replicate this skipping
# logic in other models tests
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Batching does not work properly on CPU for FAIRchem",
)
def test_validate_model_outputs(
    fairchem_model: FairChemModel, device: torch.device
) -> None:
    validate_model_outputs(fairchem_model, device, torch.float32)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
