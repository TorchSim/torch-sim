import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


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
def fairchem_model_pbc(model_path: str, device: torch.device) -> FairChemModel:
    cpu = device.type == "cpu"
    return FairChemModel(
        model=model_path,
        cpu=cpu,
        seed=0,
        pbc=True,
    )


@pytest.fixture
def ocp_calculator(model_path: str) -> OCPCalculator:
    return OCPCalculator(checkpoint_path=model_path, cpu=False, seed=0)


test_fairchem_ocp_consistency_pbc = make_model_calculator_consistency_test(
    test_name="fairchem_ocp",
    model_fixture_name="fairchem_model_pbc",
    calculator_fixture_name="ocp_calculator",
    sim_state_names=consistency_test_simstate_fixtures[:-1],
    rtol=5e-4,  # NOTE: fairchem doesn't pass at the 1e-5 level used for other models
    atol=5e-4,
)

# TODO: add test for non-PBC model

# fairchem batching is broken on CPU, do not replicate this skipping
# logic in other models tests. This is due to issues with how the models
# handle supercells (see related issue here: https://github.com/FAIR-Chem/fairchem/issues/428)
test_fairchem_ocp_model_outputs = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Batching does not work properly on CPU for FAIRchem",
)(
    make_validate_model_outputs_test(
        model_fixture_name="fairchem_model",
    )
)
