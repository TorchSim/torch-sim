import os

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
    from huggingface_hub.utils._auth import get_token

    from torch_sim.models.fairchem import FairChemModel

except ImportError:
    pytest.skip("FairChem not installed", allow_module_level=True)


@pytest.fixture(scope="session")
def model_path_oc20(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp_path = tmp_path_factory.mktemp("fairchem_checkpoints")
    model_name = "EquiformerV2-31M-S2EF-OC20-All+MD"
    return model_name_to_local_file(model_name, local_cache=str(tmp_path))


@pytest.fixture
def eqv2_oc20_model_pbc(model_path_oc20: str, device: torch.device) -> FairChemModel:
    cpu = device.type == "cpu"
    return FairChemModel(
        model=model_path_oc20,
        cpu=cpu,
        seed=0,
        pbc=True,
    )


@pytest.fixture
def eqv2_oc20_model_non_pbc(model_path_oc20: str, device: torch.device) -> FairChemModel:
    cpu = device.type == "cpu"
    return FairChemModel(
        model=model_path_oc20,
        cpu=cpu,
        seed=0,
        pbc=False,
    )


if get_token():

    @pytest.fixture(scope="session")
    def model_path_omat24(tmp_path_factory: pytest.TempPathFactory) -> str:
        tmp_path = tmp_path_factory.mktemp("fairchem_checkpoints")
        model_name = "EquiformerV2-31M-OMAT24-MP-sAlex"
        return model_name_to_local_file(model_name, local_cache=str(tmp_path))

    @pytest.fixture
    def eqv2_omat24_model_pbc(
        model_path_omat24: str, device: torch.device
    ) -> FairChemModel:
        cpu = device.type == "cpu"
        return FairChemModel(
            model=model_path_omat24,
            cpu=cpu,
            seed=0,
            pbc=True,
        )


@pytest.fixture
def ocp_calculator(model_path_oc20: str) -> OCPCalculator:
    return OCPCalculator(checkpoint_path=model_path_oc20, cpu=False, seed=0)


test_fairchem_ocp_consistency_pbc = make_model_calculator_consistency_test(
    test_name="fairchem_ocp",
    model_fixture_name="eqv2_oc20_model_pbc",
    calculator_fixture_name="ocp_calculator",
    sim_state_names=consistency_test_simstate_fixtures[:-1],
    rtol=5e-4,  # NOTE: EqV2 doesn't pass at the 1e-5 level used for other models
    atol=5e-4,
)

test_fairchem_non_pbc_benzene = make_model_calculator_consistency_test(
    test_name="fairchem_non_pbc_benzene",
    model_fixture_name="eqv2_oc20_model_non_pbc",
    calculator_fixture_name="ocp_calculator",
    sim_state_names=["benzene_sim_state"],
    rtol=5e-4,  # NOTE: EqV2 doesn't pass at the 1e-5 level used for other models
    atol=5e-4,
)


# Skip this test due to issues with how the older models
# handled supercells (see related issue here: https://github.com/FAIR-Chem/fairchem/issues/428)

test_fairchem_ocp_model_outputs = pytest.mark.skipif(
    os.environ.get("HF_TOKEN") is None,
    reason="Issues in graph construction of older models",
)(
    make_validate_model_outputs_test(
        model_fixture_name="eqv2_omat24_model_pbc",
    )
)
