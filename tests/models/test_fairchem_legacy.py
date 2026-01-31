import os
import traceback

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS, SIMSTATE_MOLECULE_GENERATORS


try:
    from fairchem.core import OCPCalculator
    from fairchem.core.models.model_registry import model_name_to_local_file
    from huggingface_hub.utils._auth import get_token

    from torch_sim.models.fairchem_legacy import FairChemV1Model

except ImportError:
    pytest.skip(
        f"FairChem not installed: {traceback.format_exc()}", allow_module_level=True
    )


@pytest.fixture(scope="session")
def model_path_oc20(tmp_path_factory: pytest.TempPathFactory) -> str:
    tmp_path = tmp_path_factory.mktemp("fairchem_checkpoints")
    model_name = "EquiformerV2-31M-S2EF-OC20-All+MD"
    return model_name_to_local_file(model_name, local_cache=str(tmp_path))


@pytest.fixture
def eqv2_oc20_model_pbc(model_path_oc20: str) -> FairChemV1Model:
    return FairChemV1Model(model=model_path_oc20, device=DEVICE, seed=0, pbc=True)


@pytest.fixture
def eqv2_oc20_model_non_pbc(
    model_path_oc20: str,
) -> FairChemV1Model:
    return FairChemV1Model(model=model_path_oc20, device=DEVICE, seed=0, pbc=False)


if get_token():

    @pytest.fixture(scope="session")
    def model_path_omat24(tmp_path_factory: pytest.TempPathFactory) -> str:
        tmp_path = tmp_path_factory.mktemp("fairchem_checkpoints")
        model_name = "EquiformerV2-31M-OMAT24-MP-sAlex"
        return model_name_to_local_file(model_name, local_cache=str(tmp_path))

    @pytest.fixture
    def eqv2_omat24_model_pbc(
        model_path_omat24: str,
    ) -> FairChemV1Model:
        return FairChemV1Model(model=model_path_omat24, device=DEVICE, seed=0, pbc=True)


@pytest.fixture
def ocp_calculator(model_path_oc20: str) -> OCPCalculator:
    return OCPCalculator(checkpoint_path=model_path_oc20, cpu=False, seed=0)


test_fairchem_ocp_consistency_pbc = make_model_calculator_consistency_test(
    test_name="fairchem_ocp",
    model_fixture_name="eqv2_oc20_model_pbc",
    calculator_fixture_name="ocp_calculator",
    sim_state_names=tuple(SIMSTATE_BULK_GENERATORS.keys()),
    energy_rtol=5e-4,  # NOTE: EqV2 doesn't pass at the 1e-5 level used for other models
    energy_atol=5e-4,
    force_rtol=5e-4,
    force_atol=5e-4,
    stress_rtol=5e-4,
    stress_atol=5e-4,
)

test_fairchem_non_pbc = make_model_calculator_consistency_test(
    test_name="fairchem_non_pbc_benzene",
    model_fixture_name="eqv2_oc20_model_non_pbc",
    calculator_fixture_name="ocp_calculator",
    sim_state_names=tuple(SIMSTATE_MOLECULE_GENERATORS.keys()),
    energy_rtol=5e-4,  # NOTE: EqV2 doesn't pass at the 1e-5 level used for other models
    energy_atol=5e-4,
    force_rtol=5e-4,
    force_atol=5e-4,
    stress_rtol=5e-4,
    stress_atol=5e-4,
)


# Skip this test due to issues with how the older models
# handled supercells (see related issue here: https://github.com/facebookresearch/fairchem/issues/428)

test_fairchem_ocp_model_outputs = pytest.mark.skipif(
    os.environ.get("HF_TOKEN") is None,
    reason="Issues in graph construction of older models",
)(make_validate_model_outputs_test(model_fixture_name="eqv2_omat24_model_pbc"))


def test_fairchem_mixed_pbc_init_raises(model_path_oc20: str) -> None:
    """Test that initializing FairChemV1Model with mixed PBC raises ValueError."""
    mixed_pbc = torch.tensor([True, False, True], dtype=torch.bool)
    with pytest.raises(ValueError, match="FairChemV1Model does not support mixed PBC"):
        FairChemV1Model(model=model_path_oc20, device=DEVICE, seed=0, pbc=mixed_pbc)


def test_fairchem_mixed_pbc_forward_raises(
    eqv2_oc20_model_pbc: FairChemV1Model, si_sim_state: ts.SimState
) -> None:
    """Test that calling forward with a SimState that has mixed PBC raises ValueError."""
    mixed_pbc_state = si_sim_state.clone()
    mixed_pbc_state.pbc = torch.tensor([True, False, True], dtype=torch.bool)
    with pytest.raises(ValueError, match="FairChemV1Model does not support mixed PBC"):
        eqv2_oc20_model_pbc(mixed_pbc_state)
