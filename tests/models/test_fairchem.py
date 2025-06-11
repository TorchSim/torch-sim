import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from huggingface_hub.utils._auth import get_token

    from torch_sim.models.fairchem import FairChemModel

except ImportError:
    pytest.skip("FairChem not installed", allow_module_level=True)


@pytest.fixture
def eqv2_uma_model_pbc(device: torch.device) -> FairChemModel:
    """Use the UMA model which is available in fairchem-core-2.2.0+."""
    cpu = device.type == "cpu"
    return FairChemModel(
        model=None, model_name="uma-s-1", task_name="omat", cpu=cpu, seed=0
    )


@pytest.fixture
def eqv2_uma_model_non_pbc(device: torch.device) -> FairChemModel:
    """Use the UMA model for non-PBC systems."""
    cpu = device.type == "cpu"
    return FairChemModel(
        model=None, model_name="uma-s-1", task_name="omat", cpu=cpu, seed=0
    )


@pytest.fixture
def fairchem_calculator() -> FAIRChemCalculator:
    """FAIRChemCalculator using the UMA model."""
    return FAIRChemCalculator.from_model_checkpoint(
        name_or_path="uma-s-1",
        task_name="omat",
        device="cpu",
        seed=0,
    )


test_fairchem_ocp_consistency_pbc = make_model_calculator_consistency_test(
    test_name="fairchem_uma",
    model_fixture_name="eqv2_uma_model_pbc",
    calculator_fixture_name="fairchem_calculator",
    sim_state_names=consistency_test_simstate_fixtures[:-1],
    energy_rtol=5e-4,  # NOTE: UMA model tolerances
    energy_atol=5e-4,
    force_rtol=5e-4,
    force_atol=5e-4,
    stress_rtol=5e-4,
    stress_atol=5e-4,
)

test_fairchem_non_pbc_benzene = make_model_calculator_consistency_test(
    test_name="fairchem_non_pbc_benzene",
    model_fixture_name="eqv2_uma_model_non_pbc",
    calculator_fixture_name="fairchem_calculator",
    sim_state_names=["benzene_sim_state"],
    energy_rtol=5e-4,  # NOTE: UMA model tolerances
    energy_atol=5e-4,
    force_rtol=5e-4,
    force_atol=5e-4,
    stress_rtol=5e-4,
    stress_atol=5e-4,
)


test_fairchem_uma_model_outputs = pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)(make_validate_model_outputs_test(model_fixture_name="eqv2_uma_model_pbc"))
