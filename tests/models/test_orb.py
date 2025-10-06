import pytest
import torch

from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
    make_validate_single_system_model_outputs_test,
)


try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    from torch_sim.models.orb import OrbModel
except ImportError:
    pytest.skip("ORB not installed", allow_module_level=True)


@pytest.fixture
def orbv3_conservative_inf_omat_model(device: torch.device) -> OrbModel:
    orb_ff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision="float32-high",
    )
    return OrbModel(model=orb_ff, device=device)


@pytest.fixture
def orbv3_direct_20_omat_model(device: torch.device) -> OrbModel:
    orb_ff = pretrained.orb_v3_direct_20_omat(
        device=device,
        precision="float32-high",
    )
    return OrbModel(model=orb_ff, device=device)


@pytest.fixture
def orbv3_conservative_inf_omat_calculator(device: torch.device) -> ORBCalculator:
    """Create an ORBCalculator for the pretrained model."""
    orb_ff = pretrained.orb_v3_conservative_inf_omat(
        device=device,
        precision="float32-high",
    )
    return ORBCalculator(model=orb_ff, device=device)


@pytest.fixture
def orbv3_direct_20_omat_calculator(device: torch.device) -> ORBCalculator:
    """Create an ORBCalculator for the pretrained model."""
    orb_ff = pretrained.orb_v3_direct_20_omat(
        device=device,
        precision="float32-high",
    )
    return ORBCalculator(model=orb_ff, device=device)


test_orb_conservative_consistency = make_model_calculator_consistency_test(
    test_name="orbv3_conservative_inf_omat",
    model_fixture_name="orbv3_conservative_inf_omat_model",
    calculator_fixture_name="orbv3_conservative_inf_omat_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
    energy_rtol=5e-5,
    energy_atol=5e-5,
)

test_orb_direct_consistency = make_model_calculator_consistency_test(
    test_name="orbv3_direct_20_omat",
    model_fixture_name="orbv3_direct_20_omat_model",
    calculator_fixture_name="orbv3_direct_20_omat_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
    energy_rtol=5e-5,
    energy_atol=5e-5,
)

test_validate_conservative_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="orbv3_conservative_inf_omat_model",
)

test_validate_direct_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="orbv3_direct_20_omat_model",
)

test_validate_conservative_model_single_output = (
    make_validate_single_system_model_outputs_test(
        model_fixture_name="orbv3_conservative_inf_omat_model",
    )
)

test_validate_direct_model_single_output = make_validate_single_system_model_outputs_test(
    model_fixture_name="orbv3_direct_20_omat_model",
)
