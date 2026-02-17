import traceback

import pytest

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_GENERATORS


try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    from torch_sim.models.orb import OrbModel
except ImportError:
    pytest.skip(f"ORB not installed: {traceback.format_exc()}", allow_module_level=True)


@pytest.fixture
def orbv3_conservative_inf_omat_model() -> OrbModel:
    orb_ff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
        device=DEVICE, precision="float32-high"
    )
    return OrbModel(orb_ff, atoms_adapter, device=DEVICE)


@pytest.fixture
def orbv3_direct_20_omat_model() -> OrbModel:
    orb_ff, atoms_adapter = pretrained.orb_v3_direct_20_omat(
        device=DEVICE, precision="float32-high"
    )
    return OrbModel(orb_ff, atoms_adapter, device=DEVICE)


@pytest.fixture
def orbv3_conservative_inf_omat_calculator() -> ORBCalculator:
    """Create an ORBCalculator for the pretrained model."""
    orb_ff, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
        device=DEVICE, precision="float32-high"
    )
    return ORBCalculator(model=orb_ff, atoms_adapter=atoms_adapter, device=DEVICE)


@pytest.fixture
def orbv3_direct_20_omat_calculator() -> ORBCalculator:
    """Create an ORBCalculator for the pretrained model."""
    orb_ff, atoms_adapter = pretrained.orb_v3_direct_20_omat(
        device=DEVICE, precision="float32-high"
    )
    return ORBCalculator(model=orb_ff, atoms_adapter=atoms_adapter, device=DEVICE)


test_orb_conservative_consistency = make_model_calculator_consistency_test(
    test_name="orbv3_conservative_inf_omat",
    model_fixture_name="orbv3_conservative_inf_omat_model",
    calculator_fixture_name="orbv3_conservative_inf_omat_calculator",
    sim_state_names=tuple(SIMSTATE_GENERATORS.keys()),
    energy_rtol=5e-5,
    energy_atol=5e-5,
)

test_orb_direct_consistency = make_model_calculator_consistency_test(
    test_name="orbv3_direct_20_omat",
    model_fixture_name="orbv3_direct_20_omat_model",
    calculator_fixture_name="orbv3_direct_20_omat_calculator",
    sim_state_names=tuple(SIMSTATE_GENERATORS.keys()),
    energy_rtol=5e-5,
    energy_atol=5e-5,
)

test_validate_conservative_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="orbv3_conservative_inf_omat_model",
)

test_validate_direct_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="orbv3_direct_20_omat_model",
)
