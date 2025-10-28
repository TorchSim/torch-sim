import traceback

import pytest
import torch

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)


try:
    from chgnet.model.dynamics import CHGNetCalculator as CHGNetAseCalculator

    from torch_sim.models.chgnet import CHGNetModel
except (ImportError, ValueError):
    pytest.skip(
        f"CHGNet not installed: {traceback.format_exc()}", allow_module_level=True
    )


DTYPE = torch.float32


@pytest.fixture
def ts_chgnet_model() -> CHGNetModel:
    """Create a TorchSim CHGNet model for testing."""
    return CHGNetModel(
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def ase_chgnet_calculator() -> CHGNetAseCalculator:
    """Create an ASE CHGNet calculator for testing."""
    # Use the official CHGNet calculator
    return CHGNetAseCalculator()


test_chgnet_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_chgnet_model", dtype=DTYPE
)

test_chgnet_consistency = make_model_calculator_consistency_test(
    test_name="chgnet",
    model_fixture_name="ts_chgnet_model",
    calculator_fixture_name="ase_chgnet_calculator",
    sim_state_names=("si_sim_state", "cu_sim_state", "mg_sim_state", "ti_sim_state"),
    dtype=DTYPE,
    energy_rtol=1e-4,
    energy_atol=1e-4,
    force_rtol=1e-4,
    force_atol=1e-4,
    stress_rtol=1e-3,
    stress_atol=1e-3,
)
