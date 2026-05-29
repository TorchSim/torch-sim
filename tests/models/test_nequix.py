from __future__ import annotations

import traceback

import pytest

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS, ModelTolerance


try:
    from nequix.calculator import NequixCalculator

    from torch_sim.models.nequix import NequixModel

    _IMPORT_ERROR: str | None = None
except (ImportError, ModuleNotFoundError):
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"nequix not installed: {_IMPORT_ERROR}"
)


@pytest.fixture(scope="session")
def nequix_model() -> NequixModel:
    return NequixModel("nequix-mp-1", device=DEVICE, dtype=DTYPE, use_kernel=False)


@pytest.fixture(scope="session")
def nequix_calculator() -> NequixCalculator:
    return NequixCalculator(
        "nequix-mp-1",
        device=DEVICE,
        backend="torch",
        use_compile=False,
        use_kernel=False,
    )


test_nequix_consistency = make_model_calculator_consistency_test(
    test_name="nequix",
    model_fixture_name="nequix_model",
    calculator_fixture_name="nequix_calculator",
    sim_state_names=tuple(SIMSTATE_BULK_GENERATORS.keys()),
    force_atol=ModelTolerance.LOOSE,
    dtype=DTYPE,
    device=DEVICE,
)

test_nequix_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="nequix_model",
    dtype=DTYPE,
    device=DEVICE,
)
