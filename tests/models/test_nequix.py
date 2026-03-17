import traceback

import pytest

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS


try:
    from nequix.calculator import NequixCalculator

    from torch_sim.models.nequix import NequixModel
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        f"nequix not installed: {traceback.format_exc()}",  # ty:ignore[too-many-positional-arguments]
        allow_module_level=True,
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
    force_atol=5e-5,
    dtype=DTYPE,
    device=DEVICE,
)

test_nequix_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="nequix_model",
    dtype=DTYPE,
    device=DEVICE,
)
