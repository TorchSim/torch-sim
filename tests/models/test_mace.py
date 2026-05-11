import random
import time
import traceback
import urllib.error
from collections.abc import Callable
from typing import Any

import pytest
import torch
from ase.atoms import Atoms

import torch_sim as ts
from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS, SIMSTATE_MOLECULE_GENERATORS


try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import mace_mp, mace_off, mace_omol

    from torch_sim.models.mace import MaceModel
except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    pytest.skip(f"MACE not installed: {traceback.format_exc()}", allow_module_level=True)

DTYPE = torch.float64
MAX_RETRIES = 3
RETRY_DELAY = 45 + random.randint(0, 15)


def _download_with_retry(fn: Callable, **kwargs: Any) -> Any:
    """Retry until the function returns a value or the maximum number of retries
    is reached.

    Args:
        fn: The function to retry.
        **kwargs: The arguments to pass to the function.
    Returns:
        The value returned by the function.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn(**kwargs)
        except (RuntimeError, urllib.error.HTTPError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))
    return None


@pytest.fixture(scope="session")
def raw_mace_mp():
    return _download_with_retry(mace_mp, model="small", return_raw_model=True)


@pytest.fixture(scope="session")
def raw_mace_off():
    return _download_with_retry(mace_off, model="small", return_raw_model=True)


@pytest.fixture(scope="session")
def raw_mace_omol():
    return _download_with_retry(mace_omol, model="extra_large", return_raw_model=True)


@pytest.fixture
def ase_mace_calculator(raw_mace_mp: torch.nn.Module) -> MACECalculator:
    dtype = str(DTYPE).removeprefix("torch.")
    return MACECalculator(
        models=raw_mace_mp,
        device=DEVICE.type,
        default_dtype=dtype,
        dispersion=False,
    )


@pytest.fixture
def ts_mace_model(raw_mace_mp: torch.nn.Module) -> MaceModel:
    return MaceModel(
        model=raw_mace_mp,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


test_mace_consistency = make_model_calculator_consistency_test(
    test_name="mace",
    model_fixture_name="ts_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=tuple(SIMSTATE_BULK_GENERATORS.keys()),
    dtype=DTYPE,
)

test_mace_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_mace_model", device=DEVICE, dtype=DTYPE
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_dtype_working(
    si_atoms: Atoms, raw_mace_mp: torch.nn.Module, dtype: torch.dtype
) -> None:
    model = MaceModel(
        model=raw_mace_mp,
        device=DEVICE,
        dtype=dtype,
        compute_forces=True,
    )
    state = ts.io.atoms_to_state([si_atoms], DEVICE, dtype)
    model.forward(state)


@pytest.fixture
def ase_mace_off_calculator(raw_mace_off: torch.nn.Module) -> MACECalculator:
    return MACECalculator(
        models=raw_mace_off,
        device=str(DEVICE),
        default_dtype=str(DTYPE).removeprefix("torch."),
        dispersion=False,
    )


@pytest.fixture
def ts_mace_off_model(raw_mace_off: torch.nn.Module) -> MaceModel:
    return MaceModel(
        model=raw_mace_off,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
    )


test_mace_off_consistency = make_model_calculator_consistency_test(
    test_name="mace_off",
    model_fixture_name="ts_mace_off_model",
    calculator_fixture_name="ase_mace_off_calculator",
    sim_state_names=tuple(SIMSTATE_MOLECULE_GENERATORS.keys()),
    dtype=DTYPE,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_off_dtype_working(
    benzene_sim_state: ts.SimState,
    raw_mace_off: torch.nn.Module,
    dtype: torch.dtype,
) -> None:
    model = MaceModel(
        model=raw_mace_off,
        device=DEVICE,
        dtype=dtype,
        compute_forces=True,
    )
    model.forward(benzene_sim_state.to(DEVICE, dtype))


@pytest.mark.parametrize(
    ("charge", "spin"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 0.0),
        (0.0, 2.0),
    ],
)
def test_mace_charge_spin(
    benzene_sim_state: ts.SimState,
    raw_mace_omol: torch.nn.Module,
    charge: float,
    spin: float,
) -> None:
    """Test that MaceModel correctly handles charge and spin."""
    benzene_sim_state = ts.SimState.from_state(
        benzene_sim_state,
        charge=torch.tensor([charge], device=DEVICE, dtype=DTYPE),
        spin=torch.tensor([spin], device=DEVICE, dtype=DTYPE),
    )
    if charge != 0.0:
        assert benzene_sim_state.charge is not None
        assert benzene_sim_state.charge[0].item() == charge
    else:
        assert (
            benzene_sim_state.charge is None or benzene_sim_state.charge[0].item() == 0.0
        )
    if spin != 0.0:
        assert benzene_sim_state.spin is not None
        assert benzene_sim_state.spin[0].item() == spin
    else:
        assert benzene_sim_state.spin is None or benzene_sim_state.spin[0].item() == 0.0
    model = MaceModel(
        model=raw_mace_omol,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
    )
    result = model.forward(benzene_sim_state)
    assert "energy" in result
    assert result["energy"].shape == (1,)
    if model.compute_forces:
        assert "forces" in result
        assert result["forces"].shape == benzene_sim_state.positions.shape
