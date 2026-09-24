"""Tests for the bundled DeePMD-kit integration (torch_sim.models.deepmd).

Uses the DPA-3.1-3M universal foundation checkpoint (full periodic-table
type_map, "Omat24" head).
"""

from __future__ import annotations

import time
import traceback
import urllib.error
import urllib.request

import pytest
import torch

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_BULK_GENERATORS, SIMSTATE_MOLECULE_GENERATORS


try:
    from deepmd.calculator import DP

    from torch_sim.models.deepmd import DeepmdModel

    _IMPORT_ERROR: str | None = None
except ImportError:
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"deepmd not installed: {_IMPORT_ERROR}"
)

DTYPE = torch.float64
MAX_RETRIES = 3
RETRY_DELAY = 30

_MODEL_URL = (
    "https://store.aissquare.com/models/35b4ce45-4f59-4868-9fd7-a0c0f5ad9464/"
    "DPA-3.1-3M.pt"
)
_MODEL_HEAD = "Omat24"


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Download the DPA-3.1-3M checkpoint once per session, with retries."""
    dest = tmp_path_factory.mktemp("deepmd") / "DPA-3.1-3M.pt"
    for attempt in range(MAX_RETRIES):
        try:
            urllib.request.urlretrieve(_MODEL_URL, dest)  # noqa: S310
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES - 1:
                pytest.skip(f"could not download DPA-3.1-3M from {_MODEL_URL}: {exc}")
            time.sleep(RETRY_DELAY * (attempt + 1))
        else:
            break
    return str(dest)


@pytest.fixture
def deepmd_model(model_path: str) -> DeepmdModel:
    return DeepmdModel(
        model_path=model_path,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
        head=_MODEL_HEAD,
    )


@pytest.fixture
def deepmd_calculator(model_path: str) -> DP:
    return DP(model=model_path, head=_MODEL_HEAD)


def test_deepmd_initialization(deepmd_model: DeepmdModel) -> None:
    assert deepmd_model.device == DEVICE
    assert deepmd_model.dtype == DTYPE
    assert deepmd_model.compute_forces is True
    assert deepmd_model.compute_stress is True
    assert "Cu" in deepmd_model.type_map  # universal periodic-table type_map


test_deepmd_consistency = make_model_calculator_consistency_test(
    test_name="deepmd",
    model_fixture_name="deepmd_model",
    calculator_fixture_name="deepmd_calculator",
    sim_state_names=tuple(SIMSTATE_BULK_GENERATORS.keys()),
    device=DEVICE,
    dtype=DTYPE,
)


@pytest.fixture
def deepmd_molecule_model(model_path: str) -> DeepmdModel:
    """Stress disabled (mirroring the mace_off molecule test): ASE's DP
    calculator raises PropertyNotImplementedError for stress on non-periodic
    systems, so molecules are checked on energy/forces only."""
    return DeepmdModel(
        model_path=model_path,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
        head=_MODEL_HEAD,
    )


test_deepmd_molecule_consistency = make_model_calculator_consistency_test(
    test_name="deepmd_molecule",
    model_fixture_name="deepmd_molecule_model",
    calculator_fixture_name="deepmd_calculator",
    sim_state_names=tuple(SIMSTATE_MOLECULE_GENERATORS.keys()),
    device=DEVICE,
    dtype=DTYPE,
)

test_deepmd_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="deepmd_model",
    device=DEVICE,
    dtype=DTYPE,
)
