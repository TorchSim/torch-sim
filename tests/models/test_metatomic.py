from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

import pytest
import torch

from tests.conftest import DEVICE
from tests.models.conftest import (
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.testing import SIMSTATE_GENERATORS, ModelTolerance


if TYPE_CHECKING:
    from metatomic.torch import AtomisticModel

try:
    from metatomic_ase import MetatomicCalculator
    from upet import get_upet

    from torch_sim.models.metatomic import MetatomicModel

    _IMPORT_ERROR: str | None = None
except ImportError:
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"metatomic not installed: {_IMPORT_ERROR}"
)


@pytest.fixture
def metatomic_module() -> AtomisticModel:
    return get_upet(model="pet-mad")


@pytest.fixture
def metatomic_calculator(metatomic_module: AtomisticModel) -> MetatomicCalculator:
    return MetatomicCalculator(model=metatomic_module, device=DEVICE)


@pytest.fixture
def metatomic_model(metatomic_module: AtomisticModel) -> MetatomicModel:
    return MetatomicModel(model=metatomic_module, device=DEVICE)


def test_metatomic_initialization() -> None:
    model = MetatomicModel(model=get_upet(model="pet-mad"), device=DEVICE)
    assert model.device == DEVICE
    assert model.dtype == torch.float32


test_metatomic_consistency = make_model_calculator_consistency_test(
    test_name="metatomic",
    model_fixture_name="metatomic_model",
    calculator_fixture_name="metatomic_calculator",
    sim_state_names=tuple(SIMSTATE_GENERATORS.keys()),
    energy_atol=ModelTolerance.LOOSE,
    dtype=torch.float32,
    device=DEVICE,
)

test_metatomic_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="metatomic_model",
    dtype=torch.float32,
    device=DEVICE,
)
