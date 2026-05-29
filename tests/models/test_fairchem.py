from __future__ import annotations

import traceback

import pytest

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test


try:
    from huggingface_hub.utils._auth import get_token

    from torch_sim.models.fairchem import FairChemModel

    _IMPORT_ERROR: str | None = None
except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    _IMPORT_ERROR = traceback.format_exc()

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None, reason=f"FairChem not installed: {_IMPORT_ERROR}"
)


@pytest.fixture
def eqv2_uma_model_pbc() -> FairChemModel:
    """UMA model for periodic boundary condition systems."""
    return FairChemModel(model="uma-s-1p1", task_name="omat", device=DEVICE)


test_fairchem_uma_model_outputs = pytest.mark.skipif(
    _IMPORT_ERROR is not None or get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)(
    make_validate_model_outputs_test(
        model_fixture_name="eqv2_uma_model_pbc", device=DEVICE, dtype=DTYPE
    )
)
