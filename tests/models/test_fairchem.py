import traceback

import pytest

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test


try:
    from huggingface_hub.utils._auth import get_token

    from torch_sim.models.fairchem import FairChemModel

except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    pytest.skip(
        f"FairChem not installed: {traceback.format_exc()}",
        allow_module_level=True,
    )


@pytest.fixture
def eqv2_uma_model_pbc() -> FairChemModel:
    """UMA model for periodic boundary condition systems."""
    return FairChemModel(model="uma-s-1p1", task_name="omat", device=DEVICE)


test_fairchem_uma_model_outputs = pytest.mark.skipif(
    get_token() is None,
    reason="Requires HuggingFace authentication for UMA model access",
)(
    make_validate_model_outputs_test(
        model_fixture_name="eqv2_uma_model_pbc", device=DEVICE, dtype=DTYPE
    )
)
