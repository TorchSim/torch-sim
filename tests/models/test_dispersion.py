"""Tests for the D3DispersionModel wrapper."""

import traceback  # noqa: I001

import pytest
import torch

from tests.conftest import DEVICE, DTYPE
from tests.models.conftest import make_validate_model_outputs_test

try:
    from nvalchemiops.torch.interactions.dispersion import D3Parameters

    from torch_sim.models.dispersion import D3DispersionModel
except (ImportError, OSError, RuntimeError):
    pytest.skip(
        f"nvalchemiops not installed: {traceback.format_exc()}",
        allow_module_level=True,
    )


def _make_d3_params(device: torch.device = DEVICE) -> D3Parameters:
    """Build minimal D3 reference parameters for testing (elements up to Fe=26)."""
    max_z = 26
    mesh = 5
    return D3Parameters(
        rcov=torch.rand(max_z + 1, device=device),
        r4r2=torch.rand(max_z + 1, device=device),
        c6ab=torch.rand(max_z + 1, max_z + 1, mesh, mesh, device=device),
        cn_ref=torch.rand(max_z + 1, max_z + 1, mesh, mesh, device=device),
    )


@pytest.fixture
def d3_model() -> D3DispersionModel:
    return D3DispersionModel(
        a1=0.4289,
        a2=4.4407,
        s8=0.7875,
        d3_params=_make_d3_params(),
        cutoff=12.0,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


test_d3_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="d3_model", device=DEVICE, dtype=DTYPE
)
