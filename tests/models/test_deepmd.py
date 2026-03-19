import os
import traceback

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE


try:
    from deepmd.calculator import DP

    from torch_sim.models.deepmd import DeepMDModel

except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    pytest.skip(
        f"DeepMD not installed: {traceback.format_exc()}", allow_module_level=True
    )

DTYPE = torch.float64

MODEL_PATH = os.environ.get("DEEPMD_MODEL_PATH", "")
if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    pytest.skip(
        "DEEPMD_MODEL_PATH not set or file not found. "
        "Set DEEPMD_MODEL_PATH to a DeepMD .pt/.pth model file to run these tests.",
        allow_module_level=True,
    )


def _make_water_state(
    device: torch.device, dtype: torch.dtype
) -> ts.SimState:
    """Create a periodic water molecule state."""
    from ase.build import molecule

    water = molecule("H2O")
    water.set_cell([10, 10, 10])
    water.set_pbc(True)
    return ts.io.atoms_to_state([water], device, dtype)


def _make_two_water_state(
    device: torch.device, dtype: torch.dtype
) -> ts.SimState:
    """Create a batched state with two water molecules."""
    from ase.build import molecule

    water1 = molecule("H2O")
    water1.set_cell([10, 10, 10])
    water1.set_pbc(True)

    water2 = molecule("H2O")
    water2.set_cell([12, 12, 12])
    water2.set_pbc(True)
    water2.positions += 0.3

    return ts.io.atoms_to_state([water1, water2], device, dtype)


@pytest.fixture
def ase_deepmd_calculator() -> DP:
    return DP(model=MODEL_PATH)


@pytest.fixture
def ts_deepmd_model() -> DeepMDModel:
    return DeepMDModel(
        model=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
    )


def test_deepmd_single_system(ts_deepmd_model: DeepMDModel) -> None:
    """Test forward pass on a single water molecule."""
    state = _make_water_state(DEVICE, DTYPE)
    result = ts_deepmd_model.forward(state)

    assert "energy" in result
    assert result["energy"].shape == (1,)
    assert result["energy"].dtype == DTYPE

    assert "forces" in result
    assert result["forces"].shape == (3, 3)
    assert result["forces"].dtype == DTYPE

    assert "stress" in result
    assert result["stress"].shape == (1, 3, 3)
    assert result["stress"].dtype == DTYPE


def test_deepmd_batched_systems(ts_deepmd_model: DeepMDModel) -> None:
    """Test forward pass on a batch of two water molecules."""
    state = _make_two_water_state(DEVICE, DTYPE)
    result = ts_deepmd_model.forward(state)

    assert result["energy"].shape == (2,)
    assert result["forces"].shape == (6, 3)
    assert result["stress"].shape == (2, 3, 3)


def test_deepmd_batch_single_consistency(ts_deepmd_model: DeepMDModel) -> None:
    """Batched results should match individual forward passes."""
    batched_state = _make_two_water_state(DEVICE, DTYPE)
    batched_result = ts_deepmd_model.forward(batched_state)

    single_state = _make_water_state(DEVICE, DTYPE)
    single_result = ts_deepmd_model.forward(single_state)

    torch.testing.assert_close(
        batched_result["energy"][0],
        single_result["energy"][0],
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        batched_result["forces"][:3],
        single_result["forces"],
        atol=1e-10,
        rtol=1e-10,
    )


def test_deepmd_calculator_consistency(
    ts_deepmd_model: DeepMDModel, ase_deepmd_calculator: DP
) -> None:
    """TorchSim model should match the ASE DP calculator."""
    from torch_sim.testing import assert_model_calculator_consistency

    state = _make_water_state(DEVICE, DTYPE)
    assert_model_calculator_consistency(
        model=ts_deepmd_model,
        calculator=ase_deepmd_calculator,
        sim_state=state,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deepmd_dtype_working(dtype: torch.dtype) -> None:
    model = DeepMDModel(
        model=MODEL_PATH,
        device=DEVICE,
        dtype=dtype,
        compute_forces=True,
    )
    state = _make_water_state(DEVICE, dtype)
    result = model.forward(state)
    assert result["energy"].dtype == dtype
    assert result["forces"].dtype == dtype


def test_deepmd_no_mutation(ts_deepmd_model: DeepMDModel) -> None:
    """Forward pass should not mutate the input state."""
    state = _make_water_state(DEVICE, DTYPE)
    original_positions = state.positions.clone()
    original_cell = state.cell.clone()

    ts_deepmd_model.forward(state)

    torch.testing.assert_close(state.positions, original_positions)
    torch.testing.assert_close(state.cell, original_cell)
