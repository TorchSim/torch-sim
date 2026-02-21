import traceback

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
    from mace.calculators.foundations_models import mace_mp, mace_off

    from torch_sim.models.mace import MaceModel, MaceUrls

except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    pytest.skip(f"MACE not installed: {traceback.format_exc()}", allow_module_level=True)

# mace_omol is optional (added in newer MACE versions)
try:
    from mace.calculators.foundations_models import mace_omol

    raw_mace_omol = mace_omol(model="extra_large", return_raw_model=True)
    HAS_MACE_OMOL = True
except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
    raw_mace_omol = None
    HAS_MACE_OMOL = False

raw_mace_mp = mace_mp(model=MaceUrls.mace_mp_small, return_raw_model=True)
raw_mace_off = mace_off(model=MaceUrls.mace_off_small, return_raw_model=True)
DTYPE = torch.float64


@pytest.fixture
def ase_mace_calculator() -> MACECalculator:
    dtype = str(DTYPE).removeprefix("torch.")
    return mace_mp(
        model=MaceUrls.mace_mp_small, device="cpu", default_dtype=dtype, dispersion=False
    )


@pytest.fixture
def ts_mace_model() -> MaceModel:
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_dtype_working(si_atoms: Atoms, dtype: torch.dtype) -> None:
    model = MaceModel(
        model=raw_mace_mp,
        device=DEVICE,
        dtype=dtype,
        compute_forces=True,
    )

    state = ts.io.atoms_to_state([si_atoms], DEVICE, dtype)
    model.forward(state)


@pytest.fixture
def ase_mace_off_calculator() -> MACECalculator:
    return mace_off(
        model=MaceUrls.mace_off_small,
        device=str(DEVICE),
        default_dtype=str(DTYPE).removeprefix("torch."),
        dispersion=False,
    )


@pytest.fixture
def ts_mace_off_model() -> MaceModel:
    return MaceModel(model=raw_mace_off, device=DEVICE, dtype=DTYPE, compute_forces=True)


test_mace_off_consistency = make_model_calculator_consistency_test(
    test_name="mace_off",
    model_fixture_name="ts_mace_off_model",
    calculator_fixture_name="ase_mace_off_calculator",
    sim_state_names=tuple(SIMSTATE_MOLECULE_GENERATORS.keys()),
    dtype=DTYPE,
)

test_mace_off_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_mace_model", dtype=DTYPE
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_off_dtype_working(
    benzene_sim_state: ts.SimState, dtype: torch.dtype
) -> None:
    model = MaceModel(model=raw_mace_off, device=DEVICE, dtype=dtype, compute_forces=True)
    model.forward(benzene_sim_state.to(DEVICE, dtype))


def test_mace_urls_enum() -> None:
    assert len(MaceUrls) > 2
    for key in MaceUrls:
        assert key.value.startswith("https://github.com/ACEsuit/mace-")
        assert key.value.endswith((".model", ".model?raw=true"))


@pytest.mark.skipif(not HAS_MACE_OMOL, reason="mace_omol not available")
@pytest.mark.parametrize(
    ("charge", "spin"),
    [
        (0.0, 0.0),  # Neutral, no spin
        (1.0, 1.0),  # +1 charge, spin=1 (doublet)
        (-1.0, 0.0),  # -1 charge, no spin (singlet)
        (0.0, 2.0),  # Neutral, spin=2 (triplet)
    ],
)
def test_mace_charge_spin(
    benzene_sim_state: ts.SimState, charge: float, spin: float
) -> None:
    """Test that MaceModel correctly handles charge and spin from atoms.info."""
    # Convert to SimState (should extract charge/spin)
    benzene_sim_state.charge = torch.tensor([charge], device=DEVICE, dtype=DTYPE)
    benzene_sim_state.spin = torch.tensor([spin], device=DEVICE, dtype=DTYPE)

    # Verify charge/spin were extracted correctly
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

    # Create model with MACE-OMOL (supports charge/spin for molecules)
    model = MaceModel(
        model=raw_mace_omol,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
    )

    # This should not raise an error
    result = model.forward(benzene_sim_state)

    # Verify outputs exist
    assert "energy" in result
    assert result["energy"].shape == (1,)
    if model.compute_forces:
        assert "forces" in result
        assert result["forces"].shape == benzene_sim_state.positions.shape
