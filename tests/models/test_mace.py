import traceback

import pytest
import torch
from ase.atoms import Atoms

import torch_sim as ts
from tests.conftest import DEVICE
from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.models.mace import MaceUrls


try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import mace_mp, mace_off, mace_omol

    from torch_sim.models.mace import MaceModel
except (ImportError, ValueError):
    pytest.skip(f"MACE not installed: {traceback.format_exc()}", allow_module_level=True)


raw_mace_mp = mace_mp(model=MaceUrls.mace_mp_small, return_raw_model=True)
raw_mace_off = mace_off(model=MaceUrls.mace_off_small, return_raw_model=True)
raw_mace_omol = mace_omol(model="extra_large", return_raw_model=True)
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
    sim_state_names=tuple(
        s for s in consistency_test_simstate_fixtures if s != "ti_sim_state"
    ),
    dtype=DTYPE,
)


test_mace_consistency_ti = make_model_calculator_consistency_test(
    test_name="mace_ti",
    model_fixture_name="ts_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=("ti_sim_state",),
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
def benzene_system(benzene_atoms: Atoms) -> dict:
    atomic_numbers = benzene_atoms.get_atomic_numbers()

    positions = torch.tensor(benzene_atoms.positions, device=DEVICE, dtype=DTYPE)
    cell = torch.tensor(benzene_atoms.cell.array, device=DEVICE, dtype=DTYPE)

    return {
        "positions": positions,
        "cell": cell,
        "atomic_numbers": atomic_numbers,
        "ase_atoms": benzene_atoms,
    }


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
    sim_state_names=("benzene_sim_state",),
    dtype=DTYPE,
)

test_mace_off_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="ts_mace_model", dtype=DTYPE
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_off_dtype_working(benzene_atoms: Atoms, dtype: torch.dtype) -> None:
    model = MaceModel(model=raw_mace_off, device=DEVICE, dtype=dtype, compute_forces=True)

    state = ts.io.atoms_to_state([benzene_atoms], DEVICE, dtype)

    model.forward(state)


def test_mace_urls_enum() -> None:
    assert len(MaceUrls) > 2
    for key in MaceUrls:
        assert key.value.startswith("https://github.com/ACEsuit/mace-")
        assert key.value.endswith((".model", ".model?raw=true"))


@pytest.mark.parametrize(
    ("charge", "spin"),
    [
        (0.0, 0.0),  # Neutral, no spin
        (1.0, 1.0),  # +1 charge, spin=1 (doublet)
        (-1.0, 0.0),  # -1 charge, no spin (singlet)
        (0.0, 2.0),  # Neutral, spin=2 (triplet)
    ],
)
def test_mace_charge_spin(benzene_atoms: Atoms, charge: float, spin: float) -> None:
    """Test that MaceModel correctly handles charge and spin from atoms.info."""
    # Set charge and spin in ASE atoms.info
    benzene_atoms.info["charge"] = charge
    benzene_atoms.info["spin"] = spin

    # Convert to SimState (should extract charge/spin)
    state = ts.io.atoms_to_state([benzene_atoms], DEVICE, DTYPE)

    # Verify charge/spin were extracted correctly
    if charge != 0.0:
        assert state.charge is not None
        assert state.charge[0].item() == charge
    else:
        assert state.charge is None or state.charge[0].item() == 0.0

    if spin != 0.0:
        assert state.spin is not None
        assert state.spin[0].item() == spin
    else:
        assert state.spin is None or state.spin[0].item() == 0.0

    # Create model with MACE-OMOL (supports charge/spin for molecules)
    model = MaceModel(
        model=raw_mace_omol,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
    )

    # This should not raise an error
    result = model.forward(state)

    # Verify outputs exist
    assert "energy" in result
    assert result["energy"].shape == (1,)
    if model.compute_forces:
        assert "forces" in result
        assert result["forces"].shape == benzene_atoms.positions.shape
