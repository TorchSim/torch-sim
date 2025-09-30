import pytest
import torch
from ase.atoms import Atoms

import torch_sim as ts
from tests.models.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_validate_model_outputs_test,
)
from torch_sim.models.mace import MaceUrls


try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import mace_mp, mace_off

    from torch_sim.models.mace import MaceModel
except (ImportError, ValueError):
    pytest.skip("MACE not installed", allow_module_level=True)


mace_model = mace_mp(model=MaceUrls.mace_mp_small, return_raw_model=True)
mace_off_model = mace_off(model=MaceUrls.mace_off_small, return_raw_model=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def ase_mace_calculator() -> MACECalculator:
    return mace_mp(
        model=MaceUrls.mace_mp_small,
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_mace_model(device: torch.device, dtype: torch.dtype) -> MaceModel:
    return MaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )


test_mace_consistency = make_model_calculator_consistency_test(
    test_name="mace",
    model_fixture_name="torchsim_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_dtype_working(
    si_atoms: Atoms, dtype: torch.dtype, device: torch.device
) -> None:
    model = MaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )

    state = ts.io.atoms_to_state([si_atoms], device, dtype)

    model.forward(state)


@pytest.fixture
def benzene_system(
    benzene_atoms: Atoms, device: torch.device, dtype: torch.dtype
) -> dict:
    atomic_numbers = benzene_atoms.get_atomic_numbers()

    positions = torch.tensor(benzene_atoms.positions, device=device, dtype=dtype)
    cell = torch.tensor(benzene_atoms.cell.array, device=device, dtype=dtype)

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
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_mace_off_model(device: torch.device, dtype: torch.dtype) -> MaceModel:
    return MaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )


test_mace_off_consistency = make_model_calculator_consistency_test(
    test_name="mace_off",
    model_fixture_name="torchsim_mace_off_model",
    calculator_fixture_name="ase_mace_off_calculator",
    sim_state_names=["benzene_sim_state"],
)

test_mace_off_model_outputs = make_validate_model_outputs_test(
    model_fixture_name="torchsim_mace_model"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_off_dtype_working(
    benzene_atoms: Atoms, dtype: torch.dtype, device: torch.device
) -> None:
    model = MaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )

    state = ts.io.atoms_to_state([benzene_atoms], device, dtype)

    model.forward(state)


def test_mace_urls_enum() -> None:
    assert len(MaceUrls) > 2
    for key in MaceUrls:
        assert key.value.startswith("https://github.com/ACEsuit/mace-")
        assert key.value.endswith((".model", ".model?raw=true"))
