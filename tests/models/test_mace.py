import pytest
import torch
from ase.atoms import Atoms

from tests.conftest import (
    consistency_test_simstate_fixtures,
    make_model_calculator_consistency_test,
    make_unbatched_model_calculator_consistency_test,
)
from torch_sim.io import atoms_to_state
from torch_sim.models.interface import validate_model_outputs


try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import mace_mp, mace_off

    from torch_sim.models.mace import MaceModel
    from torch_sim.unbatched.models.mace import UnbatchedMaceModel
except ImportError:
    pytest.skip("MACE not installed", allow_module_level=True)


mace_model = mace_mp(model="small", return_raw_model=True)
mace_off_model = mace_off(model="small", return_raw_model=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def ase_mace_calculator() -> MACECalculator:
    return mace_mp(
        model="small",
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_unbatched_mace_model(
    device: torch.device, dtype: torch.dtype
) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
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


test_unbatched_mace_consistency = make_unbatched_model_calculator_consistency_test(
    test_name="mace",
    model_fixture_name="torchsim_unbatched_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=[
        "cu_sim_state",
        "mg_sim_state",
        "sb_sim_state",
        "tio2_sim_state",
        "ga_sim_state",
        "niti_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        "benzene_sim_state",
    ],
)

test_mace_consistency = make_model_calculator_consistency_test(
    test_name="mace",
    model_fixture_name="torchsim_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=consistency_test_simstate_fixtures,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unbatched_mace_dtype_working(
    si_atoms: Atoms, dtype: torch.dtype, device: torch.device
) -> None:
    model = UnbatchedMaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )

    state = atoms_to_state(si_atoms, device, dtype)

    model.forward(state)


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

    state = atoms_to_state([si_atoms], device, dtype)

    model.forward(state)


def test_validate_model_outputs(
    torchsim_mace_model: MaceModel, device: torch.device, dtype: torch.dtype
) -> None:
    validate_model_outputs(torchsim_mace_model, device, dtype)


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
        model="small",
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_unbatched_mace_off_model(
    device: torch.device, dtype: torch.dtype
) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )


@pytest.fixture
def torchsim_mace_off_model(device: torch.device, dtype: torch.dtype) -> MaceModel:
    return MaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )


test_unbatched_mace_off_consistency = make_unbatched_model_calculator_consistency_test(
    test_name="mace_off",
    model_fixture_name="torchsim_unbatched_mace_off_model",
    calculator_fixture_name="ase_mace_off_calculator",
    sim_state_names=[
        "benzene_sim_state",
    ],
)

test_mace_off_consistency = make_model_calculator_consistency_test(
    test_name="mace_off",
    model_fixture_name="torchsim_mace_off_model",
    calculator_fixture_name="ase_mace_off_calculator",
    sim_state_names=[
        "benzene_sim_state",
    ],
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

    state = atoms_to_state([benzene_atoms], device, dtype)

    model.forward(state)
