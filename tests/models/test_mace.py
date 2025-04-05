import pytest
import torch
from ase.atoms import Atoms

from tests.conftest import make_model_calculator_consistency_test


try:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import mace_mp, mace_off
except ImportError:
    pytest.skip("MACE not installed", allow_module_level=True)

from torch_sim.io import atoms_to_state
from torch_sim.models.interface import validate_model_outputs
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.unbatched.models.mace import UnbatchedMaceModel


mace_model = mace_mp(model="small", return_raw_model=True)
mace_off_model = mace_off(model="small", return_raw_model=True)


@pytest.fixture
def dtype() -> torch.dtype:
    """Fixture to provide the default dtype for testing."""
    return torch.float32


@pytest.fixture
def si_system(si_atoms: Atoms, device: torch.device, dtype: torch.dtype) -> dict:
    atomic_numbers = si_atoms.get_atomic_numbers()

    positions = torch.tensor(si_atoms.positions, device=device, dtype=dtype)
    cell = torch.tensor(si_atoms.cell.array, device=device, dtype=dtype)

    return {
        "positions": positions,
        "cell": cell,
        "atomic_numbers": atomic_numbers,
        "ase_atoms": si_atoms,
    }


@pytest.fixture
def ti_system(ti_atoms: Atoms, device: torch.device, dtype: torch.dtype) -> dict:
    atomic_numbers = ti_atoms.get_atomic_numbers()

    positions = torch.tensor(ti_atoms.positions, device=device, dtype=dtype)
    cell = torch.tensor(ti_atoms.cell.array, device=device, dtype=dtype)

    return {
        "positions": positions,
        "cell": cell,
        "atomic_numbers": atomic_numbers,
        "ase_atoms": ti_atoms,
    }


@pytest.fixture
def torchsim_mace_model(device: torch.device, dtype: torch.dtype) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        neighbor_list_fn=vesin_nl_ts,
    )


@pytest.fixture
def ase_mace_calculator() -> MACECalculator:
    return mace_mp(
        model="small",
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_batched_mace_model(device: torch.device, dtype: torch.dtype) -> MaceModel:
    return MaceModel(
        model=mace_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        neighbor_list_fn=vesin_nl_ts,
    )


def test_mace_consistency_si(
    torchsim_mace_model: UnbatchedMaceModel,
    ase_mace_calculator: MACECalculator,
    si_system: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # Set up ASE calculator
    si_system["ase_atoms"].calc = ase_mace_calculator

    # Get FairChem results
    torchsim_mace_results = torchsim_mace_model(
        atoms_to_state([si_system["ase_atoms"]], device, dtype)
    )

    # Get OCP results
    ase_mace_forces = torch.tensor(
        si_system["ase_atoms"].get_forces(), device=device, dtype=dtype
    )
    ase_mace_energy = torch.tensor(
        si_system["ase_atoms"].get_potential_energy(),
        device=device,
        dtype=dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        torchsim_mace_results["energy"][0], ase_mace_energy, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        torchsim_mace_results["forces"], ase_mace_forces, rtol=1e-3, atol=1e-3
    )


def test_mace_consistency_ti(
    torchsim_mace_model: UnbatchedMaceModel,
    ase_mace_calculator: MACECalculator,
    ti_system: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # Set up ASE calculator
    ti_system["ase_atoms"].calc = ase_mace_calculator

    # Get FairChem results
    torchsim_mace_results = torchsim_mace_model(
        atoms_to_state([ti_system["ase_atoms"]], device, dtype)
    )

    # Get OCP results
    ase_mace_forces = torch.tensor(
        ti_system["ase_atoms"].get_forces(), device=device, dtype=dtype
    )
    ase_mace_energy = torch.tensor(
        ti_system["ase_atoms"].get_potential_energy(),
        device=device,
        dtype=dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        torchsim_mace_results["energy"][0], ase_mace_energy, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        torchsim_mace_results["forces"], ase_mace_forces, rtol=1e-3, atol=1e-3
    )


test_mace_consistency = make_model_calculator_consistency_test(
    test_name="mace",
    model_fixture_name="torchsim_mace_model",
    calculator_fixture_name="ase_mace_calculator",
    sim_state_names=[
        "cu_sim_state",
        "ti_sim_state",
        "si_sim_state",
        "sio2_sim_state",
        "benzene_sim_state",
    ],
    rtol=1e-5,
    atol=1e-5,
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

    state = atoms_to_state([si_atoms], device, dtype)

    model.forward(state)


def test_validate_model_outputs(
    torchsim_batched_mace_model: MaceModel, device: torch.device, dtype: torch.dtype
) -> None:
    validate_model_outputs(torchsim_batched_mace_model, device, dtype)


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
def torchsim_mace_off_model(
    device: torch.device, dtype: torch.dtype
) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        neighbor_list_fn=vesin_nl_ts,
    )


@pytest.fixture
def ase_mace_off_calculator() -> MACECalculator:
    return mace_off(
        model="small",
        device="cpu",
        default_dtype="float32",
        dispersion=False,
    )


@pytest.fixture
def torchsim_batched_mace_off_model(
    device: torch.device, dtype: torch.dtype
) -> MaceModel:
    return MaceModel(
        model=mace_off_model,
        device=device,
        dtype=dtype,
        compute_forces=True,
        neighbor_list_fn=vesin_nl_ts,
    )


def test_mace_off_consistency(
    torchsim_mace_off_model: UnbatchedMaceModel,
    ase_mace_off_calculator: MACECalculator,
    benzene_system: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # Set up ASE calculator
    benzene_system["ase_atoms"].calc = ase_mace_off_calculator

    # Get FairChem results
    torchsim_mace_results = torchsim_mace_off_model(
        atoms_to_state([benzene_system["ase_atoms"]], device, dtype)
    )

    # Get OCP results
    ase_mace_forces = torch.tensor(
        benzene_system["ase_atoms"].get_forces(), device=device, dtype=dtype
    )
    ase_mace_energy = torch.tensor(
        benzene_system["ase_atoms"].get_potential_energy(),
        device=device,
        dtype=dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        torchsim_mace_results["energy"][0], ase_mace_energy, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        torchsim_mace_results["forces"], ase_mace_forces, rtol=1e-3, atol=1e-3
    )


def test_mace_off_batched_consistency(
    torchsim_batched_mace_off_model: MaceModel,
    ase_mace_off_calculator: MACECalculator,
    benzene_system: dict,
    benzene_atoms: Atoms,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # Set up ASE calculator
    benzene_atoms.calc = ase_mace_off_calculator

    benzene_sim_state = atoms_to_state([benzene_atoms], device, dtype)

    # Get FairChem results
    torchsim_mace_off_results = torchsim_batched_mace_off_model(benzene_sim_state)

    # Get OCP results
    ase_mace_off_forces = torch.tensor(
        benzene_system["ase_atoms"].get_forces(), device=device, dtype=dtype
    )
    ase_mace_off_energy = torch.tensor(
        benzene_system["ase_atoms"].get_potential_energy(),
        device=device,
        dtype=dtype,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        torchsim_mace_off_results["energy"][0], ase_mace_off_energy, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        torchsim_mace_off_results["forces"], ase_mace_off_forces, rtol=1e-3, atol=1e-3
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
