import pytest
import torch
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp

from torch_sim.io import atoms_to_state
from torch_sim.models.interface import validate_model_outputs
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import wrapping_nl
from torch_sim.unbatched.models.mace import UnbatchedMaceModel


mace_model = mace_mp(model="small", return_raw_model=True)


@pytest.fixture
def si_system(si_atoms: Atoms, device: torch.device) -> dict:
    atomic_numbers = si_atoms.get_atomic_numbers()

    positions = torch.tensor(si_atoms.positions, device=device, dtype=torch.float32)
    cell = torch.tensor(si_atoms.cell.array, device=device, dtype=torch.float32)

    return {
        "positions": positions,
        "cell": cell,
        "atomic_numbers": atomic_numbers,
        "ase_atoms": si_atoms,
    }


@pytest.fixture
def torchsim_mace_model(device: torch.device) -> UnbatchedMaceModel:
    return UnbatchedMaceModel(
        model=mace_model,
        device=device,
        neighbor_list_fn=wrapping_nl,
        periodic=True,
        dtype=torch.float32,
        compute_force=True,
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
def torchsim_batched_mace_model(device: torch.device) -> MaceModel:
    return MaceModel(
        model=mace_model,
        device=device,
        periodic=True,
        dtype=torch.float32,
        compute_force=True,
    )


def test_mace_consistency(
    torchsim_mace_model: UnbatchedMaceModel,
    ase_mace_calculator: MACECalculator,
    si_system: dict,
    device: torch.device,
) -> None:
    # Set up ASE calculator
    si_system["ase_atoms"].calc = ase_mace_calculator

    # Get FairChem results
    torchsim_mace_results = torchsim_mace_model(
        atoms_to_state([si_system["ase_atoms"]], device, torch.float32)
    )

    # Get OCP results
    ase_mace_forces = torch.tensor(
        si_system["ase_atoms"].get_forces(), device=device, dtype=torch.float32
    )
    ase_mace_energy = torch.tensor(
        si_system["ase_atoms"].get_potential_energy(),
        device=device,
        dtype=torch.float32,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        torchsim_mace_results["energy"][0], ase_mace_energy, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        torchsim_mace_results["forces"], ase_mace_forces, rtol=1e-3, atol=1e-3
    )


def test_mace_batched_consistency(
    torchsim_batched_mace_model: MaceModel,
    ase_mace_calculator: MACECalculator,
    si_system: dict,
    si_atoms: Atoms,
    device: torch.device,
) -> None:
    # Set up ASE calculator
    si_atoms.calc = ase_mace_calculator

    si_sim_state = atoms_to_state([si_atoms], device, torch.float32)

    # Get FairChem results
    torchsim_mace_results = torchsim_batched_mace_model(si_sim_state)

    # Get OCP results
    ase_mace_forces = torch.tensor(
        si_system["ase_atoms"].get_forces(), device=device, dtype=torch.float32
    )
    ase_mace_energy = torch.tensor(
        si_system["ase_atoms"].get_potential_energy(),
        device=device,
        dtype=torch.float32,
    )

    # Test consistency with reasonable tolerances
    torch.testing.assert_close(
        torchsim_mace_results["energy"][0], ase_mace_energy, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        torchsim_mace_results["forces"], ase_mace_forces, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_dtype_working(
    si_atoms: Atoms, dtype: torch.dtype, device: torch.device
) -> None:
    model = MaceModel(
        model=mace_model,
        device=device,
        periodic=True,
        dtype=dtype,
        compute_force=True,
    )

    state = atoms_to_state([si_atoms], device, dtype)

    model.forward(state)


def test_validate_model_outputs(
    torchsim_batched_mace_model: MaceModel, device: torch.device
) -> None:
    validate_model_outputs(torchsim_batched_mace_model, device, torch.float32)
