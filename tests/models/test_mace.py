import pytest
import torch
from ase.atoms import Atoms
from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_mp

from torchsim.models.interface import validate_model_outputs
from torchsim.models.mace import MaceModel, UnbatchedMaceModel
from torchsim.neighbors import wrapping_nl
from torchsim.runners import atoms_to_state
from torchsim.state import BaseState


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
        si_system["positions"], si_system["cell"], si_system["atomic_numbers"]
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

    si_base_state = atoms_to_state([si_atoms], device, torch.float32)

    # Get FairChem results
    torchsim_mace_results = torchsim_batched_mace_model(
        positions=si_base_state.positions,
        cell=si_base_state.cell,
        batch=si_base_state.batch,
        atomic_numbers=si_base_state.atomic_numbers,
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

    model.forward(
        positions=state.positions,
        cell=state.cell,
        batch=state.batch,
        atomic_numbers=state.atomic_numbers,
    )


def test_validate_model_outputs(
    torchsim_batched_mace_model: MaceModel, device: torch.device
) -> None:
    validate_model_outputs(torchsim_batched_mace_model, device, torch.float32)


def test_integrate_with_autobatcher(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    torchsim_batched_mace_model: MaceModel,
) -> None:
    """Test integration with autobatcher.

    This test is honestly quite out of place but this functionality can only
    be tested with an MLIP that actually consumes memory. It's failure is
    indicative of something going wrong with the autobatcher.
    """
    mace_model = torchsim_batched_mace_model
    from torchsim.integrators import nve
    from torchsim.runners import initialize_state, integrate
    from torchsim.state import split_state

    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        mace_model.device,
        mace_model.dtype,
    )

    final_state = integrate(
        system=triple_state,
        model=mace_model,
        integrator=nve,
        n_steps=10,
        temperature=300.0,
        timestep=0.001,
        autobatcher=True,
    )

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)

    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)


def test_optimize_with_autobatcher(
    ar_base_state: BaseState,
    fe_fcc_state: BaseState,
    torchsim_batched_mace_model: MaceModel,
) -> None:
    """Test optimize with autobatcher."""
    from torchsim.optimizers import unit_cell_fire
    from torchsim.runners import generate_force_convergence_fn, initialize_state, optimize
    from torchsim.state import split_state

    mace_model = torchsim_batched_mace_model

    states = [ar_base_state, fe_fcc_state, ar_base_state]
    triple_state = initialize_state(
        states,
        mace_model.device,
        mace_model.dtype,
    )

    final_state = optimize(
        system=triple_state,
        model=mace_model,
        optimizer=unit_cell_fire,
        convergence_fn=generate_force_convergence_fn(force_tol=1e-1),
        autobatcher=True,
    )

    assert isinstance(final_state, BaseState)
    split_final_state = split_state(final_state)
    for init_state, final_state in zip(states, split_final_state, strict=False):
        assert torch.all(final_state.atomic_numbers == init_state.atomic_numbers)
        assert torch.any(final_state.positions != init_state.positions)
