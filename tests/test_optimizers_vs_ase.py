import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.optimize import FIRE

import torch_sim as ts
from torch_sim.io import atoms_to_state, state_to_atoms
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import frechet_cell_fire, unit_cell_fire


if TYPE_CHECKING:
    from mace.calculators import MACECalculator


@pytest.fixture
def osn2_sim_state(torchsim_mace_mpa: MaceModel) -> ts.state.SimState:
    """Provides an initial SimState for rhombohedral OsN2."""
    # For pymatgen Structure initialization
    from pymatgen.core import Lattice, Structure

    a = 3.211996
    lattice = Lattice.from_parameters(a, a, a, 60, 60, 60)
    species = ["Os", "N"]
    frac_coords = [[0.75, 0.7501, -0.25], [0, 0, 0]]  # Slightly perturbed
    structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    return ts.initialize_state(
        structure, dtype=torchsim_mace_mpa.dtype, device=torchsim_mace_mpa.device
    )


@pytest.fixture
def distorted_fcc_al_conventional_sim_state(
    torchsim_mace_mpa: MaceModel,
) -> ts.state.SimState:
    """Initial SimState for a slightly distorted FCC Al conventional cell (4 atoms)."""
    # Create a standard 4-atom conventional FCC Al cell
    atoms_fcc = bulk("Al", crystalstructure="fcc", a=4.05, cubic=True)

    # Define a small triclinic strain matrix (deviations from identity)
    strain_matrix = np.array([[1.0, 0.05, -0.03], [0.04, 1.0, 0.06], [-0.02, 0.03, 1.0]])

    original_cell = atoms_fcc.get_cell()
    new_cell = original_cell @ strain_matrix.T  # Apply strain
    atoms_fcc.set_cell(new_cell, scale_atoms=True)

    # Slightly perturb atomic positions to break perfect symmetry after strain
    positions = atoms_fcc.get_positions()
    np_rng = np.random.default_rng(seed=42)
    positions += np_rng.normal(scale=0.01, size=positions.shape)
    atoms_fcc.set_positions(positions)

    dtype = torchsim_mace_mpa.dtype
    device = torchsim_mace_mpa.device
    # Convert the ASE Atoms object to SimState (will be a single batch with 4 atoms)
    return atoms_to_state(atoms_fcc, device=device, dtype=dtype)


# Helper function to run and compare optimizations
def _run_and_compare_optimizers(
    initial_sim_state_fixture: ts.state.SimState,
    torchsim_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    torch_sim_optimizer_type: str,
    ase_filter_class: Any,
    checkpoints: list[int],
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
) -> None:
    pytest.importorskip("mace")
    dtype = torch.float64
    device = torchsim_mace_mpa.device

    ts_current_system_state = copy.deepcopy(initial_sim_state_fixture).to(
        dtype=dtype, device=device
    )
    ts_current_system_state.positions = (
        ts_current_system_state.positions.detach().requires_grad_()
    )
    ts_current_system_state.cell = ts_current_system_state.cell.detach().requires_grad_()
    ts_optimizer_state = None

    optimizer_builders = {
        "frechet": frechet_cell_fire,
        "unit_cell": unit_cell_fire,
    }
    if torch_sim_optimizer_type not in optimizer_builders:
        raise ValueError(f"Unknown torch_sim_optimizer_type: {torch_sim_optimizer_type}")
    ts_optimizer_builder = optimizer_builders[torch_sim_optimizer_type]

    optimizer_callable_for_ts_optimize = lambda model, **_kwargs: ts_optimizer_builder(  # noqa: E731
        model, md_flavor="ase_fire"
    )

    ase_atoms_for_run = state_to_atoms(
        copy.deepcopy(initial_sim_state_fixture).to(dtype=dtype, device=device)
    )[0]
    ase_atoms_for_run.calc = ase_mace_mpa
    filtered_ase_atoms_for_run = ase_filter_class(ase_atoms_for_run)
    ase_optimizer = FIRE(filtered_ase_atoms_for_run, logfile=None)

    last_checkpoint_step_count = 0
    convergence_fn = ts.generate_force_convergence_fn(force_tol=force_tol)

    for checkpoint_step in checkpoints:
        steps_for_current_segment = checkpoint_step - last_checkpoint_step_count

        if steps_for_current_segment > 0:
            # Ensure requires_grad is set for the input to ts.optimize
            # ts.optimize is expected to return a state suitable for further optimization
            # if optimizer_state is passed.
            ts_current_system_state.positions = (
                ts_current_system_state.positions.detach().requires_grad_()
            )
            ts_current_system_state.cell = (
                ts_current_system_state.cell.detach().requires_grad_()
            )
            new_ts_state_and_optimizer_state = ts.optimize(
                system=ts_current_system_state,
                model=torchsim_mace_mpa,
                optimizer=optimizer_callable_for_ts_optimize,
                max_steps=steps_for_current_segment,
                convergence_fn=convergence_fn,
                optimizer_state=ts_optimizer_state,
            )
            ts_current_system_state = new_ts_state_and_optimizer_state
            ts_optimizer_state = new_ts_state_and_optimizer_state

            ase_optimizer.run(fmax=force_tol, steps=steps_for_current_segment)

        current_test_id = f"{test_id_prefix} (Step {checkpoint_step})"

        final_custom_energy = ts_current_system_state.energy.item()
        final_custom_forces_max = (
            torch.norm(ts_current_system_state.forces, dim=-1).max().item()
        )
        final_custom_positions = ts_current_system_state.positions.detach()
        final_custom_cell = ts_current_system_state.row_vector_cell.squeeze(0).detach()

        final_ase_atoms = filtered_ase_atoms_for_run.atoms
        final_ase_energy = final_ase_atoms.get_potential_energy()
        ase_forces_raw = final_ase_atoms.get_forces()
        final_ase_forces_max = (
            torch.norm(torch.tensor(ase_forces_raw, device=device, dtype=dtype), dim=-1)
            .max()
            .item()
            if ase_forces_raw is not None
            else float("nan")
        )
        final_ase_positions = torch.tensor(
            final_ase_atoms.get_positions(), device=device, dtype=dtype
        )
        final_ase_cell = torch.tensor(
            final_ase_atoms.get_cell(), device=device, dtype=dtype
        )

        energy_diff = abs(final_custom_energy - final_ase_energy)
        assert energy_diff < tolerances["energy"], (
            f"{current_test_id}: Final energies differ significantly: "
            f"torch-sim={final_custom_energy:.6f}, ASE={final_ase_energy:.6f}, "
            f"Diff={energy_diff:.2e}"
        )

        avg_displacement = (
            torch.norm(final_custom_positions - final_ase_positions, dim=-1).mean().item()
        )
        assert avg_displacement < tolerances["pos"], (
            f"{current_test_id}: Final positions differ ({avg_displacement=:.4f})"
        )

        cell_diff = torch.norm(final_custom_cell - final_ase_cell).item()
        assert cell_diff < tolerances["cell"], (
            f"{current_test_id}: Final cell matrices differ (Frobenius norm: "
            f"{cell_diff:.4f})\nTorch-sim Cell:\n{final_custom_cell}"
            f"\nASE Cell:\n{final_ase_cell}"
        )

        force_max_diff = abs(final_custom_forces_max - final_ase_forces_max)
        assert force_max_diff < tolerances["force_max"], (
            f"{current_test_id}: Max forces differ significantly: "
            f"torch-sim={final_custom_forces_max:.4f}, ASE={final_ase_forces_max:.4f}, "
            f"Diff={force_max_diff:.2e}"
        )

        last_checkpoint_step_count = checkpoint_step


@pytest.mark.parametrize(
    (
        "sim_state_fixture_name",
        "torch_sim_optimizer_type",
        "ase_filter_class",
        "checkpoints",
        "force_tol",
        "tolerances",
        "test_id_prefix",
    ),
    [
        (
            "rattled_sio2_sim_state",
            "frechet",
            FrechetCellFilter,
            [33, 66, 100],
            0.02,
            {"energy": 2e-3, "pos": 1.5e-2, "cell": 1.8e-2, "force_max": 1.5e-1},
            "SiO2 (Frechet)",
        ),
        (
            "osn2_sim_state",
            "frechet",
            FrechetCellFilter,
            [16, 33, 50],
            0.02,
            {"energy": 1e-4, "pos": 1e-3, "cell": 1.8e-3, "force_max": 5e-2},
            "OsN2 (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            "frechet",
            FrechetCellFilter,
            [33, 66, 100],
            0.01,
            {"energy": 1e-2, "pos": 5e-3, "cell": 2e-2, "force_max": 5e-2},
            "Triclinic Al (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            "unit_cell",
            UnitCellFilter,
            [33, 66, 100],
            0.01,
            {"energy": 1e-2, "pos": 3e-2, "cell": 1e-1, "force_max": 5e-2},
            "Triclinic Al (UnitCell)",
        ),
        (
            "rattled_sio2_sim_state",
            "unit_cell",
            UnitCellFilter,
            [33, 66, 100],
            0.02,
            {"energy": 5e-3, "pos": 2.5e-2, "cell": 5e-2, "force_max": 1.5e-1},
            "SiO2 (UnitCell)",
        ),
    ],
)
def test_optimizer_vs_ase_parametrized(
    sim_state_fixture_name: str,
    torch_sim_optimizer_type: str,
    ase_filter_class: Any,
    checkpoints: list[int],
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
    torchsim_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    request: pytest.FixtureRequest,
) -> None:
    """Compare torch-sim optimizers with ASE FIRE and relevant filters at multiple
    checkpoints."""
    initial_sim_state_fixture = request.getfixturevalue(sim_state_fixture_name)

    _run_and_compare_optimizers(
        initial_sim_state_fixture=initial_sim_state_fixture,
        torchsim_mace_mpa=torchsim_mace_mpa,
        ase_mace_mpa=ase_mace_mpa,
        torch_sim_optimizer_type=torch_sim_optimizer_type,
        ase_filter_class=ase_filter_class,
        checkpoints=checkpoints,
        force_tol=force_tol,
        tolerances=tolerances,
        test_id_prefix=test_id_prefix,
    )
