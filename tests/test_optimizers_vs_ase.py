from typing import TYPE_CHECKING, Any

import pytest
import torch
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.optimize import FIRE
from pymatgen.analysis.structure_matcher import StructureMatcher

import torch_sim as ts
from torch_sim.io import atoms_to_state, state_to_atoms, state_to_structures
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import frechet_cell_fire, unit_cell_fire


if TYPE_CHECKING:
    from mace.calculators import MACECalculator


def _compare_ase_and_ts_states(
    ts_current_system_state: ts.state.SimState,
    filtered_ase_atoms_for_run: Any,
    tolerances: dict[str, float],
    current_test_id: str,
) -> None:
    structure_matcher = StructureMatcher(
        ltol=tolerances["lattice_tol"],
        stol=tolerances["site_tol"],
        angle_tol=tolerances["angle_tol"],
        scale=False,
    )

    tensor_kwargs = {
        "device": ts_current_system_state.device,
        "dtype": ts_current_system_state.dtype,
    }

    final_custom_energy = ts_current_system_state.energy.item()
    final_custom_forces_max = (
        torch.norm(ts_current_system_state.forces, dim=-1).max().item()
    )

    # Convert torch-sim state to pymatgen Structure
    ts_structure = state_to_structures(ts_current_system_state)[0]

    # Convert ASE atoms to pymatgen Structure
    final_ase_atoms = filtered_ase_atoms_for_run.atoms
    final_ase_energy = final_ase_atoms.get_potential_energy()
    ase_forces_raw = final_ase_atoms.get_forces()
    final_ase_forces_max = torch.norm(
        torch.tensor(ase_forces_raw, **tensor_kwargs), dim=-1
    ).max()
    ts_state = atoms_to_state(final_ase_atoms, **tensor_kwargs)
    ase_structure = state_to_structures(ts_state)[0]

    # Compare energies
    energy_diff = abs(final_custom_energy - final_ase_energy)
    assert energy_diff < tolerances["energy"], (
        f"{current_test_id}: Final energies differ significantly: "
        f"torch-sim={final_custom_energy:.6f}, ASE={final_ase_energy:.6f}, "
        f"Diff={energy_diff:.2e}"
    )

    # Compare forces
    force_max_diff = abs(final_custom_forces_max - final_ase_forces_max)
    assert force_max_diff < tolerances["force_max"], (
        f"{current_test_id}: Max forces differ significantly: "
        f"torch-sim={final_custom_forces_max:.4f}, ASE={final_ase_forces_max:.4f}, "
        f"Diff={force_max_diff:.2e}"
    )

    # Compare structures using StructureMatcher
    assert structure_matcher.fit(ts_structure, ase_structure), (
        f"{current_test_id}: Structures do not match according to StructureMatcher\n"
        f"{ts_structure=}\n{ase_structure=}"
    )


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
    """Run and compare optimizations between torch-sim and ASE."""
    pytest.importorskip("mace")
    dtype = torch.float64
    device = torchsim_mace_mpa.device

    ts_current_system_state = initial_sim_state_fixture.clone()

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
        initial_sim_state_fixture.clone().to(dtype=dtype, device=device)
    )[0]
    ase_atoms_for_run.calc = ase_mace_mpa
    filtered_ase_atoms_for_run = ase_filter_class(ase_atoms_for_run)
    ase_optimizer = FIRE(filtered_ase_atoms_for_run, logfile=None)

    last_checkpoint_step_count = 0
    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol, include_cell_forces=True
    )

    results = torchsim_mace_mpa(ts_current_system_state)
    ts_initial_system_state = ts_current_system_state.clone()
    ts_initial_system_state.forces = results["forces"]
    ts_initial_system_state.energy = results["energy"]
    ase_atoms_for_run.calc.calculate(ase_atoms_for_run)

    _compare_ase_and_ts_states(
        ts_initial_system_state,
        filtered_ase_atoms_for_run,
        tolerances,
        f"{test_id_prefix} (Initial)",
    )

    for checkpoint_step in checkpoints:
        steps_for_current_segment = checkpoint_step - last_checkpoint_step_count

        if steps_for_current_segment > 0:
            updated_ts_state = ts.optimize(
                system=ts_current_system_state,
                model=torchsim_mace_mpa,
                optimizer=optimizer_callable_for_ts_optimize,
                max_steps=steps_for_current_segment,
                convergence_fn=convergence_fn,
                steps_between_swaps=1,
            )
            ts_current_system_state = updated_ts_state.clone()

            ase_optimizer.run(fmax=force_tol, steps=steps_for_current_segment)

        current_test_id = f"{test_id_prefix} (Step {checkpoint_step})"

        _compare_ase_and_ts_states(
            ts_current_system_state,
            filtered_ase_atoms_for_run,
            tolerances,
            current_test_id,
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
            [1, 33, 66, 100],
            0.02,
            {
                "energy": 1e-2,
                "force_max": 5e-2,
                "lattice_tol": 3e-2,
                "site_tol": 3e-2,
                "angle_tol": 1e-1,
            },
            "SiO2 (Frechet)",
        ),
        (
            "osn2_sim_state",
            "frechet",
            FrechetCellFilter,
            [1, 16, 33, 50],
            0.02,
            {
                "energy": 1e-2,
                "force_max": 5e-2,
                "lattice_tol": 3e-2,
                "site_tol": 3e-2,
                "angle_tol": 1e-1,
            },
            "OsN2 (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            "frechet",
            FrechetCellFilter,
            [1, 33, 66, 100],
            0.01,
            {
                "energy": 1e-2,
                "force_max": 5e-2,
                "lattice_tol": 3e-2,
                "site_tol": 3e-2,
                "angle_tol": 5e-1,
            },
            "Triclinic Al (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            "unit_cell",
            UnitCellFilter,
            [1, 33, 66, 100],
            0.01,
            {
                "energy": 1e-2,
                "force_max": 5e-2,
                "lattice_tol": 3e-2,
                "site_tol": 3e-2,
                "angle_tol": 5e-1,
            },
            "Triclinic Al (UnitCell)",
        ),
        (
            "rattled_sio2_sim_state",
            "unit_cell",
            UnitCellFilter,
            [1, 33, 66, 100],
            0.02,
            {
                "energy": 1e-2,
                "force_max": 5e-2,
                "lattice_tol": 3e-2,
                "site_tol": 3e-2,
                "angle_tol": 1e-1,
            },
            "SiO2 (UnitCell)",
        ),
        (
            "osn2_sim_state",
            "unit_cell",
            UnitCellFilter,
            [1, 16, 33, 50],
            0.02,
            {
                "energy": 1e-2,
                "force_max": 5e-2,
                "lattice_tol": 3e-2,
                "site_tol": 3e-2,
                "angle_tol": 1e-1,
            },
            "OsN2 (UnitCell)",
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
