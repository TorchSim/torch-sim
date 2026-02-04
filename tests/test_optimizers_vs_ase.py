import traceback
from typing import TYPE_CHECKING, Any

import pytest
import torch
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.optimize import BFGS as ASE_BFGS
from ase.optimize import FIRE
from ase.optimize import LBFGS as ASE_LBFGS
from pymatgen.analysis.structure_matcher import StructureMatcher

import torch_sim as ts
from tests.conftest import DTYPE
from torch_sim.models.mace import MaceModel, MaceUrls


if TYPE_CHECKING:
    from mace.calculators import MACECalculator


@pytest.fixture
def ts_mace_mpa() -> MaceModel:
    """Provides a MACE MP model instance for the optimizer tests."""
    try:
        from mace.calculators.foundations_models import mace_mp
    except ImportError:
        pytest.skip(
            f"MACE not installed: {traceback.format_exc()}", allow_module_level=True
        )

    # Use float64 for potentially higher precision needed in optimization
    dtype = getattr(torch, dtype_str := "float64")
    raw_mace = mace_mp(
        model=MaceUrls.mace_mp_small, return_raw_model=True, default_dtype=dtype_str
    )
    return MaceModel(
        model=raw_mace,
        device=torch.device("cpu"),
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )


@pytest.fixture
def ase_mace_mpa() -> "MACECalculator":
    """Provides an ASE MACECalculator instance using mace_mp."""
    try:
        from mace.calculators.foundations_models import mace_mp
    except ImportError:
        pytest.skip(
            f"MACE not installed: {traceback.format_exc()}", allow_module_level=True
        )

    # Ensure dtype matches the one used in the torch-sim fixture (float64)
    return mace_mp(model=MaceUrls.mace_mp_small, default_dtype="float64")


def _compare_ase_and_ts_states(
    state: ts.FireState,
    filtered_ase_atoms: FrechetCellFilter | UnitCellFilter,
    tolerances: dict[str, float],
    current_test_id: str,
) -> None:
    structure_matcher = StructureMatcher(
        ltol=tolerances["lattice_tol"],
        stol=tolerances["site_tol"],
        angle_tol=tolerances["angle_tol"],
        scale=False,
    )
    tensor_kwargs = {"device": state.device, "dtype": state.dtype}

    final_custom_energy = state.energy.item()
    final_custom_forces_max = torch.norm(state.forces, dim=-1).max().item()

    # Convert torch-sim state to pymatgen Structure
    ts_structure = ts.io.state_to_structures(state)[0]

    # Convert ASE atoms to pymatgen Structure
    final_ase_atoms = filtered_ase_atoms.atoms
    final_ase_energy = final_ase_atoms.get_potential_energy()
    ase_forces_raw = final_ase_atoms.get_forces()
    final_ase_forces_max = torch.norm(
        torch.tensor(ase_forces_raw, **tensor_kwargs), dim=-1
    ).max()
    ts_state = ts.io.atoms_to_state(final_ase_atoms, **tensor_kwargs)
    ase_structure = ts.io.state_to_structures(ts_state)[0]

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
    initial_sim_state_fixture: ts.SimState,
    ts_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    fire_type: ts.Optimizer,
    cell_filter: ts.CellFilter,
    ase_filter_cls: FrechetCellFilter | UnitCellFilter,
    checkpoints: list[int],
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
    **optim_kwargs: Any,
) -> None:
    """Run and compare optimizations between torch-sim and ASE."""
    pytest.importorskip("mace")
    device = ts_mace_mpa.device

    state = initial_sim_state_fixture.clone()

    ase_atoms = ts.io.state_to_atoms(
        initial_sim_state_fixture.clone().to(dtype=DTYPE, device=device)
    )[0]
    ase_atoms.calc = ase_mace_mpa
    filtered_ase_atoms = ase_filter_cls(ase_atoms)  # type: ignore[call-non-callable]
    ase_optimizer = FIRE(filtered_ase_atoms, logfile=None)

    last_checkpoint_step_count = 0
    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol, include_cell_forces=True
    )

    results = ts_mace_mpa(state)
    ts_initial_system_state = state.clone()
    ts_initial_system_state.forces = results["forces"]
    ts_initial_system_state.energy = results["energy"]
    ase_mace_mpa.calculate(ase_atoms)

    _compare_ase_and_ts_states(
        ts_initial_system_state,
        filtered_ase_atoms,
        tolerances,
        f"{test_id_prefix} (Initial)",
    )

    for checkpoint_step in checkpoints:
        steps_for_current_segment = checkpoint_step - last_checkpoint_step_count

        if steps_for_current_segment > 0:
            updated_ts_state = ts.optimize(
                system=state,
                model=ts_mace_mpa,
                optimizer=fire_type,
                max_steps=steps_for_current_segment,
                convergence_fn=convergence_fn,
                steps_between_swaps=1,
                fire_flavor="ase_fire",  # optimizer kwargs
                init_kwargs=dict(cell_filter=cell_filter),
                **optim_kwargs,
            )
            state = updated_ts_state.clone()

            ase_optimizer.run(fmax=force_tol, steps=steps_for_current_segment)

        current_test_id = f"{test_id_prefix} (Step {checkpoint_step})"

        _compare_ase_and_ts_states(state, filtered_ase_atoms, tolerances, current_test_id)

        last_checkpoint_step_count = checkpoint_step


@pytest.mark.parametrize(
    (
        "sim_state_fixture_name",
        "fire_type",
        "cell_filter",
        "ase_filter_cls",
        "checkpoints",
        "force_tol",
        "tolerances",
        "test_id_prefix",
    ),
    [
        (
            "rattled_sio2_sim_state",
            ts.Optimizer.fire,
            ts.CellFilter.frechet,
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
            ts.Optimizer.fire,
            ts.CellFilter.frechet,
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
            ts.Optimizer.fire,
            ts.CellFilter.frechet,
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
            ts.Optimizer.fire,
            ts.CellFilter.unit,
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
            ts.Optimizer.fire,
            ts.CellFilter.unit,
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
            ts.Optimizer.fire,
            ts.CellFilter.unit,
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
    fire_type: ts.Optimizer,
    cell_filter: ts.CellFilter,
    ase_filter_cls: FrechetCellFilter | UnitCellFilter,
    checkpoints: list[int],
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
    ts_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    request: pytest.FixtureRequest,
) -> None:
    """Compare torch-sim optimizers with ASE FIRE and relevant filters at multiple
    checkpoints."""
    initial_sim_state_fixture = request.getfixturevalue(sim_state_fixture_name)

    _run_and_compare_optimizers(
        initial_sim_state_fixture=initial_sim_state_fixture,
        ts_mace_mpa=ts_mace_mpa,
        ase_mace_mpa=ase_mace_mpa,
        fire_type=fire_type,
        cell_filter=cell_filter,
        ase_filter_cls=ase_filter_cls,
        checkpoints=checkpoints,
        force_tol=force_tol,
        tolerances=tolerances,
        test_id_prefix=test_id_prefix,
    )


# TODO (AG): Can we merge these tests with the FIRE tests?


@pytest.mark.parametrize(
    (
        "sim_state_fixture_name",
        "cell_filter",
        "ase_filter_cls",
        "checkpoints",
        "force_tol",
        "tolerances",
        "test_id_prefix",
    ),
    [
        (
            "rattled_sio2_sim_state",
            ts.CellFilter.frechet,
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
            "BFGS SiO2 (Frechet)",
        ),
        (
            "osn2_sim_state",
            ts.CellFilter.frechet,
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
            "BFGS OsN2 (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            ts.CellFilter.frechet,
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
            "BFGS Triclinic Al (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            ts.CellFilter.unit,
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
            "BFGS Triclinic Al (UnitCell)",
        ),
        (
            "rattled_sio2_sim_state",
            ts.CellFilter.unit,
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
            "BFGS SiO2 (UnitCell)",
        ),
        (
            "osn2_sim_state",
            ts.CellFilter.unit,
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
            "BFGS OsN2 (UnitCell)",
        ),
    ],
)
def test_bfgs_vs_ase_parametrized(
    sim_state_fixture_name: str,
    cell_filter: ts.CellFilter,
    ase_filter_cls: type,
    checkpoints: list[int],
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
    ts_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    request: pytest.FixtureRequest,
) -> None:
    """Compare torch-sim BFGS with ASE BFGS at multiple checkpoints."""
    pytest.importorskip("mace")
    device = ts_mace_mpa.device

    initial_sim_state = request.getfixturevalue(sim_state_fixture_name)
    state = initial_sim_state.clone()

    ase_atoms = ts.io.state_to_atoms(
        initial_sim_state.clone().to(dtype=DTYPE, device=device)
    )[0]
    ase_atoms.calc = ase_mace_mpa
    filtered_ase_atoms = ase_filter_cls(ase_atoms)
    ase_optimizer = ASE_BFGS(filtered_ase_atoms, logfile=None, alpha=70.0)

    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol, include_cell_forces=True
    )

    # Compare initial state
    results = ts_mace_mpa(state)
    ts_initial = state.clone()
    ts_initial.forces = results["forces"]
    ts_initial.energy = results["energy"]
    ase_mace_mpa.calculate(ase_atoms)
    _compare_ase_and_ts_states(
        ts_initial, filtered_ase_atoms, tolerances, f"{test_id_prefix} (Initial)"
    )

    last_step = 0
    for checkpoint in checkpoints:
        steps = checkpoint - last_step
        if steps > 0:
            state = ts.optimize(
                system=state,
                model=ts_mace_mpa,
                optimizer=ts.Optimizer.bfgs,
                max_steps=steps,
                convergence_fn=convergence_fn,
                steps_between_swaps=1,
                init_kwargs=dict(cell_filter=cell_filter),
            )
            ase_optimizer.run(fmax=force_tol, steps=steps)

        _compare_ase_and_ts_states(
            state, filtered_ase_atoms, tolerances, f"{test_id_prefix} (Step {checkpoint})"
        )
        last_step = checkpoint


# TODO (AG): Can we merge these tests with the FIRE tests?


@pytest.mark.parametrize(
    (
        "sim_state_fixture_name",
        "cell_filter",
        "ase_filter_cls",
        "checkpoints",
        "force_tol",
        "tolerances",
        "test_id_prefix",
    ),
    [
        (
            "rattled_sio2_sim_state",
            ts.CellFilter.frechet,
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
            "LBFGS SiO2 (Frechet)",
        ),
        (
            "osn2_sim_state",
            ts.CellFilter.frechet,
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
            "LBFGS OsN2 (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            ts.CellFilter.frechet,
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
            "LBFGS Triclinic Al (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            ts.CellFilter.unit,
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
            "LBFGS Triclinic Al (UnitCell)",
        ),
        (
            "rattled_sio2_sim_state",
            ts.CellFilter.unit,
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
            "LBFGS SiO2 (UnitCell)",
        ),
        (
            "osn2_sim_state",
            ts.CellFilter.unit,
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
            "LBFGS OsN2 (UnitCell)",
        ),
    ],
)
def test_lbfgs_vs_ase_parametrized(
    sim_state_fixture_name: str,
    cell_filter: ts.CellFilter,
    ase_filter_cls: type,
    checkpoints: list[int],
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
    ts_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    request: pytest.FixtureRequest,
) -> None:
    """Compare torch-sim L-BFGS with ASE LBFGS at multiple checkpoints."""
    pytest.importorskip("mace")
    device = ts_mace_mpa.device

    initial_sim_state = request.getfixturevalue(sim_state_fixture_name)
    state = initial_sim_state.clone()

    ase_atoms = ts.io.state_to_atoms(
        initial_sim_state.clone().to(dtype=DTYPE, device=device)
    )[0]
    ase_atoms.calc = ase_mace_mpa
    filtered_ase_atoms = ase_filter_cls(ase_atoms)
    ase_optimizer = ASE_LBFGS(filtered_ase_atoms, logfile=None, alpha=70.0, damping=1.0)

    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol, include_cell_forces=True
    )

    # Compare initial state
    results = ts_mace_mpa(state)
    ts_initial = state.clone()
    ts_initial.forces = results["forces"]
    ts_initial.energy = results["energy"]
    ase_mace_mpa.calculate(ase_atoms)
    _compare_ase_and_ts_states(
        ts_initial, filtered_ase_atoms, tolerances, f"{test_id_prefix} (Initial)"
    )

    last_step = 0
    for checkpoint in checkpoints:
        steps = checkpoint - last_step
        if steps > 0:
            state = ts.optimize(
                system=state,
                model=ts_mace_mpa,
                optimizer=ts.Optimizer.lbfgs,
                max_steps=steps,
                convergence_fn=convergence_fn,
                steps_between_swaps=1,
                init_kwargs=dict(cell_filter=cell_filter, alpha=70.0, step_size=1.0),
                max_step=0.2,
            )
            ase_optimizer.run(fmax=force_tol, steps=steps)

        _compare_ase_and_ts_states(
            state, filtered_ase_atoms, tolerances, f"{test_id_prefix} (Step {checkpoint})"
        )
        last_step = checkpoint
