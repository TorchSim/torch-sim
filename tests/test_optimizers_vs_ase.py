import copy
import functools
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.filters import ExpCellFilter, FrechetCellFilter
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
    n_steps: int,
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
) -> None:
    pytest.importorskip("mace")
    dtype = torch.float64
    device = torchsim_mace_mpa.device

    # --- Setup torch-sim part ---
    ts_initial_state = copy.deepcopy(initial_sim_state_fixture).to(
        dtype=dtype, device=device
    )
    ts_initial_state.positions = ts_initial_state.positions.detach().requires_grad_()
    ts_initial_state.cell = ts_initial_state.cell.detach().requires_grad_()

    if torch_sim_optimizer_type == "frechet":
        ts_optimizer_builder = frechet_cell_fire
    elif torch_sim_optimizer_type == "unit_cell":
        ts_optimizer_builder = unit_cell_fire
    else:
        raise ValueError(f"Unknown torch_sim_optimizer_type: {torch_sim_optimizer_type}")

    torch_sim_optimizer_factory = functools.partial(
        ts_optimizer_builder, md_flavor="ase_fire"
    )

    custom_opt_state = ts.optimize(
        system=ts_initial_state,
        model=torchsim_mace_mpa,
        optimizer=torch_sim_optimizer_factory,
        max_steps=n_steps,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=force_tol),
    )

    # --- Setup ASE part ---
    ase_atoms_for_run = state_to_atoms(
        copy.deepcopy(initial_sim_state_fixture).to(dtype=dtype, device=device)
    )[0]
    ase_atoms_for_run.calc = ase_mace_mpa

    filtered_ase_atoms_for_run = ase_filter_class(ase_atoms_for_run)
    ase_optimizer = FIRE(filtered_ase_atoms_for_run, logfile=None)
    ase_optimizer.run(fmax=force_tol, steps=n_steps)

    # --- Compare Results ---
    final_custom_energy = custom_opt_state.energy.item()
    final_custom_forces_max = torch.norm(custom_opt_state.forces, dim=-1).max().item()
    final_custom_positions = custom_opt_state.positions.detach()
    final_custom_cell = custom_opt_state.row_vector_cell.squeeze(0).detach()

    final_ase_atoms = filtered_ase_atoms_for_run.atoms
    final_ase_energy = final_ase_atoms.get_potential_energy()
    ase_forces_raw = final_ase_atoms.get_forces()
    if ase_forces_raw is not None:
        final_ase_forces = torch.tensor(ase_forces_raw, device=device, dtype=dtype)
        final_ase_forces_max = torch.norm(final_ase_forces, dim=-1).max().item()
    else:
        final_ase_forces_max = float("nan")

    final_ase_positions = torch.tensor(
        final_ase_atoms.get_positions(), device=device, dtype=dtype
    )
    final_ase_cell = torch.tensor(final_ase_atoms.get_cell(), device=device, dtype=dtype)

    energy_diff = abs(final_custom_energy - final_ase_energy)
    assert energy_diff < tolerances["energy"], (
        f"{test_id_prefix}: Final energies differ significantly after {n_steps} steps: "
        f"torch-sim={final_custom_energy:.6f}, ASE={final_ase_energy:.6f}, "
        f"Diff={energy_diff:.2e}"
    )

    print(
        f"{test_id_prefix}: Max Force ({n_steps} steps): "
        f"torch-sim={final_custom_forces_max:.4f}, ASE={final_ase_forces_max:.4f}"
    )

    avg_displacement = (
        torch.norm(final_custom_positions - final_ase_positions, dim=-1).mean().item()
    )
    assert avg_displacement < tolerances["pos"], (
        f"{test_id_prefix}: Final positions differ ({avg_displacement=:.4f})"
    )

    cell_diff = torch.norm(final_custom_cell - final_ase_cell).item()
    assert cell_diff < tolerances["cell"], (
        f"{test_id_prefix}: Final cell matrices differ (Frobenius norm: {cell_diff:.4f})"
        f"\nTorch-sim Cell:\n{final_custom_cell}"
        f"\nASE Cell:\n{final_ase_cell}"
    )


# Parameterized test function
@pytest.mark.parametrize(
    (
        "sim_state_fixture_name",
        "torch_sim_optimizer_type",
        "ase_filter_class",
        "n_steps",
        "force_tol",
        "tolerances",
        "test_id_prefix",
    ),
    [
        (
            "rattled_sio2_sim_state",
            "frechet",
            FrechetCellFilter,
            100,
            0.02,
            {"energy": 5e-4, "pos": 1e-2, "cell": 1e-2},
            "SiO2 (Frechet)",
        ),
        (
            "osn2_sim_state",
            "frechet",
            FrechetCellFilter,
            50,
            0.02,
            {"energy": 1e-4, "pos": 1e-3, "cell": 1e-3},
            "OsN2 (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            "frechet",
            FrechetCellFilter,
            100,
            0.01,
            {"energy": 1e-2, "pos": 5e-3, "cell": 2e-2},
            "Triclinic Al (Frechet)",
        ),
        (
            "distorted_fcc_al_conventional_sim_state",
            "unit_cell",
            ExpCellFilter,
            100,
            0.01,
            {"energy": 1e-2, "pos": 3e-2, "cell": 1e-1},
            "Triclinic Al (UnitCell)",
        ),
    ],
)
def test_optimizer_vs_ase_parametrized(
    sim_state_fixture_name: str,
    torch_sim_optimizer_type: str,
    ase_filter_class: Any,
    n_steps: int,
    force_tol: float,
    tolerances: dict[str, float],
    test_id_prefix: str,
    torchsim_mace_mpa: MaceModel,
    ase_mace_mpa: "MACECalculator",
    request: pytest.FixtureRequest,
) -> None:
    """Compare torch-sim optimizers with ASE FIRE and relevant filters."""
    initial_sim_state_fixture = request.getfixturevalue(sim_state_fixture_name)

    _run_and_compare_optimizers(
        initial_sim_state_fixture=initial_sim_state_fixture,
        torchsim_mace_mpa=torchsim_mace_mpa,
        ase_mace_mpa=ase_mace_mpa,
        torch_sim_optimizer_type=torch_sim_optimizer_type,
        ase_filter_class=ase_filter_class,
        n_steps=n_steps,
        force_tol=force_tol,
        tolerances=tolerances,
        test_id_prefix=test_id_prefix,
    )
