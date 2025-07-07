"""Structural optimization with MACE using FIRE optimizer.
Comparing the ASE and VV FIRE optimizers.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
#     "plotly>=6.0.0",
# ]
# ///

import os
import time
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import torch
from ase.build import bulk
from ase.cell import Cell
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE as ASEFIRE
from mace.calculators.foundations_models import mace_mp
from mace.calculators.foundations_models import mace_mp as mace_mp_calculator_for_ase

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.optimizers import GDState, fire, frechet_cell_fire
from torch_sim.state import SimState


# Set device, data type and unit conversion
SMOKE_TEST = os.getenv("CI") is not None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
unit_conv = ts.units.UnitConversion

# Option 1: Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Number of steps to run
max_iterations = 10 if SMOKE_TEST else 500
supercell_scale = (1, 1, 1) if SMOKE_TEST else (3, 2, 2)
# Max steps for each individual ASE optimization run
ase_max_optimizer_steps = max_iterations * 10

# Set random seed for reproducibility
rng = np.random.default_rng(seed=0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat(supercell_scale)
si_dc.positions += 0.1 * rng.standard_normal(si_dc.positions.shape).clip(-1, 1)

# Create FCC Copper
cu_dc = bulk("Cu", "fcc", a=3.85).repeat([r + 1 for r in supercell_scale])
cu_dc.positions += 0.1 * rng.standard_normal(cu_dc.positions.shape).clip(-1, 1)

# Create BCC Iron
fe_dc = bulk("Fe", "bcc", a=2.95).repeat([r + 1 for r in supercell_scale])
fe_dc.positions += 0.1 * rng.standard_normal(fe_dc.positions.shape).clip(-1, 1)

si_dc_vac = si_dc.copy()
si_dc_vac.positions += 0.1 * rng.standard_normal(si_dc_vac.positions.shape).clip(-1, 1)
# select 2 numbers in range 0 to len(si_dc_vac)
indices = rng.choice(len(si_dc_vac), size=2, replace=False)
for idx in indices:
    si_dc_vac.pop(idx)


cu_dc_vac = cu_dc.copy()
cu_dc_vac.positions += 0.1 * rng.standard_normal(cu_dc_vac.positions.shape).clip(-1, 1)
# remove 2 atoms from cu_dc_vac at random
indices = rng.choice(len(cu_dc_vac), size=2, replace=False)
for idx in indices:
    index = idx + 3
    if index < len(cu_dc_vac):
        cu_dc_vac.pop(index)
    else:
        print(f"Index {index} is out of bounds for cu_dc_vac")
        cu_dc_vac.pop(0)

fe_dc_vac = fe_dc.copy()
fe_dc_vac.positions += 0.1 * rng.standard_normal(fe_dc_vac.positions.shape).clip(-1, 1)
# remove 2 atoms from fe_dc_vac at random
indices = rng.choice(len(fe_dc_vac), size=2, replace=False)
for idx in indices:
    index = idx + 2
    if index < len(fe_dc_vac):
        fe_dc_vac.pop(index)
    else:
        print(f"Index {index} is out of bounds for fe_dc_vac")
        fe_dc_vac.pop(0)


# Create a list of our atomic systems
atoms_list = [si_dc, cu_dc, fe_dc, si_dc_vac, cu_dc_vac, fe_dc_vac]

# Print structure information
print(f"Silicon atoms: {len(si_dc)}")
print(f"Copper atoms: {len(cu_dc)}")
print(f"Iron atoms: {len(fe_dc)}")
print(f"Total number of structures: {len(atoms_list)}")

# Create batched model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Convert atoms to state
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
# Run initial inference
initial_energies = model(state)["energy"]


def run_optimization_ts(  # noqa: PLR0915
    *,
    initial_state: SimState,
    ts_md_flavor: Literal["vv_fire", "ase_fire"],
    ts_use_frechet: bool,
    force_tol: float,
    max_iterations_ts: int,
) -> tuple[torch.Tensor, SimState | None]:
    """Runs Torch-Sim optimization and returns convergence steps and final state."""
    print(
        f"\n--- Running Torch-Sim optimization: flavor={ts_md_flavor}, "
        f"frechet_cell_opt={ts_use_frechet}, force_tol={force_tol} ---"
    )
    start_time = time.perf_counter()

    print("Initial cell parameters (Torch-Sim):")
    for k_idx in range(initial_state.n_graphs):
        cell_tensor_k = initial_state.cell[k_idx].cpu().numpy()
        ase_cell_k = Cell(cell_tensor_k)
        params_str = ", ".join([f"{p:.2f}" for p in ase_cell_k.cellpar()])
        print(
            f"  Structure {k_idx + 1}: Volume={ase_cell_k.volume:.2f} Å³, "
            f"Params=[{params_str}]"
        )

    if ts_use_frechet:
        init_fn_opt, update_fn_opt = frechet_cell_fire(
            model=model, md_flavor=ts_md_flavor
        )
    else:
        init_fn_opt, update_fn_opt = fire(model=model, md_flavor=ts_md_flavor)

    opt_state = init_fn_opt(initial_state.clone())

    batcher = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with="n_atoms",
        max_memory_scaler=1000,
        max_iterations=max_iterations_ts,
        return_indices=True,
    )
    batcher.load_states(opt_state)

    total_structures = opt_state.n_graphs
    convergence_steps = torch.full(
        (total_structures,), -1, dtype=torch.long, device=device
    )
    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol, include_cell_forces=ts_use_frechet
    )
    converged_tensor_global = torch.zeros(
        total_structures, dtype=torch.bool, device=device
    )
    global_step = 0
    all_converged_states = []
    convergence_tensor_for_batcher = None
    last_active_state = opt_state

    while True:
        result = batcher.next_batch(last_active_state, convergence_tensor_for_batcher)
        opt_state, converged_states_from_batcher, current_indices_list = result
        all_converged_states.extend(converged_states_from_batcher)

        if opt_state is None:
            print("All structures converged or batcher reached max iterations.")
            break

        last_active_state = opt_state
        current_indices = torch.tensor(
            current_indices_list, dtype=torch.long, device=device
        )

        steps_this_round = 1
        for _ in range(steps_this_round):
            opt_state = update_fn_opt(opt_state)
        global_step += steps_this_round

        convergence_tensor_for_batcher = convergence_fn(opt_state, None)
        newly_converged_mask_local = convergence_tensor_for_batcher & (
            convergence_steps[current_indices] == -1
        )
        converged_indices_global = current_indices[newly_converged_mask_local]

        if converged_indices_global.numel() > 0:
            convergence_steps[converged_indices_global] = global_step
            converged_tensor_global[converged_indices_global] = True
            total_converged_frac = converged_tensor_global.sum().item() / total_structures
            print(
                f"{global_step=}: Converged indices {converged_indices_global.tolist()}, "
                f"Total converged: {total_converged_frac:.2%}"
            )

        if global_step % 50 == 0:
            total_converged_frac = converged_tensor_global.sum().item() / total_structures
            active_structures = opt_state.n_graphs if opt_state else 0
            print(
                f"{global_step=}: Active structures: {active_structures}, "
                f"Total converged: {total_converged_frac:.2%}"
            )

    final_states_list = batcher.restore_original_order(all_converged_states)
    final_state_concatenated = ts.concatenate_states(final_states_list)

    if final_state_concatenated is not None and hasattr(final_state_concatenated, "cell"):
        print("Final cell parameters (Torch-Sim):")
        for k_idx in range(final_state_concatenated.n_graphs):
            cell_tensor_k = final_state_concatenated.cell[k_idx].cpu().numpy()
            ase_cell_k = Cell(cell_tensor_k)
            params_str = ", ".join([f"{p:.2f}" for p in ase_cell_k.cellpar()])
            print(
                f"  Structure {k_idx + 1}: Volume={ase_cell_k.volume:.2f} Å³, "
                f"Params=[{params_str}]"
            )
    else:
        print(
            "Final cell parameters (Torch-Sim): Not available (final_state_concatenated "
            "is None or has no cell)."
        )

    end_time = time.perf_counter()
    print(
        f"Finished Torch-Sim ({ts_md_flavor}, frechet={ts_use_frechet}) in "
        f"{end_time - start_time:.2f} seconds."
    )
    return convergence_steps, final_state_concatenated


def run_optimization_ase(  # noqa: C901, PLR0915
    *,
    initial_state: SimState,
    ase_use_frechet_filter: bool,
    force_tol: float,
    max_steps_ase: int,
) -> tuple[torch.Tensor, GDState | None]:
    """Runs ASE optimization and returns convergence steps and final state."""
    print(
        f"\n--- Running ASE optimization: frechet_filter={ase_use_frechet_filter}, "
        f"force_tol={force_tol} ---"
    )
    start_time = time.perf_counter()

    individual_initial_states = initial_state.split()
    num_structures = len(individual_initial_states)
    final_ase_atoms_list = []
    convergence_steps_list = []

    for i, single_sim_state in enumerate(individual_initial_states):
        print(f"Optimizing structure {i + 1}/{num_structures} with ASE...")
        ase_atoms_orig = ts.io.state_to_atoms(single_sim_state)[0]

        initial_cell_ase = ase_atoms_orig.get_cell()
        initial_params_str = ", ".join([f"{p:.2f}" for p in initial_cell_ase.cellpar()])
        print(
            f"  Initial cell (ASE Structure {i + 1}): "
            f"Volume={initial_cell_ase.volume:.2f} Å³, "
            f"Params=[{initial_params_str}]"
        )

        ase_calc_instance = mace_mp_calculator_for_ase(
            model=MaceUrls.mace_mpa_medium,
            device=device,
            default_dtype=str(dtype).split(".")[-1],
        )
        ase_atoms_orig.calc = ase_calc_instance

        optim_target_atoms = ase_atoms_orig
        if ase_use_frechet_filter:
            print(f"Applying FrechetCellFilter to structure {i + 1}")
            optim_target_atoms = FrechetCellFilter(ase_atoms_orig)

        dyn = ASEFIRE(optim_target_atoms, trajectory=None, logfile=None)

        try:
            dyn.run(fmax=force_tol, steps=max_steps_ase)
            if dyn.converged():
                convergence_steps_list.append(dyn.nsteps)
                print(f"ASE structure {i + 1} converged in {dyn.nsteps} steps.")
            else:
                print(
                    f"ASE optimization for structure {i + 1} did not converge within "
                    f"{max_steps_ase} steps. Steps taken: {dyn.nsteps}."
                )
                convergence_steps_list.append(-1)
        except Exception as e:  # noqa: BLE001
            print(f"ASE optimization failed for structure {i + 1}: {e}")
            convergence_steps_list.append(-1)

        final_ats_for_print = (
            optim_target_atoms.atoms if ase_use_frechet_filter else ase_atoms_orig
        )
        final_cell_ase = final_ats_for_print.get_cell()
        final_params_str = ", ".join([f"{p:.2f}" for p in final_cell_ase.cellpar()])
        print(
            f"  Final cell (ASE Structure {i + 1}): "
            f"Volume={final_cell_ase.volume:.2f} Å³, "
            f"Params=[{final_params_str}]"
        )

        final_ase_atoms_list.append(final_ats_for_print)

    all_positions = []
    all_masses = []
    all_atomic_numbers = []
    all_cells = []
    all_graphs_for_gd = []
    final_energies_ase = []
    final_forces_ase_tensors = []

    current_atom_offset = 0
    for graph_idx, ats_final in enumerate(final_ase_atoms_list):
        all_positions.append(
            torch.tensor(ats_final.get_positions(), device=device, dtype=dtype)
        )
        all_masses.append(
            torch.tensor(ats_final.get_masses(), device=device, dtype=dtype)
        )
        all_atomic_numbers.append(
            torch.tensor(ats_final.get_atomic_numbers(), device=device, dtype=torch.long)
        )
        # ASE cell is row-vector, SimState expects column-vector
        all_cells.append(
            torch.tensor(ats_final.get_cell().array.T, device=device, dtype=dtype)
        )

        num_atoms_in_current = len(ats_final)
        all_graphs_for_gd.append(
            torch.full(
                (num_atoms_in_current,), graph_idx, device=device, dtype=torch.long
            )
        )
        current_atom_offset += num_atoms_in_current

        try:
            if ats_final.calc is None:
                print(
                    "Re-attaching ASE calculator for final energy/forces for "
                    f"structure {graph_idx}."
                )
                temp_calc = mace_mp_calculator_for_ase(
                    model=MaceUrls.mace_mpa_medium,
                    device=device,
                    default_dtype=str(dtype).split(".")[-1],
                )
                ats_final.calc = temp_calc
            final_energies_ase.append(ats_final.get_potential_energy())
            final_forces_ase_tensors.append(
                torch.tensor(ats_final.get_forces(), device=device, dtype=dtype)
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"Could not get final energy/forces for an ASE structure {graph_idx}: {e}"
            )
            final_energies_ase.append(float("nan"))
            if all_positions and len(all_positions[-1]) > 0:
                final_forces_ase_tensors.append(torch.zeros_like(all_positions[-1]))
            else:
                final_forces_ase_tensors.append(
                    torch.empty((0, 3), device=device, dtype=dtype)
                )

    if not all_positions:  # If all optimizations failed early
        print("Warning: No successful ASE structures to form GDState.")
        return torch.tensor(convergence_steps_list, dtype=torch.long, device=device), None

    # Concatenate all parts
    concatenated_positions = torch.cat(all_positions, dim=0)
    concatenated_masses = torch.cat(all_masses, dim=0)
    concatenated_atomic_numbers = torch.cat(all_atomic_numbers, dim=0)
    concatenated_cells = torch.stack(all_cells, dim=0)  # Cells are (n_graphs, 3, 3)
    concatenated_graph_indices = torch.cat(all_graphs_for_gd, dim=0)

    concatenated_energies = torch.tensor(final_energies_ase, device=device, dtype=dtype)
    concatenated_forces = torch.cat(final_forces_ase_tensors, dim=0)

    # Check for NaN energies which might cause issues
    if torch.isnan(concatenated_energies).any():
        print(
            "Warning: NaN values found in final ASE energies. "
            "GDState energy tensor will contain NaNs."
        )

    # Create GDState instance
    final_state_as_gd = GDState(
        positions=concatenated_positions,
        masses=concatenated_masses,
        cell=concatenated_cells,
        pbc=initial_state.pbc,
        atomic_numbers=concatenated_atomic_numbers,
        graph_idx=concatenated_graph_indices,
        energy=concatenated_energies,
        forces=concatenated_forces,
    )

    convergence_steps = torch.tensor(
        convergence_steps_list, dtype=torch.long, device=device
    )

    end_time = time.perf_counter()
    print(
        f"Finished ASE optimization (frechet_filter={ase_use_frechet_filter}) "
        f"in {end_time - start_time:.2f} seconds."
    )
    return convergence_steps, final_state_as_gd


# --- Main Script ---
force_tol = 0.05

# Configurations to test
configs_to_run = [
    {
        "name": "torch-sim VV-FIRE (PosOnly)",
        "type": "torch_sim",
        "ts_md_flavor": "vv_fire",
        "ts_use_frechet": False,
    },
    {
        "name": "torch-sim ASE-FIRE (PosOnly)",
        "type": "torch_sim",
        "ts_md_flavor": "ase_fire",
        "ts_use_frechet": False,
    },
    {
        "name": "torch-sim VV-FIRE (Frechet Cell)",
        "type": "torch_sim",
        "ts_md_flavor": "vv_fire",
        "ts_use_frechet": True,
    },
    {
        "name": "torch-sim ASE-FIRE (Frechet Cell)",
        "type": "torch_sim",
        "ts_md_flavor": "ase_fire",
        "ts_use_frechet": True,
    },
    {
        "name": "ASE FIRE (Native, PosOnly)",
        "type": "ase",
        "ase_use_frechet_filter": False,
    },
    {
        "name": "ASE FIRE (Native Frechet Filter, CellOpt)",
        "type": "ase",
        "ase_use_frechet_filter": True,
    },
]

all_results = {}

for config_run in configs_to_run:
    print(f"\n\nStarting configuration: {config_run['name']}")
    optimizer_type_val = config_run["type"]
    ts_md_flavor_val = config_run.get("ts_md_flavor")
    ts_use_frechet_val = config_run.get("ts_use_frechet", False)
    ase_use_frechet_filter_val = config_run.get("ase_use_frechet_filter", False)

    steps: torch.Tensor | None = None
    final_state_opt: SimState | GDState | None = None

    if optimizer_type_val == "torch_sim":
        if ts_md_flavor_val is None:
            raise ValueError(f"{ts_md_flavor_val=} must be provided for torch_sim")
        steps, final_state_opt = run_optimization_ts(
            initial_state=state.clone(),
            ts_md_flavor=ts_md_flavor_val,
            ts_use_frechet=ts_use_frechet_val,
            force_tol=force_tol,
            max_iterations_ts=max_iterations,
        )
    elif optimizer_type_val == "ase":
        steps, final_state_opt = run_optimization_ase(
            initial_state=state.clone(),
            ase_use_frechet_filter=ase_use_frechet_filter_val,
            force_tol=force_tol,
            max_steps_ase=ase_max_optimizer_steps,
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type_val}")

    all_results[config_run["name"]] = {"steps": steps, "final_state": final_state_opt}


print("\n\n--- Overall Comparison ---")
print(f"{force_tol=:.2f} eV/Å")
print(f"Initial energies: {[f'{e.item():.3f}' for e in initial_energies]} eV")

for name, result_data in all_results.items():
    final_state_res = result_data["final_state"]
    steps_res = result_data["steps"]
    print(f"\nResults for: {name}")
    if (
        final_state_res is not None
        and hasattr(final_state_res, "energy")
        and final_state_res.energy is not None
    ):
        energy_str = [f"{e.item():.3f}" for e in final_state_res.energy]
        print(f"  Final energies: {energy_str} eV")
    else:
        print("  Final energies: Not available or state is None")
    print(f"  Convergence steps: {steps_res.tolist()}")

    not_converged_indices = torch.where(steps_res == -1)[0].tolist()
    if not_converged_indices:
        print(f"  Did not converge for structure indices: {not_converged_indices}")

comparison_pairs = [
    ("torch-sim ASE-FIRE (PosOnly)", "ASE FIRE (Native, PosOnly)"),
    ("torch-sim ASE-FIRE (Frechet Cell)", "ASE FIRE (Native Frechet Filter, CellOpt)"),
    ("torch-sim VV-FIRE (Frechet Cell)", "ASE FIRE (Native Frechet Filter, CellOpt)"),
    ("torch-sim VV-FIRE (PosOnly)", "ASE FIRE (Native, PosOnly)"),
]

for name1, name2 in comparison_pairs:
    if name1 in all_results and name2 in all_results:
        state1 = all_results[name1]["final_state"]
        state2 = all_results[name2]["final_state"]

        if state1 is None or state2 is None:
            print(f"\nCannot compare {name1} and {name2}, one or both states are None.")
            continue

        state1_list = state1.split()

        state2_list = state2.split()

        if len(state1_list) != len(state2_list):
            print(
                f"\nCannot compare {name1} and {name2}, different number of structures."
            )
            continue

        mean_displacements = []
        for s1, s2 in zip(state1_list, state2_list, strict=True):
            if s1.n_atoms == 0 or s2.n_atoms == 0:
                mean_displacements.append(float("nan"))
                continue
            pos1_centered = s1.positions - s1.positions.mean(dim=0, keepdim=True)
            pos2_centered = s2.positions - s2.positions.mean(dim=0, keepdim=True)
            if pos1_centered.shape != pos2_centered.shape:
                print(
                    f"Warning: Shape mismatch for {name1} vs {name2} in structure. "
                    "Skipping displacement calc."
                )
                mean_displacements.append(float("nan"))
                continue
            displacement = torch.norm(pos1_centered - pos2_centered, dim=1)
            mean_disp = torch.mean(displacement).item()
            mean_displacements.append(mean_disp)

        print(
            f"\nMean Disp ({name1} vs {name2}): "
            f"{[f'{d:.4f}' for d in mean_displacements]} Å"
        )
    else:
        print(
            f"\nSkipping displacement comparison for ({name1} vs {name2}), "
            "one or both results missing."
        )


# --- Plotting Results ---
original_structure_formulas = [ats.get_chemical_formula() for ats in atoms_list]
structure_names = [
    "Si_bulk",
    "Cu_bulk",
    "Fe_bulk",
    "Si_vac",
    "Cu_vac",
    "Fe_vac",
]
if len(structure_names) != len(atoms_list):
    print(
        f"Warning: Mismatch between custom structure_names ({len(structure_names)}) and "
        f"atoms_list ({len(atoms_list)}). Using custom names."
    )
num_structures_plot = len(structure_names)


# --- Plot 1: Convergence Steps (Multi-bar per structure) ---
plot_methods_fig1 = list(all_results)
num_methods_fig1 = len(plot_methods_fig1)
steps_data_fig1 = np.full((num_structures_plot, num_methods_fig1), np.nan)

for method_idx, method_name in enumerate(plot_methods_fig1):
    result_data = all_results[method_name]
    if result_data["final_state"] is None or result_data["steps"] is None:
        print(f"Plot1: Skipping steps for {method_name} as final_state or steps is None.")
        continue

    steps_tensor = result_data["steps"].cpu().numpy()
    penalty_steps = ase_max_optimizer_steps + 100
    steps_plot_values = np.where(steps_tensor == -1, penalty_steps, steps_tensor)

    if len(steps_plot_values) == num_structures_plot:
        steps_data_fig1[:, method_idx] = steps_plot_values
    elif len(steps_plot_values) > num_structures_plot:
        print(
            f"Warning: More step values ({len(steps_plot_values)}) than "
            f"structure names ({num_structures_plot}) for {method_name}. "
            "Truncating."
        )
        steps_data_fig1[:, method_idx] = steps_plot_values[:num_structures_plot]
    elif len(steps_plot_values) < num_structures_plot:
        print(
            f"Warning: Fewer step values ({len(steps_plot_values)}) than "
            f"structure names ({num_structures_plot}) for {method_name}. "
            "Padding with NaN."
        )
        steps_data_fig1[: len(steps_plot_values), method_idx] = steps_plot_values

fig1_plotly = go.Figure()

for i in range(num_methods_fig1):
    fig1_plotly.add_bar(
        name=plot_methods_fig1[i],
        x=structure_names,
        y=steps_data_fig1[:, i],
        text=[
            "NC"
            if all_results[plot_methods_fig1[i]]["steps"].cpu().numpy()[bar_idx] == -1
            and not np.isnan(steps_data_fig1[bar_idx, i])
            else ""
            for bar_idx in range(num_structures_plot)
        ],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=10, family="Arial, sans-serif"),
    )

fig1_plotly.update_layout(
    barmode="group",
    title_text="Convergence Steps per Structure and Method",
    xaxis_title="Structure",
    yaxis_title="Convergence Steps (NC = Not Converged, shown at penalty)",
    legend_title="Optimization Method",
    xaxis_tickangle=-45,
    yaxis_gridcolor="lightgrey",
    plot_bgcolor="white",
    height=600,
    width=max(1000, 150 * num_structures_plot),
    margin=dict(l=50, r=50, b=100, t=50, pad=4),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig1_plotly.update_xaxes(categoryorder="array", categoryarray=structure_names)


# --- Plot 2: Average Final Energy Difference from Baselines ---
baseline_ase_pos_only = "ASE FIRE (Native, PosOnly)"
baseline_ase_frechet = "ASE FIRE (Native Frechet Filter, CellOpt)"
avg_energy_diffs_fig2 = []
plot_names_fig2 = []

baseline_pos_only_data = all_results.get(baseline_ase_pos_only)
baseline_frechet_data = all_results.get(baseline_ase_frechet)

for name, result_data in all_results.items():
    if result_data["final_state"] is None or result_data["final_state"].energy is None:
        print(f"Plot2: Skipping energy diff for {name} as final_state or energy is None.")
        if name not in plot_names_fig2:
            plot_names_fig2.append(name)
        avg_energy_diffs_fig2.append(np.nan)
        continue

    if name not in plot_names_fig2:
        plot_names_fig2.append(name)

    current_energies = result_data["final_state"].energy.cpu().numpy()

    chosen_baseline_energies = None
    is_baseline_self = False
    processed_current_name = False

    if name in (baseline_ase_pos_only, baseline_ase_frechet):
        avg_energy_diffs_fig2.append(0.0)
        is_baseline_self = True
        processed_current_name = True
    elif "torch-sim" in name:
        if "PosOnly" in name:
            if (
                baseline_pos_only_data
                and baseline_pos_only_data["final_state"] is not None
                and baseline_pos_only_data["final_state"].energy is not None
            ):
                chosen_baseline_energies = (
                    baseline_pos_only_data["final_state"].energy.cpu().numpy()
                )
        elif "Frechet Cell" in name and (
            baseline_frechet_data
            and baseline_frechet_data["final_state"] is not None
            and baseline_frechet_data["final_state"].energy is not None
        ):
            chosen_baseline_energies = (
                baseline_frechet_data["final_state"].energy.cpu().numpy()
            )

    if not is_baseline_self and not processed_current_name:
        if chosen_baseline_energies is not None:
            if current_energies.shape == chosen_baseline_energies.shape:
                energy_diff = np.mean(current_energies - chosen_baseline_energies)
                avg_energy_diffs_fig2.append(energy_diff)
            else:
                avg_energy_diffs_fig2.append(np.nan)
                print(
                    f"Plot2: Shape mismatch for energy comparison: {name} vs baseline. "
                    f"{current_energies.shape} vs {chosen_baseline_energies.shape}"
                )
        else:
            print(
                f"Plot2: No appropriate baseline for {name} or baseline data missing. "
                "Setting energy diff to NaN."
            )
            avg_energy_diffs_fig2.append(np.nan)
    elif not processed_current_name and name not in [
        n
        for n, v in zip(plot_names_fig2, avg_energy_diffs_fig2, strict=False)
        if not np.isnan(v)
    ]:
        print(f"Plot2: Fallback for {name}, setting energy diff to NaN.")
        avg_energy_diffs_fig2.append(np.nan)

final_plot_names_fig2 = []
final_avg_energy_diffs_fig2 = []
all_method_names_sorted = sorted(all_results)

for name in all_method_names_sorted:
    result_data = all_results[name]
    final_plot_names_fig2.append(name)
    if result_data["final_state"] is None or result_data["final_state"].energy is None:
        final_avg_energy_diffs_fig2.append(np.nan)
        continue

    current_energies = result_data["final_state"].energy.cpu().numpy()
    energy_to_append = np.nan

    if name in (baseline_ase_pos_only, baseline_ase_frechet):
        energy_to_append = 0.0
    elif "torch-sim" in name:
        baseline_to_use_energies = None
        if "PosOnly" in name:
            if (
                baseline_pos_only_data
                and baseline_pos_only_data["final_state"] is not None
                and baseline_pos_only_data["final_state"].energy is not None
            ):
                baseline_to_use_energies = (
                    baseline_pos_only_data["final_state"].energy.cpu().numpy()
                )
        elif "Frechet Cell" in name and (
            baseline_frechet_data
            and baseline_frechet_data["final_state"] is not None
            and baseline_frechet_data["final_state"].energy is not None
        ):
            baseline_to_use_energies = (
                baseline_frechet_data["final_state"].energy.cpu().numpy()
            )

        if baseline_to_use_energies is not None:
            if current_energies.shape == baseline_to_use_energies.shape:
                energy_to_append = np.mean(current_energies - baseline_to_use_energies)
            else:
                print(
                    f"Plot2: Shape mismatch for {name} ({current_energies.shape}) "
                    f"vs baseline ({baseline_to_use_energies.shape})."
                )
    final_avg_energy_diffs_fig2.append(energy_to_append)


fig2_plotly = go.Figure()
fig2_plotly.add_bar(
    x=final_plot_names_fig2,
    y=final_avg_energy_diffs_fig2,
    marker_color="lightcoral",
    text=[
        f"{yval:.3f}" if not np.isnan(yval) else ""
        for yval in final_avg_energy_diffs_fig2
    ],
    textposition="auto",
    textfont=dict(size=10),
)

line_dict = dict(color="black", width=1, dash="dash")
x1 = len(final_plot_names_fig2) - 0.5
fig2_plotly.update_layout(
    title_text="Average Final Energy Difference from ASE Baselines",
    xaxis_title="Optimization Method",
    yaxis_title="Avg. Final Energy Diff. from Corresponding ASE Baseline (eV)",
    xaxis_tickangle=-45,
    yaxis_gridcolor="lightgrey",
    plot_bgcolor="white",
    shapes=[dict(type="line", y0=0, y1=0, x0=-0.5, x1=x1, line=line_dict)],
    height=600,
    width=max(800, 100 * len(final_plot_names_fig2)),
    margin=dict(l=50, r=50, b=150, t=50, pad=4),
)


# --- Plot 3: Mean Displacement from ASE Counterparts (Multi-bar per structure) ---
# look at sets of: (ts_method_name, ase_method_name, short_label_for_legend)
comparison_pairs_plot3_defs = [
    (
        "torch-sim ASE-FIRE (PosOnly)",
        baseline_ase_pos_only,
        "TS ASE PosOnly vs ASE Native",
    ),
    ("torch-sim VV-FIRE (PosOnly)", baseline_ase_pos_only, "TS VV PosOnly vs ASE Native"),
    (
        "torch-sim ASE-FIRE (Frechet Cell)",
        baseline_ase_frechet,
        "TS ASE Frechet vs ASE Frechet",
    ),
    (
        "torch-sim VV-FIRE (Frechet Cell)",
        baseline_ase_frechet,
        "TS VV Frechet vs ASE Frechet",
    ),
]
num_comparison_pairs_plot3 = len(comparison_pairs_plot3_defs)
# rows: structures, cols: comparison_pair
disp_data_fig3 = np.full((num_structures_plot, num_comparison_pairs_plot3), np.nan)
legend_labels_fig3 = []

for pair_idx, (ts_method_name, ase_method_name, plot_label) in enumerate(
    comparison_pairs_plot3_defs
):
    legend_labels_fig3.append(plot_label)
    if ts_method_name in all_results and ase_method_name in all_results:
        state1_data = all_results[ts_method_name]
        state2_data = all_results[ase_method_name]

        if state1_data["final_state"] is None or state2_data["final_state"] is None:
            print(
                f"Plot3: Skipping displacement for {plot_label} due to missing state data"
            )
            continue

        state1_list = state1_data["final_state"].split()
        state2_list = state2_data["final_state"].split()

        if (
            len(state1_list) != len(state2_list)
            or len(state1_list) != num_structures_plot
        ):
            print(
                f"Plot3: Structure count mismatch for {plot_label}. "
                f"Expected {num_structures_plot}, got S1:{len(state1_list)}, "
                f"S2:{len(state2_list)}"
            )
            continue

        mean_displacements_for_this_pair = []
        for s1, s2 in zip(state1_list, state2_list, strict=True):
            if s1.n_atoms == 0 or s2.n_atoms == 0 or s1.n_atoms != s2.n_atoms:
                mean_displacements_for_this_pair.append(np.nan)
                continue
            pos1_centered = s1.positions - s1.positions.mean(dim=0, keepdim=True)
            pos2_centered = s2.positions - s2.positions.mean(dim=0, keepdim=True)
            displacement = torch.norm(pos1_centered - pos2_centered, dim=1)
            mean_disp = torch.mean(displacement).item()
            mean_displacements_for_this_pair.append(mean_disp)

        if len(mean_displacements_for_this_pair) == num_structures_plot:
            disp_data_fig3[:, pair_idx] = np.array(mean_displacements_for_this_pair)
        else:
            print(f"Plot3: Inner loop displacement calculation mismatch for {plot_label}")

    else:
        print(f"Plot3: Missing data for methods in pair: {plot_label}.")

fig3_plotly = go.Figure()

for idx, name in enumerate(legend_labels_fig3):
    fig3_plotly.add_bar(name=name, x=structure_names, y=disp_data_fig3[:, idx])


title = "Mean Displacement of Torch-Sim Methods to ASE Counterparts (per Structure)"
fig3_plotly.update_layout(
    barmode="group",
    title=dict(text=title, x=0.5, y=1),
    xaxis_title="Structure",
    yaxis_title="Mean Atomic Displacement (Å) to ASE Counterpart",
    legend_title="Comparison Pair",
    xaxis_tickangle=-45,
    yaxis_gridcolor="lightgrey",
    plot_bgcolor="white",
    height=600,
    width=max(1000, 150 * num_structures_plot),  # Adjust width
    margin=dict(l=50, r=50, b=100, t=50, pad=4),
    legend=dict(orientation="h", yanchor="bottom", y=0.96, xanchor="right", x=1),
)

# Show Plotly figures
fig1_plotly.show()
fig2_plotly.show()
fig3_plotly.show()
