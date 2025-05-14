"""Structural optimization with MACE using FIRE optimizer.
Comparing the ASE and VV FIRE optimizers.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
# ]
# ///

import os
import time

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.optimizers import fire
from torch_sim.state import SimState


# Set device, data type and unit conversion
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
N_steps = 10 if os.getenv("CI") else 500

# Set random seed for reproducibility
rng = np.random.default_rng(seed=0)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((4, 4, 4))
si_dc.positions += 0.3 * rng.standard_normal(si_dc.positions.shape)

# Create FCC Copper
cu_dc = bulk("Cu", "fcc", a=3.85).repeat((5, 5, 5))
cu_dc.positions += 0.3 * rng.standard_normal(cu_dc.positions.shape)

# Create BCC Iron
fe_dc = bulk("Fe", "bcc", a=2.95).repeat((5, 5, 5))
fe_dc.positions += 0.3 * rng.standard_normal(fe_dc.positions.shape)

si_dc_vac = si_dc.copy()
si_dc_vac.positions += 0.3 * rng.standard_normal(si_dc_vac.positions.shape)
# select 2 numbers in range 0 to len(si_dc_vac)
indices = rng.choice(len(si_dc_vac), size=2, replace=False)
for idx in indices:
    si_dc_vac.pop(idx)


cu_dc_vac = cu_dc.copy()
cu_dc_vac.positions += 0.3 * rng.standard_normal(cu_dc_vac.positions.shape)
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
fe_dc_vac.positions += 0.3 * rng.standard_normal(fe_dc_vac.positions.shape)
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
atoms_list = [si_dc, cu_dc, fe_dc, si_dc_vac, cu_dc_vac]

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


def run_optimization(
    initial_state: SimState, md_flavor: str, force_tol: float = 0.05
) -> tuple[torch.Tensor, SimState]:
    """Runs FIRE optimization and returns convergence steps."""
    print(f"\n--- Running optimization with MD Flavor: {md_flavor} ---")
    start_time = time.perf_counter()

    # Re-initialize state and optimizer for this run
    init_fn, update_fn = fire(
        model=model,
        md_flavor=md_flavor,
    )
    fire_state = init_fn(initial_state.clone())  # Use a clone to start fresh

    batcher = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with="n_atoms",
        max_memory_scaler=1000,
        max_iterations=1000,  # Increased max iterations
        return_indices=True,  # Ensure indices are returned
    )

    batcher.load_states(fire_state)

    total_structures = fire_state.n_batches
    # Initialize convergence steps tensor (-1 means not converged yet)
    convergence_steps = torch.full(
        (total_structures,), -1, dtype=torch.long, device=device
    )
    convergence_fn = ts.generate_force_convergence_fn(force_tol=force_tol)

    converged_tensor_global = torch.zeros(
        total_structures, dtype=torch.bool, device=device
    )
    global_step = 0
    all_converged_states = []  # Initialize list to store completed states
    convergence_tensor_for_batcher = None  # Initialize convergence tensor for batcher

    # Keep track of the last valid state for final collection
    last_active_state = fire_state

    while True:  # Loop until batcher indicates completion
        # Get the next batch, passing the convergence status
        result = batcher.next_batch(last_active_state, convergence_tensor_for_batcher)

        fire_state, converged_states_from_batcher, current_indices_list = result
        all_converged_states.extend(
            converged_states_from_batcher
        )  # Add newly completed states

        if fire_state is None:  # No more active states
            print("All structures converged or batcher reached max iterations.")
            break

        last_active_state = fire_state  # Store the current active state

        # Get the original indices of the current active batch as a tensor
        current_indices = torch.tensor(
            current_indices_list, dtype=torch.long, device=device
        )

        # Optimize the current batch
        steps_this_round = 10
        for _ in range(steps_this_round):
            fire_state = update_fn(fire_state)
        global_step += steps_this_round  # Increment global step count

        # Check convergence *within the active batch*
        convergence_tensor_for_batcher = convergence_fn(fire_state, None)

        # Update global convergence status and steps
        # Identify structures in this batch that just converged
        newly_converged_mask_local = convergence_tensor_for_batcher & (
            convergence_steps[current_indices] == -1
        )
        converged_indices_global = current_indices[newly_converged_mask_local]

        if converged_indices_global.numel() > 0:
            # Mark convergence step
            convergence_steps[converged_indices_global] = global_step
            converged_tensor_global[converged_indices_global] = True
            converged_indices = converged_indices_global.tolist()

            total_converged = converged_tensor_global.sum().item() / total_structures
            print(f"{global_step=}: {converged_indices=}, {total_converged=:.2%}")

        # Optional: Print progress
        if global_step % 50 == 0:  # Reduced frequency
            total_converged = converged_tensor_global.sum().item() / total_structures
            active_structures = fire_state.n_batches if fire_state else 0
            print(f"{global_step=}: {active_structures=}, {total_converged=:.2%}")

    # After the loop, collect any remaining states that were active in the last batch
    # result[1] contains states completed *before* the last next_batch call.
    # We need the states that were active *in* the last batch returned by next_batch
    # If fire_state was the last active state, we might need to add it if batcher didn't
    # mark it complete. However, restore_original_order should handle all collected states
    # correctly.

    # Restore original order and concatenate
    final_states_list = batcher.restore_original_order(all_converged_states)
    final_state_concatenated = ts.concatenate_states(final_states_list)

    end_time = time.perf_counter()
    print(f"Finished {md_flavor} in {end_time - start_time:.2f} seconds.")
    # Return both convergence steps and the final state object
    return convergence_steps, final_state_concatenated


# --- Main Script ---
force_tol = 0.05

# Run with ase_fire
ase_steps, ase_final_state = run_optimization(
    state.clone(), "ase_fire", force_tol=force_tol
)
# Run with vv_fire
vv_steps, vv_final_state = run_optimization(state.clone(), "vv_fire", force_tol=force_tol)

print("\n--- Comparison ---")
print(f"{force_tol=:.2f} eV/Å")

# Calculate Mean Position Displacements
ase_final_states_list = ase_final_state.split()
vv_final_states_list = vv_final_state.split()
mean_displacements = []
for idx in range(len(ase_final_states_list)):
    ase_pos = ase_final_states_list[idx].positions
    vv_pos = vv_final_states_list[idx].positions
    displacement = torch.norm(ase_pos - vv_pos, dim=1)
    mean_disp = torch.mean(displacement).item()
    mean_displacements.append(mean_disp)


print(f"Initial energies: {[f'{e.item():.3f}' for e in initial_energies]} eV")
print(f"Final ASE energies: {[f'{e.item():.3f}' for e in ase_final_state.energy]} eV")
print(f"Final VV energies:  {[f'{e.item():.3f}' for e in vv_final_state.energy]} eV")
print(f"Mean Disp (ASE-VV): {[f'{d:.4f}' for d in mean_displacements]} Å")
print(f"Convergence steps (ASE FIRE): {ase_steps.tolist()}")
print(f"Convergence steps (VV FIRE):  {vv_steps.tolist()}")

# Identify structures that didn't converge
ase_not_converged = torch.where(ase_steps == -1)[0].tolist()
vv_not_converged = torch.where(vv_steps == -1)[0].tolist()

if ase_not_converged:
    print(f"ASE FIRE did not converge for indices: {ase_not_converged}")
if vv_not_converged:
    print(f"VV FIRE did not converge for indices: {vv_not_converged}")
