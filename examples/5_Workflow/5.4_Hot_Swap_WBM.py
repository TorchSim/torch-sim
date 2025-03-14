"""Example script demonstrating batched MACE model optimization with hot-swapping."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "matbench-discovery>=1.3.1",
# ]
# ///

import os
import time
from pathlib import Path

import torch
from mace.calculators.foundations_models import mace_mp
from matbench_discovery.data import DataFiles, ase_atoms_from_zip

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import unit_cell_fire
from torchsim.runners import atoms_to_state, state_to_atoms
from torchsim.units import UnitConversion
from torchsim.workflows.batching_utils import (
    calculate_force_convergence_mask,
    check_max_atoms_in_batch,
    swap_structure,
    write_log_line,
)


# --- Setup and Configuration ---
# Device and data type configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"job will run on {device=}")

# --- Model Initialization ---
PERIODIC = True
print("Loading MACE model...")
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

print("Initializing MACE model...")
batched_model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)
batched_model.eval()

# Optimization parameters
batch_size = 2 if os.getenv("CI") else 16
fmax = 0.05  # Force convergence threshold
n_steps = 10 if os.getenv("CI") else 200_000_000
log_path = Path(f"WBM_relaxation_log_{batch_size}.txt")
max_atoms_in_batch = 50 if os.getenv("CI") else 8_000
log_frequency = 500

# Performance optimizations
if device == "cuda":
    torch.set_num_threads(batch_size)  # Optimize CPU thread count for GPU usage
    # torch.cuda.set_device(0)  # Use primary GPU if multiple available

# Ensure log directory exists
log_path.parent.mkdir(parents=True, exist_ok=True)

# --- Data Loading ---
n_structures_to_relax = 2 if os.getenv("CI") else 100
print(f"Loading {n_structures_to_relax:,} structures...")
ase_init_atoms = ase_atoms_from_zip(
    DataFiles.wbm_initial_atoms.path, limit=n_structures_to_relax
)

# --- Optimization Setup ---
# Statistics tracking
completed_structures = 0
processed_atoms = 0

# Initialize first batch
current_idx = batch_size
struct_list = ase_init_atoms[:batch_size].copy()
fire_init, fire_update = unit_cell_fire(model=batched_model)
batch_state = fire_init(atoms_to_state(struct_list, device=device, dtype=dtype))

print(
    f"Starting optimization of {n_structures_to_relax} structures with {batch_size=}..."
)
start_time = time.perf_counter()

# --- Main Optimization Loop ---
for step in range(n_steps):
    # Calculate convergence with no gradient tracking
    with torch.no_grad():
        force_norms, force_mask = calculate_force_convergence_mask(
            forces=batch_state.forces,
            batch=batch_state.batch,
            batch_size=batch_size,
            fmax=fmax,
        )
        # Get indices of converged structures
        converged_indices = torch.where(force_mask)[0].cpu().tolist()
        # Check if we have converged structures that need replacement
        any_replaced = False

        # Cache tensors for logging
        energies = batch_state.energy
        energies_cpu = energies.cpu()
        force_norms_cpu = force_norms.cpu()

    # Periodic logging of progress
    if step % log_frequency == 0:
        with torch.no_grad():
            current_time = time.perf_counter()
            elapsed = current_time - start_time
            structures_per_sec = completed_structures / elapsed if elapsed > 0 else 0
            atoms_per_sec = processed_atoms / elapsed if elapsed > 0 else 0

            # Calculate properties for logging
            energies = batch_state.energy
            pressures = (
                torch.tensor(
                    [torch.trace(stress) / 3.0 for stress in batch_state.stress],
                    device=device,
                )
                * UnitConversion.eV_per_Ang3_to_GPa
            )

            # Convert to Python lists for logging
            energy_list = energies.cpu().tolist()
            pressure_list = pressures.cpu().tolist()
            force_list = force_norms.cpu().tolist()

        # Write detailed log to file
        with open(log_path, mode="a") as file:
            props = {
                "energy": energy_list,
                "pressure": pressure_list,
                "force": force_list,
            }
            write_log_line(
                file=file,
                step=step,
                properties=props,
                converged=force_mask.cpu().tolist(),
                batch_idx=list(range(current_idx - batch_size, current_idx)),
            )

    # Process converged structures
    for idx in sorted(converged_indices, reverse=True):
        if current_idx < n_structures_to_relax:
            # Try to replace with next structure
            next_atoms = ase_init_atoms[current_idx]
            if check_max_atoms_in_batch(
                current_struct=struct_list[idx],
                next_struct=next_atoms,
                struct_list=struct_list,
                max_atoms=max_atoms_in_batch,
            ):
                # Update statistics and log completion
                completed_structures += 1
                processed_atoms += len(struct_list[idx])

                # Replace structure and update state
                struct_list = state_to_atoms(batch_state)
                batch_state, current_idx = swap_structure(
                    idx=idx,
                    current_idx=current_idx,
                    struct_list=struct_list,
                    all_struct_list=ase_init_atoms,
                    device=device,
                    dtype=dtype,
                    optimizer_init=fire_init,
                )
                torch.cuda.empty_cache()
                any_replaced = True
                print(
                    f"Swapped structure "
                    f"{current_idx - batch_size + idx} with structure {current_idx}."
                )
            else:
                # Handle final batch case
                print(
                    f"Final structure {idx} converged. "
                    f"Reducing batch size to {batch_size - 1}."
                )
                completed_structures += 1
                processed_atoms += len(struct_list[idx])

                # Remove the converged structure
                struct_list.pop(idx)
                batch_size -= 1

                if batch_size > 0:
                    # Reinitialize with smaller batch
                    struct_list = state_to_atoms(batch_state)
                    batch_state = fire_init(
                        atoms_to_state(struct_list, device=device, dtype=dtype)
                    )
                    torch.cuda.empty_cache()

                    # Filter force_norms correctly - remove converged entries
                    force_norms = force_norms[~force_mask]
                    force_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                else:
                    # All structures are converged, exit the loop
                    print("All structures converged!")
                    break

                any_replaced = True
        else:
            completed_structures += 1
            processed_atoms += len(struct_list[idx])

            # Remove the converged structure
            struct_list.pop(idx)
            batch_size -= 1

            if batch_size > 0:
                # Reinitialize with smaller batch
                struct_list = state_to_atoms(batch_state)
                batch_state = fire_init(
                    atoms_to_state(struct_list, device=device, dtype=dtype)
                )
                torch.cuda.empty_cache()

                # Filter force_norms correctly - remove converged entries
                mask = torch.ones_like(force_norms, dtype=torch.bool)
                mask[idx] = False
                force_norms = force_norms[mask]
                force_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                # All structures are converged, exit the loop
                print("All structures converged!")
                break

            any_replaced = True

    # Skip update if any structures were replaced
    if any_replaced:
        continue

    # Check if all structures are done
    if batch_size == 0:
        break

    # Update optimization state
    batch_state = fire_update(batch_state)

# --- Final Statistics ---
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Completed {completed_structures}/{n_structures_to_relax} structures")
print(
    f"Average time per structure: {total_time / max(1, completed_structures):.2f} seconds"
)
print(
    f"Average throughput: {completed_structures / total_time:.2f} structures/s, "
    f"{processed_atoms / total_time:.2f} atoms/s"
)
