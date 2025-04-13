"""Example script demonstrating batched MACE model optimization with hot-swapping."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "matbench-discovery>=1.3.1",
# ]
# ///

import os
import time

import torch
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts


# --- Setup and Configuration ---
# Device and data type configuration
device = torch.device("cpu") if os.getenv("CI") else torch.device("cuda")
dtype = torch.float32
print(f"job will run on {device=}")

# --- Model Initialization ---
print("Loading MACE model...")
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
mace = mace_mp(model=mace_checkpoint_url, return_raw_model=True)
mace_model = ts.models.MaceModel(
    model=mace,
    device=device,
    dtype=dtype,
    compute_forces=True,
)

# Optimization parameters
fmax = 0.05  # Force convergence threshold
n_steps = 10 if os.getenv("CI") else 200_000_000
max_atoms_in_batch = 50 if os.getenv("CI") else 8_000

# --- Data Loading ---
if not os.getenv("CI"):
    n_structures_to_relax = 100
    print(f"Loading {n_structures_to_relax:,} structures...")
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip

    ase_atoms_list = ase_atoms_from_zip(
        DataFiles.wbm_initial_atoms.path, limit=n_structures_to_relax
    )
else:
    n_structures_to_relax = 2
    print(f"Loading {n_structures_to_relax:,} structures...")
    from ase.build import bulk

    al_atoms = bulk("Al", "hcp", a=4.05)
    fe_atoms = bulk("Fe", "bcc", a=2.86)
    ase_atoms_list = [al_atoms, fe_atoms]

# --- Optimization Setup ---
# Statistics tracking

# Initialize first batch
fire_init, fire_update = ts.optimizers.frechet_cell_fire(model=mace_model)
fire_states = fire_init(
    ts.io.atoms_to_state(atoms=ase_atoms_list, device=device, dtype=dtype)
)

batcher = ts.autobatching.HotSwappingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=1000 if os.getenv("CI") else None,
)
converge_max_force = ts.runners.generate_force_convergence_fn(force_tol=0.05)

start_time = time.perf_counter()

# --- Main Optimization Loop ---
batcher.load_states(fire_states)
all_completed_states, convergence_tensor, state = [], None, None
while (result := batcher.next_batch(state, convergence_tensor))[0] is not None:
    state, completed_states = result
    print(f"Starting new batch of {state.n_batches} states.")

    all_completed_states.extend(completed_states)
    print("Total number of completed states", len(all_completed_states))

    for _step in range(10):
        state = fire_update(state)
    convergence_tensor = converge_max_force(state, last_energy=None)
all_completed_states.extend(result[1])
print("Total number of completed states", len(all_completed_states))

# --- Final Statistics ---
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
