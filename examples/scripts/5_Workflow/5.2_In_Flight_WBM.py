"""Example script demonstrating batched MACE model optimization with hot-swapping."""

# /// script
# dependencies = ["mace-torch>=0.3.10", "matbench-discovery>=1.3.1"]
# ///
import os
import time

import numpy as np
import torch
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls


# --- Setup and Configuration ---
# Device and data type configuration
SMOKE_TEST = os.getenv("CI") is not None
device = torch.device("cpu" if SMOKE_TEST else "cuda")
dtype = torch.float32
print(f"job will run on {device=}")

# --- Model Initialization ---
print("Loading MACE model...")
mace = mace_mp(model=MaceUrls.mace_mpa_medium, return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=dtype,
    compute_forces=True,
)

# Optimization parameters
fmax = 0.05  # Force convergence threshold
n_steps = 10 if SMOKE_TEST else 200_000_000
max_atoms_in_batch = 50 if SMOKE_TEST else 8_000

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
    al_atoms.positions += 0.1 * np.random.randn(*al_atoms.positions.shape)  # noqa: NPY002
    fe_atoms = bulk("Fe", "bcc", a=2.86).repeat((2, 2, 2))
    fe_atoms.positions += 0.1 * np.random.randn(*fe_atoms.positions.shape)  # noqa: NPY002
    ase_atoms_list = [al_atoms, fe_atoms]

# --- Optimization Setup ---
# Statistics tracking

# Initialize first batch
fire_states = ts.fire_init(
    model=mace_model,
    state=ts.io.atoms_to_state(atoms=ase_atoms_list, device=device, dtype=dtype),
    cell_filter=ts.CellFilter.frechet,
)

batcher = ts.autobatching.InFlightAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=1000 if SMOKE_TEST else None,
)
converge_max_force = ts.runners.generate_force_convergence_fn(force_tol=0.05)

start_time = time.perf_counter()

# --- Main Optimization Loop ---
batcher.load_states(fire_states)
all_completed_states, convergence_tensor, state = [], None, None
while (result := batcher.next_batch(state, convergence_tensor))[0] is not None:
    state, completed_states = result
    print(f"Starting new batch of {state.n_systems} states.")

    all_completed_states.extend(completed_states)
    print(f"Total number of completed states {len(all_completed_states)}")

    for _step in range(10):
        state = ts.fire_step(model=mace_model, state=state)
    convergence_tensor = converge_max_force(state, last_energy=None)
all_completed_states.extend(result[1])
print(f"Total number of completed states {len(all_completed_states)}")

# --- Final Statistics ---
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
