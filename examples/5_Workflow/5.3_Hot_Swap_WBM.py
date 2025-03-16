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
from matbench_discovery.data import DataFiles, ase_atoms_from_zip

from torch_sim.autobatching import HotSwappingAutoBatcher
from torch_sim.io import atoms_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.state import BaseState


# --- Setup and Configuration ---
# Device and data type configuration
device = torch.device("cuda")
dtype = torch.float64
print(f"job will run on {device=}")

# --- Model Initialization ---
PERIODIC = True
print("Loading MACE model...")
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_force=True,
)

# Optimization parameters
fmax = 0.05  # Force convergence threshold
n_steps = 10 if os.getenv("CI") else 200_000_000
max_atoms_in_batch = 50 if os.getenv("CI") else 8_000

# --- Data Loading ---
n_structures_to_relax = 2 if os.getenv("CI") else 100
print(f"Loading {n_structures_to_relax:,} structures...")
ase_atoms_list = ase_atoms_from_zip(
    DataFiles.wbm_initial_atoms.path, limit=n_structures_to_relax
)


def convergence_fn(state: BaseState, fmax: float = 0.05) -> bool:
    """Check if the system has converged."""
    batch_wise_max_force = torch.zeros(state.n_batches, device=state.device)
    max_forces = state.forces.norm(dim=1)
    batch_wise_max_force = batch_wise_max_force.scatter_reduce(
        dim=0,
        index=state.batch,
        src=max_forces,
        reduce="amax",
    )
    return batch_wise_max_force < fmax


# --- Optimization Setup ---
# Statistics tracking

# Initialize first batch
fire_init, fire_update = unit_cell_fire(model=mace_model)
fire_states = fire_init(atoms_to_state(ase_atoms_list, device=device, dtype=dtype))

batcher = HotSwappingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=None,
)
batcher.load_states(fire_states)
start_time = time.perf_counter()

# --- Main Optimization Loop ---
all_completed_states, convergence_tensor = [], None
state = None
total_steps = 0
while True:
    if state is not None:
        print(f"Starting new batch of {state.n_batches} states.")

    state, completed_states = batcher.next_batch(state, convergence_tensor)
    print("Number of completed states", len(completed_states))

    all_completed_states.extend(completed_states)
    if state is None:
        break

    # run 10 steps, arbitrary number
    for _ in range(10):
        state = fire_update(state)
        total_steps += 1

    convergence_tensor = convergence_fn(state, fmax)

    if total_steps == n_steps:
        break

# --- Final Statistics ---
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
