"""Structural optimization with MACE using FIRE optimizer."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.models.mace import UnbatchedMaceModel
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.unbatched_optimizers import fire


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

# Number of steps to run
N_steps = 10 if os.getenv("CI") else 500

PERIODIC = True

# Create diamond cubic Silicon with random displacements
rng = np.random.default_rng(seed=0)
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
si_dc.positions = si_dc.positions + 0.2 * rng.standard_normal(si_dc.positions.shape)

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

# Initialize the unbatched MACE model
model = UnbatchedMaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

# Run initial inference
results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)

state = {
    "positions": positions,
    "masses": masses,
    "cell": cell,
    "pbc": PERIODIC,
    "atomic_numbers": atomic_numbers,
}
# Initialize FIRE optimizer for structural relaxation
fire_init, fire_update = fire(model=model)
state = fire_init(state=state)

# Run optimization loop
for step in range(N_steps):
    if step % 10 == 0:
        print(f"{step=}: Total energy: {state.energy.item()} eV")
    state = fire_update(state)

print(f"Initial energy: {results['energy'].item()} eV")
print(f"Final energy: {state.energy.item()} eV")
print(f"Initial max force: {torch.max(torch.abs(results['forces'])).item()} eV/Å")
print(f"Final max force: {torch.max(torch.abs(state.forces)).item()} eV/Å")
