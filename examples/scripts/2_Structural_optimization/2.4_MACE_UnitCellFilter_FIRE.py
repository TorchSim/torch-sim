"""Structural optimization with MACE using FIRE optimizer."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceUrls
from torch_sim.unbatched.models.mace import UnbatchedMaceModel
from torch_sim.unbatched.unbatched_optimizers import unit_cell_fire
from torch_sim.units import UnitConversion


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

# Number of steps to run
N_steps = 10 if os.getenv("CI") else 500

# Create diamond cubic Silicon with random displacements and a 5% volume compression
rng = np.random.default_rng(seed=0)
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
si_dc.positions = si_dc.positions + 0.2 * rng.standard_normal(si_dc.positions.shape)
si_dc.cell = si_dc.cell.array * 0.95

state = ts.io.atoms_to_state([si_dc], device, dtype)

# Initialize the unbatched MACE model
model = UnbatchedMaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Run initial inference
results = model(state)

# Initialize FIRE optimizer for structural relaxation
fire_init, fire_update = unit_cell_fire(model=model)
state = fire_init(state=state)

# Run optimization loop
for step in range(N_steps):
    if step % 10 == 0:
        e_pot = state.energy.item()
        pressure = (
            -torch.trace(state.stress).item() / 3.0 * UnitConversion.eV_per_Ang3_to_GPa
        )
        print(f"{step=}: Total energy: {e_pot} eV, {pressure=:.4f} GPa")
    state = fire_update(state)

print(f"Initial energy: {results['energy'].item()} eV")
print(f"Final energy: {state.energy.item()} eV")


print(f"Initial max force: {torch.max(torch.abs(results['forces'])).item()} eV/Å")
print(f"Final max force: {torch.max(torch.abs(state.forces)).item()} eV/Å")

initial_pressure = (
    -torch.trace(results["stress"]).item() / 3.0 * UnitConversion.eV_per_Ang3_to_GPa
)
print(f"{initial_pressure=:.4f} GPa")
final_pressure = (
    torch.trace(state.stress).item() / 3.0 * UnitConversion.eV_per_Ang3_to_GPa
)
print(f"{final_pressure=:.4f} GPa")
