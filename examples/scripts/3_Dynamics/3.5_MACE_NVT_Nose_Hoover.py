"""NVT simulation with MACE and Nose-Hoover thermostat."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
# ]
# ///

import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.integrators.nvt import nvt_nose_hoover, nvt_nose_hoover_invariant
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.quantities import calc_kT
from torch_sim.units import MetalUnits as Units


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
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 20 if SMOKE_TEST else 2_000

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

# Initialize the MACE model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)
state = ts.SimState(
    positions=positions,
    masses=masses,
    cell=cell,
    pbc=True,
    atomic_numbers=atomic_numbers,
)
# Run initial inference
results = model(state)

dt = 0.002 * Units.time  # Timestep (ps)
kT = (
    torch.tensor(1000, device=device, dtype=dtype) * Units.temperature
)  # Initial temperature (K)


nvt_init, nvt_update = nvt_nose_hoover(model=model, kT=kT, dt=dt)
state = nvt_init(state=state, kT=kT, seed=1)

for step in range(N_steps):
    if step % 10 == 0:
        temp = (
            calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
            / Units.temperature
        )
        invariant = nvt_nose_hoover_invariant(state, kT=kT).item()
        print(f"{step=}: Temperature: {temp.item():.4f}: invariant: {invariant:.4f}")
    state = nvt_update(state=state, kT=kT)

final_temp = (
    calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f}")
