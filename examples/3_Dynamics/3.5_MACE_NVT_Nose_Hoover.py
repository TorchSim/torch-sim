"""NVT simulation with MACE and Nose-Hoover thermostat."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///

import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torchsim.models.mace import UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.quantities import temperature
from torchsim.unbatched_integrators import nvt_nose_hoover, nvt_nose_hoover_invariant
from torchsim.units import MetalUnits as Units


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
N_steps = 20 if os.getenv("CI") else 2_000

PERIODIC = True

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

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

dt = 0.002 * Units.time  # Timestep (ps)
kT = 1000 * Units.temperature  # Initial temperature (K)

start = {
    "positions": positions,
    "masses": masses,
    "cell": cell,
    "pbc": PERIODIC,
    "atomic_numbers": atomic_numbers,
}

nvt_init, nvt_update = nvt_nose_hoover(model=model, kT=kT, dt=dt)
state = nvt_init(start, kT=kT, seed=1)

for step in range(N_steps):
    if step % 10 == 0:
        temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
        invariant = nvt_nose_hoover_invariant(state, kT=kT).item()
        print(f"{step=}: Temperature: {temp.item():.4f}: invariant: {invariant:.4f}")
    state = nvt_update(state, kT=kT)

final_temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
print(f"Final temperature: {final_temp}")
