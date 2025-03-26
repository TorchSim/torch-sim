"""NVT Langevin simulation with MACE."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///

import os

import torch
from ase.io import read
from mace.calculators.foundations_models import mace_mp

from torch_sim.io import atoms_to_state
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.quantities import temperature
from torch_sim.unbatched.models.mace import UnbatchedMaceModel
from torch_sim.unbatched.unbatched_integrators import nvt_langevin
from torch_sim.units import MetalUnits as Units


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Load the MACE model
loaded_model = mace_mp(
    model="medium",
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
    dispersion=True,
)

# Load the battery structure
battery = read("battery.xyz", format="extxyz")

# Number of steps to run
N_steps = 20 if os.getenv("CI") else 2_000

# Initialize the unbatched MACE model
model = UnbatchedMaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    compute_force=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

state = atoms_to_state(atoms=battery, device=device, dtype=dtype)

dt = 0.002 * Units.time  # Timestep (ps)
kT = 500 * Units.temperature  # Initial temperature (K)
gamma = 10 / Units.time  # Langevin friction coefficient (ps^-1)

# Initialize NVT Langevin integrator
langevin_init, langevin_update = nvt_langevin(
    model=model,
    kT=kT,
    dt=dt,
    gamma=gamma,
)

state = langevin_init(state=state, seed=1)

for step in range(N_steps):
    if step % 10 == 0:
        temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
        print(f"{step=}: Temperature: {temp:.4f}")
    state = langevin_update(state=state, kT=kT)

final_temp = temperature(masses=state.masses, momenta=state.momenta) / Units.temperature
print(f"Final temperature: {final_temp:.4f}")
