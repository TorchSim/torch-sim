"""NVE simulation with MACE and cuEquivariance enabled."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
import os
import time

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.integrators import nve_init, nve_step
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.units import MetalUnits as Units


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).lstrip("torch."),
    device=str(device),
)

# Option 2: Load from local file (comment out Option 1 to use this)
# loaded_model = torch.load("path/to/model.pt", map_location=device)

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 20 if SMOKE_TEST else 2_000

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Initialize the MACE model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=torch.cuda.is_available(),
)
state = ts.io.atoms_to_state(si_dc, device=device, dtype=dtype)

# Run initial inference
results = model(state)

# Setup NVE MD simulation parameters
kT = torch.tensor(1000, device=device, dtype=dtype) * Units.temperature
dt = torch.tensor(0.002 * Units.time, device=device, dtype=dtype)  # Timestep (ps)


# Initialize NVE integrator
state = nve_init(model=model, state=state, kT=kT, seed=1)

# Run MD simulation
print("\nStarting NVE molecular dynamics simulation...")
start_time = time.perf_counter()
for step in range(N_steps):
    total_energy = state.energy + ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )
    if step % 10 == 0:
        print(f"Step {step}: Total energy: {total_energy.item():.4f} eV")
    state = nve_step(model=model, state=state, dt=dt)
end_time = time.perf_counter()

# Report simulation results
print("\nSimulation complete!")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Average time per step: {(end_time - start_time) / N_steps:.4f} seconds")
print(f"Final total energy: {total_energy.item()} eV")
