"""MACE NVT Langevin dynamics."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.units import MetalUnits as Units


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# Option 2: Load from local file (comment out Option 1 to use this)
# loaded_model = torch.load("path/to/model.pt", map_location=device)

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
    positions=positions, masses=masses, cell=cell, atomic_numbers=atomic_numbers, pbc=True
)

dt = torch.tensor(0.002 * Units.time, device=device, dtype=dtype)  # Timestep (ps)
kT = torch.tensor(1000, device=device, dtype=dtype) * Units.temperature
gamma = torch.tensor(
    10 / Units.time, device=device, dtype=dtype
)  # Langevin friction coefficient (ps^-1)

# Initialize NVT Langevin integrator
state = nvt_langevin_init(model=model, state=state, kT=kT, seed=1)

for step in range(N_steps):
    if step % 10 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        print(f"{step=}: Temperature: {temp.item():.4f}")
    state = nvt_langevin_step(state=state, model=model, dt=dt, kT=kT, gamma=gamma)

final_temp = (
    ts.calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f}")
