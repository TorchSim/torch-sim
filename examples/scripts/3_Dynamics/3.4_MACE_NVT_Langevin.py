"""MACE NVT Langevin dynamics."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///

import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.quantities import calc_kT
from torch_sim.state import SimState
from torch_sim.unbatched.models.mace import UnbatchedMaceModel
from torch_sim.unbatched.unbatched_integrators import nvt_langevin
from torch_sim.units import MetalUnits as Units


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
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
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

state = SimState(
    positions=positions,
    masses=masses,
    cell=cell,
    pbc=True,
    atomic_numbers=atomic_numbers,
)

dt = 0.002 * Units.time  # Timestep (ps)
kT = 1000 * Units.temperature  # Initial temperature (K)
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
        temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
        print(f"{step=}: Temperature: {temp:.4f}")
    state = langevin_update(state=state, kT=kT)

final_temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
print(f"Final temperature: {final_temp:.4f}")
