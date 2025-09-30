"""MACE NVT simulation with staggered stress calculation."""

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
from torch_sim.integrators import nvt_langevin
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

# Initialize the MACE model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

state = ts.io.atoms_to_state(si_dc, device=device, dtype=dtype)

dt = 0.002 * Units.time  # Timestep (ps)
kT = (
    torch.tensor(1000, device=device, dtype=dtype) * Units.temperature
)  # Initial temperature (K)

nvt_init, nvt_update = nvt_langevin(model=model, kT=kT, dt=dt)
state = nvt_init(state, kT=kT, seed=1)

stress = torch.zeros(N_steps // 10, 3, 3, device=device, dtype=dtype)
for step in range(N_steps):
    temp = (
        calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
        / Units.temperature
    )

    # Calculate kinetic energy: KE = 0.5 * sum(p^2 / m)
    kinetic_energy = 0.5 * torch.sum(state.momenta**2 / state.masses.unsqueeze(-1))
    # Total energy = kinetic + potential
    invariant = float(kinetic_energy + state.energy)

    print(f"{step=}: Temperature: {temp.item():.4f}: {invariant=:.4f}")
    state = nvt_update(state, kT=kT)
    if step % 10 == 0:
        results = model(state)
        stress[step // 10] = results["stress"]

print(f"Stress: {stress} eV/Å^3")
