"""MACE NVT simulation with staggered stress calculation."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.quantities import calc_kT
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

state = ts.nvt_langevin_init(state=state, model=model, kT=kT)

stress = torch.zeros(N_steps // 10, 3, 3, device=device, dtype=dtype)
for step in range(N_steps):
    temp = (
        calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
        / Units.temperature
    )

    # Calculate kinetic energy: KE = 0.5 * sum(p^2 / m)
    kinetic_energy = 0.5 * torch.sum(
        torch.pow(state.momenta, 2) / state.masses.unsqueeze(-1)
    )
    # Total energy = kinetic + potential
    invariant = float(kinetic_energy + state.energy)

    print(f"{step=}: Temperature: {temp.item():.4f}: {invariant=:.4f}")
    state = ts.nvt_langevin_step(state=state, model=model, dt=torch.tensor(dt), kT=kT)
    if step % 10 == 0:
        results = model(state)
        stress[step // 10] = results["stress"]

print(f"Stress: {stress} eV/Å^3")
