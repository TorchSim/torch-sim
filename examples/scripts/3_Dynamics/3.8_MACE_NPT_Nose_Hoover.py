"""NPT simulation with MACE and Nose-Hoover thermostat."""

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
from torch_sim.integrators import npt_nose_hoover, npt_nose_hoover_invariant
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.quantities import calc_kinetic_energy, calc_kT, get_pressure
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

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

# Print shapes for verification
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")

# Initialize the MACE model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)
state = ts.SimState(
    positions=positions,
    masses=masses,
    cell=cell,
    atomic_numbers=atomic_numbers,
    pbc=True,
)
# Run initial inference
results = model(state)

N_steps_nvt = 20 if os.getenv("CI") else 2_000
N_steps_npt = 20 if os.getenv("CI") else 2_000
dt = 0.001 * Units.time  # Time step (1 ps)
kT = 300 * Units.temperature  # Initial temperature (300 K)
target_pressure = 0.0 * Units.pressure  # Target pressure (0 bar)

nvt_init, nvt_update = npt_nose_hoover(model=model, kT=kT, dt=dt)
state = nvt_init(state=state, seed=1)

for step in range(N_steps_nvt):
    if step % 10 == 0:
        temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        ).item()
        print(f"{step=}: Temperature: {temp:.4f}: invariant: {invariant:.4f}, ")
    state = nvt_update(state, kT=kT)

npt_init, npt_update = npt_nose_hoover(
    model=model, kT=kT, dt=dt, external_pressure=target_pressure
)
state = npt_init(state=state, seed=1)

for step in range(N_steps_npt):
    if step % 10 == 0:
        temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        ).item()
        stress = model(state)["stress"]
        volume = torch.det(state.current_cell)
        pressure = get_pressure(
            stress,
            calc_kinetic_energy(masses=state.masses, momenta=state.momenta),
            volume,
        ).item()
        xx, yy, zz = torch.diag(state.current_cell)
        print(
            f"{step=}: Temperature: {temp:.4f}: invariant: {invariant:.4f}, "
            f"pressure: {pressure:.4f}, "
            f"cell xx yy zz: {xx.item():.4f}, {yy.item():.4f}, {zz.item():.4f}"
        )
    state = npt_update(state, kT=kT, external_pressure=target_pressure)

final_temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
print(f"Final temperature: {final_temp:.4f}")
final_stress = model(state)["stress"]
final_volume = torch.det(state.current_cell)
final_pressure = get_pressure(
    final_stress,
    calc_kinetic_energy(masses=state.masses, momenta=state.momenta),
    final_volume,
)
print(f"Final pressure: {final_pressure:.4f}")
