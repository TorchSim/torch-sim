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
from torch_sim.integrators.npt import npt_nose_hoover, npt_nose_hoover_invariant
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

# Run initial inference
results = model(state)

SMOKE_TEST = os.getenv("CI") is not None
N_steps_nvt = 20 if SMOKE_TEST else 2_000
N_steps_npt = 20 if SMOKE_TEST else 2_000
dt = 0.001 * Units.time  # Time step (1 ps)
kT = (
    torch.tensor(300, device=device, dtype=dtype) * Units.temperature
)  # Initial temperature (300 K)
target_pressure = 0.0 * Units.pressure  # Target pressure (0 bar)

npt_init, npt_update = npt_nose_hoover(model=model, kT=kT, dt=dt)
state = npt_init(state=state, seed=1)

for step in range(N_steps_nvt):
    if step % 10 == 0:
        temp = (
            calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
            / Units.temperature
        )
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        ).item()
        print(f"{step=}: Temperature: {temp.item():.4f}: invariant: {invariant:.4f}, ")
    state = npt_update(state, kT=kT)

npt_init, npt_update = npt_nose_hoover(
    model=model, kT=kT, dt=dt, external_pressure=target_pressure
)
state = npt_init(state=state, seed=1)

for step in range(N_steps_npt):
    if step % 10 == 0:
        temp = (
            calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
            / Units.temperature
        )
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        ).item()
        stress = model(state)["stress"]
        volume = torch.det(state.current_cell)
        pressure = get_pressure(
            stress,
            calc_kinetic_energy(
                masses=state.masses, momenta=state.momenta, batch=state.batch
            ),
            volume,
        ).item()
        xx, yy, zz = torch.diag(state.current_cell)
        print(
            f"{step=}: Temperature: {temp.item():.4f}: invariant: {invariant:.4f}, "
            f"pressure: {pressure:.4f}, "
            f"cell xx yy zz: {xx.item():.4f}, {yy.item():.4f}, {zz.item():.4f}"
        )
    state = npt_update(state, kT=kT, external_pressure=target_pressure)

final_temp = (
    calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f}")
final_stress = model(state)["stress"]
final_volume = torch.det(state.current_cell)
final_pressure = get_pressure(
    final_stress,
    calc_kinetic_energy(masses=state.masses, momenta=state.momenta, batch=state.batch),
    final_volume,
)
print(f"Final pressure: {final_pressure:.4f}")
