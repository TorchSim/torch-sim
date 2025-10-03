"""NPT simulation with MACE and Nose-Hoover thermostat."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
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
dt = torch.tensor(0.001 * Units.time, device=device, dtype=dtype)  # Time step (1 ps)
kT = (
    torch.tensor(300, device=device, dtype=dtype) * Units.temperature
)  # Initial temperature (300 K)
target_pressure = torch.tensor(
    10_000 * Units.pressure, device=device, dtype=dtype
)  # Target pressure (0 bar)

state = ts.nvt_nose_hoover_init(model=model, state=state, kT=kT, dt=dt, seed=1)

for step in range(N_steps_nvt):
    if step % 10 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        invariant = float(ts.nvt_nose_hoover_invariant(state, kT=kT))
        print(f"{step=}: Temperature: {temp.item():.4f}: {invariant=:.4f}, ")
    state = ts.nvt_nose_hoover_step(model=model, state=state, dt=dt, kT=kT)

state = ts.npt_langevin_init(model=model, state=state, kT=kT, dt=dt, seed=1)

for step in range(N_steps_npt):
    if step % 10 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        stress = model(state)["stress"]
        volume = torch.det(state.cell)
        pressure = (
            ts.get_pressure(
                stress,
                ts.calc_kinetic_energy(
                    masses=state.masses,
                    momenta=state.momenta,
                    system_idx=state.system_idx,
                ),
                volume,
            ).item()
            / Units.pressure
        )
        xx, yy, zz = torch.diag(state.cell[0])
        print(
            f"{step=}: Temperature: {temp.item():.4f}, "
            f"pressure: {pressure:.4f}, "
            f"cell xx yy zz: {xx.item():.4f}, {yy.item():.4f}, {zz.item():.4f}"
        )
    state = ts.npt_langevin_step(
        model=model,
        state=state,
        dt=dt,
        kT=kT,
        external_pressure=target_pressure,
        alpha=1.0 / (100 * dt),
        cell_alpha=1.0 / (100 * dt),
        b_tau=1 / (1000 * dt),
    )

final_temp = (
    ts.calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f} K")
final_stress = model(state)["stress"]
final_volume = torch.det(state.cell)
final_pressure = (
    ts.get_pressure(
        final_stress,
        ts.calc_kinetic_energy(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        ),
        final_volume,
    ).item()
    / Units.pressure
)
print(f"Final pressure: {final_pressure:.4f} bar")
