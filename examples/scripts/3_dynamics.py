"""Molecular Dynamics Examples - Various ensembles and integrators.

This script demonstrates molecular dynamics simulations with:
- NVE (microcanonical) ensemble
- NVT (canonical) ensemble with Langevin and Nose-Hoover thermostats
- NPT (isothermal-isobaric) ensemble with Nose-Hoover barostat
- Both Lennard-Jones and MACE models
"""

# /// script
# dependencies = ["scipy>=1.15", "mace-torch>=0.3.12"]
# ///

import itertools
import os
import time

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.units import MetalUnits as Units


# Set up the device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 100 if SMOKE_TEST else 2_000

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)


# ============================================================================
# SECTION 1: Lennard-Jones NVE (Microcanonical Ensemble)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Lennard-Jones NVE Simulation")
print("=" * 70)

# Create face-centered cubic (FCC) Argon
a_len = 5.26  # Lattice constant

# Generate base FCC unit cell positions
base_positions = torch.tensor(
    [
        [0.0, 0.0, 0.0],  # Corner
        [0.0, 0.5, 0.5],  # Face centers
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    device=device,
    dtype=dtype,
)

# Create 4x4x4 supercell of FCC Argon
positions = []
for i, j, k in itertools.product(range(4), range(4), range(4)):
    for base_pos in base_positions:
        pos = base_pos + torch.tensor([i, j, k], device=device, dtype=dtype)
        positions.append(pos)

positions = torch.stack(positions) * a_len

# Create the cell tensor
cell = torch.tensor(
    [[4 * a_len, 0, 0], [0, 4 * a_len, 0], [0, 0, 4 * a_len]],
    device=device,
    dtype=dtype,
)

# Create the atomic numbers tensor (Argon = 18)
atomic_numbers = torch.full((positions.shape[0],), 18, device=device, dtype=torch.int)
# Create the masses tensor (Argon = 39.948 amu)
masses = torch.full((positions.shape[0],), 39.948, device=device, dtype=dtype)

state = ts.SimState(
    positions=positions, masses=masses, cell=cell, atomic_numbers=atomic_numbers, pbc=True
)

# Initialize the Lennard-Jones model
lj_model = LennardJonesModel(
    use_neighbor_list=False,
    sigma=3.405,
    epsilon=0.0104,
    cutoff=2.5 * 3.405,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
)

# Run initial simulation
results = lj_model(state)

# Set up NVE simulation
kT = torch.tensor(80 * Units.temperature, device=device, dtype=dtype)
dt = torch.tensor(0.001 * Units.time, device=device, dtype=dtype)

# Initialize NVE integrator
state = ts.nve_init(state=state, model=lj_model, kT=kT, seed=1)

# Run NVE simulation
for step in range(N_steps):
    if step % 100 == 0:
        # Calculate total energy (potential + kinetic)
        total_energy = state.energy + ts.calc_kinetic_energy(
            masses=state.masses, momenta=state.momenta
        )
        print(f"Step {step}: Total energy: {total_energy.item():.4f} eV")

    # Update state using NVE integrator
    state = ts.nve_step(state=state, model=lj_model, dt=dt)

final_total_energy = state.energy + ts.calc_kinetic_energy(
    masses=state.masses, momenta=state.momenta
)
print(f"Final total energy: {final_total_energy.item():.4f} eV")


# ============================================================================
# SECTION 2: MACE NVE Simulation
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: MACE NVE Simulation")
print("=" * 70)

# Load MACE model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

# Initialize the MACE model
mace_model = MaceModel(
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

# Run initial inference
results = mace_model(state)

# Setup NVE MD simulation parameters
kT = torch.tensor(1000 * Units.temperature, device=device, dtype=dtype)  # 1000 K
dt = torch.tensor(0.002 * Units.time, device=device, dtype=dtype)  # 2 fs

# Initialize NVE integrator
state = ts.nve_init(state=state, model=mace_model, kT=kT, seed=1)

# Run MD simulation
print("\nStarting NVE molecular dynamics simulation...")
start_time = time.perf_counter()
for step in range(N_steps):
    total_energy = state.energy + ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )
    if step % 100 == 0:
        print(f"Step {step}: Total energy: {total_energy.item():.4f} eV")
    state = ts.nve_step(state=state, model=mace_model, dt=dt)
end_time = time.perf_counter()

print("\nSimulation complete!")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Average time per step: {(end_time - start_time) / N_steps:.4f} seconds")


# ============================================================================
# SECTION 3: MACE NVT Langevin Simulation
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MACE NVT Langevin Simulation")
print("=" * 70)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Prepare input tensors
positions = torch.tensor(si_dc.positions, device=device, dtype=dtype)
cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
atomic_numbers = torch.tensor(si_dc.get_atomic_numbers(), device=device, dtype=torch.int)
masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

state = ts.SimState(
    positions=positions, masses=masses, cell=cell, atomic_numbers=atomic_numbers, pbc=True
)

dt = torch.tensor(0.002 * Units.time, device=device, dtype=dtype)  # 2 fs
kT = torch.tensor(1000 * Units.temperature, device=device, dtype=dtype)  # 1000 K
gamma = torch.tensor(10 / Units.time, device=device, dtype=dtype)  # ps^-1

# Initialize NVT Langevin integrator
state = ts.nvt_langevin_init(model=mace_model, state=state, kT=kT, seed=1)

print("\nStarting NVT Langevin simulation...")
for step in range(N_steps):
    if step % 100 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        print(f"Step {step}: Temperature: {temp.item():.4f} K")
    state = ts.nvt_langevin_step(state=state, model=mace_model, dt=dt, kT=kT, gamma=gamma)

final_temp = (
    ts.calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f} K")


# ============================================================================
# SECTION 4: MACE NVT Nose-Hoover Simulation
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: MACE NVT Nose-Hoover Simulation")
print("=" * 70)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

state = ts.io.atoms_to_state(si_dc, device=device, dtype=dtype)

# Run initial inference
results = mace_model(state)

dt = torch.tensor(0.002 * Units.time, device=device, dtype=dtype)  # 2 fs
kT = torch.tensor(1000 * Units.temperature, device=device, dtype=dtype)  # 1000 K

state = ts.nvt_nose_hoover_init(state=state, model=mace_model, kT=kT, dt=dt)

print("\nStarting NVT Nose-Hoover simulation...")
for step in range(N_steps):
    if step % 100 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        invariant = float(ts.nvt_nose_hoover_invariant(state, kT=kT))
        print(
            f"Step {step}: Temperature: {temp.item():.4f} K, Invariant: {invariant:.4f}"
        )
    state = ts.nvt_nose_hoover_step(state=state, model=mace_model, dt=dt, kT=kT)

final_temp = (
    ts.calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f} K")


# ============================================================================
# SECTION 5: MACE NPT Nose-Hoover Simulation
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: MACE NPT Nose-Hoover Simulation")
print("=" * 70)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))

# Create model with stress computation enabled for NPT
mace_model_stress = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

state = ts.io.atoms_to_state(si_dc, device=device, dtype=dtype)

# Run initial inference
results = mace_model_stress(state)

N_steps_nvt = 100 if SMOKE_TEST else 1_000
N_steps_npt = 100 if SMOKE_TEST else 1_000
dt = 0.001 * Units.time  # 1 fs
kT = torch.tensor(300 * Units.temperature, device=device, dtype=dtype)  # 300 K
target_pressure = torch.tensor(0.0 * Units.pressure, device=device, dtype=dtype)  # 0 bar

# Initialize NPT with NVT equilibration
state = ts.npt_nose_hoover_init(
    state=state, model=mace_model_stress, kT=kT, dt=torch.tensor(dt)
)

print("\nRunning NVT equilibration phase...")
for step in range(N_steps_nvt):
    if step % 100 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        invariant = float(
            ts.npt_nose_hoover_invariant(state, kT=kT, external_pressure=target_pressure)
        )
        print(
            f"Step {step}: Temperature: {temp.item():.4f} K, Invariant: {invariant:.4f}"
        )
    state = ts.npt_nose_hoover_step(
        state=state,
        model=mace_model_stress,
        dt=torch.tensor(dt),
        kT=kT,
        external_pressure=target_pressure,
    )

# Reinitialize for NPT phase
state = ts.npt_nose_hoover_init(
    state=state, model=mace_model_stress, kT=kT, dt=torch.tensor(dt)
)

print("\nRunning NPT simulation...")
for step in range(N_steps_npt):
    if step % 100 == 0:
        temp = (
            ts.calc_kT(
                masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
            )
            / Units.temperature
        )
        invariant = float(
            ts.npt_nose_hoover_invariant(state, kT=kT, external_pressure=target_pressure)
        )
        stress = mace_model_stress(state)["stress"]
        volume = torch.det(state.current_cell)
        e_kin = ts.calc_kinetic_energy(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        pressure = float(ts.get_pressure(stress, e_kin, volume))
        xx, yy, zz = torch.diag(state.current_cell[0])
        print(
            f"Step {step}: Temperature: {temp.item():.4f} K, Invariant: {invariant:.4f}, "
            f"Pressure: {pressure:.4f} eV/Å³, Cell: [{xx.item():.4f}, {yy.item():.4f}, {zz.item():.4f}]"
        )
    state = ts.npt_nose_hoover_step(
        state=state,
        model=mace_model_stress,
        dt=torch.tensor(dt),
        kT=kT,
        external_pressure=target_pressure,
    )

final_temp = (
    ts.calc_kT(masses=state.masses, momenta=state.momenta, system_idx=state.system_idx)
    / Units.temperature
)
print(f"Final temperature: {final_temp.item():.4f} K")

final_stress = mace_model_stress(state)["stress"]
final_volume = torch.det(state.current_cell)
final_pressure = ts.get_pressure(
    final_stress,
    ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    ),
    final_volume,
)
print(f"Final pressure: {final_pressure.item():.4f} eV/Å³")

print("\n" + "=" * 70)
print("Molecular dynamics examples completed!")
print("=" * 70)
