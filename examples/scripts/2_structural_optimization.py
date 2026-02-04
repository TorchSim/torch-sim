"""Structural Optimization Examples - FIRE and Gradient Descent with various cell filters.

This script demonstrates structural optimization techniques with:
- FIRE optimizer (Fast Inertial Relaxation Engine)
- Gradient descent optimizer
- Different cell filters (none, unit cell, Frechet cell)
- Batched optimization for multiple structures
"""

# /// script
# dependencies = ["scipy>=1.15", "mace-torch>=0.3.12"]
# ///

import itertools
import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.units import UnitConversion


# Set up the device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 10 if SMOKE_TEST else 500


# ============================================================================
# SECTION 1: Lennard-Jones FIRE Optimization
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Lennard-Jones FIRE Optimization")
print("=" * 70)

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)

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

# Create cell and atomic properties
cell = torch.tensor(
    [[4 * a_len, 0, 0], [0, 4 * a_len, 0], [0, 0, 4 * a_len]],
    device=device,
    dtype=dtype,
)
atomic_numbers = torch.full((positions.shape[0],), 18, device=device, dtype=torch.int)
# Add random perturbation to start with non-equilibrium structure
positions = positions + 0.2 * torch.randn(
    positions.shape, generator=generator, device=device, dtype=dtype
)
masses = torch.full((positions.shape[0],), 39.948, device=device, dtype=dtype)

# Initialize the Lennard-Jones model
lj_model = LennardJonesModel(
    use_neighbor_list=False,
    sigma=3.405,
    epsilon=0.0104,
    cutoff=2.5 * 3.405,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=False,
)

# Create state
state = ts.SimState(
    positions=positions,
    masses=masses,
    cell=cell.unsqueeze(0),
    atomic_numbers=atomic_numbers,
    pbc=True,
)

# Run initial simulation
results = lj_model(state)

# Initialize FIRE optimizer
state = ts.fire_init(state=state, model=lj_model, dt_start=0.005)

# Run optimization
for step in range(N_steps):
    if step % 20 == 0:
        print(f"Step {step}: Potential energy: {state.energy[0].item()} eV")
    state = ts.fire_step(state=state, model=lj_model, dt_max=0.01)

print(f"Initial energy: {results['energy'][0].item()} eV")
print(f"Final energy: {state.energy[0].item()} eV")
print(f"Initial max force: {torch.max(torch.abs(results['forces'][0])).item()} eV/Å")
print(f"Final max force: {torch.max(torch.abs(state.forces[0])).item()} eV/Å")


# ============================================================================
# SECTION 2: Batched MACE FIRE Optimization (Atomic Positions Only)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Batched MACE FIRE - Positions Only")
print("=" * 70)

# Load MACE model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# Set random seed for reproducibility
rng = np.random.default_rng(seed=0)

# Create different crystal structures with perturbations
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

print(f"Silicon atoms: {len(si_dc)}")
print(f"Copper atoms: {len(cu_dc)}")
print(f"Iron atoms: {len(fe_dc)}")

# Create batched model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Convert atoms to state
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)

# Initialize FIRE optimizer
state = ts.fire_init(state=state, model=model, dt_start=0.005)

print("\nRunning FIRE:")
for step in range(N_steps):
    if step % 20 == 0:
        print(f"Step {step}, Energy: {[energy.item() for energy in state.energy]}")

    state = ts.fire_step(state=state, model=model, dt_max=0.01)

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")


# ============================================================================
# SECTION 3: Batched MACE Gradient Descent Optimization
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Batched MACE Gradient Descent")
print("=" * 70)

# Reset structures with new perturbations
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

fe = bulk("Fe", "bcc", a=2.8665, cubic=True).repeat((3, 3, 3))
fe.positions += 0.2 * rng.standard_normal(fe.positions.shape)

atoms_list = [si_dc, fe]

state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)

# Initialize gradient descent optimizer
learning_rate = 0.01
state = ts.gradient_descent_init(state=state, model=model)

print("\nRunning batched gradient descent:")
for step in range(N_steps):
    if step % 10 == 0:
        print(f"Step {step}, Energy: {[res.item() for res in state.energy]} eV")
    state = ts.gradient_descent_step(state=state, model=model, pos_lr=learning_rate)

print(f"Initial energies: {[res.item() for res in results['energy']]} eV")
print(f"Final energies: {[res.item() for res in state.energy]} eV")


# ============================================================================
# SECTION 4: Unit Cell Filter with Gradient Descent
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Unit Cell Filter with Gradient Descent")
print("=" * 70)

# Recreate structures with perturbations
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

# Convert atoms to state
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)

# Use different learning rates for positions and cell
pos_lr, cell_lr = 0.01, 0.1

state = ts.gradient_descent_init(
    state=state,
    model=model,
    cell_filter=ts.CellFilter.unit,
    cell_factor=None,  # Will default to atoms per system
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)

print("\nRunning batched unit cell gradient descent:")
for step in range(N_steps):
    if step % 20 == 0:
        P1 = -torch.trace(state.stress[0]) * UnitConversion.eV_per_Ang3_to_GPa / 3
        P2 = -torch.trace(state.stress[1]) * UnitConversion.eV_per_Ang3_to_GPa / 3
        P3 = -torch.trace(state.stress[2]) * UnitConversion.eV_per_Ang3_to_GPa / 3

        print(
            f"Step {step}, Energy: {[energy.item() for energy in state.energy]}, "
            f"P1={P1:.4f} GPa, P2={P2:.4f} GPa, P3={P3:.4f} GPa"
        )

    state = ts.gradient_descent_step(
        state=state, model=model, pos_lr=pos_lr, cell_lr=cell_lr
    )

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")


# ============================================================================
# SECTION 5: Unit Cell Filter with FIRE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Unit Cell Filter with FIRE")
print("=" * 70)

# Recreate structures with perturbations
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

# Convert atoms to state
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)

# Initialize FIRE optimizer with unit cell filter
state = ts.fire_init(
    state=state,
    model=model,
    cell_filter=ts.CellFilter.unit,
    cell_factor=None,
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)

print("\nRunning batched unit cell FIRE:")
for step in range(N_steps):
    if step % 20 == 0:
        P1 = -torch.trace(state.stress[0]) * UnitConversion.eV_per_Ang3_to_GPa / 3
        P2 = -torch.trace(state.stress[1]) * UnitConversion.eV_per_Ang3_to_GPa / 3
        P3 = -torch.trace(state.stress[2]) * UnitConversion.eV_per_Ang3_to_GPa / 3

        print(
            f"Step {step}, Energy: {[energy.item() for energy in state.energy]}, "
            f"P1={P1:.4f} GPa, P2={P2:.4f} GPa, P3={P3:.4f} GPa"
        )

    state = ts.fire_step(state=state, model=model)

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")


# ============================================================================
# SECTION 6: Frechet Cell Filter with FIRE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Frechet Cell Filter with FIRE")
print("=" * 70)

# Recreate structures with perturbations
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

# Convert atoms to state
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)

# Initialize FIRE optimizer with Frechet cell filter
state = ts.fire_init(
    state=state,
    model=model,
    cell_filter=ts.CellFilter.frechet,
    cell_factor=None,
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)

print("\nRunning batched frechet cell filter with FIRE:")
for step in range(N_steps):
    if step % 20 == 0:
        P1 = -torch.trace(state.stress[0]) * UnitConversion.eV_per_Ang3_to_GPa / 3
        P2 = -torch.trace(state.stress[1]) * UnitConversion.eV_per_Ang3_to_GPa / 3
        P3 = -torch.trace(state.stress[2]) * UnitConversion.eV_per_Ang3_to_GPa / 3

        print(
            f"Step {step}, Energy: {[energy.item() for energy in state.energy]}, "
            f"P1={P1:.4f} GPa, P2={P2:.4f} GPa, P3={P3:.4f} GPa"
        )

    state = ts.fire_step(state=state, model=model)

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")

initial_pressure = [
    -torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
    for stress in results["stress"]
]
final_pressure = [
    -torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
    for stress in state.stress
]
print(f"Initial pressure: {initial_pressure} GPa")
print(f"Final pressure: {final_pressure} GPa")

# ============================================================================
# SECTION 7: Batched MACE L-BFGS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 7: Batched MACE L-BFGS")
print("=" * 70)

# Recreate structures with perturbations
si_dc = bulk("Si", "diamond", a=5.21).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)
state = ts.lbfgs_init(state=state, model=model, alpha=70.0, step_size=1.0)

print("\nRunning L-BFGS:")
for step in range(N_steps):
    if step % 20 == 0:
        print(f"Step {step}, Energy: {[energy.item() for energy in state.energy]}")
    state = ts.lbfgs_step(state=state, model=model, max_history=100)

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")


# ============================================================================
# SECTION 8: Batched MACE BFGS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 8: Batched MACE BFGS")
print("=" * 70)

# Recreate structures with perturbations
si_dc = bulk("Si", "diamond", a=5.21).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
results = model(state)
state = ts.bfgs_init(state=state, model=model, alpha=70.0)

print("\nRunning BFGS:")
for step in range(N_steps):
    if step % 20 == 0:
        print(f"Step {step}, Energy: {[energy.item() for energy in state.energy]}")
    state = ts.bfgs_step(state=state, model=model)

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")


print("\n" + "=" * 70)
print("Structural optimization examples completed!")
print("=" * 70)
