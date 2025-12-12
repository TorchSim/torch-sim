"""Batched MACE unit cell filter with gradient descent optimizer."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.units import UnitConversion


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
N_steps = 10 if SMOKE_TEST else 500

# Set random seed for reproducibility
rng = np.random.default_rng(seed=0)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

# Create FCC Copper
cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

# Create BCC Iron
fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

# Create a list of our atomic systems
atoms_list = [si_dc, cu_dc, fe_dc]

# Print structure information
print(f"Silicon atoms: {len(si_dc)}")
print(f"Copper atoms: {len(cu_dc)}")
print(f"Iron atoms: {len(fe_dc)}")
print(f"Total number of structures: {len(atoms_list)}")

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
# Run initial inference
results = model(state)

# Use same learning rate for all batches
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


# Run optimization for a few steps
print("\nRunning batched unit cell gradient descent:")
for step in range(N_steps):
    P1 = -torch.trace(state.stress[0]) * UnitConversion.eV_per_Ang3_to_GPa / 3
    P2 = -torch.trace(state.stress[1]) * UnitConversion.eV_per_Ang3_to_GPa / 3
    P3 = -torch.trace(state.stress[2]) * UnitConversion.eV_per_Ang3_to_GPa / 3

    if step % 20 == 0:
        print(
            f"Step {step}, Energy: {[energy.item() for energy in state.energy]}, "
            f"P1={P1:.4f} GPa, P2={P2:.4f} GPa, P3={P3:.4f} GPa"
        )

    state = ts.gradient_descent_step(
        state=state, model=model, pos_lr=pos_lr, cell_lr=cell_lr
    )

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")

initial_pressure = [
    torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
    for stress in results["stress"]
]
final_pressure = [
    torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
    for stress in state.stress
]
print(f"{initial_pressure=} GPa")
print(f"{final_pressure=} GPa")
