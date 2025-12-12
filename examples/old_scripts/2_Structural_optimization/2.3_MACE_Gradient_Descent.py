"""Batched MACE gradient descent example."""

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

# Create diamond cubic Silicon systems
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

fe = bulk("Fe", "bcc", a=2.8665, cubic=True).repeat((3, 3, 3))
fe.positions += 0.2 * rng.standard_normal(fe.positions.shape)

# Create a list of our atomic systems
atoms_list = [si_dc, fe]

# Create batched model
batched_model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

"""
# Convert data to tensors
positions_list = [
    torch.tensor(atoms.positions, device=device, dtype=dtype) for atoms in atoms_list
]
cell_list = [
    torch.tensor(atoms.cell.array, device=device, dtype=dtype) for atoms in atoms_list
]
masses_list = [
    torch.tensor(atoms.get_masses(), device=device, dtype=dtype) for atoms in atoms_list
]
atomic_numbers_list = [atoms.get_atomic_numbers() for atoms in atoms_list]

# Create batch data format
# First concatenate positions array
positions_numpy = np.concatenate([atoms.positions for atoms in atoms_list])
positions = torch.tensor(positions_numpy, device=device, dtype=dtype)

# Stack cell vectors into a (n_batch, 3, 3) array
cell_numpy = np.stack([atoms.cell.array for atoms in atoms_list])
cell = torch.tensor(cell_numpy, device=device, dtype=dtype)

# Concatenate atomic numbers
atomic_numbers_numpy = np.concatenate(
    [atoms.get_atomic_numbers() for atoms in atoms_list]
)
atomic_numbers = torch.tensor(atomic_numbers_numpy, device=device, dtype=torch.int)

# Concatenate masses
masses_numpy = np.concatenate([atoms.get_masses() for atoms in atoms_list])
masses = torch.tensor(masses_numpy, device=device, dtype=dtype)

# Create system indices tensor for scatter operations
atoms_per_system = torch.tensor(
    [len(atoms) for atoms in atoms_list], device=device, dtype=torch.int
)
system_indices = torch.repeat_interleave(
    torch.arange(len(atoms_per_system), device=device), atoms_per_system
)
"""

state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)

print(f"Positions shape: {state.positions.shape}")
print(f"Cell shape: {state.cell.shape}")
print(f"System indices shape: {state.system_idx.shape}")

# Run initial inference
results = batched_model(state)

# Use different learning rates for each batch
learning_rate = 0.01

# Initialize batched gradient descent optimizer
state = ts.gradient_descent_init(state=state, model=batched_model)

# Run batched optimization for a few steps
print("\nRunning batched gradient descent:")
for step in range(N_steps):
    if step % 10 == 0:
        print(f"Step {step}, Energy: {[res.item() for res in state.energy]} eV")
    state = ts.gradient_descent_step(
        state=state, model=batched_model, pos_lr=learning_rate
    )

print(f"Initial energies: {[res.item() for res in results['energy']]} eV")
print(f"Final energies: {[res.item() for res in state.energy]} eV")
