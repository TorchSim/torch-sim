"""Minimal MACE batched example."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load the compiled model from the local file
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms_list = [si_dc, si_dc]
PERIODIC = True

batched_model = MaceModel(
    # Pass the raw model
    model=loaded_model,
    # Or load from compiled model
    # model=compiled_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# First we will create a concatenated positions array
# This will have shape (16, 3) which is concatenated from two 8 atom systems
positions_numpy = np.concatenate([atoms.positions for atoms in atoms_list])

# stack cell vectors into a (2, 3, 3) array where the first index is batch dimension
cell_numpy = np.stack([atoms.cell.array for atoms in atoms_list])

# concatenate atomic numbers into a (16,) array
atomic_numbers_numpy = np.concatenate(
    [atoms.get_atomic_numbers() for atoms in atoms_list]
)

# convert to tensors
positions = torch.tensor(positions_numpy, device=device, dtype=dtype)
cell = torch.tensor(cell_numpy, device=device, dtype=dtype)
atomic_numbers = torch.tensor(atomic_numbers_numpy, device=device, dtype=torch.int)

# create batch index array of shape (16,) which is 0 for first 8 atoms, 1 for last 8 atoms
atoms_per_batch = torch.tensor(
    [len(atoms) for atoms in atoms_list], device=device, dtype=torch.int
)
batch = torch.repeat_interleave(
    torch.arange(len(atoms_per_batch), device=device), atoms_per_batch
)

# You can see their shapes are as expected
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")
print(f"Atomic numbers: {atomic_numbers.shape}")
print(f"Batch: {batch.shape}")

# Now we can pass them to the model
results = batched_model(
    positions=positions, cell=cell, atomic_numbers=atomic_numbers, batch=batch
)

# The energy has shape (n_batches,) as the structures in a batch
print(f"Energy: {results['energy'].shape}")

# The forces have shape (n_atoms, 3) same as positions
print(f"Forces: {results['forces'].shape}")

# The stress has shape (n_batches, 3, 3) same as cell
print(f"Stress: {results['stress'].shape}")

# Check if the energy, forces, and stress are the same for the Si system across the batch
print(torch.max(torch.abs(results["energy"][0] - results["energy"][1])))
print(torch.max(torch.abs(results["forces"][0] - results["forces"][1])))
print(torch.max(torch.abs(results["stress"][0] - results["stress"][1])))
