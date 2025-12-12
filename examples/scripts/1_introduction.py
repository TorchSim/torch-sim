"""Introduction to TorchSim - Basic Examples with Lennard-Jones and MACE models.

This script demonstrates the fundamental usage of TorchSim with:
- Lennard-Jones model for simple classical potentials
- MACE model for machine learning potentials
"""

# /// script
# dependencies = ["scipy>=1.15", "mace-torch>=0.3.12"]
# ///

import itertools

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.mace import MaceModel, MaceUrls


# Set up the device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


# ============================================================================
# SECTION 1: Lennard-Jones Model - Simple Classical Potential
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Lennard-Jones Model")
print("=" * 70)

# Create face-centered cubic (FCC) Argon
# 5.26 Å is a typical lattice constant for Ar
a_len = 5.26  # Lattice constant

# Generate base FCC unit cell positions (scaled by lattice constant)
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

# Create 4x4x4 supercell of FCC Argon manually
positions = []
for i, j, k in itertools.product(range(4), range(4), range(4)):
    for base_pos in base_positions:
        # Add unit cell position + offset for supercell
        pos = base_pos + torch.tensor([i, j, k], device=device, dtype=dtype)
        positions.append(pos)

# Stack the positions into a tensor
positions = torch.stack(positions)

# Scale by lattice constant
positions = positions * a_len

# Create the cell tensor
cell = torch.tensor(
    [[4 * a_len, 0, 0], [0, 4 * a_len, 0], [0, 0, 4 * a_len]], device=device, dtype=dtype
)

# Create the atomic numbers tensor
atomic_numbers = torch.full((positions.shape[0],), 18, device=device, dtype=torch.int)

# Initialize the Lennard-Jones model
# Parameters:
#  - sigma: distance at which potential is zero (3.405 Å for Ar)
#  - epsilon: depth of potential well (0.0104 eV for Ar)
#  - cutoff: distance beyond which interactions are ignored (typically 2.5*sigma)
lj_model = LennardJonesModel(
    use_neighbor_list=True,
    cutoff=2.5 * 3.405,
    sigma=3.405,
    epsilon=0.0104,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
    per_atom_energies=True,
    per_atom_stresses=True,
)

# State dict
state = dict(
    positions=positions, cell=cell.unsqueeze(0), atomic_numbers=atomic_numbers, pbc=True
)

# Run the simulation and get results
results = lj_model(state)

# Print the results
print(f"Energy: {results['energy']}")
print(f"Forces shape: {results['forces'].shape}")
print(f"Stress shape: {results['stress'].shape}")
print(f"Per-atom energies shape: {results['energies'].shape}")
print(f"Per-atom stresses shape: {results['stresses'].shape}")


# ============================================================================
# SECTION 2: MACE Model - Machine Learning Potential (Batched)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: MACE Model with Batched Input")
print("=" * 70)

# Load the raw model from the downloaded model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms_list = [si_dc, si_dc]

batched_model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
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

# create system idx array of shape (16,) which is 0 for first 8 atoms, 1 for last 8 atoms
atoms_per_system = torch.tensor(
    [len(atoms) for atoms in atoms_list], device=device, dtype=torch.int
)
system_idx = torch.repeat_interleave(
    torch.arange(len(atoms_per_system), device=device), atoms_per_system
)

# You can see their shapes are as expected
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")
print(f"Atomic numbers: {atomic_numbers.shape}")
print(f"System indices: {system_idx.shape}")

# Now we can pass them to the model
results = batched_model(
    dict(
        positions=positions,
        cell=cell,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
        pbc=True,
    )
)

# The energy has shape (n_systems,) as the structures in a batch
print(f"Energy shape: {results['energy'].shape}")

# The forces have shape (n_atoms, 3) same as positions
print(f"Forces shape: {results['forces'].shape}")

# The stress has shape (n_systems, 3, 3) same as cell
print(f"Stress shape: {results['stress'].shape}")

# Check if the energy, forces, and stress are the same for the Si system across batches
# Each system has 64 atoms (2x2x2 supercell of 8-atom Si diamond)
n_atoms_per_system = len(si_dc)
energy_diff = torch.max(torch.abs(results["energy"][0] - results["energy"][1]))
forces_diff = torch.max(
    torch.abs(
        results["forces"][:n_atoms_per_system] - results["forces"][n_atoms_per_system:]
    )
)
stress_diff = torch.max(torch.abs(results["stress"][0] - results["stress"][1]))

print(f"\nMax energy difference: {energy_diff}")
print(f"Max forces difference: {forces_diff}")
print(f"Max stress difference: {stress_diff}")

print("\n" + "=" * 70)
print("Introduction examples completed!")
print("=" * 70)
