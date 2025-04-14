"""Lennard-Jones simple single system example."""

# /// script
# dependencies = [
#     "scipy>=1.15",
# ]
# ///

import itertools

import torch

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.unbatched.models.lennard_jones import UnbatchedLennardJonesModel


# Set up the device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

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
model = UnbatchedLennardJonesModel(
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

# Print system information
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")

state = dict(
    positions=positions,
    cell=cell,
    atomic_numbers=atomic_numbers,
    pbc=True,
)
# Run the simulation and get results
results = model(state)

# Print the results
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")
print(f"Energies: {results['energies']}")
print(f"Stresses: {results['stresses']}")

# Batched model
batched_model = LennardJonesModel(
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

# Batched state
state = dict(
    positions=positions,
    cell=cell.unsqueeze(0),
    atomic_numbers=atomic_numbers,
    pbc=True,
)

# Run the simulation and get results
results = batched_model(state)

# Print the results
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")
print(f"Energies: {results['energies']}")
print(f"Stresses: {results['stresses']}")
