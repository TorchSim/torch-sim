"""Lennard-Jones FIRE optimization."""

# /// script
# dependencies = ["scipy>=1.15"]
# ///
import itertools
import os

import torch

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel


# Set up the device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)  # For reproducibility

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 10 if SMOKE_TEST else 2_000

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
        pos = base_pos + torch.tensor([i, j, k], device=device, dtype=dtype)
        positions.append(pos)

# Stack the positions into a tensor
positions = torch.stack(positions)

# Scale by lattice constant
positions = positions * a_len

# Create the cell tensor
cell = torch.tensor(
    [[4 * a_len, 0, 0], [0, 4 * a_len, 0], [0, 0, 4 * a_len]],
    device=device,
    dtype=dtype,
)

# Create the atomic numbers tensor
atomic_numbers = torch.full((positions.shape[0],), 18, device=device, dtype=torch.int)
# Add random perturbation to the positions to start with non-equilibrium structure
positions = positions + 0.2 * torch.randn(
    positions.shape, generator=generator, device=device, dtype=dtype
)
masses = torch.full((positions.shape[0],), 39.948, device=device, dtype=dtype)

# Initialize the Lennard-Jones model
model = LennardJonesModel(
    use_neighbor_list=False,
    sigma=3.405,
    epsilon=0.0104,
    cutoff=2.5 * 3.405,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=False,
)

# Create state with batch dimension
state = ts.SimState(
    positions=positions,
    masses=masses,
    cell=cell.unsqueeze(0),
    atomic_numbers=atomic_numbers,
    pbc=True,
)

# Run initial simulation and get results
results = model(state)

# Initialize FIRE optimizer
state = ts.fire_init(model=model, state=state, dt_start=0.005)


# Run optimization for N_steps
for step in range(N_steps):
    if step % 100 == 0:
        print(f"{step=}: Potential energy: {state.energy[0].item()} eV")
    state = ts.fire_step(model, state, dt_max=0.01)

# Print max force after optimization
print(f"Initial energy: {results['energy'][0].item()} eV")
print(f"Final energy: {state.energy[0].item()} eV")
print(f"Initial max force: {torch.max(torch.abs(results['forces'][0])).item()} eV/Å")
print(f"Final max force: {torch.max(torch.abs(state.forces[0])).item()} eV/Å")
