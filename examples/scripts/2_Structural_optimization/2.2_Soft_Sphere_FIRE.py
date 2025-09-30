"""Structural optimization with soft sphere potential using FIRE optimizer."""

# /// script
# dependencies = [
#     "scipy>=1.15",
# ]
# ///

import itertools
import os

import torch

import torch_sim as ts
from torch_sim.models.soft_sphere import SoftSphereModel
from torch_sim.optimizers import fire


# Set up the device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)  # For reproducibility

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 10 if SMOKE_TEST else 2_000

# Create face-centered cubic (FCC) Cu
# 3.61 Å is a typical lattice constant for Cu
a_len = 3.61  # Lattice constant

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

# Create 4x4x4 supercell of FCC Cu manually
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

# Add random perturbation to the positions to start with non-equilibrium structure
positions = positions + 0.2 * torch.randn(
    positions.shape, generator=generator, device=device, dtype=dtype
)

# Create the atomic numbers tensor
atomic_numbers = torch.full((positions.shape[0],), 29, device=device, dtype=torch.int)

# Cu atomic mass in atomic mass units
masses = torch.full((positions.shape[0],), 63.546, device=device, dtype=dtype)

# Create state with batch dimension
state = ts.state.SimState(
    positions=positions,
    masses=masses,
    cell=cell.unsqueeze(0),
    atomic_numbers=atomic_numbers,
    pbc=True,
)

# Initialize the Soft Sphere model
model = SoftSphereModel(
    sigma=2.5,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=False,
    use_neighbor_list=True,
)

# Run initial simulation and get results
results = model(state)

# Initialize FIRE optimizer
fire_init, fire_update = fire(
    model=model,
    dt_start=0.005,
    dt_max=0.01,
)

state = fire_init(state=state)

# Run optimization for N_steps
for step in range(N_steps):
    if step % 100 == 0:
        print(f"{step=}: Total energy: {state.energy[0].item()} eV")
    state = fire_update(state)

# Print max force after optimization
print(f"Initial energy: {results['energy'][0].item()} eV")
print(f"Final energy: {state.energy[0].item()} eV")
print(f"Initial max force: {torch.max(torch.abs(results['forces'][0])).item()} eV/Å")
print(f"Final max force: {torch.max(torch.abs(state.forces[0])).item()} eV/Å")
