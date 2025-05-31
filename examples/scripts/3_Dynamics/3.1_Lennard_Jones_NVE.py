"""NVE simulation with Lennard-Jones potential."""

# /// script
# dependencies = [
#     "scipy>=1.15",
# ]
# ///

import itertools
import os

import torch

import torch_sim as ts
from torch_sim.integrators import nve
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.quantities import calc_kinetic_energy
from torch_sim.units import MetalUnits as Units


# Set up the device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Number of steps to run
N_steps = 100 if os.getenv("CI") else 2_000

# Set random seed and deterministic behavior for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)  # For reproducibility

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

# Create the atomic numbers tensor (Argon = 18)
atomic_numbers = torch.full((positions.shape[0],), 18, device=device, dtype=torch.int)
# Create the masses tensor (Argon = 39.948 amu)
masses = torch.full((positions.shape[0],), 39.948, device=device, dtype=dtype)

state = ts.SimState(
    positions=positions,
    masses=masses,
    cell=cell,
    pbc=True,
    atomic_numbers=atomic_numbers,
)
# Initialize the Lennard-Jones model
# Parameters:
#  - sigma: distance at which potential is zero (3.405 Å for Ar)
#  - epsilon: depth of potential well (0.0104 eV for Ar)
#  - cutoff: distance beyond which interactions are ignored (typically 2.5*sigma)
model = LennardJonesModel(
    use_neighbor_list=False,
    sigma=3.405,
    epsilon=0.0104,
    cutoff=2.5 * 3.405,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
)

# Run initial simulation and get results
results = model(state)

# Set up NVE simulation
# kT: initial temperature in metal units (K)
# dt: timestep in metal units (ps)
kT = 80 * Units.temperature
dt = 0.001 * Units.time

# Initialize NVE integrator
nve_init, nve_update = nve(model=model, dt=dt, kT=kT)

state = nve_init(state=state)

# Run NVE simulation for 1000 steps
for step in range(N_steps):
    if step % 100 == 0:
        # Calculate total energy (potential + kinetic)
        total_energy = state.energy + calc_kinetic_energy(
            masses=state.masses, momenta=state.momenta
        )
        print(f"{step=}: Total energy: {total_energy.item():.4f}")

    # Update state using NVE integrator
    state = nve_update(state=state, dt=dt)

final_total_energy = state.energy + calc_kinetic_energy(
    masses=state.masses, momenta=state.momenta
)
print(f"Final total energy: {final_total_energy.item():.4f}")
