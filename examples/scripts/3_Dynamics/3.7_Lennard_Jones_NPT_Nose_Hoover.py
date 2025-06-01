"""Lennard-Jones simulation in NPT ensemble using Nose-Hoover chain."""

# /// script
# dependencies = [
#     "scipy>=1.15",
# ]
# ///

import itertools
import os

import torch

import torch_sim as ts
from torch_sim.integrators.npt import npt_nose_hoover, npt_nose_hoover_invariant
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.quantities import calc_kinetic_energy, calc_kT, get_pressure
from torch_sim.units import MetalUnits as Units


# Set up the device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Set random seed and deterministic behavior for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up the random number generator
generator = torch.Generator(device=device)
generator.manual_seed(42)  # For reproducibility

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 100 if SMOKE_TEST else 10_000

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
state = ts.SimState(
    positions=positions,
    masses=masses,
    cell=cell.unsqueeze(0),
    pbc=True,
    atomic_numbers=atomic_numbers,
)
# Run initial simulation and get results
results = model(state)

dt = 0.001 * Units.time  # Time step (1 ps)
kT = 200 * Units.temperature  # Temperature (200 K)
target_pressure = (
    torch.tensor(10, device=device, dtype=dtype) * Units.pressure
)  # Target pressure (10 kbar)

npt_init, npt_update = npt_nose_hoover(
    model=model,
    dt=dt,
    kT=kT,
    external_pressure=target_pressure,
    chain_length=3,  # Chain length
    chain_steps=1,
    sy_steps=1,
)
state = npt_init(state=state, seed=1)

# Run the simulation
for step in range(N_steps):
    if step % 50 == 0:
        temp = (
            calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
            / Units.temperature
        )
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        )
        pressure = get_pressure(
            model(state)["stress"],
            calc_kinetic_energy(
                masses=state.masses, momenta=state.momenta, batch=state.batch
            ),
            torch.det(state.current_cell),
        )
        pressure = pressure.item() / Units.pressure
        xx, yy, zz = state.cell[..., 0, 0], state.cell[..., 1, 1], state.cell[..., 2, 2]
        print(
            f"{step=}: Temperature: {temp.item():.4f}, "
            f"invariant: {invariant.item():.4f}, "
            f"{pressure=:.4f}, "
            f"cell xx yy zz: {xx.item():.4f}, {yy.item():.4f}, {zz.item():.4f}"
        )
    state = npt_update(state, kT=kT, external_pressure=target_pressure)

temp = (
    calc_kT(masses=state.masses, momenta=state.momenta, batch=state.batch)
    / Units.temperature
)
print(f"Final temperature: {temp.item():.4f}")

pressure = get_pressure(
    model(state)["stress"],
    calc_kinetic_energy(masses=state.masses, momenta=state.momenta, batch=state.batch),
    torch.det(state.current_cell),
)
pressure = pressure.item() / Units.pressure
print(f"Final {pressure=:.4f}")
