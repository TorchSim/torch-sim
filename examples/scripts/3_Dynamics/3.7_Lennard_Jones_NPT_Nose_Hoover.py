"""Lennard-Jones simulation in NPT ensemble using Nose-Hoover chain."""

import itertools
import os

import torch

from torch_sim.quantities import calc_kinetic_energy, calc_kT
from torch_sim.state import SimState
from torch_sim.unbatched.models.lennard_jones import UnbatchedLennardJonesModel
from torch_sim.unbatched.unbatched_integrators import (
    npt_nose_hoover,
    npt_nose_hoover_invariant,
)
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
N_steps = 100 if os.getenv("CI") else 10_000

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
model = UnbatchedLennardJonesModel(
    use_neighbor_list=False,
    sigma=3.405,
    epsilon=0.0104,
    cutoff=2.5 * 3.405,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
)
state = SimState(
    positions=positions,
    masses=masses,
    cell=cell,
    pbc=True,
    atomic_numbers=atomic_numbers,
)
# Run initial simulation and get results
results = model(state)

dt = 0.001 * Units.time  # Time step (1 ps)
kT = 200 * Units.temperature  # Temperature (200 K)
target_pressure = 0 * Units.pressure  # Target pressure (10 kbar)

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


def get_pressure(
    stress: torch.Tensor, kinetic_energy: torch.Tensor, volume: torch.Tensor, dim: int = 3
) -> torch.Tensor:
    """Compute the pressure from the stress tensor.

    The stress tensor is defined as 1/volume * dU/de_ij
    So the pressure is -1/volume * trace(dU/de_ij)
    """
    return 1 / (dim) * ((2 * kinetic_energy / volume) - torch.trace(stress))


# Run the simulation
for step in range(N_steps):
    if step % 50 == 0:
        temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
        invariant = npt_nose_hoover_invariant(
            state, kT=kT, external_pressure=target_pressure
        )
        pressure = get_pressure(
            model(state)["stress"],
            calc_kinetic_energy(masses=state.masses, momenta=state.momenta),
            torch.det(state.current_cell),
        )
        pressure = pressure.item() / Units.pressure
        xx, yy, zz = torch.diag(state.current_cell)
        print(
            f"{step=}: Temperature: {temp:.4f}: invariant: {invariant.item():.4f}, "
            f"{pressure=:.4f}, "
            f"cell xx yy zz: {xx.item():.4f}, {yy.item():.4f}, {zz.item():.4f}"
        )
    state = npt_update(state, kT=kT, external_pressure=target_pressure)

temp = calc_kT(masses=state.masses, momenta=state.momenta) / Units.temperature
print(f"Final temperature: {temp:.4f}")

pressure = get_pressure(
    model(state)["stress"],
    calc_kinetic_energy(masses=state.masses, momenta=state.momenta),
    torch.det(state.current_cell),
)
pressure = pressure.item() / Units.pressure
print(f"Final {pressure=:.4f}")
