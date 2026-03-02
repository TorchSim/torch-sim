"""Miscellaneous Examples - Advanced features and utilities.

This script demonstrates:
- Batched neighbor list calculations
- Velocity autocorrelation functions
- Autograd features for custom potentials
- Performance comparisons
"""

# /// script
# dependencies = ["ase>=3.26", "scipy>=1.15", "matplotlib", "numpy"]
# ///

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import torch_sim as ts
from torch_sim import transforms
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.neighbors import torch_nl_linked_cell, torch_nl_n2
from torch_sim.properties.correlations import VelocityAutoCorrelation
from torch_sim.telemetry import configure_logging, get_logger
from torch_sim.units import MetalUnits as Units


configure_logging(log_file="7_others.log")
log = get_logger(name="7_others")

SMOKE_TEST = os.getenv("CI") is not None

# ============================================================================
# SECTION 1: Batched Neighbor List Calculations
# ============================================================================
log.info("=" * 70)
log.info("SECTION 1: Batched Neighbor List Calculations")
log.info("=" * 70)

# Create multiple atomic systems
atoms_list = [
    bulk("Si", "diamond", a=5.43),
    bulk("Ge", "diamond", a=5.65),
    bulk("Cu", "fcc", a=3.61),
]

state = ts.io.atoms_to_state(atoms_list, device=torch.device("cpu"), dtype=torch.float32)
pos, cell, pbc = state.positions, state.cell, state.pbc
system_idx = state.system_idx
n_atoms = state.n_atoms
cutoff = torch.tensor(4.0, dtype=pos.dtype)
self_interaction = False

# Ensure pbc has the correct shape [n_systems, 3]
pbc_tensor = torch.tensor(pbc).repeat(state.n_systems, 1)

log.info(f"Batched system with {state.n_systems} structures:")
for i, atoms in enumerate(atoms_list):
    log.info(f"  Structure {i}: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

# Method 1: Linked cell neighbor list (efficient for large systems)
log.info("Calculating neighbor lists with linked cell method...")
mapping, mapping_system, shifts_idx = torch_nl_linked_cell(
    pos, cell, pbc_tensor, cutoff, system_idx, self_interaction
)
cell_shifts = transforms.compute_cell_shifts(cell, shifts_idx, mapping_system)
dds = transforms.compute_distances_with_cell_shifts(pos, mapping, cell_shifts)

log.info("Linked cell results:")
log.info(f"  Mapping shape: {mapping.shape}")
log.info(f"  Mapping system shape: {mapping_system.shape}")
log.info(f"  Shifts idx shape: {shifts_idx.shape}")
log.info(f"  Cell shifts shape: {cell_shifts.shape}")
log.info(f"  Distances shape: {dds.shape}")
log.info(f"  Total neighbor pairs: {mapping.shape[1]}")

# Method 2: N^2 neighbor list (simple but slower)
log.info("Calculating neighbor lists with N^2 method...")
mapping_n2, mapping_system_n2, shifts_idx_n2 = torch_nl_n2(
    pos, cell, pbc_tensor, cutoff, system_idx, self_interaction
)
cell_shifts_n2 = transforms.compute_cell_shifts(cell, shifts_idx_n2, mapping_system_n2)
dds_n2 = transforms.compute_distances_with_cell_shifts(pos, mapping_n2, cell_shifts_n2)

log.info("N^2 method results:")
log.info(f"  Mapping shape: {mapping_n2.shape}")
log.info(f"  Total neighbor pairs: {mapping_n2.shape[1]}")

# Verify consistency
if mapping.shape[1] == mapping_n2.shape[1]:
    log.info("✓ Both methods found the same number of neighbors")
else:
    log.info(
        f"⚠ Different neighbor counts: "
        f"linked cell={mapping.shape[1]}, N^2={mapping_n2.shape[1]}"
    )


# ============================================================================
# SECTION 2: Velocity Autocorrelation Function (VACF)
# ============================================================================
log.info("=" * 70)
log.info("SECTION 2: Velocity Autocorrelation Function")
log.info("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# Create solid Argon system with Lennard-Jones potential
atoms = bulk("Ar", crystalstructure="fcc", a=5.256, cubic=True)
atoms = atoms.repeat((3, 3, 3))
temperature = 50.0  # Kelvin

log.info("System: Solid Argon")
log.info(f"  Temperature: {temperature} K")
log.info(f"  Number of atoms: {len(atoms)}")
log.info(f"  Cell size: {atoms.cell.cellpar()[:3]}")

# Initialize velocities with Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

# Create Lennard-Jones model for Argon
epsilon = 0.0104  # eV
sigma = 3.4  # Å
cutoff = 2.5 * sigma

lj_model = LennardJonesModel(
    sigma=sigma,
    epsilon=epsilon,
    cutoff=cutoff,
    device=device,
    dtype=dtype,
    compute_forces=True,
)

log.info("Lennard-Jones parameters:")
log.info(f"  Epsilon: {epsilon} eV")
log.info(f"  Sigma: {sigma} Å")
log.info(f"  Cutoff: {cutoff:.2f} Å")

# Simulation parameters
timestep = 0.001  # ps (1 fs)
dt = torch.tensor(timestep * Units.time, device=device, dtype=dtype)
temp_kT = temperature * Units.temperature  # noqa: N816
kT = torch.tensor(temp_kT, device=device, dtype=dtype)

# Initialize NVE integrator
state = ts.nve_init(state=state, model=lj_model, kT=kT)

# Create VACF calculator
window_size = 150 if not SMOKE_TEST else 20  # Correlation window length
vacf_calc = VelocityAutoCorrelation(
    window_size=window_size,
    device=device,
    use_running_average=True,
    normalize=True,
)

# Set up trajectory reporter
trajectory_file = "tmp/vacf_example.h5"
correlation_dt = 10  # Steps between correlation samples
reporter = ts.TrajectoryReporter(
    trajectory_file,
    state_frequency=100,
    prop_calculators={correlation_dt: {"vacf": vacf_calc}},
)

# Run simulation
num_steps = 100 if SMOKE_TEST else 15000
log.info(f"Running NVE simulation for {num_steps} steps...")
log.info(f"VACF window size: {window_size} samples")
log.info(f"Correlation sampling interval: {correlation_dt} steps")

for step in range(num_steps):
    state = ts.nve_step(state=state, model=lj_model, dt=dt)
    reporter.report(state, step)

    if step % 1000 == 0 and not SMOKE_TEST:
        total_energy = state.energy + ts.calc_kinetic_energy(
            masses=state.masses, momenta=state.momenta
        )
        log.info(f"  Step {step}: Total energy = {total_energy.item():.4f} eV")

reporter.close()

# Calculate time axis for VACF
time_steps = np.arange(window_size)
time = time_steps * correlation_dt * timestep * 1000  # Convert to fs

if vacf_calc.vacf is not None:
    vacf_data = vacf_calc.vacf.detach().cpu().numpy()
    log.info("VACF calculation complete:")
    log.info(f"  Number of windows averaged: {vacf_calc._window_count}")  # noqa: SLF001
    log.info(f"  VACF at t=0: {vacf_data[0]:.4f}")
    log.info(f"  VACF decay at t_max: {vacf_data[-1]:.4f}")

    # Plot VACF if not in CI mode
    if not SMOKE_TEST:
        plt.figure(figsize=(10, 6))
        plt.plot(time, vacf_data, "b-", linewidth=2)
        plt.xlabel("Time (fs)", fontsize=12)
        plt.ylabel("VACF (normalized)", fontsize=12)
        plt.title(
            f"Velocity Autocorrelation Function (Argon at {temperature}K)", fontsize=14
        )
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.grid(visible=True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tmp/vacf_example.png", dpi=150)
        log.info("✓ VACF plot saved to tmp/vacf_example.png")
    else:
        log.info("Skipping plot generation in CI mode")
else:
    log.info("Warning: VACF data not available")


# ============================================================================
# SECTION 3: Summary
# ============================================================================
log.info("=" * 70)
log.info("Summary")
log.info("=" * 70)
log.info("Demonstrated features:")
log.info("  1. Batched neighbor list calculations")
log.info("     - Linked cell method (efficient)")
log.info("     - N^2 method (simple)")
log.info("  2. Velocity autocorrelation function (VACF)")
log.info("     - NVE molecular dynamics")
log.info("     - Running average over time windows")
log.info("     - Normalized correlation decay")

log.info("Key capabilities:")
log.info("  - Efficient batched computations")
log.info("  - Multiple neighbor list algorithms")
log.info("  - Advanced property calculations during MD")
log.info("  - Trajectory analysis and correlation functions")

log.info("=" * 70)
log.info("Miscellaneous examples completed!")
log.info("=" * 70)
