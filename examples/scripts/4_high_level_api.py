"""High-Level API Examples - Simplified interface for common workflows.

This script demonstrates the high-level API for:
- Integration with different models and integrators
- Trajectory logging and reporting
- Batched simulations
- Custom convergence criteria
- Support for ASE Atoms and Pymatgen Structure objects
"""

# /// script
# dependencies = ["mace-torch>=0.3.12", "pymatgen>=2025.2.18"]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from pymatgen.core import Structure

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.mace import MaceModel
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter
from torch_sim.units import MetalUnits


SMOKE_TEST = os.getenv("CI") is not None

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# SECTION 1: Basic Integration with Lennard-Jones
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Basic Integration with Lennard-Jones")
print("=" * 70)

lj_model = LennardJonesModel(
    sigma=2.0,  # Å, typical for Si-Si interaction
    epsilon=0.1,  # eV, typical for Si-Si interaction
    device=device,
    dtype=torch.float64,
)

si_atoms = bulk("Si", "fcc", a=5.43, cubic=True)

final_state = ts.integrate(
    system=si_atoms,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100 if SMOKE_TEST else 1000,
    temperature=2000,
    timestep=0.002,
)
final_atoms = ts.io.state_to_atoms(final_state)

print(f"Final energy: {final_state.energy.item():.4f} eV")
print(f"Final atoms: {len(final_atoms)} atoms")


# ============================================================================
# SECTION 2: Integration with Trajectory Logging
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Integration with Trajectory Logging")
print("=" * 70)

trajectory_file = "tmp/lj_trajectory.h5md"

# Report potential energy every 10 steps and kinetic energy every 20 steps
prop_calculators = {
    10: {"potential_energy": lambda state: state.energy},
    20: {
        "kinetic_energy": lambda state: ts.calc_kinetic_energy(
            momenta=state.momenta, masses=state.masses
        )
    },
}

reporter = TrajectoryReporter(
    trajectory_file,
    state_frequency=10,  # Report state every 10 steps
    prop_calculators=prop_calculators,
)

final_state = ts.integrate(
    system=si_atoms,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100 if SMOKE_TEST else 1000,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)

# Read trajectory data
with TorchSimTrajectory(trajectory_file) as traj:
    kinetic_energies = traj.get_array("kinetic_energy")
    potential_energies = traj.get_array("potential_energy")
    final_energy = potential_energies[-1]
    final_atoms = traj.get_atoms(-1)

print(f"Final energy from trajectory: {final_energy:.4f} eV")
print(f"Number of kinetic energy samples: {len(kinetic_energies)}")
print(f"Number of potential energy samples: {len(potential_energies)}")


# ============================================================================
# SECTION 3: MACE Model with High-Level API
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MACE Model with High-Level API")
print("=" * 70)

mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float64,
    compute_forces=True,
)

reporter = TrajectoryReporter(
    trajectory_file,
    state_frequency=10,
    prop_calculators=prop_calculators,
)

final_state = ts.integrate(
    system=si_atoms,
    model=mace_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100 if SMOKE_TEST else 1000,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)
final_atoms = ts.io.state_to_atoms(final_state)

print(f"Final energy: {final_state.energy.item():.4f} eV")


# ============================================================================
# SECTION 4: Batched Integration
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Batched Integration")
print("=" * 70)

fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True)
fe_atoms_supercell = fe_atoms.repeat([2, 2, 2])
si_atoms_supercell = si_atoms.repeat([2, 2, 2])

final_state = ts.integrate(
    system=[si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell],
    model=mace_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100 if SMOKE_TEST else 1000,
    temperature=2000,
    timestep=0.002,
)
final_atoms = ts.io.state_to_atoms(final_state)
final_fe_atoms_supercell = final_atoms[3]

print(f"Number of systems: {len(final_atoms)}")
print(f"Final energies: {[e.item() for e in final_state.energy]} eV")


# ============================================================================
# SECTION 5: Batched Integration with Trajectory Reporting
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Batched Integration with Trajectory Reporting")
print("=" * 70)

systems = (si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell)

filenames = [f"tmp/batch_traj_{i}.h5md" for i in range(len(systems))]
batch_reporter = TrajectoryReporter(
    filenames,
    state_frequency=100,
    prop_calculators=prop_calculators,
)

final_state = ts.integrate(
    system=systems,
    model=mace_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100 if SMOKE_TEST else 1000,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=batch_reporter,
)

final_energies_per_atom = []
for filename in filenames:
    with TorchSimTrajectory(filename) as traj:
        final_energy = traj.get_array("potential_energy")[-1]
        final_energies_per_atom.append(final_energy / len(traj.get_atoms(-1)))

print(f"Final energies per atom: {final_energies_per_atom}")


# ============================================================================
# SECTION 6: Structure Optimization
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Structure Optimization")
print("=" * 70)

final_state = ts.optimize(
    system=systems,
    model=mace_model,
    optimizer=ts.Optimizer.fire,
    max_steps=10 if SMOKE_TEST else 1000,
    init_kwargs=dict(cell_filter=ts.CellFilter.unit),
)

print(f"Final optimized energies: {[e.item() for e in final_state.energy]} eV")

# Add perturbations
rng = np.random.default_rng()
for system in systems:
    system.positions += rng.random(system.positions.shape) * 0.01


# ============================================================================
# SECTION 7: Optimization with Custom Convergence Criteria
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 7: Optimization with Custom Convergence")
print("=" * 70)

final_state = ts.optimize(
    system=systems,
    model=mace_model,
    optimizer=ts.Optimizer.fire,
    convergence_fn=lambda state, last_energy: last_energy - state.energy
    < 1e-6 * MetalUnits.energy,
    max_steps=10 if SMOKE_TEST else 1000,
    init_kwargs=dict(cell_filter=ts.CellFilter.unit),
)

print(f"Final converged energies: {[e.item() for e in final_state.energy]} eV")


# ============================================================================
# SECTION 8: Pymatgen Structure Support
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 8: Pymatgen Structure Support")
print("=" * 70)

lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
species = ["Si"] * 8
coords = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.0, 0.5, 0.5],
    [0.25, 0.75, 0.75],
    [0.5, 0.0, 0.5],
    [0.75, 0.25, 0.75],
    [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.25],
]
structure = Structure(lattice, species, coords)

final_state = ts.integrate(
    system=structure,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100 if SMOKE_TEST else 1000,
    temperature=2000,
    timestep=0.002,
)
final_structure = ts.io.state_to_structures(final_state)

print(f"Final structure type: {type(final_structure)}")
print(f"Final energy: {final_state.energy.item():.4f} eV")

print("\n" + "=" * 70)
print("High-level API examples completed!")
print("=" * 70)
