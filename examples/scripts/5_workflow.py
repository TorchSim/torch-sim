"""Advanced Workflows - Complex simulation workflows and utilities.

This script demonstrates:
- In-flight autobatching for efficient optimization
- Elastic constant calculations
- Force convergence utilities
"""

# /// script
# dependencies = ["mace-torch>=0.3.12", "matbench-discovery>=1.3.1"]
# ///

import os
import time

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.elastic import get_bravais_type
from torch_sim.models.mace import MaceModel, MaceUrls


# Set device
SMOKE_TEST = os.getenv("CI") is not None
device = torch.device(
    "cpu" if SMOKE_TEST else ("cuda" if torch.cuda.is_available() else "cpu")
)
dtype = torch.float32

print(f"Running on device: {device}")


# ============================================================================
# SECTION 1: In-Flight Autobatching Workflow
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: In-Flight Autobatching Workflow")
print("=" * 70)

print("Loading MACE model...")
mace = mace_mp(model=MaceUrls.mace_mpa_medium, return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=dtype,
    compute_forces=True,
)

# Optimization parameters
fmax = 0.05  # Force convergence threshold
max_atoms_in_batch = 50 if SMOKE_TEST else 8_000

# Load or create structures
if not SMOKE_TEST:
    try:
        from matbench_discovery.data import DataFiles, ase_atoms_from_zip

        n_structures_to_relax = 100
        print(f"Loading {n_structures_to_relax:,} structures from WBM dataset...")
        ase_atoms_list = ase_atoms_from_zip(
            DataFiles.wbm_initial_atoms.path, limit=n_structures_to_relax
        )
    except ImportError:
        print("matbench_discovery not available, using synthetic structures...")
        n_structures_to_relax = 10
        ase_atoms_list = []
        for _ in range(n_structures_to_relax):
            atoms = bulk("Al", "hcp", a=4.05).repeat((2, 2, 2))
            atoms.positions += 0.1 * np.random.randn(*atoms.positions.shape)
            ase_atoms_list.append(atoms)
else:
    n_structures_to_relax = 2
    print(f"Loading {n_structures_to_relax:,} test structures...")
    al_atoms = bulk("Al", "hcp", a=4.05)
    al_atoms.positions += 0.1 * np.random.randn(*al_atoms.positions.shape)
    fe_atoms = bulk("Fe", "bcc", a=2.86).repeat((2, 2, 2))
    fe_atoms.positions += 0.1 * np.random.randn(*fe_atoms.positions.shape)
    ase_atoms_list = [al_atoms, fe_atoms]

# Initialize first batch
fire_states = ts.fire_init(
    state=ts.io.atoms_to_state(atoms=ase_atoms_list, device=device, dtype=dtype),
    model=mace_model,
    cell_filter=ts.CellFilter.frechet,
)

# Create autobatcher
batcher = ts.autobatching.InFlightAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=1000 if SMOKE_TEST else None,
)

# Create convergence function
converge_max_force = ts.runners.generate_force_convergence_fn(force_tol=fmax)

start_time = time.perf_counter()

# Main optimization loop with autobatching
batcher.load_states(fire_states)
all_completed_states, convergence_tensor, state = [], None, None

print("\nStarting optimization with autobatching...")
batch_count = 0
while (result := batcher.next_batch(state, convergence_tensor))[0] is not None:
    state, completed_states = result
    batch_count += 1
    print(f"Batch {batch_count}: Optimizing {state.n_systems} structures")

    all_completed_states.extend(completed_states)
    if all_completed_states:
        print(f"Total completed structures: {len(all_completed_states)}")

    # Run optimization steps for this batch
    for _step in range(10):
        state = ts.fire_step(state=state, model=mace_model)

    # Check convergence
    convergence_tensor = converge_max_force(state, last_energy=None)

# Add final completed states
all_completed_states.extend(result[1])

end_time = time.perf_counter()
total_time = end_time - start_time

print("\nOptimization complete!")
print(f"Total completed structures: {len(all_completed_states)}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per structure: {total_time / len(all_completed_states):.2f} seconds")


# ============================================================================
# SECTION 2: Elastic Constants Calculation
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Elastic Constants Calculation")
print("=" * 70)

# Use higher precision for elastic constants
dtype_elastic = torch.float64

loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    enable_cueq=False,
    device=str(device),
    default_dtype=str(dtype_elastic).removeprefix("torch."),
    return_raw_model=True,
)

# Create FCC Copper structure
struct = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))

model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype_elastic,
    enable_cueq=False,
)

# Target force tolerance
fmax = 1e-3

# Relax positions and cell
state = ts.io.atoms_to_state(atoms=struct, device=device, dtype=dtype_elastic)
state = ts.fire_init(
    state=state, model=model, scalar_pressure=0.0, cell_filter=ts.CellFilter.frechet
)

print("\nRelaxing structure...")
unit_conv = ts.units.UnitConversion
for step in range(300):
    pressure = -torch.trace(state.stress.squeeze()) / 3 * unit_conv.eV_per_Ang3_to_GPa
    current_fmax = torch.max(torch.abs(state.forces.squeeze()))

    if step % 50 == 0:
        print(
            f"Step {step}, Energy: {state.energy.item():.4f} eV, "
            f"Pressure: {pressure.item():.4f} GPa, "
            f"Fmax: {current_fmax.item():.4f} eV/Å"
        )

    if current_fmax < fmax and abs(pressure) < 1e-2:
        print(f"Converged at step {step}")
        break

    state = ts.fire_step(state=state, model=model)

# Get bravais type
bravais_type = get_bravais_type(state)
print(f"\nBravais lattice type: {bravais_type}")

# Calculate elastic tensor
print("\nCalculating elastic tensor...")
elastic_tensor = ts.elastic.calculate_elastic_tensor(
    state=state, model=model, bravais_type=bravais_type
)

# Convert to GPa
elastic_tensor = elastic_tensor * unit_conv.eV_per_Ang3_to_GPa

# Calculate elastic moduli
bulk_modulus, shear_modulus, poisson_ratio, pugh_ratio = (
    ts.elastic.calculate_elastic_moduli(elastic_tensor)
)

# Print results
print("\nElastic tensor (GPa):")
elastic_tensor_np = elastic_tensor.cpu().numpy()
for row in elastic_tensor_np:
    print("  " + "  ".join(f"{val:10.4f}" for val in row))

print("\nElastic moduli:")
print(f"  Bulk modulus (GPa): {bulk_modulus:.4f}")
print(f"  Shear modulus (GPa): {shear_modulus:.4f}")
print(f"  Poisson's ratio: {poisson_ratio:.4f}")
print(f"  Pugh's ratio (K/G): {pugh_ratio:.4f}")

# Interpret Pugh's ratio
if pugh_ratio > 1.75:
    material_type = "ductile"
else:
    material_type = "brittle"
print(f"  Material behavior: {material_type}")

print("\n" + "=" * 70)
print("Workflow examples completed!")
print("=" * 70)
