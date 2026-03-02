"""Phonon Calculations - DOS, band structure, and thermal properties.

This script demonstrates phonon calculations with:
- Phonon density of states (DOS)
- Phonon band structure
- Batched force constant calculations
- Integration with Phonopy

Note: This example requires phonopy, pymatviz, plotly, seekpath, and ase packages.
Visualization is disabled in CI mode.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.12",
#     "phonopy>=2.35",
#     "pymatviz>=0.17.1",
#     "plotly>=6.3.0",
#     "seekpath",
#     "ase",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from phonopy import Phonopy

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.telemetry import configure_logging, get_logger


configure_logging(log_file="6_phonons.log")
log = get_logger(name="6_phonons")


def require_not_none[T](value: T | None, message: str) -> T:
    """Return value or raise RuntimeError when missing."""
    if value is None:
        raise RuntimeError(message)
    return value


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

SMOKE_TEST = os.getenv("CI") is not None

# ============================================================================
# SECTION 1: Structure Relaxation for Phonons
# ============================================================================
log.info("=" * 70)
log.info("SECTION 1: Structure Relaxation")
log.info("=" * 70)

# Load the MACE model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# Create Silicon diamond structure
struct = bulk("Si", "diamond", a=5.431, cubic=True)
supercell_matrix = 2 * np.eye(3)  # 2x2x2 supercell for phonons
mesh = [20, 20, 20]  # Phonon mesh for DOS
max_steps = 10 if SMOKE_TEST else 300
displ = 0.01  # Atomic displacement for finite differences (Angstrom)

# Create MACE model
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Relax the structure
log.info("Relaxing structure...")
final_state = ts.optimize(
    system=struct,
    model=model,
    optimizer=ts.Optimizer.fire,
    max_steps=max_steps,
    init_kwargs=dict(
        cell_filter=ts.CellFilter.frechet,
        constant_volume=True,
        hydrostatic_strain=True,
    ),
)

log.info(f"Relaxation complete. Final energy: {final_state.energy.item():.4f} eV")


# ============================================================================
# SECTION 2: Phonon DOS Calculation
# ============================================================================
log.info("=" * 70)
log.info("SECTION 2: Phonon DOS Calculation")
log.info("=" * 70)

# Convert state to Phonopy atoms
atoms = ts.io.state_to_phonopy(final_state)[0]
ph = Phonopy(atoms, supercell_matrix)

# Generate displaced supercells for force constant calculation
log.info(f"Generating displacements (distance = {displ} Å)...")
ph.generate_displacements(distance=displ)
supercells = ph.supercells_with_displacements

if supercells is None:
    raise ValueError("No supercells generated - check Phonopy settings")

log.info(f"Number of displaced supercells: {len(supercells)}")

# Convert PhonopyAtoms to batched state for efficient computation
log.info("Calculating forces for displaced structures (batched)...")
state = ts.io.phonopy_to_state(supercells, device, dtype)
results = model(state)

# Extract forces and convert back to list of numpy arrays for Phonopy
n_atoms_per_supercell = [len(cell) for cell in supercells]
force_sets = []
start_idx = 0
for n_atoms in n_atoms_per_supercell:
    end_idx = start_idx + n_atoms
    force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
    start_idx = end_idx

# Produce force constants from forces
log.info("Producing force constants...")
ph.forces = force_sets
ph.produce_force_constants()

# Calculate phonon DOS
log.info(f"Calculating phonon DOS with mesh {mesh}...")
ph.run_mesh(mesh)
ph.run_total_dos()

# Get DOS data
dos = require_not_none(ph.total_dos, "Phonopy total_dos not computed")
freq_points = dos.frequency_points
dos_values = dos.dos
if freq_points is None or dos_values is None:
    raise RuntimeError("Phonopy total_dos has missing frequency_points or dos arrays")

log.info("Phonon DOS calculated:")
log.info(f"  Frequency range: {freq_points.min():.3f} to {freq_points.max():.3f} THz")
log.info(f"  DOS values range: {dos_values.min():.6f} to {dos_values.max():.6f}")

# Check for imaginary modes (negative frequencies)
if freq_points.min() < -0.1:
    log.info("  WARNING: Imaginary modes detected (freq < -0.1 THz)")
else:
    log.info("  No imaginary modes detected (all frequencies > -0.1 THz)")


# ============================================================================
# SECTION 3: Phonon Band Structure Calculation
# ============================================================================
log.info("=" * 70)
log.info("SECTION 3: Phonon Band Structure Calculation")
log.info("=" * 70)

try:
    import seekpath
    from ase import Atoms
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    # Convert to ASE atoms for seekpath
    ase_atoms = Atoms(
        symbols=atoms.symbols,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=True,
    )

    # Get high-symmetry path using seekpath
    log.info("Finding high-symmetry path...")
    seekpath_data = seekpath.get_path(
        (ase_atoms.cell, ase_atoms.get_scaled_positions(), ase_atoms.numbers)
    )

    # Extract high symmetry points and path
    points = seekpath_data["point_coords"]
    path = []
    for segment in seekpath_data["path"]:
        start_point = points[segment[0]]
        end_point = points[segment[1]]
        path.append([start_point, end_point])

    n_points = 51  # Points per segment
    q_pts, connections = get_band_qpoints_and_path_connections(path, npoints=n_points)

    log.info(f"Calculating phonon band structure ({len(q_pts)} q-points)...")
    ph.run_band_structure(q_pts, path_connections=connections)

    # Get band structure data
    bands_dict = ph.get_band_structure_dict()
    log.info("Phonon band structure calculated:")
    log.info(f"  Number of paths: {len(bands_dict['frequencies'])}")
    log.info(f"  Number of bands: {len(bands_dict['frequencies'][0][0])}")

    # Visualize if not in CI mode
    if not SMOKE_TEST:
        try:
            import pymatviz as pmv

            log.info("Generating phonon DOS plot...")
            total_dos = require_not_none(
                ph.total_dos, "Phonopy total_dos not available for plotting"
            )
            fig_dos = pmv.phonon_dos(total_dos)
            fig_dos.update_traces(line_width=3)
            fig_dos.update_layout(
                xaxis_title="Frequency (THz)",
                yaxis_title="DOS",
                font=dict(size=18),
                width=800,
                height=600,
            )
            fig_dos.show()

            log.info("Generating phonon band structure plot...")
            ph.auto_band_structure(plot=False)
            band_structure = require_not_none(
                ph.band_structure, "Phonopy band_structure not available for plotting"
            )
            fig_bands = pmv.phonon_bands(
                band_structure,
                line_kwargs={"width": 3},
            )
            fig_bands.update_layout(
                xaxis_title="Wave Vector",
                yaxis_title="Frequency (THz)",
                font=dict(size=18),
                width=800,
                height=600,
            )
            fig_bands.show()

        except ImportError:
            log.info("pymatviz not available, skipping visualization")
    else:
        log.info("Skipping visualization in CI mode")

except ImportError as e:
    log.info(f"Skipping band structure calculation: {e}")
    log.info("Install seekpath for band structure calculations")


# ============================================================================
# SECTION 4: Summary
# ============================================================================
log.info("=" * 70)
log.info("Summary")
log.info("=" * 70)
log.info("Structure: Silicon (diamond)")
log.info("Supercell: 2x2x2")
log.info(f"Number of displaced structures: {len(supercells)}")
log.info("Batched force calculation: Yes")
log.info("Phonon DOS calculated: Yes")
log.info(f"Frequency range: {freq_points.min():.3f} to {freq_points.max():.3f} THz")

log.info("=" * 70)
log.info("Phonon calculation examples completed!")
log.info("=" * 70)
