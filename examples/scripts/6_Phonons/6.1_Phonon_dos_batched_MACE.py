"""Phonon DOS calculation with MACE in batched mode."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "phonopy>=2.35",
#     "pymatviz[export-figs]>=0.15.1",
# ]
# ///

import numpy as np
import pymatviz as pmv
import torch
from mace.calculators.foundations_models import mace_mp
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from torch_sim.io import phonopy_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Load the raw model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Create NaCl structure using PhonopyAtoms
atoms = PhonopyAtoms(
    cell=np.eye(3) * 3,
    scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    symbols=["Na", "Cl"],
    pbc=True,
)

# Create Phonopy object with supercell matrix
supercell_matrix = 2 * np.eye(3)
ph = Phonopy(atoms, supercell_matrix)

# Generate displacements
ph.generate_displacements(distance=0.01)
supercells = ph.get_supercells_with_displacements()

# Convert PhonopyAtoms to state
state = phonopy_to_state(supercells, device, dtype)

# Create batched MACE model
atomic_numbers_list = [cell.get_atomic_numbers() for cell in supercells]
model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

# Run the model in batched mode
results = model(state)

# Print result keys and shapes
print(f"Result keys: {results.keys()}")
print(f"Forces shape: {results['forces'].shape}")

# Extract forces and convert back to list of numpy arrays for phonopy
n_atoms_per_supercell = [len(cell) for cell in supercells]
force_sets = []
start_idx = 0
for n_atoms in n_atoms_per_supercell:
    end_idx = start_idx + n_atoms
    force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
    start_idx = end_idx

# Produce force constants
ph.set_forces(force_sets)
ph.produce_force_constants()

# Set mesh for DOS calculation
mesh = [20, 20, 20]
ph.run_mesh(mesh)
ph.run_total_dos()
ph.save(filename="phonopy_params.yaml")

# Visualize phonon DOS
fig = pmv.phonon_dos(ph.total_dos)
pmv.save_fig(fig, "phonon_dos.png")
