# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "phonopy>=2.35",
#     "pymatviz>=0.15.1",
# ]
# ///
import numpy as np
import pymatviz as pmv
import torch
from mace.calculators.foundations_models import mace_mp
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Load the raw model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
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
)

# Create Phonopy object with supercell matrix
supercell_matrix = 2 * np.eye(3)
ph = Phonopy(atoms, supercell_matrix)

# Generate displacements
ph.generate_displacements(distance=0.01)
supercells = ph.get_supercells_with_displacements()

# Create batched MACE model
atomic_numbers_list = [i.get_atomic_numbers() for i in supercells]
model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=True,
    compute_force=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

# First we will create a concatenated positions array from all supercells
positions_numpy = np.concatenate([i.get_positions() for i in supercells])

# Then we stack cell vectors into a (n_supercells, 3, 3) array
cell_numpy = np.stack([i.get_cell() for i in supercells])

# Then we concatenate atomic numbers into a single array
atomic_numbers_numpy = np.concatenate([i.get_atomic_numbers() for i in supercells])

# Then we convert to tensors
positions = torch.tensor(positions_numpy, device=device, dtype=dtype)
cell = torch.tensor(cell_numpy, device=device, dtype=dtype)
atomic_numbers = torch.tensor(atomic_numbers_numpy, device=device, dtype=torch.int)

# Create a batch index array to track which atoms belong to which supercell
atoms_per_batch = torch.tensor(
    [len(i) for i in supercells], device=device, dtype=torch.int
)
batch = torch.repeat_interleave(
    torch.arange(len(atoms_per_batch), device=device), atoms_per_batch
)

# Print shapes for verification
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")
print(f"Atomic numbers: {atomic_numbers.shape}")
print(f"Batch: {batch.shape}")

# Run the model in batched mode
results = model(
    positions=positions, cell=cell, atomic_numbers=atomic_numbers, batch=batch
)

# Print result keys and shapes
print(f"Result keys: {results.keys()}")
print(f"Forces shape: {results['forces'].shape}")

# Extract forces and convert back to list of numpy arrays for phonopy
n_atoms_per_supercell = [len(sc) for sc in supercells]
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
fig.write_image("phonon_dos.png")
