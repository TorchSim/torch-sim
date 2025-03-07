"""Calculate the thermal conductivity of a material using a batched MACE model."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "phono3py>=3.12",
#     "pymatgen>=2025.2.18",
# ]
# ///
import time

import numpy as np
import torch
from mace.calculators.foundations_models import mace_mp
from phono3py import Phono3py
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import get_phonopy_structure

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts


start_time = time.perf_counter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Load the raw model from URL
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Create MgO structure using pymatgen
lattice = Lattice.cubic(4.21)
mg_o_structure = Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

# Convert to phonopy atoms
unit_cell = get_phonopy_structure(mg_o_structure)
ph3 = Phono3py(unit_cell, supercell_matrix=[2, 2, 2], primitive_matrix="auto")
ph3.generate_displacements()
supercells = ph3.supercells_with_displacements

# Create batched MACE model
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

model_loading_time = time.perf_counter() - start_time
print(f"Model loading time: {model_loading_time}s")

# First we will create a concatenated positions array from all supercells
positions_numpy = np.concatenate([i.get_positions() for i in supercells])

# Then we stack cell vectors into a (n_supercells, 3, 3) array
cell_numpy = np.stack([i.get_cell() for i in supercells])

# Then we concatenate atomic numbers into a single array
atomic_numbers_numpy = np.concatenate([i.numbers for i in supercells])

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

# Run the model in batched mode
results = model(
    positions=positions, cell=cell, atomic_numbers=atomic_numbers, batch=batch
)

# Extract forces and convert back to list of numpy arrays for phonopy
n_atoms_per_supercell = [len(sc) for sc in supercells]
force_sets = []
start_idx = 0
for n_atoms in n_atoms_per_supercell:
    end_idx = start_idx + n_atoms
    force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
    start_idx = end_idx

forces_time = time.perf_counter() - start_time
print(f"Forces calculation time: {forces_time}s")

# Save phono3py yaml file
ph3.save("phono3py.yaml")
ph3yml = Phono3pyYaml()
ph3yml.read("phono3py.yaml")

# Update phono3py dataset
disp_dataset = ph3yml.dataset
ph3.dataset = disp_dataset

# Calculate force constants
ph3.forces = np.array(force_sets).reshape(-1, len(ph3.supercell), 3)
ph3.produce_fc3()

# Set mesh numbers
ph3.mesh_numbers = [3, 3, 3]

# Initialize phonon-phonon interaction
ph3.init_phph_interaction()

# Run thermal conductivity calculation
ph3.run_thermal_conductivity()

kappa_time = time.perf_counter() - start_time
print(f"Kappa calculation time: {kappa_time}s")
