"""Calculate the thermal conductivity of a material using a MACE model."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
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

from torchsim.models.mace import UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts


# Set device and data type
start_time = time.perf_counter()
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

# Create MgO structure using pymatgen
# MgO has rock salt structure with lattice constant ~4.21 Å
lattice = Lattice.cubic(4.21)
mg_o_structure = Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

# Convert to phonopy atoms
unit_cell = get_phonopy_structure(mg_o_structure)
ph3 = Phono3py(unit_cell, supercell_matrix=[2, 2, 2], primitive_matrix="auto")
ph3.generate_displacements()
supercells = ph3.supercells_with_displacements

# Create unbatched MACE model
model = UnbatchedMaceModel(
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

# Calculate forces for each displacement
set_of_forces: list[np.ndarray] = []
for displacement in ph3.supercells_with_displacements:
    positions = torch.tensor(displacement.get_positions(), device=device, dtype=dtype)
    cell = torch.tensor(displacement.get_cell(), device=device, dtype=dtype)
    atomic_numbers = torch.tensor(displacement.numbers, device=device, dtype=torch.int)
    results = model(positions=positions, cell=cell, atomic_numbers=atomic_numbers)
    set_of_forces.append(results["forces"].detach().cpu().numpy())

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
ph3.forces = np.array(set_of_forces).reshape(-1, len(ph3.supercell), 3)
ph3.produce_fc3()

# Set mesh numbers
ph3.mesh_numbers = [3, 3, 3]

# Initialize phonon-phonon interaction
ph3.init_phph_interaction()

# Run thermal conductivity calculation
ph3.run_thermal_conductivity()

kappa_time = time.perf_counter() - start_time
print(f"Kappa calculation time: {kappa_time}s")
