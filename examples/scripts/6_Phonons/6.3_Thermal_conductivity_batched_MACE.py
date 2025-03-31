"""Calculate the thermal conductivity of a material using a batched MACE model."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "phono3py>=3.12",
#     "pymatgen>=2025.2.18",
# ]
# ///

# ruff: noqa: TC002
import time

import numpy as np
import torch
from mace.calculators.foundations_models import mace_mp
from phono3py import Phono3py
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import get_phonopy_structure

from torch_sim.io import phonopy_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts
import plotly.graph_objects as go

start_time = time.perf_counter()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Load the raw model from URL
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url, return_raw_model=True, default_dtype=dtype, device=device
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
    compute_forces=True,
    compute_stress=False,
    dtype=dtype,
    enable_cueq=False,
)

model_loading_time = time.perf_counter() - start_time
print(f"Model loading time: {model_loading_time}s")

# Convert PhonopyAtoms to state
state = phonopy_to_state(supercells, device, dtype)

# Run the model in batched mode
results = model(state)

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
ph3.mesh_numbers = [5, 5, 5]

# Initialize phonon-phonon interaction
ph3.init_phph_interaction()

# Run thermal conductivity calculation
ph3.run_thermal_conductivity()

kappa_time = time.perf_counter() - start_time
print(f"Kappa calculation time: {kappa_time}s")

# Average thermal conductivity
temperatures = ph3.thermal_conductivity.temperatures
kappa = ph3.thermal_conductivity.kappa[0]
kappa_av = np.mean(kappa[:,:3], axis=1)

# Plot temperatures vs kappa using plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=temperatures, y=kappa_av, mode='lines'))
fig.update_layout(
    title="Thermal Conductivity vs Temperature",
    xaxis_title="Temperature (K)",
    yaxis_title="Thermal Conductivity (W/mK)"
)
fig.write_image("thermal_conductivity.pdf")
