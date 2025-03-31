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
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import get_phonopy_structure

from torch_sim.io import phonopy_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts

from torch_sim import optimize
from torch_sim.optimizers import frechet_cell_fire
from torch_sim.io import state_to_phonopy
import plotly.graph_objects as go
from ase.build import bulk
import tqdm

start_time = time.perf_counter()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Load the raw model from URL
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url, return_raw_model=True, default_dtype=dtype, device=device
)

# Structure and input parameters
struct = bulk('Si', 'diamond', a=5.431, cubic=True) # ASE structure
mesh = [5, 5, 5] # Phonon mesh
supercell_matrix = [2,2,2] # supercell matrix for phonon calculation
supercell_matrix_fc2 = [2,2,2] # supercell matrix for FC2 calculation
Nrelax = 300 # number of relaxation steps
displ = 0.05 # atomic displacement for phonons (in Angstrom)
conductivity_type = "wigner" # "wigner", "kubo"
temperatures = np.arange(0, 1600, 100) # temperature range for thermal conductivity calculation

# Relax atomic positions
model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)
final_state = optimize(
    system=struct,
    model=model,
    optimizer=frechet_cell_fire,
    constant_volume=True,
    hydrostatic_strain=True,
    max_steps=Nrelax,
)

# Phono3py object
phonopy_atoms = state_to_phonopy(final_state)[0]
ph3 = Phono3py(
    phonopy_atoms,
    supercell_matrix=supercell_matrix,
    primitive_matrix="auto",
    phonon_supercell_matrix=supercell_matrix_fc2
)

# FC2 displacements
ph3.generate_fc2_displacements(distance=displ)
supercells_fc2 = ph3.phonon_supercells_with_displacements
state = phonopy_to_state(supercells_fc2, device, dtype)
results = model(state)
n_atoms_per_supercell = [len(sc) for sc in supercells_fc2]
force_sets = []
start_idx = 0
for n_atoms in tqdm.tqdm(n_atoms_per_supercell, desc="FC2 calculations"):
    end_idx = start_idx + n_atoms
    force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
    start_idx = end_idx
ph3.phonon_forces = np.array(force_sets).reshape(-1, len(ph3.phonon_supercell), 3)
ph3.produce_fc2(symmetrize_fc2=True)

# FC3 displacements
ph3.generate_displacements(distance=displ)
supercells_fc3 = ph3.supercells_with_displacements
state = phonopy_to_state(supercells_fc3, device, dtype)
results = model(state)
n_atoms_per_supercell = [len(sc) for sc in supercells_fc3]
force_sets = []
start_idx = 0
for n_atoms in tqdm.tqdm(n_atoms_per_supercell, desc="FC3 calculations"):
    end_idx = start_idx + n_atoms
    force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
    start_idx = end_idx
ph3.forces = np.array(force_sets).reshape(-1, len(ph3.supercell), 3)
ph3.produce_fc3(symmetrize_fc3r=True)

# Run thermal conductivity calculation
ph3.mesh_numbers = mesh
ph3.init_phph_interaction(symmetrize_fc3q=False)
ph3.run_thermal_conductivity(
    is_isotope=True,
    temperatures=temperatures,
    conductivity_type=conductivity_type,
    boundary_mfp=1e6,
)
temperatures = ph3.thermal_conductivity.temperatures
if conductivity_type == "wigner":
    kappa = ph3.thermal_conductivity.kappa_TOT_RTA[0]
else:
    kappa = ph3.thermal_conductivity.kappa[0]

# Average thermal conductivity
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


