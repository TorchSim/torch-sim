"""Calculate quasi-harmonic thermal properties in batched mode."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "phonopy>=2.35",
#     "pymatviz[export-figs]>=0.15.1",
#     "seekpath",
#     "ase",
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

from torch_sim.optimizers import frechet_cell_fire
from ase import Atoms
import seekpath
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections, get_band_qpoints_by_seekpath
from ase.build import bulk
from torch_sim import optimize
from torch_sim.io import state_to_atoms, state_to_phonopy
from phonopy.api_qha import PhonopyQHA
import plotly.graph_objects as go
import tqdm

# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Load the raw model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Structure and input parameters
struct = bulk('Si', 'diamond', a=5.431, cubic=True) # ASE structure
supercell_matrix = 2 * np.eye(3) # supercell matrix for phonon calculation
mesh = [20, 20, 20] # Phonon mesh
Nrelax = 300 # number of relaxation steps
displ = 0.05 # atomic displacement for phonons (in Angstrom)
temperatures = np.arange(0, 1600, 100) # temperature range for quasi-harmonic calculation
length_factor = np.linspace(0.85, 1.15, 15) # length factor for quasi-harmonic calculation

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

# Define atoms and Phonopy object
atoms = state_to_phonopy(final_state)[0]

# Initialize arrays to store results
volumes = []
energies = []
free_energies = []
entropies = []
heat_capacities = []
all_thermal_props = []

# Loop over different volumes using the length factor
for factor in tqdm.tqdm(length_factor, desc="Processing volumes"):
    
    # Scale the original structure
    scaled_struct = bulk('Si', 'diamond', a=5.431*factor, cubic=True)
    
    # Relax atomic positions at this volume
    scaled_state = optimize(
        system=scaled_struct,
        model=model,
        optimizer=frechet_cell_fire,
        constant_volume=True,
        hydrostatic_strain=True,
        max_steps=Nrelax,
    )
    
    # 2) Define PhonopyAtoms and calculate force constants
    scaled_atoms = state_to_phonopy(scaled_state)[0]
    ph = Phonopy(scaled_atoms, supercell_matrix)
    
    # Generate displacements
    ph.generate_displacements(distance=displ)
    supercells = ph.supercells_with_displacements
    
    # Convert to state and calculate forces
    state = phonopy_to_state(supercells, device, dtype)
    results = model(state)
    
    # Extract forces
    n_atoms_per_supercell = [len(cell) for cell in supercells]
    force_sets = []
    start_idx = 0
    for n_atoms in n_atoms_per_supercell:
        end_idx = start_idx + n_atoms
        force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
        start_idx = end_idx
    
    # Produce force constants
    ph.forces = force_sets
    ph.produce_force_constants()
    
    # Calculate thermal properties
    ph.run_mesh(mesh)
    ph.run_thermal_properties(
        t_min=temperatures[0],
        t_max=temperatures[-1],
        t_step=int((temperatures[-1]-temperatures[0])/(len(temperatures)-1)),
    )

    thermal_props = ph.get_thermal_properties_dict()
    all_thermal_props.append(thermal_props)
    
    # Store volume
    cell = scaled_atoms.get_cell()
    volume = np.linalg.det(cell)
    volumes.append(volume)

    energies.append(results["energy"].item())
    
    # Store properties for plotting
    free_energies.append(thermal_props['free_energy'])
    entropies.append(thermal_props['entropy'])
    heat_capacities.append(thermal_props['heat_capacity'])

# Convert lists to numpy arrays for easier manipulation
volumes = np.array(volumes)
free_energies = np.array(free_energies)
entropies = np.array(entropies)
heat_capacities = np.array(heat_capacities)

# run QHA
qha = PhonopyQHA(
    volumes=volumes,
    electronic_energies=np.tile(energies, (len(temperatures), 1)),
    temperatures=temperatures,
    free_energy=free_energies.T,
    cv=heat_capacities.T,
    entropy=entropies.T,
    eos='vinet'
)

# Plot thermal expansion vs temperature
fig = go.Figure()
fig.add_trace(go.Scatter(x=temperatures, y=qha.thermal_expansion, mode='lines'))
fig.update_layout(
    title="Thermal Expansion vs Temperature",
    xaxis_title="Temperature (K)",
    yaxis_title="Thermal Expansion (1/K)"
)
fig.write_image("thermal_expansion.pdf")