"""Calculate quasi-harmonic thermal properties batching
   over FC2 calculations with MACE"""

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
from torch_sim.io import phonopy_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.optimizers import frechet_cell_fire
from ase.build import bulk
from torch_sim import optimize
from torch_sim.io import state_to_phonopy
from phonopy.api_qha import PhonopyQHA
import plotly.graph_objects as go
import tqdm
from torch_sim.runners import generate_force_convergence_fn
from torch_sim import TrajectoryReporter, TorchSimTrajectory
from phonopy.structure.atoms import PhonopyAtoms
import os

def print_relax_info(trajectory_file, device):
    with TorchSimTrajectory(trajectory_file) as traj:
        energies = traj.get_array("potential_energy")
        forces = traj.get_array("forces")
        if isinstance(forces, np.ndarray):
            forces = torch.tensor(forces, device=device)
        max_force = torch.max(torch.abs(forces), dim=1).values
    for i in range(max_force.shape[0]):
        print(f"Step {i}: Max force = {torch.max(max_force[i]).item():.4e} eV/A, Energy = {energies[i].item():.4f} eV")
    os.remove(trajectory_file)

# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# Load the raw model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)
model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Structure and input parameters
struct = bulk('Si', 'diamond', a=5.431, cubic=True) # ASE structure
supercell_matrix = 2 * np.eye(3) # supercell matrix for phonon calculation
mesh = [20, 20, 20] # Phonon mesh
Nrelax = 300 # number of relaxation steps
fmax = 1e-3 # force convergence
displ = 0.05 # atomic displacement for phonons (in Angstrom)
temperatures = np.arange(0, 1410, 10) # temperature range for quasi-harmonic calculation
length_factor = np.linspace(0.85, 1.15, 15) # length factor for quasi-harmonic calculation

# Relax structure
converge_max_force = generate_force_convergence_fn(force_tol=fmax)
trajectory_file = "qha.h5"
reporter = TrajectoryReporter(
    trajectory_file,
    state_frequency=0,
    prop_calculators={
        1: {"potential_energy": lambda state: state.energy, 
             "forces": lambda state: state.forces},
    },
)
final_state = optimize(
    system=struct,
    model=model,
    optimizer=frechet_cell_fire,
    constant_volume=True,
    hydrostatic_strain=True,
    max_steps=Nrelax,
    convergence_fn=converge_max_force,
    trajectory_reporter=reporter,
)
print(f"\nStructural relaxation")
print_relax_info(trajectory_file, device)

# Define atoms and Phonopy object
atoms = state_to_phonopy(final_state)[0]
relaxed_struct = atoms.copy()

# Initialize arrays to store results
volumes = []
energies = []
free_energies = []
entropies = []
heat_capacities = []
all_thermal_props = []

# Loop over different volumes using the length factor
for i, factor in enumerate(length_factor):
    
    print(f"\n({i+1}/{len(length_factor)}), Scale factor = {factor:.3f}")

    # Scale the original structure
    scaled_cell = relaxed_struct.cell * factor
    scaled_struct = PhonopyAtoms(
        cell=scaled_cell,
        scaled_positions=relaxed_struct.scaled_positions,
        symbols=relaxed_struct.symbols
    )
    
    # Relax atomic positions at this volume
    scaled_state = optimize(
        system=scaled_struct,
        model=model,
        optimizer=frechet_cell_fire,
        constant_volume=True,
        hydrostatic_strain=True,
        max_steps=Nrelax,
        convergence_fn=converge_max_force,
        trajectory_reporter=reporter,
    )
    print_relax_info(trajectory_file, device)
    
    # Define scaled Phonopy object
    scaled_atoms = state_to_phonopy(scaled_state)[0]
    ph = Phonopy(
        scaled_atoms, 
        supercell_matrix=supercell_matrix,
        primitive_matrix="auto",
    )
    
    # Calculate FC2
    ph.generate_displacements(distance=displ)
    supercells = ph.supercells_with_displacements
    state = phonopy_to_state(supercells, device, dtype)
    results = model(state)
    n_atoms_per_supercell = [len(cell) for cell in supercells]
    force_sets = []
    start_idx = 0
    for n_atoms in n_atoms_per_supercell:
        end_idx = start_idx + n_atoms
        force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
        start_idx = end_idx
    ph.forces = force_sets
    ph.produce_force_constants()
    
    # Calculate thermal properties
    ph.run_mesh(mesh)
    ph.run_thermal_properties(
        t_min=temperatures[0],
        t_max=temperatures[-1],
        t_step=int((temperatures[-1]-temperatures[0])/(len(temperatures)-1)),
    )
    
    # Store volume, energy, entropies, heat capacities
    thermal_props = ph.get_thermal_properties_dict()
    all_thermal_props.append(thermal_props)
    n_unit_cells = np.prod(np.diag(supercell_matrix))
    cell = scaled_atoms.cell
    volume = np.linalg.det(cell)
    volumes.append(volume)

    # Get the energy - handle multi-element tensor case
    energies.append(results["energy"][0].item() / n_unit_cells)
    free_energies.append(thermal_props['free_energy'])
    entropies.append(thermal_props['entropy'])
    heat_capacities.append(thermal_props['heat_capacity'])

# run QHA
qha = PhonopyQHA(
    volumes=volumes,
    electronic_energies=np.tile(energies, (len(temperatures), 1)),
    temperatures=temperatures,
    free_energy=np.array(free_energies).T,
    cv=np.array(heat_capacities).T,
    entropy= np.array(entropies).T,
    eos='vinet'
)

# Axis style
axis_style = dict(
    showgrid=False, 
    zeroline=False, 
    linecolor='black',
    showline=True,
    ticks="inside",
    mirror=True,
    linewidth=3,
    tickwidth=3,
    ticklen=10,
)

# Plot thermal expansion vs temperature
fig = go.Figure()
fig.add_trace(go.Scatter(x=temperatures, y=qha.thermal_expansion, mode='lines', line=dict(width=4)))
fig.update_layout(
    xaxis_title="Temperature (K)",
    yaxis_title="Thermal Expansion (1/K)",
    font=dict(size=24),
    xaxis=axis_style,
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor='white'
)
fig.write_image("thermal_expansion.pdf")

# Plot bulk modulus vs temperature
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=temperatures, 
    y=qha.bulk_modulus_temperature, 
    mode='lines', 
    line=dict(width=4)
))
fig.update_layout(
    xaxis_title="Temperature (K)",
    yaxis_title="Bulk Modulus (GPa)",
    font=dict(size=24),
    xaxis=axis_style,
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor='white'
)
fig.write_image("bulk_modulus.pdf")