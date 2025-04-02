"""Calculate the Wigner thermal conductivity batching over
FC2 and FC3 calculations with MACE.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "phono3py>=3.12",
#     "pymatgen>=2025.2.18",
# ]
# ///

import os
import time

import numpy as np
import plotly.graph_objects as go
import torch
import tqdm
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from phono3py import Phono3py

from torch_sim import TorchSimTrajectory, TrajectoryReporter, optimize
from torch_sim.io import phonopy_to_state, state_to_phonopy
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.optimizers import frechet_cell_fire
from torch_sim.runners import generate_force_convergence_fn


def print_relax_info(trajectory_file: str, device: torch.device) -> None:
    """Print relaxation information from trajectory file.

    Args:
        trajectory_file: Path to the trajectory file
        device: Torch device for calculations
    """
    with TorchSimTrajectory(trajectory_file) as traj:
        energies = traj.get_array("potential_energy")
        forces = traj.get_array("forces")
        if isinstance(forces, np.ndarray):
            forces = torch.tensor(forces, device=device)
        max_force = torch.max(torch.abs(forces), dim=1).values
    for i in range(max_force.shape[0]):
        print(
            f"Step {i}: Max force = {torch.max(max_force[i]).item():.4e} eV/A, "
            f"Energy = {energies[i].item():.4f} eV"
        )
    os.remove(trajectory_file)


start_time = time.perf_counter()
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# Load the raw model from URL
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url, return_raw_model=True, default_dtype=dtype, device=device
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
struct = bulk("Si", "diamond", a=5.431, cubic=True)  # ASE structure
mesh = [8, 8, 8]  # Phonon mesh
supercell_matrix = [
    1,
    1,
    1,
]  # supercell matrix for phonon calculation (use larger supercell for better accuracy)
supercell_matrix_fc2 = [2, 2, 2]  # supercell matrix for FC2 calculation
Nrelax = 300  # number of relaxation steps
fmax = 1e-3  # force convergence
displ = 0.05  # atomic displacement for phonons (in Angstrom)
conductivity_type = "wigner"  # "wigner", "kubo"
temperatures = np.arange(
    0, 1600, 10
)  # temperature range for thermal conductivity calculation

# Relax structure
converge_max_force = generate_force_convergence_fn(force_tol=fmax)
trajectory_file = "anha.h5"
reporter = TrajectoryReporter(
    trajectory_file,
    state_frequency=0,
    prop_calculators={
        1: {
            "potential_energy": lambda state: state.energy,
            "forces": lambda state: state.forces,
        },
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
print_relax_info(trajectory_file, device)

# Phono3py object
phonopy_atoms = state_to_phonopy(final_state)[0]
ph3 = Phono3py(
    phonopy_atoms,
    supercell_matrix=supercell_matrix,
    primitive_matrix="auto",
    phonon_supercell_matrix=supercell_matrix_fc2,
)

# Calculate FC2
ph3.generate_fc2_displacements(distance=displ)
supercells_fc2 = ph3.phonon_supercells_with_displacements
state = phonopy_to_state(supercells_fc2, device, dtype)
results = model(state)
n_atoms_per_supercell = [len(sc) for sc in supercells_fc2]
force_sets = []
start_idx = 0
for n_atoms in tqdm.tqdm(n_atoms_per_supercell, desc="FC2"):
    end_idx = start_idx + n_atoms
    force_sets.append(results["forces"][start_idx:end_idx].detach().cpu().numpy())
    start_idx = end_idx
ph3.phonon_forces = np.array(force_sets).reshape(-1, len(ph3.phonon_supercell), 3)
ph3.produce_fc2(symmetrize_fc2=True)

# Calculate FC3
ph3.generate_displacements(distance=displ)
supercells_fc3 = ph3.supercells_with_displacements
state = phonopy_to_state(supercells_fc3, device, dtype)
results = model(state)
n_atoms_per_supercell = [len(sc) for sc in supercells_fc3]
force_sets = []
start_idx = 0
for n_atoms in tqdm.tqdm(n_atoms_per_supercell, desc="FC3"):
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
kappa_av = np.mean(kappa[:, :3], axis=1)

# Print kappa at 300K
idx_300k = np.abs(temperatures - 300).argmin()
print(
    f"\nThermal conductivity at {temperatures[idx_300k]:.1f} K: "
    f"{kappa_av[idx_300k]:.2f} W/mK"
)

# Axis style
axis_style = dict(
    showgrid=False,
    zeroline=False,
    linecolor="black",
    showline=True,
    ticks="inside",
    mirror=True,
    linewidth=3,
    tickwidth=3,
    ticklen=10,
)

# Plot thermal expansion vs temperature
fig = go.Figure()
fig.add_trace(go.Scatter(x=temperatures, y=kappa_av, mode="lines", line=dict(width=4)))
fig.update_layout(
    xaxis_title="Temperature (K)",
    yaxis_title="Thermal Conductivity (W/mK)",
    font=dict(size=24),
    xaxis=axis_style,
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor="white",
)
fig.show()
