"""Calculate Phonon DOS and band structure with MACE in batched mode."""

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
import numpy as np
import pymatviz as pmv
import seekpath
import torch
from ase import Atoms
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from phonopy import Phonopy
from phonopy.phonon.band_structure import (
    get_band_qpoints_and_path_connections,
    get_band_qpoints_by_seekpath,
)

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls


def get_qpts_and_connections(
    ase_atoms: Atoms,
    n_points: int = 101,
) -> tuple[list[list[float]], list[bool]]:
    """Get the high symmetry points and path connections for the band structure."""
    # Define seekpath data
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
    qpts, connections = get_band_qpoints_and_path_connections(path, npoints=n_points)

    return qpts, connections


def get_labels_qpts(ph: Phonopy, n_points: int = 101) -> tuple[list[str], list[bool]]:
    """Get the labels and coordinates of qpoints for the phonon band structure."""
    # Get labels and coordinates for high-symmetry points
    _, qpts_labels, connections = get_band_qpoints_by_seekpath(
        ph.primitive, npoints=n_points, is_const_interval=True
    )
    connections = [True, *connections]
    connections[-1] = True
    qpts_labels_connections = []
    idx = 0
    for connection in connections:
        if connection:
            qpts_labels_connections.append(qpts_labels[idx])
            idx += 1
        else:
            qpts_labels_connections.append(f"{qpts_labels[idx]}|{qpts_labels[idx + 1]}")
            idx += 2

    qpts_labels_arr = [
        q_label.replace("\\Gamma", "Γ")
        .replace("$", "")
        .replace("\\", "")
        .replace("mathrm", "")
        .replace("{", "")
        .replace("}", "")
        for q_label in qpts_labels_connections
    ]
    bands_dict = ph.get_band_structure_dict()
    npaths = len(bands_dict["frequencies"])
    qpts_coord = [bands_dict["distances"][n][0] for n in range(npaths)] + [
        bands_dict["distances"][-1][-1]
    ]

    return qpts_labels_arr, qpts_coord


# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Load the raw model
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).lstrip("torch."),
    device=str(device),
)

# Structure and input parameters
struct = bulk("Si", "diamond", a=5.431, cubic=True)  # ASE structure
supercell_matrix = 2 * np.eye(3)  # supercell matrix for phonon calculation
mesh = [20, 20, 20]  # Phonon mesh
max_steps = 300  # number of relaxation steps
displ = 0.01  # atomic displacement for phonons (in Angstrom)

# Relax atomic positions
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)
final_state = ts.optimize(
    system=struct,
    model=model,
    optimizer=ts.OptimFlavor.fire,
    cell_filter=ts.CellFilter.frechet,
    max_steps=max_steps,
    constant_volume=True,
    hydrostatic_strain=True,
)

# Define atoms and Phonopy object
atoms = ts.io.state_to_phonopy(final_state)[0]
ph = Phonopy(atoms, supercell_matrix)

# Generate FC2 displacements
ph.generate_displacements(distance=displ)
supercells = ph.supercells_with_displacements
if supercells is None:
    raise ValueError("supercells cannot be None")

# Convert PhonopyAtoms to state
state = ts.io.phonopy_to_state(supercells, device, dtype)
results = model(state)

# Extract forces and convert back to list of numpy arrays for phonopy
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

# Set mesh for DOS calculation
ph.run_mesh(mesh)
ph.run_total_dos()

# Calculate phonon band structure
ase_atoms = Atoms(
    symbols=atoms.symbols,
    positions=atoms.positions,
    cell=atoms.cell,
    pbc=True,
)
q_pts, connections = get_qpts_and_connections(ase_atoms)
ph.run_band_structure(q_pts, path_connections=connections)

# Define axis style for plots
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

# Plot phonon DOS
fig = pmv.phonon_dos(ph.total_dos)
fig.update_traces(line_width=3)
fig.update_layout(
    xaxis_title="Frequency (THz)",
    yaxis_title="DOS",
    font=dict(size=24),
    xaxis=axis_style,
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor="white",
)
fig.show()

# Plot phonon band structure
ph.auto_band_structure(plot=False)
fig = pmv.phonon_bands(
    ph.band_structure,
    line_kwargs={"width": 3},
)
qpts_labels, qpts_coord = get_labels_qpts(ph)
for q_pt in qpts_coord:
    fig.add_vline(x=q_pt, line_dash="dash", line_color="black", line_width=2, opacity=1)
fig.update_layout(
    xaxis_title="Wave Vector",
    yaxis_title="Frequency (THz)",
    font=dict(size=24),
    xaxis=dict(
        tickmode="array",
        tickvals=qpts_coord,
        ticktext=qpts_labels,
        showgrid=False,
        zeroline=False,
        linecolor="black",
        showline=True,
        ticks="inside",
        mirror=True,
        linewidth=3,
        tickwidth=3,
        ticklen=10,
    ),
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor="white",
)
fig.show()
