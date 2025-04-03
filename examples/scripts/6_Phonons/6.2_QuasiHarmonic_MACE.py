"""Calculate quasi-harmonic thermal properties batching over
different volumes and FC2 calculations with MACE.
"""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "phonopy>=2.35",
#     "pymatviz[export-figs]>=0.15.1",
# ]
# ///

import os

import numpy as np
import plotly.graph_objects as go
import torch
from ase import Atoms
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from phonopy import Phonopy
from phonopy.api_qha import PhonopyQHA
from phonopy.structure.atoms import PhonopyAtoms

import torch_sim as ts


def get_relaxed_structure(
    struct: Atoms,
    model: torch.nn.Module | None,
    Nrelax: int = 300,
    fmax: float = 1e-3,
    *,
    use_autobatcher: bool = False,
) -> ts.state.SimState:
    """Get relaxed structure.

    Args:
        struct: ASE structure
        model: MACE model
        Nrelax: Maximum number of relaxation steps
        fmax: Force convergence criterion
        use_autobatcher: Whether to use automatic batching

    Returns:
        SimState: Relaxed structure
    """
    trajectory_file = "traj.h5"
    reporter = ts.TrajectoryReporter(
        trajectory_file,
        state_frequency=0,
        prop_calculators={
            1: {
                "potential_energy": lambda state: state.energy,
                "forces": lambda state: state.forces,
            },
        },
    )
    converge_max_force = ts.runners.generate_force_convergence_fn(force_tol=fmax)
    final_state = ts.optimize(
        system=struct,
        model=model,
        optimizer=ts.optimizers.frechet_cell_fire,
        constant_volume=True,
        hydrostatic_strain=True,
        max_steps=Nrelax,
        convergence_fn=converge_max_force,
        trajectory_reporter=reporter,
        autobatcher=use_autobatcher,
    )

    # Remove trajectory file
    os.remove(trajectory_file)

    return final_state


def get_qha_structures(
    state: ts.state.SimState,
    length_factors: np.ndarray,
    model: torch.nn.Module | None,
    Nmax: int = 300,
    fmax: float = 1e-3,
    *,
    use_autobatcher: bool = False,
) -> list[PhonopyAtoms]:
    """Get relaxed structures at different volumes.

    Args:
        state: Initial state
        length_factors: Array of scaling factors
        model: Calculator model
        Nmax: Maximum number of relaxation steps
        fmax: Force convergence criterion
        use_autobatcher: Whether to use automatic batching

    Returns:
        list[PhonopyAtoms]: Relaxed PhonopyAtoms structures at different volumes
    """
    # Convert state to PhonopyAtoms
    relaxed_struct = ts.io.state_to_phonopy(state)[0]

    # Create scaled structures
    scaled_structs = [
        PhonopyAtoms(
            cell=relaxed_struct.cell * factor,
            scaled_positions=relaxed_struct.scaled_positions,
            symbols=relaxed_struct.symbols,
        )
        for factor in length_factors
    ]

    # Relax all structures
    scaled_state = ts.optimize(
        system=scaled_structs,
        model=model,
        optimizer=ts.optimizers.frechet_cell_fire,
        constant_volume=True,
        hydrostatic_strain=True,
        max_steps=Nmax,
        convergence_fn=ts.runners.generate_force_convergence_fn(force_tol=fmax),
        autobatcher=use_autobatcher,
    )

    return scaled_state.to_phonopy()


def get_qha_phonons(
    scaled_structures: list[PhonopyAtoms],
    model: torch.nn.Module | None,
    supercell_matrix: np.ndarray | None,
    displ: float = 0.05,
    *,
    use_autobatcher: bool = False,
) -> tuple[list[Phonopy], list[list[np.ndarray]], np.ndarray]:
    """Get phonon objects for each scaled atom.

    Args:
        scaled_structures: List of PhonopyAtoms objects
        model: Calculator model
        supercell_matrix: Supercell matrix
        displ: Atomic displacement for phonons
        use_autobatcher: Whether to use automatic batching

    Returns:
        tuple[list[Phonopy], list[list[np.ndarray]], np.ndarray]: Contains:
            - List of Phonopy objects
            - List of force sets for each structure
            - Array of energies
    """
    # Generate phonon object for each scaled structure
    supercells_flat = []
    supercell_boundaries = [0]
    ph_sets = []
    if supercell_matrix is None:
        supercell_matrix = np.eye(3)
    for atoms in scaled_structures:
        ph = Phonopy(
            atoms,
            supercell_matrix=supercell_matrix,
            primitive_matrix="auto",
        )
        ph.generate_displacements(distance=displ)
        supercells = ph.supercells_with_displacements
        n_atoms = sum(len(cell) for cell in supercells)
        supercell_boundaries.append(supercell_boundaries[-1] + n_atoms)
        supercells_flat.extend(supercells)
        ph_sets.append(ph)

    # Run the model on flattened structure
    reporter = ts.TrajectoryReporter(
        None,
        state_frequency=0,
        prop_calculators={
            1: {
                "potential_energy": lambda state: state.energy,
                "forces": lambda state: state.forces,
            }
        },
    )
    results = ts.static(
        system=supercells_flat,
        model=model,
        autobatcher=use_autobatcher,
        trajectory_reporter=reporter,
    )

    # Reconstruct force sets and energies
    force_sets = []
    forces = torch.cat([r["forces"] for r in results]).detach().cpu().numpy()
    energies = (
        torch.tensor([r["potential_energy"] for r in results]).detach().cpu().numpy()
    )
    for i, ph in enumerate(ph_sets):
        start, end = supercell_boundaries[i], supercell_boundaries[i + 1]
        forces_i = forces[start:end]
        n_atoms = len(ph.supercell)
        n_displacements = len(ph.supercells_with_displacements)
        force_sets_i = []
        for j in range(n_displacements):
            start_j = j * n_atoms
            end_j = (j + 1) * n_atoms
            force_sets_i.append(forces_i[start_j:end_j])
        force_sets.append(force_sets_i)

    return ph_sets, force_sets, energies


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
autobatcher = False

# Load the raw model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)
model = ts.models.MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=ts.neighbors.vesin_nl_ts,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Structure and input parameters
struct = bulk("Si", "diamond", a=5.431, cubic=True)  # ASE structure
supercell_matrix = 2 * np.eye(3)  # supercell matrix for phonon calculation
mesh = [20, 20, 20]  # Phonon mesh
fmax = 1e-3  # force convergence
Nmax = 300  # maximum number of relaxation steps
displ = 0.05  # atomic displacement for phonons (in Angstrom)
temperatures = np.arange(0, 1410, 10)  # temperature range for quasi-harmonic calculation
length_factors = np.linspace(
    0.85, 1.15, 15
)  # length factor for quasi-harmonic calculation

# Relax initial structure
state = get_relaxed_structure(
    struct=struct, model=model, Nrelax=Nmax, fmax=fmax, use_autobatcher=autobatcher
)

# Get relaxed structures at different volumes
scaled_structures = get_qha_structures(
    state=state,
    length_factors=length_factors,
    model=model,
    Nmax=Nmax,
    fmax=fmax,
    use_autobatcher=autobatcher,
)

# Get phonons, FC2 forces, and energies for all set of scaled structures
ph_sets, force_sets, energy_sets = get_qha_phonons(
    scaled_structures=scaled_structures,
    model=model,
    supercell_matrix=supercell_matrix,
    displ=displ,
    use_autobatcher=autobatcher,
)

# Calculate thermal properties for each supercells
volumes = []
energies = []
free_energies = []
entropies = []
heat_capacities = []
n_displacements = len(ph_sets[0].supercells_with_displacements)
for i in range(len(ph_sets)):
    ph_sets[i].forces = force_sets[i]
    ph_sets[i].produce_force_constants()
    ph_sets[i].run_mesh(mesh)
    ph_sets[i].run_thermal_properties(
        t_min=temperatures[0],
        t_max=temperatures[-1],
        t_step=int((temperatures[-1] - temperatures[0]) / (len(temperatures) - 1)),
    )

    # Store volume, energy, entropies, heat capacities
    thermal_props = ph_sets[i].get_thermal_properties_dict()
    n_unit_cells = np.prod(np.diag(supercell_matrix))
    cell = scaled_structures[i].cell
    volume = np.linalg.det(cell)
    volumes.append(volume)
    energies.append(energy_sets[i * n_displacements].item() / n_unit_cells)
    free_energies.append(thermal_props["free_energy"])
    entropies.append(thermal_props["entropy"])
    heat_capacities.append(thermal_props["heat_capacity"])

# run QHA
qha = PhonopyQHA(
    volumes=volumes,
    electronic_energies=np.tile(energies, (len(temperatures), 1)),
    temperatures=temperatures,
    free_energy=np.array(free_energies).T,
    cv=np.array(heat_capacities).T,
    entropy=np.array(entropies).T,
    eos="vinet",
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
fig.add_trace(
    go.Scatter(x=temperatures, y=qha.thermal_expansion, mode="lines", line=dict(width=4))
)
fig.update_layout(
    xaxis_title="Temperature (K)",
    yaxis_title="Thermal Expansion (1/K)",
    font=dict(size=24),
    xaxis=axis_style,
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor="white",
)
fig.show()

# Plot bulk modulus vs temperature
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=temperatures, y=qha.bulk_modulus_temperature, mode="lines", line=dict(width=4)
    )
)
fig.update_layout(
    xaxis_title="Temperature (K)",
    yaxis_title="Bulk Modulus (GPa)",
    font=dict(size=24),
    xaxis=axis_style,
    yaxis=axis_style,
    width=800,
    height=600,
    plot_bgcolor="white",
)
fig.show()
