### basic lennard jones example


# %% %
from ase.build import bulk
from torchsim.runners import integrate, state_to_atoms
from torchsim.integrators import nvt_langevin
from torchsim.models.lennard_jones import LennardJonesModel
import torch

from mace.calculators.foundations_models import mace_mp
from torchsim.models.mace import MaceModel


lj_model = LennardJonesModel(
    sigma=2.0,  # Å, typical for Si-Si interaction
    epsilon=0.1,  # eV, typical for Si-Si interaction
    device=torch.device("cpu"),
    dtype=torch.float64,
)

si_atoms = bulk("Si", "fcc", a=5.43, cubic=True)

final_state = integrate(
    system=si_atoms,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=1000,
    temperature=2000,
    timestep=0.002,
)
final_atoms = state_to_atoms(final_state)


### basic lennard jones example with reporting

from torchsim.trajectory import TrajectoryReporter, TorchSimTrajectory
from torchsim.quantities import kinetic_energy

trajectory_file = "lj_trajectory.h5md"
# report potential energy every 10 steps and kinetic energy every 20 steps
prop_calculators = {
    10: {"potential_energy": lambda state: state.energy},
    20: {"kinetic_energy": lambda state: kinetic_energy(state.momenta, state.masses)},
}

reporter = TrajectoryReporter(
    trajectory_file,
    # report state every 10 steps
    state_frequency=10,
    prop_calculators=prop_calculators,
)

final_state = integrate(
    system=si_atoms,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=1000,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)

# Check energy fluctuations
with TorchSimTrajectory(trajectory_file) as traj:
    kinetic_energies = traj.get_array("kinetic_energy")
    potential_energies = traj.get_array("potential_energy")
    final_energy = potential_energies[-1]

    final_atoms = traj.get_atoms(-1)


### basic mace example

# cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"


mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_force=True,
)

reporter = TrajectoryReporter(
    trajectory_file,
    # report state every 10 steps
    state_frequency=10,
    prop_calculators=prop_calculators,
)

final_state = integrate(
    system=si_atoms,
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=100,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)
final_atoms = state_to_atoms(final_state)


### basic mace example with batching

fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True)
fe_atoms_supercell = fe_atoms.repeat([2, 2, 2])
si_atoms_supercell = si_atoms.repeat([2, 2, 2])

final_state = integrate(
    system=[si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell],
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=100,
    temperature=2000,
    timestep=0.002,
)
final_atoms = state_to_atoms(final_state)
final_fe_atoms_supercell = final_atoms[3]


### basic mace example with batching and reporting

systems = [si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell]

filenames = [f"batch_traj_{i}.h5md" for i in range(len(systems))]
batch_reporter = TrajectoryReporter(
    filenames,
    state_frequency=100,
    prop_calculators=prop_calculators,
)
final_state = integrate(
    system=systems,
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=100,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=batch_reporter,
)

final_energies_per_atom = []
for filename in filenames:
    with TorchSimTrajectory(filename) as traj:
        final_energy = traj.get_array("potential_energy")[-1]
        final_energies_per_atom.append(final_energy / len(traj.get_atoms(-1)))


### basic mace example with optimization

from torchsim.runners import optimize
from torchsim.optimizers import fire

systems = [si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell]

final_state = optimize(
    system=systems,
    model=mace_model,
    optimizer=fire,
)


### basic mace example with custom convergence function

from torchsim.runners import optimize
from torchsim.optimizers import fire
from torchsim.units import MetalUnits

systems = [si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell]

for system in systems:
    system.positions += torch.randn_like(system.positions) * 0.01


def converge_forces(state) -> bool:
    return torch.all(state.forces < 1e-2)


def converge_energy(state, last_energy) -> bool:
    return torch.all(last_energy - state.energy < 1e-6 * MetalUnits.energy)


final_state = optimize(
    system=systems,
    model=mace_model,
    optimizer=fire,
    convergence_fn=converge_energy,
)


### basic mace example with pymatgen

from pymatgen.core import Structure
from torchsim.runners import state_to_structures

lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
species = ["Si"] * 8
coords = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.0, 0.5, 0.5],
    [0.25, 0.75, 0.75],
    [0.5, 0.0, 0.5],
    [0.75, 0.25, 0.75],
    [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.25],
]
structure = Structure(lattice, species, coords)
final_state = integrate(
    system=structure,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=1000,
    temperature=2000,
    timestep=0.002,
)
final_structure = state_to_structures(final_state)
