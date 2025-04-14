# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "pymatgen>=2025.2.18",
#     "ase>=3.23.1",
# ]
# ///
# </details>


# %% [markdown]
"""
# Introduction to TorchSim

This tutorial introduces TorchSim's high-level API for molecular dynamics simulations
and geometry optimizations. The high-level API provides simple, powerful interfaces
that abstract away the complexities of setting up atomistic simulations while still
allowing for customization.

## Introduction

TorchSim's high-level API consists of three primary functions:

1. `integrate` - For running molecular dynamics simulations
2. `optimize` - For geometry optimization
3. `static` - For one-time energy/force calculations on a diversity set of systems

These functions handle:
* Automatic state initialization from various input formats
* Memory-efficient GPU operations via autobatching
* Trajectory reporting and property calculation
* Custom convergence criteria

Over the course of the tutorial, we will fully explain the example in the README
by steadily adding functionality.
"""

# %% [markdown]
"""
## Basic Molecular Dynamics

We'll start with a simple example: simulating a silicon system using a Lennard-Jones
potential. First, let's set up our model and create an atomic structure:
"""

# %%
import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.models.lennard_jones import LennardJonesModel

# Create a Lennard-Jones model with parameters suitable for Si
lj_model = LennardJonesModel(
    sigma=2.0,  # Å, typical for Si-Si interaction
    epsilon=0.1,  # eV, typical for Si-Si interaction
    device=torch.device("cpu"),
    dtype=torch.float64,
)

# Create a silicon FCC structure using ASE
cu_atoms = bulk("Cu", "fcc", a=5.43, cubic=True)


# %% [markdown]
"""
Now we can run a molecular dynamics simulation using the `integrate` function. This
function takes care of initializing the state, setting up the integrator, and running
the simulation:
"""

# %% Run NVT simulation at 2000K
n_steps = 50
final_state = ts.integrate(
    system=cu_atoms,  # Input atomic system
    model=lj_model,  # Energy/force model
    integrator=ts.nvt_langevin,  # Integrator to use
    n_steps=n_steps,  # Number of MD steps
    temperature=2000,  # Target temperature (K)
    timestep=0.002,  # Integration timestep (ps)
)

# Convert the final state back to ASE atoms
final_atoms = final_state.to_atoms()


# %% [markdown]
"""
## Trajectory Reporting

While running simulations, we often want to save trajectory data and calculate
properties. The easiest way to do this is to simply specify the `filenames`
argument in the `integrate` function. This will assume some reasonable default
settings for the trajectory reporter and write to the specified files.
"""

# %%
n_steps = 50
final_state = ts.integrate(
    system=cu_atoms,
    model=lj_model,
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter={"filenames": "lj_trajectory.h5"},
)


# %% [markdown]
"""
Behind the scenes, the `dict` is used to instantiate a `TrajectoryReporter` object,
which then handles the reporting. If you need more control over the trajectory
reporter, you can instantiate it manually and pass it to the `integrate` function.

This makes it easier to customize the trajectory reporter to your needs. Below,
we show how to periodically report additional quantities and manually specify the
frequency that the state is saved.

For more detail, see the [trajectory reporter tutorial](./trajectory_reporter.ipynb).
"""

# %% Define the output trajectory file
trajectory_file = "lj_trajectory.h5"

# Define property calculators to track energies
# - Calculate potential energy every 10 steps
# - Calculate kinetic energy every 20 steps
prop_calculators = {
    10: {"potential_energy": lambda state: state.energy},
    20: {
        "kinetic_energy": lambda state: ts.calc_kinetic_energy(
            state.momenta, state.masses
        )
    },
}

# Create a reporter that saves the state every 10 steps
reporter = ts.TrajectoryReporter(
    trajectory_file,
    state_frequency=10,  # Save the state every 10 steps
    prop_calculators=prop_calculators,
)


# %% [markdown]
"""
Now we can run the simulation with trajectory reporting:
"""

# %% Run the simulation with trajectory reporting
final_state = ts.integrate(
    system=cu_atoms,
    model=lj_model,
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,  # Add the reporter
)


# %% [markdown]
"""
After the simulation is complete, we can analyze the trajectory using the
`TorchSimTrajectory` class. This class provides a simple interface for analyzing the
trajectory data.
"""

# %% Open the trajectory file and extract data
with ts.TorchSimTrajectory(trajectory_file) as traj:
    # Read energy arrays
    kinetic_energies = traj.get_array("kinetic_energy")
    potential_energies = traj.get_array("potential_energy")
    final_energy = potential_energies[-1].item()

    # Get the final atomic configuration
    final_atoms = traj.get_atoms(-1)

print(f"Final potential energy: {final_energy:.6f} eV")
print(f"Shape of kinetic energy array: {kinetic_energies.shape}")


# %% [markdown]
"""
## Using Machine Learning Potentials

TorchSim isn't limited to classical potentials. It also supports machine learning
potentials like MACE for more accurate simulations. Let's run a similar simulation
using MACE:
"""

# %%
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the MACE "small" foundation model
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float64,
    compute_forces=True,
)

# Run the simulation with MACE
final_state = ts.integrate(
    system=cu_atoms,
    model=mace_model,
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
)

final_atoms = final_state.to_atoms()


# %% [markdown]
"""
## Batch Processing Multiple Systems

One of the most powerful features of TorchSim is the ability to simulate multiple
systems in parallel. This is especially useful when working with machine learning
potentials that benefit from GPU acceleration:
"""

# %% Create multiple systems to simulate
fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True)
fe_atoms_supercell = fe_atoms.repeat([2, 2, 2])
cu_atoms_supercell = cu_atoms.repeat([2, 2, 2])

# Pack them into a list
systems = [cu_atoms, fe_atoms, cu_atoms_supercell, fe_atoms_supercell]


# %% [markdown]
"""
We can simulate all these systems in a single call to `integrate`:
"""

# %% Run batch simulation with
final_state = ts.integrate(
    system=systems,  # List of systems to simulate
    model=mace_model,  # Single model for all systems
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
)

final_atoms = final_state.to_atoms()
print(f"Number of systems simulated: {len(final_atoms)}")
print(f"Number of atoms in last system: {len(final_atoms[3])}")


# %% [markdown]
"""
### Batch Trajectory Reporting

When simulating multiple systems, we can save each to its own trajectory file:
"""

# %% Create individual filenames for each system
filenames = [f"batch_traj_{i}.h5" for i in range(len(systems))]

# Create a reporter that handles multiple trajectories
batch_reporter = ts.TrajectoryReporter(
    filenames,
    state_frequency=10,
    prop_calculators=prop_calculators,
)

# Run the simulation with batch reporting
final_state = ts.integrate(
    system=systems,
    model=mace_model,
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=batch_reporter,
)


# %% [markdown]
"""
We can analyze each trajectory individually:
"""

# %% Calculate final energy per atom for each system
final_energies_per_atom = []
for i, filename in enumerate(filenames):
    with ts.TorchSimTrajectory(filename) as traj:
        final_energy = traj.get_array("potential_energy")[-1].item()
        n_atoms = len(traj.get_atoms(-1))
        final_energies_per_atom.append(final_energy / n_atoms)
        print(f"System {i}: {final_energy:.6f} eV, {final_energy / n_atoms:.6f} eV/atom")


# %% [markdown]
"""
## Autobatching

The `integrate` function also supports autobatching, which automatically determines
the maximum number of systems that can fit in memory and splits up the systems to make
optimal use of the GPU. This abstracts away the complexity of managing memory when
running more systems than can fit on the GPU.

Ignore the following cell, it just exists so that the example runs on CPU.
"""


# %%
def mock_determine_max_batch_size(*args, **kwargs):
    return 10


ts.autobatching.determine_max_batch_size = mock_determine_max_batch_size


# %% [markdown]
"""
We enable autobatching by simply setting the `autobatcher` argument to `True`.
"""
# %% Run the simulation with batch reporting
final_state = ts.integrate(
    system=systems,
    model=mace_model,
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
    autobatcher=True,
)

# %% [markdown]
"""
Otherwise, everything else is the same! The `integrate` function will still report out
trajectory data, calculate properties, and return a final state with the correct
ordering.

## Geometry Optimization

In addition to molecular dynamics, TorchSim provides a high-level API for geometry
optimization. The `optimize` function is similar to `integrate` in that it takes a list
of systems and a model and support reporting and autobatching. The key difference is
that instead of taking `n_steps` and `temperature`, `optimize` takes a `convergence_fn`
that determines when the optimization is converged. By default, the `convergence_fn` will
wait until the energy difference between steps is less than 1 meV.

Let's use the `optimize` function with the FIRE algorithm to relax our structures:
"""

# %% Optimize multiple systems
final_state = ts.optimize(
    system=systems,
    model=mace_model,
    optimizer=ts.unit_cell_fire,
)

final_atoms = final_state.to_atoms()


# %% [markdown]
"""
### Custom Convergence Criteria

The `optimize` function allows us to specify custom convergence criteria. The inputs to the
convergence function are `state` and `last_energy`. The `state` is a `SimState` object
that contains the current state of the system and the `last_energy` is the energy of the
previous step. The convergence function should return a boolean tensor of length
`n_batches`.

This is how we'd manually define the default `convergence_fn`:
"""


# %% Define a convergence function based on energy differences
def default_energy_convergence(state, last_energy):
    # Consider converged when energy change is less than 1e-6 eV
    if last_energy is None:
        return False
    energy_diff = torch.abs(last_energy - state.energy)
    return energy_diff < 1e-6


# we arbitrarily add energy so nothing is converged
convergence_tensor = default_energy_convergence(final_state, final_state.energy + 1)
print("Any converged?", torch.any(convergence_tensor).item())


# %% [markdown]
"""
For convenience TorchSim provides constructors for common convergence functions.
"""

# %% we use metal units for these functions
energy_convergence_fn = ts.generate_energy_convergence_fn(energy_tol=1e-6)
force_convergence_fn = ts.generate_force_convergence_fn(force_tol=1e-3)

# Run optimization with custom convergence
final_state = ts.optimize(
    system=systems,
    model=mace_model,
    optimizer=ts.unit_cell_fire,
    convergence_fn=force_convergence_fn,  # Custom convergence function
)

final_atoms = final_state.to_atoms()


# %% [markdown]
"""
## Static Calculations

TorchSim also supports static calculations, which are useful for calculating properties
across a diverse set of systems without any system evolution. This is a great way to compute
elastic properties or run a benchmark against DFT energies.

The `static` function is similar to `integrate` and `optimize` in that it takes a list of structures
and a model while supporting batching and reporting. The key difference is that `static` does not
return a final state, but rather a list of dictionaries containing the outputs of any `prop_calculators`
specified in the `TrajectoryReporter`.
"""

# %% static will report all of the properties for each system, regardless of frequency
prop_calculators = {
    10: {"potential_energy": lambda state: state.energy},
    20: {"stress": lambda state: state.stress},
}

final_results = ts.static(
    system=systems,
    model=mace_model,
    # we don't want to save any trajectories this time, just get the properties
    trajectory_reporter={"filenames": None, "prop_calculators": prop_calculators},
)

print(f"Static returns {len(final_results)} results, one for each system")
print(f"Matches the number of systems? {len(final_results) == len(systems)}")
print(f"len(final_results): {len(final_results)}")
assert len(final_results) == len(systems)

cu_results = final_results[0]
print(
    "The Cu system has a final energy of ",
    cu_results["potential_energy"][-1].item(),
    " eV",
)


# %% [markdown]
"""
## Working with PyMatGen Structures

TorchSim supports PyMatGen Structure objects in addition to ASE Atoms objects:
"""

# %%
from pymatgen.core import Structure

# Define a silicon diamond structure using PyMatGen
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

# Run a simulation starting from the PyMatGen structure
final_state = ts.integrate(
    system=structure,
    model=lj_model,
    integrator=ts.nvt_langevin,
    n_steps=n_steps,
    temperature=2000,
    timestep=0.002,
)

# Convert the final state back to a PyMatGen structure
final_structure = final_state.to_structures()


# %% [markdown]
"""
## Conclusion

TorchSim's high-level API provides a simple yet powerful interface for running
molecular simulations:

1. The `integrate` function makes it easy to run MD simulations
2. The `optimize` function handles geometry optimization with custom convergence
3. The `static` function handles static calculations with batching and reporting
4. Built-in support for batch processing multiple systems
5. Seamless integration with trajectory reporting
6. Compatible with both ASE and PyMatGen structures

By handling the complexity behind the scenes, these high-level functions let you focus
on the scientific questions rather than simulation details.

For more advanced use cases, you can still access the lower-level components
(integrators, optimizers, etc.) directly, as shown in other tutorials.
"""
