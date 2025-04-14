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
# Understanding Reporting

This tutorial explains how to save and analyze trajectory data from molecular dynamics
simulations using TorchSim's trajectory module. Though reporting can be automatically
handled by the `integrate`, `optimize`, and `static` functions, understanding the
reporting interface is helpful for developing more complex workflows.

## Introduction

TorchSim provides two classes for handling trajectories:

`TorchSimTrajectory` is a flexible low-level interface for reading/writing HDF5 files.
The TorchSimTrajectory is two things, 1) a file format for storing simulation data,
equivalent to the ASE `.traj` file format or the classical MD `.dcd` file format, and
2) a simple interface for storing and retrieving trajectory data from HDF5 files.

`TrajectoryReporter` builds on `TorchSimTrajectory` to make it easier to record
simulation data. It provides a high-level interface for saving states at regular
intervals, calculating and saving properties during simulation, and handling
multi-batch simulations.

We'll start with the low-level interface to understand the fundamentals.
"""

# %% [markdown]
"""
## TorchSimTrajectory: Low-Level Interface

The TorchSimTrajectory does not aim to be a new trajectory standard, but rather
a simple interface for storing and retrieving trajectory data from HDF5 files.
Through the power of HDF5, the TorchSimTrajectory supports:
* Saving arbitrary arrays from the user in a natural way
* First class support for `torch_sim.state.SimState` objects
* Binary encoding + compression for minimal file sizes
* Easy interoperability with ASE and pymatgen

### Basic Usage

Let's start with the basics usage of the TorchSimTrajectory, writing and reading
arrays of data. This is the operation that all other functionality is built on.
"""

# %%
import torch
import torch_sim as ts

# Open a trajectory file for writing
trajectory = ts.TorchSimTrajectory(
    "basic_traj.h5",
    mode="w",  # 'w' for write, 'r' for read, 'a' for append
    compress_data=True,  # Enable compression
    coerce_to_float32=True,  # Convert float64 to float32 to save space
)

# Write some custom arrays
data = {
    "positions": torch.randn(10, 3),  # [n_atoms, 3] array
    "velocities": torch.randn(10, 3),
}
# save the data at simulation step 1, 2, 3, 4, 5
for step in range(5):
    trajectory.write_arrays(data, steps=step + 1)

# print a summary of the trajectory
print(trajectory)

# we can read back out the positions and the steps they were saved at
positions = trajectory.get_array("positions")
steps = trajectory.get_steps("positions")

trajectory.close()


# %% [markdown]
"""
### Writing SimState Objects

While you can write individual arrays, TorchSimTrajectory provides a convenient method
to write entire SimState objects:
"""

# %%
from ase.build import bulk

# Create a bulk Si diamond structure
state = ts.initialize_state(
    bulk("Si", "diamond", a=5.43), device="cpu", dtype=torch.float64
)

# Open a new trajectory file in a context manager
with ts.TorchSimTrajectory("random_state.h5", mode="w") as traj:
    # Write the state with additional options
    for i in range(5):
        traj.write_state(
            state,
            steps=i + 1,
            save_velocities=False,  # our basic state doesn't have velocities
            save_forces=False,  # our basic state doesn't have forces
            variable_cell=False,  # True for an NPT simulation, where the cell changes
            variable_masses=False,  # True for a Monte Carlo simulation which swaps atoms
        )
    print(traj)


# %% [markdown]
"""
### Reading Trajectory Data

Once we've written a trajectory, we can get the raw arrays, a SimState object, or
convert the state to an atoms or ase.Atoms object.
"""

# %% Open for reading
with ts.TorchSimTrajectory("random_state.h5", mode="r") as traj:
    # Get raw arrays
    positions = traj.get_array("positions")
    steps = traj.get_steps("positions")

    # Get a SimState object from the first cell
    state = traj.get_state(0)

    # Get ase atoms from the second cell
    atoms = traj.get_atoms(2)

    # get pymatgen structure from the last cell
    structure = traj.get_structure(-1)

    # write ase trajectory
    traj.write_ase_trajectory("random_state.traj")


# %% [markdown]
"""
## TrajectoryReporter: High-Level Interface

While TorchSimTrajectory is powerful, it is low-level and requires too much
code to do the things that are most common in atomistic simulation. To bridge the gap
TorchSim provides the TrajectoryReporter, which makes it easier to save states at
regular intervals, calculate and save properties during simulation, and handle
multi-batch simulations.

### Basic State Saving

Let's start with the simplest use case - saving states periodically:
"""

# %% Initialize a basic reporter
reporter = ts.TrajectoryReporter(
    filenames="reported_traj.h5",
    state_frequency=5,  # Save full state every 100 steps
)

# Run a simple simulation
for step in range(50):
    # Report the state
    reporter.report(state, step + 1)


# under the hood, the reporter is using TorchSimTrajectory
traj = reporter.trajectories[0]
print(traj)


# %% [markdown]
"""
### Property Calculators

Often we want to calculate and save properties during simulation. Property calculators
are functions that:
1. Take a SimState as their first argument
2. Optionally take a model as their second argument
3. Return a tensor that will be saved in the trajectory

The property calculators are organized in a dictionary that maps frequencies to
property names and their calculator functions:

```py
prop_calculators = {
    frequency1: {
        "prop1": calc_fn1,
        "prop2": calc_fn2,
    },
    frequency2: {
        "prop3": calc_fn3,
    }
}
```

Let's see an example:
"""

# %%
from torch_sim.models import LennardJonesModel


# Define some property calculators
def calculate_com(state: ts.state.SimState) -> torch.Tensor:
    """Calculate center of mass - only needs state"""
    return torch.mean(state.positions * state.masses.unsqueeze(1), dim=0)


def calculate_energy(state: ts.state.SimState, model: torch.nn.Module) -> torch.Tensor:
    """Calculate energy - needs both state and model"""
    return model(state)["energy"]


# Create a reporter with property calculators
reporter = ts.TrajectoryReporter(
    filenames="traj_with_props.h5",
    state_frequency=50,  # Save full state every 100 steps
    prop_calculators={
        10: {"center_of_mass": calculate_com},
        20: {"energy": calculate_energy},
    },
)

# Initialize a model for energy calculation
lj_model = LennardJonesModel()

# Run simulation with property calculation
for step in range(100):
    reporter.report(state, step + 1, model=lj_model)

traj = reporter.trajectories[0]
print(traj)

reporter.close()


# %% [markdown]
"""

We can see that the center of mass is saved 10 times, the energy 5 times, and the state
twice, as we expect from the reporting frequency.

### Other Trajectory Writing Options

The TrajectoryReporter also accepts `state_kwargs` that are passed to the
`TorchSimTrajectory.write_state` method, allowing us to save velocities, forces,
and other properties that might be part of the SimState. Note that velocities and
forces are not attributes of the base SimState but are attributes of the MDState,
which it inherits from.

We can also save metadata about the simulation, which will be saved in the HDF5 file
and can be accessed later.
"""

# %%
reporter = ts.TrajectoryReporter(
    filenames="state_options.h5",
    state_frequency=100,
    metadata={"author": "John Doe"},
    state_kwargs={
        "save_velocities": True,
        "save_forces": True,
        "variable_cell": True,
        "variable_masses": False,
        "variable_atomic_numbers": False,
    },
)

traj = reporter.trajectories[0]
print(traj.metadata)


# %% [markdown]
"""
### Multi-Batch Simulations

When simulating multiple systems simultaneously, the reporter can split the data across
multiple trajectory files:
"""

# %% Create a double-batch simulation state
multi_state = ts.concatenate_states([state.clone() for _ in range(5)])

# Create a reporter with multiple files
reporter = ts.TrajectoryReporter(
    filenames=[f"system{i}.h5" for i in range(5)],
    state_frequency=100,
    prop_calculators={10: {"energy": calculate_energy}},
)

# Report state and properties
for step in range(5):
    reporter.report(multi_state, step, lj_model)

print(f"We now have {len(reporter.trajectories)} trajectories.")
reporter.close()


# %% [markdown]
"""
### Closing the Reporter

This is a bit of a niche use case, but we should mention that the reporter
can also run the prop calculators without writing to a trajectory file.
This can be useful if we have defined property calculators and want to call
all of them without writing to a trajectory file.
"""

# %%
reporter = ts.TrajectoryReporter(
    filenames=None,
    prop_calculators={
        10: {"center_of_mass": calculate_com},
        20: {"energy": calculate_energy},
    },
)

# Report state and properties
props = reporter.report(state, 0, lj_model)
print(f"We calculated the following properties: {[list(prop)[0] for prop in props]}")

reporter.close()


# %% [markdown]
"""

## Conclusion

TorchSim's complementary interfaces `TorchSimTrajectory` and `TrajectoryReporter`
provide a flexible and efficient way to save and analyze simulation data.

1. Use `TrajectoryReporter` for saving simulation data to files.
2. Use `TorchSimTrajectory` for opening and reading the trajectory files
you generate.

### HDF5 File Structure

For experienced HDF5 users, the HDF5 files created by both classes follow this
structure:
```
/
├── header/           # File metadata
├── metadata/         # User metadata
├── data/            # Array data
│   ├── positions
│   ├── velocities
│   ├── any_other_array
│   └── ...
└── steps/           # Step numbers
    ├── positions
    ├── velocities
    ├── any_other_array
    └── ...
```
"""
