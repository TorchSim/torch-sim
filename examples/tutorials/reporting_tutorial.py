# %% [markdown]
"""
# Trajectory Reporting in TorchSim

This tutorial explains how to save and analyze trajectory data from molecular dynamics 
simulations using TorchSim's trajectory module.

## Introduction

When running molecular dynamics simulations, we often want to:
- Save atomic positions, forces, and energies over time
- Calculate and store custom properties during the simulation
- Analyze the trajectory data after the simulation
- Convert trajectories to other formats for visualization

TorchSim provides two main classes for handling trajectories:
1. `TorchSimTrajectory`: Low-level interface for reading/writing HDF5 files
2. `TrajectoryReporter`: High-level interface that builds on TorchSimTrajectory

We'll start with the low-level interface to understand the fundamentals.
"""

# %%
import torch
from torch_sim.trajectory import TrajectoryReporter, TorchSimTrajectory
from torch_sim.state import SimState
from torch_sim.models.lennard_jones import LennardJonesModel


# %% [markdown]
"""
## TorchSimTrajectory: Low-Level Interface

The `TorchSimTrajectory` class provides direct access to trajectory data in HDF5 format.
It's designed to be simple and efficient, storing data in a structured way.

### Basic Usage

Let's start with the basics of writing and reading data:
"""

# %%
# Open a trajectory file for writing
trajectory = TorchSimTrajectory(
    tmp_path / "basic.h5",
    mode="w",  # 'w' for write, 'r' for read, 'a' for append
    compress_data=True,  # Enable compression
    coerce_to_float32=True,  # Convert float64 to float32 to save space
)

# Write some custom arrays
data = {
    "positions": torch.randn(10, 3),  # [n_atoms, 3] array
    "velocities": torch.randn(10, 3),
}
trajectory.write_arrays(data, steps=0)  # Save at simulation step 0

trajectory.close()

# %% [markdown]
"""
### Writing SimState Objects

While you can write individual arrays, TorchSimTrajectory provides a convenient method
to write entire SimState objects:
"""

# %%
# Create a simple simulation state
state = SimState(
    positions=torch.randn(10, 3),
    masses=torch.ones(10),
    cell=torch.eye(3).unsqueeze(0),
    atomic_numbers=torch.ones(10, dtype=torch.int),
    batch=torch.zeros(10, dtype=torch.int),
)

# Open a new trajectory file
with TorchSimTrajectory(tmp_path / "state.h5", mode="w") as traj:
    # Write the state with additional options
    traj.write_state(
        state, 
        steps=0,
        save_velocities=True,  # Save velocities if present
        save_forces=True,      # Save forces if present
        variable_cell=True,    # Cell parameters can change over time
        variable_masses=False, # Masses are constant
    )

# %% [markdown]
"""
### Reading Trajectory Data

The trajectory file can be read in several ways:
"""

# %%
# Open for reading
with TorchSimTrajectory(tmp_path / "state.h5", mode="r") as traj:
    # Get raw arrays
    positions = traj.get_array("positions")
    steps = traj.get_steps("positions")
    
    # Get a SimState object
    state = traj.get_state(frame=0)
    
    # Get structure objects for visualization
    atoms = traj.get_atoms(frame=0)  # ASE Atoms
    structure = traj.get_structure(frame=0)  # Pymatgen Structure

# %% [markdown]
"""
## TrajectoryReporter: High-Level Interface

While TorchSimTrajectory is powerful, the TrajectoryReporter makes it easier to:
- Save states at regular intervals
- Calculate and save properties during simulation
- Handle multi-batch simulations

### Basic State Saving

Let's start with the simplest use case - saving states periodically:
"""

# %%
# Initialize a basic reporter
reporter = TrajectoryReporter(
    filenames=tmp_path / "simulation.h5",
    state_frequency=100,  # Save full state every 100 steps
)

# Run a simple simulation
for step in range(5):
    # Update state (in a real simulation)
    state.positions += torch.randn_like(state.positions) * 0.01
    
    # Report the state
    reporter.report(state, step)

reporter.close()

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

```python
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
# Define some property calculators
def calculate_com(state: SimState) -> torch.Tensor:
    """Calculate center of mass - only needs state"""
    return torch.mean(state.positions * state.masses.unsqueeze(1), dim=0)

def calculate_energy(state: SimState, model: torch.nn.Module) -> torch.Tensor:
    """Calculate energy - needs both state and model"""
    output = model(state)
    return output["energy"]

# Create a reporter with property calculators
reporter = TrajectoryReporter(
    filenames=tmp_path / "props.h5",
    state_frequency=100,  # Save full state every 100 steps
    prop_calculators={
        10: {  # Calculate these properties every 10 steps
            "center_of_mass": calculate_com,
            "energy": calculate_energy,
        }
    }
)

# Initialize a model for energy calculation
lj_model = LennardJonesModel()

# Run simulation with property calculation
for step in range(5):
    state.positions += torch.randn_like(state.positions) * 0.01
    reporter.report(state, step, model=lj_model)

reporter.close()

# %% [markdown]
"""
### State Writing Options

The TrajectoryReporter accepts `state_kwargs` that control how states are written:
"""

# %%
reporter = TrajectoryReporter(
    filenames=tmp_path / "state_options.h5",
    state_frequency=100,
    state_kwargs={
        "save_velocities": True,    # Save velocities if present
        "save_forces": True,        # Save forces if present
        "variable_cell": True,      # Cell parameters can change
        "variable_masses": False,   # Masses are constant
        "variable_atomic_numbers": False,  # Atomic numbers are constant
    }
)

# %% [markdown]
"""
### Multi-Batch Simulations

When simulating multiple systems simultaneously, the reporter can split the data across
multiple trajectory files:
"""

# %%
# Create a double-batch simulation state
positions = torch.randn(20, 3)  # 20 atoms total
batch = torch.cat([torch.zeros(10), torch.ones(10)])  # Two batches of 10 atoms
multi_state = SimState(
    positions=positions,
    masses=torch.ones(20),
    cell=torch.eye(3).unsqueeze(0),
    atomic_numbers=torch.ones(20, dtype=torch.int),
    batch=batch
)

# Create a reporter with multiple files
reporter = TrajectoryReporter(
    filenames=[tmp_path / "system1.h5", tmp_path / "system2.h5"],
    state_frequency=100,
    prop_calculators={10: {"energy": calculate_energy}}
)

# Report state and properties
for step in range(5):
    reporter.report(multi_state, step, lj_model)

reporter.close()

# %% [markdown]
"""
### Data Organization

The HDF5 files created by both classes follow this structure:
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

## Best Practices

1. Use context managers (`with` statements) to ensure files are properly closed
2. Enable compression for large trajectories
3. Consider using float32 precision to reduce file sizes
4. Use property calculators for analysis during simulation
5. Split large multi-batch simulations across multiple files
6. Set appropriate state_kwargs to avoid saving unnecessary data

## Conclusion

TorchSim's trajectory module provides two complementary interfaces:
- `TorchSimTrajectory` for direct, low-level data access
- `TrajectoryReporter` for high-level simulation output

Together, they provide a flexible and efficient way to save and analyze simulation data!
"""