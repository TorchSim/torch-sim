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
1. `TrajectoryReporter`: High-level interface for saving simulation data
2. `TorchSimTrajectory`: Low-level interface for reading/writing HDF5 files
"""

# %%
import torch
import numpy as np
from pathlib import Path
from torch_sim.trajectory import TrajectoryReporter, TorchSimTrajectory
from torch_sim.state import SimState
from torch_sim.models.lennard_jones import LennardJonesModel

# Create a temporary directory for our trajectory files
from pathlib import Path
tmp_path = Path("tutorial_trajectories")
tmp_path.mkdir(exist_ok=True)

# %% [markdown]
"""
## TrajectoryReporter: High-Level Interface

The `TrajectoryReporter` manages writing simulation data to one or more trajectory 
files. It handles:
- Periodic saving of system states
- Custom property calculations
- Multi-batch simulations

Let's start with a basic example:
"""

# %%
# Create a simple property calculator that computes center of mass
def calculate_com(state: SimState) -> torch.Tensor:
    return torch.mean(state.positions * state.masses.unsqueeze(1), dim=0)

# Initialize a reporter with property calculators
reporter = TrajectoryReporter(
    filenames=tmp_path / "simulation.h5",
    state_frequency=100,  # Save full state every 100 steps
    prop_calculators={
        10: {  # Calculate properties every 10 steps
            "center_of_mass": calculate_com,
        }
    }
)

# %% [markdown]
"""
### Property Calculators

Property calculators are functions that compute custom properties during the simulation.
They can:
- Take a state and optionally a model as input
- Return scalar or tensor quantities
- Be scheduled at different frequencies

Here's an example using a Lennard-Jones model to calculate energies:
"""

# %%
# Initialize a Lennard-Jones model
lj_model = LennardJonesModel()

# Create an energy calculator that uses the model
def energy_calculator(state: SimState, model: torch.nn.Module) -> torch.Tensor:
    output = model(state)
    return output["energy"]

# Create a reporter with model-dependent properties
reporter = TrajectoryReporter(
    filenames=[tmp_path / "system1.h5", tmp_path / "system2.h5"],
    state_frequency=100,
    prop_calculators={
        10: {"energy": energy_calculator}
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
state = SimState(
    positions=positions,
    masses=torch.ones(20),
    cell=torch.eye(3).unsqueeze(0),
    atomic_numbers=torch.ones(20, dtype=torch.int),
    batch=batch
)

# Report state and properties
for step in range(5):
    reporter.report(state, step, lj_model)

reporter.close()

# %% [markdown]
"""
## TorchSimTrajectory: Low-Level Interface

The `TorchSimTrajectory` class provides direct access to trajectory data in HDF5 format.
It supports:
- Reading and writing arbitrary arrays
- Data type coercion for reduced file sizes
- Conversion to common molecular formats (ASE, pymatgen)
"""

# %%
# Open a trajectory file for writing
trajectory = TorchSimTrajectory(
    tmp_path / "direct_write.h5",
    mode="w",  # 'w' for write, 'r' for read, 'a' for append
    compress_data=True,  # Enable compression
    coerce_to_float32=True,  # Convert float64 to float32
)

# Write custom arrays
data = {
    "positions": torch.randn(10, 3),
    "velocities": torch.randn(10, 3),
}
trajectory.write_arrays(data, steps=0)

# Write a full state
trajectory.write_state(state, steps=1)

trajectory.close()

# %% [markdown]
"""
### Reading Trajectory Data

The trajectory file can be read in several ways:
"""

# %%
# Open for reading
with TorchSimTrajectory(tmp_path / "direct_write.h5", mode="r") as traj:
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
### Data Organization

The HDF5 file structure is organized as:
```
/
├── header/           # File metadata
├── metadata/         # User metadata
├── data/            # Array data
│   ├── positions
│   ├── velocities
│   └── ...
└── steps/           # Step numbers
    ├── positions
    ├── velocities
    └── ...
```

This organization allows efficient storage and retrieval of large trajectories.

## Best Practices

1. Use context managers (`with` statements) to ensure files are properly closed
2. Enable compression for large trajectories
3. Consider using float32 precision to reduce file sizes
4. Use property calculators for analysis during simulation
5. Split large multi-batch simulations across multiple files

## Conclusion

The trajectory module provides flexible tools for handling simulation data:
- `TrajectoryReporter` for high-level simulation output
- `TorchSimTrajectory` for direct data access
- Support for custom properties and multiple formats

These tools make it easy to save, analyze, and visualize simulation results!
"""
