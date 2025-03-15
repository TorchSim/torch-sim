# TorchSim

[![CI](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/radical-ai/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/radical-ai/torch-sim)

TorchSim is an open-source simulation engine in PyTorch focused on atomistic simulations. It provides a easy to use and efficient interface for running simulations on both CPU and GPU. Being built on PyTorch, it allows for automatic differentiation of the simulation equations, making it easy to compute quantities of interest.

## Installation

```sh
git clone https://github.com/radical-ai/torch-sim
cd torch-sim
pip install .
```

## Running Example Scripts

`torchsim` has dozens of demos in the [`examples/`](examples) folder. To run any of the them, use the following command:

```sh
# if uv is not yet installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick any of the examples
uv run --with . examples/2_Structural_optimization/2.3_MACE_FIRE.py
uv run --with . examples/3_Dynamics/3.3_MACE_NVE_cueq.py
uv run --with . examples/4_High_level_api/4.1_high_level_api.py
```

## High-level API Example with ASE

```python
from ase.build import bulk
from torchsim.runners import integrate, state_to_atoms
from torchsim.integrators import nvt_langevin
from torchsim.models.lennard_jones import LennardJonesModel
import torch

# instantiate a lennard jones model for Si
lj_model = LennardJonesModel(
    sigma=2.0,
    epsilon=0.1,
    device=torch.device("cpu"),
    dtype=torch.float64,
)
# create a bulk silicon system
si_atoms = bulk("Si", "fcc", a=5.43, cubic=True)

# integrate the system with NVT Langevin dynamics
final_state = integrate(
    system=si_atoms,
    model=lj_model,
    integrator=nvt_langevin,
    n_steps=1000,
    temperature=2000,
    timestep=0.002,
)
# retrieve the final state as an ASE Atoms object
final_atoms = state_to_atoms(final_state)
```

## Low-level API Example with ASE

```python
# the model and atoms will remain the same

from torchsim.runners import atoms_to_state
from torchsim.units import MetalUnits

# instantiate the state
initial_state = atoms_to_state(si_atoms, device=lj_model.device, dtype=lj_model.dtype)

# instantiate the integrator
init_state_fn, update_fn = nvt_langevin(
    model=lj_model,
    dt=0.002 * MetalUnits.time,
    kT=2000 * MetalUnits.temperature,
)

# initialize the state
initial_state = init_state_fn(initial_state, kT=2000 * MetalUnits.temperature)

# run the simulation
for step in range(1000):
    state = update_fn(initial_state, kT=2000 * MetalUnits.temperature)
```

## High-level API with reporting

```python
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

# integrate the system with NVT Langevin dynamics
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
```

## High-level API with MACE and batching

```python
from mace.calculators.foundations_models import mace_mp
from torchsim.models.mace import MaceModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# simply load the model from mace-mp
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_force=True,
)

# create a bulk example systems
fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True)
fe_atoms_supercell = fe_atoms.repeat([2, 2, 2])
si_atoms_supercell = si_atoms.repeat([2, 2, 2])

# create a reporter to report the trajectories
trajectory_files = [
    "si_traj.h5md",
    "fe_traj.h5md",
    "si_supercell_traj.h5md",
    "fe_supercell_traj.h5md",
]
reporter = TrajectoryReporter(
    filenames=trajectory_files,
    # report state every 10 steps
    state_frequency=50,
    prop_calculators=prop_calculators,
)

# seamlessly run a batched simulation
final_state = integrate(
    system=[si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell],
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=100,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)
final_atoms = state_to_atoms(final_state)
final_fe_atoms_supercell = final_atoms[3]

for filename in trajectory_files:
    with TorchSimTrajectory(filename) as traj:
        print(traj)
```

## Core modules

TorchSim is built around the following core modules:

- [`torchsim.integrators`](torchsim/integrators.py): Provides batched molecular dynamics integrators for simulating the time evolution of molecular systems.

- [`torchsim.optimizers`](torchsim/optimizers.py): Provides optimization algorithms for molecular systems, including gradient descent and the FIRE algorithm.

- [`torchsim.unbatched`](torchsim/unbatched): Contains unbatched versions of the integrators and optimizers.

- [`torchsim.monte_carlo`](torchsim/monte_carlo.py): Contains functions for performing Monte Carlo simulations, including swap-based Monte Carlo.

- [`torchsim.neighbors`](torchsim/neighbors.py): Contains functions for constructing neighbor lists.

- [`torchsim.quantities`](torchsim/quantities.py): Functions for computing physical quantities, such as kinetic energy, from simulation data.

- [`torchsim.runners`](torchsim/runners.py): Provides functions for running molecular dynamics simulations and optimizations, handling simulation state and conversions between different representations.

- [`torchsim.state`](torchsim/state.py): Defines the `BaseState` class, a `dataclass` for representing the state of molecular systems.

- [`torchsim.trajectory`](torchsim/trajectory.py): Implements classes for handling trajectory data, allowing for reading and writing of simulation data to HDF5 files.

- [`torchsim.transforms`](torchsim/transforms.py): Functions for handling coordinate transformations and periodic boundary conditions in molecular simulations.

- [`torchsim.elastic`](torchsim/elastic.py): Contains classes and functions for calculating crystal elasticity, including the representation of elastic states and computation of elastic tensors.

- [`torchsim.utils`](torchsim/utils.py): Utility functions.

- [`torchsim.units`](torchsim/units.py): Unit system and conversion factors

- [`torchsim.autobatching`](torchsim/autobatching.py): Contains classes for automatically batching simulations.

Each module is designed to work seamlessly with PyTorch, enabling efficient and flexible molecular simulations.

## Citation

If you use TorchSim in your research, please cite:

```bib
@repository{gangan-2025-torchsim,
  ...
}
```
