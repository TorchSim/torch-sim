# Core Concepts

## Runners

TorchSim makes atomistic simulation easy with a fully featured high-level API. It introduces three "runner" functions: `integrate` for molecular dynamics, `optimize` for relaxation, and `static` for static evaluation. All functions share a similar signature and support auto batching, trajectory reporting, diverse models, and IO with popular libraries. Further, they support all of this across various simulation types, such as integration with NVT or NPT and optimization with gradient descent or FIRE.
All runners use the [metal unit system](https://docs.lammps.org/units.html).

Learn more in [Introduction to TorchSim](../tutorials/high_level_tutorial.ipynb)

## State

`SimState` is the core atomistic representation for the TorchSim package. It contains the atoms, atomic numbers, cell, and everything else required to represent the simulation. It is the equivalent of the pymatgen `Structure`, the ASE `Atoms`, or the OpenMM `System`. The unique feature of `SimState` is 1) attributes are pytorch tensors and 2) it is a batched state that can represent a single system or many. Every different kind of simulation, from NVT Langevin to Frechet cell fire, has its own `State` type that inherits from `SimState,` initializing it with a unique initialization function and updating with a unique update function. Both the mathematical operations of the integrators and optimizers and the forward passes of the models act directly on the batched state, letting the operations make much more efficient use of GPUs.

Learn more in [Understanding State](../tutorials/state_tutorial.ipynb)

## Autobatching

Even when batching is possible, using GPU memory efficiently isn't easy, a problem that TorchSim also solves. The first issue is that different models have different memory footprints for the same system. Second, the memory footprint scales differently depending on how the neighbor list is computed. For example, the MACE model scales with the number of atoms multiplied by the number density (radial cutoff), while the Fairchem models scale with just the number of atoms (max neighbors). Thus, predicting the memory footprint of a batched simulation, optimally packing systems into memory, and correctly evolving and reporting the trajectories is no easy task. TorchSim automatically determines the memory footprint of a model on the fly and arranges the simulations to make optimal use of available memory. It does so for for molecular dynamics, where states are typically evolved for a fixed time, and for optimization, where states must be removed as they converge.

Learn more in [Understanding Autobatching](../tutorials/autobatching_tutorial.ipynb)

## Reporting

Efficiently tracking trajectory information is a core feature of simulation engines. TorchSim introduces a new trajectory format that allows native integration with TorchSim batched state, binary encoding of any properties, and on-the-fly compression. Writing a new trajectory format was not undertaken lightly; the developers are painfully aware of the great redundancy of trajectory formats. Ultimately, none could be adapted to meet the project's needs. The `TorchSimTrajectory` is based on [HDF5](https://docs.h5py.org/en/stable/) and is best thought of as an efficient container for arbitrary arrays. It wraps an `hdf5` file in convenient utilities tailored to atomistic simulation, making it easy and fast to save the state. Rather than efficiently storing positions and velocities with hacked-on solutions for anything else, the `TorchSimTrajectory` stores any properties with the same binary encoding and compression, such as temperature, forces, per-atom-energies, or electric fields.

Learn more in [Understanding Reporting](../tutorials/reporting_tutorial.ipynb)

## High-level vs Low-Level

Under the hood, TorchSim takes a modular functional approach to atomistic simulation. Each integrator or optimizer function, such as `nvt_langevin,` takes in a model and parameters and returns `init` and `update` functions that act on a unique `State.` The state inherits from `SimState` and tracks the fixed and fluctuating parameters of the simulation, such as the `momenta` for NVT or the timestep for FIRE. The runner functions take this basic structure and wrap it in a convenient interface with autobatching and reporting.

Learn more in [Fundamentals of TorchSim](../tutorials/low_level_tutorial.ipynb) and [Implementing New Methods](../tutorials/hybrid_swap_tutorial.ipynb)
