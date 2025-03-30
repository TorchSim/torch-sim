# TorchSim

[![CI](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/radical-ai/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/radical-ai/torch-sim)
[![This project supports Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/torch-sim?logo=pypi&logoColor=white)](https://pypi.org/project/torch-sim)
[![Zenodo](https://img.shields.io/badge/DOI-TODO-blue?logo=Zenodo&logoColor=white)](https://zenodo.org/records/TODO)

<!-- help docs find start of prose in readme, DO NOT REMOVE -->
TorchSim is an next-generation open-source atomistic simulation engine for the MLIP era. By rewriting the core primitives of atomistic simulation in Pytorch, it allows orders of magnitude acceleration of popular machine learning potentials.

* Automatic batching and GPU memory management allowing significant simulation speedup
* Support for MACE and Fairchem MLIP models
* Support for classical lennard jones, morse, and soft-sphere potentials
* Molecular dynamics integration schemes like NVE, NVT Langevin, and NPT langevin
* Relaxation of atomic positions and cell with gradient descent and FIRE
* Swap monte carlo and hybrid swap monte carlo algorithm
* An extensible binary trajectory writing format with support for arbitrary properties
* A simple and intuitive high-level API for new users
* Integration with ASE, Pymatgen, and Phonopy
* and more: differentiable simulation, elastic properties, custom workflows...

## Quick Start

Here is a quick demonstration of many of the core features of TorchSim:
native support for GPUs, MLIP models, ASE integration, simple API,
autobatching, and trajectory reporting, all in under 40 lines of code.

```python
import torch
import torch_sim as ts

# run natively on gpus
device = torch.device("cuda")

# easily load the model from mace-mp
from mace.calculators.foundations_models import mace_mp
from torch_sim.models import MaceModel
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)

# create many replicates of a cu system using ase
from ase.build import bulk
cu_atoms = bulk("Cu", "fcc", a=5.26, cubic=True).repeat(2, 2, 2)
many_cu_atoms = [cu_atoms] * 50
trajectory_files = [f"fe_traj_{i}" for i in many_cu_atoms]

# run them all simultaneously with batching
final_state = ts.optimize(
    system=many_cu_atoms,
    model=mace_model,
    integrator=ts.nvt_langevin,
    n_steps=50,
    temperature=1000,
    timestep=0.002,
    trajectory_reporter=dict(filenames=trajectory_files, state_frequency=10),
    autobatching=True
)
final_atoms_list = final_state.to_atoms()

# extract the final energy from the trajectory file
final_energies = []
for filename in trajectory_files:
    with ts.TorchSimTrajectory(filename) as traj:
        final_energies.append(traj.get_array("potential_energy")[-1])

print(final_energies)
```

## Installation
### PyPI Installation
```sh
pip install --upgrade pip
pip install torch-sim
```

### Installing from source
```sh
git clone https://github.com/radical-ai/torch-sim
cd torch-sim
pip install .
```

## Examples

`torch-sim` has dozens of example scripts and tutorials in the [`examples/`](examples/readme.md) folder.

## Core Modules

(Link to API docs)

* [`torch_sim.integrators`](torch_sim/integrators.py): Provides batched molecular dynamics integrators for simulating the time evolution of atomistic systems.

## Citation

If you use TorchSim in your research, please cite:

```bib
@repository{gangan-2025-torchsim,
  ...
}
```
