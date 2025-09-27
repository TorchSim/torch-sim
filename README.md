# TorchSim

[![CI](https://github.com/torchsim/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/torchsim/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/torchsim/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/torchsim/torch-sim)
[![This project supports Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/torch-sim-atomistic?logo=pypi&logoColor=white)](https://pypi.org/project/torch-sim-atomistic)
[![Zenodo](https://img.shields.io/badge/Zenodo-15127004-blue?logo=Zenodo&logoColor=white)][zenodo]

[zenodo]: https://zenodo.org/records/15127004

<!-- help docs find start of prose in readme, DO NOT REMOVE -->
TorchSim is a next-generation open-source atomistic simulation engine for the MLIP
era. By rewriting the core primitives of atomistic simulation in Pytorch, it allows
orders of magnitude acceleration of popular machine learning potentials.

* Automatic batching and GPU memory management allowing significant simulation speedup
* Support for MACE, Fairchem, SevenNet, ORB, MatterSim, graph-pes, and metatomic MLIP models
* Support for classical lennard jones, morse, and soft-sphere potentials
* Molecular dynamics integration schemes like NVE, NVT Langevin, and NPT Langevin
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

### Running batched MD
<!-- tested in tests/test_runners::test_readme_example, update as needed -->

```py
import torch
import torch_sim as ts

# run natively on gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# easily load the model from mace-mp
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)

from ase.build import bulk
cu_atoms = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))
many_cu_atoms = [cu_atoms] * 50
trajectory_files = [f"Cu_traj_{i}.h5md" for i in range(len(many_cu_atoms))]

# run them all simultaneously with batching
final_state = ts.integrate(
    system=many_cu_atoms,
    model=mace_model,
    n_steps=50,
    timestep=0.002,
    temperature=1000,
    integrator=ts.integrators.nvt_langevin,
    trajectory_reporter=dict(filenames=trajectory_files, state_frequency=10),
)
final_atoms_list = final_state.to_atoms()

# extract the final energy from the trajectory file
final_energies = []
for filename in trajectory_files:
    with ts.TorchSimTrajectory(filename) as traj:
        final_energies.append(traj.get_array("potential_energy")[-1])

print(final_energies)
```

### Running batched relaxation

To then relax those structures with FIRE is just a few more lines.

```py
# relax all of the high temperature states
relaxed_state = ts.optimize(
    system=final_state,
    model=mace_model,
    optimizer=ts.OptimFlavor.fire,
    cell_filter=ts.CellFilter.frechet,
    autobatcher=True,
)

print(relaxed_state.energy)
```

## Installation

### PyPI Installation

```sh
pip install torch-sim-atomistic
```

### Installing from source

```sh
git clone https://github.com/torchsim/torch-sim
cd torch-sim
pip install .
```

## Examples

To understand how TorchSim works, start with the [comprehensive tutorials](https://torchsim.github.io/torch-sim/user/overview.html) in the documentation.

## Core Modules

TorchSim's package structure is summarized in the [API reference](https://torchsim.github.io/torch-sim/reference/index.html) documentation and drawn as a treemap below.

![TorchSim package treemap](https://github.com/user-attachments/assets/56f894ad-b995-4108-a6de-a48714276d89)

## License

TorchSim is released under an [MIT license](license).

## Citation

+If you use TorchSim in your research, please cite the [arXiv preprint](https://arxiv.org/abs/2508.06628).
