# TorchSim

[![CI](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/radical-ai/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/radical-ai/torch-sim)
[![This project supports Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/torch_sim_atomistic?logo=pypi&logoColor=white)](https://pypi.org/project/torch_sim_atomistic)
[![Zenodo](https://img.shields.io/badge/Zenodo-15127004-blue?logo=Zenodo&logoColor=white)][zenodo]

[zenodo]: https://zenodo.org/records/15127004

<!-- help docs find start of prose in readme, DO NOT REMOVE -->
TorchSim is a next-generation open-source atomistic simulation engine for the MLIP
era. By rewriting the core primitives of atomistic simulation in Pytorch, it allows
orders of magnitude acceleration of popular machine learning potentials.

* Automatic batching and GPU memory management allowing significant simulation speedup
* Support for MACE, Fairchem, SevenNet, ORB, MatterSim and metatensor MLIP models
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
from torch_sim.models import MaceModel
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
    integrator=ts.nvt_langevin,
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
    optimizer=ts.frechet_cell_fire,
    autobatcher=True,
)

print(relaxed_state.energy)
```

## Speedup

TorchSim achieves up to 100x speedup compared to ASE with popular MLIPs.

![Speedup comparison](/docs/_static/speedup_plot.svg)

This figure compares the time per atom of ASE and `torch_sim`. Time per atom is defined
as the number of atoms / total time. While ASE can only run a single system of `n_atoms`
(on the $x$ axis), `torch_sim` can run as many systems as will fit in memory. On an H100 80 GB card,
the max atoms that could fit in memory was ~8,000 for [GemNet](https://github.com/FAIR-Chem/fairchem), ~10,000 for [MACE](https://github.com/ACEsuit/mace), and ~2,500
for [SevenNet](https://github.com/MDIL-SNU/SevenNet). This metric describes model performance by capturing speed and memory
usage simultaneously.

## Installation

### PyPI Installation

```sh
pip install torch-sim-atomistic
```

### Installing from source

```sh
git clone https://github.com/radical-ai/torch-sim
cd torch-sim
pip install .
```

## Examples

To understand how TorchSim works, start with the [comprehensive tutorials](https://radical-ai.github.io/torch-sim/user/overview.html) in the documentation.

## Core Modules

TorchSim's structure is summarized in the [API reference](https://radical-ai.github.io/torch-sim/reference/index.html) documentation.

> `torch-sim` module graph. Each node represents a Python module. Arrows indicate imports between modules. Node color indicates connectedness: blue nodes have fewer dependents, red nodes have more (up to 16). The number in parentheses is the number of lines of code in the module.

## License

TorchSim is released under an [MIT license](LICENSE).

## Citation

A manuscript is in preparation. Meanwhile, if you use TorchSim in your research, please [cite the Zenodo archive][zenodo].
