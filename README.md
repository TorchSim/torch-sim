# TorchSim

[![CI](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml/badge.svg)](https://github.com/radical-ai/torch-sim/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/radical-ai/torch-sim/branch/main/graph/badge.svg)](https://codecov.io/gh/radical-ai/torch-sim)


TorchSim is an next-generation open-source atomistic simulation engine for the MLIP era. By rewriting the core primitives of atomistic simulation in Pytorch, it allows orders of magnitude acceleration of popular machine learning potentials.

* automatic batching and GPU memory management allowing up to 200x simulation speedup
* Support for MACE and Fairchem MLIP models
* Support for classical lennard jones, morse, and soft-sphere potentials
* integration with NVE, NVT Langevin, and NPT langevin integrators
* optimization with gradient descent, unit cell FIRE, or frechet cell FIRE
* swap monte carlo and hybrid swap monte carlo
* an extensible binary trajectory writing format with support for arbitrary properties
* a simple and intuitive high-level API for new users
* integration with ASE, Pymatgen, and Phonopy
* and more: differentiable simulation, elastic properties, a2c workflow...

## Quick Start


```python
from ase.build import bulk
from torch_sim.runners import integrate
from torch_sim.integrators import nvt_langevin
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# simply load the model from mace-mp
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_forces=True,
)

# create a bulk example systems
si_atoms = bulk("Si", "fcc", a=5.43, cubic=True)
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
final_state = optimize(
    system=[si_atoms, fe_atoms, si_atoms_supercell, fe_atoms_supercell],
    model=mace_model,
    integrator=nvt_langevin,
    n_steps=100,
    temperature=2000,
    timestep=0.002,
    trajectory_reporter=reporter,
)
final_atoms = state.to_atoms()
final_fe_atoms_supercell = final_atoms[3]

for filename in trajectory_files:
    with TorchSimTrajectory(filename) as traj:
        print(traj)
```


## Installation

```sh
git clone https://github.com/radical-ai/torch-sim
cd torch-sim
pip install .
```

## Running Example Scripts

`torch-sim` has dozens of demos in the [`examples/`](examples) folder. To run any of the them, use the following command:

```sh
# if uv is not yet installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick any of the examples
uv run --with . examples/2_Structural_optimization/2.3_MACE_FIRE.py
uv run --with . examples/3_Dynamics/3.3_MACE_NVE_cueq.py
uv run --with . examples/4_High_level_api/4.1_high_level_api.py
```

## Core Modules
(Link to API docs)

- [`torch_sim.integrators`](torch_sim/integrators.py): Provides batched molecular dynamics integrators for simulating the time evolution of atomistic systems.


## Citation

If you use TorchSim in your research, please cite:

```bib
@repository{gangan-2025-torchsim,
  ...
}
```
