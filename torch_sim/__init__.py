"""Torch-Sim package base module."""

import os
from datetime import datetime

from torch_sim._version import __version__

# high level runners and support
from torch_sim.runners import (
    integrate,
    optimize,
    static,
    generate_energy_convergence_fn,
    generate_force_convergence_fn,
)
from torch_sim.trajectory import TrajectoryReporter, TorchSimTrajectory
from torch_sim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher

# state propagators
from torch_sim.monte_carlo import swap_monte_carlo
from torch_sim.integrators import nvt_langevin, npt_langevin, nve
from torch_sim.optimizers import (
    gradient_descent,
    unit_cell_gradient_descent,
    unit_cell_fire,
    frechet_cell_fire,
)

# state and state manipulation
from torch_sim.state import initialize_state, concatenate_states

# quantities
from torch_sim.quantities import kinetic_energy, temperature

__all__ = [
    "__version__",
    "integrate",
    "optimize",
    "static",
    "generate_energy_convergence_fn",
    "generate_force_convergence_fn",
    "TrajectoryReporter",
    "TorchSimTrajectory",
    "ChunkingAutoBatcher",
    "HotSwappingAutoBatcher",
    "swap_monte_carlo",
    "nvt_langevin",
    "npt_langevin",
    "nve",
    "gradient_descent",
    "unit_cell_gradient_descent",
    "unit_cell_fire",
    "frechet_cell_fire",
    "initialize_state",
    "concatenate_states",
    "kinetic_energy",
    "temperature",
]

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)

SCRIPTS_DIR = f"{ROOT}/examples"

today = f"{datetime.now().astimezone():%Y-%m-%d}"
