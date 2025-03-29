"""Torch-Sim package base module."""

# ruff: noqa: F401

import os
from datetime import datetime

from torch_sim import (
    io,
    models,
    neighbors,
    optimizers,
    quantities,
    runners,
    state,
    trajectory,
    units,
)
from torch_sim._version import __version__
from torch_sim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher
from torch_sim.integrators import npt_langevin, nve, nvt_langevin

# state propagators
from torch_sim.monte_carlo import swap_monte_carlo
from torch_sim.optimizers import (
    frechet_cell_fire,
    gradient_descent,
    unit_cell_fire,
    unit_cell_gradient_descent,
)

# quantities
from torch_sim.quantities import calc_kinetic_energy, calc_kT

# high level runners and support
from torch_sim.runners import (
    generate_energy_convergence_fn,
    generate_force_convergence_fn,
    integrate,
    optimize,
    static,
)

# state and state manipulation
from torch_sim.state import concatenate_states, initialize_state
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter


__all__ = [
    "ChunkingAutoBatcher",
    "HotSwappingAutoBatcher",
    "TorchSimTrajectory",
    "TrajectoryReporter",
    "__version__",
    "calc_kT",
    "calc_kinetic_energy",
    "concatenate_states",
    "frechet_cell_fire",
    "generate_energy_convergence_fn",
    "generate_force_convergence_fn",
    "gradient_descent",
    "initialize_state",
    "integrate",
    "npt_langevin",
    "nve",
    "nvt_langevin",
    "optimize",
    "static",
    "swap_monte_carlo",
    "unit_cell_fire",
    "unit_cell_gradient_descent",
]

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)

SCRIPTS_DIR = f"{ROOT}/examples"

today = f"{datetime.now().astimezone():%Y-%m-%d}"
