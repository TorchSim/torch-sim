"""TorchSim package base module."""

# ruff: noqa: F401

import os
from datetime import datetime

from torch_sim import (
    autobatching,
    elastic,
    integrators,
    io,
    math,
    models,
    monte_carlo,
    neighbors,
    optimizers,
    quantities,
    runners,
    state,
    trajectory,
    transforms,
    units,
)
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
from torch_sim.properties.correlations import CorrelationCalculator

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


PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)

SCRIPTS_DIR = f"{ROOT}/examples"

today = f"{datetime.now().astimezone():%Y-%m-%d}"
