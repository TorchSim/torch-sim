"""TorchSim package base module."""

# ruff: noqa: F401
import os
from datetime import datetime
from importlib.metadata import version

import torch_sim as ts
from torch_sim import (
    autobatching,
    constraints,
    elastic,
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
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher
from torch_sim.integrators import (
    INTEGRATOR_REGISTRY,
    Integrator,
    NVTNoseHooverState,
    nve_init,
    nve_step,
    nvt_langevin_init,
    nvt_langevin_step,
    nvt_nose_hoover_init,
    nvt_nose_hoover_invariant,
    nvt_nose_hoover_step,
)
from torch_sim.integrators.npt import (
    NPTLangevinState,
    NPTNoseHooverState,
    npt_langevin_init,
    npt_langevin_step,
    npt_nose_hoover_init,
    npt_nose_hoover_invariant,
    npt_nose_hoover_step,
)
from torch_sim.monte_carlo import SwapMCState, swap_mc_init, swap_mc_step
from torch_sim.optimizers import (
    OPTIM_REGISTRY,
    FireState,
    Optimizer,
    OptimState,
    fire_init,
    fire_step,
    gradient_descent_init,
    gradient_descent_step,
)
from torch_sim.optimizers.cell_filters import (
    CELL_FILTER_REGISTRY,
    CellFilter,
    CellFireState,
    CellOptimState,
    get_cell_filter,
)
from torch_sim.properties.correlations import CorrelationCalculator
from torch_sim.quantities import (
    calc_kinetic_energy,
    calc_kT,
    calc_temperature,
    get_pressure,
    system_wise_max_force,
)
from torch_sim.runners import (
    generate_energy_convergence_fn,
    generate_force_convergence_fn,
    integrate,
    optimize,
    static,
)
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter


PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)
SCRIPTS_DIR = f"{ROOT}/examples"

__version__ = version("torch-sim-atomistic")
