"""Integrators for molecular dynamics simulations.

This module provides a collection of integrators for molecular dynamics simulations,
supporting NVE (microcanonical), NVT (canonical), and NPT (isothermal-isobaric) ensembles.
Each integrator handles batched simulations efficiently using PyTorch tensors and
supports periodic boundary conditions.

Examples:
    >>> import torch_sim as ts
    >>> state = ts.nvt_langevin_init(model, initial_state, kT=300.0 * units.temperature)
    >>> for _ in range(1000):
    ...     state = ts.nvt_langevin_step(
    ...         model, state, dt=1e-3 * units.time, kT=300.0 * units.temperature
    ...     )

Notes:
    All integrators support batched operations for efficient parallel simulation
    of multiple systems.
"""

# ruff: noqa: F401
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Final

import torch_sim as ts

from .md import MDState, calculate_momenta, momentum_step, position_step, velocity_verlet
from .npt import (
    NPTLangevinState,
    NPTNoseHooverState,
    npt_langevin_init,
    npt_langevin_step,
    npt_nose_hoover_init,
    npt_nose_hoover_invariant,
    npt_nose_hoover_step,
)
from .nve import nve_init, nve_step
from .nvt import (
    NVTNoseHooverState,
    nvt_langevin_init,
    nvt_langevin_step,
    nvt_nose_hoover_init,
    nvt_nose_hoover_invariant,
    nvt_nose_hoover_step,
)


class Integrator(StrEnum):
    """Enumeration of available molecular dynamics (MD) integrators.

    Each member represents a different simulation ensemble or thermostat/barostat
    scheme. These values are used as keys in :data:`INTEGRATOR_REGISTRY`
    to select the corresponding initialization and stepping functions.

    Available options:
        - ``nve``: Constant energy (microcanonical) ensemble.
        - ``nvt_langevin``: Langevin thermostat for constant temperature.
        - ``nvt_nose_hoover``: Nosé-Hoover thermostat for constant temperature.
        - ``npt_langevin``: Langevin barostat for constant temperature and pressure.
        - ``npt_nose_hoover``: Nosé-Hoover barostat for constant temperature
                and constant pressure.

    Example:
        >>> integrator = Integrator.nvt_langevin
        >>> print(integrator.value)
        'nvt_langevin'

    """

    nve = "nve"
    nvt_langevin = "nvt_langevin"
    nvt_nose_hoover = "nvt_nose_hoover"
    npt_langevin = "npt_langevin"
    npt_nose_hoover = "npt_nose_hoover"


#: Integrator registry - maps integrator names to (init_fn, step_fn) pairs.
#:
#: This dictionary associates each :class:`Integrator` enum value with a pair
#: of callables:
#:
#: - **init_fn**: A function used to initialize the integrator state.
#: - **step_fn**: A function that advances the state by one simulation step.
#:
#: Example:
#:
#:     >>> init_fn, step_fn = INTEGRATOR_REGISTRY[Integrator.nvt_langevin]
#:     >>> state = init_fn(...)
#:     >>> new_state = step_fn(state, ...)
#:
#: The available integrators are:
#:
#: - ``Integrator.nve``: Velocity Verlet (microcanonical)
#: - ``Integrator.nvt_langevin``: Langevin thermostat
#: - ``Integrator.nvt_nose_hoover``: Nosé-Hoover thermostat
#: - ``Integrator.npt_langevin``: Langevin barostat
#: - ``Integrator.npt_nose_hoover``: Nosé-Hoover barostat
#:
#: :type: dict[Integrator, tuple[Callable[..., Any], Callable[..., Any]]]
INTEGRATOR_REGISTRY: Final[
    dict[Integrator, tuple[Callable[..., Any], Callable[..., Any]]]
] = {
    Integrator.nve: (nve_init, nve_step),
    Integrator.nvt_langevin: (nvt_langevin_init, nvt_langevin_step),
    Integrator.nvt_nose_hoover: (nvt_nose_hoover_init, nvt_nose_hoover_step),
    Integrator.npt_langevin: (npt_langevin_init, npt_langevin_step),
    Integrator.npt_nose_hoover: (npt_nose_hoover_init, npt_nose_hoover_step),
}
