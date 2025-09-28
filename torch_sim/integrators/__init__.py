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


class MdFlavor(StrEnum):
    """Flavor of molecular dynamics simulation."""

    nve = "nve"
    nvt_langevin = "nvt_langevin"
    nvt_nose_hoover = "nvt_nose_hoover"
    npt_langevin = "npt_langevin"
    npt_nose_hoover = "npt_nose_hoover"


# Integrator registry - maps integrator names to (init_fn, step_fn) pairs
INTEGRATOR_REGISTRY: Final[
    dict[MdFlavor, tuple[Callable[..., Any], Callable[..., Any]]]
] = {
    MdFlavor.nve: (nve_init, nve_step),
    MdFlavor.nvt_langevin: (nvt_langevin_init, nvt_langevin_step),
    MdFlavor.nvt_nose_hoover: (nvt_nose_hoover_init, nvt_nose_hoover_step),
    MdFlavor.npt_langevin: (npt_langevin_init, npt_langevin_step),
    MdFlavor.npt_nose_hoover: (npt_nose_hoover_init, npt_nose_hoover_step),
}
