"""Integrators for molecular dynamics simulations.

This module provides a collection of integrators for molecular dynamics simulations,
supporting NVE (microcanonical), NVT (canonical), and NPT (isothermal-isobaric) ensembles.
Each integrator handles batched simulations efficiently using PyTorch tensors and
supports periodic boundary conditions.

Examples:
    >>> from torch_sim.integrators import nve
    >>> nve_init, nve_update = nve(
    ...     model, dt=1e-3 * units.time, kT=300.0 * units.temperature
    ... )
    >>> state = nve_init(initial_state)
    >>> for _ in range(1000):
    ...     state = nve_update(state)

Notes:
    All integrators support batched operations for efficient parallel simulation
    of multiple systems.
"""

# ruff: noqa: F401

from .md import MDState, calculate_momenta, momentum_step, position_step, velocity_verlet
from .npt import (
    NPTLangevinState,
    npt_langevin,
    npt_nose_hoover,
    npt_nose_hoover_invariant,
)
from .nve import nve
from .nvt import nvt_langevin, nvt_nose_hoover, nvt_nose_hoover_invariant
