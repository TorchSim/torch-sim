"""Implementations of NVE integrators."""

from typing import Any

import torch

from torch_sim.integrators.md import (
    MDState,
    initialize_momenta,
    momentum_step,
    position_step,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


def nve_init(
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    **_kwargs: Any,
) -> MDState:
    """Initialize an NVE state from input data.

    Creates an initial state for NVE molecular dynamics by computing initial
    energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
    at the specified temperature.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: SimState containing positions, masses, cell, pbc, and other
            required state variables
        kT: Temperature in energy units for initializing momenta,
            scalar or with shape [n_systems]

    Returns:
        MDState: Initialized state for NVE integration containing positions,
            momenta, forces, energy, and other required attributes

    Notes:
        - Initial velocities sampled from Maxwell-Boltzmann distribution
        - Time integration error scales as O(dt²)
    """
    model_output = model(state)

    momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions,
            state.masses,
            state.system_idx,
            kT,
            state.rng,
        )

    return MDState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
    )


def nve_step(
    state: MDState, model: ModelInterface, *, dt: float | torch.Tensor, **_kwargs: Any
) -> MDState:
    r"""Perform one complete NVE (microcanonical) integration step.

    Implements the velocity Verlet algorithm for NVE dynamics, which provides
    energy-conserving, time-reversible integration of Hamilton's equations of motion.

    **Equations** (standard velocity Verlet):

    .. math::

        \mathbf{p}_i(t + \Delta t/2) &= \mathbf{p}_i(t)
            + \frac{\Delta t}{2}\,\mathbf{F}_i(t) \\
        \mathbf{r}_i(t + \Delta t) &= \mathbf{r}_i(t)
            + \Delta t\,\frac{\mathbf{p}_i(t + \Delta t/2)}{m_i} \\
        \mathbf{F}_i(t + \Delta t) &= -\nabla_{\mathbf{r}_i} U\bigl(
            \mathbf{r}(t + \Delta t)\bigr) \\
        \mathbf{p}_i(t + \Delta t) &= \mathbf{p}_i(t + \Delta t/2)
            + \frac{\Delta t}{2}\,\mathbf{F}_i(t + \Delta t)

    **Variable mapping (equation -> code):**

    ============================================  ============================
    Equation symbol                               Code variable
    ============================================  ============================
    :math:`\mathbf{r}_i`  (positions)             ``state.positions``
    :math:`\mathbf{p}_i`  (momenta)               ``state.momenta``
    :math:`m_i`           (masses)                ``state.masses``
    :math:`\mathbf{F}_i`  (forces)                ``state.forces``
    :math:`\Delta t`      (timestep)              ``dt``
    ============================================  ============================

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep, either scalar or shape [n_systems]

    Returns:
        MDState: Updated state after one complete NVE step with new positions,
            momenta, forces, and energy

    Notes:
        - Symplectic, time-reversible integrator of second order accuracy O(dt^2)
        - Conserves energy in the absence of numerical errors
    """
    dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    state = momentum_step(state, dt / 2)
    state = position_step(state, dt)

    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]

    return momentum_step(state, dt / 2)
