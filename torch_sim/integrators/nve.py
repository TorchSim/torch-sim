"""Implementations of NVE integrators."""

from typing import Any

import torch

from torch_sim.integrators.md import (
    MDState,
    calculate_momenta,
    momentum_step,
    position_step,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.typing import StateDict


def nve_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: torch.Tensor,
    seed: int | None = None,
    **_kwargs: Any,
) -> MDState:
    """Initialize an NVE state from input data.

    Creates an initial state for NVE molecular dynamics by computing initial
    energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
    at the specified temperature.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Either a SimState object or a dictionary containing positions,
            masses, cell, pbc, and other required state variables
        kT: Temperature in energy units for initializing momenta,
            scalar or with shape [n_systems]
        seed: Random seed for reproducibility

    Returns:
        MDState: Initialized state for NVE integration containing positions,
            momenta, forces, energy, and other required attributes

    Notes:
        - Initial velocities sampled from Maxwell-Boltzmann distribution
        - Time integration error scales as O(dt²)
    """
    if not isinstance(state, SimState):
        state = SimState(**state)

    model_output = model(state)

    momenta = getattr(
        state,
        "momenta",
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    return MDState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
    )


def nve_step(
    state: MDState, model: ModelInterface, *, dt: torch.Tensor, **_kwargs: Any
) -> MDState:
    """Perform one complete NVE (microcanonical) integration step.

    This function implements the velocity Verlet algorithm for NVE dynamics,
    which provides energy-conserving time evolution. The integration sequence is:
    1. Half momentum update using current forces
    2. Full position update using updated momenta
    3. Force update at new positions
    4. Half momentum update using new forces

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep, either scalar or shape [n_systems]

    Returns:
        MDState: Updated state after one complete NVE step with new positions,
            momenta, forces, and energy

    Notes:
        - Uses velocity Verlet algorithm for time reversible integration
        - Conserves energy in the absence of numerical errors
        - Handles periodic boundary conditions if enabled in state
        - Symplectic integrator preserving phase space volume
    """
    state = momentum_step(state, dt / 2)
    state = position_step(state, dt)

    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]

    return momentum_step(state, dt / 2)
