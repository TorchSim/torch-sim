"""Implementations of NVE integrators."""

from collections.abc import Callable

import torch

from torch_sim.integrators.md import (
    MDState,
    calculate_momenta,
    momentum_step,
    position_step,
)
from torch_sim.state import SimState
from torch_sim.typing import StateDict


def nve(
    model: torch.nn.Module,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    seed: int | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NVE (microcanonical) integrator.

    This function sets up integration in the NVE ensemble, where particle number (N),
    volume (V), and total energy (E) are conserved. It returns both an initialization
    function and an update function for time evolution.

    The initialization function samples initial momenta from a Maxwell-Boltzmann
    distribution at the specified temperature, while the update function
    implements the velocity Verlet algorithm for energy-conserving dynamics.

    Args:
        model (torch.nn.Module): Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_systems]
        kT (torch.Tensor): Temperature in energy units for initializing momenta,
            either scalar or with shape [n_systems]
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - callable: Function to initialize the MDState from input data and kT
              with signature: init_fn(state, kT=kT, seed=seed) -> MDState
            - callable: Update function that evolves system by one timestep
              with signature: update_fn(state, dt=dt) -> MDState

    Notes:
        - Uses velocity Verlet algorithm for time-reversible integration
        - Conserves total energy in the absence of numerical errors
        - Initial velocities sampled from Maxwell-Boltzmann distribution
        - Time integration error scales as O(dt²)
    """

    def nve_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        seed: int | None = seed,
    ) -> MDState:
        """Initialize an NVE state from input data.

        Creates an initial state for NVE molecular dynamics by computing initial
        energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
        at the specified temperature.

        Args:
            state (SimState | StateDict): Either a SimState object or a dictionary
                containing positions, masses, cell, pbc, and other required state
                variables
            kT (torch.Tensor): Temperature in energy units for initializing momenta,
                scalar or with shape [n_systems]
            seed (int, optional): Random seed for reproducibility

        Returns:
            MDState: Initialized state for NVE integration containing positions,
                momenta, forces, energy, and other required attributes
        """
        # Extract required data from input
        if not isinstance(state, SimState):
            state = SimState(**state)

        model_output = model(state)

        momenta = getattr(
            state,
            "momenta",
            calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
        )

        initial_state = MDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            system_idx=state.system_idx,
            atomic_numbers=state.atomic_numbers,
        )
        return initial_state  # noqa: RET504

    def nve_update(state: MDState, dt: torch.Tensor = dt, **_) -> MDState:
        """Perform one complete NVE (microcanonical) integration step.

        This function implements the velocity Verlet algorithm for NVE dynamics,
        which provides energy-conserving time evolution. The integration sequence is:
        1. Half momentum update using current forces
        2. Full position update using updated momenta
        3. Force update at new positions
        4. Half momentum update using new forces

        Args:
            state (MDState): Current system state containing positions, momenta, forces
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
            **_: Additional unused keyword arguments (for compatibility)

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

    return nve_init, nve_update
