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

from dataclasses import dataclass

import torch

from torch_sim import transforms
from torch_sim.state import SimState


@dataclass
class MDState(SimState):
    """State information for molecular dynamics simulations.

    This class represents the complete state of a molecular system being integrated
    with molecular dynamics. It extends the base SimState class to include additional
    attributes required for MD simulations, such as momenta, energy, and forces.
    The class also provides computed properties like velocities.

    Attributes:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        momenta (torch.Tensor): Particle momenta [n_particles, n_dim]
        energy (torch.Tensor): Total energy of the system [n_batches]
        forces (torch.Tensor): Forces on particles [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        cell (torch.Tensor): Simulation cell matrix [n_batches, n_dim, n_dim]
        pbc (bool): Whether to use periodic boundary conditions
        batch (torch.Tensor): Batch indices [n_particles]
        atomic_numbers (torch.Tensor): Atomic numbers [n_particles]

    Properties:
        velocities (torch.Tensor): Particle velocities [n_particles, n_dim]
        n_batches (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    momenta: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor

    @property
    def velocities(self) -> torch.Tensor:
        """Velocities calculated from momenta and masses with shape
        [n_particles, n_dimensions].
        """
        return self.momenta / self.masses.unsqueeze(-1)


def calculate_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    batch: torch.Tensor,
    kT: torch.Tensor | float,
    seed: int | None = None,
) -> torch.Tensor:
    """Initialize particle momenta based on temperature.

    Generates random momenta for particles following the Maxwell-Boltzmann
    distribution at the specified temperature. The center of mass motion
    is removed to prevent system drift.

    Args:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        batch (torch.Tensor): Batch indices [n_particles]
        kT (torch.Tensor): Temperature in energy units [n_batches]
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        torch.Tensor: Initialized momenta [n_particles, n_dim]
    """
    device = positions.device
    dtype = positions.dtype

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    if isinstance(kT, torch.Tensor) and len(kT.shape) > 0:
        # kT is a tensor with shape (n_batches,)
        kT = kT[batch]

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    batchwise_momenta = torch.zeros(
        (batch[-1] + 1, momenta.shape[1]), device=device, dtype=dtype
    )

    # create 3 copies of batch
    batch_3 = batch.view(-1, 1).repeat(1, 3)
    bincount = torch.bincount(batch)
    mean_momenta = torch.scatter_reduce(
        batchwise_momenta,
        dim=0,
        index=batch_3,
        src=momenta,
        reduce="sum",
    ) / bincount.view(-1, 1)

    return torch.where(
        torch.repeat_interleave(bincount > 1, bincount).view(-1, 1),
        momenta - mean_momenta[batch],
        momenta,
    )


def momentum_step(state: MDState, dt: torch.Tensor) -> MDState:
    """Update particle momenta using current forces.

    This function performs the momentum update step of velocity Verlet integration
    by applying forces over the timestep dt. It implements the equation:
    p(t+dt) = p(t) + F(t) * dt

    Args:
        state (MDState): Current system state containing forces and momenta
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_batches]

    Returns:
        MDState: Updated state with new momenta after force application

    """
    new_momenta = state.momenta + state.forces * dt
    state.momenta = new_momenta
    return state


def position_step(state: MDState, dt: torch.Tensor) -> MDState:
    """Update particle positions using current velocities.

    This function performs the position update step of velocity Verlet integration
    by propagating particles according to their velocities over timestep dt.
    It implements the equation: r(t+dt) = r(t) + v(t) * dt

    Args:
        state (MDState): Current system state containing positions and velocities
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_batches]

    Returns:
        MDState: Updated state with new positions after propagation

    """
    new_positions = state.positions + state.velocities * dt

    if state.pbc:
        # Split positions and cells by batch
        new_positions = transforms.pbc_wrap_batched(
            new_positions, state.cell, state.batch
        )

    state.positions = new_positions
    return state
