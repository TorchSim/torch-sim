"""Functions for computing physical quantities."""

import torch
from torchsim.state import BaseState


# @torch.jit.script
def count_dof(tensor: torch.Tensor) -> int:
    """Count the degrees of freedom in the system."""
    return tensor.numel()


# @torch.jit.script
def temperature(
    momenta: torch.Tensor,
    masses: torch.Tensor,
    velocities: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate temperature from momenta/velocities and masses.
    Temperature returned in energy units.

    Args:
        momenta: Particle momenta, shape (n_particles, n_dim)
        masses: Particle masses, shape (n_particles,)
        velocities: Particle velocities, shape (n_particles, n_dim)
        batch: Optional tensor indicating batch membership of each particle

    Returns:
        Scalar temperature value
    """
    if momenta is not None and velocities is not None:
        raise ValueError("Must pass either momenta or velocities, not both")

    if momenta is None and velocities is None:
        raise ValueError("Must pass either momenta or velocities")

    if momenta is not None:
        # If momentum provided, calculate v^2 = p^2/m^2
        squared_term = (momenta**2) / masses.unsqueeze(-1)
    else:
        # If velocity provided, calculate mv^2
        squared_term = (velocities**2) * masses.unsqueeze(-1)

    if batch is None:
        # Count total degrees of freedom
        dof = count_dof(squared_term)
        return torch.sum(squared_term) / dof
    # Sum squared terms for each batch
    flattened_squared = torch.sum(squared_term, dim=-1)

    # Count degrees of freedom per batch
    batch_sizes = torch.bincount(batch)
    dof_per_batch = batch_sizes * squared_term.shape[-1]  # multiply by n_dimensions

    # Calculate temperature per batch
    batch_sums = torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=batch_sizes
    )
    return batch_sums / dof_per_batch


# @torch.jit.script
def kinetic_energy(
    momenta: torch.Tensor,
    masses: torch.Tensor,
    velocities: torch.Tensor | None = None,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the kinetic energy of a system.

    Args:
        momenta: Particle momenta, shape (n_particles, n_dim)
        masses: Particle masses, shape (n_particles,)
        velocities: Particle velocities, shape (n_particles, n_dim)
        batch: Optional tensor indicating batch membership of each particle

    Returns:
        If batch is None: Scalar tensor containing the total kinetic energy
        If batch is provided: Tensor of kinetic energies per batch
    """
    if momenta is not None and velocities is not None:
        raise ValueError("Must pass either momenta or velocities, not both")
    if momenta is None and velocities is None:
        raise ValueError("Must pass either momenta or velocities")

    if momenta is None:
        # Using velocities
        squared_term = (velocities**2) * masses.unsqueeze(-1)
    else:
        # Using momentum
        squared_term = (momenta**2) / masses.unsqueeze(-1)

    if batch is None:
        return 0.5 * torch.sum(squared_term)
    flattened_squared = torch.sum(squared_term, dim=-1)
    return 0.5 * torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=torch.bincount(batch)
    )


def batchwise_max_force(state: BaseState) -> torch.Tensor:
    batch_wise_max_force = torch.zeros(
        state.n_batches, device=state.device, dtype=state.dtype
    )
    max_forces = state.forces.norm(dim=1)
    return batch_wise_max_force.scatter_reduce(
        dim=0,
        index=state.batch,
        src=max_forces,
        reduce="amax",
    )
