"""Functions for computing physical quantities."""

from typing import cast

import torch

from torch_sim.state import SimState
from torch_sim.units import MetalUnits


# @torch.jit.script
def count_dof(tensor: torch.Tensor) -> int:
    """Count the degrees of freedom in the system.

    Args:
        tensor: Tensor to count the degrees of freedom in

    Returns:
        Number of degrees of freedom
    """
    return tensor.numel()


# @torch.jit.script
def calc_kT(  # noqa: N802
    *,
    masses: torch.Tensor,
    momenta: torch.Tensor | None = None,
    velocities: torch.Tensor | None = None,
    system_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate temperature in energy units from momenta/velocities and masses.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        system_idx (torch.Tensor | None): Optional tensor indicating system membership of
        each particle

    Returns:
        torch.Tensor: Scalar temperature value
    """
    if not ((momenta is not None) ^ (velocities is not None)):
        raise ValueError("Must pass either one of momenta or velocities")

    if momenta is None:
        # If velocity provided, calculate mv^2
        velocities = cast("torch.Tensor", velocities)
        squared_term = (velocities**2) * masses.unsqueeze(-1)
    else:
        # If momentum provided, calculate v^2 = p^2/m^2
        squared_term = (momenta**2) / masses.unsqueeze(-1)

    if system_idx is None:
        # Count total degrees of freedom
        dof = count_dof(squared_term)
        return torch.sum(squared_term) / dof
    # Sum squared terms for each system
    flattened_squared = torch.sum(squared_term, dim=-1)

    # Count degrees of freedom per system
    system_sizes = torch.bincount(system_idx)
    dof_per_system = system_sizes * squared_term.shape[-1]  # multiply by n_dimensions

    # Calculate temperature per system
    system_sums = torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=system_sizes
    )
    return system_sums / dof_per_system


def calc_temperature(
    *,
    masses: torch.Tensor,
    momenta: torch.Tensor | None = None,
    velocities: torch.Tensor | None = None,
    system_idx: torch.Tensor | None = None,
    units: MetalUnits = MetalUnits.temperature,
) -> torch.Tensor:
    """Calculate temperature from momenta/velocities and masses.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        system_idx (torch.Tensor | None): Optional tensor indicating system membership of
        each particle
        units (object): Units to return the temperature in

    Returns:
        torch.Tensor: Temperature value in specified units
    """
    kT = calc_kT(
        masses=masses, momenta=momenta, velocities=velocities, system_idx=system_idx
    )
    return kT / units


# @torch.jit.script
def calc_kinetic_energy(
    *,
    masses: torch.Tensor,
    momenta: torch.Tensor | None = None,
    velocities: torch.Tensor | None = None,
    system_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes the kinetic energy of a system.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        system_idx (torch.Tensor | None): Optional tensor indicating system membership of
        each particle

    Returns:
        If system_idx is None: Scalar tensor containing the total kinetic energy
        If system_idx is provided: Tensor of kinetic energies per system
    """
    if not ((momenta is not None) ^ (velocities is not None)):
        raise ValueError("Must pass either one of momenta or velocities")

    if momenta is None:  # Using velocities
        squared_term = (velocities**2) * masses.unsqueeze(-1)
    else:  # Using momenta
        squared_term = (momenta**2) / masses.unsqueeze(-1)

    if system_idx is None:
        return 0.5 * torch.sum(squared_term)
    flattened_squared = torch.sum(squared_term, dim=-1)
    return 0.5 * torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=torch.bincount(system_idx)
    )


def get_pressure(
    stress: torch.Tensor, kinetic_energy: torch.Tensor, volume: torch.Tensor, dim: int = 3
) -> torch.Tensor:
    """Compute the pressure from the stress tensor.

    The stress tensor is defined as 1/volume * dU/de_ij
    So the pressure is -1/volume * trace(dU/de_ij)
    """
    return 1 / dim * ((2 * kinetic_energy / volume) - torch.einsum("...ii", stress))


def systemwise_max_force(state: SimState) -> torch.Tensor:
    """Compute the maximum force per system.

    Args:
        state (SimState): State to compute the maximum force per system for.

    Returns:
        torch.Tensor: Maximum forces per system
    """
    system_wise_max_force = torch.zeros(
        state.n_systems, device=state.device, dtype=state.dtype
    )
    max_forces = state.forces.norm(dim=1)
    return system_wise_max_force.scatter_reduce(
        dim=0, index=state.system_idx, src=max_forces, reduce="amax"
    )
