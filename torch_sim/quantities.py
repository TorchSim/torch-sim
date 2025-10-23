"""Functions for computing physical quantities."""

from typing import TYPE_CHECKING

import torch

from torch_sim.units import MetalUnits


if TYPE_CHECKING:
    from torch_sim.integrators.md import MDState
    from torch_sim.optimizers import OptimState


# @torch.jit.script
def calc_kT(  # noqa: N802
    *,
    masses: torch.Tensor,
    momenta: torch.Tensor | None = None,
    velocities: torch.Tensor | None = None,
    system_idx: torch.Tensor | None = None,
    dof_per_system: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate temperature in energy units from momenta/velocities and masses.

    Args:
        momenta (torch.Tensor): Particle momenta, shape (n_particles, n_dim)
        masses (torch.Tensor): Particle masses, shape (n_particles,)
        velocities (torch.Tensor | None): Particle velocities, shape (n_particles, n_dim)
        system_idx (torch.Tensor | None): Optional tensor indicating system membership of
        each particle
        dof_per_system (torch.Tensor | None): Optional tensor indicating
        degrees of freedom per system

    Returns:
        torch.Tensor: Scalar temperature value
    """
    if not ((momenta is not None) ^ (velocities is not None)):
        raise ValueError("Must pass either one of momenta or velocities")

    if momenta is None and velocities is not None:
        # If velocity provided, calculate mv^2
        squared_term = torch.square(velocities) * masses.unsqueeze(-1)
    elif momenta is not None and velocities is None:
        # If momentum provided, calculate v^2 = p^2/m^2
        squared_term = torch.square(momenta) / masses.unsqueeze(-1)
    else:
        raise ValueError("Must pass either one of momenta or velocities")

    if system_idx is None:
        # Count total degrees of freedom
        dof = squared_term.numel()
        return torch.sum(squared_term) / dof
    # Sum squared terms for each system
    flattened_squared = torch.sum(squared_term, dim=-1)

    # Count degrees of freedom per system
    system_sizes = torch.bincount(system_idx)
    if dof_per_system is None:
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

    if momenta is None and velocities is not None:  # Using velocities
        squared_term = torch.square(velocities) * masses.unsqueeze(-1)
    elif momenta is not None and velocities is None:  # Using momenta
        squared_term = torch.square(momenta) / masses.unsqueeze(-1)
    else:
        raise ValueError("Must pass either one of momenta or velocities")

    if system_idx is None:
        return 0.5 * torch.sum(squared_term)
    flattened_squared = torch.sum(squared_term, dim=-1)
    return 0.5 * torch.segment_reduce(
        flattened_squared, reduce="sum", lengths=torch.bincount(system_idx)
    )


def get_pressure(
    stress: torch.Tensor,
    kinetic_energy: float | torch.Tensor,
    volume: torch.Tensor,
    dim: int = 3,
) -> torch.Tensor:
    """Compute the pressure from the stress tensor.

    The stress tensor is defined as 1/volume * dU/de_ij
    So the pressure is -1/volume * trace(dU/de_ij)
    """
    return 1 / dim * ((2 * kinetic_energy / volume) - torch.einsum("...ii", stress))


def calc_heat_flux(
    momenta: torch.Tensor | None,
    masses: torch.Tensor,
    velocities: torch.Tensor | None,
    energies: torch.Tensor,
    stresses: torch.Tensor,
    batch: torch.Tensor | None = None,
    *,  # Force keyword arguments for booleans
    is_centroid_stress: bool = False,
    is_virial_only: bool = False,
) -> torch.Tensor:
    r"""Calculate the heat flux vector.

    Computes the microscopic heat flux, :math:`\mathbf{J}`
    defined as:

    .. math::
        \mathbf{J} = \mathbf{J}^c + \mathbf{J}^v

    where the convective part :math:`\mathbf{J}^c` and virial part
    :math:`\mathbf{J}^v` are:

    .. math::
        \mathbf{J}^c &= \sum_i \epsilon_i \mathbf{v}_i \\
        \mathbf{J}^v &= \sum_i \sum_j \mathbf{S}_{ij} \cdot \mathbf{v}_j

    where :math:`\epsilon_i` is the per-atom energy (p.e. + k.e.),
    :math:`\mathbf{v}_i` is velocity, and :math:`\mathbf{S}_{ij}` is the
    per-atom stress tensor.

    Args:
        momenta: Particle momenta, shape (n_particles, n_dim)
        masses: Particle masses, shape (n_particles,)
        velocities: Particle velocities, shape (n_particles, n_dim)
        energies: Per-atom energies (p.e. + k.e.), shape (n_particles,)
        stresses: Per-atom stress tensor components:
            - If is_centroid_stress=False: shape (n_particles, 6) for
              :math:`[\sigma_{xx}, \sigma_{yy}, \sigma_{zz},
              \sigma_{xy}, \sigma_{xz}, \sigma_{yz}]`
            - If is_centroid_stress=True: shape (n_particles, 9) for
              :math:`[\mathbf{r}_{ix}f_{ix}, \mathbf{r}_{iy}f_{iy},
              \mathbf{r}_{iz}f_{iz}, \mathbf{r}_{ix}f_{iy},
              \mathbf{r}_{ix}f_{iz}, \mathbf{r}_{iy}f_{iz},
              \mathbf{r}_{iy}f_{ix}, \mathbf{r}_{iz}f_{ix},
              \mathbf{r}_{iz}f_{iy}]`
        batch: Optional tensor indicating system membership
        is_centroid_stress: Whether stress uses centroid formulation
        is_virial_only: If True, returns only virial part :math:`\mathbf{J}^v`

    Returns:
        Heat flux vector of shape (3,) or (n_systems, 3)
    """
    if momenta is not None and velocities is not None:
        raise ValueError("Must pass either momenta or velocities, not both")
    if momenta is None and velocities is None:
        raise ValueError("Must pass either momenta or velocities")

    # Deduce velocities
    if velocities is None:
        velocities = momenta / masses.unsqueeze(-1)

    convective_flux = energies.unsqueeze(-1) * velocities

    # Calculate virial flux
    if is_centroid_stress:
        # Centroid formulation: r_i[x,y,z] . f_i[x,y,z]
        virial_x = -(
            stresses[:, 0] * velocities[:, 0]  # r_ix.f_ix.v_x
            + stresses[:, 3] * velocities[:, 1]  # r_ix.f_iy.v_y
            + stresses[:, 4] * velocities[:, 2]  # r_ix.f_iz.v_z
        )
        virial_y = -(
            stresses[:, 6] * velocities[:, 0]  # r_iy.f_ix.v_x
            + stresses[:, 1] * velocities[:, 1]  # r_iy.f_iy.v_y
            + stresses[:, 5] * velocities[:, 2]  # r_iy.f_iz.v_z
        )
        virial_z = -(
            stresses[:, 7] * velocities[:, 0]  # r_iz.f_ix.v_x
            + stresses[:, 8] * velocities[:, 1]  # r_iz.f_iy.v_y
            + stresses[:, 2] * velocities[:, 2]  # r_iz.f_iz.v_z
        )
    else:
        # Standard stress tensor components
        virial_x = -(
            stresses[:, 0] * velocities[:, 0]  # s_xx.v_x
            + stresses[:, 3] * velocities[:, 1]  # s_xy.v_y
            + stresses[:, 4] * velocities[:, 2]  # s_xz.v_z
        )
        virial_y = -(
            stresses[:, 3] * velocities[:, 0]  # s_xy.v_x
            + stresses[:, 1] * velocities[:, 1]  # s_yy.v_y
            + stresses[:, 5] * velocities[:, 2]  # s_yz.v_z
        )
        virial_z = -(
            stresses[:, 4] * velocities[:, 0]  # s_xz.v_x
            + stresses[:, 5] * velocities[:, 1]  # s_yz.v_y
            + stresses[:, 2] * velocities[:, 2]  # s_zz.v_z
        )

    virial_flux = torch.stack([virial_x, virial_y, virial_z], dim=-1)

    if batch is None:
        # All atoms
        virial_sum = torch.sum(virial_flux, dim=0)
        if is_virial_only:
            return virial_sum
        conv_sum = torch.sum(convective_flux, dim=0)
        return conv_sum + virial_sum

    # All atoms in each system
    n_systems = int(torch.max(batch) + 1)
    virial_sum = torch.zeros(
        (n_systems, 3), device=velocities.device, dtype=velocities.dtype
    )
    virial_sum.scatter_add_(0, batch.unsqueeze(-1).expand(-1, 3), virial_flux)

    if is_virial_only:
        return virial_sum

    conv_sum = torch.zeros(
        (n_systems, 3), device=velocities.device, dtype=velocities.dtype
    )
    conv_sum.scatter_add_(0, batch.unsqueeze(-1).expand(-1, 3), convective_flux)
    return conv_sum + virial_sum


def system_wise_max_force[T: MDState | OptimState](state: T) -> torch.Tensor:
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
