"""Implementations of NPT integrators."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import torch_sim as ts
from torch_sim.integrators.md import (
    MDState,
    NoseHooverChain,
    NoseHooverChainFns,
    calculate_momenta,
    construct_nose_hoover_chain,
    momentum_step,
)
from torch_sim.integrators.nvt import _vrescale_update
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.typing import StateDict


@dataclass(kw_only=True)
class NPTLangevinState(MDState):
    """State information for an NPT system with Langevin dynamics.

    This class represents the complete state of a molecular system being integrated
    in the NPT (constant particle number, pressure, temperature) ensemble using
    Langevin dynamics. In addition to particle positions and momenta, it tracks
    cell dimensions and their dynamics for volume fluctuations.

    Attributes:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        velocities (torch.Tensor): Particle velocities [n_particles, n_dim]
        energy (torch.Tensor): Energy of the system [n_systems]
        forces (torch.Tensor): Forces on particles [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        cell (torch.Tensor): Simulation cell matrix [n_systems, n_dim, n_dim]
        pbc (bool): Whether to use periodic boundary conditions
        system_idx (torch.Tensor): System indices [n_particles]
        atomic_numbers (torch.Tensor): Atomic numbers [n_particles]
        stress (torch.Tensor): Stress tensor [n_systems, n_dim, n_dim]
        reference_cell (torch.Tensor): Original cell vectors used as reference for
            scaling [n_systems, n_dim, n_dim]
        cell_positions (torch.Tensor): Cell positions [n_systems, n_dim, n_dim]
        cell_velocities (torch.Tensor): Cell velocities [n_systems, n_dim, n_dim]
        cell_masses (torch.Tensor): Masses associated with the cell degrees of freedom
            shape [n_systems]

    Properties:
        momenta (torch.Tensor): Particle momenta calculated as velocities*masses
            with shape [n_particles, n_dimensions]
        n_systems (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    # System state variables
    stress: torch.Tensor

    alpha: torch.Tensor
    cell_alpha: torch.Tensor
    b_tau: torch.Tensor

    # Cell variables
    reference_cell: torch.Tensor
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_masses: torch.Tensor

    _system_attributes = MDState._system_attributes | {  # noqa: SLF001
        "stress",
        "cell_positions",
        "cell_velocities",
        "cell_masses",
        "reference_cell",
        "alpha",
        "cell_alpha",
        "b_tau",
    }


def _npt_langevin_beta(
    state: NPTLangevinState,
    kT: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Calculate random noise term for particle Langevin dynamics.

    This function generates the stochastic force term for the Langevin thermostat
    according to the fluctuation-dissipation theorem, ensuring proper thermal
    sampling at the target temperature.

    Args:
        state (NPTLangevinState): Current NPT state
        alpha (torch.Tensor): Friction coefficient, either scalar or
            shape [n_systems]
        kT (torch.Tensor): Temperature in energy units, either scalar or
            shape [n_systems]
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]

    Returns:
        torch.Tensor: Random noise term for force calculation [n_particles, n_dim]
    """
    # Generate system-specific noise with correct shape
    noise = torch.randn_like(state.momenta)

    # Calculate the thermal noise amplitude by system
    batch_kT = kT
    if kT.ndim == 0:
        batch_kT = kT.expand(state.n_systems)

    # Map system kT to atoms
    atom_kT = batch_kT[state.system_idx]

    # Calculate the prefactor for each atom
    # The standard deviation should be sqrt(2*alpha*kB*T*dt)
    prefactor = torch.sqrt(2 * state.alpha * atom_kT * dt)

    return prefactor.unsqueeze(-1) * noise


def _npt_langevin_cell_beta(
    state: NPTLangevinState,
    kT: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Generate random noise for cell fluctuations in NPT dynamics.

    This function creates properly scaled random noise for cell dynamics in NPT
    simulations, following the fluctuation-dissipation theorem to ensure correct
    thermal sampling of cell degrees of freedom.

    Args:
        state (NPTLangevinState): Current NPT state
        cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
            with shape [n_systems]
        kT (torch.Tensor): System temperature in energy units, either scalar or
            with shape [n_systems]
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
        device (torch.device): Device for tensor operations
        dtype (torch.dtype): Data type for tensor operations

    Returns:
        torch.Tensor: Scaled random noise for cell dynamics with shape
            [n_systems, n_dimensions, n_dimensions]
    """
    # Generate standard normal distribution (zero mean, unit variance)
    noise = torch.randn_like(state.cell_positions, device=state.device, dtype=state.dtype)

    if kT.ndim == 0:
        kT = kT.expand(state.n_systems)

    # Reshape for broadcasting
    cell_alpha_expanded = state.cell_alpha.view(-1, 1, 1)  # shape: (n_systems, 1, 1)
    kT = kT.view(-1, 1, 1)  # shape: (n_systems, 1, 1)
    dt = dt.expand(state.n_systems).view(-1, 1, 1) if dt.ndim == 0 else dt.view(-1, 1, 1)

    # Scale to satisfy the fluctuation-dissipation theorem
    # The standard deviation should be sqrt(2*alpha*kB*T*dt)
    scaling_factor = torch.sqrt(2.0 * cell_alpha_expanded * kT * dt)

    return scaling_factor * noise


def _npt_langevin_cell_position_step(
    state: NPTLangevinState,
    dt: torch.Tensor,
    pressure_force: torch.Tensor,
    kT: torch.Tensor,
) -> NPTLangevinState:
    """Update the cell position in NPT dynamics.

    This function updates the cell position (effectively the volume) in NPT dynamics
    using the current cell velocities, pressure forces, and thermal noise. It
    implements the position update part of the Langevin barostat algorithm.

    Args:
        state (NPTLangevinState): Current NPT state
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
        pressure_force (torch.Tensor): Pressure force for barostat
            [n_systems, n_dim, n_dim]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_systems]
        cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
            with shape [n_systems]

    Returns:
        NPTLangevinState: Updated state with new cell positions
    """
    # Calculate effective mass term
    Q_2 = 2 * state.cell_masses.view(-1, 1, 1)  # shape: (n_systems, 1, 1)

    # Ensure parameters have batch dimension
    if dt.ndim == 0:
        dt = dt.expand(state.n_systems)

    # Reshape for broadcasting
    dt_expanded = dt.view(-1, 1, 1)
    cell_alpha_expanded = state.cell_alpha.view(-1, 1, 1)

    # Calculate damping factor for cell position update
    cell_b = 1 / (1 + ((cell_alpha_expanded * dt_expanded) / Q_2))

    # Deterministic velocity contribution
    c_1 = cell_b * dt_expanded * state.cell_velocities

    # Force contribution
    c_2 = cell_b * dt_expanded * dt_expanded * pressure_force / Q_2

    # Random noise contribution (thermal fluctuations)
    c_3 = cell_b * dt_expanded * _npt_langevin_cell_beta(state, kT, dt) / Q_2

    # Update cell positions with all contributions
    state.cell_positions = state.cell_positions + c_1 + c_2 + c_3
    return state


def _npt_langevin_cell_velocity_step(
    state: NPTLangevinState,
    F_p_n: torch.Tensor,
    dt: torch.Tensor,
    pressure_force: torch.Tensor,
    kT: torch.Tensor,
) -> NPTLangevinState:
    """Update the cell velocities in NPT dynamics.

    This function updates the cell velocities using a Langevin-type integrator,
    accounting for both deterministic forces from pressure differences and
    stochastic thermal noise. It implements the velocity update part of the
    Langevin barostat algorithm.

    Args:
        state (NPTLangevinState): Current NPT state
        F_p_n (torch.Tensor): Initial pressure force with shape
            [n_systems, n_dimensions, n_dimensions]
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
        pressure_force (torch.Tensor): Final pressure force
            shape [n_systems, n_dim, n_dim]
        cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
            shape [n_systems]
        kT (torch.Tensor): Temperature in energy units, either scalar or
            shape [n_systems]

    Returns:
        NPTLangevinState: Updated state with new cell velocities
    """
    # Ensure parameters have batch dimension
    if dt.ndim == 0:
        dt = dt.expand(state.n_systems)
    if kT.ndim == 0:
        kT = kT.expand(state.n_systems)

    # Reshape for broadcasting - need to maintain 3x3 dimensions
    dt_expanded = dt.view(-1, 1, 1)  # shape: (n_systems, 1, 1)
    cell_alpha_expanded = state.cell_alpha.view(-1, 1, 1)  # shape: (n_systems, 1, 1)

    # Calculate cell masses per system - reshape to match 3x3 cell matrices
    cell_masses_expanded = state.cell_masses.view(-1, 1, 1)  # shape: (n_systems, 1, 1)

    # These factors come from the Langevin integration scheme
    a = (1 - (cell_alpha_expanded * dt_expanded) / cell_masses_expanded) / (
        1 + (cell_alpha_expanded * dt_expanded) / cell_masses_expanded
    )
    b = 1 / (1 + (cell_alpha_expanded * dt_expanded) / cell_masses_expanded)

    # Calculate the three terms for velocity update
    # a will broadcast from (n_systems, 1, 1) to (n_systems, 3, 3)
    c_1 = a * state.cell_velocities  # Damped old velocity

    # Force contribution (average of initial and final forces)
    c_2 = dt_expanded * ((a * F_p_n) + pressure_force) / (2 * cell_masses_expanded)

    # Generate system-specific cell noise with correct shape (n_systems, 3, 3)
    cell_noise = torch.randn_like(state.cell_velocities)

    # Calculate thermal noise amplitude
    noise_prefactor = torch.sqrt(
        2 * cell_alpha_expanded * kT.view(-1, 1, 1) * dt_expanded
    )
    noise_term = noise_prefactor * cell_noise / torch.sqrt(cell_masses_expanded)

    # Random noise contribution
    c_3 = b * noise_term

    # Update velocities with all contributions
    state.cell_velocities = c_1 + c_2 + c_3
    return state


def _npt_langevin_position_step(
    state: NPTLangevinState,
    L_n: torch.Tensor,  # This should be shape (n_systems,)
    dt: torch.Tensor,
    kT: torch.Tensor,
) -> NPTLangevinState:
    """Update the particle positions in NPT dynamics.

    This function updates particle positions accounting for both the changing
    cell dimensions and the particle velocities/forces. It handles the scaling
    of positions due to volume changes as well as the normal position updates
    from velocities.

    Args:
        state (NPTLangevinState): Current NPT state
        L_n (torch.Tensor): Previous cell length scale with shape [n_systems]
        dt: Integration timestep, either scalar or with shape [n_systems]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_systems]
        alpha (torch.Tensor | None): Friction coefficient, either scalar or with
            shape [n_systems].

    Returns:
        NPTLangevinState: Updated state with new positions
    """
    # Calculate effective mass term by system
    # Map masses to have batch dimension
    M_2 = 2 * state.masses.unsqueeze(-1)  # shape: (n_atoms, 1)

    # Calculate new cell length scale (cube root of volume for isotropic scaling)
    L_n_new = torch.pow(
        state.cell_positions.reshape(state.n_systems, -1)[:, 0], 1 / 3
    )  # shape: (n_systems,)

    # Map system-specific L_n and L_n_new to atom-level using system indices
    # Make sure L_n is the right shape (n_systems,) before indexing
    if L_n.ndim != 1 or L_n.shape[0] != state.n_systems:
        # If L_n has wrong shape, calculate it again to ensure correct shape
        L_n = torch.pow(state.cell_positions.reshape(state.n_systems, -1)[:, 0], 1 / 3)

    # Map system-specific values to atoms using system indices
    L_n_atoms = L_n[state.system_idx]  # shape: (n_atoms,)
    L_n_new_atoms = L_n_new[state.system_idx]  # shape: (n_atoms,)

    # Calculate damping factor
    alpha_atoms = state.alpha[state.system_idx]
    dt_atoms = dt
    if dt.ndim > 0:
        dt_atoms = dt[state.system_idx]

    b = 1 / (1 + ((alpha_atoms * dt_atoms) / (2 * state.masses)))

    # Scale positions due to cell volume change
    c_1 = (L_n_new_atoms / L_n_atoms).unsqueeze(-1) * state.positions

    # Time step factor with average length scale
    c_2 = (2 * L_n_new_atoms / (L_n_new_atoms + L_n_atoms)) * b * dt_atoms

    # Generate atom-specific noise
    noise = torch.randn_like(state.momenta)
    batch_kT = kT
    if kT.ndim == 0:
        batch_kT = kT.expand(state.n_systems)
    atom_kT = batch_kT[state.system_idx]

    # Calculate noise prefactor according to fluctuation-dissipation theorem
    noise_prefactor = torch.sqrt(2 * alpha_atoms * atom_kT * dt_atoms)
    noise_term = noise_prefactor.unsqueeze(-1) * noise

    # Velocity and force contributions with random noise
    c_3 = (
        state.velocities + dt_atoms.unsqueeze(-1) * state.forces / M_2 + noise_term / M_2
    )

    # Update positions with all contributions
    state.set_positions(c_1 + c_2.unsqueeze(-1) * c_3)
    return state


def _npt_langevin_velocity_step(
    state: NPTLangevinState,
    forces: torch.Tensor,
    dt: torch.Tensor,
    kT: torch.Tensor,
) -> NPTLangevinState:
    """Update the particle velocities in NPT dynamics.

    This function updates particle velocities using a Langevin-type integrator,
    accounting for both deterministic forces and stochastic thermal noise.
    It implements the velocity update part of the Langevin thermostat algorithm.

    Args:
        state (NPTLangevinState): Current NPT state
        forces: Forces on particles
        dt: Integration timestep, either scalar or with shape [n_systems]
        kT: Target temperature in energy units, either scalar or
            with shape [n_systems]
        alpha (torch.Tensor | None): Friction coefficient, either scalar or with
            shape [n_systems].

    Returns:
        NPTLangevinState: Updated state with new velocities
    """
    # Calculate denominator for update equations
    M_2 = 2 * state.masses  # shape: (n_atoms, 1)

    # Map batch parameters to atom level
    alpha_atoms = state.alpha[state.system_idx]
    dt_atoms = dt
    if dt.ndim > 0:
        dt_atoms = dt[state.system_idx]

    # Calculate damping factors for Langevin integration
    a = (1 - (alpha_atoms * dt_atoms) / M_2) / (1 + (alpha_atoms * dt_atoms) / M_2)
    a = a.unsqueeze(-1)
    b = 1 / (1 + (alpha_atoms * dt_atoms) / M_2).unsqueeze(-1)

    # Velocity contribution with damping
    c_1 = a * state.velocities

    # Force contribution (average of initial and final forces)
    c_2 = dt_atoms.unsqueeze(-1) * ((a * forces) + state.forces) / M_2.unsqueeze(-1)

    # Generate atom-specific noise
    noise = torch.randn_like(state.momenta)
    batch_kT = kT
    if kT.ndim == 0:
        batch_kT = kT.expand(state.n_systems)
    atom_kT = batch_kT[state.system_idx]

    # Calculate noise prefactor according to fluctuation-dissipation theorem
    noise_prefactor = torch.sqrt(2 * alpha_atoms * atom_kT * dt_atoms)
    noise_term = noise_prefactor.unsqueeze(-1) * noise

    # Random noise contribution
    c_3 = b * noise_term / state.masses.unsqueeze(-1)

    # Update momenta (velocities * masses) with all contributions
    new_velocities = c_1 + c_2 + c_3
    # Apply constraints.
    state.set_momenta(new_velocities * state.masses.unsqueeze(-1))
    return state


def _compute_cell_force(
    state: NPTLangevinState,
    external_pressure: torch.Tensor,
    kT: torch.Tensor,
) -> torch.Tensor:
    """Compute forces on the cell for NPT dynamics.

    This function calculates the forces acting on the simulation cell
    based on the difference between internal stress and external pressure,
    plus a kinetic contribution. These forces drive the volume changes
    needed to maintain constant pressure.

    Args:
        state (NPTLangevinState): Current NPT state
        external_pressure (torch.Tensor): Target external pressure, either scalar or
            tensor with shape [n_systems, n_dimensions, n_dimensions]
        kT (torch.Tensor): Temperature in energy units, either scalar or
            shape [n_systems]

    Returns:
        torch.Tensor: Force acting on the cell [n_systems, n_dim, n_dim]
    """
    # Convert external_pressure to tensor if it's not already one
    if not isinstance(external_pressure, torch.Tensor):
        external_pressure = torch.tensor(
            external_pressure, device=state.device, dtype=state.dtype
        )

    # Convert kT to tensor if it's not already one
    if not isinstance(kT, torch.Tensor):
        kT = torch.tensor(kT, device=state.device, dtype=state.dtype)

    # Get current volumes for each batch
    volumes = torch.linalg.det(state.cell)  # shape: (n_systems,)

    # Reshape for broadcasting
    volumes = volumes.view(-1, 1, 1)  # shape: (n_systems, 1, 1)

    # Create pressure tensor (diagonal with external pressure)
    if external_pressure.ndim == 0:
        # Scalar pressure - create diagonal pressure tensors for each batch
        pressure_tensor = external_pressure * torch.eye(
            3, device=state.device, dtype=state.dtype
        )
        pressure_tensor = pressure_tensor.unsqueeze(0).expand(state.n_systems, -1, -1)
    else:
        # Already a tensor with shape compatible with n_systems
        pressure_tensor = external_pressure

    # Calculate virials from stress and external pressure
    # Internal stress is negative of virial tensor divided by volume
    virial = -volumes * (state.stress + pressure_tensor)

    # Add kinetic contribution (kT * Identity)
    batch_kT = kT
    if kT.ndim == 0:
        batch_kT = kT.expand(state.n_systems)

    e_kin_per_atom = batch_kT.view(-1, 1, 1) * torch.eye(
        3, device=state.device, dtype=state.dtype
    ).unsqueeze(0)

    # Correct implementation with scaling by n_atoms_per_system
    return virial + e_kin_per_atom * state.n_atoms_per_system.view(-1, 1, 1)


def npt_langevin_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    cell_alpha: float | torch.Tensor | None = None,
    b_tau: float | torch.Tensor | None = None,
    seed: int | None = None,
    **_kwargs: Any,
) -> NPTLangevinState:
    """Initialize an NPT Langevin state from input data.

    This function creates the initial state for NPT Langevin dynamics,
    setting up all necessary variables including particle velocities,
    cell parameters, and barostat variables. It computes initial forces
    and stress using the provided model.

    Args:
        model (ModelInterface): Neural network model that computes energies, forces,
            and stress. Must return a dict with 'energy', 'forces', and 'stress' keys.
        state (MDState | StateDict): Either a MDState object or a dictionary
            containing positions, masses, cell, pbc
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_systems]
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
        alpha (torch.Tensor, optional): Friction coefficient for particle Langevin
            thermostat, either scalar or shape [n_systems]. Defaults to 1/(100*dt).
        cell_alpha (torch.Tensor, optional): Friction coefficient for cell Langevin
            thermostat, either scalar or shape [n_systems]. Defaults to same as alpha.
        b_tau (torch.Tensor, optional): Barostat time constant controlling how quickly
            the system responds to pressure differences, either scalar or shape
            [n_systems]. Defaults to 1/(1000*dt).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        NPTLangevinState: Initialized state for NPT Langevin integration containing
            all required attributes for particle and cell dynamics

    Notes:
        - The model must provide stress tensor calculations for proper pressure coupling
    """
    device, dtype = model.device, model.dtype

    # Set default values if not provided
    if alpha is None:
        alpha = 1.0 / (100 * dt)  # Default friction based on timestep
    if cell_alpha is None:
        cell_alpha = alpha  # Use same friction for cell by default
    if b_tau is None:
        b_tau = 1 / (1000 * dt)  # Default barostat time constant

    # Convert all parameters to tensors with correct device and dtype
    alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    cell_alpha = torch.as_tensor(cell_alpha, device=device, dtype=dtype)
    b_tau = torch.as_tensor(b_tau, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)
    dt = torch.as_tensor(dt, device=device, dtype=dtype)

    if not isinstance(state, SimState):
        state = SimState(**state)

    if alpha.ndim == 0:
        alpha = alpha.expand(state.n_systems)
    if cell_alpha.ndim == 0:
        cell_alpha = cell_alpha.expand(state.n_systems)
    if b_tau.ndim == 0:
        b_tau = b_tau.expand(state.n_systems)

    # Get model output to initialize forces and stress
    model_output = model(state)

    # Initialize momenta if not provided
    momenta = getattr(
        state,
        "momenta",
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    # Initialize cell parameters
    reference_cell = state.cell.clone()

    # Calculate initial cell_positions (volume)
    cell_positions = (
        torch.linalg.det(state.cell).unsqueeze(-1).unsqueeze(-1)
    )  # shape: (n_systems, 1, 1)

    # Initialize cell velocities to zero
    cell_velocities = torch.zeros((state.n_systems, 3, 3), device=device, dtype=dtype)

    # Calculate cell masses based on system size and temperature
    # This follows standard NPT barostat mass scaling
    n_atoms_per_system = torch.bincount(state.system_idx)
    batch_kT = (
        kT.expand(state.n_systems)
        if isinstance(kT, torch.Tensor) and kT.ndim == 0
        else kT
    )
    cell_masses = (n_atoms_per_system + 1) * batch_kT * b_tau * b_tau

    if state.constraints:
        # warn if constraints are present
        warnings.warn(
            "Constraints are present in the system. "
            "Make sure they are compatible with NPT Langevin dynamics."
            "We recommend not using constraints with NPT dynamics for now.",
            UserWarning,
            stacklevel=3,
        )

    # Create the initial state
    return NPTLangevinState(
        positions=state.positions,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        stress=model_output["stress"],
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        system_idx=state.system_idx,
        atomic_numbers=state.atomic_numbers,
        alpha=alpha,
        b_tau=b_tau,
        reference_cell=reference_cell,
        cell_positions=cell_positions,
        cell_velocities=cell_velocities,
        cell_masses=cell_masses,
        cell_alpha=cell_alpha,
        _constraints=state.constraints,
    )


def npt_langevin_step(
    state: NPTLangevinState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTLangevinState:
    """Perform one complete NPT Langevin dynamics integration step.

    This function implements a modified integration scheme for NPT dynamics,
    handling both atomic and cell updates with Langevin thermostats to maintain
    constant temperature and pressure. The integration scheme couples particle
    motion with cell volume fluctuations.

    Args:
        model (ModelInterface): Neural network model that computes energies, forces,
            and stress. Must return a dict with 'energy', 'forces', and 'stress' keys.
        state (NPTLangevinState): Current NPT state with particle and cell variables
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            shape [n_systems]
        external_pressure (torch.Tensor): Target external pressure, either scalar or
            tensor with shape [n_systems, n_dim, n_dim]
        alpha (torch.Tensor): Position friction coefficient, either scalar or
            shape [n_systems]
        cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
            shape [n_systems]
        b_tau (torch.Tensor): Barostat time constant, either scalar or shape [n_systems]

    Returns:
        NPTLangevinState: Updated NPT state after one timestep with new positions,
            velocities, cell parameters, forces, energy, and stress
    """
    device, dtype = model.device, model.dtype

    # Convert any scalar parameters to tensors with batch dimension if needed
    if isinstance(state.alpha, float):
        state.alpha = torch.tensor(state.alpha, device=device, dtype=dtype)
    if isinstance(kT, float):
        kT = torch.tensor(kT, device=device, dtype=dtype)
    if isinstance(state.cell_alpha, float):
        state.cell_alpha = torch.tensor(state.cell_alpha, device=device, dtype=dtype)
    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    # Make sure parameters have batch dimension if they're scalars
    batch_kT = kT.expand(state.n_systems) if kT.ndim == 0 else kT

    # Update barostat mass based on current temperature
    # This ensures proper coupling between system and barostat
    n_atoms_per_system = torch.bincount(state.system_idx)
    state.cell_masses = (n_atoms_per_system + 1) * batch_kT * torch.square(state.b_tau)

    # Compute model output for current state
    model_output = model(state)
    state.forces = model_output["forces"]
    state.stress = model_output["stress"]

    # Store initial values for integration
    forces = state.forces
    F_p_n = _compute_cell_force(state=state, external_pressure=external_pressure, kT=kT)
    L_n = torch.pow(
        state.cell_positions.reshape(state.n_systems, -1)[:, 0], 1 / 3
    )  # shape: (n_systems,)

    # Step 1: Update cell position
    state = _npt_langevin_cell_position_step(state, dt, F_p_n, kT)

    # Update cell (currently only isotropic fluctuations)
    dim = state.positions.shape[1]  # Usually 3 for 3D
    # V_0 and V are shape: (n_systems,)
    V_0 = torch.linalg.det(state.reference_cell)
    V = state.cell_positions.reshape(state.n_systems, -1)[:, 0]

    # Scale cell uniformly in all dimensions
    scaling = (V / V_0) ** (1.0 / dim)  # shape: (n_systems,)

    # Apply scaling to reference cell to get new cell
    new_cell = torch.zeros_like(state.cell)
    for sys_idx in range(state.n_systems):
        new_cell[sys_idx] = scaling[sys_idx] * state.reference_cell[sys_idx]

    state.cell = new_cell

    # Step 2: Update particle positions
    state = _npt_langevin_position_step(state, L_n, dt, kT)

    # Recompute model output after position updates
    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]
    state.stress = model_output["stress"]

    # Compute updated pressure force
    F_p_n_new = _compute_cell_force(
        state=state, external_pressure=external_pressure, kT=kT
    )

    # Step 3: Update cell velocities
    state = _npt_langevin_cell_velocity_step(state, F_p_n, dt, F_p_n_new, kT)

    # Step 4: Update particle velocities
    return _npt_langevin_velocity_step(state, forces, dt, kT)


@dataclass(kw_only=True)
class NPTNoseHooverState(MDState):
    """State information for an NPT system with Nose-Hoover chain thermostats.

    This class represents the complete state of a molecular system being integrated
    in the NPT (constant particle number, pressure, temperature) ensemble using
    Nose-Hoover chain thermostats for both temperature and pressure control.

    The cell dynamics are parameterized using a logarithmic coordinate system where
    cell_position = (1/d)ln(V/V_0), with V being the current volume, V_0 the reference
    volume, and d the spatial dimension. This ensures volume positivity and simplifies
    the equations of motion.

    Attributes:
        positions (torch.Tensor): Particle positions with shape [n_particles, n_dims]
        momenta (torch.Tensor): Particle momenta with shape [n_particles, n_dims]
        forces (torch.Tensor): Forces on particles with shape [n_particles, n_dims]
        masses (torch.Tensor): Particle masses with shape [n_particles]
        reference_cell (torch.Tensor): Reference simulation cell matrix with shape
            [n_systems, n_dimensions, n_dimensions]. Used to measure relative volume
            changes.
        cell_position (torch.Tensor): Logarithmic cell coordinate with shape [n_systems].
            Represents (1/d)ln(V/V_0) where V is current volume and V_0 is reference
            volume.
        cell_momentum (torch.Tensor): Cell momentum (velocity) conjugate to cell_position
            with shape [n_systems]. Controls volume changes.
        cell_mass (torch.Tensor): Mass parameter for cell dynamics with shape [n_systems].
            Controls coupling between volume fluctuations and pressure.
        barostat (NoseHooverChain): Chain thermostat coupled to cell dynamics for
            pressure control
        thermostat (NoseHooverChain): Chain thermostat coupled to particle dynamics
            for temperature control
        barostat_fns (NoseHooverChainFns): Functions for barostat chain updates
        thermostat_fns (NoseHooverChainFns): Functions for thermostat chain updates

    Properties:
        velocities (torch.Tensor): Particle velocities computed as momenta
            divided by masses. Shape: [n_particles, n_dimensions]
        current_cell (torch.Tensor): Current simulation cell matrix derived from
            cell_position. Shape: [n_systems, n_dimensions, n_dimensions]

    Notes:
        - The cell parameterization ensures volume positivity
        - Nose-Hoover chains provide deterministic control of T and P
        - Extended system approach conserves an extended Hamiltonian
        - Time-reversible when integrated with appropriate algorithms
        - All cell-related properties now support batch dimensions
    """

    # Cell variables - now with batch dimensions
    reference_cell: torch.Tensor  # [n_systems, 3, 3]
    cell_position: torch.Tensor  # [n_systems]
    cell_momentum: torch.Tensor  # [n_systems]
    cell_mass: torch.Tensor  # [n_systems]

    # Thermostat variables
    thermostat: NoseHooverChain
    thermostat_fns: NoseHooverChainFns

    # Barostat variables
    barostat: NoseHooverChain
    barostat_fns: NoseHooverChainFns

    _system_attributes = MDState._system_attributes | {  # noqa: SLF001
        "reference_cell",
        "cell_position",
        "cell_momentum",
        "cell_mass",
    }
    _global_attributes = MDState._global_attributes | {  # noqa: SLF001
        "thermostat",
        "barostat",
        "thermostat_fns",
        "barostat_fns",
    }

    @property
    def velocities(self) -> torch.Tensor:
        """Calculate particle velocities from momenta and masses.

        Returns:
            torch.Tensor: Particle velocities with shape [n_particles, n_dimensions]
        """
        return self.momenta / self.masses.unsqueeze(-1)

    @property
    def current_cell(self) -> torch.Tensor:
        """Calculate current simulation cell from cell position.

        The cell is computed from the reference cell and cell_position using:
        cell = (V/V_0)^(1/d) * reference_cell
        where V = V_0 * exp(d * cell_position)

        Returns:
            torch.Tensor: Current simulation cell matrix with shape
                [n_systems, n_dimensions, n_dimensions]
        """
        dim = self.positions.shape[1]
        V_0 = torch.det(self.reference_cell)  # [n_systems]
        V = V_0 * torch.exp(dim * self.cell_position)  # [n_systems]
        scale = (V / V_0) ** (1.0 / dim)  # [n_systems]
        # Expand scale to [n_systems, 1, 1] for broadcasting
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        return scale * self.reference_cell

    def get_number_of_degrees_of_freedom(self) -> torch.Tensor:
        """Calculate degrees of freedom per system."""
        dof = super().get_number_of_degrees_of_freedom()
        return dof - 3  # Subtract 3 degrees of freedom for center of mass motion


def _npt_nose_hoover_cell_info(
    state: NPTNoseHooverState,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Gets the current volume and a function to compute the cell from volume.

    This helper function computes the current system volume and returns a function
    that can compute the simulation cell for any given volume. This is useful for
    integration algorithms that need to update the cell based on volume changes.

    Args:
        state (NPTNoseHooverState): Current state of the NPT system

    Returns:
        tuple:
            - torch.Tensor: Current system volume with shape [n_systems]
            - callable: Function that takes a volume tensor [n_systems] and returns
                the corresponding cell matrix [n_systems, n_dimensions, n_dimensions]

    Notes:
        - Uses logarithmic cell coordinate parameterization
        - Volume changes are measured relative to reference cell
        - Cell scaling preserves shape while changing volume
        - Supports batched operations
    """
    dim = state.positions.shape[1]
    ref = state.reference_cell  # [n_systems, dim, dim]
    V_0 = torch.det(ref)  # [n_systems] - Reference volume
    V = V_0 * torch.exp(dim * state.cell_position)  # [n_systems] - Current volume

    def volume_to_cell(V: torch.Tensor) -> torch.Tensor:
        """Compute cell matrix for given volumes.

        Args:
            V (torch.Tensor): Volumes with shape [n_systems]

        Returns:
            torch.Tensor: Cell matrices with shape [n_systems, dim, dim]
        """
        scale = torch.pow(V / V_0, 1.0 / dim)  # [n_systems]
        # Expand scale to [n_systems, 1, 1] for broadcasting
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        return scale * ref

    return V, volume_to_cell


def _npt_nose_hoover_update_cell_mass(
    state: NPTNoseHooverState,
    kT: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> NPTNoseHooverState:
    """Update the cell mass parameter in an NPT simulation.

    This function updates the mass parameter associated with cell volume fluctuations
    based on the current system size and target temperature. The cell mass controls
    how quickly the volume can change and is chosen to maintain stable pressure
    control.

    Args:
        state (NPTNoseHooverState): Current state of the NPT system
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            shape [n_systems]
        device (torch.device): Device for tensor operations
        dtype (torch.dtype): Data type for tensor operations

    Returns:
        NPTNoseHooverState: Updated state with new cell mass

    Notes:
        - Cell mass scales with system size (N+1) and dimensionality
        - Larger cell mass gives slower but more stable volume fluctuations
        - Mass depends on barostat relaxation time (tau)
        - Supports batched operations
    """
    _n_particles, dim = state.positions.shape

    # Convert kT to tensor if it's not already one
    if not isinstance(kT, torch.Tensor):
        kT = torch.tensor(kT, device=device, dtype=dtype)

    # Handle both scalar and batched kT
    kT_system = kT.expand(state.n_systems) if kT.ndim == 0 else kT

    # Calculate cell masses for each system
    n_atoms_per_system = torch.bincount(state.system_idx, minlength=state.n_systems)
    cell_mass = (
        dim * (n_atoms_per_system + 1) * kT_system * torch.square(state.barostat.tau)
    )

    # Update state with new cell masses
    state.cell_mass = cell_mass.to(device=device, dtype=dtype)
    return state


def _npt_nose_hoover_sinhx_x(x: torch.Tensor) -> torch.Tensor:
    """Compute sinh(x)/x using Taylor series expansion near x=0.

    This function implements a Taylor series approximation of sinh(x)/x that is
    accurate near x=0. The series expansion is:
    sinh(x)/x = 1 + x²/6 + x⁴/120 + x⁶/5040 + x⁸/362880 + x¹⁰/39916800

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Approximation of sinh(x)/x

    Notes:
        - Uses 6 terms of Taylor series for good accuracy near x=0
        - Relative error < 1e-12 for |x| < 0.5
        - More efficient than direct sinh(x)/x computation for small x
        - Avoids division by zero at x=0

    Example:
        >>> x = torch.tensor([0.0, 0.1, 0.2])
        >>> y = sinhx_x(x)
        >>> print(y)  # tensor([1, 1.0017, 1.0067])
    """
    return (
        1
        + torch.pow(x, 2) / 6
        + torch.pow(x, 4) / 120
        + torch.pow(x, 6) / 5040
        + torch.pow(x, 8) / 362_880
        + torch.pow(x, 10) / 39_916_800
    )


def _npt_nose_hoover_exp_iL1(  # noqa: N802
    state: NPTNoseHooverState,
    velocities: torch.Tensor,
    cell_velocity: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Apply the exp(iL1) operator for NPT dynamics position updates.

    This function implements the position update operator for NPT dynamics using
    a symplectic integration scheme. It accounts for both particle motion and
    cell scaling effects through the cell velocity, with optional periodic boundary
    conditions.

    The update follows the form:
    R_new = R + (exp(x) - 1)R + dt*V*exp(x/2)*sinh(x/2)/(x/2)
    where x = V_b * dt is the cell velocity term

    Args:
        state (NPTNoseHooverState): Current simulation state
        velocities (torch.Tensor): Particle velocities [n_particles, n_dimensions]
        cell_velocity (torch.Tensor): Cell velocity with shape [n_systems]
        dt (torch.Tensor): Integration timestep

    Returns:
        torch.Tensor: Updated particle positions with optional periodic wrapping

    Notes:
        - Uses Taylor series for sinh(x)/x near x=0 for numerical stability
        - Properly handles cell scaling through cell_velocity
        - Maintains time-reversibility of the integration scheme
        - Applies periodic boundary conditions if state.pbc is True
        - Supports batched operations with proper atom-to-system mapping
    """
    # Map system-level cell velocities to atom level using system indices
    cell_velocity_atoms = cell_velocity[state.system_idx]  # [n_atoms]

    # Compute cell velocity terms per atom
    x = cell_velocity_atoms * dt  # [n_atoms]
    x_2 = x / 2  # [n_atoms]

    # Compute sinh(x/2)/(x/2) using stable Taylor series
    sinh_term = _npt_nose_hoover_sinhx_x(x_2)  # [n_atoms]

    # Expand dimensions for broadcasting with positions [n_atoms, 3]
    x_expanded = x.unsqueeze(-1)  # [n_atoms, 1]
    x_2_expanded = x_2.unsqueeze(-1)  # [n_atoms, 1]
    sinh_expanded = sinh_term.unsqueeze(-1)  # [n_atoms, 1]

    # Compute position updates
    new_positions = (
        state.positions * (torch.exp(x_expanded) - 1)
        + dt * velocities * torch.exp(x_2_expanded) * sinh_expanded
    )
    return state.positions + new_positions


def _npt_nose_hoover_exp_iL2(  # noqa: N802
    state: NPTNoseHooverState,
    alpha: torch.Tensor,
    momenta: torch.Tensor,
    forces: torch.Tensor,
    cell_velocity: torch.Tensor,
    dt_2: torch.Tensor,
) -> torch.Tensor:
    """Apply the exp(iL2) operator for NPT dynamics momentum updates.

    This function implements the momentum update operator for NPT dynamics using
    a symplectic integration scheme. It accounts for both force terms and
    cell velocity scaling effects.

    The update follows the form:
    P_new = P*exp(-x) + dt/2 * F * exp(-x/2) * sinh(x/2)/(x/2)
    where x = alpha * V_b * dt/2

    Args:
        state (NPTNoseHooverState): Current simulation state for batch mapping
        alpha (torch.Tensor): Cell scaling parameter with shape [n_systems]
        momenta (torch.Tensor): Current particle momenta [n_particles, n_dimensions]
        forces (torch.Tensor): Forces on particles [n_particles, n_dimensions]
        cell_velocity (torch.Tensor): Cell velocity with shape [n_systems]
        dt_2 (torch.Tensor): Half timestep (dt/2)

    Returns:
        torch.Tensor: Updated particle momenta

    Notes:
        - Uses Taylor series for sinh(x)/x near x=0 for numerical stability
        - Properly handles cell velocity scaling effects
        - Maintains time-reversibility of the integration scheme
        - Part of the NPT integration algorithm
        - Supports batched operations with proper atom-to-system mapping
    """
    # Map system-level cell velocities to atom level using system indices
    cell_velocity_atoms = cell_velocity[state.system_idx]  # [n_atoms]

    # Compute scaling terms per atom
    alpha_atoms = alpha[state.system_idx]  # [n_atoms]
    x = alpha_atoms * cell_velocity_atoms * dt_2  # [n_atoms]
    x_2 = x / 2  # [n_atoms]

    # Compute sinh(x/2)/(x/2) using stable Taylor series
    sinh_term = _npt_nose_hoover_sinhx_x(x_2)  # [n_atoms]

    # Expand dimensions for broadcasting with momenta [n_atoms, 3]
    x_expanded = x.unsqueeze(-1)  # [n_atoms, 1]
    x_2_expanded = x_2.unsqueeze(-1)  # [n_atoms, 1]
    sinh_expanded = sinh_term.unsqueeze(-1)  # [n_atoms, 1]

    # Update momenta with both scaling and force terms
    return momenta * torch.exp(-x_expanded) + dt_2 * forces * sinh_expanded * torch.exp(
        -x_2_expanded
    )


def _npt_nose_hoover_compute_cell_force(
    alpha: torch.Tensor,
    volume: torch.Tensor,
    positions: torch.Tensor,
    momenta: torch.Tensor,
    masses: torch.Tensor,
    stress: torch.Tensor,
    external_pressure: torch.Tensor,
    system_idx: torch.Tensor,
) -> torch.Tensor:
    """Compute the force on the cell degree of freedom in NPT dynamics.

    This function calculates the force driving cell volume changes in NPT simulations.
    The force includes contributions from:
    1. Kinetic energy scaling (alpha * KE)
    2. Internal stress (from stress_fn)
    3. External pressure (P*V)

    Args:
        alpha (torch.Tensor): Cell scaling parameter
        volume (torch.Tensor): Current system volume with shape [n_systems]
        positions (torch.Tensor): Particle positions [n_particles, n_dimensions]
        momenta (torch.Tensor): Particle momenta [n_particles, n_dimensions]
        masses (torch.Tensor): Particle masses [n_particles]
        stress (torch.Tensor): Stress tensor [n_systems, n_dimensions, n_dimensions]
        external_pressure (torch.Tensor): Target external pressure
        system_idx (torch.Tensor): System indices for atoms [n_particles]

    Returns:
        torch.Tensor: Force on the cell degree of freedom with shape [n_systems]

    Notes:
        - Force drives volume changes to maintain target pressure
        - Includes both kinetic and potential contributions
        - Uses stress tensor for potential energy contribution
        - Properly handles periodic boundary conditions
        - Supports batched operations
    """
    _N, dim = positions.shape
    n_systems = len(volume)

    # Compute kinetic energy contribution per system
    # Split momenta and masses by system
    KE_per_system = torch.zeros(n_systems, device=positions.device, dtype=positions.dtype)
    for sys_idx in range(n_systems):
        system_mask = system_idx == sys_idx
        if system_mask.any():
            system_momenta = momenta[system_mask]
            system_masses = masses[system_mask]
            KE_per_system[sys_idx] = ts.calc_kinetic_energy(
                masses=system_masses, momenta=system_momenta
            )

    # Get stress tensor and compute trace per system
    # Handle stress tensor with batch dimension
    if stress.ndim == 3:
        internal_pressure = torch.diagonal(stress, dim1=-2, dim2=-1).sum(
            dim=-1
        )  # [n_systems]
    else:
        # Single system case - expand to batch dimension
        internal_pressure = torch.trace(stress).unsqueeze(0).expand(n_systems)

    # Compute force on cell coordinate per system
    # F = alpha * KE - dU/dV - P*V*d
    return (
        (alpha * KE_per_system)
        - (internal_pressure * volume)
        - (external_pressure * volume * dim)
    )


def _npt_nose_hoover_inner_step(
    state: NPTNoseHooverState,
    model: ModelInterface,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTNoseHooverState:
    """Perform one inner step of NPT integration using velocity Verlet algorithm.

    This function implements a single integration step for NPT dynamics, including:
    1. Cell momentum and particle momentum updates (half step)
    2. Position and cell position updates (full step)
    3. Force updates with new positions and cell
    4. Final momentum updates (half step)

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTNoseHooverState): Current system state
        dt (torch.Tensor): Integration timestep
        external_pressure (torch.Tensor): Target external pressure

    Returns:
        NPTNoseHooverState: Updated state after one integration step
    """
    # Get target pressure from kwargs or use default
    dt_2 = dt / 2

    # Unpack state variables for clarity
    positions = state.positions
    momenta = state.momenta
    masses = state.masses
    forces = state.forces
    cell_position = state.cell_position  # [n_systems]
    cell_momentum = state.cell_momentum  # [n_systems]
    cell_mass = state.cell_mass  # [n_systems]

    # Get current volume and cell function
    volume, volume_to_cell = _npt_nose_hoover_cell_info(state)
    cell = volume_to_cell(volume)

    # Get model output
    state.cell = cell
    model_output = model(state)

    # First half step: Update momenta
    n_atoms_per_system = torch.bincount(state.system_idx, minlength=state.n_systems)
    alpha = 1 + 1 / n_atoms_per_system  # [n_systems]

    cell_force_val = _npt_nose_hoover_compute_cell_force(
        alpha=alpha,
        volume=volume,
        positions=positions,
        momenta=momenta,
        masses=masses,
        stress=model_output["stress"],
        external_pressure=external_pressure,
        system_idx=state.system_idx,
    )

    # Update cell momentum and particle momenta
    cell_momentum = cell_momentum + dt_2 * cell_force_val.unsqueeze(-1)
    cell_velocities = cell_momentum.squeeze(-1) / cell_mass
    momenta = _npt_nose_hoover_exp_iL2(
        state, alpha, momenta, forces, cell_velocities, dt_2
    )

    # Full step: Update positions
    cell_position = cell_position + cell_velocities * dt

    # Update state with new cell_position before calling functions that depend on it
    state.cell_position = cell_position

    # Get updated cell
    volume, volume_to_cell = _npt_nose_hoover_cell_info(state)
    cell = volume_to_cell(volume)

    # Update particle positions and forces
    positions = _npt_nose_hoover_exp_iL1(state, state.velocities, cell_velocities, dt)
    state.set_positions(positions)
    state.cell = cell
    model_output = model(state)

    # Second half step: Update momenta
    momenta = _npt_nose_hoover_exp_iL2(
        state, alpha, momenta, model_output["forces"], cell_velocities, dt_2
    )
    cell_force_val = _npt_nose_hoover_compute_cell_force(
        alpha=alpha,
        volume=volume,
        positions=positions,
        momenta=momenta,
        masses=masses,
        stress=model_output["stress"],
        external_pressure=external_pressure,
        system_idx=state.system_idx,
    )
    cell_momentum = cell_momentum + dt_2 * cell_force_val.unsqueeze(-1)

    # Return updated state
    state.set_positions(positions)
    state.set_momenta(momenta)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.cell_position = cell_position
    state.cell_momentum = cell_momentum
    state.cell_mass = cell_mass
    return state


def npt_nose_hoover_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: torch.Tensor,
    dt: torch.Tensor,
    chain_length: int = 3,
    chain_steps: int = 2,
    sy_steps: int = 3,
    t_tau: torch.Tensor | None = None,
    b_tau: torch.Tensor | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> NPTNoseHooverState:
    """Initialize the NPT Nose-Hoover state.

    This function initializes a state for NPT molecular dynamics with Nose-Hoover
    chain thermostats for both temperature and pressure control. It sets up the
    system with appropriate initial conditions including particle positions, momenta,
    cell variables, and thermostat chains.

    Args:
        model (ModelInterface): Model to compute forces and energies
        state: Initial system state as MDState or dict containing positions, masses,
            cell, and PBC information
        kT: Target temperature in energy units
        external_pressure: Target external pressure
        dt: Integration timestep
        chain_length: Length of Nose-Hoover chains. Defaults to 3.
        chain_steps: Chain integration substeps. Defaults to 2.
        sy_steps: Suzuki-Yoshida integration order. Defaults to 3.
        t_tau: Thermostat relaxation time. Controls how quickly temperature
            equilibrates. Defaults to 100*dt
        b_tau: Barostat relaxation time. Controls how quickly pressure equilibrates.
            Defaults to 1000*dt
        seed: Random seed for momenta initialization. Used for reproducible runs
        **kwargs: Additional state variables like atomic_numbers or
            pre-initialized momenta

    Returns:
        NPTNoseHooverState: Initialized state containing:
            - Particle positions, momenta, forces
            - Cell position, momentum and mass (all with batch dimensions)
            - Reference cell matrix (with batch dimensions)
            - Thermostat and barostat chain variables
            - System energy
            - Other state variables (masses, PBC, etc.)

    Notes:
        - Uses separate Nose-Hoover chains for temperature and pressure control
        - Cell mass is set based on system size and barostat relaxation time
        - Initial momenta are drawn from Maxwell-Boltzmann distribution if not
          provided
        - Cell dynamics use logarithmic coordinates for volume updates
        - All cell properties are properly initialized with batch dimensions
    """
    device, dtype = model.device, model.dtype

    # Initialize the NPT Nose-Hoover state
    # Thermostat relaxation time
    if t_tau is None:
        t_tau = 100 * dt

    # Barostat relaxation time
    if b_tau is None:
        b_tau = 1000 * dt

    # Setup thermostats with appropriate timescales
    barostat_fns = construct_nose_hoover_chain(
        dt, chain_length, chain_steps, sy_steps, b_tau
    )
    thermostat_fns = construct_nose_hoover_chain(
        dt, chain_length, chain_steps, sy_steps, t_tau
    )

    if not isinstance(state, SimState):
        state = SimState(**state)

    _n_particles, dim = state.positions.shape
    n_systems = state.n_systems
    atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

    # Initialize cell variables with proper system dimensions
    # cell_momentum: [n_systems, 1] for compatibility with half_step
    cell_position = torch.zeros(n_systems, device=device, dtype=dtype)
    cell_momentum = torch.zeros(n_systems, 1, device=device, dtype=dtype)

    # Convert kT to tensor if it's not already one
    if not isinstance(kT, torch.Tensor):
        kT = torch.tensor(kT, device=device, dtype=dtype)

    # Handle both scalar and batched kT
    kT_system = kT.expand(n_systems) if kT.ndim == 0 else kT

    # Calculate cell masses for each system
    n_atoms_per_system = torch.bincount(state.system_idx, minlength=n_systems)
    cell_mass = dim * (n_atoms_per_system + 1) * kT_system * torch.square(b_tau)
    cell_mass = cell_mass.to(device=device, dtype=dtype)

    # Calculate cell kinetic energy (using first system for initialization)
    dof_barostat = torch.ones(n_systems, device=device, dtype=dtype)
    KE_cell = (cell_momentum.squeeze(-1) ** 2) / (2 * cell_mass)

    # Initialize momenta
    momenta = kwargs.get(
        "momenta",
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    # Compute total DOF for thermostat initialization and a zero KE placeholder
    dof_per_system = torch.bincount(state.system_idx, minlength=n_systems) * dim
    KE_thermostat = ts.calc_kinetic_energy(
        masses=state.masses, momenta=momenta, system_idx=state.system_idx
    )

    # Ensure reference_cell has proper system dimensions
    if state.cell.ndim == 2:
        # Single cell matrix - expand to batch dimension
        reference_cell = state.cell.unsqueeze(0).expand(n_systems, -1, -1).clone()
    else:
        # Already has batch dimension
        reference_cell = state.cell.clone()

    # Handle scalar cell input
    if (torch.is_tensor(state.cell) and state.cell.ndim == 0) or isinstance(
        state.cell, int | float
    ):
        cell_matrix = torch.eye(dim, device=device, dtype=dtype) * state.cell
        reference_cell = cell_matrix.unsqueeze(0).expand(n_systems, -1, -1).clone()
        state.cell = reference_cell

    # Get model output
    model_output = model(state)
    forces = model_output["forces"]
    energy = model_output["energy"]

    if state.constraints:
        # warn if constraints are present
        warnings.warn(
            "Constraints are present in the system. "
            "Make sure they are compatible with NPT Nosé Hoover dynamics."
            "We recommend not using constraints with NPT dynamics for now.",
            UserWarning,
            stacklevel=3,
        )

    # Create initial state
    return NPTNoseHooverState(
        positions=state.positions,
        momenta=momenta,
        energy=energy,
        forces=forces,
        masses=state.masses,
        atomic_numbers=atomic_numbers,
        cell=state.cell,
        pbc=state.pbc,
        system_idx=state.system_idx,
        reference_cell=reference_cell,
        cell_position=cell_position,
        cell_momentum=cell_momentum,
        cell_mass=cell_mass,
        barostat=barostat_fns.initialize(dof_barostat, KE_cell, kT),
        thermostat=thermostat_fns.initialize(dof_per_system, KE_thermostat, kT),
        barostat_fns=barostat_fns,
        thermostat_fns=thermostat_fns,
        _constraints=state.constraints,
    )


def npt_nose_hoover_step(
    state: NPTNoseHooverState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTNoseHooverState:
    """Perform a complete NPT integration step with Nose-Hoover chain thermostats.
    If the center of mass motion is removed initially, it remains removed throughout
    the simulation, so the degrees of freedom decreases by 3.

    This function performs a full NPT integration step including:
    1. Mass parameter updates for thermostats and cell
    2. Thermostat chain updates (half step)
    3. Inner NPT dynamics step
    4. Energy updates for thermostats
    5. Final thermostat chain updates (half step)

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTNoseHooverState): Current system state
        dt (torch.Tensor): Integration timestep
        kT (torch.Tensor): Target temperature
        external_pressure (torch.Tensor): Target external pressure

    Returns:
        NPTNoseHooverState: Updated state after complete integration step
    """
    device, dtype = model.device, model.dtype

    # Unpack state variables for clarity
    barostat = state.barostat
    thermostat = state.thermostat

    # Update mass parameters
    state.barostat = state.barostat_fns.update_mass(barostat, kT)
    state.thermostat = state.thermostat_fns.update_mass(thermostat, kT)
    state = _npt_nose_hoover_update_cell_mass(state, kT, device, dtype)

    # First half step of thermostat chains
    cell_system_idx = torch.arange(state.n_systems, device=device)
    state.cell_momentum, state.barostat = state.barostat_fns.half_step(
        state.cell_momentum, state.barostat, kT, cell_system_idx
    )
    state.momenta, state.thermostat = state.thermostat_fns.half_step(
        state.momenta, state.thermostat, kT, state.system_idx
    )

    # Perform inner NPT step
    state = _npt_nose_hoover_inner_step(state, model, dt, external_pressure)

    # Update kinetic energies for thermostats
    KE = ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )
    state.thermostat.kinetic_energy = KE

    KE_cell = (torch.square(state.cell_momentum.squeeze(-1))) / (2 * state.cell_mass)
    state.barostat.kinetic_energy = KE_cell

    # Second half step of thermostat chains
    state.momenta, state.thermostat = state.thermostat_fns.half_step(
        state.momenta, state.thermostat, kT, state.system_idx
    )
    state.cell_momentum, state.barostat = state.barostat_fns.half_step(
        state.cell_momentum, state.barostat, kT, cell_system_idx
    )
    return state


def _compute_chain_energy(
    chain: NoseHooverChain, kT: torch.Tensor, e_tot: torch.Tensor, dof: torch.Tensor
) -> torch.Tensor:
    """Compute energy contribution from a Nose-Hoover chain.

    Args:
        chain: The Nose-Hoover chain state
        kT: Target temperature
        e_tot: Current total energy for broadcasting
        dof: Degrees of freedom (only used for first chain element)

    Returns:
        Total chain energy contribution
    """
    chain_energy = torch.zeros_like(e_tot)

    # First chain element with DOF weighting
    ke_0 = torch.square(chain.momenta[:, 0]) / (2 * chain.masses[:, 0])
    pe_0 = dof * kT * chain.positions[:, 0]

    chain_energy += ke_0 + pe_0

    # Remaining chain elements
    for i in range(1, chain.positions.shape[1]):
        ke = torch.square(chain.momenta[:, i]) / (2 * chain.masses[:, i])
        pe = kT * chain.positions[:, i]
        chain_energy += ke + pe

    return chain_energy


def npt_nose_hoover_invariant(
    state: NPTNoseHooverState,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
) -> torch.Tensor:
    """Computes the conserved quantity for NPT ensemble with Nose-Hoover thermostat.

    This function calculates the Hamiltonian of the extended NPT dynamics, which should
    be conserved during the simulation. It's useful for validating the correctness of
    NPT simulations.

    The conserved quantity includes:
    - Potential energy of the systems
    - Kinetic energy of the particles
    - Energy contributions from thermostat chains (per system)
    - Energy contributions from barostat chains (per system)
    - PV work term
    - Cell kinetic energy

    Args:
        state: Current state of the NPT simulation system.
            Must contain position, momentum, cell, cell_momentum, cell_mass, thermostat,
            and barostat with proper batching for multiple systems.
        external_pressure: Target external pressure of the system.
        kT: Target thermal energy (Boltzmann constant x temperature).

    Returns:
        torch.Tensor: The conserved quantity (extended Hamiltonian) of the NPT system.
            Returns a scalar for a single system or tensor with shape [n_systems] for
            multiple systems.
    """
    # Calculate volume and potential energy
    volume = torch.det(state.current_cell)  # [n_systems]
    e_pot = state.energy  # Should be scalar or [n_systems]

    # Calculate kinetic energy of particles per system
    e_kin_per_system = ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )

    # Calculate degrees of freedom per system
    n_atoms_per_system = torch.bincount(state.system_idx, minlength=state.n_systems)
    dof_per_system = n_atoms_per_system * state.positions.shape[-1]  # n_atoms * n_dim

    # Initialize total energy with PE + KE
    e_tot = e_pot + e_kin_per_system

    # Add thermostat chain contributions (batched per system, DOF = n_atoms * 3)
    e_tot += _compute_chain_energy(state.thermostat, kT, e_tot, dof_per_system)

    # Add barostat chain contributions (batched per system, DOF = 1)
    barostat_dof = torch.ones_like(dof_per_system)  # 1 DOF per system for barostat
    e_tot += _compute_chain_energy(state.barostat, kT, e_tot, barostat_dof)

    # Add PV term and cell kinetic energy (both are per system)
    e_tot += external_pressure * volume

    # Ensure cell_momentum has the right shape [n_systems]
    cell_momentum = state.cell_momentum.squeeze()

    e_tot += torch.square(cell_momentum) / (2 * state.cell_mass)

    return e_tot


@dataclass(kw_only=True)
class NPTCRescaleState(MDState):
    """State for NPT ensemble with cell rescaling barostat.

    This class extends the MDState to include variables and properties
    specific to the NPT ensemble with a cell rescaling barostat.
    """

    # System state variables
    stress: torch.Tensor
    isothermal_compressibility: torch.Tensor  # shape: [n_systems]
    tau_p: torch.Tensor  # shape: [n_systems]

    _system_attributes = MDState._system_attributes | {  # noqa: SLF001
        "stress",
        "isothermal_compressibility",
        "tau_p",
    }

    def get_number_of_degrees_of_freedom(self) -> torch.Tensor:
        """Calculate degrees of freedom for each system in the batch.

        Returns:
            torch.Tensor: Degrees of freedom for each system, shape [n_systems]
        """
        # Subtract 3 for center of mass motion
        return super().get_number_of_degrees_of_freedom() - 3


def rotate_gram_schmidt(box: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 3x3 box matrices into lower-triangular form.

    Args:
        box (torch.Tensor): shape [n_systems, 3, 3]

    Returns:
        torch.Tensor: shape [n_systems, 3, 3] lower-triangular boxes
    """
    out = torch.zeros_like(box)

    # Columns (a, b, c) correspond to box vectors in column form
    a = box[:, :, 0]
    b = box[:, :, 1]
    c = box[:, :, 2]

    # --- Compute the lower-triangular entries ---

    # a-axis
    out[:, 0, 0] = torch.norm(a, dim=1)

    # b projections
    out[:, 1, 0] = torch.sum(a * b, dim=1) / out[:, 0, 0]
    out[:, 1, 1] = torch.sqrt(torch.sum(b * b, dim=1) - out[:, 1, 0] ** 2)

    # c projections
    out[:, 2, 0] = torch.sum(a * c, dim=1) / out[:, 0, 0]
    out[:, 2, 1] = (torch.sum(b * c, dim=1) - out[:, 2, 0] * out[:, 1, 0]) / out[:, 1, 1]
    out[:, 2, 2] = torch.sqrt(
        torch.sum(c * c, dim=1) - out[:, 2, 0] ** 2 - out[:, 2, 1] ** 2
    )

    # Upper-triangular entries are 0 by initialization
    return out


def batch_matrix_vector(
    matrices: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Perform batch matrix-vector multiplication.

    Args:
        matrices (torch.Tensor): shape [n_systems, n, n]
        vectors (torch.Tensor): shape [n_systems, n, m]

    Returns:
        torch.Tensor: shape [n_systems, n, m] result of multiplication
    """
    return torch.matmul(matrices, vectors.unsqueeze(-1)).squeeze(-1)


def _crescale_anisotropic_barostat_step(
    state: NPTCRescaleState,
    kT: torch.Tensor,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTCRescaleState:
    volume = torch.det(state.cell)  # shape: (n_systems,)
    P_int = ts.quantities.compute_instantaneous_pressure_tensor(
        momenta=state.momenta,
        masses=state.masses,
        system_idx=state.system_idx,
        stress=state.stress,
        volumes=volume,
    )
    sqrt_vol = torch.sqrt(volume)
    trace_P_int = torch.einsum("bii->b", P_int)
    prefactor_random = torch.sqrt(
        kT * state.isothermal_compressibility * dt / (4 * state.tau_p)
    )
    prefactor = state.isothermal_compressibility * sqrt_vol / (2 * state.tau_p)
    change_sqrt_vol = -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * torch.randn_like(sqrt_vol)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol
    ## Step 2: compute deformation matrix
    prefactor_random_matrix = (
        torch.sqrt(2 * state.isothermal_compressibility * kT * dt / (3 * state.tau_p))
        / new_sqrt_volume
    )
    a_tilde = -(state.isothermal_compressibility / (3 * state.tau_p))[:, None, None] * (
        P_int
        - trace_P_int[:, None, None]
        / 3
        * torch.eye(
            3, device=state.positions.device, dtype=state.positions.dtype
        ).expand_as(P_int)
    )
    random_matrix = torch.randn(
        state.n_systems,
        3,
        3,
        device=state.positions.device,
        dtype=state.positions.dtype,
    )
    random_matrix_tilde = random_matrix - torch.einsum("bii->b", random_matrix)[
        :, None, None
    ] / 3 * torch.eye(
        3, device=state.positions.device, dtype=state.positions.dtype
    ).expand_as(random_matrix)
    deformation_matrix = torch.matrix_exp(
        a_tilde * dt + prefactor_random_matrix[:, None, None] * random_matrix_tilde
    )
    deformation_matrix = rotate_gram_schmidt(deformation_matrix)

    ## Step 3: propagate sqrt(volume) for dt/2
    new_sqrt_volume += -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * torch.randn_like(sqrt_vol)
    rscaling = deformation_matrix * torch.pow((new_sqrt_volume / sqrt_vol), 2 / 3).view(
        -1, 1, 1
    )
    vscaling = torch.inverse(rscaling).transpose(-2, -1)

    # Update positions and momenta (barostat + half momentum step)
    state.positions = batch_matrix_vector(
        rscaling[state.system_idx], state.positions
    ) + batch_matrix_vector(
        (vscaling + rscaling)[state.system_idx], state.momenta
    ) * dt / (2 * state.masses.unsqueeze(-1))
    state.momenta = batch_matrix_vector(vscaling[state.system_idx], state.momenta)
    state.cell = rscaling.mT @ state.cell
    return state


def _crescale_independent_lengths_barostat_step(
    state: NPTCRescaleState,
    kT: torch.Tensor,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTCRescaleState:
    volume = torch.det(state.cell)  # shape: (n_systems,)
    P_int = ts.quantities.compute_instantaneous_pressure_tensor(
        momenta=state.momenta,
        masses=state.masses,
        system_idx=state.system_idx,
        stress=state.stress,
        volumes=volume,
    )
    sqrt_vol = torch.sqrt(volume)
    trace_P_int = torch.einsum("bii->b", P_int)
    prefactor_random = torch.sqrt(
        kT * state.isothermal_compressibility * dt / (4 * state.tau_p)
    )
    prefactor = state.isothermal_compressibility * sqrt_vol / (2 * state.tau_p)
    change_sqrt_vol = -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * torch.randn_like(sqrt_vol)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol
    ## Step 2: compute deformation matrix
    prefactor_random_matrix = (
        torch.sqrt(2 * state.isothermal_compressibility * kT * dt / (3 * state.tau_p))
        / new_sqrt_volume
    )
    # Note: it corresponds to using a diagonal isothermal compressibility tensor
    P_int_diagonal = torch.diagonal(P_int, dim1=-2, dim2=-1)
    a_tilde = -(state.isothermal_compressibility / (3 * state.tau_p))[:, None] * (
        P_int_diagonal - trace_P_int[:, None] / 3
    )

    random_matrix = torch.randn(
        state.n_systems,
        3,
        device=state.positions.device,
        dtype=state.positions.dtype,
    )
    random_matrix_tilde = random_matrix - torch.mean(random_matrix, dim=1, keepdim=True)
    deformation_matrix = torch.exp(
        a_tilde * dt + prefactor_random_matrix[:, None] * random_matrix_tilde
    )

    ## Step 3: propagate sqrt(volume) for dt/2
    new_sqrt_volume += -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * torch.randn_like(sqrt_vol)
    rscaling = deformation_matrix * torch.pow(
        (new_sqrt_volume / sqrt_vol), 2 / 3
    ).unsqueeze(-1)

    # Update positions and momenta (barostat + half momentum step)
    state.positions = rscaling[state.system_idx] * state.positions + (
        rscaling + 1 / rscaling
    )[state.system_idx] * state.momenta * dt / (2 * state.masses.unsqueeze(-1))
    state.momenta = (1 / rscaling)[state.system_idx] * state.momenta
    state.cell = torch.diag_embed(rscaling) @ state.cell
    return state


def compute_average_pressure_tensor(
    *,
    degrees_of_freedom: torch.Tensor,
    kT: torch.Tensor,
    stress: torch.Tensor,
    volumes: torch.Tensor,
) -> torch.Tensor:
    """Compute forces on the cell for NPT dynamics.

    This function calculates the instantaneous internal pressure tensor.

    Args:
        degrees_of_freedom (torch.Tensor): Degrees of freedom of
            the system, shape (n_systems,)
        kT (torch.Tensor): Thermal energy (k_B * T), shape (n_systems,)
        stress (torch.Tensor): Stress tensor of the system, shape (n_systems, 3, 3)
        volumes (torch.Tensor): Volumes of the systems, shape (n_systems,)

    Returns:
        torch.Tensor: Instanteneous internal pressure tesnor [n_systems, 3, 3]
    """
    # Calculate virials: 2/V * (N_{atoms}k_B T / 2 - Virial_{tensor})
    n_systems = stress.shape[0]
    prefactor = degrees_of_freedom * kT / volumes  # shape: (n_systems,)
    average_kinetic_energy_tensor = prefactor[:, None, None] * torch.eye(
        3, device=stress.device, dtype=stress.dtype
    ).expand(n_systems, 3, 3)
    return average_kinetic_energy_tensor - stress


def _crescale_average_anisotropic_barostat_step(
    state: NPTCRescaleState,
    kT: torch.Tensor,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTCRescaleState:
    volume = torch.det(state.cell)  # shape: (n_systems,)
    P_int = compute_average_pressure_tensor(
        # Should it be degrees_of_freedom=state.get_number_of_degrees_of_freedom() / 3,
        degrees_of_freedom=state.n_atoms_per_system,
        kT=kT,
        stress=state.stress,
        volumes=volume,
    )
    sqrt_vol = torch.sqrt(volume)
    trace_P_int = torch.einsum("bii->b", P_int)
    prefactor_random = torch.sqrt(
        kT * state.isothermal_compressibility * dt / (4 * state.tau_p)
    )
    prefactor = state.isothermal_compressibility * sqrt_vol / (2 * state.tau_p)
    change_sqrt_vol = -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * torch.randn_like(sqrt_vol)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol
    ## Step 2: compute deformation matrix
    prefactor_random_matrix = (
        torch.sqrt(2 * state.isothermal_compressibility * kT * dt / (3 * state.tau_p))
        / new_sqrt_volume
    )
    a_tilde = -(state.isothermal_compressibility / (3 * state.tau_p))[:, None, None] * (
        P_int
        - trace_P_int[:, None, None]
        / 3
        * torch.eye(
            3, device=state.positions.device, dtype=state.positions.dtype
        ).expand_as(P_int)
    )
    random_matrix = torch.randn(
        state.n_systems,
        3,
        3,
        device=state.positions.device,
        dtype=state.positions.dtype,
    )
    random_matrix_tilde = random_matrix - torch.einsum("bii->b", random_matrix)[
        :, None, None
    ] / 3 * torch.eye(
        3, device=state.positions.device, dtype=state.positions.dtype
    ).expand_as(random_matrix)
    deformation_matrix = torch.matrix_exp(
        a_tilde * dt + prefactor_random_matrix[:, None, None] * random_matrix_tilde
    )
    deformation_matrix = rotate_gram_schmidt(deformation_matrix)

    ## Step 3: propagate sqrt(volume) for dt/2
    new_sqrt_volume += -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * torch.randn_like(sqrt_vol)
    rscaling = deformation_matrix * torch.pow((new_sqrt_volume / sqrt_vol), 2 / 3).view(
        -1, 1, 1
    )

    # Update positions and momenta (barostat + half momentum step)
    state.positions = batch_matrix_vector(
        rscaling[state.system_idx], state.positions
    ) + batch_matrix_vector(
        (
            torch.eye(
                3, device=state.positions.device, dtype=state.positions.dtype
            ).expand_as(rscaling)
            + rscaling
        )[state.system_idx],
        state.momenta,
    ) * dt / (2 * state.masses.unsqueeze(-1))
    state.cell = rscaling.mT @ state.cell
    return state


def _crescale_isotropic_barostat_step(
    state: NPTCRescaleState,
    kT: torch.Tensor,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTCRescaleState:
    volume = torch.det(state.cell)  # shape: (n_systems,)
    P_int = ts.quantities.compute_instantaneous_pressure_tensor(
        momenta=state.momenta,
        masses=state.masses,
        system_idx=state.system_idx,
        stress=state.stress,
        volumes=volume,
    )
    sqrt_vol = torch.sqrt(volume)
    trace_P_int = torch.einsum("bii->b", P_int)
    prefactor_random = torch.sqrt(
        kT * state.isothermal_compressibility * dt / (4 * state.tau_p)
    )
    prefactor = state.isothermal_compressibility * sqrt_vol / (2 * state.tau_p)
    change_sqrt_vol = -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt + prefactor_random * torch.randn_like(sqrt_vol)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol

    # Update positions and momenta (barostat + half momentum step)
    # SI (S13ab): notice there is a typo in the SI where q_i(t)
    # should be scaled as well by rscaling
    rscaling = torch.pow((new_sqrt_volume / sqrt_vol), 2 / 3).unsqueeze(-1)
    state.positions = rscaling[state.system_idx] * state.positions + (
        rscaling + 1 / rscaling
    )[state.system_idx] * state.momenta * (0.5 * dt) / state.masses.unsqueeze(-1)
    state.momenta = (1 / rscaling)[state.system_idx] * state.momenta
    rscaling = rscaling.unsqueeze(-1)  # make [n_systems, 1, 1]
    state.cell = rscaling * state.cell
    return state


def npt_crescale_anisotropic_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> NPTCRescaleState:
    """Perform one NPT integration step with cell rescaling barostat.

    This function performs a single integration step for NPT dynamics using
    a cell rescaling barostat. It updates particle positions, momenta, and
    the simulation cell based on the target temperature and pressure.

    Trotter based splitting:
    1. Half Thermostat (velocity scaling)
    2. Half Update momenta with forces
    3. Barostat (cell rescaling)
    4. Update positions (from barostat + half momenta)
    5. Update forces with new positions and cell
    6. Compute forces
    7. Half Update momenta with forces
    8. Half Thermostat (velocity scaling)

    Only allow isotropic external stress. This method performs anisotropic
    cell rescaling. Lengths and angles can change independently. Based on
    pressure using kinetic energy. Positions and momenta are scaled when scaling the cell.

    Inspired from: https://github.com/bussilab/crescale/blob/master/simplemd_anisotropic/simplemd.cpp
    - Time reversible integrator
    - Instantaneous kinetic energy (not not the average from equipartition)

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTCRescaleState): Current system state
        dt (torch.Tensor): Integration timestep
        kT (torch.Tensor): Target temperature
        external_pressure (torch.Tensor): Target external pressure
        tau (torch.Tensor | None): V-Rescale thermostat relaxation time. If None,
            defaults to 100*dt

    Returns:
        NPTCRescaleState: Updated state after one integration step
    """
    # Note: would probably be better to have tau in NVTCRescaleState
    if tau is None:
        tau = 100 * dt
    state = _vrescale_update(state, tau, kT, dt / 2)

    state = momentum_step(state, dt / 2)

    # Barostat step
    state = _crescale_anisotropic_barostat_step(state, kT, dt, external_pressure)

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]

    # Final momentum step
    state = momentum_step(state, dt / 2)

    # Final thermostat step
    return _vrescale_update(state, tau, kT, dt / 2)


def npt_crescale_independent_lengths_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> NPTCRescaleState:
    """Perform one NPT integration step with cell rescaling barostat.

    This function performs a single integration step for NPT dynamics using
    a cell rescaling barostat. It updates particle positions, momenta, and
    the simulation cell based on the target temperature and pressure.

    Trotter based splitting:
    1. Half Thermostat (velocity scaling)
    2. Half Update momenta with forces
    3. Barostat (cell rescaling)
    4. Update positions (from barostat + half momenta)
    5. Update forces with new positions and cell
    6. Compute forces
    7. Half Update momenta with forces
    8. Half Thermostat (velocity scaling)

    Only allow isotropic external stress.
    This method has 3 degrees of freedom for each cell length,
    allowing independent scaling of each cell vector.

    Inspired from: https://github.com/bussilab/crescale/blob/master/simplemd_anisotropic/simplemd.cpp
    - Time reversible integrator
    - Instantaneous kinetic energy (not not the average from equipartition)

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTCRescaleState): Current system state
        dt (torch.Tensor): Integration timestep
        kT (torch.Tensor): Target temperature
        external_pressure (torch.Tensor): Target external pressure
        tau (torch.Tensor | None): V-Rescale thermostat relaxation time. If None,
            defaults to 100*dt

    Returns:
        NPTCRescaleState: Updated state after one integration step
    """
    # Note: would probably be better to have tau in NVTCRescaleState
    if tau is None:
        tau = 100 * dt
    state = _vrescale_update(state, tau, kT, dt / 2)

    state = momentum_step(state, dt / 2)

    # Barostat step
    state = _crescale_independent_lengths_barostat_step(state, kT, dt, external_pressure)

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]

    # Final momentum step
    state = momentum_step(state, dt / 2)

    # Final thermostat step
    return _vrescale_update(state, tau, kT, dt / 2)


def npt_crescale_average_anisotropic_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> NPTCRescaleState:
    """Perform one NPT integration step with cell rescaling barostat.

    This function performs a single integration step for NPT dynamics using
    a cell rescaling barostat. It updates particle positions, momenta, and
    the simulation cell based on the target temperature and pressure.

    Trotter based splitting:
    1. Half Thermostat (velocity scaling)
    2. Half Update momenta with forces
    3. Barostat (cell rescaling)
    4. Update positions (from barostat + half momenta)
    5. Update forces with new positions and cell
    6. Compute forces
    7. Half Update momenta with forces
    8. Half Thermostat (velocity scaling)

    Only allow isotropic external stress. This method performs anisotropic
    cell rescaling. Lengths and angles can change independently. Based on
    pressure using average kinetic energy from equipartition theorem.
    Only positions are scaled when scaling the cell.

    Inspired from: https://github.com/bussilab/crescale/blob/master/simplemd_anisotropic/simplemd.cpp
    - Time reversible integrator
    - Average kinetic energy, scaling only positions

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTCRescaleState): Current system state
        dt (torch.Tensor): Integration timestep
        kT (torch.Tensor): Target temperature
        external_pressure (torch.Tensor): Target external pressure
        tau (torch.Tensor | None): V-Rescale thermostat relaxation time. If None,
            defaults to 100*dt

    Returns:
        NPTCRescaleState: Updated state after one integration step
    """
    # Note: would probably be better to have tau in NVTCRescaleState
    if tau is None:
        tau = 100 * dt
    state = _vrescale_update(state, tau, kT, dt / 2)

    state = momentum_step(state, dt / 2)

    # Barostat step
    state = _crescale_average_anisotropic_barostat_step(state, kT, dt, external_pressure)

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]

    # Final momentum step
    state = momentum_step(state, dt / 2)

    # Final thermostat step
    return _vrescale_update(state, tau, kT, dt / 2)


def npt_crescale_isotropic_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    tau: torch.Tensor | None = None,
) -> NPTCRescaleState:
    """Perform one NPT integration step with cell rescaling barostat.

    This function performs a single integration step for NPT dynamics using
    a cell rescaling barostat. It updates particle positions, momenta, and
    the simulation cell based on the target temperature and pressure.

    Trotter based splitting:
    1. Half Thermostat (velocity scaling)
    2. Half Update momenta with forces
    3. Barostat (cell rescaling)
    4. Update positions (from barostat + half momenta)
    5. Update forces with new positions and cell
    6. Compute forces
    7. Half Update momenta with forces
    8. Half Thermostat (velocity scaling)

    Only allow isotropic external stress. This performs isotropic
    cell rescaling: cell shape is preserved, cell lengths are scaled equally.
    For anisotropic cell rescaling, use npt_crescale_anisotropic_step.

    References:
        - Bernetti, Mattia, and Giovanni Bussi.
        "Pressure control using stochastic cell rescaling."
        The Journal of Chemical Physics 153.11 (2020).
        - And the corresponding Supplementary Information which details
        the integration scheme. Notice an error in scaling of positions in SI Eq. S13a.

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTCRescaleState): Current system state
        dt (torch.Tensor): Integration timestep
        kT (torch.Tensor): Target temperature
        external_pressure (torch.Tensor): Target external pressure
        tau (torch.Tensor | None): V-Rescale thermostat relaxation time. If None,
            defaults to 100*dt

    Returns:
        NPTCRescaleState: Updated state after one integration step
    """
    # Note: would probably be better to have tau in NVTCRescaleState
    if tau is None:
        tau = 100 * dt
    state = _vrescale_update(state, tau, kT, dt / 2)

    state = momentum_step(state, dt / 2)

    # Barostat step
    state = _crescale_isotropic_barostat_step(state, kT, dt, external_pressure)

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]

    # Final momentum step
    state = momentum_step(state, dt / 2)

    # Final thermostat step
    return _vrescale_update(state, tau, kT, dt / 2)


def npt_crescale_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: torch.Tensor,
    dt: torch.Tensor,
    tau_p: torch.Tensor | None = None,
    isothermal_compressibility: torch.Tensor | None = None,
    seed: int | None = None,
) -> NPTCRescaleState:
    """Initialize the NPT cell rescaling state.

    This function initializes a state for NPT molecular dynamics with a
    cell rescaling barostat. It sets up the system with appropriate initial
    conditions including particle positions, momenta, and cell variables.

    Only allow isotropic external stress, but can run both isotropic and
    anisotropic cell rescaling.

    Args:
        state: Initial system state as MDState or dict containing positions, masses,
            cell, and PBC information
        model (ModelInterface): Model to compute forces and energies
        kT: Target temperature in energy units
        dt: Integration timestep
        tau_p: Barostat relaxation time. Controls how quickly pressure equilibrates.
        isothermal_compressibility: Isothermal compressibility of the system.
        seed: Random seed for momenta initialization.
    """
    device, dtype = model.device, model.dtype

    # Set default values if not provided
    if tau_p is None:
        tau_p = 5000 * dt  # 5ps for dt=1fs
    if isothermal_compressibility is None:
        isothermal_compressibility = 1e-1  # (eV/A^3)^-1

    # Convert all parameters to tensors with correct device and dtype
    tau_p = torch.as_tensor(tau_p, device=device, dtype=dtype)
    isothermal_compressibility = torch.as_tensor(
        isothermal_compressibility, device=device, dtype=dtype
    )
    if tau_p.ndim == 0:
        tau_p = tau_p.expand(state.n_systems)
    if isothermal_compressibility.ndim == 0:
        isothermal_compressibility = isothermal_compressibility.expand(state.n_systems)
    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    if isinstance(kT, float):
        kT = torch.tensor(kT, device=device, dtype=dtype)

    if not isinstance(state, SimState):
        state = SimState(**state)

    # Get model output to initialize forces and stress
    model_output = model(state)

    # Initialize momenta if not provided
    momenta = getattr(
        state,
        "momenta",
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    # Create the initial state
    return NPTCRescaleState(
        positions=state.positions,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        stress=model_output["stress"],
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        system_idx=state.system_idx,
        atomic_numbers=state.atomic_numbers,
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
    )
