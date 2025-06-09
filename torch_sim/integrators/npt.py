"""Implementations of NPT integrators."""

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
)
from torch_sim.quantities import calc_kinetic_energy
from torch_sim.state import SimState
from torch_sim.typing import StateDict


@dataclass
class NPTLangevinState(SimState):
    """State information for an NPT system with Langevin dynamics.

    This class represents the complete state of a molecular system being integrated
    in the NPT (constant particle number, pressure, temperature) ensemble using
    Langevin dynamics. In addition to particle positions and momenta, it tracks
    cell dimensions and their dynamics for volume fluctuations.

    Attributes:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        velocities (torch.Tensor): Particle velocities [n_particles, n_dim]
        energy (torch.Tensor): Energy of the system [n_batches]
        forces (torch.Tensor): Forces on particles [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        cell (torch.Tensor): Simulation cell matrix [n_batches, n_dim, n_dim]
        pbc (bool): Whether to use periodic boundary conditions
        batch (torch.Tensor): Batch indices [n_particles]
        atomic_numbers (torch.Tensor): Atomic numbers [n_particles]
        stress (torch.Tensor): Stress tensor [n_batches, n_dim, n_dim]
        reference_cell (torch.Tensor): Original cell vectors used as reference for
            scaling [n_batches, n_dim, n_dim]
        cell_positions (torch.Tensor): Cell positions [n_batches, n_dim, n_dim]
        cell_velocities (torch.Tensor): Cell velocities [n_batches, n_dim, n_dim]
        cell_masses (torch.Tensor): Masses associated with the cell degrees of freedom
            shape [n_batches]

    Properties:
        momenta (torch.Tensor): Particle momenta calculated as velocities*masses
            with shape [n_particles, n_dimensions]
        n_batches (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    # System state variables
    energy: torch.Tensor
    forces: torch.Tensor
    velocities: torch.Tensor
    stress: torch.Tensor

    # Cell variables
    reference_cell: torch.Tensor
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_masses: torch.Tensor

    @property
    def momenta(self) -> torch.Tensor:
        """Calculate momenta from velocities and masses."""
        return self.velocities * self.masses.unsqueeze(-1)


# Extracted out from npt_langevin body to test fix in https://github.com/Radical-AI/torch-sim/pull/153
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
            tensor with shape [n_batches, n_dimensions, n_dimensions]
        kT (torch.Tensor): Temperature in energy units, either scalar or
            shape [n_batches]

    Returns:
        torch.Tensor: Force acting on the cell [n_batches, n_dim, n_dim]
    """
    # Get current volumes for each batch
    volumes = torch.linalg.det(state.cell)  # shape: (n_batches,)

    # Reshape for broadcasting
    volumes = volumes.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

    # Create pressure tensor (diagonal with external pressure)
    if external_pressure.ndim == 0:
        # Scalar pressure - create diagonal pressure tensors for each batch
        pressure_tensor = external_pressure * torch.eye(
            3, device=state.device, dtype=state.dtype
        )
        pressure_tensor = pressure_tensor.unsqueeze(0).expand(state.n_batches, -1, -1)
    else:
        # Already a tensor with shape compatible with n_batches
        pressure_tensor = external_pressure

    # Calculate virials from stress and external pressure
    # Internal stress is negative of virial tensor divided by volume
    virial = -volumes * (state.stress + pressure_tensor)

    # Add kinetic contribution (kT * Identity)
    batch_kT = kT
    if kT.ndim == 0:
        batch_kT = kT.expand(state.n_batches)

    e_kin_per_atom = batch_kT.view(-1, 1, 1) * torch.eye(
        3, device=state.device, dtype=state.dtype
    ).unsqueeze(0)

    # Correct implementation with scaling by n_atoms_per_batch
    return virial + e_kin_per_atom * state.n_atoms_per_batch.view(-1, 1, 1)


def npt_langevin(  # noqa: C901, PLR0915
    model: torch.nn.Module,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    alpha: torch.Tensor | None = None,
    cell_alpha: torch.Tensor | None = None,
    b_tau: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], NPTLangevinState],
    Callable[[NPTLangevinState, torch.Tensor], NPTLangevinState],
]:
    """Initialize and return an NPT (isothermal-isobaric) integrator with Langevin
    dynamics.

    This function sets up integration in the NPT ensemble, where particle number (N),
    pressure (P), and temperature (T) are conserved. It allows the simulation cell to
    fluctuate to maintain the target pressure, while using Langevin dynamics to
    maintain constant temperature.

    Args:
        model (torch.nn.Module): Neural network model that computes energies, forces,
            and stress. Must return a dict with 'energy', 'forces', and 'stress' keys.
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_batches]
        external_pressure (torch.Tensor): Target pressure to maintain, either scalar
            or shape [n_batches, n_dim, n_dim] for anisotropic pressure
        alpha (torch.Tensor, optional): Friction coefficient for particle Langevin
            thermostat, either scalar or shape [n_batches]. Defaults to 1/(100*dt).
        cell_alpha (torch.Tensor, optional): Friction coefficient for cell Langevin
            thermostat, either scalar or shape [n_batches]. Defaults to same as alpha.
        b_tau (torch.Tensor, optional): Barostat time constant controlling how quickly
            the system responds to pressure differences, either scalar or shape
            [n_batches]. Defaults to 1/(1000*dt).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - callable: Function to initialize the NPTLangevinState from input data
              with signature: init_fn(state, kT=kT, seed=seed) -> NPTLangevinState
            - callable: Update function that evolves system by one timestep
              with signature: update_fn(state, dt=dt, kT=kT,
              external_pressure=external_pressure, alpha=alpha,
              cell_alpha=cell_alpha) -> NPTLangevinState

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
    if isinstance(alpha, float):
        alpha = torch.tensor(alpha, device=device, dtype=dtype)
    if isinstance(cell_alpha, float):
        cell_alpha = torch.tensor(cell_alpha, device=device, dtype=dtype)
    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    if isinstance(kT, float):
        kT = torch.tensor(kT, device=device, dtype=dtype)
    if isinstance(b_tau, float):
        b_tau = torch.tensor(b_tau, device=device, dtype=dtype)
    if isinstance(external_pressure, float):
        external_pressure = torch.tensor(external_pressure, device=device, dtype=dtype)

    def beta(
        state: NPTLangevinState,
        alpha: torch.Tensor,
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
                shape [n_batches]
            kT (torch.Tensor): Temperature in energy units, either scalar or
                shape [n_batches]
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]

        Returns:
            torch.Tensor: Random noise term for force calculation [n_particles, n_dim]
        """
        # Generate batch-specific noise with correct shape
        noise = torch.randn_like(state.velocities)

        # Calculate the thermal noise amplitude by batch
        batch_kT = kT
        if kT.ndim == 0:
            batch_kT = kT.expand(state.n_batches)

        # Map batch kT to atoms
        atom_kT = batch_kT[state.batch]

        # Calculate the prefactor for each atom
        # The standard deviation should be sqrt(2*alpha*kB*T*dt)
        prefactor = torch.sqrt(2 * alpha * atom_kT * dt)

        return prefactor.unsqueeze(-1) * noise

    def cell_beta(
        state: NPTLangevinState,
        cell_alpha: torch.Tensor,
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
                with shape [n_batches]
            kT (torch.Tensor): System temperature in energy units, either scalar or
                with shape [n_batches]
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]

        Returns:
            torch.Tensor: Scaled random noise for cell dynamics with shape
                [n_batches, n_dimensions, n_dimensions]
        """
        # Generate standard normal distribution (zero mean, unit variance)
        noise = torch.randn_like(state.cell_positions, device=device, dtype=dtype)

        # Ensure cell_alpha and kT have batch dimension if they're scalars
        if cell_alpha.ndim == 0:
            cell_alpha = cell_alpha.expand(state.n_batches)
        if kT.ndim == 0:
            kT = kT.expand(state.n_batches)

        # Reshape for broadcasting
        cell_alpha = cell_alpha.view(-1, 1, 1)  # shape: (n_batches, 1, 1)
        kT = kT.view(-1, 1, 1)  # shape: (n_batches, 1, 1)
        if dt.ndim == 0:
            dt = dt.expand(state.n_batches).view(-1, 1, 1)
        else:
            dt = dt.view(-1, 1, 1)

        # Scale to satisfy the fluctuation-dissipation theorem
        # The standard deviation should be sqrt(2*alpha*kB*T*dt)
        scaling_factor = torch.sqrt(2.0 * cell_alpha * kT * dt)

        return scaling_factor * noise

    def compute_cell_force(
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
                tensor with shape [n_batches, n_dimensions, n_dimensions]
            kT (torch.Tensor): Temperature in energy units, either scalar or
                shape [n_batches]

        Returns:
            torch.Tensor: Force acting on the cell [n_batches, n_dim, n_dim]
        """
        return _compute_cell_force(state, external_pressure, kT)

    def cell_position_step(
        state: NPTLangevinState,
        dt: torch.Tensor,
        pressure_force: torch.Tensor,
        kT: torch.Tensor = kT,
        cell_alpha: torch.Tensor = cell_alpha,
    ) -> NPTLangevinState:
        """Update the cell position in NPT dynamics.

        This function updates the cell position (effectively the volume) in NPT dynamics
        using the current cell velocities, pressure forces, and thermal noise. It
        implements the position update part of the Langevin barostat algorithm.

        Args:
            state (NPTLangevinState): Current NPT state
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            pressure_force (torch.Tensor): Pressure force for barostat
                [n_batches, n_dim, n_dim]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_batches]
            cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
                with shape [n_batches]

        Returns:
            NPTLangevinState: Updated state with new cell positions
        """
        # Calculate effective mass term
        Q_2 = 2 * state.cell_masses.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

        # Ensure parameters have batch dimension
        if dt.ndim == 0:
            dt = dt.expand(state.n_batches)
        if cell_alpha.ndim == 0:
            cell_alpha = cell_alpha.expand(state.n_batches)

        # Reshape for broadcasting
        dt_expanded = dt.view(-1, 1, 1)
        cell_alpha_expanded = cell_alpha.view(-1, 1, 1)

        # Calculate damping factor for cell position update
        cell_b = 1 / (1 + ((cell_alpha_expanded * dt_expanded) / Q_2))

        # Deterministic velocity contribution
        c_1 = cell_b * dt_expanded * state.cell_velocities

        # Force contribution
        c_2 = cell_b * dt_expanded * dt_expanded * pressure_force / Q_2

        # Random noise contribution (thermal fluctuations)
        c_3 = (
            cell_b
            * dt_expanded
            * cell_beta(state=state, cell_alpha=cell_alpha, kT=kT, dt=dt)
            / Q_2
        )

        # Update cell positions with all contributions
        state.cell_positions = state.cell_positions + c_1 + c_2 + c_3
        return state

    def cell_velocity_step(
        state: NPTLangevinState,
        F_p_n: torch.Tensor,
        dt: torch.Tensor,
        pressure_force: torch.Tensor,
        cell_alpha: torch.Tensor,
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
                [n_batches, n_dimensions, n_dimensions]
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            pressure_force (torch.Tensor): Final pressure force
                shape [n_batches, n_dim, n_dim]
            cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
                shape [n_batches]
            kT (torch.Tensor): Temperature in energy units, either scalar or
                shape [n_batches]

        Returns:
            NPTLangevinState: Updated state with new cell velocities
        """
        # Ensure parameters have batch dimension
        if dt.ndim == 0:
            dt = dt.expand(state.n_batches)
        if cell_alpha.ndim == 0:
            cell_alpha = cell_alpha.expand(state.n_batches)
        if kT.ndim == 0:
            kT = kT.expand(state.n_batches)

        # Reshape for broadcasting - need to maintain 3x3 dimensions
        dt_expanded = dt.view(-1, 1, 1)  # shape: (n_batches, 1, 1)
        cell_alpha_expanded = cell_alpha.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

        # Calculate cell masses per batch - reshape to match 3x3 cell matrices
        cell_masses_expanded = state.cell_masses.view(
            -1, 1, 1
        )  # shape: (n_batches, 1, 1)

        # These factors come from the Langevin integration scheme
        a = (1 - (cell_alpha_expanded * dt_expanded) / cell_masses_expanded) / (
            1 + (cell_alpha_expanded * dt_expanded) / cell_masses_expanded
        )
        b = 1 / (1 + (cell_alpha_expanded * dt_expanded) / cell_masses_expanded)

        # Calculate the three terms for velocity update
        # a will broadcast from (n_batches, 1, 1) to (n_batches, 3, 3)
        c_1 = a * state.cell_velocities  # Damped old velocity

        # Force contribution (average of initial and final forces)
        c_2 = dt_expanded * ((a * F_p_n) + pressure_force) / (2 * cell_masses_expanded)

        # Generate batch-specific cell noise with correct shape (n_batches, 3, 3)
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

    def langevin_position_step(
        state: NPTLangevinState,
        L_n: torch.Tensor,  # This should be shape (n_batches,)
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
            L_n (torch.Tensor): Previous cell length scale with shape [n_batches]
            dt: Integration timestep, either scalar or with shape [n_batches]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_batches]

        Returns:
            NPTLangevinState: Updated state with new positions
        """
        # Calculate effective mass term by batch
        # Map masses to have batch dimension
        M_2 = 2 * state.masses.unsqueeze(-1)  # shape: (n_atoms, 1)

        # Calculate new cell length scale (cube root of volume for isotropic scaling)
        L_n_new = torch.pow(
            state.cell_positions.reshape(state.n_batches, -1)[:, 0], 1 / 3
        )  # shape: (n_batches,)

        # Map batch-specific L_n and L_n_new to atom-level using batch indices
        # Make sure L_n is the right shape (n_batches,) before indexing
        if L_n.ndim != 1 or L_n.shape[0] != state.n_batches:
            # If L_n has wrong shape, calculate it again to ensure correct shape
            L_n = torch.pow(
                state.cell_positions.reshape(state.n_batches, -1)[:, 0], 1 / 3
            )

        # Map batch values to atoms using batch indices
        L_n_atoms = L_n[state.batch]  # shape: (n_atoms,)
        L_n_new_atoms = L_n_new[state.batch]  # shape: (n_atoms,)

        # Calculate damping factor
        alpha_atoms = alpha
        if alpha.ndim > 0:
            alpha_atoms = alpha[state.batch]
        dt_atoms = dt
        if dt.ndim > 0:
            dt_atoms = dt[state.batch]

        b = 1 / (1 + ((alpha_atoms * dt_atoms) / M_2))

        # Scale positions due to cell volume change
        c_1 = (L_n_new_atoms / L_n_atoms).unsqueeze(-1) * state.positions

        # Time step factor with average length scale
        c_2 = (
            (2 * L_n_new_atoms / (L_n_new_atoms + L_n_atoms)).unsqueeze(-1)
            * b
            * dt_atoms.unsqueeze(-1)
        )

        # Generate atom-specific noise
        noise = torch.randn_like(state.velocities)
        batch_kT = kT
        if kT.ndim == 0:
            batch_kT = kT.expand(state.n_batches)
        atom_kT = batch_kT[state.batch]

        # Calculate noise prefactor according to fluctuation-dissipation theorem
        noise_prefactor = torch.sqrt(2 * alpha_atoms * atom_kT * dt_atoms)
        noise_term = noise_prefactor.unsqueeze(-1) * noise

        # Velocity and force contributions with random noise
        c_3 = (
            state.velocities
            + dt_atoms.unsqueeze(-1) * state.forces / M_2
            + noise_term / M_2
        )

        # Update positions with all contributions
        state.positions = c_1 + c_2 * c_3

        # Apply periodic boundary conditions if needed
        if state.pbc:
            state.positions = ts.transforms.pbc_wrap_batched(
                state.positions, state.cell, state.batch
            )

        return state

    def langevin_velocity_step(
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
            dt: Integration timestep, either scalar or with shape [n_batches]
            kT: Target temperature in energy units, either scalar or
                with shape [n_batches]

        Returns:
            NPTLangevinState: Updated state with new velocities
        """
        # Calculate denominator for update equations
        M_2 = 2 * state.masses.unsqueeze(-1)  # shape: (n_atoms, 1)

        # Map batch parameters to atom level
        alpha_atoms = alpha
        if alpha.ndim > 0:
            alpha_atoms = alpha[state.batch]
        dt_atoms = dt
        if dt.ndim > 0:
            dt_atoms = dt[state.batch]

        # Calculate damping factors for Langevin integration
        a = (1 - (alpha_atoms * dt_atoms) / M_2) / (1 + (alpha_atoms * dt_atoms) / M_2)
        b = 1 / (1 + (alpha_atoms * dt_atoms) / M_2)

        # Velocity contribution with damping
        c_1 = a * state.velocities

        # Force contribution (average of initial and final forces)
        c_2 = dt_atoms.unsqueeze(-1) * ((a * forces) + state.forces) / M_2

        # Generate atom-specific noise
        noise = torch.randn_like(state.velocities)
        batch_kT = kT
        if kT.ndim == 0:
            batch_kT = kT.expand(state.n_batches)
        atom_kT = batch_kT[state.batch]

        # Calculate noise prefactor according to fluctuation-dissipation theorem
        noise_prefactor = torch.sqrt(2 * alpha_atoms * atom_kT * dt_atoms)
        noise_term = noise_prefactor.unsqueeze(-1) * noise

        # Random noise contribution
        c_3 = b * noise_term / state.masses.unsqueeze(-1)

        # Update velocities with all contributions
        state.velocities = c_1 + c_2 + c_3
        return state

    def npt_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        seed: int | None = seed,
    ) -> NPTLangevinState:
        """Initialize an NPT Langevin state from input data.

        This function creates the initial state for NPT Langevin dynamics,
        setting up all necessary variables including particle velocities,
        cell parameters, and barostat variables. It computes initial forces
        and stress using the provided model.

        Args:
            state (SimState | StateDict): Either a SimState object or a dictionary
                containing positions, masses, cell, pbc
            kT (torch.Tensor): Temperature in energy units for initializing momenta
            seed (int, optional): Random seed for reproducibility

        Returns:
            NPTLangevinState: Initialized state for NPT Langevin integration containing
                all required attributes for particle and cell dynamics
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Get model output to initialize forces and stress
        model_output = model(state)

        # Initialize momenta if not provided
        momenta = getattr(
            state,
            "momenta",
            calculate_momenta(state.positions, state.masses, state.batch, kT, seed),
        )

        # Initialize cell parameters
        reference_cell = state.cell.clone()

        # Calculate initial cell_positions (volume)
        cell_positions = (
            torch.linalg.det(state.cell).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (n_batches, 1, 1)

        # Initialize cell velocities to zero
        cell_velocities = torch.zeros((state.n_batches, 3, 3), device=device, dtype=dtype)

        # Calculate cell masses based on system size and temperature
        # This follows standard NPT barostat mass scaling
        n_atoms_per_batch = torch.bincount(state.batch)
        batch_kT = (
            kT.expand(state.n_batches)
            if isinstance(kT, torch.Tensor) and kT.ndim == 0
            else kT
        )
        cell_masses = (n_atoms_per_batch + 1) * batch_kT * b_tau * b_tau

        # Create the initial state
        return NPTLangevinState(
            positions=state.positions,
            velocities=momenta / state.masses.unsqueeze(-1),
            energy=model_output["energy"],
            forces=model_output["forces"],
            stress=model_output["stress"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            batch=state.batch,
            atomic_numbers=state.atomic_numbers,
            reference_cell=reference_cell,
            cell_positions=cell_positions,
            cell_velocities=cell_velocities,
            cell_masses=cell_masses,
        )

    def npt_update(
        state: NPTLangevinState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        external_pressure: torch.Tensor = external_pressure,
        alpha: torch.Tensor = alpha,
        cell_alpha: torch.Tensor = cell_alpha,
    ) -> NPTLangevinState:
        """Perform one complete NPT Langevin dynamics integration step.

        This function implements a modified integration scheme for NPT dynamics,
        handling both atomic and cell updates with Langevin thermostats to maintain
        constant temperature and pressure. The integration scheme couples particle
        motion with cell volume fluctuations.

        Args:
            state (NPTLangevinState): Current NPT state with particle and cell variables
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                shape [n_batches]
            external_pressure (torch.Tensor): Target external pressure, either scalar or
                tensor with shape [n_batches, n_dim, n_dim]
            alpha (torch.Tensor): Position friction coefficient, either scalar or
                shape [n_batches]
            cell_alpha (torch.Tensor): Cell friction coefficient, either scalar or
                shape [n_batches]

        Returns:
            NPTLangevinState: Updated NPT state after one timestep with new positions,
                velocities, cell parameters, forces, energy, and stress
        """
        # Convert any scalar parameters to tensors with batch dimension if needed
        if isinstance(alpha, float):
            alpha = torch.tensor(alpha, device=device, dtype=dtype)
        if isinstance(kT, float):
            kT = torch.tensor(kT, device=device, dtype=dtype)
        if isinstance(cell_alpha, float):
            cell_alpha = torch.tensor(cell_alpha, device=device, dtype=dtype)
        if isinstance(dt, float):
            dt = torch.tensor(dt, device=device, dtype=dtype)

        # Make sure parameters have batch dimension if they're scalars
        batch_kT = kT.expand(state.n_batches) if kT.ndim == 0 else kT

        # Update barostat mass based on current temperature
        # This ensures proper coupling between system and barostat
        n_atoms_per_batch = torch.bincount(state.batch)
        state.cell_masses = (n_atoms_per_batch + 1) * batch_kT * b_tau * b_tau

        # Compute model output for current state
        model_output = model(state)
        state.forces = model_output["forces"]
        state.stress = model_output["stress"]

        # Store initial values for integration
        forces = state.forces
        F_p_n = compute_cell_force(
            state=state, external_pressure=external_pressure, kT=kT
        )
        L_n = torch.pow(
            state.cell_positions.reshape(state.n_batches, -1)[:, 0], 1 / 3
        )  # shape: (n_batches,)

        # Step 1: Update cell position
        state = cell_position_step(state=state, dt=dt, pressure_force=F_p_n, kT=kT)

        # Update cell (currently only isotropic fluctuations)
        dim = state.positions.shape[1]  # Usually 3 for 3D
        # V_0 and V are shape: (n_batches,)
        V_0 = torch.linalg.det(state.reference_cell)
        V = state.cell_positions.reshape(state.n_batches, -1)[:, 0]

        # Scale cell uniformly in all dimensions
        scaling = (V / V_0) ** (1.0 / dim)  # shape: (n_batches,)

        # Apply scaling to reference cell to get new cell
        new_cell = torch.zeros_like(state.cell)
        for b in range(state.n_batches):
            new_cell[b] = scaling[b] * state.reference_cell[b]

        state.cell = new_cell

        # Step 2: Update particle positions
        state = langevin_position_step(state=state, L_n=L_n, dt=dt, kT=kT)

        # Recompute model output after position updates
        model_output = model(state)
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]
        state.stress = model_output["stress"]

        # Compute updated pressure force
        F_p_n_new = compute_cell_force(
            state=state, external_pressure=external_pressure, kT=kT
        )

        # Step 3: Update cell velocities
        state = cell_velocity_step(
            state=state,
            F_p_n=F_p_n,
            dt=dt,
            pressure_force=F_p_n_new,
            cell_alpha=cell_alpha,
            kT=kT,
        )

        # Step 4: Update particle velocities
        state = langevin_velocity_step(state=state, forces=forces, dt=dt, kT=kT)

        return state  # noqa: RET504

    return npt_init, npt_update


@dataclass
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
            [n_dimensions, n_dimensions]. Used to measure relative volume changes.
        cell_position (torch.Tensor): Logarithmic cell coordinate.
            Scalar value representing (1/d)ln(V/V_0) where V is current volume
            and V_0 is reference volume.
        cell_momentum (torch.Tensor): Cell momentum (velocity) conjugate to cell_position.
            Scalar value controlling volume changes.
        cell_mass (torch.Tensor): Mass parameter for cell dynamics. Controls coupling
            between volume fluctuations and pressure.
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
            cell_position. Shape: [n_dimensions, n_dimensions]

    Notes:
        - The cell parameterization ensures volume positivity
        - Nose-Hoover chains provide deterministic control of T and P
        - Extended system approach conserves an extended Hamiltonian
        - Time-reversible when integrated with appropriate algorithms
    """

    # Cell variables
    reference_cell: torch.Tensor
    cell_position: torch.Tensor
    cell_momentum: torch.Tensor
    cell_mass: torch.Tensor

    # Thermostat variables
    thermostat: NoseHooverChain
    thermostat_fns: NoseHooverChainFns

    # Barostat variables
    barostat: NoseHooverChain
    barostat_fns: NoseHooverChainFns

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
                [n_dimensions, n_dimensions]
        """
        dim = self.positions.shape[1]
        V_0 = torch.det(self.reference_cell)  # Reference volume
        V = V_0 * torch.exp(dim * self.cell_position)  # Current volume
        scale = (V / V_0) ** (1.0 / dim)
        return scale * self.reference_cell


def npt_nose_hoover(  # noqa: C901, PLR0915
    *,
    model: torch.nn.Module,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    dt: torch.Tensor,
    chain_length: int = 3,
    chain_steps: int = 2,
    sy_steps: int = 3,
) -> tuple[
    Callable[[SimState | StateDict], NPTNoseHooverState],
    Callable[[NPTNoseHooverState, torch.Tensor], NPTNoseHooverState],
]:
    """Create an NPT simulation with Nose-Hoover chain thermostats.

    This function returns initialization and update functions for NPT molecular dynamics
    with Nose-Hoover chain thermostats for temperature and pressure control.

    Args:
        model (torch.nn.Module): Model to compute forces and energies
        kT (torch.Tensor): Target temperature in energy units
        external_pressure (torch.Tensor): Target external pressure
        dt (torch.Tensor): Integration timestep
        chain_length (int, optional): Length of Nose-Hoover chains. Defaults to 3.
        chain_steps (int, optional): Chain integration substeps. Defaults to 2.
        sy_steps (int, optional): Suzuki-Yoshida integration order. Defaults to 3.

    Returns:
        tuple:
            - Callable[[SimState | StateDict], NPTNoseHooverState]: Initialization
              function
            - Callable[[NPTNoseHooverState, torch.Tensor], NPTNoseHooverState]: Update
              function

    Notes:
        - Uses Nose-Hoover chains for both temperature and pressure control
        - Implements symplectic integration with Suzuki-Yoshida decomposition
        - Cell dynamics use logarithmic coordinates for volume updates
        - Conserves extended system Hamiltonian
    """
    device, dtype = model.device, model.dtype

    def _npt_cell_info(
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
                - torch.Tensor: Current system volume
                - callable: Function that takes a volume and returns the corresponding
                    cell matrix

        Notes:
            - Uses logarithmic cell coordinate parameterization
            - Volume changes are measured relative to reference cell
            - Cell scaling preserves shape while changing volume
        """
        dim = state.positions.shape[1]
        ref = state.reference_cell
        V_0 = torch.det(ref)  # Reference volume
        V = V_0 * torch.exp(dim * state.cell_position)  # Current volume

        def volume_to_cell(V: torch.Tensor) -> torch.Tensor:
            """Compute cell matrix for a given volume."""
            return (V / V_0) ** (1.0 / dim) * ref

        return V, volume_to_cell

    def update_cell_mass(
        state: NPTNoseHooverState, kT: torch.Tensor
    ) -> NPTNoseHooverState:
        """Update the cell mass parameter in an NPT simulation.

        This function updates the mass parameter associated with cell volume fluctuations
        based on the current system size and target temperature. The cell mass controls
        how quickly the volume can change and is chosen to maintain stable pressure
        control.

        Args:
            state (NPTNoseHooverState): Current state of the NPT system
            kT (torch.Tensor): Target temperature in energy units

        Returns:
            NPTNoseHooverState: Updated state with new cell mass

        Notes:
            - Cell mass scales with system size (N+1) and dimensionality
            - Larger cell mass gives slower but more stable volume fluctuations
            - Mass depends on barostat relaxation time (tau)
        """
        n_particles, dim = state.positions.shape
        cell_mass = torch.tensor(
            dim * (n_particles + 1) * kT * state.barostat.tau**2,
            device=device,
            dtype=dtype,
        )
        # Create new state with updated cell mass
        state.cell_mass = cell_mass
        return state

    def sinhx_x(x: torch.Tensor) -> torch.Tensor:
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
            1 + x**2 / 6 + x**4 / 120 + x**6 / 5040 + x**8 / 362_880 + x**10 / 39_916_800
        )

    def exp_iL1(  # noqa: N802
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
            cell_velocity (torch.Tensor): Cell velocity (scalar)
            dt (torch.Tensor): Integration timestep

        Returns:
            torch.Tensor: Updated particle positions with optional periodic wrapping

        Notes:
            - Uses Taylor series for sinh(x)/x near x=0 for numerical stability
            - Properly handles cell scaling through cell_velocity
            - Maintains time-reversibility of the integration scheme
            - Applies periodic boundary conditions if state.pbc is True
        """
        # Compute cell velocity terms
        x = cell_velocity * dt
        x_2 = x / 2

        # Compute sinh(x/2)/(x/2) using stable Taylor series
        sinh_term = sinhx_x(x_2)

        # Compute position updates
        new_positions = (
            state.positions * (torch.exp(x) - 1)
            + dt * velocities * torch.exp(x_2) * sinh_term
        )
        new_positions = state.positions + new_positions

        # Apply periodic boundary conditions
        return ts.transforms.pbc_wrap_general(new_positions, state.current_cell.T)

    def exp_iL2(  # noqa: N802
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
            alpha (torch.Tensor): Cell scaling parameter
            momenta (torch.Tensor): Current particle momenta [n_particles, n_dimensions]
            forces (torch.Tensor): Forces on particles [n_particles, n_dimensions]
            cell_velocity (torch.Tensor): Cell velocity (scalar)
            dt_2 (torch.Tensor): Half timestep (dt/2)

        Returns:
            torch.Tensor: Updated particle momenta

        Notes:
            - Uses Taylor series for sinh(x)/x near x=0 for numerical stability
            - Properly handles cell velocity scaling effects
            - Maintains time-reversibility of the integration scheme
            - Part of the NPT integration algorithm
        """
        # Compute scaling terms
        x = alpha * cell_velocity * dt_2
        x_2 = x / 2

        # Compute sinh(x/2)/(x/2) using stable Taylor series
        sinh_term = sinhx_x(x_2)

        # Update momenta with both scaling and force terms
        return momenta * torch.exp(-x) + dt_2 * forces * sinh_term * torch.exp(-x_2)

    def compute_cell_force(
        alpha: torch.Tensor,
        volume: torch.Tensor,
        positions: torch.Tensor,
        momenta: torch.Tensor,
        masses: torch.Tensor,
        stress: torch.Tensor,
        external_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the force on the cell degree of freedom in NPT dynamics.

        This function calculates the force driving cell volume changes in NPT simulations.
        The force includes contributions from:
        1. Kinetic energy scaling (alpha * KE)
        2. Internal stress (from stress_fn)
        3. External pressure (P*V)

        Args:
            alpha (torch.Tensor): Cell scaling parameter
            volume (torch.Tensor): Current system volume
            positions (torch.Tensor): Particle positions [n_particles, n_dimensions]
            momenta (torch.Tensor): Particle momenta [n_particles, n_dimensions]
            masses (torch.Tensor): Particle masses [n_particles]
            stress (torch.Tensor): Stress tensor [n_dimensions, n_dimensions]
            external_pressure (torch.Tensor): Target external pressure


        Returns:
            torch.Tensor: Force on the cell degree of freedom

        Notes:
            - Force drives volume changes to maintain target pressure
            - Includes both kinetic and potential contributions
            - Uses stress tensor for potential energy contribution
            - Properly handles periodic boundary conditions
        """
        N, dim = positions.shape

        # Compute kinetic energy contribution
        KE2 = 2.0 * calc_kinetic_energy(momenta, masses)

        # Get stress tensor and compute trace
        internal_pressure = torch.trace(stress)

        # Compute force on cell coordinate
        # F = alpha * KE - dU/dV - P*V*d
        return (
            (alpha * KE2)
            - (internal_pressure * volume)
            - (external_pressure * volume * dim)
        )

    def npt_inner_step(
        state: NPTNoseHooverState,
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
        cell_position = state.cell_position
        cell_momentum = state.cell_momentum
        cell_mass = state.cell_mass

        n_particles, dim = positions.shape

        # Get current volume and cell function
        volume, volume_to_cell = _npt_cell_info(state)
        cell = volume_to_cell(volume)

        # Get model output
        state.cell = cell
        model_output = model(state)

        # First half step: Update momenta
        alpha = 1 + 1 / n_particles
        cell_force_val = compute_cell_force(
            alpha=alpha,
            volume=volume,
            positions=positions,
            momenta=momenta,
            masses=masses,
            stress=model_output["stress"],
            external_pressure=external_pressure,
        )

        # Update cell momentum and particle momenta
        cell_momentum = cell_momentum + dt_2 * cell_force_val
        momenta = exp_iL2(alpha, momenta, forces, cell_momentum / cell_mass, dt_2)

        # Full step: Update positions
        cell_position = cell_position + cell_momentum / cell_mass * dt

        # Update state with new cell_position before calling functions that depend on it
        state.cell_position = cell_position

        # Get updated cell
        volume, volume_to_cell = _npt_cell_info(state)
        cell = volume_to_cell(volume)

        # Update particle positions and forces
        positions = exp_iL1(state, state.velocities, cell_momentum / cell_mass, dt)
        state.positions = positions
        state.cell = cell
        model_output = model(state)

        # Second half step: Update momenta
        momenta = exp_iL2(
            alpha, momenta, model_output["forces"], cell_momentum / cell_mass, dt_2
        )
        cell_force_val = compute_cell_force(
            alpha=alpha,
            volume=volume,
            positions=positions,
            momenta=momenta,
            masses=masses,
            stress=model_output["stress"],
            external_pressure=external_pressure,
        )
        cell_momentum = cell_momentum + dt_2 * cell_force_val

        # Return updated state
        state.positions = positions
        state.momenta = momenta
        state.forces = model_output["forces"]
        state.energy = model_output["energy"]
        state.cell_position = cell_position
        state.cell_momentum = cell_momentum
        state.cell_mass = cell_mass
        return state

    def npt_nose_hoover_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
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
            state: Initial system state as SimState or dict containing positions, masses,
                cell, and PBC information
            kT: Target temperature in energy units
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
                - Cell position, momentum and mass
                - Reference cell matrix
                - Thermostat and barostat chain variables
                - System energy
                - Other state variables (masses, PBC, etc.)

        Notes:
            - Uses separate Nose-Hoover chains for temperature and pressure control
            - Cell mass is set based on system size and barostat relaxation time
            - Initial momenta are drawn from Maxwell-Boltzmann distribution if not
              provided
            - Cell dynamics use logarithmic coordinates for volume updates
        """
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

        # Check if there is an extra batch dimension
        if state.cell.dim() == 3:
            state.cell = state.cell.squeeze(0)

        n_particles, dim = state.positions.shape
        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Initialize cell variables
        cell_position = torch.zeros((), device=device, dtype=dtype)
        cell_momentum = torch.zeros((), device=device, dtype=dtype)
        cell_mass = torch.tensor(
            dim * (n_particles + 1) * kT * b_tau**2, device=device, dtype=dtype
        )

        # Calculate cell kinetic energy
        KE_cell = calc_kinetic_energy(cell_momentum, cell_mass)

        # Handle scalar cell input
        if (torch.is_tensor(state.cell) and state.cell.ndim == 0) or isinstance(
            state.cell, int | float
        ):
            state.cell = torch.eye(dim, device=device, dtype=dtype) * state.cell

        # Get model output
        model_output = model(state)
        forces = model_output["forces"]
        energy = model_output["energy"]

        # Create initial state
        state = NPTNoseHooverState(
            positions=state.positions,
            momenta=None,
            energy=energy,
            forces=forces,
            masses=state.masses,
            atomic_numbers=atomic_numbers,
            cell=state.cell,
            pbc=state.pbc,
            reference_cell=state.cell,
            cell_position=cell_position,
            cell_momentum=cell_momentum,
            cell_mass=cell_mass,
            barostat=barostat_fns.initialize(1, KE_cell, kT),
            thermostat=None,
            barostat_fns=barostat_fns,
            thermostat_fns=thermostat_fns,
        )

        # Initialize momenta
        momenta = kwargs.get(
            "momenta",
            calculate_momenta(state.positions, state.masses, state.batch, kT, seed),
        )

        # Initialize thermostat
        state.momenta = momenta
        KE = calc_kinetic_energy(state.momenta, state.masses)
        state.thermostat = thermostat_fns.initialize(state.positions.numel(), KE, kT)

        return state

    def npt_nose_hoover_update(
        state: NPTNoseHooverState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        external_pressure: torch.Tensor = external_pressure,
    ) -> NPTNoseHooverState:
        """Perform a complete NPT integration step with Nose-Hoover chain thermostats.

        This function performs a full NPT integration step including:
        1. Mass parameter updates for thermostats and cell
        2. Thermostat chain updates (half step)
        3. Inner NPT dynamics step
        4. Energy updates for thermostats
        5. Final thermostat chain updates (half step)

        Args:
            state (NPTNoseHooverState): Current system state
            dt (torch.Tensor): Integration timestep
            kT (torch.Tensor): Target temperature
            external_pressure (torch.Tensor): Target external pressure

        Returns:
            NPTNoseHooverState: Updated state after complete integration step
        """
        # Unpack state variables for clarity
        barostat = state.barostat
        thermostat = state.thermostat

        # Update mass parameters
        state.barostat = state.barostat_fns.update_mass(barostat, kT)
        state.thermostat = state.thermostat_fns.update_mass(thermostat, kT)
        state = update_cell_mass(state, kT)

        # First half step of thermostat chains
        state.cell_momentum, state.barostat = state.barostat_fns.half_step(
            state.cell_momentum, state.barostat, kT
        )
        state.momenta, state.thermostat = state.thermostat_fns.half_step(
            state.momenta, state.thermostat, kT
        )

        # Perform inner NPT step
        state = npt_inner_step(
            state=state,
            dt=dt,
            external_pressure=external_pressure,
        )

        # Update kinetic energies for thermostats
        KE = calc_kinetic_energy(state.momenta, state.masses)
        state.thermostat.kinetic_energy = KE

        KE_cell = calc_kinetic_energy(state.cell_momentum, state.cell_mass)
        state.barostat.kinetic_energy = KE_cell

        # Second half step of thermostat chains
        state.momenta, state.thermostat = state.thermostat_fns.half_step(
            state.momenta, state.thermostat, kT
        )
        state.cell_momentum, state.barostat = state.barostat_fns.half_step(
            state.cell_momentum, state.barostat, kT
        )
        return state

    return npt_nose_hoover_init, npt_nose_hoover_update


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
    - Potential energy of the system
    - Kinetic energy of the particles
    - Energy contributions from thermostat chains
    - Energy contributions from barostat chains
    - PV work term
    - Cell kinetic energy

    Args:
        state: Current state of the NPT simulation system.
            Must contain position, momentum, cell, cell_momentum, cell_mass, thermostat,
            and barostat.
        external_pressure: Target external pressure of the system.
        kT: Target thermal energy (Boltzmann constant x temperature).

    Returns:
        torch.Tensor: The conserved quantity (extended Hamiltonian) of the NPT system.
    """
    # Calculate volume and potential energy
    volume = torch.det(state.current_cell)
    e_pot = state.energy

    # Calculate kinetic energy of particles
    e_kin = calc_kinetic_energy(state.momenta, state.masses)

    # Total degrees of freedom
    DOF = state.positions.numel()

    # Initialize total energy with PE + KE
    e_tot = e_pot + e_kin

    # Add thermostat chain contributions
    e_tot += (state.thermostat.momenta[0] ** 2) / (2 * state.thermostat.masses[0])
    e_tot += DOF * kT * state.thermostat.positions[0]

    # Add remaining thermostat terms
    for pos, momentum, mass in zip(
        state.thermostat.positions[1:],
        state.thermostat.momenta[1:],
        state.thermostat.masses[1:],
        strict=True,
    ):
        e_tot += (momentum**2) / (2 * mass) + kT * pos

    # Add barostat chain contributions
    for pos, momentum, mass in zip(
        state.barostat.positions,
        state.barostat.momenta,
        state.barostat.masses,
        strict=True,
    ):
        e_tot += (momentum**2) / (2 * mass) + kT * pos

    # Add PV term and cell kinetic energy
    e_tot += external_pressure * volume
    e_tot += (state.cell_momentum**2) / (2 * state.cell_mass)

    return e_tot
