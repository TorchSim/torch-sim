"""Integrators for atomistic dynamics simulations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from torch_sim.quantities import calc_kinetic_energy, count_dof
from torch_sim.state import SimState, StateDict
from torch_sim.transforms import pbc_wrap_general


@dataclass
class MDState(SimState):
    """State information for MD.

    This class represents the complete state of a molecular system being integrated
    with Langevin dynamics in the NVT (constant particle number, volume, temperature)
    ensemble. The Langevin thermostat adds stochastic noise and friction to maintain
    constant temperature.

    Attributes:
        positions: Particle positions with shape [n_particles, n_dimensions]
        momenta: Particle momenta with shape [n_particles, n_dimensions]
        energy: Energy of the system
        forces: Forces on particles with shape [n_particles, n_dimensions]
        masses: Particle masses with shape [n_particles]
        cell: Simulation cell matrix with shape [n_dimensions, n_dimensions]
        pbc: Whether to use periodic boundary conditions

    Properties:
        velocities: Particle velocities computed as momenta/masses
            Has shape [n_particles, n_dimensions]
    """

    momenta: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor

    @property
    def velocities(self) -> torch.Tensor:
        """Calculate velocities from momenta and masses.

        Returns:
            The velocities of the particles
        """
        return self.momenta / self.masses.unsqueeze(-1)


def calculate_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    kT: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    """Calculate momenta from positions and masses.

    Args:
        positions: The positions of the particles
        masses: The masses of the particles
        kT: The temperature of the system
        device: The device to use for the calculation
        dtype: The data type to use for the calculation
        seed: The seed to use for the calculation

    Returns:
        The momenta of the particles
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    # Center the momentum if more than one particle
    if positions.shape[0] > 1:
        momenta = momenta - torch.mean(momenta, dim=0, keepdim=True)

    return momenta


def initialize_momenta(
    state: MDState,
    kT: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> MDState:
    """Initialize particle momenta from Maxwell-Boltzmann distribution.

    Args:
        state: Current system state containing particle masses and positions
        kT: Temperature in energy units to initialize momenta at
        device: Device to initialize momenta on
        dtype: Data type to initialize momenta as
        seed: Random seed for reproducibility (optional)

    Returns:
        Updated state with initialized momenta
    """
    state.momenta = calculate_momenta(
        state.positions, state.masses, kT, device, dtype, seed
    )
    return state


def momentum_step(state: MDState, dt: torch.Tensor) -> MDState:
    """Update particle momenta using current forces.

    This function performs the momentum update step of velocity Verlet integration
    by applying forces over the timestep dt.

    Args:
        state: Current system state containing forces and momenta
        dt: Integration timestep

    Returns:
        Updated state with new momenta after force application

    Notes:
        - Implements p(t+dt) = p(t) + F(t)*dt
        - Used as half-steps in velocity Verlet algorithm
        - Preserves time-reversibility when used symmetrically
    """
    new_momenta = state.momenta + state.forces * dt
    state.momenta = new_momenta
    return state


def position_step(state: MDState, dt: torch.Tensor) -> MDState:
    """Update particle positions using current velocities.

    This function performs the position update step of velocity Verlet integration
    by propagating particles according to their velocities over timestep dt.

    Args:
        state: Current system state containing positions and velocities
        dt: Integration timestep

    Returns:
        Updated state with new positions after propagation

    Notes:
        - Implements r(t+dt) = r(t) + v(t)*dt
        - Handles periodic boundary conditions if enabled
        - Used as half-steps in velocity Verlet algorithm
    """
    new_positions = state.positions + state.velocities * dt

    if state.pbc:
        new_positions = pbc_wrap_general(
            positions=new_positions, lattice_vectors=state.cell.T
        )

    state.positions = new_positions
    return state


def velocity_verlet(state: MDState, dt: torch.Tensor, model: torch.nn.Module) -> MDState:
    """Perform one complete velocity Verlet integration step.

    This function implements the velocity Verlet algorithm, which provides
    time-reversible integration of the equations of motion. The integration
    sequence is:
    1. Half momentum update
    2. Full position update
    3. Force update
    4. Half momentum update

    Args:
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep
        model: Neural network model that computes energies and forces

    Returns:
        Updated state after one complete velocity Verlet step

    Notes:
        - Time-reversible and symplectic integrator
        - Conserves energy in the absence of numerical errors
        - Handles periodic boundary conditions if enabled in state
    """
    dt_2 = dt / 2
    state = momentum_step(state, dt_2)
    state = position_step(state, dt)

    model_output = model(state)

    state.energy = model_output["energy"]
    state.forces = model_output["forces"]
    return momentum_step(state, dt_2)


def nve(
    *,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
) -> tuple[
    Callable[[SimState | dict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NVE (microcanonical) integrator.

    This function sets up integration in the NVE ensemble, where particle number (N),
    volume (V), and total energy (E) are conserved. It returns both an initial state
    and an update function for time evolution.

    Args:
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Temperature in energy units for initializing momenta

    Returns:
        tuple:
            - Callable[[SimState | StateDict, torch.Tensor], MDState]: Function to
              initialize the MDState from input data and kT
            - Callable[[MDState, torch.Tensor], MDState]: Update function that evolves
              system by one timestep

    Notes:
        - Uses velocity Verlet algorithm for time-reversible integration
        - Conserves total energy in the absence of numerical errors
        - Initial velocities sampled from Maxwell-Boltzmann distribution
        - Model must return dict with 'energy' and 'forces' keys
    """
    device = model.device
    dtype = model.dtype

    def nve_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MDState:
        """Initialize an NVE state from input data.

        Args:
            state: Either a SimState object or a dictionary containing positions,
                masses, cell, pbc
            kT: Temperature in energy units for initializing momenta
            seed: Random seed for reproducibility
            **kwargs: Additional state arguments

        Returns:
            MDState: Initialized state for NVE integration
        """
        # Extract required data from input
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Check if there is an extra batch dimension
        if state.cell.dim() == 3:
            state.cell = state.cell.squeeze(0)

        # Override with kwargs if provided
        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        model_output = model(state)

        momenta = kwargs.get(
            "momenta",
            calculate_momenta(state.positions, state.masses, kT, device, dtype, seed),
        )

        initial_state = MDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
        )
        return initial_state  # noqa: RET504

    def nve_update(state: MDState, dt: torch.Tensor = dt, **_) -> MDState:
        """Perform one complete NVE (microcanonical) integration step.

        This function implements the velocity Verlet algorithm for NVE dynamics,
        which provides energy-conserving time evolution. The integration sequence is:
        1. Half momentum update
        2. Full position update
        3. Force update
        4. Half momentum update

        Args:
            state: Current system state containing positions, momenta, forces
            dt: Integration timestep

        Returns:
            Updated state after one complete NVE step

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


def nvt_langevin(
    *,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
    gamma: torch.Tensor | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NVT (canonical) integrator using Langevin dynamics.

    This function sets up integration in the NVT ensemble, where particle number (N),
    volume (V), and temperature (T) are conserved. It returns both an initial state
    and an update function for time evolution.

    Args:
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Target temperature in energy units
        gamma: Friction coefficient for Langevin thermostat (default: 1/(100*dt))

    Returns:
        tuple:
            - Callable[[SimState | StateDict, torch.Tensor], MDState]: Function to
              initialize the MDState from input data and kT
            - Callable[[MDState, torch.Tensor], MDState]: Update function that evolves
              system by one timestep

    Notes:
        - Uses BAOAB splitting scheme for Langevin dynamics
        - Preserves detailed balance for correct NVT sampling
        - Handles periodic boundary conditions if enabled in state
    """
    device = model.device
    dtype = model.dtype
    gamma = gamma or 1 / (100 * dt)

    if isinstance(gamma, float):
        gamma = torch.tensor(gamma, device=device, dtype=dtype)

    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    def ou_step(
        state: MDState,
        dt: torch.Tensor,
        kT: torch.Tensor,
        gamma: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MDState:
        """Apply stochastic noise and friction for Langevin dynamics.

        This function implements the stochastic part of Langevin dynamics by applying
        random noise and friction forces to particle momenta. The noise amplitude is
        chosen to maintain the target temperature kT.

        Args:
            state: Current system state containing positions, momenta, etc.
            dt: Integration timestep
            kT: Target temperature in energy units
            gamma: Friction coefficient controlling noise strength
            device: Device to initialize momenta on
            dtype: Data type to initialize momenta as

        Returns:
            Updated state with new momenta after stochastic step

        Notes:
            - Uses Ornstein-Uhlenbeck process for correct thermal sampling
            - Noise amplitude scales with sqrt(mass) for equipartition
            - Preserves detailed balance through fluctuation-dissipation relation
        """
        c1 = torch.exp(-gamma * dt)
        c2 = torch.sqrt(kT * (1 - c1**2))

        # Generate random noise from normal distribution
        noise = torch.randn_like(state.momenta, device=device, dtype=dtype)
        new_momenta = (
            c1 * state.momenta + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
        )
        state.momenta = new_momenta
        return state

    def langevin_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MDState:
        """Initialize an NVT Langevin state from input data.

        Args:
            state: Either a SimState object or a dictionary containing positions,
                masses, cell, pbc
            kT: Temperature in energy units for initializing momenta
            seed: Random seed for reproducibility
            **kwargs: Additional state arguments

        Returns:
            MDState: Initialized state for NVT Langevin integration
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Check if there is an extra batch dimension
        if state.cell.dim() == 3:
            state.cell = state.cell.squeeze(0)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        model_output = model(state)

        momenta = kwargs.get(
            "momenta",
            calculate_momenta(state.positions, state.masses, kT, device, dtype, seed),
        )

        initial_state = MDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
        )
        return initial_state  # noqa: RET504

    def langevin_update(
        state: MDState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        gamma: torch.Tensor = gamma,
    ) -> MDState:
        """Perform one complete Langevin dynamics integration step.

        This function implements the BAOAB splitting scheme for Langevin dynamics,
        which provides accurate sampling of the canonical ensemble. The integration
        sequence is:
        1. Half momentum update (B)
        2. Half position update (A)
        3. Full stochastic update (O)
        4. Half position update (A)
        5. Half momentum update (B)

        Args:
            state: Current system state containing positions, momenta, forces
            dt: Integration timestep
            kT: Target temperature in energy units
            gamma: Friction coefficient for Langevin thermostat

        Returns:
            MDState: Updated state after one complete Langevin step
        """
        if isinstance(gamma, float):
            gamma = torch.tensor(gamma, device=device, dtype=dtype)

        if isinstance(dt, float):
            dt = torch.tensor(dt, device=device, dtype=dtype)

        state = momentum_step(state, dt / 2)
        state = position_step(state, dt / 2)
        state = ou_step(state, dt, kT, gamma, device, dtype)
        state = position_step(state, dt / 2)

        model_output = model(state)
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]

        return momentum_step(state, dt / 2)

    return langevin_init, langevin_update


@dataclass
class NPTLangevinState(SimState):
    """State information for an NPT system with Langevin dynamics.

    This class represents the complete state of a molecular system being integrated
    in the NPT (constant particle number, pressure, temperature) ensemble using
    Langevin dynamics.

    Attributes:
        energy: Total energy of the system
        forces: Forces acting on each particle
        stress: Stress tensor of the system
        velocities: Velocities of each particle
        reference_cell: Original cell vectors used as reference for scaling
        cell_positions: Cell positions (effectively the volume)
        cell_velocities: Cell velocities (rate of volume change)
        cell_masses: Masses associated with the cell degrees of freedom
    """

    # System state variables
    energy: torch.Tensor
    forces: torch.Tensor
    stress: torch.Tensor
    velocities: torch.Tensor

    # Cell variables
    reference_cell: torch.Tensor
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_masses: torch.Tensor

    @property
    def momenta(self) -> torch.Tensor:
        """Calculate momenta from velocities and masses.

        Returns:
            The momenta of the particles
        """
        return self.masses.unsqueeze(-1) * self.velocities


def npt_langevin(  # noqa: C901, PLR0915
    *,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
    external_pressure: torch.Tensor,
    alpha: torch.Tensor | None = None,
    cell_alpha: torch.Tensor | None = None,
    b_tau: torch.Tensor | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NPT (canonical) integrator using Langevin dynamics.

    This function sets up integration in the NPT ensemble, where particle number (N),
    pressure (P), and temperature (T) are conserved. It returns both an initialization
    function and an update function for time evolution.

    Args:
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Target temperature in energy units
        external_pressure: Target pressure for the system
        alpha: Friction coefficient for position updates (default: 1/(100*dt))
        cell_alpha: Friction coefficient for cell updates (default: 1/(100*dt))
        b_tau: Pressure damping parameter (default: 1/(1000*dt))

    Returns:
        tuple:
            - Callable[[SimState | StateDict, torch.Tensor], MDState]: Function to
              initialize the MDState from input data and kT
            - Callable[[MDState, torch.Tensor], MDState]: Update function that evolves
              system by one timestep
    """
    device = model.device
    dtype = model.dtype

    # Set default values for coupling parameters if not provided
    alpha = alpha or 1 / (100 * dt)
    cell_alpha = cell_alpha or 1 / (100 * dt)
    b_tau = b_tau or 1 / (1000 * dt)

    # Convert float parameters to tensors with appropriate device and dtype
    if isinstance(alpha, float):
        alpha = torch.tensor(alpha, device=device, dtype=dtype)

    if isinstance(cell_alpha, float):
        cell_alpha = torch.tensor(cell_alpha, device=device, dtype=dtype)

    if isinstance(b_tau, float):
        b_tau = torch.tensor(b_tau, device=device, dtype=dtype)

    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    def cell_beta(
        state: NPTLangevinState,
        cell_alpha: torch.Tensor,
        kT: torch.Tensor,
        dt: torch.Tensor,
        device: torch.device = device,
        dtype: torch.dtype = dtype,
    ) -> torch.Tensor:
        """Generate random noise for cell fluctuations in NPT dynamics.

        This function creates properly scaled random noise for cell dynamics in NPT
        simulations, following the fluctuation-dissipation theorem. The noise amplitude
        is scaled to maintain the target temperature.

        Args:
            state: Current NPT state
            cell_alpha: Coupling parameter controlling noise strength
            kT: System temperature in energy units
            dt: Integration timestep
            device: Device to place the tensor on (CPU or GPU)
            dtype: Data type for the tensor

        Returns:
            torch.Tensor: Scaled random noise for cell dynamics
        """
        # Generate standard normal distribution (zero mean, unit variance)
        noise = torch.randn_like(state.cell_positions, device=device, dtype=dtype)

        # Scale to satisfy the fluctuation-dissipation theorem
        # The standard deviation should be sqrt(2*alpha*kB*T*dt)
        scaling_factor = torch.sqrt(2.0 * cell_alpha * kT * dt)

        return scaling_factor * noise

    def beta(
        state: NPTLangevinState,
        alpha: torch.Tensor,
        kT: torch.Tensor,
        dt: torch.Tensor,
        device: torch.device = device,
        dtype: torch.dtype = dtype,
    ) -> torch.Tensor:
        """Generate random noise for particle fluctuations in NPT dynamics.

        This function creates properly scaled random noise for particle dynamics in NPT
        simulations, following the fluctuation-dissipation theorem. The noise amplitude
        is scaled to maintain the target temperature.

        Args:
            state: Current NPT state
            alpha: Coupling parameter controlling noise strength
            kT: System temperature in energy units
            dt: Integration timestep
            device: Device to place the tensor on (CPU or GPU)
            dtype: Data type for the tensor

        Returns:
            torch.Tensor: Scaled random noise for particle dynamics
        """
        # Generate standard normal distribution (zero mean, unit variance)
        noise = torch.randn_like(state.positions, device=device, dtype=dtype)

        # Scale to satisfy the fluctuation-dissipation theorem
        # The standard deviation should be sqrt(2*alpha*kB*T*dt)
        scaling_factor = torch.sqrt(2.0 * alpha * kT * dt)

        return scaling_factor * noise

    def cell_position_step(
        state: NPTLangevinState,
        dt: torch.Tensor,
        pressure_force: torch.Tensor,
        kT: torch.Tensor = kT,
        cell_alpha: torch.Tensor = cell_alpha,
    ) -> NPTLangevinState:
        """Update the cell position in NPT dynamics.

        This function updates the cell position in NPT dynamics using the barostat force.
        It applies a half-step update to the cell position based on the barostat force.

        Args:
            state: Current NPT state
            dt: Integration timestep
            pressure_force: Pressure force for barostat
            kT: Target temperature in energy units
            cell_alpha: Cell friction coefficient

        Returns:
            NPTLangevinState: Updated state with new cell positions
        """
        # Calculate effective mass term
        Q_2 = 2 * state.cell_masses

        # Calculate damping factor for cell position update
        cell_b = 1 / (1 + ((cell_alpha * dt) / Q_2))

        # Deterministic velocity contribution
        c_1 = cell_b * dt * state.cell_velocities

        # Force contribution
        c_2 = cell_b * dt * dt * pressure_force / Q_2

        # Random noise contribution (thermal fluctuations)
        c_3 = (
            cell_b
            * dt
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
        kT: torch.Tensor = kT,
    ) -> NPTLangevinState:
        """Update the cell momentum in NPT dynamics.

        This function updates the cell velocities based on the pressure forces and
        thermal fluctuations, following the Langevin dynamics equations.

        Args:
            state: Current NPT state
            F_p_n: Previous pressure force
            dt: Integration timestep
            pressure_force: Updated pressure force
            cell_alpha: Cell friction coefficient
            kT: Target temperature in energy units

        Returns:
            NPTLangevinState: Updated state with new cell velocities
        """
        # Calculate denominator for update equations
        Q_2 = 2 * state.cell_masses

        # Calculate damping factors
        cell_a = (1 - (cell_alpha * dt) / Q_2) / (1 + (cell_alpha * dt) / Q_2)
        cell_b = 1 / (1 + (cell_alpha * dt) / Q_2)

        # Deterministic velocity contribution
        c_1 = cell_alpha * state.cell_velocities

        # Force contribution (average of initial and final forces)
        c_2 = dt / Q_2 * (cell_a * F_p_n + pressure_force)

        # Random noise contribution (thermal fluctuations)
        c_3 = (
            cell_b
            * cell_beta(state=state, cell_alpha=cell_alpha, kT=kT, dt=dt)
            / state.cell_masses
        )

        # Update cell velocities with all contributions
        state.cell_velocities = c_1 + c_2 + c_3
        return state

    def compute_cell_force(
        state: NPTLangevinState,
        external_pressure: torch.Tensor,
        kT: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cell force in NPT dynamics.

        This function calculates the force on the cell based on the difference between
        the internal stress and the external pressure.

        Args:
            state: Current NPT state
            external_pressure: Target external pressure
            kT: System temperature in energy units

        Returns:
            torch.Tensor: Computed cell force for barostat
        """
        # Kinetic contribution
        N = state.n_atoms
        volume = state.cell_positions
        KE_cell = N * kT

        # Calculate internal pressure from stress tensor
        # (average of diagonal elements for isotropic pressure)
        internal_pressure = torch.trace(state.stress) / state.positions.shape[1]

        # Force is proportional to pressure difference
        # F = V * (P_internal - P_external) + kinetic contribution
        return KE_cell - (internal_pressure * volume) - (external_pressure * volume)

    def langevin_position_step(
        state: NPTLangevinState,
        L_n: torch.Tensor,
        dt: torch.Tensor,
        kT: torch.Tensor,
    ) -> NPTLangevinState:
        """Update the particle positions in NPT dynamics.

        This function updates the particle positions in NPT dynamics, accounting for
        both the forces on particles and the cell volume changes.

        Args:
            state: Current NPT state
            L_n: Previous cell length scale
            dt: Integration timestep
            kT: Target temperature in energy units

        Returns:
            NPTLangevinState: Updated state with new positions
        """
        # Calculate effective mass term
        M_2 = 2 * state.masses.unsqueeze(-1)

        # Calculate new cell length scale (cube root of volume for isotropic scaling)
        L_n_new = torch.pow(state.cell_positions, 1 / 3)

        # Calculate damping factor
        b = 1 / (1 + ((alpha * dt) / M_2))

        # Scale positions due to cell volume change
        c_1 = (L_n_new / L_n) * state.positions

        # Time step factor with average length scale
        c_2 = (2 * L_n_new / (L_n_new + L_n)) * b * dt

        # Velocity and force contributions with random noise
        c_3 = (
            state.velocities
            + dt * state.forces / (M_2)
            + 1 / (M_2) * beta(state=state, alpha=alpha, kT=kT, dt=dt)
        )

        # Update positions with all contributions
        state.positions = c_1 + c_2 * c_3

        # Apply periodic boundary conditions if needed
        if state.pbc:
            new_positions = pbc_wrap_general(
                positions=state.positions, lattice_vectors=state.cell.T
            )
            state.positions = new_positions

        return state

    def langevin_velocity_step(
        state: NPTLangevinState,
        forces: torch.Tensor,
        dt: torch.Tensor,
        kT: torch.Tensor,
        device: torch.device = device,
        dtype: torch.dtype = dtype,
    ) -> NPTLangevinState:
        """Update the particle velocities in NPT dynamics.

        This function updates the particle velocities based on the forces and
        thermal fluctuations, following the Langevin dynamics equations.

        Args:
            state: Current NPT state
            forces: Forces on particles
            dt: Integration timestep
            kT: Target temperature in energy units
            device: Device to place the tensor on (CPU or GPU)
            dtype: Data type for the tensor

        Returns:
            NPTLangevinState: Updated state with new velocities
        """
        # Calculate denominator for update equations
        M_2 = 2 * state.masses.unsqueeze(-1)

        # Calculate damping factors
        a = (1 - (alpha * dt) / M_2) / (1 + (alpha * dt) / M_2)
        b = 1 / (1 + (alpha * dt) / M_2)

        # Velocity contribution with damping
        c_1 = a * state.velocities

        # Force contribution (average of initial and final forces)
        c_2 = dt * ((a * forces) + state.forces) / M_2

        # Random noise contribution (thermal fluctuations)
        c_3 = (
            b
            * beta(state=state, alpha=alpha, kT=kT, dt=dt, device=device, dtype=dtype)
            / state.masses.unsqueeze(-1)
        )

        # Update velocities with all contributions
        state.velocities = c_1 + c_2 + c_3
        return state

    def npt_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        device: torch.device = device,
        dtype: torch.dtype = dtype,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MDState:
        """Initialize an NPT state from input data.

        This function creates an initial NPT state from the provided state or
        state dictionary, initializing all necessary variables for NPT simulation.

        Args:
            state: Either a SimState object or a dictionary containing positions,
                momenta, cell, pbc
            kT: Target temperature in energy units
            device: Device to place the tensor on (CPU or GPU)
            dtype: Data type for the tensor
            seed: Random seed for reproducibility
            **kwargs: Additional state arguments

        Returns:
            MDState: Initialized state for NPT integration
        """
        # Convert dictionary to BaseState if needed
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Check if there is an extra batch dimension
        if state.cell.dim() == 3:
            state.cell = state.cell.squeeze(0)

        # Get atomic numbers from kwargs or state
        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Compute initial energy, forces, and stress
        model_output = model(state)

        # Initialize cell variables
        # Cell position is the volume
        cell_positions = torch.linalg.det(state.cell)
        # Initial cell velocity is zero
        cell_velocities = torch.zeros_like(cell_positions)
        # Cell mass depends on system size, temperature and barostat time constant
        cell_masses = (state.n_atoms + 1) * kT * b_tau * b_tau

        # Initialize momenta (from kwargs or calculated)
        momenta = kwargs.get(
            "momenta",
            calculate_momenta(state.positions, state.masses, kT, device, dtype, seed),
        )

        # Create and return the NPT state
        return NPTLangevinState(
            positions=state.positions,
            velocities=momenta / state.masses.unsqueeze(-1),
            cell=state.cell,
            pbc=state.pbc,
            masses=state.masses,
            energy=model_output["energy"],
            forces=model_output["forces"],
            stress=model_output["stress"],
            reference_cell=state.cell.clone(),
            cell_positions=cell_positions,
            cell_velocities=cell_velocities,
            cell_masses=cell_masses,
            atomic_numbers=atomic_numbers,
        )

    def npt_update(
        state: NPTLangevinState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        external_pressure: torch.Tensor = external_pressure,
        alpha: torch.Tensor = alpha,
        cell_alpha: torch.Tensor = cell_alpha,
    ) -> NPTLangevinState:
        """Update the NPT state for one timestep.

        This function performs a single timestep of NPT integration for the given state.
        It includes both position and cell updates, following a modified BAOAB scheme
        adapted for NPT dynamics.

        Args:
            state: Current NPT state
            dt: Integration timestep
            kT: Target temperature in energy units
            external_pressure: Target external pressure
            alpha: Position friction coefficient
            cell_alpha: Cell friction coefficient

        Returns:
            NPTLangevinState: Updated NPT state after one timestep
        """
        # Convert float parameters to tensors if needed
        if isinstance(alpha, float):
            alpha = torch.tensor(alpha, device=device, dtype=dtype)

        if isinstance(kT, float):
            kT = torch.tensor(kT, device=device, dtype=dtype)

        if isinstance(cell_alpha, float):
            cell_alpha = torch.tensor(cell_alpha, device=device, dtype=dtype)

        if isinstance(dt, float):
            dt = torch.tensor(dt, device=device, dtype=dtype)

        # Update barostat mass based on current temperature
        # This ensures proper coupling as temperature changes
        state.cell_masses = (state.n_atoms + 1) * kT * b_tau * b_tau

        # Compute model output for current state
        model_output = model(state)
        state.forces = model_output["forces"]
        state.stress = model_output["stress"]

        # Store initial values for integration
        forces = state.forces
        F_p_n = compute_cell_force(
            state=state, external_pressure=external_pressure, kT=kT
        )
        L_n = torch.pow(state.cell_positions, 1 / 3)  # Current length scale

        # Step 1: Update cell position
        state = cell_position_step(state=state, dt=dt, pressure_force=F_p_n, kT=kT)

        # Update cell (currently only isotropic fluctuations)
        dim = state.positions.shape[1]
        V_0 = torch.det(state.reference_cell)
        V = state.cell_positions

        # Scale cell uniformly in all dimensions
        new_cell = (V / V_0) ** (1.0 / dim) * state.reference_cell
        state.cell = new_cell

        # NOTE: Better to scale each dimension independently?
        # state.cell = torch.tensor([[L_x, 0, 0],
        #                            [0, L_x, 0],
        #                            [0, 0, L_x]], device=device, dtype=dtype)

        # Step 2: Update particle positions
        state = langevin_position_step(state=state, L_n=L_n, dt=dt, kT=kT)

        # state.positions = state.positions + dt * state.velocities

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

        # state.velocities = state.velocities + dt * forces / state.masses.unsqueeze(-1)

        return state  # noqa: RET504

    return npt_init, npt_update


@dataclass
class NoseHooverChain:
    """State information for a Nose-Hoover chain thermostat.

    The Nose-Hoover chain is a deterministic thermostat that maintains constant
    temperature by coupling the system to a chain of thermostats. Each thermostat
    in the chain has its own positions, momenta, and masses.

    Attributes:
        positions: Positions of the chain thermostats. Shape: [chain_length]
        momenta: Momenta of the chain thermostats. Shape: [chain_length]
        masses: Masses of the chain thermostats. Shape: [chain_length]
        tau: Thermostat relaxation time. Longer values give better stability
            but worse temperature control. Shape: scalar
        kinetic_energy: Current kinetic energy of the coupled system. Shape: scalar
        degrees_of_freedom: Number of degrees of freedom in the coupled system
    """

    positions: torch.Tensor
    momenta: torch.Tensor
    masses: torch.Tensor
    tau: torch.Tensor
    kinetic_energy: torch.Tensor
    degrees_of_freedom: int


@dataclass
class NoseHooverChainFns:
    """Collection of functions for operating on a Nose-Hoover chain.

    Attributes:
        initialize (Callable): Function to initialize the chain state
        half_step (Callable): Function to perform half-step integration of chain
        update_mass (Callable): Function to update the chain masses
    """

    initialize: Callable
    half_step: Callable
    update_mass: Callable


# Suzuki-Yoshida weights for multi-timestep integration
SUZUKI_YOSHIDA_WEIGHTS = {
    1: torch.tensor([1.0]),
    3: torch.tensor([0.828981543588751, -0.657963087177502, 0.828981543588751]),
    5: torch.tensor(
        [
            0.2967324292201065,
            0.2967324292201065,
            -0.186929716880426,
            0.2967324292201065,
            0.2967324292201065,
        ]
    ),
    7: torch.tensor(
        [
            0.784513610477560,
            0.235573213359357,
            -1.17767998417887,
            1.31518632068391,
            -1.17767998417887,
            0.235573213359357,
            0.784513610477560,
        ]
    ),
}


def construct_nose_hoover_chain(
    dt: torch.Tensor,
    chain_length: int,
    chain_steps: int,
    sy_steps: int,
    tau: torch.Tensor,
) -> NoseHooverChainFns:
    """Creates functions to simulate a Nose-Hoover Chain thermostat.

    Implements the direct translation method from Martyna et al. for thermal ensemble
    sampling using Nose-Hoover chains. The chains are updated using a symmetric
    splitting scheme with two half-steps per simulation step.

    The integration uses a multi-timestep approach with Suzuki-Yoshida (SY) splitting:
    - The chain evolution is split into nc substeps (chain_steps)
    - Each substep is further split into sy_steps
    - Each SY step has length δi = Δt*wi/nc where wi are the SY weights

    Args:
        dt: Simulation timestep
        chain_length: Number of thermostats in the chain
        chain_steps: Number of outer substeps for chain integration
        sy_steps: Number of Suzuki-Yoshida steps (must be 1, 3, 5, or 7)
        tau: Temperature equilibration timescale (in units of dt)
            Larger values give better stability but slower equilibration

    Returns:
        NoseHooverChainFns containing:
        - initialize: Function to create initial chain state
        - half_step: Function to evolve chain for half timestep
        - update_mass: Function to update chain masses

    References:
        Martyna et al. "Nose-Hoover chains: the canonical ensemble via
            continuous dynamics"
        J. Chem. Phys. 97, 2635 (1992)
    """

    def init_fn(
        degrees_of_freedom: int, KE: torch.Tensor, kT: torch.Tensor
    ) -> NoseHooverChain:
        """Initialize a Nose-Hoover chain state.

        Args:
            degrees_of_freedom: Number of degrees of freedom in coupled system
            KE: Initial kinetic energy of the system
            kT: Target temperature in energy units

        Returns:
            Initial NoseHooverChain state
        """
        device = KE.device
        dtype = KE.dtype

        xi = torch.zeros(chain_length, dtype=dtype, device=device)
        p_xi = torch.zeros(chain_length, dtype=dtype, device=device)

        Q = kT * tau**2 * torch.ones(chain_length, dtype=dtype, device=device)
        Q[0] *= degrees_of_freedom

        return NoseHooverChain(xi, p_xi, Q, tau, KE, degrees_of_freedom)

    def substep_fn(
        delta: torch.Tensor, P: torch.Tensor, state: NoseHooverChain, kT: torch.Tensor
    ) -> tuple[torch.Tensor, NoseHooverChain, torch.Tensor]:
        """Perform single update of chain parameters and rescale velocities.

        Args:
            delta: Integration timestep for this substep
            P: System momenta to be rescaled
            state: Current chain state
            kT: Target temperature

        Returns:
            Tuple of (rescaled momenta, updated chain state, temperature)
        """
        xi, p_xi, Q, _tau, KE, DOF = (
            state.positions,
            state.momenta,
            state.masses,
            state.tau,
            state.kinetic_energy,
            state.degrees_of_freedom,
        )

        delta_2 = delta / 2.0
        delta_4 = delta_2 / 2.0
        delta_8 = delta_4 / 2.0

        M = chain_length - 1

        # Update chain momenta backwards
        G = p_xi[M - 1] ** 2 / Q[M - 1] - kT
        p_xi[M] += delta_4 * G

        for m in range(M - 1, 0, -1):
            G = p_xi[m - 1] ** 2 / Q[m - 1] - kT
            scale = torch.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
            p_xi[m] = scale * (scale * p_xi[m] + delta_4 * G)

        # Update system coupling
        G = 2.0 * KE - DOF * kT
        scale = torch.exp(-delta_8 * p_xi[1] / Q[1])
        p_xi[0] = scale * (scale * p_xi[0] + delta_4 * G)

        # Rescale system momenta
        scale = torch.exp(-delta_2 * p_xi[0] / Q[0])
        KE = KE * scale**2
        P = P * scale

        # Update positions
        xi = xi + delta_2 * p_xi / Q

        # Update chain momenta forwards
        G = 2.0 * KE - DOF * kT
        for m in range(M):
            scale = torch.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
            p_xi[m] = scale * (scale * p_xi[m] + delta_4 * G)
            G = p_xi[m] ** 2 / Q[m] - kT
        p_xi[M] += delta_4 * G

        return P, NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF), kT

    def half_step_chain_fn(
        P: torch.Tensor, state: NoseHooverChain, kT: torch.Tensor
    ) -> tuple[torch.Tensor, NoseHooverChain]:
        """Evolve chain for half timestep using multi-timestep integration.

        Args:
            P: System momenta to be rescaled
            state: Current chain state
            kT: Target temperature

        Returns:
            Tuple of (rescaled momenta, updated chain state)
        """
        if chain_steps == 1 and sy_steps == 1:
            P, state, _ = substep_fn(dt, P, state, kT)
            return P, state

        delta = dt / chain_steps
        weights = SUZUKI_YOSHIDA_WEIGHTS[sy_steps]

        for step in range(chain_steps * sy_steps):
            d = delta * weights[step % sy_steps]
            P, state, _ = substep_fn(d, P, state, kT)

        return P, state

    def update_chain_mass_fn(state: NoseHooverChain, kT: torch.Tensor) -> NoseHooverChain:
        """Update chain masses to maintain target oscillation period.

        Args:
            state: Current chain state
            kT: Target temperature

        Returns:
            Updated chain state with new masses
        """
        device = state.positions.device
        dtype = state.positions.dtype

        Q = kT * state.tau**2 * torch.ones(chain_length, dtype=dtype, device=device)
        Q[0] *= state.degrees_of_freedom

        return NoseHooverChain(
            state.positions,
            state.momenta,
            Q,
            state.tau,
            state.kinetic_energy,
            state.degrees_of_freedom,
        )

    return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)


@dataclass
class NVTNoseHooverState(MDState):
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    This class represents the complete state of a molecular system being integrated
    in the NVT (constant particle number, volume, temperature) ensemble using a
    Nose-Hoover chain thermostat. The thermostat maintains constant temperature
    through a deterministic extended system approach.

    Attributes:
        positions: Particle positions with shape [n_particles, n_dimensions]
        momenta: Particle momenta with shape [n_particles, n_dimensions]
        energy: Energy of the system
        forces: Forces on particles with shape [n_particles, n_dimensions]
        masses: Particle masses with shape [n_particles]
        cell: Simulation cell matrix with shape [n_dimensions, n_dimensions]
        pbc: Whether to use periodic boundary conditions
        chain: State variables for the Nose-Hoover chain thermostat

    Properties:
        velocities: Particle velocities computed as momenta/masses
            Has shape [n_particles, n_dimensions]

    Notes:
        - The Nose-Hoover chain provides deterministic temperature control
        - Extended system approach conserves an extended energy quantity
        - Chain variables evolve to maintain target temperature
        - Time-reversible when integrated with appropriate algorithms
    """

    chain: NoseHooverChain
    _chain_fns: NoseHooverChainFns

    @property
    def velocities(self) -> torch.Tensor:
        """Calculate velocities from momenta and masses.

        Returns:
            torch.Tensor: Particle velocities with shape [n_particles, n_dimensions]
        """
        return self.momenta / self.masses.unsqueeze(-1)


def nvt_nose_hoover(
    *,
    model: torch.nn.Module,
    dt: torch.Tensor,
    kT: torch.Tensor,
    chain_length: int = 3,
    chain_steps: int = 3,
    sy_steps: int = 3,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor, int | None, Any], NVTNoseHooverState],
    Callable[[NVTNoseHooverState, torch.Tensor], NVTNoseHooverState],
]:
    """Initialize NVT Nose-Hoover chain thermostat integration.

    This function sets up integration of an NVT system using a Nose-Hoover chain
    thermostat. The Nose-Hoover chain provides deterministic temperature control by
    coupling the system to a chain of thermostats. The integration scheme is
    time-reversible and conserves an extended energy quantity.

    Args:
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Target temperature in energy units
        chain_length: Number of thermostats in Nose-Hoover chain (default: 3)
        chain_steps: Number of chain integration substeps (default: 3)
        sy_steps: Number of Suzuki-Yoshida steps - must be 1, 3, 5, or 7 (default: 3)

    Returns:
        Tuple containing:
        - Initialization function that takes a state and returns NVTNoseHooverState
        - Update function that performs one complete integration step

    Notes:
        The initialization function accepts:
        - state: Initial system state (SimState or dict)
        - kT: Target temperature (optional, defaults to constructor value)
        - tau: Thermostat relaxation time (optional, defaults to 100*dt)
        - seed: Random seed for momenta initialization (optional)
        - **kwargs: Additional state variables

        The update function accepts:
        - state: Current NVTNoseHooverState
        - dt: Integration timestep (optional, defaults to constructor value)
        - kT: Target temperature (optional, defaults to constructor value)

        The integration sequence is:
        1. Update chain masses
        2. First half-step of chain evolution
        3. Full velocity Verlet step
        4. Update chain kinetic energy
        5. Second half-step of chain evolution
    """
    device = model.device
    dtype = model.dtype

    def nvt_nose_hoover_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        tau: torch.Tensor | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> NVTNoseHooverState:
        """Initialize the NVT Nose-Hoover state.

        Args:
            state: Initial system state as SimState or dict
            kT: Target temperature in energy units
            tau: Thermostat relaxation time (defaults to 100*dt)
            seed: Random seed for momenta initialization
            **kwargs: Additional state variables

        Returns:
            Initialized NVTNoseHooverState with positions, momenta, forces,
            and thermostat chain variables
        """
        # Set default tau if not provided
        if tau is None:
            tau = dt * 100.0

        # Create thermostat functions
        chain_fns = construct_nose_hoover_chain(
            dt, chain_length, chain_steps, sy_steps, tau
        )

        if not isinstance(state, SimState):
            state = SimState(**state)

        # Check if there is an extra batch dimension
        if state.cell.dim() == 3:
            state.cell = state.cell.squeeze(0)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        model_output = model(state)
        momenta = kwargs.get(
            "momenta",
            calculate_momenta(state.positions, state.masses, kT, device, dtype, seed),
        )

        # Calculate initial kinetic energy
        KE = calc_kinetic_energy(momenta, state.masses)

        # Initialize chain with calculated KE
        dof = count_dof(state.positions)

        # Initialize state
        state = NVTNoseHooverState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            chain=chain_fns.initialize(dof, KE, kT),
            _chain_fns=chain_fns,  # Store the chain functions
        )
        return state  # noqa: RET504

    def nvt_nose_hoover_update(
        state: NVTNoseHooverState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
    ) -> NVTNoseHooverState:
        """Perform one complete Nose-Hoover chain integration step.

        Args:
            state: Current system state containing positions, momenta, forces, and chain
            dt: Integration timestep
            kT: Target temperature in energy units

        Returns:
            Updated state after one complete Nose-Hoover step

        Notes:
            Integration sequence:
            1. Update chain masses based on target temperature
            2. First half-step of chain evolution
            3. Full velocity Verlet step
            4. Update chain kinetic energy
            5. Second half-step of chain evolution
        """
        # Get chain functions from state
        chain_fns = state._chain_fns  # noqa: SLF001
        chain = state.chain

        # Update chain masses based on target temperature
        chain = chain_fns.update_mass(chain, kT)

        # First half-step of chain evolution
        momenta, chain = chain_fns.half_step(state.momenta, chain, kT)
        state.momenta = momenta

        # Full velocity Verlet step
        state = velocity_verlet(state=state, dt=dt, model=model)

        # Update chain kinetic energy
        KE = calc_kinetic_energy(state.momenta, state.masses)
        chain.kinetic_energy = KE

        # Second half-step of chain evolution
        momenta, chain = chain_fns.half_step(state.momenta, chain, kT)
        state.momenta = momenta
        state.chain = chain

        return state

    return nvt_nose_hoover_init, nvt_nose_hoover_update


def nvt_nose_hoover_invariant(
    state: NVTNoseHooverState,
    kT: torch.Tensor,
) -> torch.Tensor:
    """Calculate the conserved quantity for NVT ensemble with Nose-Hoover thermostat.

    This function computes the conserved Hamiltonian of the extended system for
    NVT dynamics with a Nose-Hoover chain thermostat. The invariant includes:
    1. System potential energy
    2. System kinetic energy
    3. Chain thermostat energy terms

    This quantity should remain approximately constant during simulation and is
    useful for validating the thermostat implementation.

    Args:
        energy_fn: Function that computes system potential energy given positions
        state: Current state of the system including chain variables
        kT: Target temperature in energy units

    Returns:
        torch.Tensor: The conserved Hamiltonian of the extended NVT dynamics

    Notes:
        - Conservation indicates correct thermostat implementation
        - Drift in this quantity suggests numerical instability
        - Includes both physical and thermostat degrees of freedom
        - Useful for debugging thermostat behavior
    """
    # Calculate system energy terms
    e_pot = state.energy
    e_kin = calc_kinetic_energy(state.momenta, state.masses)

    # Get system degrees of freedom
    dof = count_dof(state.positions)

    # Start with system energy
    e_tot = e_pot + e_kin

    # Add first thermostat term
    c = state.chain
    e_tot = e_tot + c.momenta[0] ** 2 / (2 * c.masses[0]) + dof * kT * c.positions[0]

    # Add remaining chain terms
    for pos, momentum, mass in zip(
        c.positions[1:], c.momenta[1:], c.masses[1:], strict=True
    ):
        e_tot = e_tot + momentum**2 / (2 * mass) + kT * pos

    return e_tot


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
    device = model.device
    dtype = model.dtype

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
            >>> print(y)  # tensor([1.0000, 1.0017, 1.0067])
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
        return pbc_wrap_general(new_positions, state.current_cell.T)

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

        dim, n_particles = state.positions.shape
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
            calculate_momenta(state.positions, state.masses, kT, device, dtype, seed),
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
