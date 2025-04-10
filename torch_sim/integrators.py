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

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_sim.state import SimState, StateDict
from torch_sim.transforms import pbc_wrap_batched


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
        """Calculate velocities from momenta and masses.

        Returns:
            torch.Tensor: Velocities with shape [n_particles, n_dimensions]
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
        new_positions = pbc_wrap_batched(
            new_positions, state.cell.swapaxes(1, 2), state.batch
        )

    state.positions = new_positions
    return state


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
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_batches]
        kT (torch.Tensor): Temperature in energy units for initializing momenta,
            either scalar or with shape [n_batches]
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
                scalar or with shape [n_batches]
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
            calculate_momenta(state.positions, state.masses, state.batch, kT, seed),
        )

        initial_state = MDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            batch=state.batch,
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
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
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


def nvt_langevin(
    model: torch.nn.Module,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    gamma: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NVT (canonical) integrator using Langevin dynamics.

    This function sets up integration in the NVT ensemble, where particle number (N),
    volume (V), and temperature (T) are conserved. It returns both an initial state
    and an update function for time evolution.

    It uses Langevin dynamics with stochastic noise and friction to maintain constant
    temperature. The integration scheme combines deterministic velocity Verlet steps with
    stochastic Ornstein-Uhlenbeck processes following the BAOAB splitting scheme.

    Args:
        model (torch.nn.Module): Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_batches]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_batches]
        gamma (torch.Tensor, optional): Friction coefficient for Langevin thermostat,
            either scalar or with shape [n_batches]. Defaults to 1/(100*dt).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - callable: Function to initialize the MDState from input data
              with signature: init_fn(state, kT=kT, seed=seed) -> MDState
            - callable: Update function that evolves system by one timestep
              with signature: update_fn(state, dt=dt, kT=kT, gamma=gamma) -> MDState

    Notes:
        - Uses BAOAB splitting scheme for Langevin dynamics
        - Preserves detailed balance for correct NVT sampling
        - Handles periodic boundary conditions if enabled in state
        - Friction coefficient gamma controls the thermostat coupling strength
        - Weak coupling (small gamma) preserves dynamics but with slower thermalization
        - Strong coupling (large gamma) faster thermalization but may distort dynamics
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
    ) -> MDState:
        """Apply stochastic noise and friction for Langevin dynamics.

        This function implements the Ornstein-Uhlenbeck process for Langevin dynamics,
        applying random noise and friction forces to particle momenta. The noise amplitude
        is chosen to satisfy the fluctuation-dissipation theorem, ensuring proper
        sampling of the canonical ensemble at temperature kT.

        Args:
            state (MDState): Current system state containing positions, momenta, etc.
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_batches]
            gamma (torch.Tensor): Friction coefficient controlling noise strength,
                either scalar or with shape [n_batches]

        Returns:
            MDState: Updated state with new momenta after stochastic step

        Notes:
            - Implements the "O" step in the BAOAB Langevin integration scheme
            - Uses Ornstein-Uhlenbeck process for correct thermal sampling
            - Noise amplitude scales with sqrt(mass) for equipartition
            - Preserves detailed balance through fluctuation-dissipation relation
            - The equation implemented is:
              p(t+dt) = c1*p(t) + c2*sqrt(m)*N(0,1)
              where c1 = exp(-gamma*dt) and c2 = sqrt(kT*(1-c1²))
        """
        c1 = torch.exp(-gamma * dt)

        if isinstance(kT, torch.Tensor) and len(kT.shape) > 0:
            # kT is a tensor with shape (n_batches,)
            kT = kT[state.batch]

        c2 = torch.sqrt(kT * (1 - c1**2)).unsqueeze(-1)

        # Generate random noise from normal distribution
        noise = torch.randn_like(state.momenta, device=state.device, dtype=state.dtype)
        new_momenta = (
            c1 * state.momenta + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
        )
        state.momenta = new_momenta
        return state

    def langevin_init(
        state: SimState | StateDict,
        kT: torch.Tensor = kT,
        seed: int | None = seed,
    ) -> MDState:
        """Initialize an NVT state from input data for Langevin dynamics.

        Creates an initial state for NVT molecular dynamics by computing initial
        energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
        at the specified temperature.

        Args:
            state (SimState | StateDict): Either a SimState object or a dictionary
                containing positions, masses, cell, pbc, and other required state vars
            kT (torch.Tensor): Temperature in energy units for initializing momenta,
                either scalar or with shape [n_batches]
            seed (int, optional): Random seed for reproducibility

        Returns:
            MDState: Initialized state for NVT integration containing positions,
                momenta, forces, energy, and other required attributes

        Notes:
            The initial momenta are sampled from a Maxwell-Boltzmann distribution
            at the specified temperature. This provides a proper thermal initial
            state for the subsequent Langevin dynamics.
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        model_output = model(state)

        momenta = getattr(
            state,
            "momenta",
            calculate_momenta(state.positions, state.masses, state.batch, kT, seed),
        )

        initial_state = MDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            batch=state.batch,
            atomic_numbers=state.atomic_numbers,
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
        1. Half momentum update using forces (B step)
        2. Half position update using updated momenta (A step)
        3. Full stochastic update with noise and friction (O step)
        4. Half position update using updated momenta (A step)
        5. Half momentum update using new forces (B step)

        Args:
            state (MDState): Current system state containing positions, momenta, forces
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_batches]
            gamma (torch.Tensor): Friction coefficient for Langevin thermostat,
                either scalar or with shape [n_batches]

        Returns:
            MDState: Updated state after one complete Langevin step with new positions,
                momenta, forces, and energy
        """
        if isinstance(gamma, float):
            gamma = torch.tensor(gamma, device=device, dtype=dtype)

        if isinstance(dt, float):
            dt = torch.tensor(dt, device=device, dtype=dtype)

        state = momentum_step(state, dt / 2)
        state = position_step(state, dt / 2)
        state = ou_step(state, dt, kT, gamma)
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
    device = model.device
    dtype = model.dtype

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
        # Get current volumes for each batch
        volumes = torch.linalg.det(state.cell)  # shape: (n_batches,)

        # Reshape for broadcasting
        volumes = volumes.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

        # Create pressure tensor (diagonal with external pressure)
        if external_pressure.ndim == 0:
            # Scalar pressure - create diagonal pressure tensors for each batch
            pressure_tensor = external_pressure * torch.eye(3, device=device, dtype=dtype)
            pressure_tensor = pressure_tensor.unsqueeze(0).expand(state.n_batches, -1, -1)
        else:
            # Already a tensor with shape compatible with n_batches
            pressure_tensor = external_pressure

        # Calculate virials from stress and external pressure
        # Internal stress is negative of virial tensor divided by volume
        virial = -volumes * state.stress + pressure_tensor * volumes

        # Add kinetic contribution (kT * Identity)
        batch_kT = kT
        if kT.ndim == 0:
            batch_kT = kT.expand(state.n_batches)

        kinetic_term = batch_kT.view(-1, 1, 1) * torch.eye(
            3, device=device, dtype=dtype
        ).unsqueeze(0)

        return virial + kinetic_term

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
            state.positions = pbc_wrap_batched(
                state.positions, state.cell.swapaxes(1, 2), state.batch
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
