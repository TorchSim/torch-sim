"""Implementations of NVT integrators."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from torch_sim import transforms
from torch_sim.integrators.md import (
    MDState,
    calculate_momenta,
    momentum_step,
    position_step,
    velocity_verlet,
)
from torch_sim.quantities import calc_kinetic_energy, count_dof
from torch_sim.state import SimState
from torch_sim.typing import StateDict


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
    device, dtype = model.device, model.dtype

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
        """Velocities calculated from momenta and masses with shape
        [n_particles, n_dimensions].
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
        tuple containing:
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
    device, dtype = model.device, model.dtype

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
        return transforms.pbc_wrap_general(new_positions, state.current_cell.T)

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
