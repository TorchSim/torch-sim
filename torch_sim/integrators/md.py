"""Core molecular dynamics state and operations."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


@dataclass(kw_only=True)
class MDState(SimState):
    """State information for molecular dynamics simulations.

    This class represents the complete state of a molecular system being integrated
    with molecular dynamics. It extends the base SimState class to include additional
    attributes required for MD simulations, such as momenta, energy, and forces.
    The class also provides computed properties like velocities.

    Attributes:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        momenta (torch.Tensor): Particle momenta [n_particles, n_dim]
        energy (torch.Tensor): Total energy of the system [n_systems]
        forces (torch.Tensor): Forces on particles [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        cell (torch.Tensor): Simulation cell matrix [n_systems, n_dim, n_dim]
        pbc (bool): Whether to use periodic boundary conditions
        system_idx (torch.Tensor): System indices [n_particles]
        atomic_numbers (torch.Tensor): Atomic numbers [n_particles]

    Properties:
        velocities (torch.Tensor): Particle velocities [n_particles, n_dim]
        n_systems (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    momenta: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor

    _atom_attributes = (
        SimState._atom_attributes | {"momenta", "forces"}  # noqa: SLF001
    )
    _system_attributes = (
        SimState._system_attributes | {"energy"}  # noqa: SLF001
    )

    @property
    def velocities(self) -> torch.Tensor:
        """Velocities calculated from momenta and masses with shape
        [n_particles, n_dimensions].
        """
        return self.momenta / self.masses.unsqueeze(-1)

    def set_momenta(self, new_momenta: torch.Tensor) -> None:
        """Set new momenta, applying any constraints as needed."""
        if self.constraints is not None:
            for constraint in self.constraints:
                constraint.adjust_momenta(self, new_momenta)
        self.momenta = new_momenta


def calculate_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    system_idx: torch.Tensor,
    kT: float | torch.Tensor,
    seed: int | None = None,
) -> torch.Tensor:
    """Initialize particle momenta based on temperature.

    Generates random momenta for particles following the Maxwell-Boltzmann
    distribution at the specified temperature. The center of mass motion
    is removed to prevent system drift.

    Args:
        positions (torch.Tensor): Particle positions [n_particles, n_dim]
        masses (torch.Tensor): Particle masses [n_particles]
        system_idx (torch.Tensor): System indices [n_particles]
        kT (torch.Tensor): Temperature in energy units [n_systems]
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
        # kT is a tensor with shape (n_systems,)
        kT = kT[system_idx]

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    systemwise_momenta = torch.zeros(
        size=(int(system_idx[-1]) + 1, momenta.shape[1]), device=device, dtype=dtype
    )

    # create 3 copies of system_idx
    system_idx_3 = system_idx.view(-1, 1).repeat(1, 3)
    bincount = torch.bincount(system_idx)
    mean_momenta = torch.scatter_reduce(
        systemwise_momenta,
        dim=0,
        index=system_idx_3,
        src=momenta,
        reduce="sum",
    ) / bincount.view(-1, 1)

    return torch.where(
        torch.repeat_interleave(bincount > 1, bincount).view(-1, 1),
        momenta - mean_momenta[system_idx],
        momenta,
    )


def momentum_step[T: MDState](state: T, dt: float | torch.Tensor) -> T:
    """Update particle momenta using current forces.

    This function performs the momentum update step of velocity Verlet integration
    by applying forces over the timestep dt. It implements the equation:
    p(t+dt) = p(t) + F(t) * dt

    Args:
        state (MDState): Current system state containing forces and momenta
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_systems]

    Returns:
        MDState: Updated state with new momenta after force application

    """
    new_momenta = state.momenta + state.forces * dt
    state.set_momenta(new_momenta)
    return state


def position_step[T: MDState](state: T, dt: float | torch.Tensor) -> T:
    """Update particle positions using current velocities.

    This function performs the position update step of velocity Verlet integration
    by propagating particles according to their velocities over timestep dt.
    It implements the equation: r(t+dt) = r(t) + v(t) * dt

    Args:
        state (MDState): Current system state containing positions and velocities
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_systems]

    Returns:
        MDState: Updated state with new positions after propagation

    """
    new_positions = state.positions + state.velocities * dt
    state.set_positions(new_positions)

    # if state.pbc:
    #     # Split positions and cells by system
    #     new_positions = transforms.pbc_wrap_batched(
    #         state.positions, state.cell, state.system_idx
    #     )
    #     state.positions = new_positions  # no constraints applied
    return state


def velocity_verlet[T: MDState](state: T, dt: torch.Tensor, model: ModelInterface) -> T:
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

        Q = kT * torch.square(tau) * torch.ones(chain_length, dtype=dtype, device=device)
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
        G = torch.square(p_xi[M - 1]) / Q[M - 1] - kT
        p_xi[M] += delta_4 * G

        for m in range(M - 1, 0, -1):
            G = torch.square(p_xi[m - 1]) / Q[m - 1] - kT
            scale = torch.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
            p_xi[m] = scale * (scale * p_xi[m] + delta_4 * G)

        # Update system coupling
        G = 2.0 * KE - DOF * kT
        scale = torch.exp(-delta_8 * p_xi[1] / Q[1])
        p_xi[0] = scale * (scale * p_xi[0] + delta_4 * G)

        # Rescale system momenta
        scale = torch.exp(-delta_2 * p_xi[0] / Q[0])
        KE = KE * torch.square(scale)
        P = P * scale

        # Update positions
        xi = xi + delta_2 * p_xi / Q

        # Update chain momenta forwards
        G = 2.0 * KE - DOF * kT
        for m in range(M):
            scale = torch.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
            p_xi[m] = scale * (scale * p_xi[m] + delta_4 * G)
            G = torch.square(p_xi[m]) / Q[m] - kT
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

    def update_chain_mass_fn(
        chain_state: NoseHooverChain, kT: torch.Tensor
    ) -> NoseHooverChain:
        """Update chain masses to maintain target oscillation period.

        Args:
            chain_state: Current chain state
            kT: Target temperature

        Returns:
            Updated chain state with new masses
        """
        device = chain_state.positions.device
        dtype = chain_state.positions.dtype

        Q = (
            kT
            * torch.square(chain_state.tau)
            * torch.ones(chain_length, dtype=dtype, device=device)
        )
        Q[0] *= chain_state.degrees_of_freedom

        return NoseHooverChain(
            chain_state.positions,
            chain_state.momenta,
            Q,
            chain_state.tau,
            chain_state.kinetic_energy,
            chain_state.degrees_of_freedom,
        )

    return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)
