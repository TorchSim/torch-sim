"""Implementations of NVT integrators."""

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
    position_step,
    velocity_verlet,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.typing import StateDict


def _ou_step(
    state: MDState,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    gamma: float | torch.Tensor,
) -> MDState:
    """Apply stochastic noise and friction for Langevin dynamics.

    This function implements the Ornstein-Uhlenbeck process for Langevin dynamics,
    applying random noise and friction forces to particle momenta. The noise amplitude
    is chosen to satisfy the fluctuation-dissipation theorem, ensuring proper
    sampling of the canonical ensemble at temperature kT.

    Args:
        state (MDState): Current system state containing positions, momenta, etc.
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_systems]
        gamma (torch.Tensor): Friction coefficient controlling noise strength,
            either scalar or with shape [n_systems]

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
        # kT is a tensor with shape (n_systems,)
        kT = kT[state.system_idx]

    # Index c1 and c2 with state.system_idx to align shapes with state.momenta
    if isinstance(c1, torch.Tensor) and len(c1.shape) > 0:
        c1 = c1[state.system_idx]

    c2 = torch.sqrt(kT * (1 - torch.square(c1))).unsqueeze(-1)

    # Generate random noise from normal distribution
    noise = torch.randn_like(state.momenta, device=state.device, dtype=state.dtype)
    new_momenta = (
        c1.unsqueeze(-1) * state.momenta
        + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
    )
    state.momenta = new_momenta
    return state


def nvt_langevin_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    seed: int | None = None,
    **_kwargs: Any,
) -> MDState:
    """Initialize an NVT state from input data for Langevin dynamics.

    Creates an initial state for NVT molecular dynamics by computing initial
    energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
    at the specified temperature.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Either a SimState object or a dictionary containing positions,
            masses, cell, pbc, and other required state vars
        kT: Temperature in energy units for initializing momenta,
            either scalar or with shape [n_systems]
        seed: Random seed for reproducibility

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
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    return MDState(
        positions=state.positions,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        system_idx=state.system_idx,
        atomic_numbers=state.atomic_numbers,
    )


def nvt_langevin_step(
    model: ModelInterface,
    state: MDState,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    gamma: float | torch.Tensor | None = None,
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
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep, either scalar or shape [n_systems]
        kT: Target temperature in energy units, either scalar or
            with shape [n_systems]
        gamma: Friction coefficient for Langevin thermostat,
            either scalar or with shape [n_systems]. Defaults to 1/(100*dt).

    Returns:
        MDState: Updated state after one complete Langevin step with new positions,
            momenta, forces, and energy

    Notes:
        - Uses BAOAB splitting scheme for Langevin dynamics
        - Preserves detailed balance for correct NVT sampling
        - Handles periodic boundary conditions if enabled in state
        - Friction coefficient gamma controls the thermostat coupling strength
        - Weak coupling (small gamma) preserves dynamics but with slower thermalization
        - Strong coupling (large gamma) faster thermalization but may distort dynamics
    """
    device, dtype = model.device, model.dtype

    if gamma is None:
        gamma = 1 / (100 * dt)

    if isinstance(gamma, float):
        gamma = torch.tensor(gamma, device=device, dtype=dtype)

    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    state = momentum_step(state, dt / 2)
    state = position_step(state, dt / 2)
    state = _ou_step(state, dt, kT, gamma)
    state = position_step(state, dt / 2)

    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]

    return momentum_step(state, dt / 2)


@dataclass(kw_only=True)
class NVTNoseHooverState(MDState):
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    This class represents the complete state of a molecular system being integrated
    in the NVT (constant particle number, volume, temperature) ensemble using a
    Nose-Hoover chain thermostat. The thermostat maintains constant temperature
    through a deterministic extended system approach.

    Attributes:
        positions: Particle positions with shape [n_particles, n_dimensions]
        masses: Particle masses with shape [n_particles]
        cell: Simulation cell matrix with shape [n_dimensions, n_dimensions]
        pbc: Whether to use periodic boundary conditions
        momenta: Particle momenta with shape [n_particles, n_dimensions]
        energy: Energy of the system
        forces: Forces on particles with shape [n_particles, n_dimensions]
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

    _global_attributes = (
        MDState._global_attributes | {"chain", "_chain_fns"}  # noqa: SLF001
    )

    @property
    def velocities(self) -> torch.Tensor:
        """Velocities calculated from momenta and masses with shape
        [n_particles, n_dimensions].
        """
        return self.momenta / self.masses.unsqueeze(-1)

    def get_number_of_degrees_of_freedom(self) -> torch.Tensor:
        """Calculate degrees of freedom per system."""
        dof = super().get_number_of_degrees_of_freedom()
        return dof - 3  # Subtract 3 degrees of freedom for center of mass motion


def nvt_nose_hoover_init(
    model: ModelInterface,
    state: SimState | StateDict,
    *,
    kT: torch.Tensor,
    dt: torch.Tensor,
    tau: torch.Tensor | None = None,
    chain_length: int = 3,
    chain_steps: int = 3,
    sy_steps: int = 3,
    seed: int | None = None,
    **kwargs: Any,
) -> NVTNoseHooverState:
    """Initialize the NVT Nose-Hoover state.

    This function sets up integration of an NVT system using a Nose-Hoover chain
    thermostat. The Nose-Hoover chain provides deterministic temperature control by
    coupling the system to a chain of thermostats. The integration scheme is
    time-reversible and conserves an extended energy quantity.

    Args:
        model: Neural network model that computes energies and forces
        state: Initial system state as SimState or dict
        kT: Target temperature in energy units
        dt: Integration timestep
        tau: Thermostat relaxation time (defaults to 100*dt)
        chain_length: Number of thermostats in Nose-Hoover chain (default: 3)
        chain_steps: Number of chain integration substeps (default: 3)
        sy_steps: Number of Suzuki-Yoshida steps - must be 1, 3, 5, or 7 (default: 3)
        seed: Random seed for momenta initialization
        **kwargs: Additional state variables

    Returns:
        Initialized NVTNoseHooverState with positions, momenta, forces,
        and thermostat chain variables

    Notes:
        - The Nose-Hoover chain provides deterministic temperature control
        - Extended system approach conserves an extended energy quantity
        - Chain variables evolve to maintain target temperature
        - Time-reversible when integrated with appropriate algorithms
    """
    if tau is None:  # Set default tau if not provided
        tau = dt * 100.0

    # Create thermostat functions
    chain_fns = construct_nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)
    if not isinstance(state, SimState):
        state = SimState(**state)

    atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

    model_output = model(state)
    momenta = kwargs.get(
        "momenta",
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    # Calculate initial kinetic energy per system
    KE = ts.calc_kinetic_energy(
        masses=state.masses, momenta=momenta, system_idx=state.system_idx
    )

    # Calculate degrees of freedom per system
    n_atoms_per_system = torch.bincount(state.system_idx)
    dof_per_system = (
        n_atoms_per_system * state.positions.shape[-1]
    )  # n_atoms * n_dimensions

    # Initialize state
    return NVTNoseHooverState(
        positions=state.positions,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        atomic_numbers=atomic_numbers,
        system_idx=state.system_idx,
        chain=chain_fns.initialize(dof_per_system, KE, kT),
        _chain_fns=chain_fns,  # Store the chain functions
    )


def nvt_nose_hoover_step(
    model: ModelInterface,
    state: NVTNoseHooverState,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
) -> NVTNoseHooverState:
    """Perform one complete Nose-Hoover chain integration step.

    This function performs one integration step for an NVT system using a Nose-Hoover
    chain thermostat. The integration scheme is time-reversible and conserves an
    extended energy quantity. If the center of mass motion is removed initially,
    it remains removed throughout the simulation, so the degrees of freedom decreases
    by 3.

    Args:
        model: Neural network model that computes energies and forces
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
    momenta, chain = chain_fns.half_step(state.momenta, chain, kT, state.system_idx)
    state.momenta = momenta

    # Full velocity Verlet step
    state = velocity_verlet(state=state, dt=dt, model=model)

    # Update chain kinetic energy per system
    KE = ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )
    chain.kinetic_energy = KE

    # Second half-step of chain evolution
    momenta, chain = chain_fns.half_step(state.momenta, chain, kT, state.system_idx)
    state.momenta = momenta
    state.chain = chain

    return state


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
    # Calculate system energy terms per system
    e_pot = state.energy
    e_kin = ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )

    # Get system degrees of freedom per system
    n_atoms_per_system = torch.bincount(state.system_idx)
    dof = n_atoms_per_system * state.positions.shape[-1]  # n_atoms * n_dimensions

    # Start with system energy
    e_tot = e_pot + e_kin

    # Add first thermostat term
    c = state.chain
    # Ensure chain momenta and masses broadcast correctly with batch dimensions
    chain_ke_0 = torch.square(c.momenta[:, 0]) / (2 * c.masses[:, 0])
    chain_pe_0 = dof * kT * c.positions[:, 0]

    # If chain variables are scalars but we have batches, broadcast them
    if chain_ke_0.numel() == 1 and e_tot.numel() > 1:
        chain_ke_0 = chain_ke_0.expand_as(e_tot)
    if chain_pe_0.numel() == 1 and e_tot.numel() > 1:
        chain_pe_0 = chain_pe_0.expand_as(e_tot)

    e_tot = e_tot + chain_ke_0 + chain_pe_0

    # Add remaining chain terms
    for i in range(1, c.positions.shape[1]):
        pos = c.positions[:, i]
        momentum = c.momenta[:, i]
        mass = c.masses[:, i]

        chain_ke = momentum**2 / (2 * mass)
        chain_pe = kT * pos

        # Ensure proper broadcasting for batch dimensions
        if chain_ke.numel() == 1 and e_tot.numel() > 1:
            chain_ke = chain_ke.expand_as(e_tot)
        if chain_pe.numel() == 1 and e_tot.numel() > 1:
            chain_pe = chain_pe.expand_as(e_tot)

        e_tot = e_tot + chain_ke + chain_pe

    return e_tot


class NVTVRescaleState(MDState):
    """State information for an NVT system with a V-Rescale thermostat.

    This class represents the complete state of a molecular system being integrated
    in the NVT (constant particle number, volume, temperature) ensemble using a
    Velocity Rescaling thermostat. The thermostat maintains constant temperature
    through stochastic velocity rescaling.

    Attributes:
        positions: Particle positions with shape [n_particles, n_dimensions]
        masses: Particle masses with shape [n_particles]
        cell: Simulation cell matrix with shape [n_dimensions, n_dimensions]
        pbc: Whether to use periodic boundary conditions
        momenta: Particle momenta with shape [n_particles, n_dimensions]
        energy: Energy of the system
        forces: Forces on particles with shape [n_particles, n_dimensions]

    Notes:
        - The V-Rescale thermostat provides proper canonical sampling
        - Stochastic velocity rescaling ensures correct temperature distribution
        - Time-reversible when integrated with appropriate algorithms
    """

    def calc_dof(self) -> torch.Tensor:
        """Calculate the degrees of freedom per system."""
        return super().calc_dof() - 3  # Subtract 3 for center of mass motion


def _vrescale_update(
    state: MDState,
    tau: float | torch.Tensor,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
) -> MDState:
    """Update the momentum by a scaling factor as described by Eq.A7 Bussi et al.

    Note that we don't implement the optimize code from Bussi, which won't be useful
    on a high level framework like PyTorch.

    Args:
        state: Current MD state
        tau: Thermostat relaxation time
        kT: Target temperature
        dt: Integration timestep

    Returns:
        Updated state with rescaled momenta
    """
    device, dtype = state.device, state.dtype

    # Convert all inputs to tensors
    tau_tensor = torch.as_tensor(tau, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)

    # Calculate current temperature per system
    current_kT = ts.quantities.calc_kT(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )

    # Calculate degrees of freedom per system
    dof = state.calc_dof()

    # Ensure kT and tau have proper batch dimensions
    n_systems = current_kT.shape[0]
    if kT_tensor.dim() == 0:
        kT_tensor = kT_tensor.expand(n_systems)
    if tau_tensor.dim() == 0:
        tau_tensor = tau_tensor.expand(n_systems)

    # Calculate kinetic energies
    KE_old = dof * current_kT / 2
    KE_new = dof * kT_tensor / 2

    # Generate random numbers
    r1 = torch.randn(n_systems, device=device, dtype=dtype)
    # Sample Gamma((dof - 1)/2, 1/2) = \sum_2^{dof} X_i^2 where X_i ~ N(0,1)
    r2 = torch.distributions.Gamma((dof - 1) / 2, torch.ones_like(dof) / 2).sample()

    # Calculate scaling coefficients
    c1 = torch.exp(-dt_tensor / tau_tensor)
    c2 = (1 - c1) * KE_new / KE_old / dof

    # Calculate scaling factor
    scale = c1 + (c2 * (torch.square(r1) + r2)) + (2 * r1 * torch.sqrt(c1 * c2))
    lam = torch.sqrt(scale)

    # Apply scaling to momenta - map from system to atom indices
    state.momenta = state.momenta * lam[state.system_idx].unsqueeze(-1)
    return state


def nvt_vrescale_init(
    state: SimState | StateDict,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    seed: int | None = None,
    **_kwargs: Any,
) -> NVTVRescaleState:
    """Initialize an NVT state from input data for velocity rescaling dynamics.

    Creates an initial state for NVT molecular dynamics using the canonical
    sampling through velocity rescaling (CSVR) thermostat. This thermostat
    samples from the canonical ensemble by rescaling velocities with an
    appropriately chosen random factor.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Either a SimState object or a dictionary containing positions,
            masses, cell, pbc, and other required state vars
        kT: Temperature in energy units for initializing momenta,
            either scalar or with shape [n_systems]
        seed: Random seed for reproducibility

    Returns:
        MDState: Initialized state for NVT integration containing positions,
            momenta, forces, energy, and other required attributes

    Notes:
        The initial momenta are sampled from a Maxwell-Boltzmann distribution
        at the specified temperature. The V-Rescale thermostat provides proper
        canonical sampling through stochastic velocity rescaling.
    """
    if not isinstance(state, SimState):
        state = SimState(**state)

    model_output = model(state)

    momenta = getattr(
        state,
        "momenta",
        calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
    )

    return NVTVRescaleState(
        positions=state.positions,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        masses=state.masses,
        cell=state.cell,
        pbc=state.pbc,
        system_idx=state.system_idx,
        atomic_numbers=state.atomic_numbers,
    )


def nvt_vrescale_step(
    model: ModelInterface,
    state: NVTVRescaleState,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
) -> NVTVRescaleState:
    """Perform one complete V-Rescale dynamics integration step.

    This function implements the canonical sampling through velocity rescaling (V-Rescale)
    thermostat combined with velocity Verlet integration. The V-Rescale thermostat samples
    the canonical distribution by rescaling velocities with a properly chosen random
    factor that ensures correct canonical sampling.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep, either scalar or shape [n_systems]
        kT: Target temperature in energy units, either scalar or
            with shape [n_systems]
        tau: Thermostat relaxation time controlling the coupling strength,
            either scalar or with shape [n_systems]. Defaults to 100*dt.
        seed: Random seed for reproducibility

    Returns:
        MDState: Updated state after one complete V-Rescale step with new positions,
            momenta, forces, and energy

    Notes:
        - Uses V-Rescale thermostat for proper canonical ensemble sampling
        - Unlike Berendsen thermostat, V-Rescale samples the true canonical distribution
        - Integration sequence: V-Rescale rescaling + Velocity Verlet step
        - The rescaling factor follows the distribution derived in Bussi et al.

    References:
        Bussi G, Donadio D, Parrinello M. "Canonical sampling through velocity rescaling."
        The Journal of chemical physics, 126(1), 014101 (2007).
    """
    device, dtype = model.device, model.dtype

    if tau is None:
        tau = 100 * dt

    if isinstance(tau, float):
        tau = torch.tensor(tau, device=device, dtype=dtype)
    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    if isinstance(kT, float):
        kT = torch.tensor(kT, device=device, dtype=dtype)

    # Apply V-Rescale rescaling
    state = _vrescale_update(state, tau, kT, dt)

    # Perform velocity Verlet step
    return velocity_verlet(state=state, dt=dt, model=model)
