"""Implementations of NVT integrators."""

from dataclasses import dataclass
from typing import Any

import torch

import torch_sim as ts
from torch_sim._duecredit import dcite
from torch_sim.integrators.md import (
    MDState,
    NoseHooverChain,
    NoseHooverChainFns,
    construct_nose_hoover_chain,
    initialize_momenta,
    momentum_step,
    position_step,
    velocity_verlet_step,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


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
    gamma_dt = -gamma * dt
    exp_arg = torch.as_tensor(gamma_dt, device=state.device, dtype=state.dtype)
    c1 = torch.exp(exp_arg)

    if isinstance(kT, torch.Tensor) and len(kT.shape) > 0:
        # kT is a tensor with shape (n_systems,)
        kT = kT[state.system_idx]

    # Index c1 and c2 with state.system_idx to align shapes with state.momenta
    if isinstance(c1, torch.Tensor) and len(c1.shape) > 0:
        c1 = c1[state.system_idx]

    c2 = torch.sqrt(kT * (1 - torch.square(c1))).unsqueeze(-1)

    # Generate random noise from normal distribution
    noise = torch.randn(
        state.momenta.shape,
        device=state.device,
        dtype=state.dtype,
        generator=state.rng,
    )
    new_momenta = (
        c1.unsqueeze(-1) * state.momenta
        + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
    )
    state.set_constrained_momenta(new_momenta)
    return state


def nvt_langevin_init(
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    **_kwargs: Any,
) -> MDState:
    """Initialize an NVT state from input data for Langevin dynamics.

    Creates an initial state for NVT molecular dynamics by computing initial
    energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
    at the specified temperature.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: SimState containing positions, masses, cell, pbc, and other
            required state vars
        kT: Temperature in energy units for initializing momenta,
            either scalar or with shape [n_systems]

    Returns:
        MDState: Initialized state for NVT integration containing positions,
            momenta, forces, energy, and other required attributes

    Notes:
        The initial momenta are sampled from a Maxwell-Boltzmann distribution
        at the specified temperature. This provides a proper thermal initial
        state for the subsequent Langevin dynamics.
    """
    model_output = model(state)

    momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions,
            state.masses,
            state.system_idx,
            kT,
            state.rng,
        )
    return MDState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
    )


@dcite("10.1098/rspa.2016.0138")
def nvt_langevin_step(
    state: MDState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    gamma: float | torch.Tensor | None = None,
) -> MDState:
    r"""Perform one complete Langevin dynamics integration step using the BAOAB scheme.

    Implements the BAOAB splitting of the Langevin equation from
    Leimkuhler & Matthews (2013, 2016) [2]_. The Langevin SDE is:

    .. math::

        d\mathbf{q} &= M^{-1}\mathbf{p}\,dt \\
        d\mathbf{p} &= -\nabla U(\mathbf{q})\,dt
            - \gamma\,\mathbf{p}\,dt
            + \sigma M^{1/2}\,d\mathbf{W}

    where :math:`\sigma = \sqrt{2\gamma k_BT}` (fluctuation-dissipation relation).

    **BAOAB splitting** (B = kick, A = drift, O = Ornstein-Uhlenbeck):

    .. math::

        \text{B:}\quad \mathbf{p} &\leftarrow \mathbf{p}
            + \tfrac{\Delta t}{2}\,\mathbf{F}(\mathbf{q}) \\
        \text{A:}\quad \mathbf{q} &\leftarrow \mathbf{q}
            + \tfrac{\Delta t}{2}\,M^{-1}\mathbf{p} \\
        \text{O:}\quad \mathbf{p} &\leftarrow
            c_1\,\mathbf{p} + c_2\,M^{1/2}\mathbf{R},
            \quad \mathbf{R}\sim\mathcal{N}(0,I) \\
        \text{A:}\quad \mathbf{q} &\leftarrow \mathbf{q}
            + \tfrac{\Delta t}{2}\,M^{-1}\mathbf{p} \\
        \text{B:}\quad \mathbf{p} &\leftarrow \mathbf{p}
            + \tfrac{\Delta t}{2}\,\mathbf{F}(\mathbf{q})

    with :math:`c_1 = e^{-\gamma\Delta t}` and
    :math:`c_2 = \sqrt{k_BT\,(1-c_1^2)}`.

    **Variable mapping (equation -> code):**

    ============================================  ============================
    Equation symbol                               Code variable
    ============================================  ============================
    :math:`\mathbf{q}`  (positions)               ``state.positions``
    :math:`\mathbf{p}`  (momenta)                 ``state.momenta``
    :math:`M`           (mass matrix)             ``state.masses``
    :math:`\mathbf{F}`  (forces)                  ``state.forces``
    :math:`\gamma`      (friction coefficient)    ``gamma``
    :math:`k_BT`        (thermal energy)          ``kT``
    :math:`\Delta t`    (timestep)                ``dt``
    :math:`c_1`                                   ``c1`` in ``_ou_step``
    :math:`c_2`                                   ``c2`` in ``_ou_step``
    ============================================  ============================

    Args:
        state: Current system state containing positions, momenta, forces
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        dt: Integration timestep, either scalar or shape [n_systems]
        kT: Target temperature in energy units, either scalar or
            with shape [n_systems]
        gamma: Friction coefficient for Langevin thermostat,
            either scalar or with shape [n_systems]. Defaults to 1/(100*dt).

    Returns:
        MDState: Updated state after one complete Langevin step with new positions,
            momenta, forces, and energy

    References:
        .. [2] Leimkuhler B, Matthews C. "Efficient molecular dynamics using geodesic
           integration and solvent-solute splitting." Proc. R. Soc. A 472: 20160138
           (2016). Original BAOAB analysis in: Leimkuhler B, Matthews C. "Rational
           construction of stochastic numerical methods for molecular sampling."
           Appl. Math. Res. Express 2013(1), 34-56 (2013).
    """
    device, dtype = model.device, model.dtype

    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    gamma_val = (1 / (100 * dt_tensor)) if gamma is None else gamma
    gamma_tensor = torch.as_tensor(gamma_val, device=device, dtype=dtype)

    state = momentum_step(state, dt_tensor / 2)
    state = position_step(state, dt_tensor / 2)
    state = _ou_step(state, dt_tensor, kT_tensor, gamma_tensor)
    state = position_step(state, dt_tensor / 2)

    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]

    return momentum_step(state, dt_tensor / 2)


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
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
    chain_length: int = 3,
    chain_steps: int = 3,
    sy_steps: int = 3,
    **kwargs: Any,
) -> NVTNoseHooverState:
    """Initialize the NVT Nose-Hoover state.

    This function sets up integration of an NVT system using a Nose-Hoover chain
    thermostat. The Nose-Hoover chain provides deterministic temperature control by
    coupling the system to a chain of thermostats. The integration scheme is
    time-reversible and conserves an extended energy quantity.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        state: Initial system state as SimState
        model: Neural network model that computes energies and forces
        kT: Target temperature in energy units
        dt: Integration timestep
        tau: Thermostat relaxation time (defaults to 100*dt)
        chain_length: Number of thermostats in Nose-Hoover chain (default: 3)
        chain_steps: Number of chain integration substeps (default: 3)
        sy_steps: Number of Suzuki-Yoshida steps - must be 1, 3, 5, or 7 (default: 3)
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
    dt_tensor = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    kT_tensor = torch.as_tensor(kT, device=state.device, dtype=state.dtype)
    tau_tensor = torch.as_tensor(
        10.0 * dt_tensor if tau is None else tau, device=state.device, dtype=state.dtype
    )

    # Create thermostat functions
    chain_fns = construct_nose_hoover_chain(
        dt_tensor, chain_length, chain_steps, sy_steps, tau_tensor
    )

    atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

    model_output = model(state)
    momenta = kwargs.get("momenta")
    if momenta is None:
        momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions, state.masses, state.system_idx, kT_tensor, state.rng
        )

    # Calculate initial kinetic energy per system
    KE = ts.calc_kinetic_energy(
        masses=state.masses, momenta=momenta, system_idx=state.system_idx
    )

    # Calculate degrees of freedom per system (subtract 3 for COM motion,
    # matching LAMMPS compute_temp which uses dof = 3N - 3)
    dof_per_system = state.get_number_of_degrees_of_freedom() - 3

    # Initialize state
    return NVTNoseHooverState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        atomic_numbers=atomic_numbers,
        chain=chain_fns.initialize(dof_per_system, KE, kT_tensor),
        _chain_fns=chain_fns,
    )


@dcite("10.1080/00268979600100761")
def nvt_nose_hoover_step(
    state: NVTNoseHooverState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
) -> NVTNoseHooverState:
    r"""Perform one complete Nose-Hoover chain (NHC) integration step.

    Implements the NHC thermostat from Martyna et al. (1996) [3]_ with
    Suzuki-Yoshida integration of the chain variables.

    **Equations of motion** (Martyna et al. 1996, Eqs. 13-18):

    .. math::

        \dot{\mathbf{r}}_i &= \mathbf{p}_i / m_i \\
        \dot{\mathbf{p}}_i &= \mathbf{F}_i
            - \frac{p_{\xi_1}}{Q_1}\,\mathbf{p}_i \\
        \dot{\xi}_j &= p_{\xi_j} / Q_j \\
        \dot{p}_{\xi_1} &= \bigl(2K - N_f k_BT\bigr)
            - \frac{p_{\xi_2}}{Q_2}\,p_{\xi_1} \\
        \dot{p}_{\xi_j} &= \left(\frac{p_{\xi_{j-1}}^2}{Q_{j-1}}
            - k_BT\right) - \frac{p_{\xi_{j+1}}}{Q_{j+1}}\,p_{\xi_j}
            \quad (j = 2,\ldots,M{-}1) \\
        \dot{p}_{\xi_M} &= \frac{p_{\xi_{M-1}}^2}{Q_{M-1}} - k_BT

    where :math:`K = \sum_i p_i^2/(2m_i)` is the kinetic energy,
    :math:`N_f = 3N - 3` the degrees of freedom, and :math:`Q_j = k_BT\tau^2`
    (with :math:`Q_1 = N_f k_BT\tau^2`) are the chain masses.

    **Symmetric propagator** (Trotter factorization):

    .. math::

        e^{i\mathcal{L}\Delta t} = e^{i\mathcal{L}_{\text{NHC}}\Delta t/2}
        \;e^{i\mathcal{L}_{\text{VV}}\Delta t}
        \;e^{i\mathcal{L}_{\text{NHC}}\Delta t/2}

    where :math:`i\mathcal{L}_{\text{VV}}` is the velocity Verlet propagator
    and :math:`i\mathcal{L}_{\text{NHC}}` integrates the chain with
    :math:`n_c \times n_{\text{sy}}` sub-steps (Suzuki-Yoshida decomposition).

    **Variable mapping (equation -> code):**

    ============================================  ============================
    Equation symbol                               Code variable
    ============================================  ============================
    :math:`\mathbf{r}_i`  (positions)             ``state.positions``
    :math:`\mathbf{p}_i`  (momenta)               ``state.momenta``
    :math:`m_i`           (masses)                ``state.masses``
    :math:`\mathbf{F}_i`  (forces)                ``state.forces``
    :math:`\xi_j`         (chain positions)       ``state.chain.positions``
    :math:`p_{\xi_j}`     (chain momenta)         ``state.chain.momenta``
    :math:`Q_j`           (chain masses)           ``state.chain.masses``
    :math:`K`             (kinetic energy)         ``state.chain.kinetic_energy``
    :math:`N_f`           (degrees of freedom)    ``state.chain.degrees_of_freedom``
    :math:`\tau`          (relaxation time)        ``state.chain.tau``
    :math:`k_BT`          (thermal energy)        ``kT``
    :math:`\Delta t`      (timestep)              ``dt``
    :math:`M`             (chain length)          ``chain_length``
    :math:`n_c`           (chain substeps)        ``chain_steps``
    :math:`n_{\text{sy}}` (SY steps)              ``sy_steps``
    ============================================  ============================

    Args:
        state: Current system state containing positions, momenta, forces, and chain
        model: Neural network model that computes energies and forces
        dt: Integration timestep
        kT: Target temperature in energy units

    Returns:
        Updated state after one complete Nose-Hoover step

    References:
        .. [3] Martyna, G. J., Tuckerman, M. E., Tobias, D. J. & Klein, M. L.
           "Explicit reversible integrators for extended systems dynamics."
           Molecular Physics 87(5), 1117-1157 (1996).
    """
    # Get chain functions from state
    chain_fns = state._chain_fns  # noqa: SLF001
    chain = state.chain

    dt = torch.as_tensor(dt, device=state.device, dtype=state.dtype)
    kT = torch.as_tensor(kT, device=state.device, dtype=state.dtype)

    # Update chain masses based on target temperature
    chain = chain_fns.update_mass(chain, kT)

    # First half-step of chain evolution
    momenta, chain = chain_fns.half_step(state.momenta, chain, kT, state.system_idx)
    state.set_constrained_momenta(momenta)

    # Full velocity Verlet step
    state = velocity_verlet_step(state=state, dt=dt, model=model)

    # Update chain kinetic energy per system
    KE = ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )
    chain.kinetic_energy = KE

    # Second half-step of chain evolution
    momenta, chain = chain_fns.half_step(state.momenta, chain, kT, state.system_idx)
    state.set_constrained_momenta(momenta)
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

    # Get system degrees of freedom per system (3N - 3 for COM correction)
    dof = state.get_number_of_degrees_of_freedom()

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

    def get_number_of_degrees_of_freedom(self) -> torch.Tensor:
        """Calculate the degrees of freedom per system."""
        # Subtract 3 for center of mass motion
        return super().get_number_of_degrees_of_freedom() - 3


def _vrescale_update[T: MDState](
    state: T,
    tau: float | torch.Tensor,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
) -> T:
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
    current_kT = state.calc_kT()

    # Calculate degrees of freedom per system
    dof = state.get_number_of_degrees_of_freedom()

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
    rng = state.rng
    r1 = torch.randn(n_systems, device=device, dtype=dtype, generator=rng)
    # Sample Gamma((dof - 1)/2, 1/2) via _standard_gamma so we can seed it
    r2 = torch._standard_gamma((dof - 1) / 2, generator=rng) * 2  # noqa: SLF001

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
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    **_kwargs: Any,
) -> NVTVRescaleState:
    """Initialize an NVT state from input data for velocity rescaling dynamics.

    Creates an initial state for NVT molecular dynamics using the canonical
    sampling through velocity rescaling (CSVR) thermostat. This thermostat
    samples from the canonical ensemble by rescaling velocities with an
    appropriately chosen random factor.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: SimState containing positions, masses, cell, pbc, and other
            required state vars
        kT: Temperature in energy units for initializing momenta,
            either scalar or with shape [n_systems]

    Returns:
        MDState: Initialized state for NVT integration containing positions,
            momenta, forces, energy, and other required attributes

    Notes:
        The initial momenta are sampled from a Maxwell-Boltzmann distribution
        at the specified temperature. The V-Rescale thermostat provides proper
        canonical sampling through stochastic velocity rescaling.
    """
    model_output = model(state)

    momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions,
            state.masses,
            state.system_idx,
            kT,
            state.rng,
        )

    return NVTVRescaleState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
    )


@dcite("10.1063/1.2408420")
def nvt_vrescale_step(
    model: ModelInterface,
    state: NVTVRescaleState,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
) -> NVTVRescaleState:
    r"""Perform one complete V-Rescale (CSVR) dynamics integration step.

    Implements canonical sampling through velocity rescaling from
    Bussi, Donadio & Parrinello (2007) [1]_.

    **Stochastic differential equation** for kinetic energy (Eq. 7 of [1]_):

    .. math::

        dK = \frac{\bar{K} - K}{\tau}\,dt
            + 2\sqrt{\frac{K\bar{K}}{N_f\tau}}\,dW

    where :math:`\bar{K} = N_f k_BT/2` is the target kinetic energy.

    **Discrete rescaling factor** :math:`\alpha^2 = K'/K` (Eq. 22 of [1]_):

    .. math::

        \alpha^2 = e^{-\Delta t/\tau}
            + \frac{\bar{K}}{N_f K}\bigl(1-e^{-\Delta t/\tau}\bigr)
              \Bigl(R_1^2 + \sum_{i=2}^{N_f} R_i^2\Bigr)
            + 2\,e^{-\Delta t/(2\tau)}
              \sqrt{\frac{\bar{K}}{N_f K}
              \bigl(1-e^{-\Delta t/\tau}\bigr)}\;R_1

    where :math:`R_1 \sim \mathcal{N}(0,1)` and
    :math:`\sum_{i=2}^{N_f} R_i^2 \sim \text{Gamma}\bigl((N_f-1)/2,\,2\bigr)`.
    Momenta are then rescaled as :math:`\mathbf{p} \leftarrow \alpha\,\mathbf{p}`.

    **Variable mapping (equation -> code):**

    ============================================  ============================
    Equation symbol                               Code variable
    ============================================  ============================
    :math:`K`             (kinetic energy)        ``KE_old``
    :math:`\bar{K}`       (target KE)            ``KE_new``
    :math:`N_f`           (degrees of freedom)   ``dof``
    :math:`\tau`          (relaxation time)       ``tau``
    :math:`k_BT`          (thermal energy)       ``kT``
    :math:`\Delta t`      (timestep)             ``dt``
    :math:`e^{-\Delta t/\tau}`                   ``c1``
    :math:`(1-c_1)\bar{K}/(N_f K)`              ``c2``
    :math:`R_1`                                  ``r1``
    :math:`\sum_{i=2}^{N_f} R_i^2`              ``r2``
    :math:`\alpha^2`      (scale factor)         ``scale``
    :math:`\alpha`        (velocity rescaling)   ``lam``
    ============================================  ============================

    Args:
        model: Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        state: Current system state containing positions, momenta, forces
        dt: Integration timestep, either scalar or shape [n_systems]
        kT: Target temperature in energy units, either scalar or
            with shape [n_systems]
        tau: Thermostat relaxation time controlling the coupling strength,
            either scalar or with shape [n_systems]. Defaults to 100*dt.

    Returns:
        MDState: Updated state after one complete V-Rescale step with new positions,
            momenta, forces, and energy

    References:
        .. [1] Bussi G, Donadio D, Parrinello M. "Canonical sampling through velocity
           rescaling." J. Chem. Phys. 126(1), 014101 (2007).
    """
    device, dtype = model.device, model.dtype

    tau = torch.as_tensor(10 * dt if tau is None else tau, device=device, dtype=dtype)
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)

    # Apply V-Rescale rescaling
    state = _vrescale_update(state, tau, kT, dt)

    # Perform velocity Verlet step
    return velocity_verlet_step(state=state, dt=dt, model=model)
