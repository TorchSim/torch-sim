"""Implementations of NPT integrators."""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

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
)
from torch_sim.integrators.nvt import _vrescale_update
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.units import MetalUnits


logger = logging.getLogger(__name__)


def _randn_for_state(state: MDState, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """Sample standard normal noise on the state's device/dtype using state RNG."""
    return torch.randn(shape, device=state.device, dtype=state.dtype, generator=state.rng)


@dataclass(kw_only=True)
class NPTState(MDState):
    """State information for an NPT system.

    This class extends MDState with the stress tensor needed for
    constant-pressure simulations. Integrator-specific NPT states
    (e.g., NPTLangevinAnisotropicState, NPTNoseHooverIsotropicState) inherit from this
    class and add their own auxiliary variables.

    Attributes:
        stress (torch.Tensor): Stress tensor [n_systems, n_dim, n_dim]
    """

    # System state variables
    stress: torch.Tensor

    _system_attributes = MDState._system_attributes | {  # noqa: SLF001
        "stress",
    }


@dataclass(kw_only=True)
class NPTLangevinAnisotropicState(NPTState):
    """State for NPT Langevin dynamics with independent per-dimension cell lengths.

    Each spatial dimension has its own logarithmic strain coordinate
    εi = ln(Li/Li0), driven by the corresponding diagonal pressure
    component P_ii. This is analogous to LAMMPS ``fix press/langevin``
    with ``couple none``.

    With three identical target pressures the sum of forces equals the
    isotropic strain force, so the isotropic limit is recovered.

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
        reference_cell (torch.Tensor): Original cell [n_systems, d, d]
        cell_positions (torch.Tensor): Per-dimension strain εi [n_systems, 3]
        cell_velocities (torch.Tensor): dεi/dt [n_systems, 3]
        cell_masses (torch.Tensor): Mass for strain DOFs [n_systems]
        alpha (torch.Tensor): Particle friction [n_systems]
        cell_alpha (torch.Tensor): Cell friction [n_systems]
        b_tau (torch.Tensor): Barostat time constant [n_systems]

    Properties:
        momenta (torch.Tensor): Particle momenta calculated as velocities*masses
            with shape [n_particles, n_dimensions]
        current_cell (torch.Tensor): Cell reconstructed from strain and reference_cell
        volume (torch.Tensor): Current volume from cell determinant
        n_systems (int): Number of independent systems in the batch
        device (torch.device): Device on which tensors are stored
        dtype (torch.dtype): Data type of tensors
    """

    alpha: torch.Tensor
    cell_alpha: torch.Tensor
    b_tau: torch.Tensor

    # Cell variables
    reference_cell: torch.Tensor
    cell_positions: torch.Tensor  # (n_systems, 3) per-dimension strain
    cell_velocities: torch.Tensor  # (n_systems, 3)
    cell_masses: torch.Tensor

    _system_attributes = NPTState._system_attributes | {  # noqa: SLF001
        "cell_positions",
        "cell_velocities",
        "cell_masses",
        "reference_cell",
        "alpha",
        "cell_alpha",
        "b_tau",
    }

    @property
    def current_cell(self) -> torch.Tensor:
        """Compute cell from per-dimension strain: cell[i,:] = exp(εi) · ref[i,:]."""
        scale = torch.exp(self.cell_positions)  # (n_systems, 3)
        return scale.unsqueeze(-1) * self.reference_cell

    @property
    def volume(self) -> torch.Tensor:
        """Current volume from cell determinant."""
        return torch.linalg.det(self.cell)


def _npt_langevin_particle_beta(
    state: "NPTLangevinAnisotropicState | NPTLangevinIsotropicState",
    kT: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Calculate random noise term for particle Langevin dynamics.

    This function generates the stochastic force term for the Langevin thermostat
    according to the fluctuation-dissipation theorem, ensuring proper thermal
    sampling at the target temperature. Only particle degrees of freedom are
    involved (not cell DOFs), so it works for both isotropic and anisotropic
    cell dynamics.

    Args:
        state (NPTLangevinAnisotropicState | NPTLangevinIsotropicState): Current NPT state
        kT (torch.Tensor): Temperature in energy units, either scalar or
            shape [n_systems]
        dt (torch.Tensor): Integration timestep, either scalar or shape [n_systems]

    Returns:
        torch.Tensor: Random noise term for force calculation [n_particles, n_dim]
    """
    # Generate system-specific noise with correct shape
    noise = _randn_for_state(state, state.momenta.shape)

    # Calculate the thermal noise amplitude by system
    batch_kT = kT
    if kT.ndim == 0:
        batch_kT = kT.expand(state.n_systems)

    # Map system kT to atoms
    atom_kT = batch_kT[state.system_idx]
    atom_alpha = state.alpha[state.system_idx]

    atom_dt = dt
    if dt.ndim == 0:
        atom_dt = dt.expand(state.n_systems)[state.system_idx]

    # Calculate the prefactor for each atom
    # The standard deviation should be sqrt(2*alpha*kB*T*dt)
    prefactor = torch.sqrt(2 * atom_alpha * atom_kT * atom_dt)

    return prefactor.unsqueeze(-1) * noise


def _npt_langevin_anisotropic_cell_beta(
    state: NPTLangevinAnisotropicState,
    kT: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Generate per-dimension noise for cell length fluctuations.

    Args:
        state: Current NPT state
        kT: Temperature in energy units (scalar or [n_systems])
        dt: Timestep (scalar or [n_systems])

    Returns:
        torch.Tensor: Noise [n_systems, 3]
    """
    noise = _randn_for_state(state, (state.n_systems, 3))
    batch_kT = kT if kT.ndim > 0 else kT.expand(state.n_systems)
    dt_expanded = dt if dt.ndim > 0 else dt.expand(state.n_systems)
    scaling = torch.sqrt(2.0 * state.cell_alpha * batch_kT * dt_expanded)
    return scaling.unsqueeze(-1) * noise


def _npt_langevin_anisotropic_cell_position_step(
    state: NPTLangevinAnisotropicState,
    dt: torch.Tensor,
    strain_force: torch.Tensor,
    cell_beta: torch.Tensor,
) -> NPTLangevinAnisotropicState:
    """GJF position step for per-dimension strain εi.

    Args:
        state: Current NPT state
        dt: Timestep
        strain_force: F_εi [n_systems, 3]
        cell_beta: Noise [n_systems, 3]

    Returns:
        Updated state with new cell_positions (strain)
    """
    Q_2 = (2 * state.cell_masses).unsqueeze(-1)  # (n_systems, 1)
    dt_expanded = dt if dt.ndim > 0 else dt.expand(state.n_systems)
    dt_3 = dt_expanded.unsqueeze(-1) if dt_expanded.ndim > 0 else dt_expanded

    cell_b = 1 / (1 + (state.cell_alpha.unsqueeze(-1) * dt_3) / Q_2)

    c_1 = cell_b * dt_3 * state.cell_velocities
    c_2 = cell_b * dt_3 * dt_3 * strain_force / Q_2
    c_3 = cell_b * dt_3 * cell_beta / Q_2

    state.cell_positions = state.cell_positions + c_1 + c_2 + c_3
    return state


def _npt_langevin_anisotropic_cell_velocity_step(
    state: NPTLangevinAnisotropicState,
    F_eps_n: torch.Tensor,
    dt: torch.Tensor,
    strain_force: torch.Tensor,
    cell_beta: torch.Tensor,
) -> NPTLangevinAnisotropicState:
    """GJF velocity step for per-dimension strain εi.

    Args:
        state: Current NPT state
        F_eps_n: Initial strain force [n_systems, 3]
        dt: Timestep
        strain_force: Final strain force [n_systems, 3]
        cell_beta: Noise (SAME as in position step) [n_systems, 3]

    Returns:
        Updated state with new cell_velocities
    """
    dt_expanded = dt if dt.ndim > 0 else dt.expand(state.n_systems)
    dt_3 = dt_expanded.unsqueeze(-1) if dt_expanded.ndim > 0 else dt_expanded

    Q = state.cell_masses.unsqueeze(-1)  # (n_systems, 1)
    alpha_c = state.cell_alpha.unsqueeze(-1)  # (n_systems, 1)
    a = (1 - (alpha_c * dt_3) / (2 * Q)) / (1 + (alpha_c * dt_3) / (2 * Q))
    b = 1 / (1 + (alpha_c * dt_3) / (2 * Q))

    c_1 = a * state.cell_velocities
    c_2 = dt_3 * ((a * F_eps_n) + strain_force) / (2 * Q)
    c_3 = b * cell_beta / Q

    state.cell_velocities = c_1 + c_2 + c_3
    return state


def _npt_langevin_anisotropic_position_step(
    state: NPTLangevinAnisotropicState,
    eps_old: torch.Tensor,
    dt: torch.Tensor,
    particle_beta: torch.Tensor,
) -> NPTLangevinAnisotropicState:
    """Update particle positions with per-dimension strain scaling.

    Each component of position is scaled by exp(εi_new - εi_old).

    Args:
        state: Current state (cell_positions already updated)
        eps_old: Previous strain [n_systems, 3]
        dt: Timestep
        particle_beta: Noise [n_particles, n_dim]

    Returns:
        Updated state with new positions
    """
    M_2 = 2 * state.masses.unsqueeze(-1)  # (n_atoms, 1)

    # Per-dimension scale factor
    scale = torch.exp(state.cell_positions - eps_old)  # (n_systems, 3)
    scale_atoms = scale[state.system_idx]  # (n_atoms, 3)

    # Damping factor
    alpha_atoms = state.alpha[state.system_idx]
    dt_atoms = dt
    if dt.ndim > 0:
        dt_atoms = dt[state.system_idx]

    b = 1 / (1 + ((alpha_atoms * dt_atoms) / (2 * state.masses)))

    # Scale each position component independently
    c_1 = scale_atoms * state.positions  # (n_atoms, 3)

    # Time step factor: 2·s/(s+1) per dimension
    c_2 = (2 * scale_atoms / (scale_atoms + 1)) * b.unsqueeze(-1) * dt_atoms.unsqueeze(-1)

    c_3 = (
        state.velocities
        + dt_atoms.unsqueeze(-1) * state.forces / M_2
        + particle_beta / M_2
    )

    state.set_constrained_positions(c_1 + c_2 * c_3)
    return state


def _npt_langevin_particle_velocity_step(
    state: "NPTLangevinAnisotropicState | NPTLangevinIsotropicState",
    forces: torch.Tensor,
    dt: torch.Tensor,
    particle_beta: torch.Tensor,
) -> "NPTLangevinAnisotropicState | NPTLangevinIsotropicState":
    """Update the particle velocities in NPT dynamics.

    This function updates particle velocities using a Langevin-type integrator,
    accounting for both deterministic forces and pre-generated thermal noise.
    Only particle degrees of freedom are involved (not cell DOFs), so it works
    for both isotropic and anisotropic cell dynamics.

    Args:
        state (NPTLangevinAnisotropicState | NPTLangevinIsotropicState): Current NPT state
        forces: Forces on particles (from before position update)
        dt: Integration timestep, either scalar or with shape [n_systems]
        particle_beta (torch.Tensor): Pre-generated GJF noise term β for particle
            dynamics. Must be the SAME realization used in the position step.
            Shape [n_particles, n_dim]

    Returns:
        Updated state with new velocities (same type as input).
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

    # GJF noise term: b * β / m
    c_3 = b * particle_beta / state.masses.unsqueeze(-1)

    # Update momenta (velocities * masses) with all contributions
    new_velocities = c_1 + c_2 + c_3
    # Apply constraints.
    state.set_constrained_momenta(new_velocities * state.masses.unsqueeze(-1))
    return state


def _npt_langevin_anisotropic_compute_cell_force(
    state: NPTLangevinAnisotropicState,
    external_pressure: torch.Tensor,
    kT: torch.Tensor,
) -> torch.Tensor:
    """Compute per-dimension force on the strain coordinates.

    F_εi = V · (P_ii - P_ext_i)

    where P_ii = -σ_ii + N·kT/V is the ii diagonal pressure component.
    The force is in energy units (eV).

    Args:
        state: Current NPT state
        external_pressure: Target pressure per dimension [3] or [n_systems, 3]
        kT: Temperature in energy units (scalar or [n_systems])

    Returns:
        torch.Tensor: Force per dimension [n_systems, 3]
    """
    volumes = state.volume  # (n_systems,)

    # Diagonal stress components \sigma_ii
    stress_diag = torch.diagonal(state.stress, dim1=-2, dim2=-1)  # (n_systems, 3)

    # P_ii = -\sigma_ii (virial part)
    P_virial_diag = -stress_diag  # (n_systems, 3)

    # Kinetic contribution per dimension: N·kT/V (target temperature)
    batch_kT = kT if kT.ndim > 0 else kT.expand(state.n_systems)
    n_atoms = state.n_atoms_per_system.to(dtype=state.dtype)
    kinetic_pressure = (n_atoms * batch_kT / volumes).unsqueeze(-1)  # (n_systems, 1)

    P_diag = P_virial_diag + kinetic_pressure  # (n_systems, 3)

    # F_εi = V · (P_ii - P_ext_i)
    return volumes.unsqueeze(-1) * (P_diag - external_pressure)


def npt_langevin_anisotropic_init(
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    cell_alpha: float | torch.Tensor | None = None,
    b_tau: float | torch.Tensor | None = None,
    **_kwargs: Any,
) -> NPTLangevinAnisotropicState:
    """Initialize NPT Langevin state with independent per-dimension cell lengths.

    Each spatial dimension gets its own strain DOF εi = ln(Li/Li0),
    driven by the corresponding diagonal pressure component.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        state: SimState containing positions, masses, cell, pbc
        model: Model computing energy, forces, stress
        kT: Target temperature in energy units
        dt: Integration timestep
        alpha: Particle friction. Defaults to 1/(5·dt).
        cell_alpha: Cell friction. Defaults to 1/(30·dt).
        b_tau: Barostat time constant. Defaults to 300·dt.

    Returns:
        NPTLangevinAnisotropicState with εi = 0 for all dimensions
    """
    device, dtype = model.device, model.dtype

    if alpha is None:
        alpha = 1.0 / (5 * dt)
    if cell_alpha is None:
        cell_alpha = 1.0 / (30 * dt)
    if b_tau is None:
        b_tau = 300 * dt

    alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    cell_alpha = torch.as_tensor(cell_alpha, device=device, dtype=dtype)
    b_tau = torch.as_tensor(b_tau, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)
    dt = torch.as_tensor(dt, device=device, dtype=dtype)

    if alpha.ndim == 0:
        alpha = alpha.expand(state.n_systems)
    if cell_alpha.ndim == 0:
        cell_alpha = cell_alpha.expand(state.n_systems)
    if b_tau.ndim == 0:
        b_tau = b_tau.expand(state.n_systems)

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

    reference_cell = state.cell.clone()
    dim = state.positions.shape[1]

    # εi = 0 at initialization (V = V₀)
    cell_positions = torch.zeros(state.n_systems, dim, device=device, dtype=dtype)
    cell_velocities = torch.zeros(state.n_systems, dim, device=device, dtype=dtype)

    batch_kT = kT.expand(state.n_systems) if kT.ndim == 0 else kT
    n_atoms_per_system = torch.bincount(state.system_idx)
    cell_masses = (n_atoms_per_system + 1) * batch_kT * b_tau * b_tau

    if state.constraints:
        msg = (
            "Constraints are present in the system. "
            "Make sure they are compatible with NPT Langevin dynamics. "
            "We recommend not using constraints with NPT dynamics for now."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    return NPTLangevinAnisotropicState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        stress=model_output["stress"],
        alpha=alpha,
        b_tau=b_tau,
        reference_cell=reference_cell,
        cell_positions=cell_positions,
        cell_velocities=cell_velocities,
        cell_masses=cell_masses,
        cell_alpha=cell_alpha,
    )


@dcite("10.1063/1.4901303")
def npt_langevin_anisotropic_step(
    state: NPTLangevinAnisotropicState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
) -> NPTLangevinAnisotropicState:
    r"""Perform one NPT Langevin step with independent per-dimension cell lengths.

    Implements constant-pressure Langevin dynamics based on Gronbech-Jensen &
    Farago (2014) [4]_ and the LAMMPS ``fix press/langevin`` scheme [5]_.

    Each spatial dimension *i* has its own logarithmic strain
    :math:`\varepsilon_i = \ln(L_i/L_{i,0})` driven by the diagonal
    pressure component :math:`P_{ii}`.

    **Per-dimension strain force:**

    .. math::

        F_{\varepsilon_i} = V \cdot (P_{ii} - P_{\text{ext},i})

    where :math:`P_{ii} = -\sigma_{ii} + N k_B T / V`.

    With three identical target pressures the sum
    :math:`\sum_i F_{\varepsilon_i}` equals the isotropic strain force.

    **Cell reconstruction:**

    .. math::

        \mathbf{h}_i = e^{\varepsilon_i}\,\mathbf{h}_{i,0}

    **Particle scaling (per component):**

    .. math::

        r_{k,i} \to e^{\varepsilon_i^{n+1} - \varepsilon_i^n}\, r_{k,i}

    Args:
        state: Current NPT state
        model: Model computing energy, forces, stress
        dt: Integration timestep
        kT: Target temperature in energy units
        external_pressure: Target pressure — scalar (same for all dims),
            shape [3] (per-dimension), or [n_systems, 3]

    Returns:
        NPTLangevinAnisotropicState: Updated state

    References:
        .. [4] Gronbech-Jensen, N. & Farago, O. "Constant pressure and temperature
           discrete-time Langevin molecular dynamics." J. Chem. Phys. 141(19) (2014).
        .. [5] LAMMPS fix press/langevin:
           https://docs.lammps.org/fix_press_langevin.html
    """
    device, dtype = model.device, model.dtype

    state.alpha = torch.as_tensor(state.alpha, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    state.cell_alpha = torch.as_tensor(state.cell_alpha, device=device, dtype=dtype)
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    external_pressure_tensor = torch.as_tensor(
        external_pressure, device=device, dtype=dtype
    )

    # Broadcast external_pressure to (n_systems, 3)
    if external_pressure_tensor.ndim == 0:
        external_pressure_tensor = external_pressure_tensor.expand(state.n_systems, 3)
    elif external_pressure_tensor.ndim == 1 and external_pressure_tensor.shape[0] == 3:
        external_pressure_tensor = external_pressure_tensor.unsqueeze(0).expand(
            state.n_systems, 3
        )

    batch_kT = kT_tensor.expand(state.n_systems) if kT_tensor.ndim == 0 else kT_tensor

    # Update barostat mass
    n_atoms_per_system = torch.bincount(state.system_idx)
    state.cell_masses = (n_atoms_per_system + 1) * batch_kT * torch.square(state.b_tau)

    # Store initial values
    forces = state.forces
    eps_old = state.cell_positions.clone()

    F_eps_n = _npt_langevin_anisotropic_compute_cell_force(
        state=state,
        external_pressure=external_pressure_tensor,
        kT=kT_tensor,
    )

    # Generate GJF noise ONCE
    cell_beta = _npt_langevin_anisotropic_cell_beta(state, kT_tensor, dt_tensor)
    particle_beta = _npt_langevin_particle_beta(state, kT_tensor, dt_tensor)

    # Step 1: Update per-dimension strain
    state = _npt_langevin_anisotropic_cell_position_step(
        state, dt_tensor, F_eps_n, cell_beta
    )

    # Reconstruct cell from updated strain
    state.cell = state.current_cell

    # Step 2: Update particle positions
    state = _npt_langevin_anisotropic_position_step(
        state, eps_old, dt_tensor, particle_beta
    )

    # Recompute model output
    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]
    state.stress = model_output["stress"]

    # Updated strain force
    F_eps_new = _npt_langevin_anisotropic_compute_cell_force(
        state=state,
        external_pressure=external_pressure_tensor,
        kT=kT_tensor,
    )

    # Step 3: Update strain velocities (uses SAME cell_beta)
    state = _npt_langevin_anisotropic_cell_velocity_step(
        state, F_eps_n, dt_tensor, F_eps_new, cell_beta
    )

    # Step 4: Update particle velocities (uses SAME particle_beta)
    return cast(
        "NPTLangevinAnisotropicState",
        _npt_langevin_particle_velocity_step(state, forces, dt_tensor, particle_beta),
    )


# =============================================================================
# NPT Langevin Strain integrator — isotropic logarithmic strain coordinate
# =============================================================================


@dataclass(kw_only=True)
class NPTLangevinIsotropicState(NPTState):
    """State for NPT Langevin dynamics using logarithmic strain coordinate.

    The cell degree of freedom is the isotropic logarithmic strain
    ε = (1/d)·ln(V/V₀), which is dimensionless. This guarantees V > 0
    and gives the conjugate force F_ε = d·V·(P_avg - P_ext) in energy units,
    providing numerically well-scaled dynamics.

    Attributes:
        reference_cell (torch.Tensor): Original cell [n_systems, d, d]
        cell_positions (torch.Tensor): Strain ε = (1/d)·ln(V/V₀) [n_systems]
        cell_velocities (torch.Tensor): dε/dt [n_systems]
        cell_masses (torch.Tensor): Mass for strain DOF [n_systems]
        alpha (torch.Tensor): Particle friction [n_systems]
        cell_alpha (torch.Tensor): Cell friction [n_systems]
        b_tau (torch.Tensor): Barostat time constant [n_systems]
    """

    alpha: torch.Tensor
    cell_alpha: torch.Tensor
    b_tau: torch.Tensor

    reference_cell: torch.Tensor
    cell_positions: torch.Tensor  # strain ε (dimensionless)
    cell_velocities: torch.Tensor  # dε/dt
    cell_masses: torch.Tensor

    _system_attributes = NPTState._system_attributes | {  # noqa: SLF001
        "cell_positions",
        "cell_velocities",
        "cell_masses",
        "reference_cell",
        "alpha",
        "cell_alpha",
        "b_tau",
    }

    @property
    def current_cell(self) -> torch.Tensor:
        """Compute cell from strain: cell = exp(ε) · reference_cell."""
        scale = torch.exp(self.cell_positions)  # exp(ε), shape (n_systems,)
        return scale.unsqueeze(-1).unsqueeze(-1) * self.reference_cell

    @property
    def volume(self) -> torch.Tensor:
        """Current volume V = V₀ · exp(d·ε)."""
        dim = self.positions.shape[1]
        V_0 = torch.linalg.det(self.reference_cell)
        return V_0 * torch.exp(dim * self.cell_positions)


def _compute_isotropic_cell_force(
    state: NPTLangevinIsotropicState,
    external_pressure: float | torch.Tensor,
    kT: float | torch.Tensor,
) -> torch.Tensor:
    """Compute force on the strain coordinate ε.

    F_ε = d · V · (P_avg - P_ext)

    where P_avg = -(1/3)Tr(σ) + NkT/V and d·V is the Jacobian dV/dε.
    This force is in energy units (eV), making it numerically well-scaled.

    Args:
        state: Current strain-based NPT state
        external_pressure: Target pressure (scalar or [n_systems])
        kT: Temperature in energy units (scalar or [n_systems])

    Returns:
        torch.Tensor: Force on strain per system [n_systems]
    """
    external_pressure = torch.as_tensor(
        external_pressure, device=state.device, dtype=state.dtype
    )
    kT = torch.as_tensor(kT, device=state.device, dtype=state.dtype)

    dim = state.positions.shape[1]
    volumes = state.volume  # (n_systems,)

    # Isotropic virial pressure: P_virial = -(1/3)Tr(stress)
    stress_trace = torch.einsum("nii->n", state.stress)
    avg_virial_pressure = -stress_trace / 3  # (n_systems,)

    # Kinetic contribution: NkT/V
    batch_kT = kT if kT.ndim > 0 else kT.expand(state.n_systems)
    n_atoms = state.n_atoms_per_system.to(dtype=state.dtype)
    kinetic_pressure = n_atoms * batch_kT / volumes  # (n_systems,)

    if external_pressure.ndim >= 2:
        raise ValueError(
            f"External pressure tensor provided with shape {external_pressure.shape}. "
            "Only scalar or per-system external pressure is supported."
        )

    P_avg = avg_virial_pressure + kinetic_pressure
    # F_ε = d · V · (P_avg - P_ext)
    return dim * volumes * (P_avg - external_pressure)


def _npt_langevin_isotropic_cell_beta(
    state: NPTLangevinIsotropicState,
    kT: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Generate scalar random noise for isotropic strain fluctuations.

    Returns:
        torch.Tensor: Noise [n_systems]
    """
    noise = _randn_for_state(state, (state.n_systems,))
    batch_kT = kT if kT.ndim > 0 else kT.expand(state.n_systems)
    dt_expanded = dt if dt.ndim > 0 else dt.expand(state.n_systems)
    scaling = torch.sqrt(2.0 * state.cell_alpha * batch_kT * dt_expanded)
    return scaling * noise


def _npt_langevin_isotropic_cell_position_step(
    state: NPTLangevinIsotropicState,
    dt: torch.Tensor,
    strain_force: torch.Tensor,
    cell_beta: torch.Tensor,
) -> NPTLangevinIsotropicState:
    """GJF position step for the strain coordinate ε.

    ε_{n+1} = ε_n + b·dt·dε/dt + b·dt²·F_ε/(2Q) + b·dt·β/(2Q)

    Args:
        state: Current state
        dt: Timestep
        strain_force: F_ε [n_systems]
        cell_beta: Noise term β_c [n_systems]

    Returns:
        Updated state with new cell_positions (strain)
    """
    Q_2 = 2 * state.cell_masses
    dt_expanded = dt if dt.ndim > 0 else dt.expand(state.n_systems)

    cell_b = 1 / (1 + (state.cell_alpha * dt_expanded) / Q_2)

    c_1 = cell_b * dt_expanded * state.cell_velocities
    c_2 = cell_b * dt_expanded * dt_expanded * strain_force / Q_2
    c_3 = cell_b * dt_expanded * cell_beta / Q_2

    state.cell_positions = state.cell_positions + c_1 + c_2 + c_3
    return state


def _npt_langevin_isotropic_cell_velocity_step(
    state: NPTLangevinIsotropicState,
    F_eps_n: torch.Tensor,
    dt: torch.Tensor,
    strain_force: torch.Tensor,
    cell_beta: torch.Tensor,
) -> NPTLangevinIsotropicState:
    """GJF velocity step for the strain coordinate ε.

    dε/dt_{n+1} = a·dε/dt_n + dt/(2Q)·(a·F_ε^n + F_ε^{n+1}) + b·β/Q

    Args:
        state: Current state
        F_eps_n: Initial strain force [n_systems]
        dt: Timestep
        strain_force: Final strain force [n_systems]
        cell_beta: Noise term β_c (SAME as in position step) [n_systems]

    Returns:
        Updated state with new cell_velocities (dε/dt)
    """
    dt_expanded = dt if dt.ndim > 0 else dt.expand(state.n_systems)

    Q = state.cell_masses
    a = (1 - (state.cell_alpha * dt_expanded) / (2 * Q)) / (
        1 + (state.cell_alpha * dt_expanded) / (2 * Q)
    )
    b = 1 / (1 + (state.cell_alpha * dt_expanded) / (2 * Q))

    c_1 = a * state.cell_velocities
    c_2 = dt_expanded * ((a * F_eps_n) + strain_force) / (2 * Q)
    c_3 = b * cell_beta / Q

    state.cell_velocities = c_1 + c_2 + c_3
    return state


def _npt_langevin_isotropic_position_step(
    state: NPTLangevinIsotropicState,
    eps_old: torch.Tensor,
    dt: torch.Tensor,
    particle_beta: torch.Tensor,
) -> NPTLangevinIsotropicState:
    """Update particle positions accounting for strain change.

    Positions are scaled by exp(ε_new - ε_old) for the volume change,
    then the standard GJF position update is applied.

    Args:
        state: Current state (cell_positions already updated to ε_new)
        eps_old: Strain before the cell position step [n_systems]
        dt: Timestep
        particle_beta: Noise [n_particles, n_dim]

    Returns:
        Updated state with new positions
    """
    M_2 = 2 * state.masses.unsqueeze(-1)  # (n_atoms, 1)

    # Scale factor from strain change: L_new/L_old = exp(ε_new - ε_old)
    scale = torch.exp(state.cell_positions - eps_old)  # (n_systems,)
    scale_atoms = scale[state.system_idx]  # (n_atoms,)

    # Damping factor
    alpha_atoms = state.alpha[state.system_idx]
    dt_atoms = dt
    if dt.ndim > 0:
        dt_atoms = dt[state.system_idx]

    b = 1 / (1 + ((alpha_atoms * dt_atoms) / (2 * state.masses)))

    # Scale positions due to volume change
    c_1 = scale_atoms.unsqueeze(-1) * state.positions

    # Time step factor: 2·s/(s+1) where s = scale
    c_2 = (2 * scale_atoms / (scale_atoms + 1)) * b * dt_atoms

    c_3 = (
        state.velocities
        + dt_atoms.unsqueeze(-1) * state.forces / M_2
        + particle_beta / M_2
    )

    state.set_constrained_positions(c_1 + c_2.unsqueeze(-1) * c_3)
    return state


def npt_langevin_isotropic_init(
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    cell_alpha: float | torch.Tensor | None = None,
    b_tau: float | torch.Tensor | None = None,
    **_kwargs: Any,
) -> NPTLangevinIsotropicState:
    """Initialize an NPT Langevin state using logarithmic strain coordinate.

    The strain coordinate ε = (1/d)·ln(V/V₀) provides well-scaled dynamics
    where the conjugate force F_ε = d·V·(P_avg - P_ext) is in energy units.

    Args:
        state: Initial SimState
        model: Model that computes energy, forces, stress
        kT: Target temperature in energy units
        dt: Integration timestep
        alpha: Particle friction coefficient. Defaults to 1/(5·dt).
        cell_alpha: Cell friction coefficient. Defaults to 1/(30·dt).
        b_tau: Barostat time constant. Defaults to 300·dt.

    Returns:
        NPTLangevinIsotropicState: Initialized state with ε = 0
    """
    device, dtype = model.device, model.dtype

    if alpha is None:
        alpha = 1.0 / (5 * dt)
    if cell_alpha is None:
        cell_alpha = 1.0 / (30 * dt)
    if b_tau is None:
        b_tau = 300 * dt

    alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
    cell_alpha = torch.as_tensor(cell_alpha, device=device, dtype=dtype)
    b_tau = torch.as_tensor(b_tau, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)
    dt = torch.as_tensor(dt, device=device, dtype=dtype)

    if alpha.ndim == 0:
        alpha = alpha.expand(state.n_systems)
    if cell_alpha.ndim == 0:
        cell_alpha = cell_alpha.expand(state.n_systems)
    if b_tau.ndim == 0:
        b_tau = b_tau.expand(state.n_systems)

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

    reference_cell = state.cell.clone()

    # ε = 0 at initialization (V = V₀)
    cell_positions = torch.zeros(state.n_systems, device=device, dtype=dtype)
    cell_velocities = torch.zeros(state.n_systems, device=device, dtype=dtype)

    batch_kT = kT.expand(state.n_systems) if kT.ndim == 0 else kT
    n_atoms_per_system = torch.bincount(state.system_idx)
    cell_masses = (n_atoms_per_system + 1) * batch_kT * b_tau * b_tau

    if state.constraints:
        msg = (
            "Constraints are present in the system. "
            "Make sure they are compatible with NPT Langevin dynamics. "
            "We recommend not using constraints with NPT dynamics for now."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    return NPTLangevinIsotropicState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        stress=model_output["stress"],
        alpha=alpha,
        b_tau=b_tau,
        reference_cell=reference_cell,
        cell_positions=cell_positions,
        cell_velocities=cell_velocities,
        cell_masses=cell_masses,
        cell_alpha=cell_alpha,
    )


@dcite("10.1063/1.4901303")
def npt_langevin_isotropic_step(
    state: NPTLangevinIsotropicState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
) -> NPTLangevinIsotropicState:
    r"""Perform one NPT Langevin step using logarithmic strain coordinate.

    Uses the same GJF integrator as :func:`npt_langevin_anisotropic_step` but with the
    cell degree of freedom being the isotropic logarithmic strain
    :math:`\varepsilon = \frac{1}{d}\ln(V/V_0)` instead of the raw volume.

    **Strain force:**

    .. math::

        F_\varepsilon = d \cdot V \cdot (P_{\text{avg}} - P_{\text{ext}})

    where the Jacobian :math:`dV/d\varepsilon = d \cdot V` naturally provides
    a volume factor that makes :math:`F_\varepsilon` an energy (eV), giving
    numerically well-scaled dynamics.

    **Cell reconstruction:**

    .. math::

        V = V_0 \exp(d\,\varepsilon), \quad
        \mathbf{h} = e^\varepsilon \, \mathbf{h}_0

    **Particle scaling:**

    .. math::

        \mathbf{r}_i \to e^{\varepsilon_{n+1} - \varepsilon_n} \, \mathbf{r}_i

    Args:
        state: Current strain-based NPT state
        model: Model computing energy, forces, stress
        dt: Integration timestep
        kT: Target temperature in energy units
        external_pressure: Target pressure

    Returns:
        NPTLangevinIsotropicState: Updated state
    """
    device, dtype = model.device, model.dtype

    state.alpha = torch.as_tensor(state.alpha, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    state.cell_alpha = torch.as_tensor(state.cell_alpha, device=device, dtype=dtype)
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    external_pressure_tensor = torch.as_tensor(
        external_pressure, device=device, dtype=dtype
    )

    batch_kT = kT_tensor.expand(state.n_systems) if kT_tensor.ndim == 0 else kT_tensor

    # Update barostat mass
    n_atoms_per_system = torch.bincount(state.system_idx)
    state.cell_masses = (n_atoms_per_system + 1) * batch_kT * torch.square(state.b_tau)

    # Store initial values
    forces = state.forces
    eps_old = state.cell_positions.clone()

    F_eps_n = _compute_isotropic_cell_force(
        state=state,
        external_pressure=external_pressure_tensor,
        kT=kT_tensor,
    )

    # Generate GJF noise ONCE
    cell_beta = _npt_langevin_isotropic_cell_beta(state, kT_tensor, dt_tensor)
    particle_beta = _npt_langevin_particle_beta(state, kT_tensor, dt_tensor)

    # Step 1: Update strain (cell position step)
    state = _npt_langevin_isotropic_cell_position_step(
        state, dt_tensor, F_eps_n, cell_beta
    )

    # Reconstruct cell from updated strain
    state.cell = state.current_cell

    # Step 2: Update particle positions (with strain-based scaling)
    state = _npt_langevin_isotropic_position_step(
        state, eps_old, dt_tensor, particle_beta
    )

    # Recompute model output
    model_output = model(state)
    state.energy = model_output["energy"]
    state.forces = model_output["forces"]
    state.stress = model_output["stress"]

    # Compute updated strain force
    F_eps_new = _compute_isotropic_cell_force(
        state=state,
        external_pressure=external_pressure_tensor,
        kT=kT_tensor,
    )

    # Step 3: Update strain velocity (uses SAME cell_beta)
    state = _npt_langevin_isotropic_cell_velocity_step(
        state, F_eps_n, dt_tensor, F_eps_new, cell_beta
    )

    # Step 4: Update particle velocities (uses SAME particle_beta)
    return cast(
        "NPTLangevinIsotropicState",
        _npt_langevin_particle_velocity_step(
            state,
            forces,
            dt_tensor,
            particle_beta,
        ),
    )


@dataclass(kw_only=True)
class NPTNoseHooverIsotropicState(NPTState):
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
        stress (torch.Tensor): Stress tensor with shape [n_systems, n_dims, n_dims]
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

    _system_attributes = NPTState._system_attributes | {  # noqa: SLF001
        "reference_cell",
        "cell_position",
        "cell_momentum",
        "cell_mass",
    }
    _global_attributes = NPTState._global_attributes | {  # noqa: SLF001
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


def _npt_nose_hoover_isotropic_cell_info(
    state: NPTNoseHooverIsotropicState,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Gets the current volume and a function to compute the cell from volume.

    This helper function computes the current system volume and returns a function
    that can compute the simulation cell for any given volume. This is useful for
    integration algorithms that need to update the cell based on volume changes.

    Args:
        state (NPTNoseHooverIsotropicState): Current state of the NPT system

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


def _npt_nose_hoover_isotropic_update_cell_mass(
    state: NPTNoseHooverIsotropicState,
    kT: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> NPTNoseHooverIsotropicState:
    """Update the cell mass parameter in an NPT simulation.

    This function updates the mass parameter associated with cell volume fluctuations
    based on the current system size and target temperature. The cell mass controls
    how quickly the volume can change and is chosen to maintain stable pressure
    control.

    Args:
        state (NPTNoseHooverIsotropicState): Current state of the NPT system
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            shape [n_systems]
        device (torch.device): Device for tensor operations
        dtype (torch.dtype): Data type for tensor operations

    Returns:
        NPTNoseHooverIsotropicState: Updated state with new cell mass

    Notes:
        - Cell mass scales with system size (N+1) and dimensionality
        - Larger cell mass gives slower but more stable volume fluctuations
        - Mass depends on barostat relaxation time (tau)
        - Supports batched operations
    """
    _n_particles, dim = state.positions.shape

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


def _npt_nose_hoover_isotropic_sinhx_x(x: torch.Tensor) -> torch.Tensor:
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


def _npt_nose_hoover_isotropic_exp_iL1(  # noqa: N802
    state: NPTNoseHooverIsotropicState,
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
        state (NPTNoseHooverIsotropicState): Current simulation state
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
    sinh_term = _npt_nose_hoover_isotropic_sinhx_x(x_2)  # [n_atoms]

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


def _npt_nose_hoover_isotropic_exp_iL2(  # noqa: N802
    state: NPTNoseHooverIsotropicState,
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
        state (NPTNoseHooverIsotropicState): Current simulation state for batch mapping
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
    sinh_term = _npt_nose_hoover_isotropic_sinhx_x(x_2)  # [n_atoms]

    # Expand dimensions for broadcasting with momenta [n_atoms, 3]
    x_expanded = x.unsqueeze(-1)  # [n_atoms, 1]
    x_2_expanded = x_2.unsqueeze(-1)  # [n_atoms, 1]
    sinh_expanded = sinh_term.unsqueeze(-1)  # [n_atoms, 1]

    # Update momenta with both scaling and force terms
    return momenta * torch.exp(-x_expanded) + dt_2 * forces * sinh_expanded * torch.exp(
        -x_2_expanded
    )


def _npt_nose_hoover_isotropic_compute_cell_force(
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
    # F = alpha * (2 * KE) - dU/dV - P*V*d
    return (
        (alpha * 2 * KE_per_system)
        - (internal_pressure * volume)
        - (external_pressure * volume * dim)
    )


def _npt_nose_hoover_isotropic_inner_step(
    state: NPTNoseHooverIsotropicState,
    model: ModelInterface,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTNoseHooverIsotropicState:
    """Perform one inner step of NPT integration using velocity Verlet algorithm.

    This function implements a single integration step for NPT dynamics, including:
    1. Cell momentum and particle momentum updates (half step)
    2. Position and cell position updates (full step)
    3. Force updates with new positions and cell
    4. Final momentum updates (half step)

    Args:
        model (ModelInterface): Model to compute forces and energies
        state (NPTNoseHooverIsotropicState): Current system state
        dt (torch.Tensor): Integration timestep
        external_pressure (torch.Tensor): Target external pressure

    Returns:
        NPTNoseHooverIsotropicState: Updated state after one integration step
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
    volume, volume_to_cell = _npt_nose_hoover_isotropic_cell_info(state)
    cell = volume_to_cell(volume)

    # First half step: Update momenta
    # alpha = 1 + dim / degrees_of_freedom (3 * natoms - 3)
    alpha = 1 + 3 / state.get_number_of_degrees_of_freedom()  # [n_systems]

    # Reuse stress from previous step since positions and cell unchanged
    cell_force_val = _npt_nose_hoover_isotropic_compute_cell_force(
        alpha=alpha,
        volume=volume,
        positions=positions,
        momenta=momenta,
        masses=masses,
        stress=state.stress,
        external_pressure=external_pressure,
        system_idx=state.system_idx,
    )

    # Update cell momentum and particle momenta
    cell_momentum = cell_momentum + dt_2 * cell_force_val.unsqueeze(-1)
    cell_velocities = cell_momentum.squeeze(-1) / cell_mass
    momenta = _npt_nose_hoover_isotropic_exp_iL2(
        state, alpha, momenta, forces, cell_velocities, dt_2
    )

    # Full step: Update positions
    cell_position = cell_position + cell_velocities * dt

    # Update state with new cell_position before calling functions that depend on it
    state.cell_position = cell_position

    # Get updated cell
    volume, volume_to_cell = _npt_nose_hoover_isotropic_cell_info(state)
    cell = volume_to_cell(volume)

    # Update particle positions and forces
    state.set_constrained_momenta(momenta)
    positions = _npt_nose_hoover_isotropic_exp_iL1(
        state, state.velocities, cell_velocities, dt
    )
    state.set_constrained_positions(positions)
    state.cell = cell
    model_output = model(state)

    # Second half step: Update momenta
    momenta = _npt_nose_hoover_isotropic_exp_iL2(
        state, alpha, momenta, model_output["forces"], cell_velocities, dt_2
    )
    cell_force_val = _npt_nose_hoover_isotropic_compute_cell_force(
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
    state.set_constrained_positions(positions)
    state.set_constrained_momenta(momenta)
    state.forces = model_output["forces"]
    state.stress = model_output["stress"]
    state.energy = model_output["energy"]
    state.cell_position = cell_position
    state.cell_momentum = cell_momentum
    state.cell_mass = cell_mass
    return state


def npt_nose_hoover_isotropic_init(
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
    chain_length: int = 3,
    chain_steps: int = 2,
    sy_steps: int = 3,
    t_tau: float | torch.Tensor | None = None,
    b_tau: float | torch.Tensor | None = None,
    **kwargs: Any,
) -> NPTNoseHooverIsotropicState:
    """Initialize the NPT Nose-Hoover state.

    This function initializes a state for NPT molecular dynamics with Nose-Hoover
    chain thermostats for both temperature and pressure control. It sets up the
    system with appropriate initial conditions including particle positions, momenta,
    cell variables, and thermostat chains.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        model (ModelInterface): Model to compute forces and energies
        state: Initial system state as SimState containing positions, masses,
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
        **kwargs: Additional state variables like atomic_numbers or
            pre-initialized momenta

    Returns:
        NPTNoseHooverIsotropicState: Initialized state containing:
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
    device, dtype = state.device, state.dtype
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    t_tau_tensor = torch.as_tensor(
        10 * dt_tensor if t_tau is None else t_tau, device=device, dtype=dtype
    )
    b_tau_tensor = torch.as_tensor(
        100 * dt_tensor if b_tau is None else b_tau, device=device, dtype=dtype
    )

    # Setup thermostats with appropriate timescales
    barostat_fns = construct_nose_hoover_chain(
        dt_tensor, chain_length, chain_steps, sy_steps, b_tau_tensor
    )
    thermostat_fns = construct_nose_hoover_chain(
        dt_tensor, chain_length, chain_steps, sy_steps, t_tau_tensor
    )

    _n_particles, dim = state.positions.shape
    n_systems = state.n_systems
    atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

    # Initialize cell variables with proper system dimensions
    # cell_momentum: [n_systems, 1] for compatibility with half_step
    cell_position = torch.zeros(n_systems, device=device, dtype=dtype)
    cell_momentum = torch.zeros(n_systems, 1, device=device, dtype=dtype)

    # Handle both scalar and batched kT
    kT_system = kT_tensor.expand(n_systems) if kT_tensor.ndim == 0 else kT_tensor

    # Calculate cell masses for each system
    n_atoms_per_system = torch.bincount(state.system_idx, minlength=n_systems)
    cell_mass = dim * (n_atoms_per_system + 1) * kT_system * torch.square(b_tau_tensor)
    cell_mass = cell_mass.to(device=device, dtype=dtype)

    # Calculate cell kinetic energy (using first system for initialization)
    dof_barostat = torch.ones(n_systems, device=device, dtype=dtype)
    KE_cell = (cell_momentum.squeeze(-1) ** 2) / (2 * cell_mass)

    # Initialize momenta
    momenta = kwargs.get("momenta")
    if momenta is None:
        momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions,
            state.masses,
            state.system_idx,
            kT_tensor,
            state.rng,
        )

    # Compute total DOF for thermostat initialization and a zero KE placeholder
    dof_per_system = state.get_number_of_degrees_of_freedom() - 3

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
    stress = model_output["stress"]

    if state.constraints:
        # warn if constraints are present
        msg = (
            "Constraints are present in the system. "
            "Make sure they are compatible with NPT Nosé Hoover dynamics."
            "We recommend not using constraints with NPT dynamics for now."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    # Create initial state
    return NPTNoseHooverIsotropicState.from_state(
        state,
        momenta=momenta,
        energy=energy,
        forces=forces,
        stress=stress,
        atomic_numbers=atomic_numbers,
        reference_cell=reference_cell,
        cell_position=cell_position,
        cell_momentum=cell_momentum,
        cell_mass=cell_mass,
        barostat=barostat_fns.initialize(dof_barostat, KE_cell, kT_tensor),
        thermostat=thermostat_fns.initialize(dof_per_system, KE_thermostat, kT_tensor),
        barostat_fns=barostat_fns,
        thermostat_fns=thermostat_fns,
    )


@dcite("10.1080/00268979600100761")
@dcite("10.1088/0305-4470/39/19/S18")
def npt_nose_hoover_isotropic_step(
    state: NPTNoseHooverIsotropicState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
) -> NPTNoseHooverIsotropicState:
    r"""Perform a complete NPT integration step with Nose-Hoover chain thermostats.

    Implements the MTK (Martyna-Tobias-Klein) NPT scheme from Tuckerman et al.
    (2006) [10]_ with Nose-Hoover chains from Martyna et al. (1996) [3]_.

    **Equations of motion** (Tuckerman et al. 2006, Eqs. 1-6):

    .. math::

        \dot{\mathbf{r}}_i &= \frac{\mathbf{p}_i}{m_i}
            + \frac{p_\epsilon}{W}\,\mathbf{r}_i \\
        \dot{\mathbf{p}}_i &= \mathbf{F}_i
            - \alpha\,\frac{p_\epsilon}{W}\,\mathbf{p}_i \\
        \dot{\epsilon} &= \frac{p_\epsilon}{W} \\
        \dot{p}_\epsilon &= G_\epsilon
            = \alpha\,(2K) + \text{Tr}(\boldsymbol{\sigma}_{\text{int}})\,V
            - P_{\text{ext}}\,V\,d

    where :math:`\epsilon = (1/d)\ln(V/V_0)` is the logarithmic cell coordinate,
    :math:`\alpha = 1 + d/N_f`, :math:`d=3` is spatial dimension, and
    :math:`N_f = 3N - 3` the degrees of freedom.

    **Symmetric propagator** (Trotter factorization):

    .. math::

        e^{i\mathcal{L}\Delta t} =
          e^{i\mathcal{L}_{\text{NHC-baro}}\frac{\Delta t}{2}}
          \;e^{i\mathcal{L}_{\text{NHC-part}}\frac{\Delta t}{2}}
          \;e^{i\mathcal{L}_2\frac{\Delta t}{2}}
          \;e^{i\mathcal{L}_1\Delta t}
          \;e^{i\mathcal{L}_2\frac{\Delta t}{2}}
          \;e^{i\mathcal{L}_{\text{NHC-part}}\frac{\Delta t}{2}}
          \;e^{i\mathcal{L}_{\text{NHC-baro}}\frac{\Delta t}{2}}

    **Position update** :math:`e^{i\mathcal{L}_1\Delta t}`:

    .. math::

        \mathbf{r}_i \leftarrow \mathbf{r}_i
            + \bigl(e^{v_\epsilon\Delta t} - 1\bigr)\,\mathbf{r}_i
            + \Delta t\,\mathbf{v}_i\,e^{v_\epsilon\Delta t/2}
            \,\frac{\sinh(v_\epsilon\Delta t/2)}{v_\epsilon\Delta t/2}

    **Momentum update** :math:`e^{i\mathcal{L}_2\Delta t/2}`:

    .. math::

        \mathbf{p}_i \leftarrow \mathbf{p}_i\,e^{-\alpha v_\epsilon\Delta t/2}
            + \frac{\Delta t}{2}\,\mathbf{F}_i\,
            e^{-\alpha v_\epsilon\Delta t/4}
            \,\frac{\sinh(\alpha v_\epsilon\Delta t/4)}
            {\alpha v_\epsilon\Delta t/4}

    where :math:`v_\epsilon = p_\epsilon / W` is the cell velocity.

    **Variable mapping (equation -> code):**

    ============================================  ============================
    Equation symbol                               Code variable
    ============================================  ============================
    :math:`\mathbf{r}_i`  (positions)             ``state.positions``
    :math:`\mathbf{p}_i`  (momenta)               ``state.momenta``
    :math:`m_i`           (masses)                ``state.masses``
    :math:`\mathbf{F}_i`  (forces)                ``state.forces``
    :math:`\epsilon`      (log-cell coordinate)   ``state.cell_position``
    :math:`p_\epsilon`    (cell momentum)         ``state.cell_momentum``
    :math:`W`             (cell mass)             ``state.cell_mass``
    :math:`\alpha`        (1 + d/Nf)              ``alpha`` (local)
    :math:`v_\epsilon`    (cell velocity)         ``cell_velocities`` (local)
    :math:`V_0`           (reference volume)      ``det(state.reference_cell)``
    :math:`G_\epsilon`    (cell force)            ``cell_force_val``
    :math:`P_{\text{ext}}` (target pressure)      ``external_pressure``
    :math:`k_BT`          (thermal energy)        ``kT``
    :math:`\Delta t`      (timestep)              ``dt``
    ============================================  ============================

    If the center of mass motion is removed initially, it remains removed throughout
    the simulation, so the degrees of freedom decreases by 3.

    Args:
        model: Model to compute forces and energies
        state: Current system state
        dt: Integration timestep
        kT: Target temperature
        external_pressure: Target external pressure

    Returns:
        NPTNoseHooverIsotropicState: Updated state after complete integration step

    References:
        .. [10] Tuckerman, M. E., et al. "A Liouville-operator derived
           measure-preserving integrator for molecular dynamics simulations in
           the isothermal-isobaric ensemble." J. Phys. A 39(19), 5629-5651 (2006).
        .. [3] Martyna, G. J., et al. "Explicit reversible integrators for extended
           systems dynamics." Mol. Phys. 87(5), 1117-1157 (1996).
    """
    device, dtype = model.device, model.dtype
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    external_pressure_tensor = torch.as_tensor(
        external_pressure, device=device, dtype=dtype
    )

    # Unpack state variables for clarity
    barostat = state.barostat
    thermostat = state.thermostat

    # Update mass parameters
    state.barostat = state.barostat_fns.update_mass(barostat, kT_tensor)
    state.thermostat = state.thermostat_fns.update_mass(thermostat, kT_tensor)
    state = _npt_nose_hoover_isotropic_update_cell_mass(state, kT_tensor, device, dtype)

    # First half step of thermostat chains
    cell_system_idx = torch.arange(state.n_systems, device=device)
    state.cell_momentum, state.barostat = state.barostat_fns.half_step(
        state.cell_momentum, state.barostat, kT_tensor, cell_system_idx
    )
    state.momenta, state.thermostat = state.thermostat_fns.half_step(
        state.momenta, state.thermostat, kT_tensor, state.system_idx
    )

    # Perform inner NPT step
    state = _npt_nose_hoover_isotropic_inner_step(
        state, model, dt_tensor, external_pressure_tensor
    )

    # Update kinetic energies for thermostats
    KE = ts.calc_kinetic_energy(
        masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
    )
    state.thermostat.kinetic_energy = KE

    KE_cell = (torch.square(state.cell_momentum.squeeze(-1))) / (2 * state.cell_mass)
    state.barostat.kinetic_energy = KE_cell

    # Second half step of thermostat chains
    state.momenta, state.thermostat = state.thermostat_fns.half_step(
        state.momenta, state.thermostat, kT_tensor, state.system_idx
    )
    state.cell_momentum, state.barostat = state.barostat_fns.half_step(
        state.cell_momentum, state.barostat, kT_tensor, cell_system_idx
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


def npt_nose_hoover_isotropic_invariant(
    state: NPTNoseHooverIsotropicState,
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
    dof_per_system = state.get_number_of_degrees_of_freedom()

    # Initialize total energy with PE + KE
    e_tot = e_pot + e_kin_per_system

    # Add thermostat chain contributions (batched per system, DOF = 3 * n_atoms - 3)
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
class NPTCRescaleState(NPTState):
    """State for NPT ensemble with cell rescaling barostat.

    This class extends the NPTState to include variables and properties
    specific to the NPT ensemble with a cell rescaling barostat.
    """

    isothermal_compressibility: torch.Tensor  # shape: [n_systems]
    tau_p: torch.Tensor  # shape: [n_systems]
    initial_cell: torch.Tensor  # shape: [n_systems, 3, 3]
    initial_cell_inv: torch.Tensor  # shape: [n_systems, 3, 3]
    initial_volume: torch.Tensor  # shape: [n_systems]

    _system_attributes = NPTState._system_attributes | {  # noqa: SLF001
        "isothermal_compressibility",
        "tau_p",
        "initial_cell",
        "initial_cell_inv",
        "initial_volume",
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


def _compute_deviatoric_correction(
    cell: torch.Tensor,
    volume: torch.Tensor,
    initial_cell_inv: torch.Tensor,
    initial_volume: torch.Tensor,
    external_pressure_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute deviatoric pressure correction for non-hydrostatic external stress.

    Follows the algorithm from Bussi's crescale reference implementation:
    projects the deviatoric part of the external stress through the reference
    cell frame.

    Args:
        cell: Current cell matrix, shape [n_systems, 3, 3]
        volume: Current volume, shape [n_systems]
        initial_cell_inv: Inverse of initial cell, shape [n_systems, 3, 3]
        initial_volume: Initial volume, shape [n_systems]
        external_pressure_tensor: Full external pressure tensor [n_systems, 3, 3]

    Returns:
        pressure_hydro: Hydrostatic pressure scalar [n_systems]
        pressure_dev: Deviatoric pressure correction [n_systems, 3, 3]
        trace_pressure_dev: Trace of pressure_dev [n_systems]
    """
    pressure_hydro = torch.einsum("bii->b", external_pressure_tensor) / 3
    I = torch.eye(3, device=cell.device, dtype=cell.dtype)  # noqa: E741
    stress_dev = external_pressure_tensor - pressure_hydro[:, None, None] * I.expand_as(
        external_pressure_tensor
    )
    # Project to reference coordinates: sigma = V0 * h0_inv^T @ stress_dev @ h0_inv
    sigma = initial_volume[:, None, None] * (
        initial_cell_inv.transpose(-2, -1) @ stress_dev @ initial_cell_inv
    )
    # Symmetrize and project back: pressure_dev = h^T @ 0.5*(sigma+sigma^T) @ h / V
    sigma_sym = 0.5 * (sigma + sigma.transpose(-2, -1))
    pressure_dev = cell.transpose(-2, -1) @ sigma_sym @ cell / volume[:, None, None]
    trace_pressure_dev = torch.einsum("bii->b", pressure_dev)
    return pressure_hydro, pressure_dev, trace_pressure_dev


def _crescale_triclinic_barostat_step(
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

    # Deviatoric correction for non-hydrostatic external stress
    deviatoric = external_pressure.ndim >= 2
    if deviatoric:
        # Expand [3,3] -> [n_systems, 3, 3] if needed
        ext_p_tensor = external_pressure
        if ext_p_tensor.ndim == 2:
            ext_p_tensor = ext_p_tensor.unsqueeze(0).expand(state.n_systems, -1, -1)
        pressure_hydro, pressure_dev, trace_pressure_dev = _compute_deviatoric_correction(
            cell=state.cell,
            volume=volume,
            initial_cell_inv=state.initial_cell_inv,
            initial_volume=state.initial_volume,
            external_pressure_tensor=ext_p_tensor,
        )
        effective_p_ext = pressure_hydro + trace_pressure_dev / 3
    else:
        effective_p_ext = external_pressure

    ## Step 1: propagate sqrt(volume) for dt/2
    change_sqrt_vol = -prefactor * (
        effective_p_ext - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * _randn_for_state(state, sqrt_vol.shape)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol

    ## Step 2: compute deformation matrix
    random_coeff = 2 * state.isothermal_compressibility * kT * dt / (3 * state.tau_p)
    prefactor_random_matrix = torch.sqrt(random_coeff) / new_sqrt_volume
    I = torch.eye(  # noqa: E741
        3, device=state.positions.device, dtype=state.positions.dtype
    ).expand_as(P_int)
    # Driving force: traceless part of (P_int - pressure_dev)
    P_drive = P_int
    if deviatoric:
        P_drive = P_int - pressure_dev
    trace_P_drive = torch.einsum("bii->b", P_drive)
    a_tilde = (state.isothermal_compressibility / (3 * state.tau_p))[:, None, None] * (
        P_drive - trace_P_drive[:, None, None] / 3 * I
    )
    random_matrix = torch.randn(
        state.n_systems,
        3,
        3,
        device=state.positions.device,
        dtype=state.positions.dtype,
        generator=state.rng,
    )
    random_matrix_tilde = (
        random_matrix - torch.einsum("bii->b", random_matrix)[:, None, None] / 3 * I
    )
    deformation_matrix = torch.matrix_exp(
        a_tilde * dt + prefactor_random_matrix[:, None, None] * random_matrix_tilde
    )
    deformation_matrix = rotate_gram_schmidt(deformation_matrix)

    ## Step 3: propagate sqrt(volume) for dt/2
    new_sqrt_volume += -prefactor * (
        effective_p_ext - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * _randn_for_state(state, sqrt_vol.shape)
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
    # Right multiply: cell @ rscaling^T preserves fractional coordinates
    state.cell = state.cell @ rscaling.mT
    return state


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
    ) * dt / 2 + prefactor_random * _randn_for_state(state, sqrt_vol.shape)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol
    ## Step 2: compute deformation matrix
    prefactor_random_matrix = (
        torch.sqrt(2 * state.isothermal_compressibility * kT * dt / (3 * state.tau_p))
        / new_sqrt_volume
    )
    # Note: it corresponds to using a diagonal isothermal compressibility tensor
    P_int_diagonal = torch.diagonal(P_int, dim1=-2, dim2=-1)
    a_tilde = (state.isothermal_compressibility / (3 * state.tau_p))[:, None] * (
        P_int_diagonal - trace_P_int[:, None] / 3
    )

    random_matrix = torch.randn(
        state.n_systems,
        3,
        device=state.positions.device,
        dtype=state.positions.dtype,
        generator=state.rng,
    )
    random_matrix_tilde = random_matrix - torch.mean(random_matrix, dim=1, keepdim=True)
    deformation_matrix = torch.exp(
        a_tilde * dt + prefactor_random_matrix[:, None] * random_matrix_tilde
    )

    ## Step 3: propagate sqrt(volume) for dt/2
    new_sqrt_volume += -prefactor * (
        external_pressure - trace_P_int / 3 - kT / (2 * volume)
    ) * dt / 2 + prefactor_random * _randn_for_state(state, sqrt_vol.shape)
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


def _crescale_triclinic_average_barostat_step(
    state: NPTCRescaleState,
    kT: torch.Tensor,
    dt: torch.Tensor,
    external_pressure: torch.Tensor,
) -> NPTCRescaleState:
    volume = torch.det(state.cell)  # shape: (n_systems,)
    P_int = compute_average_pressure_tensor(
        degrees_of_freedom=state.get_number_of_degrees_of_freedom() / 3,
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
    ) * dt / 2 + prefactor_random * _randn_for_state(state, sqrt_vol.shape)
    new_sqrt_volume = sqrt_vol + change_sqrt_vol
    ## Step 2: compute deformation matrix
    prefactor_random_matrix = (
        torch.sqrt(2 * state.isothermal_compressibility * kT * dt / (3 * state.tau_p))
        / new_sqrt_volume
    )
    a_tilde = (state.isothermal_compressibility / (3 * state.tau_p))[:, None, None] * (
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
        generator=state.rng,
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
    ) * dt / 2 + prefactor_random * _randn_for_state(state, sqrt_vol.shape)
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
    # Right multiply: cell @ rscaling^T preserves fractional coordinates
    state.cell = state.cell @ rscaling.mT
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
    ) * dt + torch.sqrt(
        2 * torch.ones_like(sqrt_vol)
    ) * prefactor_random * _randn_for_state(state, sqrt_vol.shape)
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


def _coerce_crescale_step_inputs(
    state: NPTCRescaleState,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
    tau: float | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize scalar-or-tensor C-rescale step parameters to state tensors."""
    device, dtype = state.device, state.dtype
    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    kT_tensor = torch.as_tensor(kT, device=device, dtype=dtype)
    external_pressure_tensor = torch.as_tensor(
        external_pressure, device=device, dtype=dtype
    )
    tau_tensor = torch.as_tensor(
        1 * dt_tensor if tau is None else tau, device=device, dtype=dtype
    )
    return dt_tensor, kT_tensor, external_pressure_tensor, tau_tensor


@dcite("10.1063/5.0020514")
@dcite("10.3390/app12031139")
def npt_crescale_triclinic_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
) -> NPTCRescaleState:
    r"""Perform one NPT integration step with anisotropic stochastic cell rescaling.

    Implements the anisotropic C-Rescale barostat from Del Tatto et al.
    (2022) [7]_ extending the isotropic scheme of Bernetti & Bussi (2020) [6]_.
    Cell lengths and angles can change independently. Uses instantaneous kinetic
    energy. Both positions and momenta are scaled.

    **Trotter splitting:**

    V-Rescale(dt/2) -> B(dt/2) -> Barostat(dt) -> Force eval -> B(dt/2) ->
    V-Rescale(dt/2)

    **Barostat sub-steps** (3-step volume + deformation update):

    Step 1 -- Propagate :math:`\sqrt{V}` for :math:`\Delta t/2` (same SDE as
    isotropic, Eq. 7 of [6]_):

    .. math::

        \Delta\lambda = -\frac{\beta_T\lambda}{2\tau_p}
            \left(P_0 - \frac{\text{Tr}(\mathbf{P}_{\text{int}})}{3}
            - \frac{k_BT}{2V}\right)\frac{\Delta t}{2}
            + \sqrt{\frac{k_BT\beta_T\Delta t}{4\tau_p}}\;R

    Step 2 -- Compute deviatoric deformation matrix:

    .. math::

        \tilde{\mathbf{A}} &= \frac{\beta_T}{3\tau_p}
            \left(\mathbf{P}_{\text{int}}
            - \frac{\text{Tr}(\mathbf{P}_{\text{int}})}{3}\,\mathbf{I}\right) \\
        \boldsymbol{\mu}_{\text{dev}} &= \exp\bigl(\tilde{\mathbf{A}}\,\Delta t
            + \sigma\,\tilde{\mathbf{R}}\bigr)

    where :math:`\sigma = \sqrt{2\beta_T k_BT\Delta t/(3\tau_p)}\;/\;\sqrt{V'}`
    and :math:`\tilde{\mathbf{R}}` is a traceless random matrix.

    Step 3 -- Propagate :math:`\sqrt{V}` for :math:`\Delta t/2` (same as step 1).

    **Total scaling and update:**

    .. math::

        \boldsymbol{\mu} &= \boldsymbol{\mu}_{\text{dev}}
            \cdot (V'/V)^{1/3} \\
        \mathbf{r}_i &\leftarrow \boldsymbol{\mu}\,\mathbf{r}_i
            + (\boldsymbol{\mu}^{-T} + \boldsymbol{\mu})\,
            \frac{\mathbf{p}_i}{2m_i}\,\Delta t \\
        \mathbf{p}_i &\leftarrow \boldsymbol{\mu}^{-T}\,\mathbf{p}_i \\
        \mathbf{h} &\leftarrow \mathbf{h}\,\boldsymbol{\mu}^T

    **Variable mapping (equation -> code):**

    ============================================  ================================
    Equation symbol                               Code variable
    ============================================  ================================
    :math:`V`             (volume)                ``volume``
    :math:`\lambda`       (:math:`\sqrt{V}`)      ``sqrt_vol``
    :math:`\beta_T`       (compressibility)       ``state.isothermal_compressibility``
    :math:`\tau_p`        (barostat relax. time)  ``state.tau_p``
    :math:`P_0`           (target pressure)       ``external_pressure``
    :math:`\mathbf{P}_{\text{int}}` (press. tensor) ``P_int``
    :math:`\tilde{\mathbf{A}}`  (deviator drive)  ``a_tilde``
    :math:`\boldsymbol{\mu}_{\text{dev}}`         ``deformation_matrix``
    :math:`\boldsymbol{\mu}`  (total scaling)     ``rscaling``
    :math:`\boldsymbol{\mu}^{-T}` (mom. scaling)  ``vscaling``
    :math:`\tilde{\mathbf{R}}`  (traceless noise)  ``random_matrix_tilde``
    :math:`\sigma`        (noise prefactor)       ``prefactor_random_matrix``
    :math:`k_BT`          (thermal energy)        ``kT``
    :math:`\Delta t`      (timestep)              ``dt``
    :math:`\tau`          (thermostat relax.)     ``tau`` (V-Rescale)
    ============================================  ================================

    Args:
        model: Model to compute forces and energies
        state: Current system state
        dt: Integration timestep
        kT: Target temperature
        external_pressure: Target external pressure
        tau: V-Rescale thermostat relaxation time. If None, defaults to 100*dt

    Returns:
        NPTCRescaleState: Updated state after one integration step

    References:
        .. [7] Del Tatto, V., et al. "Molecular dynamics of solids at constant
           pressure and stress using anisotropic stochastic cell rescaling."
           Applied Sciences 12(3), 1139 (2022).
        .. [6] Bernetti, M. & Bussi, G. "Pressure control using stochastic cell
           rescaling." J. Chem. Phys. 153, 114107 (2020).
    """
    dt_tensor, kT_tensor, external_pressure_tensor, tau_tensor = (
        _coerce_crescale_step_inputs(state, dt, kT, external_pressure, tau)
    )
    state = _vrescale_update(state, tau_tensor, kT_tensor, dt_tensor / 2)

    state = momentum_step(state, dt_tensor / 2)

    # Barostat step
    state = _crescale_triclinic_barostat_step(
        state, kT_tensor, dt_tensor, external_pressure_tensor
    )

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]

    # Final momentum step
    state = momentum_step(state, dt_tensor / 2)

    # Final thermostat step
    return _vrescale_update(state, tau_tensor, kT_tensor, dt_tensor / 2)


@dcite("10.1063/5.0020514")
@dcite("10.3390/app12031139")
def npt_crescale_anisotropic_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
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
    device, dtype = model.device, model.dtype
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)
    external_pressure = torch.as_tensor(external_pressure, device=device, dtype=dtype)

    # Note: would probably be better to have tau in NVTCRescaleState
    tau = torch.as_tensor(tau or 100 * dt, device=device, dtype=dtype)

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


@dcite("10.1063/5.0020514")
@dcite("10.3390/app12031139")
def npt_crescale_triclinic_average_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
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
    device, dtype = model.device, model.dtype
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)
    external_pressure = torch.as_tensor(external_pressure, device=device, dtype=dtype)

    # Note: would probably be better to have tau in NVTCRescaleState
    tau = torch.as_tensor(tau or 1 * dt, device=device, dtype=dtype)

    state = _vrescale_update(state, tau, kT, dt / 2)

    state = momentum_step(state, dt / 2)

    # Barostat step
    state = _crescale_triclinic_average_barostat_step(state, kT, dt, external_pressure)

    # Forces
    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]

    # Final momentum step
    state = momentum_step(state, dt / 2)

    # Final thermostat step
    return _vrescale_update(state, tau, kT, dt / 2)


@dcite("10.1063/5.0020514")
def npt_crescale_isotropic_step(
    state: NPTCRescaleState,
    model: ModelInterface,
    *,
    dt: float | torch.Tensor,
    kT: float | torch.Tensor,
    external_pressure: float | torch.Tensor,
    tau: float | torch.Tensor | None = None,
) -> NPTCRescaleState:
    r"""Perform one NPT integration step with isotropic stochastic cell rescaling.

    Implements isotropic C-Rescale from Bernetti & Bussi (2020) [6]_.
    Cell shape is preserved; cell lengths are scaled equally.

    **Trotter splitting:**

    V-Rescale(dt/2) -> B(dt/2) -> Barostat(dt) -> Force eval -> B(dt/2) ->
    V-Rescale(dt/2)

    **Isotropic volume SDE** (Eq. 7 of [6]_, using :math:`\lambda = \sqrt{V}`):

    .. math::

        d\lambda = -\frac{\beta_T\lambda}{2\tau_p}
            \left(P_0 - \frac{\text{Tr}(\mathbf{P}_{\text{int}})}{3}
            - \frac{k_BT}{2V}\right) dt
            + \sqrt{\frac{k_BT\,\beta_T}{2\tau_p}}\;dW

    where :math:`\beta_T` is the isothermal compressibility and
    :math:`\mathbf{P}_{\text{int}}` is the instantaneous pressure tensor
    (including the kinetic contribution).

    **Position and momentum scaling** (SI Eqs. S13a-b of [6]_, corrected):

    .. math::

        \mathbf{r}_i &\leftarrow \mu\,\mathbf{r}_i
            + (\mu + \mu^{-1})\,\frac{\mathbf{p}_i}{2m_i}\,\Delta t \\
        \mathbf{p}_i &\leftarrow \mu^{-1}\,\mathbf{p}_i \\
        \mathbf{h} &\leftarrow \mu\,\mathbf{h}

    where :math:`\mu = (V'/V)^{1/3}` is the isotropic scaling factor and
    :math:`\mathbf{h}` is the cell matrix.

    **Variable mapping (equation -> code):**

    ============================================  ================================
    Equation symbol                               Code variable
    ============================================  ================================
    :math:`V`             (volume)                ``volume``
    :math:`\lambda`       (:math:`\sqrt{V}`)      ``sqrt_vol``
    :math:`\beta_T`       (compressibility)       ``state.isothermal_compressibility``
    :math:`\tau_p`        (barostat relax. time)  ``state.tau_p``
    :math:`P_0`           (target pressure)       ``external_pressure``
    :math:`\mathbf{P}_{\text{int}}` (press. tensor) ``P_int``
    :math:`\text{Tr}(\mathbf{P}_{\text{int}})`   ``trace_P_int``
    :math:`\mu`           (scaling factor)        ``rscaling``
    :math:`k_BT`          (thermal energy)        ``kT``
    :math:`\Delta t`      (timestep)              ``dt``
    :math:`\tau`          (thermostat relax.)     ``tau`` (V-Rescale)
    ============================================  ================================

    Args:
        model: Model to compute forces and energies
        state: Current system state
        dt: Integration timestep
        kT: Target temperature
        external_pressure: Target external pressure
        tau: V-Rescale thermostat relaxation time. If None, defaults to 100*dt

    Returns:
        NPTCRescaleState: Updated state after one integration step

    References:
        .. [6] Bernetti, M. & Bussi, G. "Pressure control using stochastic cell
           rescaling." J. Chem. Phys. 153, 114107 (2020). Note: SI Eq. S13a has a
           typo (positions must also be scaled by mu).
    """
    device, dtype = model.device, model.dtype
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)
    external_pressure = torch.as_tensor(external_pressure, device=device, dtype=dtype)

    # Note: would probably be better to have tau in NVTCRescaleState
    tau = torch.as_tensor(tau or 1 * dt, device=device, dtype=dtype)

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
    state: SimState,
    model: ModelInterface,
    *,
    kT: float | torch.Tensor,
    dt: float | torch.Tensor,
    tau_p: float | torch.Tensor | None = None,
    isothermal_compressibility: float | torch.Tensor | None = None,
) -> NPTCRescaleState:
    """Initialize the NPT cell rescaling state.

    This function initializes a state for NPT molecular dynamics with a
    cell rescaling barostat. It sets up the system with appropriate initial
    conditions including particle positions, momenta, and cell variables.

    Only allow isotropic external stress, but can run both isotropic and
    anisotropic cell rescaling.

    To seed the RNG set ``state.rng = seed`` before calling.

    Args:
        state: Initial system state as SimState containing positions, masses,
            cell, and PBC information
        model (ModelInterface): Model to compute forces and energies
        kT: Target temperature in energy units
        dt: Integration timestep
        tau_p: Barostat relaxation time. Controls how quickly pressure equilibrates.
        isothermal_compressibility: Isothermal compressibility of the system.
    """
    device, dtype = model.device, model.dtype

    # Convert all parameters to tensors with correct device and dtype
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    kT = torch.as_tensor(kT, device=device, dtype=dtype)

    # Set default values if not provided
    tau_p = torch.as_tensor(tau_p or 3 * dt, device=device, dtype=dtype)  # 5ps for dt=1fs
    isothermal_compressibility = torch.as_tensor(
        isothermal_compressibility
        or 1e-6 / MetalUnits.pressure,  # 1e-6 bar^-1 for metals
        device=device,
        dtype=dtype,  # (eV/A^3)^-1
    )

    if tau_p.ndim == 0:
        tau_p = tau_p.expand(state.n_systems)
    if isothermal_compressibility.ndim == 0:
        isothermal_compressibility = isothermal_compressibility.expand(state.n_systems)

    # Get model output to initialize forces and stress
    model_output = model(state)

    # Initialize momenta if not provided
    momenta = getattr(state, "momenta", None)
    if momenta is None:
        momenta = initialize_momenta(
            state.positions,
            state.masses,
            state.system_idx,
            kT,
            state.rng,
        )

    # Store initial cell for deviatoric correction
    initial_cell = state.cell.clone()
    initial_cell_inv = torch.inverse(initial_cell)
    initial_volume = torch.det(initial_cell)

    # Create the initial state
    return NPTCRescaleState.from_state(
        state,
        momenta=momenta,
        energy=model_output["energy"],
        forces=model_output["forces"],
        stress=model_output["stress"],
        tau_p=tau_p,
        isothermal_compressibility=isothermal_compressibility,
        initial_cell=initial_cell,
        initial_cell_inv=initial_cell_inv,
        initial_volume=initial_volume,
    )
