"""Optimizers for structure optimization."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torch_sim.math import expm, expm_frechet
from torch_sim.math import matrix_log_33 as logm
from torch_sim.state import BaseState, StateDict
from torch_sim.unbatched.unbatched_integrators import velocity_verlet


@dataclass
class OptimizerState(BaseState):
    """Base state class for optimization algorithms.

    Contains the common state variables needed across different optimizers.

    Attributes:
        positions: Atomic positions tensor of shape (n_atoms, 3)
        forces: Forces tensor of shape (n_atoms, 3)
        energy: Total energy scalar
        masses: Atomic masses tensor of shape (n_atoms,)
        cell: Unit cell tensor of shape (3, 3)
        pbc: Periodic boundary conditions flags
    """

    forces: torch.Tensor
    energy: torch.Tensor


@dataclass
class GDState(OptimizerState):
    """State class for gradient descent optimization."""


def gradient_descent(
    *,
    model: torch.nn.Module,
    lr: float = 0.01,
) -> tuple[Callable[[StateDict | BaseState], GDState], Callable[[GDState], GDState]]:
    """Initialize a simple gradient descent optimization.

    Gradient descent updates atomic positions by moving along the direction of the forces
    (negative energy gradient) with a fixed learning rate. While simpler than more
    sophisticated optimizers like FIRE, it can be effective for well-behaved potential
    energy surfaces.

    Args:
        model: Neural network model that computes energies and forces
        lr: Step size for position updates (default: 0.01)

    Returns:
        Tuple containing:
        - Initialization function that creates the initial GDState
        - Update function that performs one gradient descent step

    Notes:
        - Best suited for systems close to their minimum energy configuration
    """
    device = model.device
    dtype = model.dtype

    # Convert learning rate to tensor
    if not isinstance(lr, torch.Tensor):
        lr = torch.tensor(lr, device=device, dtype=dtype)

    def gd_init(state: BaseState | StateDict, **kwargs) -> GDState:
        """Initialize the gradient descent optimizer state.

        Args:
            state: Initial system state
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial GDState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get initial forces and energy from model
        model_output = model(state)

        return GDState(
            positions=state.positions,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            forces=model_output["forces"],
            energy=model_output["energy"],
        )

    def gd_step(state: GDState, lr: torch.Tensor = lr) -> GDState:
        """Perform one gradient descent optimization step.

        Args:
            state: Current optimization state
            lr: Learning rate for position updates (default: value from initialization)

        Returns:
            Updated state after one optimization step
        """
        # Update positions using forces and learning rate
        state.positions = state.positions + lr * state.forces

        # Update forces and energy at new positions
        results = model(state)
        state.forces = results["forces"]
        state.energy = results["energy"]

        return state

    return gd_init, gd_step


@dataclass
class FIREState(OptimizerState):
    """State class for FIRE optimization.

    Extends OptimizerState with additional variables needed for FIRE dynamics.

    Attributes:
        momenta: Atomic momenta tensor of shape (n_atoms, 3)
        dt: Current timestep
        alpha: Current mixing parameter
        n_pos: Number of consecutive steps with positive power
    """

    momenta: torch.Tensor
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: int

    @property
    def velocities(self) -> torch.Tensor:
        """Calculate velocities from momenta and masses.

        Returns:
            Atomic velocities tensor of shape (n_atoms, 3)
        """
        return self.momenta / self.masses.unsqueeze(-1)


def fire(
    *,
    model: torch.nn.Module,
    dt_max: float = 0.4,
    dt_start: float = 0.01,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    alpha_start: float = 0.1,
    eps: float = 1e-8,
) -> tuple[
    Callable[[BaseState | StateDict], FIREState], Callable[[FIREState], FIREState]
]:
    """Initialize a FIRE (Fast Inertial Relaxation Engine) optimization.

    FIRE is a molecular dynamics-based optimization algorithm that combines velocity
    Verlet integration with a form of adaptive velocity damping. It is particularly
    effective for atomic structure optimization.

    Args:
        model: Neural network model that computes energies and forces.
        dt_max: Maximum allowed timestep (default: 0.4).
        dt_start: Initial timestep (default: 0.01).
        n_min: Minimum number of steps before timestep increase (default: 5).
        f_inc: Factor for timestep increase (default: 1.1).
        f_dec: Factor for timestep decrease (default: 0.5).
        f_alpha: Factor for damping parameter decrease (default: 0.99).
        alpha_start: Initial value of damping parameter (default: 0.1).
        eps: Small value for numerical stability (default: 1e-8).

    Returns:
        A tuple containing:
        - Initialization function that creates the initial FIREState
        - Update function that performs one FIRE optimization step

    Notes:
        - The algorithm adaptively adjusts the timestep and damping based on the
          optimization trajectory
        - Larger dt_max allows faster convergence but may cause instability
        - n_min prevents premature timestep increases
        - f_inc and f_dec control how quickly the timestep changes
        - alpha_start and f_alpha control the strength and adaptation of damping
    References:
        - Bitzek et al., PRL 97, 170201 (2006) - Original FIRE paper
    """
    device = model.device
    dtype = model.dtype

    # Convert parameters to tensors
    params = [dt_max, n_min, f_inc, f_dec, f_alpha, dt_start, alpha_start]
    dt_max, n_min, f_inc, f_dec, f_alpha, dt_start, alpha_start = [
        p if isinstance(p, torch.Tensor) else torch.tensor(p, device=device, dtype=dtype)
        for p in params
    ]

    def fire_init(state: BaseState | StateDict, **kwargs) -> FIREState:
        """Initialize the FIRE optimizer state.

        Args:
            state: Initial system state
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial FIREState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)
        # Get initial forces and energy from model
        model_output = model(state)
        momenta = torch.zeros_like(state.positions, device=device, dtype=dtype)

        initial_state = FIREState(
            positions=state.positions,
            forces=model_output["forces"],
            energy=model_output["energy"],
            masses=state.masses,
            momenta=momenta,
            atomic_numbers=atomic_numbers,
            cell=state.cell,
            pbc=state.pbc,
            dt=dt_max,
            alpha=alpha_start,
            n_pos=0,
        )
        return initial_state  # noqa: RET504

    def fire_update(
        state: FIREState,
        dt_max: torch.Tensor = dt_max,
        n_min: torch.Tensor = n_min,
        f_inc: torch.Tensor = f_inc,
        f_dec: torch.Tensor = f_dec,
        f_alpha: torch.Tensor = f_alpha,
        alpha_start: torch.Tensor = alpha_start,
    ) -> FIREState:
        """Perform one FIRE optimization step.

        Args:
            state: Current optimization state
            dt_max: Maximum allowed timestep
            n_min: Minimum number of steps before timestep increase
            f_inc: Factor for timestep increase
            f_dec: Factor for timestep decrease
            f_alpha: Factor for damping parameter decrease
            alpha_start: Initial value of damping parameter

        Returns:
            Updated FIREState after one optimization step
        """
        # Perform NVE step
        dt_curr = state.dt
        alpha_curr = state.alpha
        n_pos = state.n_pos

        state = velocity_verlet(state, dt_curr, model=model)

        state.dt = dt_curr
        state.alpha = alpha_curr
        state.n_pos = n_pos

        R, P, F, m = state.positions, state.momenta, state.forces, state.masses
        dt, alpha, n_pos = state.dt, state.alpha, state.n_pos

        # Calculate norms (adding small epsilon for numerical stability)
        F_norm = torch.sqrt(torch.sum(F**2, dtype=dtype) + eps)
        P_norm = torch.sqrt(torch.sum(P**2, dtype=dtype))

        # Calculate F dot P
        F_dot_P = torch.sum(F * P, dtype=dtype)

        # Update momentum using FIRE mixing
        P = P + alpha * (F * P_norm / F_norm - P)

        # Update parameters based on F dot P
        if F_dot_P > 0:  # Power is positive
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)  # Increase timestep but don't exceed dt_max
                alpha = alpha * f_alpha  # Decrease mixing parameter
        else:  # Power is negative
            n_pos = 0
            dt = dt * f_dec  # Decrease timestep
            alpha = alpha_start  # Reset mixing parameter
            P.zero_()  # Reset momentum to zero

        state.positions = R
        state.momenta = P
        state.forces = F
        state.masses = m
        state.dt = dt
        state.alpha = alpha
        state.n_pos = n_pos
        return state

    return fire_init, fire_update


def fire_ase(  # noqa: PLR0915
    *,
    model: torch.nn.Module,
    dt: float = 0.1,
    max_step: float = 0.2,
    dt_max: float = 1.0,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    downhill_check: bool = False,
) -> tuple[
    Callable[[BaseState | StateDict], FIREState], Callable[[FIREState], FIREState]
]:
    """Initialize a FIRE (Fast Inertial Relaxation Engine) optimization following
    ASE's implementation.

    FIRE is a molecular dynamics-based optimization algorithm that combines velocity
    Verlet integration with adaptive velocity damping. The key idea is to introduce
    an artificial "friction" that adapts based on the angle between the
    velocity and force.

    The algorithm works by:
    1. Performing standard velocity Verlet MD steps
    2. Calculating the power P = F x v (force dot velocity)
    3. If P > 0 (moving downhill):
       - Mixing velocity with normalized force: v = (1-a)v + a|v|F/|F|
       - If moving downhill for > N_min steps:
         * Increase timestep: dt = min(dt x f_inc, dt_max)
         * Decrease mixing: a = a x f_alpha
    4. If P ≤ 0 (moving uphill):
       - Reset velocity to zero
       - Decrease timestep: dt = dt x f_dec
       - Reset mixing parameter: a = alpha_start
    Args:
        model: Neural network model that computes energies and forces
        dt: Initial timestep (default: 0.1)
        max_step: Maximum allowed atomic displacement per step (default: 0.2)
        dt_max: Maximum allowed timestep (default: 1.0)
        n_min: Minimum steps before timestep increase (default: 5)
        f_inc: Factor for timestep increase (default: 1.1)
        f_dec: Factor for timestep decrease (default: 0.5)
        alpha_start: Initial mixing parameter (default: 0.1)
        f_alpha: Factor for mixing parameter decrease (default: 0.99)
        downhill_check: Whether to verify energy decreases each step (default: False)

    Returns:
        Tuple containing:
        - Initial FIREState with system state and optimization parameters
        - Update function that performs one FIRE step
    Notes:
        - The implementation closely follows ASE's FIRE optimizer
        - Downhill checking compares energies between steps rather than relying on P > 0
        - The max_step parameter prevents atoms from moving too far in a single step
        - The algorithm is particularly effective for atomic structure optimization
        - Parameters can be tuned for specific systems:
          * Larger dt_max allows faster convergence but may cause instability
          * Smaller max_step increases stability but slows convergence
          * n_min prevents premature timestep increases
          * f_inc and f_dec control how quickly the timestep adapts
          * alpha_start and f_alpha control the strength of velocity mixing
    References:
        - Bitzek et al., PRL 97, 170201 (2006) - Original FIRE paper
        - ASE implementation: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
    """
    device = model.device
    dtype = model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Convert scalar parameters to tensors
    params = [dt, dt_max, max_step, f_inc, f_dec, f_alpha, alpha_start]
    dt, dt_max, max_step, f_inc, f_dec, f_alpha, alpha_start = [
        p if isinstance(p, torch.Tensor) else torch.tensor(p, device=device, dtype=dtype)
        for p in params
    ]

    def fire_init(state: BaseState | StateDict, **kwargs) -> FIREState:
        """Initialize the FIRE optimizer state.

        Args:
            state: Initial system state
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial FIREState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get initial forces and energy from model
        model_output = model(state)

        # Initialize momenta as zeros
        momenta = torch.zeros_like(state.positions, device=device, dtype=dtype)

        return FIREState(
            positions=state.positions,
            forces=model_output["forces"],
            energy=model_output["energy"],
            masses=state.masses,
            momenta=momenta,
            atomic_numbers=atomic_numbers,
            cell=state.cell,
            pbc=state.pbc,
            dt=dt,
            alpha=alpha_start,
            n_pos=0,
        )

    def fire_step(state: FIREState) -> FIREState:
        """Perform one FIRE optimization step.

        This function implements the core FIRE algorithm update, which combines
        velocity Verlet integration with adaptive velocity mixing and timestep
        adjustment. The algorithm modifies atomic velocities based on the power
        (dot product of forces and velocities) to efficiently navigate the energy
        landscape.

        Args:
            state: Current optimization state containing positions, momenta, forces, etc.

        Returns:
            Updated FIREState after one optimization step
        """
        # Store previous state if doing downhill check
        if downhill_check:
            prev_positions = state.positions.clone()
            prev_momenta = state.momenta.clone()
            prev_energy = state.energy.clone()
        # Perform velocity Verlet step
        state = velocity_verlet(state, state.dt, model=model)
        # Get current velocities
        velocities = state.velocities
        # Calculate power (force dot velocity)
        power = torch.sum(state.forces * velocities, dtype=dtype)
        if downhill_check and state.energy > prev_energy:
            # Revert to previous state if energy increased
            state.positions = prev_positions
            state.momenta = prev_momenta
            # Recalculate forces and energy at reverted positions
            results = model(state)
            state.forces = results["forces"]
            state.energy = results["energy"]
            power = torch.tensor(
                -1.0, device=device, dtype=dtype
            )  # Force uphill response
        if power > 0:  # Moving downhill
            # Mix velocity with normalized force
            f_norm = torch.sqrt(torch.sum(state.forces**2, dtype=dtype) + eps)
            v_norm = torch.sqrt(torch.sum(velocities**2, dtype=dtype))
            velocities = (1.0 - state.alpha) * velocities + state.alpha * (
                state.forces / f_norm
            ) * v_norm
            # Update momenta from modified velocities
            state.momenta = velocities * state.masses.unsqueeze(-1)
            # Update timestep and mixing parameter if we've had enough good steps
            if state.n_pos > n_min:
                state.dt = torch.min(state.dt * f_inc, dt_max)
                state.alpha = state.alpha * f_alpha
            state.n_pos += 1
        else:  # Moving uphill
            state.momenta.zero_()  # Reset velocities
            state.dt = state.dt * f_dec
            state.alpha = alpha_start
            state.n_pos = 0
        # Limit maximum position change
        dr = state.dt * velocities
        dr_norm = torch.sqrt(torch.sum(dr**2, dim=-1, keepdim=True))
        mask = dr_norm > max_step
        if mask.any():
            # Broadcast the scaling factor to match dr's shape
            scale_factor = (max_step / dr_norm) * mask + (~mask).float()
            dr = dr * scale_factor
        state.positions = state.positions + dr
        # Update forces and energy at new positions
        results = model(state)
        state.forces = results["forces"]
        state.energy = results["energy"]
        return state

    return fire_init, fire_step


@dataclass
class UnitCellFIREState(OptimizerState):
    """State class for FIRE optimization with unit cell.
    Extends OptimizerState with additional variables needed for FIRE dynamics
    and unit cell optimization.

    Attributes:
        velocities: Atomic velocities tensor of shape (n_atoms, 3)
        dt: Current timestep
        alpha: Current mixing parameter
        n_pos: Number of consecutive steps with positive power
        orig_cell: Original unit cell tensor of shape (3, 3)
        cell_factor: Scaling factor for cell optimization
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        pressure: Applied pressure tensor
        stress: Stress tensor of shape (3, 3)
        cell_positions: Cell positions tensor of shape (3, 3)
        cell_velocities: Cell velocities tensor of shape (3, 3)
        cell_forces: Cell forces tensor of shape (3, 3)
        cell_masses: Cell masses tensor of shape (3)
    """

    # Required attributes not in OptimizerState
    velocities: torch.Tensor
    stress: torch.Tensor

    # optimization-specific attributes
    orig_cell: torch.Tensor
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool

    # cell attributes
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor

    @property
    def momenta(self) -> torch.Tensor:
        """Calculate momenta from velocities and masses."""
        return self.velocities * self.masses.unsqueeze(-1)


def unit_cell_fire(  # noqa: PLR0915, C901
    model: torch.nn.Module,
    dt_max: float = 0.4,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    hydrostatic_strain: bool = False,  # noqa: FBT001, FBT002
    constant_volume: bool = False,  # noqa: FBT001, FBT002
    scalar_pressure: float = 0.0,
    cell_factor: float | None = None,
) -> tuple[
    Callable[[BaseState | StateDict], UnitCellFIREState],
    Callable[[UnitCellFIREState], UnitCellFIREState],
]:
    """Initialize a FIRE optimization with unit cell.

    Args:
        model: Neural network model that computes energies and forces
        dt_max: Maximum allowed timestep (default: 0.4)
        dt_start: Initial timestep (default: 0.1)
        n_min: Minimum steps before timestep increase (default: 5)
        f_inc: Factor for timestep increase (default: 1.1)
        f_dec: Factor for timestep decrease (default: 0.5)
        alpha_start: Initial mixing parameter (default: 0.1)
        f_alpha: Factor for mixing parameter decrease (default: 0.99)
        hydrostatic_strain: Whether to only allow hydrostatic deformation (default: False)
        constant_volume: Whether to maintain constant volume (default: False)
        scalar_pressure: Applied pressure in GPa (default: 0.0)
        cell_factor: Scaling factor for cell optimization (default: number of atoms)

    Returns:
        Tuple containing:
        - Initialization function that creates a UnitCellFIREState
        - Update function that performs one FIRE step
    """
    device = model.device
    dtype = model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    params = [dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min]
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min = [
        p if isinstance(p, torch.Tensor) else torch.tensor(p, device=device, dtype=dtype)
        for p in params
    ]

    def fire_init(
        state: BaseState | StateDict,
        cell_factor: torch.Tensor | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        **kwargs,
    ) -> UnitCellFIREState:
        """Initialize the FIRE optimization state.

        Args:
            state: Initial system state containing positions, masses, cell, etc.
            cell_factor: Scaling factor for cell optimization (default: number of atoms)
            scalar_pressure: External pressure tensor (default: 0.0)
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial UnitCellFIREState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Setup cell factor
        if cell_factor is None:
            cell_factor = float(len(state.positions))
        if isinstance(cell_factor, (int, float)):
            cell_factor = torch.full((1, 1), cell_factor, device=device, dtype=dtype)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)

        # Get initial forces and energy from model
        results = model(state)
        forces = results["forces"]
        energy = results["energy"]
        stress = results["stress"]

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.cell, state.cell), 0, 1
        )  # Identity matrix

        # Calculate cell positions
        cell_positions = (cur_deform_grad * cell_factor).reshape(3, 3)

        # Calculate virial
        volume = torch.linalg.det(state.cell).view(1, 1)
        virial = -volume * stress + pressure

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = diag_mean * torch.eye(3, device=device)

        if constant_volume:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = virial - diag_mean * torch.eye(3, device=device)

        virial = virial / cell_factor
        cell_forces = virial

        # Create cell masses
        cell_masses = torch.full((3,), state.masses.sum(), device=device, dtype=dtype)

        return UnitCellFIREState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            stress=stress,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            velocities=torch.zeros_like(forces),
            dt=dt_start,
            alpha=alpha_start,
            n_pos=0,
            orig_cell=state.cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
            atomic_numbers=atomic_numbers,
            cell_positions=cell_positions,
            cell_velocities=torch.zeros_like(cell_positions),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
        )

    def fire_step(  # noqa: PLR0915
        state: UnitCellFIREState,
    ) -> UnitCellFIREState:
        """Perform one FIRE optimization step.

        This function implements a single step of the FIRE optimization algorithm
        for systems with variable unit cell. It handles atomic positions and cell
        vectors separately while following the same FIRE algorithm logic.

        Args:
            state: Current UnitCellFIREState containing positions, cell, forces, etc.

        Returns:
            Updated UnitCellFIREState after one optimization step
        """
        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.orig_cell, state.cell), 0, 1
        )

        # Calculate cell positions from deformation gradient
        cell_positions = (cur_deform_grad * state.cell_factor).reshape(3, 3)

        # Velocity Verlet first half step
        state.velocities += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * state.dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Update positions
        atomic_positions_new = state.positions + state.dt * state.velocities
        cell_positions_new = cell_positions + state.dt * state.cell_velocities

        # Update cell
        cell_update = (cell_positions_new / state.cell_factor).reshape(3, 3)
        new_cell = torch.mm(state.orig_cell, cell_update.t())

        # Get new forces and energy
        state.positions = atomic_positions_new
        state.cell = new_cell
        results = model(state)

        atomic_forces = results["forces"]
        energy = results["energy"]
        stress = results["stress"]

        # Update state
        state.positions = atomic_positions_new
        state.cell = new_cell
        state.stress = stress
        state.energy = energy
        state.forces = atomic_forces
        state.cell_positions = cell_positions_new

        # Calculate virial for cell forces
        volume = torch.linalg.det(new_cell).view(1, 1)
        virial = -volume * stress + state.pressure

        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = diag_mean * torch.eye(3, device=device)

        if state.constant_volume:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = virial - diag_mean * torch.eye(3, device=device)

        virial = virial / state.cell_factor
        state.cell_forces = virial

        # Velocity Verlet second half step
        state.velocities += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * state.dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Calculate power (F·V) for atoms and cell
        atomic_power = torch.sum(state.forces * state.velocities)
        cell_power = torch.sum(state.cell_forces * state.cell_velocities)
        total_power = atomic_power + cell_power

        # FIRE updates
        if total_power > 0:
            state.n_pos += 1
            if state.n_pos > n_min:
                state.dt = torch.min(state.dt * f_inc, dt_max)
                state.alpha = state.alpha * f_alpha
        else:
            state.n_pos = 0
            state.dt = state.dt * f_dec
            state.alpha = alpha_start
            state.velocities.zero_()
            state.cell_velocities.zero_()

        # Mix velocity and force direction for atoms
        v_norm = torch.norm(state.velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        state.velocities = (
            1.0 - state.alpha
        ) * state.velocities + state.alpha * state.forces * v_norm / (f_norm + eps)

        # Mix velocity and force direction for cell
        cell_v_norm = torch.norm(state.cell_velocities, dim=1, keepdim=True)
        cell_f_norm = torch.norm(state.cell_forces, dim=1, keepdim=True)
        state.cell_velocities = (
            1.0 - state.alpha
        ) * state.cell_velocities + state.alpha * state.cell_forces * cell_v_norm / (
            cell_f_norm + eps
        )

        return state

    return fire_init, fire_step


@dataclass
class FrechetCellFIREState(OptimizerState):
    """State class for FIRE optimization with Frechet cell derivatives.

    Extends OptimizerState with additional variables needed for FIRE dynamics
    and unit cell optimization using matrix logarithm for cell parameterization.

    Attributes:
        velocities: Atomic velocities tensor of shape (n_atoms, 3)
        stress: Stress tensor of shape (3, 3)

        # optimization-specific attributes
        orig_cell: Original unit cell tensor of shape (3, 3)
        cell_factor: Scaling factor for cell optimization
        pressure: Applied pressure tensor
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume

        # cell attributes
        cell_positions: Cell positions tensor of shape (3, 3)
        cell_velocities: Cell velocities tensor of shape (3, 3)
        cell_forces: Cell forces tensor of shape (3, 3)
        cell_masses: Cell masses tensor of shape (3)

        # FIRE algorithm parameters
        dt: Current timestep
        alpha: Current mixing parameter
        n_pos: Number of consecutive steps with positive power
    """

    # Required attributes not in OptimizerState
    velocities: torch.Tensor
    stress: torch.Tensor

    # optimization-specific attributes
    orig_cell: torch.Tensor
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool

    # cell attributes
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: int

    @property
    def momenta(self) -> torch.Tensor:
        """Calculate momenta from velocities and masses."""
        return self.velocities * self.masses.unsqueeze(-1)

    def deform_grad(self) -> torch.Tensor:
        """Calculate the deformation gradient from original cell to current cell."""
        return torch.transpose(torch.linalg.solve(self.orig_cell, self.cell), 0, 1)


def frechet_cell_fire(  # noqa: PLR0915, C901
    *,
    model: torch.nn.Module,
    dt_max: float = 0.4,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
    cell_factor: float | None = None,
) -> tuple[
    Callable[[BaseState | StateDict], FrechetCellFIREState],
    Callable[[FrechetCellFIREState], FrechetCellFIREState],
]:
    """Initialize a FIRE optimization with Frechet cell parameterization.

    This implementation uses matrix logarithm to parameterize the cell degrees of freedom,
    providing forces consistent with numerical derivatives of the potential energy
    with respect to the cell variables.

    Args:
        model: Neural network model that computes energies and forces
        dt_max: Maximum allowed timestep (default: 0.4)
        dt_start: Initial timestep (default: 0.1)
        n_min: Minimum steps before timestep increase (default: 5)
        f_inc: Factor for timestep increase (default: 1.1)
        f_dec: Factor for timestep decrease (default: 0.5)
        alpha_start: Initial mixing parameter (default: 0.1)
        f_alpha: Factor for mixing parameter decrease (default: 0.99)
        hydrostatic_strain: Whether to only allow hydrostatic deformation (default: False)
        constant_volume: Whether to maintain constant volume (default: False)
        scalar_pressure: Applied pressure in GPa (default: 0.0)
        cell_factor: Scaling factor for cell optimization (default: number of atoms)

    Returns:
        Tuple containing:
        - Initialization function that creates a FrechetCellFIREState
        - Update function that performs one FIRE step with Frechet derivatives

    References:
        - https://github.com/lan496/lan496.github.io/blob/main/notes/cell_grad.pdf
        - https://github.com/JuliaMolSim/JuLIP.jl/blob/master/src/expcell.jl
    """
    device = model.device
    dtype = model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    params = [dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha]
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha = [
        p if isinstance(p, torch.Tensor) else torch.tensor(p, device=device, dtype=dtype)
        for p in params
    ]

    def fire_init(
        state: BaseState | StateDict,
        cell_factor: float | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        **kwargs,
    ) -> FrechetCellFIREState:
        """Initialize the FIRE optimization state with Frechet cell parameterization.

        Args:
            state: Initial system state containing positions, masses, cell, etc.
            cell_factor: Scaling factor for cell optimization (default: number of atoms)
            scalar_pressure: External pressure in GPa (default: 0.0)
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial FrechetCellFIREState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Setup cell factor
        if cell_factor is None:
            cell_factor = float(len(state.positions))
        if isinstance(cell_factor, (int, float)):
            cell_factor = torch.tensor(cell_factor, device=device, dtype=dtype)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)

        # Get initial forces and energy from model
        results = model(state)
        forces = results["forces"]
        energy = results["energy"]
        stress = results["stress"]

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.cell, state.cell), 0, 1
        )  # Identity matrix

        # Calculate cell positions using log parameterization
        # For identity matrix, logm gives zero matrix, so we multiply by cell_factor
        cell_positions = logm(cur_deform_grad, sim_dtype=dtype) * cell_factor

        # Calculate virial
        volume = torch.linalg.det(state.cell).view(1, 1)
        virial = -volume * stress + pressure

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = diag_mean * torch.eye(3, device=device, dtype=dtype)

        if constant_volume:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = virial - diag_mean * torch.eye(3, device=device, dtype=dtype)

        # Calculate initial cell forces (simplified for identity matrix case)
        ucf_cell_grad = virial @ torch.linalg.inv(cur_deform_grad.transpose(0, 1))
        cell_forces = ucf_cell_grad / cell_factor

        # Create cell masses
        cell_masses = torch.full((3,), state.masses.sum(), device=device, dtype=dtype)

        return FrechetCellFIREState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            stress=stress,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            velocities=torch.zeros_like(forces),
            dt=dt_start,
            alpha=alpha_start,
            n_pos=0,
            orig_cell=state.cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
            atomic_numbers=atomic_numbers,
            cell_positions=cell_positions,
            cell_velocities=torch.zeros_like(cell_positions),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
        )

    def fire_step(  # noqa: PLR0915
        state: FrechetCellFIREState,
    ) -> FrechetCellFIREState:
        """Perform one FIRE optimization step with Frechet cell parameterization.

        This function implements a single step of the FIRE optimization algorithm
        for systems with variable unit cell using the Frechet derivative approach.
        It handles atomic positions and cell vectors separately while following
        the same FIRE algorithm logic.

        Args:
            state: Current FrechetCellFIREState containing positions, cell, forces, etc.

        Returns:
            Updated FrechetCellFIREState after one optimization step
        """
        # Get current deformation gradient
        cur_deform_grad = state.deform_grad()

        # Get log of deformation gradient
        cur_deform_grad_log = logm(cur_deform_grad, sim_dtype=dtype)

        # Scale to get cell positions
        cell_positions = cur_deform_grad_log * state.cell_factor

        # Velocity Verlet first half step
        state.velocities += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * state.dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Update positions
        atomic_positions_new = state.positions + state.dt * state.velocities
        cell_positions_new = cell_positions + state.dt * state.cell_velocities

        # Convert cell positions to deformation gradient
        deform_grad_log_new = cell_positions_new
        deform_grad_new = expm.apply(deform_grad_log_new / state.cell_factor)

        # Update cell
        new_cell = torch.mm(state.orig_cell, deform_grad_new.transpose(0, 1))

        # Get new forces and energy
        state.positions = atomic_positions_new
        state.cell = new_cell
        results = model(state)

        atomic_forces = results["forces"]
        energy = results["energy"]
        stress = results["stress"]

        # Calculate virial for cell forces
        volume = torch.linalg.det(new_cell).view(1, 1)
        virial = -volume * stress + state.pressure

        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = diag_mean * torch.eye(3, device=device, dtype=dtype)

        if state.constant_volume:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = virial - diag_mean * torch.eye(3, device=device, dtype=dtype)

        # Calculate UCF-style cell gradient
        ucf_cell_grad = virial @ torch.linalg.inv(deform_grad_new.transpose(0, 1))

        # Calculate cell forces using Frechet derivative approach
        cell_forces = torch.zeros((3, 3), device=device, dtype=dtype)
        for mu in range(3):
            for nu in range(3):
                # Create directional derivative
                direction = torch.zeros((3, 3), device=device, dtype=dtype)
                direction[mu, nu] = 1.0

                # Calculate Frechet derivative
                expm_deriv = expm_frechet(
                    deform_grad_log_new, direction, compute_expm=False
                )

                # Sum the element-wise product
                cell_forces[mu, nu] = torch.sum(expm_deriv * ucf_cell_grad)

        # Scale by cell_factor
        cell_forces = cell_forces / state.cell_factor

        # Update state
        state.positions = atomic_positions_new
        state.cell = new_cell
        state.stress = stress
        state.energy = energy
        state.forces = atomic_forces
        state.cell_positions = cell_positions_new
        state.cell_forces = cell_forces

        # Velocity Verlet second half step
        state.velocities += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * state.dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Calculate power (F·V) for atoms and cell
        atomic_power = torch.sum(state.forces * state.velocities)
        cell_power = torch.sum(state.cell_forces * state.cell_velocities)
        total_power = atomic_power + cell_power

        # FIRE updates
        if total_power > 0:
            state.n_pos += 1
            if state.n_pos > n_min:
                state.dt = torch.min(state.dt * f_inc, dt_max)
                state.alpha = state.alpha * f_alpha
        else:
            state.n_pos = 0
            state.dt = state.dt * f_dec
            state.alpha = alpha_start
            state.velocities.zero_()
            state.cell_velocities.zero_()

        # Mix velocity and force direction for atoms
        v_norm = torch.norm(state.velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        state.velocities = (1.0 - state.alpha) * state.velocities
        state.velocities += state.alpha * state.forces * v_norm / (f_norm + eps)

        # Mix velocity and force direction for cell
        cell_v_norm = torch.norm(state.cell_velocities, dim=1, keepdim=True)
        cell_f_norm = torch.norm(state.cell_forces, dim=1, keepdim=True)
        state.cell_velocities = (1.0 - state.alpha) * state.cell_velocities
        state.cell_velocities += (
            state.alpha * state.cell_forces * cell_v_norm / (cell_f_norm + eps)
        )

        return state

    return fire_init, fire_step
