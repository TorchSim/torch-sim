"""Optimizers for structure optimization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from torchsim.state import BaseState
from torchsim.unbatched_integrators import velocity_verlet


StateDict = dict[
    Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "batch"], torch.Tensor
]

eps = 1e-8


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
    """State class for gradient descent optimization.

    Extends OptimizerState with learning rate parameter.

    Attributes:
        lr: Learning rate for position updates
    """


def gradient_descent(
    *,
    model: torch.nn.Module,
    lr: float = 0.01,
) -> tuple[Callable[[StateDict], GDState], Callable[[GDState], GDState]]:
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

    def gd_init(state: BaseState | StateDict, **extra_state_kwargs) -> GDState:
        """Initialize the gradient descent optimizer state.

        Args:
            state: Initial system state
            **extra_state_kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial GDState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = extra_state_kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get initial forces and energy from model
        model_output = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=atomic_numbers,
        )

        return GDState(
            positions=state.positions,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=state.atomic_numbers,
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
        results = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
        )
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


def fire(  # noqa: PLR0915, C901
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
    """
    device = model.device
    dtype = model.dtype

    # parameters to be set in fire_update
    if not isinstance(dt_max, torch.Tensor):
        dt_max = torch.tensor(dt_max, device=device, dtype=dtype)
    if not isinstance(n_min, torch.Tensor):
        n_min = torch.tensor(n_min, device=device, dtype=dtype)
    if not isinstance(f_inc, torch.Tensor):
        f_inc = torch.tensor(f_inc, device=device, dtype=dtype)
    if not isinstance(f_dec, torch.Tensor):
        f_dec = torch.tensor(f_dec, device=device, dtype=dtype)
    if not isinstance(f_alpha, torch.Tensor):
        f_alpha = torch.tensor(f_alpha, device=device, dtype=dtype)
    if not isinstance(dt_start, torch.Tensor):
        dt_start = torch.tensor(dt_start, device=device, dtype=dtype)
    if not isinstance(alpha_start, torch.Tensor):
        alpha_start = torch.tensor(alpha_start, device=device, dtype=dtype)

    def fire_init(state: BaseState | StateDict, **extra_state_kwargs) -> FIREState:
        """Initialize the FIRE optimizer state.

        Args:
            state: Initial system state
            **extra_state_kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial FIREState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = extra_state_kwargs.get("atomic_numbers", state.atomic_numbers)
        # Get initial forces and energy from model
        model_output = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=atomic_numbers,
        )
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
    positions: torch.Tensor,
    masses: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    dt: float = 0.1,
    max_step: float = 0.2,
    dt_max: float = 1.0,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    downhill_check: bool = False,
    **extra_state_kwargs,
) -> tuple[FIREState, Callable[[FIREState], FIREState]]:
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
        positions: Atomic positions tensor of shape (n_atoms, 3)
        masses: Atomic masses tensor of shape (n_atoms,)
        cell: Unit cell tensor of shape (3, 3)
        pbc: Periodic boundary conditions flags
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
    device = positions.device
    dtype = positions.dtype
    # Convert scalar parameters to tensors
    dt = torch.tensor(dt, device=device, dtype=dtype)
    dt_max = torch.tensor(dt_max, device=device, dtype=dtype)
    max_step = torch.tensor(max_step, device=device, dtype=dtype)
    f_inc = torch.tensor(f_inc, device=device, dtype=dtype)
    f_dec = torch.tensor(f_dec, device=device, dtype=dtype)
    f_alpha = torch.tensor(f_alpha, device=device, dtype=dtype)
    alpha_start = torch.tensor(alpha_start, device=device, dtype=dtype)

    def fire_step(state: FIREState) -> FIREState:
        """Perform one FIRE optimization step."""
        # Store previous state if doing downhill check
        if downhill_check:
            prev_positions = state.positions.clone()
            prev_momenta = state.momenta.clone()
            prev_energy = state.energy.clone()
        # Perform velocity Verlet step
        state = velocity_verlet(state, state.dt, model=model)
        # Get updated energy after velocity Verlet step
        results = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
        )
        state.energy = results["energy"]
        state.forces = results["forces"]
        # Get current velocities
        velocities = state.velocities
        # Calculate power (force dot velocity)
        power = torch.sum(state.forces * velocities, dtype=dtype)
        if downhill_check and state.energy > prev_energy:
            # Revert to previous state if energy increased
            state.positions = prev_positions
            state.momenta = prev_momenta
            # Recalculate forces and energy at reverted positions
            results = model(
                positions=prev_positions,
                cell=state.cell,
                atomic_numbers=state.atomic_numbers,
            )
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
        results = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
        )
        state.forces = results["forces"]
        state.energy = results["energy"]
        return state

    model_output = model(
        positions=positions,
        cell=cell,
        atomic_numbers=extra_state_kwargs.get("atomic_numbers"),
    )

    momenta = torch.zeros_like(positions, device=device, dtype=dtype)

    initial_state = FIREState(
        positions=positions,
        forces=model_output["forces"],
        energy=model_output["energy"],
        masses=masses,
        momenta=momenta,
        atomic_numbers=extra_state_kwargs.get("atomic_numbers"),
        cell=cell,
        pbc=pbc,
        dt=dt_max,
        alpha=alpha_start,
        n_pos=0,
    )

    return initial_state, fire_step


@dataclass
class UnitCellFIREState(OptimizerState):
    """State class for FIRE optimization with unit cell.
    Extends OptimizerState with additional variables needed for FIRE dynamics
    and unit cell optimization.

    Attributes:
        momenta: Atomic momenta tensor of shape (n_atoms + 3, 3)
        dt: Current timestep
        alpha: Current mixing parameter
        n_pos: Number of consecutive steps with positive power
        orig_cell: Original unit cell tensor of shape (3, 3)
        cell_factor: Scaling factor for cell optimization
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        pressure: Applied pressure tensor
        stress: Stress tensor of shape (3, 3)
    """

    momenta: torch.Tensor
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: int
    orig_cell: torch.Tensor
    cell_factor: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool
    pressure: torch.Tensor
    stress: torch.Tensor

    @property
    def velocities(self) -> torch.Tensor:
        """Calculate velocities from momenta and masses."""
        return self.momenta / self.masses.unsqueeze(-1)


def unit_cell_fire(  # noqa: PLR0915
    model: torch.nn.Module,
    positions: torch.Tensor,
    masses: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,  # noqa: FBT001
    dt_max: float = 0.4,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | None = None,
    hydrostatic_strain: bool = False,  # noqa: FBT001, FBT002
    constant_volume: bool = False,  # noqa: FBT001, FBT002
    scalar_pressure: float = 0.0,
    **extra_state_kwargs,
) -> tuple[UnitCellFIREState, Callable[[UnitCellFIREState], UnitCellFIREState]]:
    """Initialize a FIRE optimization with unit cell.

    Args:
        model: Neural network model that computes energies and forces
        positions: Atomic positions tensor of shape (n_atoms, 3)
        masses: Atomic masses tensor of shape (n_atoms,)
        cell: Unit cell tensor of shape (3, 3)
        pbc: Whether to use periodic boundary conditions
        dt_max: Maximum allowed timestep (default: 0.4)
        dt_start: Initial timestep (default: 0.1)
        n_min: Minimum steps before timestep increase (default: 5)
        f_inc: Factor for timestep increase (default: 1.1)
        f_dec: Factor for timestep decrease (default: 0.5)
        alpha_start: Initial mixing parameter (default: 0.1)
        f_alpha: Factor for mixing parameter decrease (default: 0.99)
        cell_factor: Scaling factor for cell optimization (default: n_atoms)
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        scalar_pressure: Applied pressure in GPa (default: 0.0)
        **extra_state_kwargs: Additional keyword arguments for state
    Returns:
        Tuple containing:
        - Initial UnitCellFIREState with system state
        - Update function that performs one FIRE step
    """
    device = positions.device
    dtype = positions.dtype

    # Setup parameters
    dt_max = torch.tensor(dt_max, device=device, dtype=dtype)
    dt_start = torch.tensor(dt_start, device=device, dtype=dtype)
    alpha_start = torch.tensor(alpha_start, device=device, dtype=dtype)

    # Setup cell factor
    if cell_factor is None:
        cell_factor = float(len(positions))
    cell_factor = torch.full((1, 1), cell_factor, device=device, dtype=dtype)

    # Setup pressure tensor
    pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)

    def initialize_state(
        positions: torch.Tensor,
        masses: torch.Tensor,
        cell: torch.Tensor,
        pbc: bool,  # noqa: FBT001
    ) -> UnitCellFIREState:
        """Initialize the FIRE optimization state."""
        # Get initial forces and energy from model
        results = model(
            positions=positions,
            cell=cell,
            atomic_numbers=extra_state_kwargs.get("atomic_numbers"),
        )
        forces = results["forces"]
        energy = results["energy"]
        stress = results["stress"]

        # Total number of DOFs (atoms + cell)
        n_atoms = len(positions)
        total_dofs = n_atoms + 3

        # Combine atomic forces and cell forces
        forces_combined = torch.zeros((total_dofs, 3), device=device, dtype=dtype)
        forces_combined[:n_atoms] = forces

        # Cell forces (from stress)
        volume = torch.linalg.det(cell).view(1, 1)
        virial = -volume * stress + pressure

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = diag_mean * torch.eye(3, device=device)

        if constant_volume:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = virial - diag_mean * torch.eye(3, device=device)

        virial = virial / cell_factor
        forces_combined[n_atoms:] = virial.reshape(-1, 3)

        # Initialize masses for cell degrees of freedom
        masses_combined = torch.zeros(total_dofs, device=device, dtype=dtype)
        masses_combined[:n_atoms] = masses
        masses_combined[n_atoms:] = masses.sum()  # Use total mass for cell DOFs

        return UnitCellFIREState(
            positions=positions,
            forces=forces_combined,
            energy=energy,
            stress=stress,
            masses=masses_combined,
            cell=cell,
            pbc=pbc,
            momenta=torch.zeros_like(forces_combined),
            dt=dt_start,
            alpha=alpha_start,
            n_pos=0,
            orig_cell=cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
            **extra_state_kwargs,
        )

    def fire_step(state: UnitCellFIREState) -> UnitCellFIREState:
        """Perform one FIRE optimization step."""
        n_atoms = len(state.positions)

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.orig_cell, state.cell), 0, 1
        )

        # Split positions and forces
        atomic_positions = state.positions
        cell_positions = (cur_deform_grad * state.cell_factor).reshape(-1, 3)

        # Velocity Verlet first half step
        velocities = state.velocities
        velocities += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)

        # Update positions
        atomic_positions_new = atomic_positions + state.dt * velocities[:n_atoms]
        cell_positions_new = cell_positions + state.dt * velocities[n_atoms:]

        # Update cell
        cell_update = (cell_positions_new / state.cell_factor).reshape(3, 3)
        new_cell = torch.mm(state.orig_cell, cell_update.t())

        # Get new forces and energy
        results = model(
            positions=atomic_positions_new,
            cell=new_cell,
            atomic_numbers=extra_state_kwargs.get("atomic_numbers"),
        )

        forces = results["forces"]
        energy = results["energy"]
        stress = results["stress"]

        # Update state
        state.positions = atomic_positions_new
        state.cell = new_cell
        state.stress = stress
        state.energy = energy

        # Combine forces
        forces_combined = torch.zeros_like(state.forces)
        forces_combined[:n_atoms] = forces

        # Calculate virial
        volume = torch.linalg.det(new_cell).view(1, 1)
        virial = -volume * stress + state.pressure

        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = diag_mean * torch.eye(3, device=device)

        if state.constant_volume:
            diag_mean = torch.diagonal(virial).mean().view(1, 1)
            virial = virial - diag_mean * torch.eye(3, device=device)

        virial = virial / state.cell_factor
        forces_combined[n_atoms:] = virial.reshape(-1, 3)
        state.forces = forces_combined

        # Velocity Verlet second half step
        velocities += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)
        state.momenta = velocities * state.masses.unsqueeze(-1)

        # Calculate power
        power = torch.sum(state.forces * velocities)

        # FIRE updates
        if power > 0:
            state.n_pos += 1
            if state.n_pos > n_min:
                state.dt = torch.min(state.dt * f_inc, dt_max)
                state.alpha = state.alpha * f_alpha
        else:
            state.n_pos = 0
            state.dt = state.dt * f_dec
            state.alpha = alpha_start
            state.momenta.zero_()
            velocities.zero_()

        # Mix velocity and force direction
        v_norm = torch.norm(velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        velocities = (
            1.0 - state.alpha
        ) * velocities + state.alpha * state.forces * v_norm / (f_norm + 1e-10)
        state.momenta = velocities * state.masses.unsqueeze(-1)

        return state

    initial_state = initialize_state(
        positions=positions, masses=masses, cell=cell, pbc=pbc
    )
    return initial_state, fire_step
