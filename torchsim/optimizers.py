"""Batched optimizers for structure optimization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from torchsim.state import BaseState
from torchsim.unbatched_optimizers import OptimizerState


StateDict = dict[
    Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "batch"],
    torch.Tensor,
]


@dataclass
class BatchedGDState(OptimizerState):
    """State class for batched gradient descent optimization."""


def gradient_descent(
    *,
    model: torch.nn.Module,
    lr: torch.Tensor | float = 0.01,
) -> tuple[
    Callable[[StateDict | BaseState], BatchedGDState],
    Callable[[BatchedGDState], BatchedGDState],
]:
    """Initialize a batched gradient descent optimization.

    Args:
        model: Neural network model that computes energies and forces
        lr: Learning rate(s) for optimization. Can be a single float applied to all
            batches or a tensor with shape [n_batches] for batch-specific rates

    Returns:
        Tuple containing:
        - Initialization function that creates the initial BatchedGDState
        - Update function that performs one gradient descent step
    """
    device = model.device
    dtype = model.dtype

    def gd_init(
        state: BaseState | StateDict,
        **kwargs: Any,
    ) -> BatchedGDState:
        """Initialize the batched gradient descent optimization state.

        Args:
            state: Base state containing positions, masses, cell, etc.
            kwargs: Additional keyword arguments to override state attributes

        Returns:
            Initialized BatchedGDState with forces and energy
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get initial forces and energy from model
        model_output = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=atomic_numbers,
            batch=state.batch,
        )
        energy = model_output["energy"]
        forces = model_output["forces"]

        return BatchedGDState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            batch=state.batch,
        )

    def gd_step(state: BatchedGDState, lr: torch.Tensor = lr) -> BatchedGDState:
        """Perform one gradient descent optimization step.

        Args:
            state: Current optimization state
            lr: Learning rate(s) to use for this step, overriding the default

        Returns:
            Updated BatchedGDState after one optimization step
        """
        # Get per-atom learning rates by mapping batch learning rates to atoms
        if isinstance(lr, float):
            lr = torch.full((state.n_batches,), lr, device=device, dtype=dtype)

        atom_lr = lr[state.batch].unsqueeze(-1)  # shape: (total_atoms, 1)

        # Update positions using forces and per-atom learning rates
        state.positions = state.positions + atom_lr * state.forces

        # Get updated forces and energy from model
        model_output = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            batch=state.batch,
        )

        # Update state with new forces and energy
        state.forces = model_output["forces"]
        state.energy = model_output["energy"]

        return state

    return gd_init, gd_step


@dataclass
class BatchedUnitCellGDState(BatchedGDState):
    """State class for batched gradient descent optimization with unit cell.

    Extends BatchedGDState with unit cell optimization parameters.

    Attributes:
        stress: Stress tensor of shape (n_batches, 3, 3)
        reference_cell: Reference unit cells tensor of shape (n_batches, 3, 3)
        cell_factor: Scaling factor for cell optimization
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        pressure: Applied pressure tensor
        cell_positions: Cell positions tensor of shape (n_batches, 3, 3)
        cell_forces: Cell forces tensor of shape (n_batches, 3, 3)
        cell_masses: Cell masses tensor of shape (n_batches, 3)
    """

    reference_cell: torch.Tensor
    cell_factor: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool
    pressure: torch.Tensor
    stress: torch.Tensor

    cell_positions: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor


def unit_cell_gradient_descent(  # noqa: PLR0915, C901
    model: torch.nn.Module,
    *,
    positions_lr: float = 0.01,
    cell_lr: float = 0.1,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    Callable[[BaseState | StateDict], BatchedUnitCellGDState],
    Callable[[BatchedUnitCellGDState], BatchedUnitCellGDState],
]:
    """Initialize a batched gradient descent optimization with unit cell.

    This optimizer extends the standard gradient descent to also optimize the unit cell
    parameters along with atomic positions. It supports hydrostatic strain constraints,
    constant volume constraints, and external pressure.

    Args:
        model: Neural network model that computes energies, forces, and stress
        positions_lr: Learning rate for atomic positions optimization
        cell_lr: Learning rate for unit cell optimization
        cell_factor: Scaling factor for cell optimization (default: number of atoms)
        hydrostatic_strain: Whether to only allow hydrostatic deformation (default: False)
        constant_volume: Whether to maintain constant volume (default: False)
        scalar_pressure: Applied pressure in GPa (default: 0.0)

    Returns:
        Tuple containing:
        - Initialization function that creates a BatchedUnitCellGDState
        - Update function that performs one gradient descent step with cell optimization
    """
    device = model.device
    dtype = model.dtype

    def gd_init(
        state: BaseState,
        cell_factor: float | torch.Tensor | None = cell_factor,
        hydrostatic_strain: bool = hydrostatic_strain,  # noqa: FBT001
        constant_volume: bool = constant_volume,  # noqa: FBT001
        scalar_pressure: float = scalar_pressure,
        **kwargs: Any,
    ) -> BatchedUnitCellGDState:
        """Initialize the batched gradient descent optimization state with unit cell.

        Args:
            state: Initial system state containing positions, masses, cell, etc.
            cell_factor: Scaling factor for cell optimization (default: number of atoms)
            hydrostatic_strain: Whether to only allow hydrostatic deformation
            constant_volume: Whether to maintain constant volume
            scalar_pressure: Applied pressure in GPa
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial BatchedUnitCellGDState with system configuration and forces
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per batch
            _, counts = torch.unique(state.batch, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all batches
            cell_factor = torch.full(
                (state.n_batches,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_batches, 1, 1) for broadcasting
        cell_factor = cell_factor.view(-1, 1, 1)

        scalar_pressure = torch.full(
            (state.n_batches, 1, 1), scalar_pressure, device=device, dtype=dtype
        )
        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device)

        # Get initial forces and energy from model
        model_output = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=atomic_numbers,
            batch=state.batch,
        )
        energy = model_output["energy"]
        forces = model_output["forces"]
        stress = model_output["stress"]  # Already shape: (n_batches, 3, 3)

        # Create cell masses
        cell_masses = torch.ones(
            (state.n_batches, 3), device=device, dtype=dtype
        )  # One mass per cell DOF

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.cell, state.cell), 1, 2
        )  # Identity matrix shape: (n_batches, 3, 3)

        # Calculate cell positions
        cell_factor_expanded = cell_factor.expand(
            state.n_batches, 3, 1
        )  # shape: (n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(state.n_batches, 3, 3) * cell_factor_expanded
        )  # shape: (n_batches, 3, 3)

        # Calculate virial
        volumes = torch.linalg.det(state.cell).view(-1, 1, 1)
        virial = -volumes * stress + pressure

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(state.n_batches, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(state.n_batches, -1, -1)

        # Scale virial by cell_factor
        virial = virial / cell_factor

        # Reshape virial for cell forces
        cell_forces = virial  # shape: (n_batches, 3, 3)

        return BatchedUnitCellGDState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            stress=stress,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
            atomic_numbers=atomic_numbers,
            batch=state.batch,
            cell_positions=cell_positions,
            cell_forces=cell_forces,
            cell_masses=cell_masses,
        )

    def gd_step(
        state: BatchedUnitCellGDState,
        positions_lr: torch.Tensor = positions_lr,
        cell_lr: torch.Tensor = cell_lr,
    ) -> BatchedUnitCellGDState:
        """Perform one gradient descent optimization step with unit cell.

        Updates both atomic positions and cell parameters based on forces and stress.

        Args:
            state: Current optimization state
            positions_lr: Learning rate for atomic positions optimization
            cell_lr: Learning rate for unit cell optimization

        Returns:
            Updated BatchedUnitCellGDState after one optimization step
        """
        # Get dimensions
        n_batches = state.n_batches

        # Get per-atom learning rates by mapping batch learning rates to atoms
        if isinstance(positions_lr, float):
            positions_lr = torch.full(
                (state.n_batches,), positions_lr, device=device, dtype=dtype
            )

        if isinstance(cell_lr, float):
            cell_lr = torch.full((state.n_batches,), cell_lr, device=device, dtype=dtype)

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.reference_cell, state.cell), 1, 2
        )  # shape: (n_batches, 3, 3)

        # Calculate cell positions from deformation gradient
        cell_factor_expanded = state.cell_factor.expand(n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(n_batches, 3, 3) * cell_factor_expanded
        )  # shape: (n_batches, 3, 3)

        # Get per-atom and per-cell learning rates
        atom_wise_lr = positions_lr[state.batch].unsqueeze(-1)
        cell_wise_lr = cell_lr.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

        # Update atomic and cell positions
        atomic_positions_new = state.positions + atom_wise_lr * state.forces
        cell_positions_new = cell_positions + cell_wise_lr * state.cell_forces

        # Update cell with deformation gradient
        cell_update = cell_positions_new / cell_factor_expanded
        new_cell = torch.bmm(state.reference_cell, cell_update.transpose(1, 2))

        # Get new forces and energy
        model_output = model(
            positions=atomic_positions_new,
            cell=new_cell,
            atomic_numbers=state.atomic_numbers,
            batch=state.batch,
        )

        # Update state
        state.positions = atomic_positions_new
        state.cell = new_cell
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]
        state.stress = model_output["stress"]

        # Calculate virial for cell forces
        volumes = torch.linalg.det(new_cell).view(-1, 1, 1)
        virial = -volumes * state.stress + state.pressure
        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        # Scale virial by cell_factor
        virial = virial / state.cell_factor

        # Update cell forces
        state.cell_positions = cell_positions_new
        state.cell_forces = virial

        return state

    return gd_init, gd_step


@dataclass
class BatchedUnitCellFireState(BaseState):
    """State information for batched FIRE optimization with unit cell degrees of freedom.

    This class extends BaseState to include additional attributes needed for FIRE
    optimization with unit cell degrees of freedom. It handles both atomic and cell
    optimization in a batched manner, where multiple systems can be optimized
    simultaneously.

    The state tracks atomic positions, forces, velocities as well as cell parameters and
    their associated quantities (positions, forces, velocities). It also maintains
    FIRE-specific optimization parameters like timesteps and mixing parameters.

    Attributes:
        # Atomic quantities
        forces: Forces on atoms [n_total_atoms, 3]
        velocity: Atomic velocities [n_total_atoms, 3]
        energy: Energy per batch [n_batches]
        stress: Stress tensor [n_batches, 3, 3]

        # Cell quantities
        cell_positions: Cell positions [n_batches, 3, 3]
        cell_velocities: Cell velocities [n_batches, 3, 3]
        cell_forces: Cell forces [n_batches, 3, 3]
        cell_masses: Cell masses [n_batches, 3]

        # Cell optimization parameters
        orig_cell: Original unit cells [n_batches, 3, 3]
        cell_factor: Cell optimization scaling factor [n_batches, 1, 1]
        pressure: Applied pressure tensor [n_batches, 3, 3]

        # FIRE optimization parameters
        dt: Current timestep per batch [n_batches]
        alpha: Current mixing parameter per batch [n_batches]
        n_pos: Number of positive power steps per batch [n_batches]
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
    """

    # Required attributes not in BaseState
    forces: torch.Tensor  # [n_total_atoms, 3]
    energy: torch.Tensor  # [n_batches]
    stress: torch.Tensor  # [n_batches, 3, 3]
    velocities: torch.Tensor  # [n_total_atoms, 3]

    # cell attributes
    cell_positions: torch.Tensor  # [n_batches, 3, 3]
    cell_velocities: torch.Tensor  # [n_batches, 3, 3]
    cell_forces: torch.Tensor  # [n_batches, 3, 3]
    cell_masses: torch.Tensor  # [n_batches, 3]

    # Optimization-specific attributes
    orig_cell: torch.Tensor  # [n_batches, 3, 3]
    cell_factor: torch.Tensor  # [n_batches, 1, 1]
    pressure: torch.Tensor  # [n_batches, 3, 3]

    # FIRE algorithm parameters
    dt: torch.Tensor  # [n_batches]
    alpha: torch.Tensor  # [n_batches]
    n_pos: torch.Tensor  # [n_batches]
    hydrostatic_strain: bool
    constant_volume: bool

    @property
    def momenta(self) -> torch.Tensor:
        """Atomwise momenta of the system."""
        return self.velocities * self.masses.unsqueeze(-1)


def unit_cell_fire(  # noqa: C901, PLR0915
    model: torch.nn.Module,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    BatchedUnitCellFireState,
    Callable[[BatchedUnitCellFireState], BatchedUnitCellFireState],
]:
    """Initialize a batched FIRE optimization with unit cell degrees of freedom.

    This function sets up FIRE (Fast Inertial Relaxation Engine) optimization
    for both atomic positions and unit cell parameters in a batched manner.
    FIRE combines molecular dynamics with adaptive velocity damping
    to efficiently find local minima.

    The optimization proceeds by:
    1. Performing velocity Verlet MD steps for both atoms and cell
    2. Computing power P = F·v (force dot velocity) for both atomic and cell degrees of
       freedom
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
        model: Neural network model computing energies, forces, and stress
        dt_max: Maximum allowed timestep (default: 1.0)
        dt_start: Initial timestep (default: 0.1)
        n_min: Minimum steps before timestep increase (default: 5)
        f_inc: Factor for timestep increase (default: 1.1)
        f_dec: Factor for timestep decrease (default: 0.5)
        alpha_start: Initial velocity mixing parameter (default: 0.1)
        f_alpha: Factor for mixing parameter decrease (default: 0.99)
        cell_factor: Scaling factor for cell optimization (default: number of atoms)
        hydrostatic_strain: Whether to only allow hydrostatic deformation (default: False)
        constant_volume: Whether to maintain constant volume (default: False)
        scalar_pressure: Applied pressure in GPa (default: 0.0)

    Returns:
        Tuple containing:
        - Initialization function that creates a BatchedUnitCellFireState
        - Update function that performs one FIRE optimization step

    Notes:
        - The cell_factor parameter controls the relative scale of atomic vs cell
          optimization
        - hydrostatic_strain=True restricts cell deformation to volume changes only
        - constant_volume=True maintains cell volume while allowing shape changes
        - Pressure can be applied through the scalar_pressure parameter
    """
    device = model.device
    dtype = model.dtype

    # Setup parameters
    params = [dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min]
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min = [
        (
            p
            if isinstance(p, torch.Tensor)
            else torch.tensor(p, device=device, dtype=dtype)
        )
        for p in params
    ]

    def fire_init(
        state: BaseState | StateDict,
        cell_factor: torch.Tensor | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
        **kwargs: Any,
    ) -> BatchedUnitCellFireState:
        """Initialize a batched FIRE optimization state with unit cell.

        Args:
            state: Input state as BaseState object or state parameter dict
            cell_factor: Cell optimization scaling factor. If None, uses atoms per batch.
                Single value or tensor of shape [n_batches].
            scalar_pressure: Applied pressure in energy units
            dt_start: Initial timestep per batch
            alpha_start: Initial mixing parameter per batch
            **kwargs: Additional state attribute overrides

        Returns:
            BatchedUnitCellFireState with initialized optimization tensors
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get dimensions
        n_batches = state.n_batches

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per batch
            _, counts = torch.unique(state.batch, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all batches
            cell_factor = torch.full(
                (state.n_batches,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_batches, 1, 1) for broadcasting
        cell_factor = cell_factor.view(-1, 1, 1)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
        pressure = pressure.unsqueeze(0).expand(n_batches, -1, -1)

        # Get initial forces and energy from model
        model_output = model(
            positions=state.positions,
            cell=state.cell,
            atomic_numbers=atomic_numbers,
            batch=state.batch,
        )

        energy = model_output["energy"]  # [n_batches]
        forces = model_output["forces"]  # [n_total_atoms, 3]
        stress = model_output["stress"]  # [n_batches, 3, 3]

        volumes = torch.linalg.det(state.cell).view(-1, 1, 1)
        virial = -volumes * stress + pressure

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        virial = virial / cell_factor
        cell_forces = virial

        # Sum masses per batch using segment_reduce
        # TODO (AG): check this
        batch_counts = torch.bincount(state.batch)

        cell_masses = torch.segment_reduce(
            state.masses, reduce="sum", lengths=batch_counts
        )  # shape: (n_batches,)
        cell_masses = cell_masses.unsqueeze(-1).expand(-1, 3)  # shape: (n_batches, 3)

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)

        n_pos = torch.zeros((n_batches,), device=device, dtype=torch.int32)

        # Create initial state
        return BatchedUnitCellFireState(
            # Copy base state attributes
            positions=state.positions.clone(),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            batch=state.batch.clone(),
            pbc=state.pbc,
            # new attrs
            velocities=torch.zeros_like(state.positions),
            forces=forces,
            energy=energy,
            stress=stress,
            # cell attrs
            cell_positions=torch.zeros(n_batches, 3, 3, device=device, dtype=dtype),
            cell_velocities=torch.zeros(n_batches, 3, 3, device=device, dtype=dtype),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
            # optimization attrs
            orig_cell=state.cell.clone(),
            cell_factor=cell_factor,
            pressure=pressure,
            dt=dt_start,
            alpha=alpha_start,
            n_pos=n_pos,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )

    def fire_step(  # noqa: PLR0915
        state: BatchedUnitCellFireState,
        alpha_start: float = alpha_start,
        dt_start: float = dt_start,
    ) -> BatchedUnitCellFireState:
        """Perform one FIRE optimization step for batched atomic systems with unit cell
        optimization.

        Implements one step of the Fast Inertial Relaxation Engine (FIRE) algorithm for
        optimizing atomic positions and unit cell parameters in a batched setting. Uses
        velocity Verlet integration with adaptive velocity mixing.

        Args:
            state: Current optimization state containing atomic and cell parameters
            alpha_start: Initial mixing parameter for velocity update
            dt_start: Initial timestep for velocity Verlet integration

        Returns:
            Updated state after performing one FIRE step
        """
        n_batches = state.n_batches

        # Setup parameters
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)

        # Calculate current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.orig_cell, state.cell), 1, 2
        )  # shape: (n_batches, 3, 3)

        # Calculate cell positions from deformation gradient
        cell_factor_expanded = state.cell_factor.expand(n_batches, 3, 1)
        cell_positions = cur_deform_grad * cell_factor_expanded

        # Velocity Verlet first half step (v += 0.5*a*dt)
        atom_wise_dt = state.dt[state.batch].unsqueeze(-1)
        cell_wise_dt = state.dt.unsqueeze(-1).unsqueeze(-1)

        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions  # shape: (n_atoms, 3)

        # Update atomic and cell positions
        atomic_positions_new = atomic_positions + atom_wise_dt * state.velocities
        cell_positions_new = cell_positions + cell_wise_dt * state.cell_velocities

        # Update cell with deformation gradient
        cell_update = cell_positions_new / cell_factor_expanded
        new_cell = torch.bmm(state.orig_cell, cell_update.transpose(1, 2))

        # Get new forces and energy
        results = model(
            positions=atomic_positions_new,
            cell=new_cell,
            atomic_numbers=state.atomic_numbers,
            batch=state.batch,
        )

        # Update state with new positions and cell
        state.positions = atomic_positions_new
        state.cell_positions = cell_positions_new
        state.cell = new_cell
        state.energy = results["energy"]

        # Combine new atomic forces and cell forces
        forces = results["forces"]
        stress = results["stress"]

        state.forces = forces
        state.stress = stress
        # Calculate virial
        volumes = torch.linalg.det(new_cell).view(-1, 1, 1)
        virial = -volumes * stress + state.pressure
        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_batches, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_batches, -1, -1)

        virial = virial / state.cell_factor
        state.cell_forces = virial

        # Velocity Verlet first half step (v += 0.5*a*dt)
        state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Calculate power (F·V) for atoms
        atomic_power = (state.forces * state.velocities).sum(dim=1)  # [n_atoms]
        atomic_power_per_batch = torch.zeros(
            n_batches, device=device, dtype=atomic_power.dtype
        )
        atomic_power_per_batch.scatter_add_(
            dim=0, index=state.batch, src=atomic_power
        )  # [n_batches]

        # Calculate power for cell DOFs
        cell_power = (state.cell_forces * state.cell_velocities).sum(
            dim=(1, 2)
        )  # [n_batches]
        batch_power = atomic_power_per_batch + cell_power

        for batch_idx in range(n_batches):
            # FIRE specific updates
            if batch_power[batch_idx] > 0:  # Power is positive
                state.n_pos[batch_idx] += 1
                if state.n_pos[batch_idx] > n_min:
                    state.dt[batch_idx] = min(state.dt[batch_idx] * f_inc, dt_max)
                    state.alpha[batch_idx] = state.alpha[batch_idx] * f_alpha
            else:  # Power is negative
                state.n_pos[batch_idx] = 0
                state.dt[batch_idx] = state.dt[batch_idx] * f_dec
                state.alpha[batch_idx] = alpha_start[batch_idx]
                # Reset velocities for both atoms and cell
                state.velocities[state.batch == batch_idx] = 0
                state.cell_velocities[batch_idx] = 0

        # Mix velocity and force direction using FIRE for atoms
        v_norm = torch.norm(state.velocities, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        # Avoid division by zero
        # mask = f_norm > 1e-10
        # state.velocity = torch.where(
        #     mask,
        #     (1.0 - state.alpha) * state.velocity
        #     + state.alpha * state.forces * v_norm / f_norm,
        #     state.velocity,
        # )
        batch_wise_alpha = state.alpha[state.batch].unsqueeze(-1)
        state.velocities = (
            1.0 - batch_wise_alpha
        ) * state.velocities + batch_wise_alpha * state.forces * v_norm / (f_norm + 1e-10)

        # Mix velocity and force direction for cell DOFs
        cell_v_norm = torch.norm(state.cell_velocities, dim=(1, 2), keepdim=True)
        cell_f_norm = torch.norm(state.cell_forces, dim=(1, 2), keepdim=True)
        cell_wise_alpha = state.alpha.unsqueeze(-1).unsqueeze(-1)
        cell_mask = cell_f_norm > 1e-10
        state.cell_velocities = torch.where(
            cell_mask,
            (1.0 - cell_wise_alpha) * state.cell_velocities
            + cell_wise_alpha * state.cell_forces * cell_v_norm / cell_f_norm,
            state.cell_velocities,
        )

        return state

    return fire_init, fire_step
