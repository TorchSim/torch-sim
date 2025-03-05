"""Batched optimizers for structure optimization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from torchsim.state import BaseState
from torchsim.unbatched_optimizers import OptimizerState


StateDict = dict[
    Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "batch"], torch.Tensor
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
        **extra_state_kwargs: Any,
    ) -> BatchedGDState:
        """Initialize the batched gradient descent optimization state.

        Args:
            state: Base state containing positions, masses, cell, etc.
            extra_state_kwargs: Additional keyword arguments to override state attributes

        Returns:
            Initialized BatchedGDState with forces and energy
        """
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        atomic_numbers = extra_state_kwargs.get("atomic_numbers", state.atomic_numbers)

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
        orig_cell: Original unit cells tensor of shape (n_batches, 3, 3)
        cell_factor: Scaling factor for cell optimization
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        pressure: Applied pressure tensor
    """

    orig_cell: torch.Tensor
    cell_factor: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool
    pressure: torch.Tensor


def batched_unit_cell_gradient_descent(  # noqa: PLR0915
    model: torch.nn.Module,
    # list of tensors with shape (n_atoms_per_batch, 3)
    positions_list: list[torch.Tensor],
    masses_list: list[torch.Tensor],  # list of tensors with shape (n_atoms_per_batch,)
    cell_list: list[torch.Tensor],  # list of tensors with shape (3, 3)
    batch_indices: torch.Tensor,  # shape: (total_atoms,)
    learning_rates: torch.Tensor | float = 0.01,
    *,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    BatchedUnitCellGDState, Callable[[BatchedUnitCellGDState], BatchedUnitCellGDState]
]:
    """Initialize a batched gradient descent optimization with unit cell."""
    device = positions_list[0].device
    dtype = positions_list[0].dtype

    # Get dimensions
    n_batches = len(positions_list)
    n_atoms_per_batch = [pos.shape[0] for pos in positions_list]
    total_atoms = sum(n_atoms_per_batch)

    # Convert learning rates to tensor if needed
    if isinstance(learning_rates, float):
        lr = torch.full((n_batches,), learning_rates, device=device, dtype=dtype)
    else:
        lr = learning_rates.to(device=device, dtype=dtype)
        assert len(lr) == n_batches, (
            "Number of learning rates must match number of batches"
        )

    # Setup cell_factor
    if isinstance(cell_factor, float):
        cell_factor = torch.full(
            (n_batches, 1, 1), cell_factor, device=device, dtype=dtype
        )
    elif cell_factor is None:
        cell_factor = torch.tensor(
            [float(n) for n in n_atoms_per_batch], device=device, dtype=dtype
        )
        cell_factor = cell_factor.view(-1, 1, 1)  # shape: (n_batches, 1, 1)

    # Setup pressure tensor
    pressure = scalar_pressure * torch.eye(3, device=device)
    pressure = pressure.unsqueeze(0).expand(n_batches, -1, -1)  # shape: (n_batches, 3, 3)

    def initialize_state(
        positions_list: list[torch.Tensor],
        masses_list: list[torch.Tensor],
        cell_list: list[torch.Tensor],
        *,
        pbc: bool,
        lr: torch.Tensor = lr,
    ) -> BatchedUnitCellGDState:
        """Initialize the batched gradient descent optimization state."""
        # Get initial forces and energy from model
        results = model(positions_list, cell_list)
        energy = results["energy"]
        forces_list = results["forces"]
        stress = results["stress"]  # Already shape: (n_batches, 3, 3)

        # Concatenate atomic positions, forces, and masses
        positions_cat = torch.cat(positions_list, dim=0)
        forces_cat = torch.cat(forces_list, dim=0)
        masses_cat = torch.cat(masses_list, dim=0)

        # Stack cells and create extended masses for cell DOFs
        cell_stack = torch.stack(cell_list, dim=0)
        cell_masses = torch.ones(
            n_batches * 3, device=device, dtype=dtype
        )  # One mass per cell DOF
        masses_extended = torch.cat([masses_cat, cell_masses])

        # Combine atomic forces and cell forces
        forces_combined = torch.zeros(
            (total_atoms + 3 * n_batches, 3), device=device, dtype=dtype
        )

        # Atomic forces
        forces_combined[:total_atoms] = forces_cat

        # Cell forces (from stress)
        volumes = torch.stack([torch.linalg.det(cell) for cell in cell_list])
        volumes = volumes.view(-1, 1, 1)

        # Calculate virial
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

        # Scale virial by cell_factor
        virial = virial / cell_factor

        # Reshape virial for forces_combined
        virial_flat = virial.reshape(n_batches * 3, 3)
        forces_combined[total_atoms:] = virial_flat

        return BatchedUnitCellGDState(
            positions=positions_cat,
            forces=forces_combined,
            energy=energy,
            masses=masses_extended,
            cell=cell_stack,
            pbc=pbc,
            lr=lr,
            orig_cell=cell_stack.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
        )

    def gd_step(state: BatchedUnitCellGDState) -> BatchedUnitCellGDState:
        """Perform one gradient descent optimization step."""
        # Get dimensions
        n_atoms = total_atoms

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.orig_cell, state.cell), 1, 2
        )  # shape: (n_batches, 3, 3)

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions[:n_atoms]

        # Fix cell positions calculation
        cell_factor_expanded = state.cell_factor.expand(
            n_batches, 3, 1
        )  # shape: (n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(n_batches, 3, 3) * cell_factor_expanded
        ).reshape(-1, 3)  # shape: (n_batches * 3, 3)

        # Get per-atom and per-cell learning rates
        atom_lr = state.lr[batch_indices].unsqueeze(-1)
        cell_lr = state.lr.repeat_interleave(3).unsqueeze(-1)

        # Update atomic and cell positions
        atomic_positions_new = atomic_positions + atom_lr * state.forces[:n_atoms]
        cell_positions_new = cell_positions + cell_lr * state.forces[n_atoms:]

        # Update cell with deformation gradient
        cell_update = (cell_positions_new / cell_factor_expanded.reshape(-1, 1)).reshape(
            n_batches, 3, 3
        )
        new_cell = torch.bmm(state.orig_cell, cell_update.transpose(1, 2))

        # Split positions back into list for model input
        positions_split = torch.split(atomic_positions_new, n_atoms_per_batch)
        positions_list = [pos.clone() for pos in positions_split]
        cell_list = [new_cell[i].clone() for i in range(n_batches)]

        # Get new forces and energy
        results = model(positions_list, cell_list)

        # Update state
        state.positions = torch.cat([atomic_positions_new, cell_positions_new], dim=0)
        state.cell = new_cell
        state.energy = results["energy"]

        # Combine new atomic forces and cell forces
        forces = torch.cat(results["forces"], dim=0)
        stress = results["stress"]

        forces_combined = torch.zeros_like(state.forces)
        forces_combined[:n_atoms] = forces

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
        virial_flat = virial.reshape(n_batches * 3, 3)
        forces_combined[n_atoms:] = virial_flat

        state.forces = forces_combined
        return state

    initial_state = initialize_state(positions_list, masses_list, cell_list, pbc=True)
    return initial_state, gd_step


@dataclass
class BatchedUnitCellFIREState:
    """State class for batched FIRE optimization with unit cell.

    Attributes:
        positions: Atomic positions tensor of shape (total_atoms + n_batches * 3, 3)
        forces: Forces tensor of shape (total_atoms + n_batches * 3, 3)
        energy: Energy tensor of shape (n_batches,)
        masses: Masses tensor of shape (total_atoms + n_batches * 3,)
        cell: Unit cell tensor of shape (n_batches, 3, 3)
        pbc: Periodic boundary conditions flags
        velocity: Velocity tensor of shape (total_atoms + n_batches * 3, 3)
        dt: Current timestep
        alpha: Current mixing parameter
        n_pos: Number of positive power steps
        orig_cell: Original unit cells tensor of shape (n_batches, 3, 3)
        cell_factor: Scaling factor for cell optimization
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        pressure: Applied pressure tensor
    """

    positions: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    masses: torch.Tensor
    cell: torch.Tensor

    # cell_positions: torch.Tensor
    # cell_velocities: torch.Tensor
    # cell_forces: torch.Tensor
    # cell_masses: torch.Tensor

    pbc: bool
    velocity: torch.Tensor
    dt: float
    alpha: float
    n_pos: int
    orig_cell: torch.Tensor
    cell_factor: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool
    pressure: torch.Tensor


def batched_unit_cell_fire(  # noqa: C901, PLR0915
    model: torch.nn.Module,
    positions: torch.Tensor,  # shape: (n_batches, n_atoms_per_batch, 3)
    masses: torch.Tensor,  # shape: (n_batches, n_atoms_per_batch)
    cell: torch.Tensor,  # shape: (n_batches, 3, 3)
    batch_indices: torch.Tensor,  # shape: (total_atoms,)
    *,
    dt_max: float = 0.4,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    BatchedUnitCellFIREState,
    Callable[[BatchedUnitCellFIREState], BatchedUnitCellFIREState],
]:
    """Initialize a batched FIRE optimization with unit cell."""
    device = positions.device
    dtype = positions.dtype

    # Get dimensions
    n_batches = positions.shape[0]
    n_atoms_per_batch = positions.shape[1]

    # Setup cell_factor
    if isinstance(cell_factor, float):
        cell_factor = torch.full(
            (n_batches, 1, 1), cell_factor, device=device, dtype=dtype
        )
    elif cell_factor is None:
        cell_factor = torch.full(
            (n_batches, 1, 1), float(n_atoms_per_batch), device=device, dtype=dtype
        )

    # Setup pressure tensor
    pressure = scalar_pressure * torch.eye(3, device=device)
    pressure = pressure.unsqueeze(0).expand(n_batches, -1, -1)

    def initialize_state(
        positions: torch.Tensor,
        masses: torch.Tensor,
        cell: torch.Tensor,
        *,
        pbc: bool,
    ) -> FireState:
        """Initialize the batched FIRE optimization state."""
        # Get initial forces and energy from model
        results = model(positions, cell)
        energy = results["energy"]
        forces = results["forces"]
        stress = results["stress"]

        # Total number of DOFs (atoms + cell)
        n_atoms = len(batch_indices)
        total_dofs = n_atoms + 3 * n_batches

        # Combine atomic forces and cell forces
        forces_combined = torch.zeros((total_dofs, 3), device=device, dtype=dtype)
        # Atomic forces
        forces_combined[:n_atoms] = forces.reshape(-1, 3)

        # Cell forces (from stress)
        volumes = torch.linalg.det(cell).view(-1, 1, 1)
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
        virial_flat = virial.reshape(n_batches * 3, 3)
        forces_combined[n_atoms:] = virial_flat

        # Initialize masses for cell degrees of freedom
        masses_combined = torch.zeros(n_atoms + 3 * n_batches, device=device, dtype=dtype)
        masses_combined[:n_atoms] = masses.reshape(-1)
        masses_combined[n_atoms:] = masses.sum(dim=1).repeat_interleave(
            3
        )  # Use total mass for cell DOFs

        return FireState(
            positions=positions.reshape(-1, 3),
            forces=forces_combined,
            energy=energy,
            masses=masses_combined,
            cell=cell,
            pbc=pbc,
            velocity=torch.zeros_like(forces_combined),
            dt=dt_start,
            alpha=alpha_start,
            n_pos=0,
            orig_cell=cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
        )

    def fire_step(state: FireState) -> FireState:  # noqa: PLR0915
        """Perform one FIRE optimization step."""
        # Get dimensions
        n_atoms = len(batch_indices)  # Total number of atoms

        # Get current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.orig_cell, state.cell), 1, 2
        )  # shape: (n_batches, 3, 3)

        # Fix cell positions calculation
        cell_factor_expanded = state.cell_factor.expand(
            n_batches, 3, 1
        )  # shape: (n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(n_batches, 3, 3) * cell_factor_expanded
        ).reshape(-1, 3)  # shape: (n_batches * 3, 3)

        # Velocity Verlet first half step (v += 0.5*a*dt)
        state.velocity += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions[:n_atoms]  # shape: (n_atoms, 3)

        # Update atomic and cell positions
        atomic_positions_new = atomic_positions + state.dt * state.velocity[:n_atoms]
        cell_positions_new = cell_positions + state.dt * state.velocity[n_atoms:]

        # Update cell with deformation gradient
        cell_update = (cell_positions_new / cell_factor_expanded.reshape(-1, 1)).reshape(
            n_batches, 3, 3
        )
        new_cell = torch.bmm(state.orig_cell, cell_update.transpose(1, 2))

        # Get new forces and energy
        positions_batched = atomic_positions_new.reshape(n_batches, -1, 3)
        results = model(positions_batched, new_cell)

        # Update state with new positions and cell
        state.positions = torch.cat([atomic_positions_new, cell_positions_new], dim=0)
        state.cell = new_cell
        state.energy = results["energy"]

        # Combine new atomic forces and cell forces
        forces = results["forces"]
        stress = results["stress"]

        forces_combined = torch.zeros_like(state.forces)
        forces_combined[:n_atoms] = forces.reshape(-1, 3)

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
        virial_flat = virial.reshape(n_batches * 3, 3)
        forces_combined[n_atoms:] = virial_flat

        state.forces = forces_combined

        # Velocity Verlet second half step (v += 0.5*a*dt)
        state.velocity += 0.5 * state.dt * state.forces / state.masses.unsqueeze(-1)

        # Create extended batch indices to include cell DOFs
        extended_batch_indices = torch.cat(
            [
                batch_indices,  # For atoms
                torch.arange(n_batches, device=device).repeat_interleave(
                    3
                ),  # For cell DOFs
            ]
        )

        # Calculate power (F·V) for each batch after velocity update
        power_per_dof = (state.forces * state.velocity).sum(dim=1)
        power = torch.zeros(n_batches, device=device, dtype=dtype)
        power.scatter_add_(dim=0, index=extended_batch_indices, src=power_per_dof)
        # Map power back to all atoms and cell DOFs
        power = power[extended_batch_indices]

        # FIRE specific updates
        if power.sum() > 0:  # Power is positive
            state.n_pos += 1
            if state.n_pos > n_min:
                state.dt = min(state.dt * f_inc, dt_max)
                state.alpha = state.alpha * f_alpha
        else:  # Power is negative
            state.n_pos = 0
            state.dt = state.dt * f_dec
            state.alpha = alpha_start
            state.velocity.zero_()  # Reset velocities

        # Mix velocity and force direction using FIRE
        v_norm = torch.norm(state.velocity, dim=1, keepdim=True)
        f_norm = torch.norm(state.forces, dim=1, keepdim=True)
        state.velocity = (
            1.0 - state.alpha
        ) * state.velocity + state.alpha * state.forces * v_norm / (f_norm + 1e-10)

        return state

    return initialize_state(positions, masses, cell, pbc=True), fire_step


@dataclass
class FireState(BaseState):
    """State information for FIRE optimization with unit cell degrees of freedom.

    Attributes:
        positions: Atomic positions with shape [n_total_atoms, 3]
        forces: Forces on atoms with shape [n_total_atoms, 3]
        masses: Atomic masses with shape [n_total_atoms]
        atomic_numbers: Atomic numbers with shape [n_total_atoms]
        cell: Unit cell matrices with shape [n_batches, 3, 3]
        batch: Batch indices with shape [n_total_atoms]
        pbc: Whether to use periodic boundary conditions

        energy: Energy per batch with shape [n_batches]
        velocity: Atomic velocities with shape [n_total_atoms, 3]

        # Parameters for unit cell optimization
        orig_cell: Original unit cells with shape [n_batches, 3, 3]
        cell_factor: Scaling factor for cell optimization with shape [n_batches, 1, 1]
        pressure: Applied pressure tensor with shape [n_batches, 3, 3]

        # FIRE algorithm parameters
        dt: Current timestep (scalar)
        alpha: Current mixing parameter (scalar)
        n_pos: Number of positive power steps (scalar)
        hydrostatic_strain: Whether to only allow hydrostatic deformation (scalar)
        constant_volume: Whether to maintain constant volume (scalar)
    """

    # Required attributes not in BaseState
    forces: torch.Tensor  # [n_total_atoms, 3]
    energy: torch.Tensor  # [n_batches]
    velocity: torch.Tensor  # [n_total_atoms, 3]

    # cell attributes
    cell_positions: torch.Tensor  # [n_batches * 3, 3]
    cell_velocities: torch.Tensor  # [n_batches * 3, 3]
    cell_forces: torch.Tensor  # [n_batches * 3, 3]
    cell_masses: torch.Tensor  # [n_batches * 3]

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


def fire(  # noqa: C901, PLR0915
    state: BaseState,
    model: torch.nn.Module,
    *,
    dt_max: float = 0.4,
    dt_start: float | torch.Tensor = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float = 10,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    FireState,
    Callable[[FireState], FireState],
]:
    """Initialize a batched FIRE optimization with unit cell.

    Args:
        state: Initial system state containing positions, masses, cell
        model: Neural network model that computes energies, forces, and stress
        dt_max: Maximum timestep
        dt_start: Initial timestep
        n_min: Minimum number of steps before increasing timestep
        f_inc: Factor to increase timestep by
        f_dec: Factor to decrease timestep by
        alpha_start: Initial mixing parameter
        f_alpha: Factor to decrease alpha by
        cell_factor: Scaling factor for cell optimization
        hydrostatic_strain: Whether to only allow hydrostatic deformation
        constant_volume: Whether to maintain constant volume
        scalar_pressure: Applied pressure in energy units

    Returns:
        tuple:
            - BatchedUnitCellFIREState: Initial state with forces and velocities
            - callable: Update function that performs one FIRE step
    """
    device = state.device
    dtype = state.dtype

    # Get dimensions
    n_batches = state.n_batches

    # Setup cell_factor
    # TODO: allow different cell_factors per batch
    cell_factor = torch.full((n_batches, 1, 1), cell_factor, device=device, dtype=dtype)

    # Setup pressure tensor
    pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
    pressure = pressure.unsqueeze(0).expand(n_batches, -1, -1)

    # Get initial forces and energy from model
    model_output = model(positions=state.positions, cell=state.cell, batch=state.batch)

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
        virial = virial - diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
            0
        ).expand(n_batches, -1, -1)

    virial = virial / cell_factor
    cell_forces = virial.reshape(n_batches * 3, 3)

    # Sum masses per batch using segment_reduce
    # TODO: check this
    batch_counts = torch.bincount(state.batch)

    cell_masses = torch.segment_reduce(
        state.masses, reduce="sum", lengths=batch_counts
    )  # shape: (n_batches,)
    cell_masses = cell_masses.repeat_interleave(3)  # shape: (n_batches * 3,)

    if isinstance(dt_start, float):
        dt_start = torch.full((n_batches,), dt_start, device=device, dtype=dtype)
    if isinstance(alpha_start, float):
        alpha_start = torch.full((n_batches,), alpha_start, device=device, dtype=dtype)
    n_pos = torch.zeros((n_batches,), device=device, dtype=torch.int32)

    # Create initial state
    initial_state = FireState(
        # Copy base state attributes
        positions=state.positions.clone(),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        atomic_numbers=state.atomic_numbers.clone(),
        batch=state.batch.clone(),
        pbc=state.pbc,
        # new attrs
        velocity=torch.zeros_like(state.positions),
        forces=forces,
        energy=energy,
        # cell attrs
        cell_positions=torch.zeros(3 * n_batches, 3, device=device, dtype=dtype),
        cell_velocities=torch.zeros(3 * n_batches, 3, device=device, dtype=dtype),
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

    def fire_step(state: FireState) -> FireState:  # noqa: PLR0915
        """Perform one FIRE optimization step."""
        n_batches = state.n_batches

        # Calculate current deformation gradient
        cur_deform_grad = torch.transpose(
            torch.linalg.solve(state.orig_cell, state.cell), 1, 2
        )  # shape: (n_batches, 3, 3)

        # Calculate cell positions from deformation gradient
        cell_factor_expanded = state.cell_factor.expand(n_batches, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(n_batches, 3, 3) * cell_factor_expanded
        ).reshape(-1, 3)

        # Velocity Verlet first half step (v += 0.5*a*dt)
        atom_wise_dt = state.dt[state.batch].unsqueeze(-1)
        cell_wise_dt = state.dt.repeat_interleave(3).unsqueeze(-1)

        state.velocity += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Split positions and forces into atomic and cell components
        atomic_positions = state.positions  # shape: (n_atoms, 3)

        # Update atomic and cell positions
        atomic_positions_new = atomic_positions + atom_wise_dt * state.velocity
        cell_positions_new = cell_positions + cell_wise_dt * state.cell_velocities

        # Update cell with deformation gradient
        cell_update = (cell_positions_new / cell_factor_expanded.reshape(-1, 1)).reshape(
            n_batches, 3, 3
        )
        new_cell = torch.bmm(state.orig_cell, cell_update.transpose(1, 2))

        # Get new forces and energy
        results = model(atomic_positions_new, new_cell, batch=state.batch)

        # Update state with new positions and cell
        state.positions = atomic_positions_new
        state.cell_positions = cell_positions_new
        state.cell = new_cell
        state.energy = results["energy"]

        # Combine new atomic forces and cell forces
        forces = results["forces"]
        stress = results["stress"]

        state.forces = forces

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
        virial_flat = virial.reshape(n_batches * 3, 3)
        state.cell_forces = virial_flat

        # Velocity Verlet first half step (v += 0.5*a*dt)
        state.velocity += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

        # Calculate power (F·V) for atoms
        atomic_power = (state.forces * state.velocity).sum(dim=1)  # [n_atoms]
        atomic_power_per_batch = torch.zeros(n_batches, device=device, dtype=dtype)
        atomic_power_per_batch.scatter_add_(
            dim=0, index=state.batch, src=atomic_power
        )  # [n_batches]

        # Calculate power for cell DOFs
        cell_power = (state.cell_forces * state.cell_velocities).sum(
            dim=1
        )  # [n_batches*3]
        cell_batch = torch.arange(n_batches, device=device).repeat_interleave(3)
        cell_power_per_batch = torch.zeros(n_batches, device=device, dtype=dtype)
        cell_power_per_batch.scatter_add_(
            dim=0, index=cell_batch, src=cell_power
        )  # [n_batches]

        # Calculate total power per batch and sum
        batch_power = atomic_power_per_batch + cell_power_per_batch

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
                state.velocity[state.batch == batch_idx] = 0
                cell_batch = torch.arange(n_batches, device=device).repeat_interleave(3)
                state.cell_velocities[cell_batch == batch_idx] = 0

        # Mix velocity and force direction using FIRE for atoms
        v_norm = torch.norm(state.velocity, dim=1, keepdim=True)
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
        state.velocity = (
            1.0 - batch_wise_alpha
        ) * state.velocity + batch_wise_alpha * state.forces * v_norm / (f_norm + 1e-10)

        # Mix velocity and force direction for cell DOFs
        cell_v_norm = torch.norm(state.cell_velocities, dim=1, keepdim=True)
        cell_f_norm = torch.norm(state.cell_forces, dim=1, keepdim=True)
        cell_wise_alpha = state.alpha.repeat_interleave(3).unsqueeze(-1)
        cell_mask = cell_f_norm > 1e-10
        state.cell_velocities = torch.where(
            cell_mask,
            (1.0 - cell_wise_alpha) * state.cell_velocities
            + cell_wise_alpha * state.cell_forces * cell_v_norm / cell_f_norm,
            state.cell_velocities,
        )

        return state

    return initial_state, fire_step
