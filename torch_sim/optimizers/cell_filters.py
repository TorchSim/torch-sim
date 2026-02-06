"""Cell filters for optimization algorithms.

This module provides filter functions that can be applied to optimization algorithms
to handle different types of cell optimization constraints and parameterizations.
Filters encapsulate the logic for computing cell forces and updating cell parameters
during optimization.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

import torch_sim.math as fm
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers.state import BFGSState, FireState, LBFGSState, OptimState
from torch_sim.state import SimState


def _setup_cell_factor(
    state: SimState,
    cell_factor: float | torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Setup cell factor tensor."""
    n_systems = state.n_systems

    if cell_factor is None:
        # Count atoms per system
        _, counts = torch.unique(state.system_idx, return_counts=True)
        cell_factor_tensor = counts.to(dtype=dtype)
    elif isinstance(cell_factor, (int, float)):
        cell_factor_tensor = torch.full(
            (n_systems,), cell_factor, device=device, dtype=dtype
        )
    else:
        cell_factor_tensor = torch.tensor(cell_factor, device=device, dtype=dtype)
        if (n_cft := cell_factor_tensor.numel()) != n_systems:
            raise ValueError(
                f"cell_factor tensor must have {n_systems} elements, got {n_cft}"
            )

    return cell_factor_tensor.view(n_systems, 1, 1)


def _setup_pressure(
    n_systems: int, scalar_pressure: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Setup pressure tensor."""
    pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
    return pressure.unsqueeze(0).expand(n_systems, -1, -1)


def _compute_cell_masses(state: SimState) -> torch.Tensor:
    """Compute cell masses by summing atomic masses per system."""
    system_counts = torch.bincount(state.system_idx)
    cell_masses = torch.segment_reduce(state.masses, reduce="sum", lengths=system_counts)
    return cell_masses.unsqueeze(-1).expand(-1, 3)


def _apply_constraints(
    virial: torch.Tensor, *, hydrostatic_strain: bool, constant_volume: bool
) -> torch.Tensor:
    """Apply hydrostatic strain and constant volume constraints to virial."""
    n_systems, device = virial.shape[0], virial.device

    if hydrostatic_strain:
        diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
        virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
            0
        ).expand(n_systems, -1, -1)

    if constant_volume:
        diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
        virial = virial - diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
            0
        ).expand(n_systems, -1, -1)

    return virial


def deform_grad(reference_cell: torch.Tensor, current_cell: torch.Tensor) -> torch.Tensor:
    """Compute deformation gradient between current and reference cells."""
    return torch.linalg.solve(reference_cell, current_cell).transpose(-2, -1)


def unit_cell_filter_init[T: AnyCellState](
    state: T,
    model: ModelInterface,
    *,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
    **_kwargs: Any,
) -> None:
    """Initialize unit cell filter state."""
    device, dtype = model.device, model.dtype
    n_systems = state.n_systems

    # Setup parameters
    cell_factor_tensor = _setup_cell_factor(state, cell_factor, device, dtype)
    pressure = _setup_pressure(n_systems, scalar_pressure, device, dtype)
    cell_masses = _compute_cell_masses(state)

    # Get initial model output for stress
    model_output = model(state)

    # Calculate initial cell forces
    stress = model_output["stress"].clone()

    # Apply stress constraints if any exist
    if state.constraints:
        for constraint in state.constraints:
            constraint.adjust_stress(state, stress)

    volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
    virial = -volumes * (stress + pressure)
    virial = _apply_constraints(
        virial, hydrostatic_strain=hydrostatic_strain, constant_volume=constant_volume
    )
    cell_forces = virial / cell_factor_tensor

    # Calculate initial cell positions from deformation gradient
    # Use current cell as reference (matches reference implementation)
    reference_cell = state.cell.clone()
    cur_deform_grad = deform_grad(reference_cell.mT, state.row_vector_cell)
    cell_factor_expanded = cell_factor_tensor.expand(n_systems, 3, 1)
    cell_positions = cur_deform_grad.reshape(n_systems, 3, 3) * cell_factor_expanded

    # update state cell attributes in place
    state.cell_factor = cell_factor_tensor
    state.pressure = pressure
    state.hydrostatic_strain = hydrostatic_strain
    state.constant_volume = constant_volume
    state.reference_cell = reference_cell
    state.cell_positions = cell_positions
    state.cell_forces = cell_forces
    state.cell_masses = cell_masses


def frechet_cell_filter_init[T: AnyCellState](
    state: T,
    model: ModelInterface,
    *,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
    **_kwargs: Any,
) -> None:
    """Initialize Frechet cell filter state."""
    device, dtype = model.device, model.dtype
    n_systems = state.n_systems

    # Setup parameters
    cell_factor_tensor = _setup_cell_factor(state, cell_factor, device, dtype)
    pressure = _setup_pressure(n_systems, scalar_pressure, device, dtype)
    cell_masses = _compute_cell_masses(state)

    # Initialize cell positions to zeros (log of identity matrix)
    cell_positions = torch.zeros((n_systems, 3, 3), device=device, dtype=dtype)

    # Get initial model output for stress
    model_output = model(state)

    # Calculate initial cell forces using Frechet approach
    stress = model_output["stress"].clone()

    # Apply stress constraints if present
    for constraint in state.constraints:
        constraint.adjust_stress(state, stress)

    volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
    virial = -volumes * (stress + pressure)
    virial = _apply_constraints(
        virial, hydrostatic_strain=hydrostatic_strain, constant_volume=constant_volume
    )

    # Get current deformation gradient (identity at start for Frechet)
    reference_cell = state.cell.clone()
    cur_deform_grad = deform_grad(reference_cell.mT, state.row_vector_cell)
    ucf_cell_grad = torch.bmm(
        virial, torch.linalg.inv(torch.transpose(cur_deform_grad, 1, 2))
    )

    # For identity matrix (initial state), Frechet derivative gives zero forces
    # This matches the reference implementation behavior
    cell_forces = ucf_cell_grad / cell_factor_tensor

    # update state cell attributes in place
    state.cell_factor = cell_factor_tensor
    state.pressure = pressure
    state.hydrostatic_strain = hydrostatic_strain
    state.constant_volume = constant_volume
    state.reference_cell = reference_cell
    state.cell_positions = cell_positions
    state.cell_forces = cell_forces
    state.cell_masses = cell_masses


class CellFilter(StrEnum):
    """Enumeration of the cell filters."""

    unit = "unit"
    frechet = "frechet"


# Filter type definitions for convenience
def unit_cell_step[T: AnyCellState](state: T, cell_lr: float | torch.Tensor) -> None:
    """Update cell using unit cell approach."""
    if isinstance(cell_lr, (int, float)):
        cell_lr = torch.full(
            (state.n_systems,), cell_lr, device=state.device, dtype=state.dtype
        )

    # Get current deformation gradient
    cur_deform_grad = deform_grad(state.reference_cell.mT, state.row_vector_cell)

    # Calculate cell positions from current deformation gradient
    cell_factor_expanded = state.cell_factor.expand(state.n_systems, 3, 1)
    current_cell_positions = (
        cur_deform_grad.reshape(state.n_systems, 3, 3) * cell_factor_expanded
    )

    # Update cell positions
    cell_wise_lr = cell_lr.view(state.n_systems, 1, 1)
    cell_step = cell_wise_lr * state.cell_forces
    cell_positions_new = current_cell_positions + cell_step

    # Update cell from new positions
    cell_update = cell_positions_new / cell_factor_expanded
    new_cell = torch.bmm(state.reference_cell.mT, cell_update.transpose(-2, -1))

    # Apply cell constraints (in-place, column vector convention)
    state.set_constrained_cell(new_cell.mT.clone())
    state.cell_positions = cell_positions_new


def frechet_cell_step[T: AnyCellState](state: T, cell_lr: float | torch.Tensor) -> None:
    """Update cell using frechet approach."""
    if isinstance(cell_lr, (int, float)):
        cell_lr = torch.full(
            (state.n_systems,), cell_lr, device=state.device, dtype=state.dtype
        )
    cell_wise_lr = cell_lr.view(state.n_systems, 1, 1)

    # Compute cell step and update cell positions in log space
    cell_step = cell_wise_lr * state.cell_forces
    cell_positions_new = state.cell_positions + cell_step

    # Convert from log space to deformation gradient
    cell_factor_reshaped = state.cell_factor.view(state.n_systems, 1, 1)
    deform_grad_log_new = cell_positions_new / cell_factor_reshaped
    deform_grad_new = torch.matrix_exp(deform_grad_log_new)

    # Update cell from new deformation gradient
    new_row_vector_cell = torch.bmm(
        state.reference_cell.mT, deform_grad_new.transpose(-2, -1)
    )

    # Apply cell constraints (in-place, column vector convention)
    state.set_constrained_cell(new_row_vector_cell.mT.clone())
    state.cell_positions = cell_positions_new


def compute_cell_forces[T: AnyCellState](
    model_output: dict[str, torch.Tensor], state: T
) -> None:
    """Compute cell forces for both unit and frechet methods."""
    stress = model_output["stress"].clone()

    # Apply stress constraints if any exist
    if state.constraints:
        for constraint in state.constraints:
            constraint.adjust_stress(state, stress)

    volumes = torch.linalg.det(state.cell).view(state.n_systems, 1, 1)
    virial = -volumes * (stress + state.pressure)
    virial = _apply_constraints(
        virial,
        hydrostatic_strain=state.hydrostatic_strain,
        constant_volume=state.constant_volume,
    )

    # Check if this is Frechet method by examining the stored cell filter functions
    cell_filter_funcs = getattr(state, "cell_filter", None)
    is_frechet = (
        cell_filter_funcs is not None and cell_filter_funcs[0] is frechet_cell_filter_init
    )

    if is_frechet:
        # Frechet cell force computation
        cur_deform_grad = deform_grad(state.reference_cell.mT, state.row_vector_cell)
        ucf_cell_grad = torch.bmm(
            virial, torch.linalg.inv(torch.transpose(cur_deform_grad, 1, 2))
        )

        # Calculate Frechet derivative for non-identity deformation gradients
        device, dtype = virial.device, virial.dtype
        n_systems = state.n_systems

        # Create direction matrices for Frechet derivative
        directions = torch.zeros((9, 3, 3), device=device, dtype=dtype)
        for idx, (mu, nu) in enumerate([(i, j) for i in range(3) for j in range(3)]):
            directions[idx, mu, nu] = 1.0

        # Compute deformation gradient log
        deform_grad_log = torch.zeros_like(cur_deform_grad)
        for sys_idx in range(n_systems):
            deform_grad_log[sys_idx] = fm.matrix_log_33(cur_deform_grad[sys_idx])

        # Compute Frechet derivatives
        cell_forces = torch.zeros_like(ucf_cell_grad)
        for sys_idx in range(n_systems):
            expm_derivs = torch.stack(
                [
                    fm.expm_frechet(deform_grad_log[sys_idx], direction)[1]
                    for direction in directions
                ]
            )
            forces_flat = torch.sum(
                expm_derivs * ucf_cell_grad[sys_idx].unsqueeze(0), dim=(1, 2)
            )
            cell_forces[sys_idx] = forces_flat.reshape(3, 3)

        state.cell_forces = cell_forces / state.cell_factor
    else:  # Unit cell force computation
        # Note (AG): ASE transforms virial as:
        # virial = np.linalg.solve(cur_deform_grad, virial.T).T
        cur_deform_grad = deform_grad(state.reference_cell.mT, state.row_vector_cell)
        virial_transformed = torch.linalg.solve(
            cur_deform_grad, virial.transpose(-2, -1)
        ).transpose(-2, -1)
        state.cell_forces = virial_transformed / state.cell_factor


CellFilterFuncs = tuple[Callable[..., None], Callable[..., None]]  # (init_fn, update_fn)

CELL_FILTER_REGISTRY: dict[CellFilter, CellFilterFuncs] = {
    CellFilter.unit: (unit_cell_filter_init, unit_cell_step),
    CellFilter.frechet: (frechet_cell_filter_init, frechet_cell_step),
}


def get_cell_filter(cell_filter: "CellFilter | tuple") -> CellFilterFuncs:
    """Resolve cell filter into a tuple of init and update functions."""
    if isinstance(cell_filter, CellFilter):
        return CELL_FILTER_REGISTRY[cell_filter]
    if (
        isinstance(cell_filter, tuple)
        and len(cell_filter) == 2
        and all(map(callable, cell_filter))
    ):
        return cell_filter
    raise ValueError(
        f"Unknown {cell_filter=}, must be one of {list(map(str, CellFilter))} or "
        "2-tuple of callables"
    )


@dataclass(kw_only=True)
class CellOptimState(OptimState):
    """State class for cell optimization."""

    reference_cell: torch.Tensor
    cell_filter: CellFilterFuncs
    cell_factor: torch.Tensor = field(default_factory=lambda: None)
    pressure: torch.Tensor = field(default_factory=lambda: None)
    hydrostatic_strain: bool = False
    constant_volume: bool = False
    cell_positions: torch.Tensor = field(default_factory=lambda: None)
    cell_forces: torch.Tensor = field(default_factory=lambda: None)
    cell_masses: torch.Tensor = field(default_factory=lambda: None)

    _system_attributes = OptimState._system_attributes | {  # noqa: SLF001
        "cell_factor",
        "pressure",
        "cell_positions",
        "cell_forces",
        "cell_masses",
        "reference_cell",
        "cell_filter",
    }
    _global_attributes = OptimState._global_attributes | {  # noqa: SLF001
        "hydrostatic_strain",
        "constant_volume",
    }


@dataclass(kw_only=True)
class CellFireState(CellOptimState, FireState):
    """State class for FIRE optimization with cell optimization."""

    cell_velocities: torch.Tensor = field(default_factory=lambda: None)

    _system_attributes = (
        CellOptimState._system_attributes  # noqa: SLF001
        | FireState._system_attributes  # noqa: SLF001
        | {"cell_velocities"}
    )


@dataclass(kw_only=True)
class CellBFGSState(CellOptimState, BFGSState):
    """State class for BFGS optimization with cell optimization.

    Combines BFGS position optimization with cell filter for simultaneous
    optimization of atomic positions and unit cell parameters using a unified
    extended coordinate space (positions + cell DOFs).
    """

    # Previous cell state for Hessian update
    prev_cell_positions: torch.Tensor = field(default_factory=lambda: None)
    prev_cell_forces: torch.Tensor = field(default_factory=lambda: None)

    _atom_attributes = (
        CellOptimState._atom_attributes  # noqa: SLF001
        | BFGSState._atom_attributes  # noqa: SLF001
    )
    _system_attributes = (
        CellOptimState._system_attributes  # noqa: SLF001
        | BFGSState._system_attributes  # noqa: SLF001
        | {"prev_cell_positions", "prev_cell_forces"}
    )
    _global_attributes = (
        CellOptimState._global_attributes  # noqa: SLF001
        | BFGSState._global_attributes  # noqa: SLF001
    )


@dataclass(kw_only=True)
class CellLBFGSState(CellOptimState, LBFGSState):
    """State class for L-BFGS optimization with cell optimization.

    Combines L-BFGS position optimization with cell filter for simultaneous
    optimization of atomic positions and unit cell parameters using a unified
    extended coordinate space (positions + cell DOFs).
    """

    # Previous cell state for history update
    prev_cell_positions: torch.Tensor = field(default_factory=lambda: None)
    prev_cell_forces: torch.Tensor = field(default_factory=lambda: None)

    _atom_attributes = (
        CellOptimState._atom_attributes  # noqa: SLF001
        | LBFGSState._atom_attributes  # noqa: SLF001
    )
    _system_attributes = (
        CellOptimState._system_attributes  # noqa: SLF001
        | LBFGSState._system_attributes  # noqa: SLF001
        | {"prev_cell_positions", "prev_cell_forces"}
    )
    _global_attributes = (
        CellOptimState._global_attributes  # noqa: SLF001
        | LBFGSState._global_attributes  # noqa: SLF001
    )


AnyCellState = CellFireState | CellOptimState | CellBFGSState | CellLBFGSState
