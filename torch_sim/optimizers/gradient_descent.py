"""Gradient descent optimizer implementation."""

from typing import TYPE_CHECKING, Any

import torch

import torch_sim as ts
from torch_sim.optimizers import cell_filters
from torch_sim.state import SimState
from torch_sim.typing import StateDict


if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import CellOptimState, OptimState
    from torch_sim.optimizers.cell_filters import CellFilter, CellFilterFuncs


def gradient_descent_init(
    state: SimState | StateDict,
    model: "ModelInterface",
    *,
    cell_filter: "CellFilter | CellFilterFuncs | None" = None,
    **filter_kwargs: Any,
) -> "OptimState | CellOptimState":
    """Initialize a gradient descent optimization state.

    Args:
        model: Model that computes energies, forces, and optionally stress
        state: SimState containing positions, masses, cell, etc.
        cell_filter: Filter for cell optimization (None for position-only optimization)
        **filter_kwargs: Additional arguments passed to cell filter initialization

    Returns:
        Initialized OptimState with forces, energy, and optional cell state

    Notes:
        Use cell_filter=None for position-only optimization.
        Use cell_filter=UNIT_CELL_FILTER or FRECHET_CELL_FILTER for cell optimization.
    """
    # Import here to avoid circular imports
    from torch_sim.optimizers import CellOptimState, OptimState

    if not isinstance(state, SimState):
        state = SimState(**state)

    # Get initial forces and energy from model
    model_output = model(state)
    energy = model_output["energy"]
    forces = model_output["forces"]
    stress = model_output.get("stress")

    # Common state arguments
    common_args = {
        "positions": state.positions,
        "forces": forces,
        "energy": energy,
        "stress": stress,
        "masses": state.masses,
        "cell": state.cell,
        "pbc": state.pbc,
        "atomic_numbers": state.atomic_numbers,
        "system_idx": state.system_idx,
        "_constraints": state.constraints,
    }

    if cell_filter is not None:  # Create cell optimization state
        cell_filter_funcs = init_fn, _step_fn = ts.get_cell_filter(cell_filter)
        common_args["reference_cell"] = state.cell.clone()
        common_args["cell_filter"] = cell_filter_funcs
        cell_state = CellOptimState(**common_args)

        # Initialize cell-specific attributes
        init_fn(cell_state, model, **filter_kwargs)

        return cell_state
    # Create regular OptimState without cell optimization
    return OptimState(**common_args)


def gradient_descent_step(
    state: "OptimState | CellOptimState",
    model: "ModelInterface",
    *,
    pos_lr: float | torch.Tensor = 0.01,
    cell_lr: float | torch.Tensor = 0.1,
) -> "OptimState | CellOptimState":
    """Perform one gradient descent optimization step.

    Updates atomic positions and optionally cell parameters based on the filter.

    Args:
        model: Model that computes energies, forces, and optionally stress
        state: Current optimization state
        pos_lr: Learning rate(s) for atomic positions
        cell_lr: Learning rate(s) for cell optimization (ignored if no cell filter)

    Returns:
        Updated OptimState after one optimization step
    """
    from torch_sim.optimizers import CellOptimState

    device, dtype = model.device, model.dtype

    # Get per-atom learning rates
    if isinstance(pos_lr, (int, float)):
        pos_lr = torch.full((state.n_systems,), pos_lr, device=device, dtype=dtype)
    atom_lr = pos_lr[state.system_idx].unsqueeze(-1)

    # Update atomic positions
    # state.positions = state.positions + atom_lr * state.forces
    state.set_positions(state.positions + atom_lr * state.forces)

    # Update cell if using cell optimization
    if isinstance(state, CellOptimState):
        # Compute cell step and update cell
        _init_fn, step_fn = state.cell_filter
        step_fn(state, cell_lr)

    # Get updated forces, energy, and stress
    model_output = model(state)
    state.set_forces(model_output["forces"])
    state.energy = model_output["energy"]
    if "stress" in model_output:
        state.stress = model_output["stress"]

    # Update cell forces
    if isinstance(state, CellOptimState):
        cell_filters.compute_cell_forces(model_output, state)

    return state
