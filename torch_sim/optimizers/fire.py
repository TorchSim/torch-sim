"""FIRE (Fast Inertial Relaxation Engine) optimizer implementation."""

from typing import TYPE_CHECKING, Any, get_args

import torch

import torch_sim as ts
import torch_sim.math as fm
from torch_sim.optimizers import cell_filters
from torch_sim.state import SimState
from torch_sim.typing import StateDict


if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import FireFlavor, FireState
    from torch_sim.optimizers.cell_filters import (
        CellFilter,
        CellFilterFuncs,
        CellFireState,
    )


def fire_init(
    state: SimState | StateDict,
    model: "ModelInterface",
    *,
    dt_start: float = 0.1,
    alpha_start: float = 0.1,
    fire_flavor: "FireFlavor" = "ase_fire",
    cell_filter: "CellFilter | CellFilterFuncs | None" = None,
    **filter_kwargs: Any,
) -> "FireState | CellFireState":
    """Initialize a FIRE optimization state.

    Creates an optimizer that performs FIRE (Fast Inertial Relaxation Engine)
    optimization on atomic positions and optionally cell parameters.

    Args:
        model: Model that computes energies, forces, and optionally stress
        state: Input state as SimState object or state parameter dict
        dt_start: Initial timestep per system
        alpha_start: Initial mixing parameter per system
        fire_flavor: Optimization flavor ("vv_fire" or "ase_fire")
        cell_filter: Filter for cell optimization (None for position-only optimization)
        **filter_kwargs: Additional arguments passed to cell filter initialization

    Returns:
        FireState with initialized optimization tensors

    Notes:
        - fire_flavor="vv_fire" follows the original paper closely
        - fire_flavor="ase_fire" mimics the ASE implementation
        - Use cell_filter=UNIT_CELL_FILTER or FRECHET_CELL_FILTER for cell optimization
    """
    # Import here to avoid circular imports
    from torch_sim.optimizers import CellFireState, FireFlavor, FireState

    if fire_flavor not in get_args(FireFlavor):
        raise ValueError(f"Unknown {fire_flavor=}, must be one of {get_args(FireFlavor)}")

    tensor_args = dict(device=model.device, dtype=model.dtype)

    if not isinstance(state, SimState):
        state = SimState(**state)

    n_systems = state.n_systems

    # Get initial forces and energy from model
    model_output = model(state)
    energy = model_output["energy"]
    forces = model_output["forces"]
    stress = model_output.get("stress")

    # Common state arguments
    common_args = {
        # Copy SimState attributes
        "positions": state.positions.clone(),
        "masses": state.masses.clone(),
        "cell": state.cell.clone(),
        "atomic_numbers": state.atomic_numbers.clone(),
        "system_idx": state.system_idx.clone(),
        "_constraints": state.constraints,
        "pbc": state.pbc,
        # Optimization state
        "forces": forces,
        "energy": energy,
        "stress": stress,
        "velocities": torch.full(state.positions.shape, torch.nan, **tensor_args),
        # FIRE parameters
        "dt": torch.full((n_systems,), dt_start, **tensor_args),
        "alpha": torch.full((n_systems,), alpha_start, **tensor_args),
        "n_pos": torch.zeros((n_systems,), device=model.device, dtype=torch.int32),
    }

    if cell_filter is not None:  # Create cell optimization state
        cell_filter_funcs = init_fn, _step_fn = ts.get_cell_filter(cell_filter)
        common_args["reference_cell"] = state.cell.clone()
        common_args["cell_filter"] = cell_filter_funcs
        cell_state = CellFireState(**common_args)

        # Initialize cell-specific attributes
        init_fn(cell_state, model, **filter_kwargs)

        # Initialize cell velocities after cell_forces is set
        cell_state.cell_velocities = torch.full(
            cell_state.cell_forces.shape, torch.nan, **tensor_args
        )

        return cell_state
    # Create regular FireState without cell optimization
    return FireState(**common_args)


def fire_step(
    state: "FireState | CellFireState",
    model: "ModelInterface",
    *,
    dt_max: float = 1.0,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    max_step: float = 0.2,
    fire_flavor: "FireFlavor" = "ase_fire",
) -> "FireState | CellFireState":
    """Perform one FIRE optimization step.

    Args:
        model: Model that computes energies, forces, and optionally stress
        state: Current FIRE optimization state
        dt_max: Maximum allowed timestep
        n_min: Minimum steps before timestep increase
        f_inc: Factor for timestep increase when power is positive
        f_dec: Factor for timestep decrease when power is negative
        alpha_start: Initial velocity mixing parameter
        f_alpha: Factor for mixing parameter decrease
        max_step: Maximum distance an atom can move per iteration
        fire_flavor: Optimization flavor ("vv_fire" or "ase_fire")

    Returns:
        Updated FireState after one optimization step
    """
    # Import here to avoid circular imports
    from torch_sim.optimizers import FireFlavor, ase_fire_key, vv_fire_key

    if fire_flavor not in get_args(FireFlavor):
        raise ValueError(f"Unknown {fire_flavor=}, must be one of {get_args(FireFlavor)}")

    device, dtype = model.device, model.dtype
    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    dt_max, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step = (
        torch.as_tensor(p, device=device, dtype=dtype)
        for p in (dt_max, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step)
    )

    step_func_kwargs = dict(
        model=model,
        dt_max=dt_max,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        alpha_start=alpha_start,
        f_alpha=f_alpha,
        eps=eps,
    )
    if fire_flavor == ase_fire_key:
        step_func_kwargs["max_step"] = max_step

    step_func = {vv_fire_key: _vv_fire_step, ase_fire_key: _ase_fire_step}[fire_flavor]
    return step_func(state, **step_func_kwargs)


def _vv_fire_step[T: "FireState | CellFireState"](  # noqa: PLR0915
    state: T,
    model: "ModelInterface",
    *,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_inc: torch.Tensor,
    f_dec: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    eps: float,
) -> T:
    """Perform one Velocity-Verlet based FIRE optimization step."""
    from torch_sim.optimizers import CellFireState

    n_systems, device, dtype = state.n_systems, state.device, state.dtype

    # Initialize velocities if NaN
    nan_velocities = state.velocities.isnan().any(dim=1)
    if nan_velocities.any():
        state.velocities[nan_velocities] = torch.zeros_like(
            state.positions[nan_velocities]
        )
        if isinstance(state, CellFireState):  # update velocities to zero if NaN
            nan_cell_velocities = state.cell_velocities.isnan().any(dim=(1, 2))
            state.cell_velocities[nan_cell_velocities] = torch.zeros_like(
                state.cell_positions[nan_cell_velocities]
            )

    alpha_start_system = torch.full(
        (n_systems,), alpha_start.item(), device=device, dtype=dtype
    )

    # First half of velocity update
    atom_wise_dt = state.dt[state.system_idx].unsqueeze(-1)
    state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)

    # Position update
    state.set_constrained_positions(state.positions + atom_wise_dt * state.velocities)

    # Cell position updates are handled in the velocity update step above

    # Get new forces and energy
    model_output = model(state)
    state.set_constrained_forces(model_output["forces"])
    state.energy = model_output["energy"]
    if "stress" in model_output:
        state.stress = model_output["stress"]

    # Update cell forces
    if isinstance(state, CellFireState):
        cell_filters.compute_cell_forces(model_output, state)

    # Second half of velocity update
    state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
    if isinstance(state, CellFireState):
        cell_wise_dt = state.dt.view(n_systems, 1, 1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

    # Calculate power
    system_power = fm.batched_vdot(state.forces, state.velocities, state.system_idx)
    if isinstance(state, CellFireState):
        system_power += (state.cell_forces * state.cell_velocities).sum(dim=(1, 2))

    # Update dt, alpha, n_pos
    pos_mask_system = system_power > 0.0
    neg_mask_system = ~pos_mask_system

    state.n_pos[pos_mask_system] += 1
    inc_mask = (state.n_pos > n_min) & pos_mask_system
    state.dt[inc_mask] = torch.minimum(state.dt[inc_mask] * f_inc, dt_max)
    state.alpha[inc_mask] *= f_alpha

    state.dt[neg_mask_system] *= f_dec
    state.alpha[neg_mask_system] = alpha_start_system[neg_mask_system]
    state.n_pos[neg_mask_system] = 0

    # Velocity mixing
    v_scaling_system = fm.batched_vdot(
        state.velocities, state.velocities, state.system_idx
    )
    f_scaling_system = fm.batched_vdot(state.forces, state.forces, state.system_idx)

    if isinstance(state, CellFireState):
        v_scaling_system += state.cell_velocities.pow(2).sum(dim=(1, 2))
        f_scaling_system += state.cell_forces.pow(2).sum(dim=(1, 2))

        v_scaling_cell = torch.sqrt(v_scaling_system.view(n_systems, 1, 1))
        f_scaling_cell = torch.sqrt(f_scaling_system.view(n_systems, 1, 1))
        v_mixing_cell = state.cell_forces / (f_scaling_cell + eps) * v_scaling_cell

        alpha_cell_bc = state.alpha.view(n_systems, 1, 1)
        state.cell_velocities = torch.where(
            pos_mask_system.view(n_systems, 1, 1),
            (1.0 - alpha_cell_bc) * state.cell_velocities + alpha_cell_bc * v_mixing_cell,
            torch.zeros_like(state.cell_velocities),
        )

    v_scaling_atom = torch.sqrt(v_scaling_system[state.system_idx].unsqueeze(-1))
    f_scaling_atom = torch.sqrt(f_scaling_system[state.system_idx].unsqueeze(-1))
    v_mixing_atom = state.forces * (v_scaling_atom / (f_scaling_atom + eps))

    alpha_atom = state.alpha[state.system_idx].unsqueeze(-1)
    state.velocities = torch.where(
        pos_mask_system[state.system_idx].unsqueeze(-1),
        (1.0 - alpha_atom) * state.velocities + alpha_atom * v_mixing_atom,
        torch.zeros_like(state.velocities),
    )

    return state


def _ase_fire_step[T: "FireState | CellFireState"](  # noqa: C901, PLR0915
    state: T,
    model: "ModelInterface",
    *,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_inc: torch.Tensor,
    f_dec: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    max_step: torch.Tensor,
    eps: float,
) -> T:
    """Perform one ASE-style FIRE optimization step."""
    from torch_sim.optimizers import CellFireState

    n_systems, device, dtype = state.n_systems, state.device, state.dtype

    # Initialize velocities if NaN
    nan_velocities = state.velocities.isnan().any(dim=1)
    if nan_velocities.any():
        state.velocities[nan_velocities] = torch.zeros_like(
            state.velocities[nan_velocities]
        )
        forces = state.forces
        if isinstance(state, CellFireState):
            nan_cell_velocities = state.cell_velocities.isnan().any(dim=(1, 2))
            state.cell_velocities[nan_cell_velocities] = torch.zeros_like(
                state.cell_velocities[nan_cell_velocities]
            )
    else:
        alpha_start_system = torch.full(
            (n_systems,), alpha_start.item(), device=device, dtype=dtype
        )

        # Transform forces for cell optimization
        if isinstance(state, CellFireState):
            # Get deformation gradient for force transformation
            cur_deform_grad = cell_filters.deform_grad(
                state.row_vector_cell,
                getattr(state, "reference_row_vector_cell", state.row_vector_cell),
            )
            forces = torch.bmm(
                state.forces.unsqueeze(1), cur_deform_grad[state.system_idx]
            ).squeeze(1)
        else:
            forces = state.forces

        # Calculate power
        system_power = fm.batched_vdot(forces, state.velocities, state.system_idx)
        if isinstance(state, CellFireState):
            system_power += (state.cell_forces * state.cell_velocities).sum(dim=(1, 2))

        # Update dt, alpha, n_pos
        pos_mask_system = system_power > 0.0
        neg_mask_system = ~pos_mask_system

        inc_mask = (state.n_pos > n_min) & pos_mask_system
        state.dt[inc_mask] = torch.minimum(state.dt[inc_mask] * f_inc, dt_max)
        state.alpha[inc_mask] *= f_alpha
        state.n_pos[pos_mask_system] += 1

        state.dt[neg_mask_system] *= f_dec
        state.alpha[neg_mask_system] = alpha_start_system[neg_mask_system]
        state.n_pos[neg_mask_system] = 0

        # Velocity mixing BEFORE acceleration (ASE ordering)
        v_scaling_system = fm.batched_vdot(
            state.velocities, state.velocities, state.system_idx
        )
        f_scaling_system = fm.batched_vdot(forces, forces, state.system_idx)

        if isinstance(state, CellFireState):
            v_scaling_system += state.cell_velocities.pow(2).sum(dim=(1, 2))
            f_scaling_system += state.cell_forces.pow(2).sum(dim=(1, 2))

            v_scaling_cell = torch.sqrt(v_scaling_system.view(n_systems, 1, 1))
            f_scaling_cell = torch.sqrt(f_scaling_system.view(n_systems, 1, 1))
            v_mixing_cell = state.cell_forces / (f_scaling_cell + eps) * v_scaling_cell

            alpha_cell_bc = state.alpha.view(n_systems, 1, 1)
            state.cell_velocities = torch.where(
                pos_mask_system.view(n_systems, 1, 1),
                (1.0 - alpha_cell_bc) * state.cell_velocities
                + alpha_cell_bc * v_mixing_cell,
                torch.zeros_like(state.cell_velocities),
            )

        v_scaling_atom = torch.sqrt(v_scaling_system[state.system_idx].unsqueeze(-1))
        f_scaling_atom = torch.sqrt(f_scaling_system[state.system_idx].unsqueeze(-1))
        v_mixing_atom = forces * (v_scaling_atom / (f_scaling_atom + eps))

        alpha_atom = state.alpha[state.system_idx].unsqueeze(-1)
        state.velocities = torch.where(
            pos_mask_system[state.system_idx].unsqueeze(-1),
            (1.0 - alpha_atom) * state.velocities + alpha_atom * v_mixing_atom,
            torch.zeros_like(state.velocities),
        )

    # Acceleration (single forward-Euler, no mass for ASE FIRE)
    state.velocities += forces * state.dt[state.system_idx].unsqueeze(-1)
    dr_atom = state.velocities * state.dt[state.system_idx].unsqueeze(-1)
    dr_scaling_system = fm.batched_vdot(dr_atom, dr_atom, state.system_idx)

    if isinstance(state, CellFireState):
        state.cell_velocities += state.cell_forces * state.dt.view(n_systems, 1, 1)
        dr_cell = state.cell_velocities * state.dt.view(n_systems, 1, 1)

        dr_scaling_system += dr_cell.pow(2).sum(dim=(1, 2))
        dr_scaling_cell = torch.sqrt(dr_scaling_system).view(n_systems, 1, 1)
        dr_cell = torch.where(
            dr_scaling_cell > max_step,
            max_step * dr_cell / (dr_scaling_cell + eps),
            dr_cell,
        )

    dr_scaling_atom = torch.sqrt(dr_scaling_system)[state.system_idx].unsqueeze(-1)
    dr_atom = torch.where(
        dr_scaling_atom > max_step,
        max_step * dr_atom / (dr_scaling_atom + eps),
        dr_atom,
    )

    # Position updates
    if isinstance(state, CellFireState):
        # For cell optimization, handle both atomic and cell position updates
        # This follows the ASE FIRE implementation pattern
        # Transform atomic positions to fractional coordinates
        cur_deform_grad = cell_filters.deform_grad(
            state.reference_cell.mT, state.row_vector_cell
        )
        frac_positions = torch.linalg.solve(
            cur_deform_grad[state.system_idx], state.positions.unsqueeze(-1)
        ).squeeze(-1)
        new_frac_positions = frac_positions + dr_atom
        # Back to Cartesian coordinates
        new_positions = torch.bmm(
            new_frac_positions.unsqueeze(1), cur_deform_grad[state.system_idx]
        ).squeeze(1)
        state.set_constrained_positions(new_positions)

        # Update cell positions directly based on stored cell filter type
        if hasattr(state, "cell_filter") and state.cell_filter is not None:
            from torch_sim.optimizers.cell_filters import frechet_cell_filter_init

            init_fn, _step_fn = state.cell_filter
            is_frechet = init_fn is frechet_cell_filter_init

            # Update cell positions
            cell_positions_new = state.cell_positions + dr_cell
            state.cell_positions = cell_positions_new

            if is_frechet:  # Frechet: convert from log space to deformation gradient
                cell_factor_reshaped = state.cell_factor.view(state.n_systems, 1, 1)
                deform_grad_log_new = cell_positions_new / cell_factor_reshaped
                deform_grad_new = torch.matrix_exp(deform_grad_log_new)
            else:  # Unit cell: positions are scaled deformation gradient
                cell_factor_expanded = state.cell_factor.expand(state.n_systems, 3, 1)
                deform_grad_new = cell_positions_new / cell_factor_expanded

            # Compute new cell from deformation gradient
            new_col_vector_cell = torch.bmm(state.reference_cell, deform_grad_new)

            # Apply cell constraints and scale positions to preserve fractional coords
            state.set_constrained_cell(new_col_vector_cell, scale_atoms=True)
    else:
        state.set_constrained_positions(state.positions + dr_atom)

    # Get new forces, energy, and stress
    model_output = model(state)
    state.set_constrained_forces(model_output["forces"])
    state.energy = model_output["energy"]
    if "stress" in model_output:
        state.stress = model_output["stress"]

    # Update cell forces
    if isinstance(state, CellFireState):
        cell_filters.compute_cell_forces(model_output, state)

    return state
