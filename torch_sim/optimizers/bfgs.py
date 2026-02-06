"""BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer implementation.

This module provides a batched BFGS optimizer that maintains the full Hessian
matrix for each system. This is suitable for systems with a small to moderate
number of atoms, where the $O(N^2)$ memory cost is acceptable.

The implementation handles batches of systems with different numbers of atoms
by padding vectors to the maximum number of atoms in the batch. The Hessian
matrices are similarly padded to shape (n_systems, 3*max_atoms, 3*max_atoms).

Note: When cell_filter is active, forces are transformed using the deformation gradient
to work in the same scaled coordinate space as ASE's UnitCellFilter/FrechetCellFilter.
The prev_forces and prev_positions are stored in the scaled/fractional space to match
ASE's behavior.
"""

from typing import TYPE_CHECKING, Any

import torch

import torch_sim as ts
from torch_sim.optimizers import cell_filters
from torch_sim.optimizers.cell_filters import frechet_cell_filter_init
from torch_sim.state import SimState
from torch_sim.typing import StateDict


if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import BFGSState, CellBFGSState
    from torch_sim.optimizers.cell_filters import CellFilter, CellFilterFuncs


def _get_atom_indices_per_system(
    system_idx: torch.Tensor, n_systems: int
) -> torch.Tensor:
    """Compute the index of each atom within its system.

    Assumes atoms are grouped contiguously by system.

    Args:
        system_idx: Tensor of system indices [n_atoms]
        n_systems: Number of systems

    Returns:
        Tensor of [0, 1, 2, ..., 0, 1, ...] [n_atoms]
    """
    # We assume contiguous atoms for each system, which is standard in SimState
    counts = torch.bincount(system_idx, minlength=n_systems)
    # Create ranges [0...n-1] for each system and concatenate
    indices = [torch.arange(c, device=system_idx.device) for c in counts]
    return torch.cat(indices)


def _pad_to_dense(
    flat_tensor: torch.Tensor,
    system_idx: torch.Tensor,
    atom_idx_in_system: torch.Tensor,
    n_systems: int,
    max_atoms: int,
) -> torch.Tensor:
    """Convert a packed tensor to a padded dense tensor.

    Args:
        flat_tensor: [n_atoms, D]
        system_idx: [n_atoms]
        atom_idx_in_system: [n_atoms]
        n_systems: int
        max_atoms: int

    Returns:
        dense_tensor: [n_systems, max_atoms, D]
    """
    D = flat_tensor.shape[1]
    dense = torch.zeros(
        (n_systems, max_atoms, D), dtype=flat_tensor.dtype, device=flat_tensor.device
    )
    dense[system_idx, atom_idx_in_system] = flat_tensor
    return dense


def bfgs_init(
    state: SimState | StateDict,
    model: "ModelInterface",
    *,
    max_step: float = 0.2,
    alpha: float = 70.0,
    cell_filter: "CellFilter | CellFilterFuncs | None" = None,
    **filter_kwargs: Any,
) -> "BFGSState | CellBFGSState":
    """Create an initial BFGSState.

    Initializes the Hessian as Identity matrix * alpha.

    Shape notation:
        N = total atoms across all systems (n_atoms)
        S = number of systems (n_systems)
        M = max atoms per system (max_atoms)
        D = 3*M (position DOFs)
        D_ext = 3*M + 9 (extended DOFs with cell)

    Args:
        state: Input state
        model: Model
        max_step: Maximum step size (Angstrom)
        alpha: Initial Hessian stiffness (eV/A^2)
        cell_filter: Filter for cell optimization (None for position-only optimization)
        **filter_kwargs: Additional arguments passed to cell filter initialization

    Returns:
        BFGSState or CellBFGSState if cell_filter is provided
    """
    from torch_sim.optimizers import BFGSState, CellBFGSState

    tensor_args = {"device": model.device, "dtype": model.dtype}

    if not isinstance(state, SimState):
        state = SimState(**state)

    n_systems = state.n_systems  # S

    counts = state.n_atoms_per_system  # [S]
    global_max_atoms = int(counts.max().item()) if len(counts) > 0 else 0  # M
    # Per-system max_atoms for padding/unpadding support
    max_atoms = counts.clone()  # [S] - each system's atom count
    atom_idx = _get_atom_indices_per_system(state.system_idx, n_systems)  # [N]

    model_output = model(state)
    energy = model_output["energy"]  # [S]
    forces = model_output["forces"]  # [N, 3]
    stress = model_output.get("stress")  # [S, 3, 3] or None

    alpha_t = torch.full((n_systems,), alpha, **tensor_args)  # [S]
    max_step_t = torch.full((n_systems,), max_step, **tensor_args)  # [S]
    n_iter = torch.zeros((n_systems,), device=model.device, dtype=torch.int32)  # [S]

    if cell_filter is not None:
        # Extended Hessian: (3*global_max_atoms + 9) x (3*global_max_atoms + 9)
        # The extra 9 DOFs are for cell parameters (3x3 matrix flattened)
        dim = 3 * global_max_atoms + (3 * 3)  # D_ext
        hessian = (
            torch.eye(dim, **tensor_args).unsqueeze(0).repeat(n_systems, 1, 1) * alpha
        )  # [S, D_ext, D_ext]

        cell_filter_funcs = init_fn, _step_fn = ts.get_cell_filter(cell_filter)

        # Note (AG): At initialization, deform_grad is identity, so we have:
        # fractional = Cartesian / cell and scaled forces = forces @ I = forces
        # For ASE compatibility, we need to store prev_positions as fractional coords
        # and prev_forces as scaled forces

        # Get initial deform_grad (identity at start since reference_cell = current_cell)
        reference_cell = state.cell.clone()  # [S, 3, 3]
        cur_deform_grad = cell_filters.deform_grad(
            reference_cell.mT, state.cell.mT
        )  # [S, 3, 3]

        # Initial fractional positions = solve(deform_grad, positions) = positions
        # cur_deform_grad[system_idx]: [N, 3, 3], positions: [N, 3]
        frac_positions = torch.linalg.solve(
            cur_deform_grad[state.system_idx],  # [N, 3, 3]
            state.positions.unsqueeze(-1),  # [N, 3, 1]
        ).squeeze(-1)  # [N, 3]

        # Initial scaled forces = forces @ deform_grad = forces
        # forces: [N, 3], cur_deform_grad[system_idx]: [N, 3, 3] -> [N, 3]
        scaled_forces = torch.bmm(
            forces.unsqueeze(1),  # [N, 1, 3]
            cur_deform_grad[state.system_idx],  # [N, 3, 3]
        ).squeeze(1)

        common_args = {
            "positions": state.positions.clone(),  # [N, 3]
            "masses": state.masses.clone(),  # [N]
            "cell": state.cell.clone(),  # [S, 3, 3]
            "atomic_numbers": state.atomic_numbers.clone(),  # [N]
            "forces": forces,  # [N, 3]
            "energy": energy,  # [S]
            "stress": stress,  # [S, 3, 3] or None
            "hessian": hessian,  # [S, D_ext, D_ext]
            # Note (AG): Store fractional positions and scaled forces
            # for ASE compatibility
            "prev_forces": scaled_forces,  # [N, 3] (scaled)
            "prev_positions": frac_positions,  # [N, 3] (fractional)
            "alpha": alpha_t,  # [S]
            "max_step": max_step_t,  # [S]
            "n_iter": n_iter,  # [S]
            "atom_idx_in_system": atom_idx,  # [N]
            "max_atoms": max_atoms,  # scalar M
            "system_idx": state.system_idx.clone(),  # [N]
            "pbc": state.pbc,  # [S, 3]
            "reference_cell": reference_cell,  # [S, 3, 3]
            "cell_filter": cell_filter_funcs,
            "charge": state.charge,  # preserve charge
            "spin": state.spin,  # preserve spin
            "_constraints": state.constraints,  # preserve constraints
        }

        cell_state = CellBFGSState(**common_args)

        # Initialize cell-specific attributes (cell_positions, cell_forces, etc.)
        # After init: cell_positions [S, 3, 3], cell_forces [S, 3, 3], cell_factor [S]
        init_fn(cell_state, model, **filter_kwargs)

        # Store prev_cell_positions and prev_cell_forces for Hessian update
        cell_state.prev_cell_positions = cell_state.cell_positions.clone()  # [S, 3, 3]
        cell_state.prev_cell_forces = cell_state.cell_forces.clone()  # [S, 3, 3]

        return cell_state

    # Position-only Hessian: 3*global_max_atoms x 3*global_max_atoms
    dim = 3 * global_max_atoms  # D
    hessian = (
        torch.eye(dim, **tensor_args).unsqueeze(0).repeat(n_systems, 1, 1) * alpha
    )  # [S, D, D]

    common_args = {
        "positions": state.positions.clone(),  # [N, 3]
        "masses": state.masses.clone(),  # [N]
        "cell": state.cell.clone(),  # [S, 3, 3]
        "atomic_numbers": state.atomic_numbers.clone(),  # [N]
        "forces": forces,  # [N, 3]
        "energy": energy,  # [S]
        "stress": stress,  # [S, 3, 3] or None
        "hessian": hessian,  # [S, D, D]
        "prev_forces": forces.clone(),  # [N, 3]
        "prev_positions": state.positions.clone(),  # [N, 3]
        "alpha": alpha_t,  # [S]
        "max_step": max_step_t,  # [S]
        "n_iter": n_iter,  # [S]
        "atom_idx_in_system": atom_idx,  # [N]
        "max_atoms": max_atoms,  # scalar M
        "system_idx": state.system_idx.clone(),  # [N]
        "pbc": state.pbc,  # [S, 3]
        "charge": state.charge,  # preserve charge
        "spin": state.spin,  # preserve spin
        "_constraints": state.constraints,  # preserve constraints
    }

    return BFGSState(**common_args)


def bfgs_step(  # noqa: C901, PLR0915
    state: "BFGSState | CellBFGSState",
    model: "ModelInterface",
) -> "BFGSState | CellBFGSState":
    """Perform one BFGS optimization step.

    Updates the Hessian estimate and moves atoms. If state is a CellBFGSState,
    forces are transformed using the deformation gradient to work in the same
    scaled coordinate space as ASE's cell filters (matching FIRE's approach).

    For cell optimization, prev_positions are stored as fractional coordinates
    and prev_forces as scaled forces, exactly matching ASE's pos0/forces0.

    Shape notation:
        N = total atoms across all systems (n_atoms)
        S = number of systems (n_systems)
        M = max atoms per system (max_atoms)
        D = 3*M (position DOFs)
        D_ext = 3*M + 9 (extended DOFs with cell)

    Args:
        state: Current optimization state
        model: Calculator model

    Returns:
        Updated state
    """
    from torch_sim.optimizers import CellBFGSState

    # Note (AG): eps kept same as ASE's BFGS.
    eps = 1e-7
    is_cell_state = isinstance(state, CellBFGSState)

    # Derive global_max_atoms from hessian shape
    hessian_dim = state.hessian.shape[1]
    global_max_atoms = (hessian_dim - 9) // 3 if is_cell_state else hessian_dim // 3

    if is_cell_state:
        # Get current deformation gradient
        # reference_cell.mT: [S, 3, 3], row_vector_cell: [S, 3, 3]
        cur_deform_grad = cell_filters.deform_grad(
            state.reference_cell.mT, state.row_vector_cell
        )  # [S, 3, 3]

        # Transform forces to scaled coordinates
        # forces: [N, 3], cur_deform_grad[system_idx]: [N, 3, 3]
        forces_scaled = torch.bmm(
            state.forces.unsqueeze(1),  # [N, 1, 3]
            cur_deform_grad[state.system_idx],  # [N, 3, 3]
        ).squeeze(1)  # [N, 3]

        # Current fractional positions
        # positions: [N, 3] -> frac_positions: [N, 3]
        frac_positions = torch.linalg.solve(
            cur_deform_grad[state.system_idx],  # [N, 3, 3]
            state.positions.unsqueeze(-1),  # [N, 3, 1]
        ).squeeze(-1)  # [N, 3]

        # Pack into dense tensors [N, 3] -> [S, M, 3] -> [S, D]
        # For cell state, prev_positions is already fractional (stored that way)
        # prev_forces is already scaled
        # Note (AG): Optimization potential here.
        forces_new = _pad_to_dense(
            forces_scaled,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        forces_old = _pad_to_dense(
            state.prev_forces,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        pos_new = _pad_to_dense(
            frac_positions,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        pos_old = _pad_to_dense(
            state.prev_positions,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        # Extend with cell DOFs: [S, 3, 3] -> [S, 9]
        cell_pos_new = state.cell_positions.reshape(state.n_systems, 9)  # [S, 9]
        cell_forces_new = state.cell_forces.reshape(state.n_systems, 9)  # [S, 9]
        cell_pos_old = state.prev_cell_positions.reshape(state.n_systems, 9)  # [S, 9]
        cell_forces_old = state.prev_cell_forces.reshape(state.n_systems, 9)  # [S, 9]

        # Concatenate: extended = [positions, cell_positions]
        # [S, D] + [S, 9] -> [S, D_ext]
        pos_new = torch.cat([pos_new, cell_pos_new], dim=1)  # [S, D_ext]
        forces_new = torch.cat([forces_new, cell_forces_new], dim=1)  # [S, D_ext]
        pos_old = torch.cat([pos_old, cell_pos_old], dim=1)  # [S, D_ext]
        forces_old = torch.cat([forces_old, cell_forces_old], dim=1)  # [S, D_ext]
    else:
        forces_scaled = state.forces  # [N, 3]

        # Pack into dense tensors [N, 3] -> [S, M, 3] -> [S, D]
        forces_new = _pad_to_dense(
            state.forces,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        forces_old = _pad_to_dense(
            state.prev_forces,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        pos_new = _pad_to_dense(
            state.positions,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

        pos_old = _pad_to_dense(
            state.prev_positions,  # [N, 3]
            state.system_idx,
            state.atom_idx_in_system,
            state.n_systems,
            global_max_atoms,
        ).reshape(state.n_systems, -1)  # [S, D]

    # Calculate displacements and force changes
    # dim = D or D_ext depending on cell_state
    dpos = pos_new - pos_old  # [S, dim]
    dforces = forces_new - forces_old  # [S, dim]

    # Identify systems with significant movement
    max_disp = torch.max(torch.abs(dpos), dim=1).values  # [S]
    update_mask = max_disp >= eps  # [S] bool

    # Update Hessian for active systems (BFGS update formula)
    if update_mask.any():
        idx = update_mask
        H = state.hessian[idx]  # [S_active, dim, dim]

        dp = dpos[idx].unsqueeze(2)  # [S_active, dim, 1]
        df = dforces[idx].unsqueeze(2)  # [S_active, dim, 1]

        # a = dp^T @ df: [S_active, 1, dim] @ [S_active, dim, 1] -> [S_active, 1, 1]
        a = torch.bmm(dp.transpose(1, 2), df).squeeze(2)  # [S_active, 1]
        # dg = H @ dp: [S_active, dim, dim] @ [S_active, dim, 1] -> [S_active, dim, 1]
        dg = torch.bmm(H, dp)  # [S_active, dim, 1]
        # b = dp^T @ dg: [S_active, 1, dim] @ [S_active, dim, 1] -> [S_active, 1, 1]
        b = torch.bmm(dp.transpose(1, 2), dg).squeeze(2)  # [S_active, 1]

        # term1 = df @ df^T / a: [S_active, dim, dim]
        term1 = torch.bmm(df, df.transpose(1, 2)) / (a.unsqueeze(2) + 1e-30)
        # term2 = dg @ dg^T / b: [S_active, dim, dim]
        term2 = torch.bmm(dg, dg.transpose(1, 2)) / (b.unsqueeze(2) + 1e-30)

        state.hessian[idx] = H - term1 - term2  # [S_active, dim, dim]

    # Calculate step direction using eigendecomposition
    # step = V @ (|omega|^-1) @ V^T @ forces (pseudo-inverse via eigendecomposition)
    # Note (AG): We use eigendecomposition rather than directly inverting H so we can
    # take the absolute value of eigenvalues (|omega|). This ensures the step is always
    # in a descent direction even if the Hessian approximation has negative eigenvalues.

    # Size-binned eigendecomposition: group systems by actual Hessian size
    hessian_dim = state.hessian.shape[1]
    step_dense = torch.zeros(
        state.n_systems, hessian_dim, device=state.device, dtype=state.dtype
    )  # [S, dim]

    # Get unique sizes and process each group with batched eigendecomp
    # TODO(AG): If we sort and get the sizes before hand we can reduce the
    # python loop overhead.
    unique_sizes = state.max_atoms.unique()

    for size in unique_sizes:
        actual_dim = int(3 * size.item()) + (9 if is_cell_state else 0)
        mask = state.max_atoms == size  # [S] bool - systems with this size

        # Extract actual-sized Hessians and forces for this group
        H_group = state.hessian[mask, :actual_dim, :actual_dim]  # [G, d, d]
        f_group = forces_new[mask, :actual_dim]  # [G, d]

        # Batched eigendecomposition on actual size (no padding overhead)
        omega, V = torch.linalg.eigh(H_group)  # omega: [G, d], V: [G, d, d]
        abs_omega = torch.abs(omega).clamp(min=1e-30)  # [G, d]

        # Compute step: V @ (V^T @ f / |omega|)
        vt_f = torch.bmm(V.transpose(1, 2), f_group.unsqueeze(2))  # [G, d, 1]
        step_group = torch.bmm(V, vt_f / abs_omega.unsqueeze(2)).squeeze(2)  # [G, d]

        # Place results back into step_dense (padded to hessian_dim)
        indices = mask.nonzero(as_tuple=True)[0]
        step_dense[indices, :actual_dim] = step_group

    # Split step into position and cell components
    atom_dim = 3 * global_max_atoms  # D
    if is_cell_state:
        step_pos = step_dense[:, :atom_dim]  # [S, D]
        step_cell = step_dense[:, atom_dim:]  # [S, 9]
    else:
        step_pos = step_dense  # [S, D]

    # Scale step if it exceeds max_step
    step_atoms = step_pos.view(state.n_systems, global_max_atoms, 3)  # [S, M, 3]
    atom_norms = torch.norm(step_atoms, dim=2)  # [S, M]

    if is_cell_state:
        step_cell_reshaped = step_cell.view(state.n_systems, 3, 3)  # [S, 3, 3]
        cell_norms = torch.norm(step_cell_reshaped, dim=2)  # [S, 3]
        all_norms = torch.cat([atom_norms, cell_norms], dim=1)  # [S, M+3]
        max_disp_per_sys = torch.max(all_norms, dim=1).values  # [S]
    else:
        max_disp_per_sys = torch.max(atom_norms, dim=1).values  # [S]

    scale = torch.ones_like(max_disp_per_sys)  # [S]
    needs_scale = max_disp_per_sys > state.max_step  # [S] bool
    scale[needs_scale] = state.max_step[needs_scale] / (
        max_disp_per_sys[needs_scale] + 1e-30
    )

    step_pos = step_pos * scale.unsqueeze(1)  # [S, D]
    if is_cell_state:
        step_cell = step_cell * scale.unsqueeze(1)  # [S, 9]

    # Unpack dense step to flat: [S, M, 3] -> [N, 3]
    flat_step = step_pos.view(state.n_systems, global_max_atoms, 3)[
        state.system_idx, state.atom_idx_in_system
    ]  # [N, 3]

    # Save previous state for next Hessian update
    # For cell state: store fractional positions and scaled forces (ASE convention)
    if is_cell_state:
        state.prev_positions = frac_positions.clone()  # [N, 3] (fractional)
        state.prev_forces = forces_scaled.clone()  # [N, 3] (scaled)
        state.prev_cell_positions = state.cell_positions.clone()  # [S, 3, 3]
        state.prev_cell_forces = state.cell_forces.clone()  # [S, 3, 3]

        # Apply cell step: [S, 9] -> [S, 3, 3]
        dr_cell = step_cell.view(state.n_systems, 3, 3)  # [S, 3, 3]
        cell_positions_new = state.cell_positions + dr_cell  # [S, 3, 3]
        state.cell_positions = cell_positions_new  # [S, 3, 3]

        # Determine if Frechet filter
        init_fn, _step_fn = state.cell_filter
        is_frechet = init_fn is frechet_cell_filter_init

        if is_frechet:
            # Frechet: deform_grad = exp(cell_positions / cell_factor)
            cell_factor_reshaped = state.cell_factor.view(state.n_systems, 1, 1)
            deform_grad_log_new = cell_positions_new / cell_factor_reshaped  # [S, 3, 3]
            deform_grad_new = torch.matrix_exp(deform_grad_log_new)  # [S, 3, 3]
        else:
            # UnitCell: deform_grad = cell_positions / cell_factor
            cell_factor_expanded = state.cell_factor.expand(state.n_systems, 3, 1)
            deform_grad_new = cell_positions_new / cell_factor_expanded  # [S, 3, 3]

        # Update cell: new_cell = reference_cell @ deform_grad^T
        # reference_cell.mT: [S, 3, 3], deform_grad_new: [S, 3, 3]
        state.row_vector_cell = torch.bmm(
            state.reference_cell.mT, deform_grad_new.transpose(-2, -1)
        )  # [S, 3, 3]

        # Apply position step in fractional space, then convert to Cartesian
        new_frac = frac_positions + flat_step  # [N, 3]

        new_deform_grad = cell_filters.deform_grad(
            state.reference_cell.mT, state.row_vector_cell
        )  # [S, 3, 3]
        # new_positions = new_frac @ deform_grad^T
        new_positions = torch.bmm(
            new_frac.unsqueeze(1),  # [N, 1, 3]
            new_deform_grad[state.system_idx].transpose(-2, -1),  # [N, 3, 3]
        ).squeeze(1)  # [N, 3]
        state.set_constrained_positions(new_positions)  # [N, 3]
    else:
        state.prev_positions = state.positions.clone()  # [N, 3]
        state.prev_forces = state.forces.clone()  # [N, 3]
        state.set_constrained_positions(state.positions + flat_step)  # [N, 3]

    # Evaluate new forces and energy
    model_output = model(state)
    state.set_constrained_forces(model_output["forces"])  # [N, 3]
    state.energy = model_output["energy"]  # [S]
    if "stress" in model_output:
        state.stress = model_output["stress"]  # [S, 3, 3]

    # Update cell forces for next step
    # Update cell forces for cell state: [S, 3, 3]
    if is_cell_state:
        cell_filters.compute_cell_forces(model_output, state)

    state.n_iter += 1

    return state
