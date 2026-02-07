"""L-BFGS (Limited-memory BFGS) optimizer implementation.

This module provides a batched L-BFGS optimizer for atomic structure relaxation.
L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
a limited history of position and gradient differences, making it memory-efficient
for large systems while achieving superlinear convergence near the minimum.

When cell_filter is active, forces are transformed using the deformation gradient
to work in the same scaled coordinate space as ASE's UnitCellFilter/FrechetCellFilter.
The prev_forces and prev_positions are stored in the scaled/fractional space to match
ASE's behavior exactly.
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
    from torch_sim.optimizers import CellLBFGSState, LBFGSState
    from torch_sim.optimizers.cell_filters import CellFilter, CellFilterFuncs


def _compute_atom_idx(system_idx: torch.Tensor, n_systems: int) -> torch.Tensor:
    """Compute per-system atom indices, vectorized.

    Args:
        system_idx: System index for each atom [N]
        n_systems: Number of systems S

    Returns:
        Tensor [N] with per-system atom indices
    """
    device = system_idx.device
    counts = torch.bincount(system_idx, minlength=n_systems)
    offsets = torch.zeros(n_systems, device=device, dtype=torch.long)
    if n_systems > 1:
        offsets[1:] = counts[:-1].cumsum(0)
    return torch.arange(len(system_idx), device=device) - offsets[system_idx]


def _atoms_to_padded(
    x: torch.Tensor,
    system_idx: torch.Tensor,
    n_systems: int,
    max_atoms: int,
) -> torch.Tensor:
    """Convert atom-indexed [N, 3] to padded per-system [S, M, 3].

    Args:
        x: Tensor of shape [N, 3] where N = total atoms
        system_idx: System index for each atom [N]
        n_systems: Number of systems S
        max_atoms: Maximum atoms per system M

    Returns:
        Tensor of shape [S, M, 3] with zeros for padding
    """
    device, dtype = x.device, x.dtype
    out = torch.zeros((n_systems, max_atoms, 3), device=device, dtype=dtype)
    atom_idx = _compute_atom_idx(system_idx, n_systems)
    out[system_idx, atom_idx] = x
    return out


def _padded_to_atoms(
    x: torch.Tensor,
    system_idx: torch.Tensor,
) -> torch.Tensor:
    """Convert padded per-system [S, M, 3] to atom-indexed [N, 3].

    Args:
        x: Tensor of shape [S, M, 3]
        system_idx: System index for each atom [N]

    Returns:
        Tensor of shape [N, 3]
    """
    n_systems = x.shape[0]
    atom_idx = _compute_atom_idx(system_idx, n_systems)
    return x[system_idx, atom_idx]  # [N, 3]


def _per_system_vdot(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-system dot product with padding mask.

    Args:
        a: Tensor of shape [S, M, 3]
        b: Tensor of shape [S, M, 3]
        mask: Boolean mask [S, M] where True = valid atom

    Returns:
        Tensor of shape [S] with per-system dot products
    """
    # Element-wise product then sum over atoms and coordinates
    prod = (a * b).sum(dim=-1)  # [S, M]
    prod = prod * mask.float()  # Zero out padded atoms
    return prod.sum(dim=-1)  # [S]


def lbfgs_init(
    state: SimState | StateDict,
    model: "ModelInterface",
    *,
    step_size: float = 0.1,
    alpha: float | None = None,
    cell_filter: "CellFilter | CellFilterFuncs | None" = None,
    **filter_kwargs: Any,
) -> "LBFGSState | CellLBFGSState":
    r"""Create an initial LBFGSState from a SimState or state dict.

    Initializes forces/energy, clears the (s, y) memory, and broadcasts the
    fixed step size to all systems.

    Shape notation:
        N = total atoms across all systems (n_atoms)
        S = number of systems (n_systems)
        M = max atoms per system (global_max_atoms)
        H = history length (starts at 0)
        M_ext = M + 3 (extended with cell DOFs per system)

    Args:
        state: Input state as SimState object or state parameter dict
        model: Model that computes energies, forces, and optionally stress
        step_size: Fixed per-system step length (damping factor).
            If using ASE mode (fixed alpha), set this to 1.0 (or your damping).
            If using dynamic mode (default), 0.1 is a safe starting point.
        alpha: Initial inverse Hessian stiffness guess (ASE parameter).
            If provided (e.g. 70.0), fixes H0 = 1/alpha for all steps (ASE-style).
            If None (default), H0 is updated dynamically (Standard L-BFGS).
        cell_filter: Filter for cell optimization (None for position-only optimization)
        **filter_kwargs: Additional arguments passed to cell filter initialization

    Returns:
        LBFGSState with initialized optimization tensors, or CellLBFGSState if
        cell_filter is provided

    Notes:
        The optimizer supports two modes of operation:
        1. **Standard L-BFGS (default)**: Set `alpha=None`. The inverse Hessian
           diagonal $H_0$ is updated dynamically at each step using the scaling
           $\gamma_k = (s^T y) / (y^T y)$. This is the standard behavior described
           by Nocedal & Wright.
        2. **ASE Compatibility Mode**: Set `alpha` (e.g. 70.0) and `step_size=1.0`.
           The inverse Hessian diagonal is fixed at $H_0 = 1/\alpha$ throughout the
           optimization, and the step is scaled by `step_size` (damping).
           This matches `ase.optimize.LBFGS(alpha=70.0, damping=1.0)`.
    """
    from torch_sim.optimizers import CellLBFGSState, LBFGSState

    tensor_args = {"device": model.device, "dtype": model.dtype}

    if not isinstance(state, SimState):
        state = SimState(**state)

    n_systems = state.n_systems  # S

    # Compute max atoms per system for per-system history storage
    counts = state.n_atoms_per_system  # [S]
    global_max_atoms = int(counts.max().item()) if len(counts) > 0 else 0  # M
    max_atoms = counts.clone()  # [S] - each system's atom count

    # Get initial forces and energy from model
    model_output = model(state)
    energy = model_output["energy"]  # [S]
    forces = model_output["forces"]  # [N, 3]
    stress = model_output.get("stress")  # [S, 3, 3] or None

    # Initialize empty per-system history tensors
    # History shape: [S, H, M, 3] where H=0 at start, M = global_max_atoms
    s_history = torch.zeros(
        (n_systems, 0, global_max_atoms, 3), **tensor_args
    )  # [S, 0, M, 3]
    y_history = torch.zeros(
        (n_systems, 0, global_max_atoms, 3), **tensor_args
    )  # [S, 0, M, 3]

    # Alpha tensor: 0.0 means dynamic, >0 means fixed
    alpha_val = 0.0 if alpha is None else alpha
    alpha_tensor = torch.full((n_systems,), alpha_val, **tensor_args)  # [S]

    common_args = {
        # Copy SimState attributes
        "positions": state.positions.clone(),  # [N, 3]
        "masses": state.masses.clone(),  # [N]
        "cell": state.cell.clone(),  # [S, 3, 3]
        "atomic_numbers": state.atomic_numbers.clone(),  # [N]
        "system_idx": state.system_idx.clone(),  # [N]
        "pbc": state.pbc,  # [S, 3]
        "charge": state.charge,  # preserve charge
        "spin": state.spin,  # preserve spin
        "_constraints": state.constraints,  # preserve constraints
        # Optimization state
        "forces": forces,  # [N, 3]
        "energy": energy,  # [S]
        "stress": stress,  # [S, 3, 3] or None
        # L-BFGS specific state
        "prev_forces": forces.clone(),  # [N, 3]
        "prev_positions": state.positions.clone(),  # [N, 3]
        "s_history": s_history,  # [S, 0, M, 3]
        "y_history": y_history,  # [S, 0, M, 3]
        "step_size": torch.full((n_systems,), step_size, **tensor_args),  # [S]
        "alpha": alpha_tensor,  # [S]
        "n_iter": torch.zeros((n_systems,), device=model.device, dtype=torch.int32),
        "max_atoms": max_atoms,  # [S] atoms per system for padding
    }

    if cell_filter is not None:
        cell_filter_funcs = init_fn, _step_fn = ts.get_cell_filter(cell_filter)

        # At initialization, deform_grad is identity since reference_cell = current_cell
        # Store prev_positions as fractional (same as Cartesian for identity deform_grad)
        # Store prev_forces as scaled (same as Cartesian for identity deform_grad)
        reference_cell = state.cell.clone()  # [S, 3, 3]
        cur_deform_grad = cell_filters.deform_grad(
            reference_cell.mT, state.cell.mT
        )  # [S, 3, 3]

        # Initial fractional positions = positions
        # cur_deform_grad[system_idx]: [N, 3, 3], positions: [N, 3] -> [N, 3]
        frac_positions = torch.linalg.solve(
            cur_deform_grad[state.system_idx],  # [N, 3, 3]
            state.positions.unsqueeze(-1),  # [N, 3, 1]
        ).squeeze(-1)  # [N, 3]

        # Initial scaled forces = forces @ deform_grad = forces
        # forces: [N, 3], cur_deform_grad[system_idx]: [N, 3, 3] -> [N, 3]
        scaled_forces = torch.bmm(
            forces.unsqueeze(1),  # [N, 1, 3]
            cur_deform_grad[state.system_idx],  # [N, 3, 3]
        ).squeeze(1)  # [N, 3]

        common_args["reference_cell"] = reference_cell  # [S, 3, 3]
        common_args["cell_filter"] = cell_filter_funcs
        # Store fractional positions and scaled forces for ASE compatibility
        common_args["prev_positions"] = frac_positions  # [N, 3]
        common_args["prev_forces"] = scaled_forces  # [N, 3]

        # Extended per-system history includes cell DOFs (3 "virtual atoms" per system)
        # History shape: [S, H, M+3, 3] where M = global_max_atoms
        extended_size_per_system = global_max_atoms + 3  # M_ext = M + 3
        common_args["s_history"] = torch.zeros(
            (n_systems, 0, extended_size_per_system, 3), **tensor_args
        )  # [S, 0, M_ext, 3]
        common_args["y_history"] = torch.zeros(
            (n_systems, 0, extended_size_per_system, 3), **tensor_args
        )  # [S, 0, M_ext, 3]

        cell_state = CellLBFGSState(**common_args)

        # Initialize cell-specific attributes
        # After init: cell_positions [S, 3, 3], cell_forces [S, 3, 3], cell_factor [S]
        init_fn(cell_state, model, **filter_kwargs)

        # Store prev_cell_positions and prev_cell_forces for history update
        cell_state.prev_cell_positions = cell_state.cell_positions.clone()  # [S, 3, 3]
        cell_state.prev_cell_forces = cell_state.cell_forces.clone()  # [S, 3, 3]

        return cell_state

    return LBFGSState(**common_args)


def lbfgs_step(  # noqa: PLR0915, C901
    state: "LBFGSState | CellLBFGSState",
    model: "ModelInterface",
    *,
    max_history: int = 20,
    max_step: float = 0.2,
    curvature_eps: float = 1e-12,
) -> "LBFGSState | CellLBFGSState":
    r"""Advance one L-BFGS iteration using the two-loop recursion.

    Computes the search direction via the two-loop recursion, applies a
    fixed step with optional per-system capping, evaluates new forces and
    energy, and updates the limited-memory history with a curvature check.

    When cell_filter is active, forces are transformed using the deformation
    gradient to work in the same scaled coordinate space as ASE's cell filters.
    The prev_positions are stored as fractional coordinates and prev_forces as
    scaled forces, exactly matching ASE's pos0/forces0.

    Shape notation:
        N = total atoms across all systems (n_atoms)
        S = number of systems (n_systems)
        M = max atoms per system (history dimension)
        H = current history length
        M_ext = M + 3 (extended with cell DOFs per system)

    Args:
        state: Current L-BFGS optimization state
        model: Model that computes energies, forces, and optionally stress
        max_history: Number of (s, y) pairs retained for the two-loop recursion.
        max_step: If set, caps the maximum per-atom displacement per iteration.
        curvature_eps: Threshold for the curvature ⟨y, s⟩ used to accept new
            history pairs.

    Returns:
        Updated LBFGSState after one optimization step

    Notes:
        - If `state.alpha > 0` (ASE mode), the initial inverse Hessian estimate is
          fixed at $H_0 = 1/\alpha$.
        - Otherwise (Standard mode), $H_0$ varies at each step based on the
          curvature of the most recent history pair.

    References:
        - Nocedal & Wright, Numerical Optimization (L-BFGS two-loop recursion).
    """
    from torch_sim.optimizers import CellLBFGSState

    is_cell_state = isinstance(state, CellLBFGSState)
    device, dtype = model.device, model.dtype
    eps = 1e-8 if dtype == torch.float32 else 1e-16
    n_systems = state.n_systems  # S

    # Derive max_atoms from history shape: [S, H, M, 3] or [S, H, M_ext, 3]
    history_dim = state.s_history.shape[2]  # M or M_ext
    if is_cell_state:
        max_atoms_ext = history_dim  # M_ext = M + 3
        max_atoms = max_atoms_ext - 3  # M
    else:
        max_atoms = history_dim  # M
        max_atoms_ext = max_atoms

    # Create valid atom mask for per-system operations: [S, M]
    atom_mask = torch.arange(max_atoms, device=device)[None] < state.max_atoms[:, None]

    # Extended mask including cell DOFs: [S, M_ext]
    if is_cell_state:
        ext_mask = torch.cat(
            [
                atom_mask,
                torch.ones((n_systems, 3), device=device, dtype=torch.bool),
            ],
            dim=1,
        )  # [S, M_ext]
    else:
        ext_mask = atom_mask  # [S, M]

    if is_cell_state:
        # Get current deformation gradient
        # reference_cell.mT: [S, 3, 3], row_vector_cell: [S, 3, 3]
        cur_deform_grad = cell_filters.deform_grad(
            state.reference_cell.mT, state.row_vector_cell
        )  # [S, 3, 3]

        # Transform forces to scaled coordinates
        # forces: [N, 3], cur_deform_grad[system_idx]: [N, 3, 3] -> [N, 3]
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

        # Convert to padded per-system format: [S, M, 3]
        g_atoms = _atoms_to_padded(-forces_scaled, state.system_idx, n_systems, max_atoms)
        # Cell forces: [S, 3, 3] -> [S, 3, 3]
        g_cell = -state.cell_forces  # [S, 3, 3]
        # Extended gradient: [S, M_ext, 3] = [S, M+3, 3]
        g = torch.cat([g_atoms, g_cell], dim=1)  # [S, M_ext, 3]
    else:
        # Convert to padded per-system format: [S, M, 3]
        g = _atoms_to_padded(-state.forces, state.system_idx, n_systems, max_atoms)

    # Two-loop recursion to compute search direction d = -H_k g_k
    # History shape: [S, H, M_ext, 3] or [S, H, M, 3]
    cur_history_len = state.s_history.shape[1]  # H
    q = g.clone()  # [S, M_ext, 3] or [S, M, 3]
    alphas: list[torch.Tensor] = []  # list of [S] tensors

    # First loop (from newest to oldest)
    for i in range(cur_history_len - 1, -1, -1):
        s_i = state.s_history[:, i]  # [S, M_ext, 3] or [S, M, 3]
        y_i = state.y_history[:, i]  # [S, M_ext, 3] or [S, M, 3]

        # ys = y^T s per system: [S]
        ys = _per_system_vdot(y_i, s_i, ext_mask)  # [S]
        rho = torch.where(
            ys.abs() > curvature_eps,
            1.0 / (ys + eps),
            torch.zeros_like(ys),
        )  # [S]
        sq = _per_system_vdot(s_i, q, ext_mask)  # [S]
        alpha = rho * sq  # [S]
        alphas.append(alpha)

        # q <- q - alpha * y_i (broadcast alpha to [S, 1, 1])
        q = q - alpha.view(-1, 1, 1) * y_i  # [S, M_ext, 3]

    # Initial H0 scaling: gamma = (s^T y)/(y^T y) using the last pair
    if cur_history_len > 0:
        s_last = state.s_history[:, -1]  # [S, M_ext, 3]
        y_last = state.y_history[:, -1]  # [S, M_ext, 3]
        sy = _per_system_vdot(s_last, y_last, ext_mask)  # [S]
        yy = _per_system_vdot(y_last, y_last, ext_mask)  # [S]
        gamma_dynamic = torch.where(
            yy.abs() > curvature_eps,
            sy / (yy + eps),
            torch.ones_like(yy),
        )  # [S]
    else:
        gamma_dynamic = torch.ones((n_systems,), device=device, dtype=dtype)  # [S]

    # Fixed gamma (ASE style: 1/alpha)
    # If state.alpha > 0, use that. Else use dynamic.
    is_fixed = state.alpha > 1e-6  # [S] bool
    gamma_fixed = 1.0 / (state.alpha + eps)  # [S]
    gamma = torch.where(is_fixed, gamma_fixed, gamma_dynamic)  # [S]

    # z = gamma * q (broadcast gamma to [S, 1, 1])
    z = gamma.view(-1, 1, 1) * q  # [S, M_ext, 3]

    # Second loop (from oldest to newest)
    for i in range(cur_history_len):
        s_i = state.s_history[:, i]  # [S, M_ext, 3]
        y_i = state.y_history[:, i]  # [S, M_ext, 3]

        ys = _per_system_vdot(y_i, s_i, ext_mask)  # [S]
        rho = torch.where(
            ys.abs() > curvature_eps,
            1.0 / (ys + eps),
            torch.zeros_like(ys),
        )  # [S]
        yz = _per_system_vdot(y_i, z, ext_mask)  # [S]
        beta = rho * yz  # [S]

        alpha_i = alphas[cur_history_len - 1 - i]  # [S]
        # z <- z + s_i * (alpha - beta)
        coeff = (alpha_i - beta).view(-1, 1, 1)  # [S, 1, 1]
        z = z + s_i * coeff  # [S, M_ext, 3]

    d = -z  # search direction: [S, M_ext, 3]

    # Apply step_size scaling per system: [S, 1, 1]
    step = state.step_size.view(-1, 1, 1) * d  # [S, M_ext, 3]

    # Per-system max norm (only over valid atoms/DOFs)
    step_norms = torch.linalg.norm(step, dim=-1)  # [S, M_ext]
    step_norms = step_norms * ext_mask.float()  # Zero out padded
    sys_max = step_norms.max(dim=1).values  # [S]

    # Scaling factors per system: <= 1.0
    scale = torch.where(
        sys_max > max_step,
        max_step / (sys_max + eps),
        torch.ones_like(sys_max),
    )  # [S]
    step = scale.view(-1, 1, 1) * step  # [S, M_ext, 3]

    # Split step into position and cell components
    if is_cell_state:
        step_padded = step[:, :max_atoms]  # [S, M, 3]
        step_cell = step[:, max_atoms:]  # [S, 3, 3]
        # Convert padded step to atom-level
        step_positions = _padded_to_atoms(step_padded, state.system_idx)
    else:
        step_padded = step  # [S, M, 3]
        step_positions = _padded_to_atoms(step_padded, state.system_idx)

    # Save previous state for history update
    # For cell state: store fractional positions and scaled forces (ASE convention)
    if is_cell_state:
        state.prev_positions = frac_positions.clone()  # [N, 3] (fractional)
        state.prev_forces = forces_scaled.clone()  # [N, 3] (scaled)
        state.prev_cell_positions = state.cell_positions.clone()  # [S, 3, 3]
        state.prev_cell_forces = state.cell_forces.clone()  # [S, 3, 3]

        # Apply cell step
        dr_cell = step_cell  # [S, 3, 3]
        cell_positions_new = state.cell_positions + dr_cell  # [S, 3, 3]
        state.cell_positions = cell_positions_new  # [S, 3, 3]

        # Determine if Frechet filter
        init_fn, _step_fn = state.cell_filter
        is_frechet = init_fn is frechet_cell_filter_init

        if is_frechet:
            # Frechet: deform_grad = exp(cell_positions / cell_factor)
            cell_factor_reshaped = state.cell_factor.view(n_systems, 1, 1)
            deform_grad_log_new = cell_positions_new / cell_factor_reshaped  # [S, 3, 3]
            deform_grad_new = torch.matrix_exp(deform_grad_log_new)  # [S, 3, 3]
        else:
            # UnitCell: deform_grad = cell_positions / cell_factor
            cell_factor_expanded = state.cell_factor.expand(n_systems, 3, 1)
            deform_grad_new = cell_positions_new / cell_factor_expanded  # [S, 3, 3]

        # Update cell: new_cell = reference_cell @ deform_grad^T
        # Use set_constrained_cell to apply cell constraints (e.g. FixSymmetry)
        new_col_vector_cell = torch.bmm(
            deform_grad_new, state.reference_cell
        )  # [S, 3, 3]
        state.set_constrained_cell(new_col_vector_cell, scale_atoms=True)

        # Apply position step in fractional space, then convert to Cartesian
        new_frac = frac_positions + step_positions  # [N, 3]

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
        state.set_constrained_positions(state.positions + step_positions)  # [N, 3]

    # Evaluate new forces/energy
    model_output = model(state)
    new_forces = model_output["forces"]  # [N, 3]
    new_energy = model_output["energy"]  # [S]
    new_stress = model_output.get("stress")  # [S, 3, 3] or None

    # Update cell forces for next step: [S, 3, 3]
    if is_cell_state:
        cell_filters.compute_cell_forces(model_output, state)

    # Update state
    state.set_constrained_forces(new_forces)  # [N, 3]
    state.energy = new_energy  # [S]
    state.stress = new_stress  # [S, 3, 3] or None

    # Build new (s, y) for history in per-system format [S, M_ext, 3] or [S, M, 3]
    # s = position difference, y = gradient difference
    if is_cell_state:
        # Get new scaled forces and fractional positions for history
        new_deform_grad = cell_filters.deform_grad(
            state.reference_cell.mT, state.row_vector_cell
        )  # [S, 3, 3]
        # new_forces: [N, 3] -> new_forces_scaled: [N, 3]
        new_forces_scaled = torch.bmm(
            new_forces.unsqueeze(1),  # [N, 1, 3]
            new_deform_grad[state.system_idx],  # [N, 3, 3]
        ).squeeze(1)  # [N, 3]
        # positions: [N, 3] -> new_frac_positions: [N, 3]
        new_frac_positions = torch.linalg.solve(
            new_deform_grad[state.system_idx],  # [N, 3, 3]
            state.positions.unsqueeze(-1),  # [N, 3, 1]
        ).squeeze(-1)  # [N, 3]

        # s_new_pos = frac_pos_new - frac_pos_old: [N, 3] -> [S, M, 3]
        s_new_pos_atoms = new_frac_positions - state.prev_positions  # [N, 3]
        s_new_pos = _atoms_to_padded(
            s_new_pos_atoms, state.system_idx, n_systems, max_atoms
        )  # [S, M, 3]
        # s_new_cell = cell_pos_new - cell_pos_old: [S, 3, 3]
        s_new_cell = state.cell_positions - state.prev_cell_positions  # [S, 3, 3]
        # Concatenate to extended format: [S, M_ext, 3]
        s_new = torch.cat([s_new_pos, s_new_cell], dim=1)  # [S, M_ext, 3]

        # y_new = grad_diff for positions and cell (gradient = -forces)
        # y = grad_new - grad_old = -forces_new - (-forces_old) = forces_old - forces_new
        y_new_pos_atoms = -new_forces_scaled - (-state.prev_forces)  # [N, 3]
        y_new_pos = _atoms_to_padded(
            y_new_pos_atoms, state.system_idx, n_systems, max_atoms
        )  # [S, M, 3]
        y_new_cell = -state.cell_forces - (-state.prev_cell_forces)  # [S, 3, 3]
        y_new = torch.cat([y_new_pos, y_new_cell], dim=1)  # [S, M_ext, 3]
    else:
        # s_new = pos_new - pos_old: [N, 3] -> [S, M, 3]
        s_new_atoms = state.positions - state.prev_positions  # [N, 3]
        s_new = _atoms_to_padded(
            s_new_atoms, state.system_idx, n_systems, max_atoms
        )  # [S, M, 3]
        # y_new = grad_diff: [N, 3] -> [S, M, 3]
        y_new_atoms = -new_forces - (-state.prev_forces)  # [N, 3]
        y_new = _atoms_to_padded(
            y_new_atoms, state.system_idx, n_systems, max_atoms
        )  # [S, M, 3]

    # Append history and trim if needed
    # Note: ASE's L-BFGS doesn't have a curvature check for adding to history.
    # Invalid curvatures are handled in the two-loop by checking rho.
    # History tensors: [S, H, M_ext, 3] or [S, H, M, 3]
    cur_history_len = state.s_history.shape[1]  # H
    if cur_history_len == 0:
        # First entry: [S, 1, M_ext, 3] or [S, 1, M, 3]
        s_hist = s_new.unsqueeze(1)  # [S, 1, M_ext, 3]
        y_hist = y_new.unsqueeze(1)  # [S, 1, M_ext, 3]
    else:
        # Append new entry: [S, H, ...] cat [S, 1, ...] -> [S, H+1, ...]
        s_hist = torch.cat([state.s_history, s_new.unsqueeze(1)], dim=1)
        y_hist = torch.cat([state.y_history, y_new.unsqueeze(1)], dim=1)
    # Trim to max_history
    if s_hist.shape[1] > max_history:
        s_hist = s_hist[:, -max_history:]  # [S, max_history, ...]
        y_hist = y_hist[:, -max_history:]

    if is_cell_state:
        # Store fractional/scaled for next iteration
        state.prev_positions = new_frac_positions.clone()  # [N, 3] (fractional)
        state.prev_forces = new_forces_scaled.clone()  # [N, 3] (scaled)
    else:
        state.prev_forces = new_forces.clone()  # [N, 3]
        state.prev_positions = state.positions.clone()  # [N, 3]

    state.s_history = s_hist  # [S, H, M_ext, 3] or [S, H, M, 3]
    state.y_history = y_hist  # [S, H, M_ext, 3] or [S, H, M, 3]
    state.n_iter = state.n_iter + 1  # [S]

    return state
