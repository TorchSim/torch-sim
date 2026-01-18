"""L-BFGS (Limited-memory BFGS) optimizer implementation.

This module provides a batched L-BFGS optimizer for atomic structure relaxation.
L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
a limited history of position and gradient differences, making it memory-efficient
for large systems while achieving superlinear convergence near the minimum.
"""

from typing import TYPE_CHECKING

import torch

import torch_sim.math as tsm
from torch_sim.state import SimState
from torch_sim.typing import StateDict


if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import LBFGSState


def lbfgs_init(
    state: SimState | StateDict,
    model: "ModelInterface",
    *,
    step_size: float = 0.1,
    alpha: float | None = None,
) -> "LBFGSState":
    r"""Create an initial LBFGSState from a SimState or state dict.

    Initializes forces/energy, clears the (s, y) memory, and broadcasts the
    fixed step size to all systems.

    Args:
        state: Input state as SimState object or state parameter dict
        model: Model that computes energies, forces, and optionally stress
        step_size: Fixed per-system step length (damping factor).
            If using ASE mode (fixed alpha), set this to 1.0 (or your damping).
            If using dynamic mode (default), 0.1 is a safe starting point.
        alpha: Initial inverse Hessian stiffness guess (ASE parameter).
            If provided (e.g. 70.0), fixes H0 = 1/alpha for all steps (ASE-style).
            If None (default), H0 is updated dynamically (Standard L-BFGS).

    Returns:
        LBFGSState with initialized optimization tensors

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
    from torch_sim.optimizers import LBFGSState

    tensor_args = {"device": model.device, "dtype": model.dtype}

    if not isinstance(state, SimState):
        state = SimState(**state)

    n_systems = state.n_systems

    # Get initial forces and energy from model
    model_output = model(state)
    energy = model_output["energy"]
    forces = model_output["forces"]
    stress = model_output["stress"]

    # Initialize empty history tensors
    # History shape: [max_history, n_atoms, 3] but we start with 0 entries
    s_history = torch.zeros((0, state.n_atoms, 3), **tensor_args)
    y_history = torch.zeros((0, state.n_atoms, 3), **tensor_args)

    # Alpha tensor: 0.0 means dynamic, >0 means fixed
    alpha_val = 0.0 if alpha is None else alpha
    alpha_tensor = torch.full((n_systems,), alpha_val, **tensor_args)

    return LBFGSState(
        # Copy SimState attributes
        positions=state.positions.clone(),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        atomic_numbers=state.atomic_numbers.clone(),
        system_idx=state.system_idx.clone(),
        pbc=state.pbc,
        # Optimization state
        forces=forces,
        energy=energy,
        stress=stress,
        # L-BFGS specific state
        prev_forces=forces.clone(),
        prev_positions=state.positions.clone(),
        s_history=s_history,
        y_history=y_history,
        step_size=torch.full((n_systems,), step_size, **tensor_args),
        alpha=alpha_tensor,
        n_iter=torch.zeros((n_systems,), device=model.device, dtype=torch.int32),
    )


def lbfgs_step(  # noqa: PLR0915
    state: "LBFGSState",
    model: "ModelInterface",
    *,
    max_history: int = 10,
    max_step: float = 0.2,
    curvature_eps: float = 1e-12,
) -> "LBFGSState":
    r"""Advance one L-BFGS iteration using the two-loop recursion.

    Computes the search direction via the two-loop recursion, applies a
    fixed step with optional per-system capping, evaluates new forces and
    energy, and updates the limited-memory history with a curvature check.

    Algorithm (per system s):
        1) Evaluate gradient g_k = ∇E(x_k) = -f(x_k)
        2) Perform L-BFGS two-loop recursion using up to `max_history` pairs
           (s_i, y_i) to compute d_k = -H_k g_k
        3) Fixed step update with optional per-system step capping by `max_step`
        4) Curvature check and history update: accept (s_k, y_k) if ⟨y_k, s_k⟩ > ε

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
    device, dtype = model.device, model.dtype
    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Current gradient
    g = -state.forces

    # Two-loop recursion to compute search direction d = -H_k g_k
    q = g.clone()
    alphas: list[torch.Tensor] = []  # per-history, shape [n_systems]

    # First loop (from newest to oldest)
    for i in range(state.s_history.shape[0] - 1, -1, -1):
        s_i = state.s_history[i]
        y_i = state.y_history[i]

        ys = tsm.batched_vdot(y_i, s_i, state.system_idx)  # y^T s per system
        rho = torch.where(
            ys.abs() > curvature_eps,
            1.0 / (ys + eps),
            torch.zeros_like(ys),
        )
        sq = tsm.batched_vdot(s_i, q, state.system_idx)
        alpha = rho * sq
        alphas.append(alpha)

        # q <- q - alpha * y_i (broadcast per system to atoms)
        alpha_atom = alpha[state.system_idx].unsqueeze(-1)
        q = q - alpha_atom * y_i

    # Initial H0 scaling: gamma = (s^T y)/(y^T y) using the last pair
    # Dynamic gamma (Standard L-BFGS)
    if state.s_history.shape[0] > 0:
        s_last = state.s_history[-1]
        y_last = state.y_history[-1]
        sy = tsm.batched_vdot(s_last, y_last, state.system_idx)
        yy = tsm.batched_vdot(y_last, y_last, state.system_idx)
        gamma_dynamic = torch.where(
            yy.abs() > curvature_eps,
            sy / (yy + eps),
            torch.ones_like(yy),
        )
    else:
        gamma_dynamic = torch.ones((state.n_systems,), device=device, dtype=dtype)

    # Fixed gamma (ASE style: 1/alpha)
    # If state.alpha > 0, use that. Else use dynamic.
    is_fixed = state.alpha > 1e-6
    gamma_fixed = 1.0 / (state.alpha + eps)
    gamma = torch.where(is_fixed, gamma_fixed, gamma_dynamic)

    z = gamma[state.system_idx].unsqueeze(-1) * q

    # Second loop (from oldest to newest)
    for i in range(state.s_history.shape[0]):
        s_i = state.s_history[i]
        y_i = state.y_history[i]

        ys = tsm.batched_vdot(y_i, s_i, state.system_idx)
        rho = torch.where(
            ys.abs() > curvature_eps,
            1.0 / (ys + eps),
            torch.zeros_like(ys),
        )
        yz = tsm.batched_vdot(y_i, z, state.system_idx)
        beta = rho * yz

        alpha = alphas[state.s_history.shape[0] - 1 - i]
        # z <- z + s_i * (alpha - beta)
        coeff = (alpha - beta)[state.system_idx].unsqueeze(-1)
        z = z + s_i * coeff

    d = -z  # search direction

    # Optional per-system max step cap
    # Compute per-atom step with current step_size
    t_atoms = state.step_size[state.system_idx].unsqueeze(-1)
    step = t_atoms * d

    # Per-atom norms
    norms = torch.linalg.norm(step, dim=1)

    # Per-system max norm
    sys_max = torch.zeros(state.n_systems, device=device, dtype=dtype)
    sys_max.scatter_reduce_(0, state.system_idx, norms, reduce="amax", include_self=False)

    # Scaling factors per system: <= 1.0
    scale = torch.where(
        sys_max > max_step,
        max_step / (sys_max + eps),
        torch.ones_like(sys_max),
    )
    scale_atoms = scale[state.system_idx].unsqueeze(-1)
    step = scale_atoms * step

    # Update positions
    new_positions = state.positions + step

    # Evaluate new forces/energy
    state.positions = new_positions
    model_output = model(state)
    new_forces = model_output["forces"]
    new_energy = model_output["energy"]
    new_stress = model_output["stress"]

    # Build new (s, y)
    s_new = state.positions - state.prev_positions
    y_new = -new_forces - (-state.prev_forces)  # g_new - g_prev = -(f_new - f_prev)

    # Curvature check per system; if bad, clear history (conservative)
    sy = tsm.batched_vdot(s_new, y_new, state.system_idx)
    bad_curv = sy <= curvature_eps

    if bad_curv.any():
        # Clear entire history to preserve correctness
        s_hist = torch.zeros((0, state.n_atoms, 3), device=device, dtype=dtype)
        y_hist = torch.zeros((0, state.n_atoms, 3), device=device, dtype=dtype)
    else:
        # Append and trim if needed
        if state.s_history.shape[0] == 0:
            s_hist = s_new.unsqueeze(0)
            y_hist = y_new.unsqueeze(0)
        else:
            s_hist = torch.cat([state.s_history, s_new.unsqueeze(0)], dim=0)
            y_hist = torch.cat([state.y_history, y_new.unsqueeze(0)], dim=0)
        if s_hist.shape[0] > max_history:
            s_hist = s_hist[-max_history:]
            y_hist = y_hist[-max_history:]

    # Update state
    state.forces = new_forces
    state.energy = new_energy
    state.stress = new_stress

    state.prev_forces = new_forces.clone()
    state.prev_positions = state.positions.clone()
    state.s_history = s_hist
    state.y_history = y_hist
    state.n_iter = state.n_iter + 1

    return state
