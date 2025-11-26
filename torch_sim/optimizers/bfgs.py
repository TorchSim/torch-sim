"""BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer implementation.

This module provides a batched BFGS optimizer that maintains the full Hessian
matrix for each system. This is suitable for systems with a small to moderate
number of atoms, where the $O(N^2)$ memory cost is acceptable.

The implementation handles batches of systems with different numbers of atoms
by padding vectors to the maximum number of atoms in the batch. The Hessian
matrices are similarly padded to shape (n_systems, 3*max_atoms, 3*max_atoms).
"""

from typing import TYPE_CHECKING

import torch

from torch_sim.state import SimState
from torch_sim.typing import StateDict


if TYPE_CHECKING:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.optimizers import BFGSState


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
) -> "BFGSState":
    """Create an initial BFGSState.

    Initializes the Hessian as Identity * alpha.

    Args:
        state: Input state
        model: Model
        max_step: Maximum step size (Angstrom)
        alpha: Initial Hessian stiffness (eV/A^2)

    Returns:
        BFGSState
    """
    from torch_sim.optimizers import BFGSState

    tensor_args = {"device": model.device, "dtype": model.dtype}

    if not isinstance(state, SimState):
        state = SimState(**state)

    n_systems = state.n_systems

    counts = state.n_atoms_per_system
    max_atoms = int(counts.max().item()) if len(counts) > 0 else 0
    atom_idx = _get_atom_indices_per_system(state.system_idx, n_systems)

    model_output = model(state)
    energy = model_output["energy"]
    forces = model_output["forces"]
    stress = model_output["stress"]

    # shape: (n_systems, 3*max_atoms, 3*max_atoms)
    dim = 3 * max_atoms
    hessian = torch.eye(dim, **tensor_args).unsqueeze(0).repeat(n_systems, 1, 1) * alpha

    alpha_t = torch.full((n_systems,), alpha, **tensor_args)
    max_step_t = torch.full((n_systems,), max_step, **tensor_args)
    n_iter = torch.zeros((n_systems,), device=model.device, dtype=torch.int32)

    return BFGSState(
        positions=state.positions.clone(),
        masses=state.masses.clone(),
        cell=state.cell.clone(),
        atomic_numbers=state.atomic_numbers.clone(),
        forces=forces,
        energy=energy,
        stress=stress,
        hessian=hessian,
        prev_forces=forces.clone(),
        prev_positions=state.positions.clone(),
        alpha=alpha_t,
        max_step=max_step_t,
        n_iter=n_iter,
        atom_idx_in_system=atom_idx,
        max_atoms=max_atoms,
        # passed to __post_init__
        system_idx=state.system_idx.clone(),
        pbc=state.pbc,
    )


def bfgs_step(
    state: "BFGSState",
    model: "ModelInterface",
) -> "BFGSState":
    """Perform one BFGS optimization step.

    Updates the Hessian estimate and moves atoms.

    Args:
        state: Current optimization state
        model: Calculator model

    Returns:
        Updated state
    """
    eps = 1e-7

    # Pack flat tensors into dense batched tensors
    # shape: (n_systems, max_atoms * 3)
    pos_new = _pad_to_dense(
        state.positions,
        state.system_idx,
        state.atom_idx_in_system,
        state.n_systems,
        state.max_atoms,
    ).reshape(state.n_systems, -1)

    forces_new = _pad_to_dense(
        state.forces,
        state.system_idx,
        state.atom_idx_in_system,
        state.n_systems,
        state.max_atoms,
    ).reshape(state.n_systems, -1)

    pos_old = _pad_to_dense(
        state.prev_positions,
        state.system_idx,
        state.atom_idx_in_system,
        state.n_systems,
        state.max_atoms,
    ).reshape(state.n_systems, -1)

    forces_old = _pad_to_dense(
        state.prev_forces,
        state.system_idx,
        state.atom_idx_in_system,
        state.n_systems,
        state.max_atoms,
    ).reshape(state.n_systems, -1)

    # Calculate displacements and force changes
    # dpos: (n_systems, max_atoms * 3)
    dpos = pos_new - pos_old
    dforces = -(forces_new - forces_old)

    # Identify systems with significant movement
    max_disp = torch.max(torch.abs(dpos), dim=1).values
    update_mask = max_disp >= eps

    # Update Hessian for active systems
    if update_mask.any():
        idx = update_mask
        H = state.hessian[idx]

        # shape: (n_active, D, 1)
        dp = dpos[idx].unsqueeze(2)
        df = dforces[idx].unsqueeze(2)  # noqa: PD901

        # shape: (n_active, 1)
        a = torch.bmm(dp.transpose(1, 2), df).squeeze(2)

        # shape: (n_active, D, 1)
        dg = torch.bmm(H, dp)

        # shape: (n_active, 1)
        b = torch.bmm(dp.transpose(1, 2), dg).squeeze(2)

        # Rank-2 update
        # shape: (n_active, D, D)
        term1 = torch.bmm(df, df.transpose(1, 2)) / (a.unsqueeze(2) + 1e-30)
        term2 = torch.bmm(dg, dg.transpose(1, 2)) / (b.unsqueeze(2) + 1e-30)

        state.hessian[idx] = H - term1 - term2

    # Calculate step direction using eigendecomposition
    # gradient: (n_systems, D, 1)
    # Step p = H^-1 * F
    direction = forces_new.unsqueeze(2)

    # omega: (n_systems, D), V: (n_systems, D, D)
    omega, V = torch.linalg.eigh(state.hessian)

    # shape: (n_systems, 1, D)
    abs_omega = torch.abs(omega).unsqueeze(1)
    abs_omega = torch.where(abs_omega < 1e-30, torch.ones_like(abs_omega), abs_omega)

    # Project direction onto eigenvectors and scale
    # shape: (n_systems, D, 1)
    vt_g = torch.bmm(V.transpose(1, 2), direction)
    scaled = vt_g / abs_omega.transpose(1, 2)

    # Transform back to original basis
    # shape: (n_systems, D)
    step_dense = torch.bmm(V, scaled).squeeze(2)

    # Scale step if it exceeds max_step
    # step_atoms: (n_systems, max_atoms, 3)
    step_atoms = step_dense.view(state.n_systems, state.max_atoms, 3)
    # atom_norms: (n_systems, max_atoms)
    atom_norms = torch.norm(step_atoms, dim=2)

    # max_disp_per_sys: (n_systems,)
    max_disp_per_sys = torch.max(atom_norms, dim=1).values

    scale = torch.ones_like(max_disp_per_sys)
    needs_scale = max_disp_per_sys > state.max_step
    scale[needs_scale] = state.max_step[needs_scale] / (
        max_disp_per_sys[needs_scale] + 1e-30
    )

    # shape: (n_systems, D)
    step_dense = step_dense * scale.unsqueeze(1)

    # Unpack dense step back to flat valid atoms
    flat_step = step_dense.view(state.n_systems, state.max_atoms, 3)[
        state.system_idx, state.atom_idx_in_system
    ]

    new_positions = state.positions + flat_step

    state.prev_positions = state.positions.clone()
    state.prev_forces = state.forces.clone()
    state.positions = new_positions

    model_output = model(state)
    state.forces = model_output["forces"]
    state.energy = model_output["energy"]
    state.stress = model_output["stress"]
    state.n_iter += 1

    return state
