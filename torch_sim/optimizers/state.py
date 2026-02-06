"""Optimizer state classes."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from torch_sim.state import SimState


@dataclass(kw_only=True)
class OptimState(SimState):
    """Unified state class for optimization algorithms.

    This class extends SimState to store and track the evolution of system state
    during optimization. It maintains the energies, forces, and optional cell
    optimization state needed for structure relaxation.
    """

    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor

    _atom_attributes = SimState._atom_attributes | {"forces"}  # noqa: SLF001
    _system_attributes = SimState._system_attributes | {"energy", "stress"}  # noqa: SLF001

    def set_constrained_forces(self, new_forces: torch.Tensor) -> None:
        """Set new forces in the optimization state."""
        for constraint in self._constraints:
            constraint.adjust_forces(self, new_forces)
        self.forces = new_forces

    def __post_init__(self) -> None:
        """Post-initialization to ensure SimState setup."""
        self.set_constrained_forces(self.forces)


@dataclass(kw_only=True)
class FireState(OptimState):
    """State class for FIRE optimization.

    Extends OptimState with FIRE-specific parameters for velocity-based optimization.
    """

    velocities: torch.Tensor
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor

    _atom_attributes = OptimState._atom_attributes | {"velocities"}  # noqa: SLF001
    _system_attributes = OptimState._system_attributes | {"dt", "alpha", "n_pos"}  # noqa: SLF001


@dataclass(kw_only=True)
class BFGSState(OptimState):
    """State for batched BFGS optimization.

    Stores the state needed to run a batched BFGS optimizer that maintains
    an approximate Hessian matrix.

    Attributes:
        hessian: Hessian matrix [n_systems, dim, dim] where dim = 3*max_atoms
            for position-only or 3*max_atoms + 9 with cell filter.
            May be padded when systems have different sizes.
        prev_forces: Previous-step forces [n_atoms, 3]. For cell filter,
            these are scaled forces (forces @ deform_grad) for ASE compatibility.
        prev_positions: Previous-step positions [n_atoms, 3]. For cell filter,
            these are fractional coordinates for ASE compatibility.
        alpha: Initial Hessian scale (stiffness) [n_systems]
        max_step: Maximum step size per atom [n_systems]
        n_iter: Per-system iteration counter [n_systems] (int32)
        atom_idx_in_system: Index of each atom within its system [n_atoms]
        max_atoms: Atoms per system [n_systems] - used for size-binned eigendecomp
    """

    hessian: torch.Tensor
    prev_forces: torch.Tensor
    prev_positions: torch.Tensor
    alpha: torch.Tensor
    max_step: torch.Tensor
    n_iter: torch.Tensor
    atom_idx_in_system: torch.Tensor
    max_atoms: torch.Tensor  # Changed from int to Tensor for padding support

    _atom_attributes = OptimState._atom_attributes | {  # noqa: SLF001
        "prev_forces",
        "prev_positions",
        "atom_idx_in_system",
    }
    _system_attributes = OptimState._system_attributes | {  # noqa: SLF001
        "hessian",
        "alpha",
        "max_step",
        "n_iter",
        "max_atoms",
    }
    # Attributes that need padding when concatenating different-sized systems
    _padded_system_attributes: ClassVar[set[str]] = {"hessian"}


@dataclass(kw_only=True)
class LBFGSState(OptimState):
    """State for batched L-BFGS minimization (no line search).

    Stores the state needed to run a batched Limited-memory BFGS optimizer that
    uses a fixed step size and the classical two-loop recursion to compute
    approximate inverse-Hessian-vector products. All tensors are batched across
    systems via `system_idx`.

    Attributes:
        prev_forces: Previous-step forces [n_atoms, 3]. For cell filter,
            these are scaled forces (forces @ deform_grad) for ASE compatibility.
        prev_positions: Previous-step positions [n_atoms, 3]. For cell filter,
            these are fractional coordinates for ASE compatibility.
        s_history: Displacement history [n_systems, h, max_atoms, 3] per-system.
            For cell filter: [n_systems, h, max_atoms + 3, 3] to include cell DOFs.
            May be padded when systems have different sizes.
        y_history: Gradient-diff history [n_systems, h, max_atoms, 3] per-system.
            For cell filter: [n_systems, h, max_atoms + 3, 3] to include cell DOFs.
            May be padded when systems have different sizes.
        step_size: Per-system fixed step size [n_systems]
        alpha: Initial inverse Hessian scale (stiffness) [n_systems]
        n_iter: Per-system iteration counter [n_systems] (int32)
        max_atoms: Atoms per system [n_systems] - used for size-binned operations
    """

    prev_forces: torch.Tensor
    prev_positions: torch.Tensor
    s_history: torch.Tensor
    y_history: torch.Tensor
    step_size: torch.Tensor
    alpha: torch.Tensor
    n_iter: torch.Tensor
    max_atoms: torch.Tensor  # [S] atoms per system for padding support

    _atom_attributes = OptimState._atom_attributes | {  # noqa: SLF001
        "prev_forces",
        "prev_positions",
    }
    _system_attributes = OptimState._system_attributes | {  # noqa: SLF001
        "step_size",
        "alpha",
        "n_iter",
        "max_atoms",
        "s_history",
        "y_history",
    }
    # Attributes that need padding when concatenating different-sized systems
    _padded_system_attributes: ClassVar[set[str]] = {"s_history", "y_history"}


# there's no GradientDescentState, it's the same as OptimState
