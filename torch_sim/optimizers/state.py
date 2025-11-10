"""Optimizer state classes."""

from dataclasses import dataclass

import torch

from torch_sim.state import SimState


@dataclass(kw_only=True, init=False)
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

    def set_forces(self, new_forces: torch.Tensor) -> None:
        """Set new forces in the optimization state."""
        for constraint in self.constraints:
            constraint.adjust_forces(self, new_forces)
        self.forces = new_forces

    def __init__(
        self,
        *,
        positions: torch.Tensor,
        forces: torch.Tensor,
        energy: torch.Tensor,
        stress: torch.Tensor | None = None,
        masses: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        atomic_numbers: torch.Tensor,
        system_idx: torch.Tensor,
        _constraints: list | None = None,
    ) -> None:
        """Initialize optimization state."""
        self.positions = positions
        self.masses = masses
        self.cell = cell
        self.pbc = pbc
        self.atomic_numbers = atomic_numbers
        self.system_idx = system_idx
        self._constraints = _constraints
        self.energy = energy
        self.set_forces(forces)
        self.stress = stress
        SimState.__post_init__(self)


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


# there's no GradientDescentState, it's the same as OptimState
