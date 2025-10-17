"""Constraints for molecular dynamics simulations.

This module implements constraints inspired by ASE's constraint system,
adapted for the torch-sim framework with support for batched operations
and PyTorch tensors.

The constraints affect degrees of freedom counting and modify forces, momenta,
and positions during MD simulations.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch


if TYPE_CHECKING:
    from torch_sim.state import SimState


class FixConstraint(ABC):
    """Base class for all constraints in torch-sim.

    This is the abstract base class that all constraints must inherit from.
    It defines the interface that constraints must implement to work with
    the torch-sim MD system.
    """

    @abstractmethod
    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get the number of degrees of freedom removed by this constraint.

        Args:
            state: The simulation state

        Returns:
            Number of degrees of freedom removed by this constraint
        """

    @abstractmethod
    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Adjust positions to satisfy the constraint.

        This method should modify new_positions in-place to ensure the
        constraint is satisfied.

        Args:
            state: Current simulation state
            new_positions: Proposed new positions to be adjusted
        """

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Adjust momenta to satisfy the constraint.

        This method should modify momenta in-place to ensure the constraint
        is satisfied. By default, it calls adjust_forces with the momenta.

        Args:
            state: Current simulation state
            momenta: Momenta to be adjusted
        """
        # Default implementation: treat momenta like forces
        self.adjust_forces(state, momenta)

    @abstractmethod
    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Adjust forces to satisfy the constraint.

        This method should modify forces in-place to ensure the constraint
        is satisfied.

        Args:
            state: Current simulation state
            forces: Forces to be adjusted
        """

    def copy(self) -> FixConstraint:
        """Create a copy of this constraint.

        Returns:
            A new instance of this constraint with the same parameters
        """
        return type(self)(**self.__dict__)

    def todict(self) -> dict[str, Any]:
        """Convert constraint to dictionary representation.

        Returns:
            Dictionary representation of the constraint
        """
        return {"name": self.__class__.__name__, "kwargs": self.__dict__.copy()}


class IndexedConstraint(FixConstraint):
    """Base class for constraints that act on specific atom indices.

    This class provides common functionality for constraints that operate
    on a subset of atoms, identified by their indices.
    """

    def __init__(self, indices: torch.Tensor | list[int] | None = None) -> None:
        """Initialize indexed constraint.

        Args:
            indices: Indices of atoms to constrain. Can be a tensor or list of integers.

        Raises:
            ValueError: If both indices and mask are provided, or if indices have
                       wrong shape/type
        """
        if indices is None:
            # Empty constraint
            self.index = torch.empty(0, dtype=torch.long)
            return

        # Convert to tensor if needed
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)

        # Ensure we have the right shape and type
        indices = torch.atleast_1d(indices)
        if indices.ndim > 1:
            raise ValueError(
                "indices has wrong number of dimensions. "
                f"Got {indices.ndim}, expected ndim <= 1"
            )

        if indices.dtype == torch.bool:
            # Convert boolean mask to indices
            indices = torch.where(indices)[0]
        elif len(indices) == 0:
            indices = torch.empty(0, dtype=torch.long)
        elif torch.is_floating_point(indices):
            raise ValueError(
                f"Indices must be integers or boolean mask, not dtype={indices.dtype}"
            )

        # Check for duplicates
        if len(torch.unique(indices)) < len(indices):
            raise ValueError(
                "The indices array contains duplicates. "
                "Perhaps you want to specify a mask instead, but "
                "forgot the mask= keyword."
            )

        self.index = indices.long()

    def get_indices(self) -> torch.Tensor:
        """Get the constrained atom indices.

        Returns:
            Tensor of atom indices affected by this constraint
        """
        return self.index.clone()


class FixAtoms(IndexedConstraint):
    """Constraint that fixes specified atoms in place.

    This constraint prevents the specified atoms from moving by:
    - Resetting their positions to original values
    - Setting their forces to zero
    - Removing 3 degrees of freedom per fixed atom

    Examples:
        Fix atoms with indices [0, 1, 2]:
        >>> constraint = FixAtoms(indices=[0, 1, 2])

        Fix atoms using a boolean mask:
        >>> mask = torch.tensor([True, True, True, False, False])
        >>> constraint = FixAtoms(mask=mask)
    """

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        Each fixed atom removes 3 degrees of freedom (x, y, z motion).

        Args:
            state: Simulation state

        Returns:
            Number of degrees of freedom removed (3 * number of fixed atoms)
        """
        fixed_atoms_system_idx = torch.bincount(
            state.system_idx[self.index], minlength=state.n_systems
        )
        return 3 * fixed_atoms_system_idx

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Reset positions of fixed atoms to their current values.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        new_positions[self.index] = state.positions[self.index]

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:  # noqa: ARG002
        """Set forces on fixed atoms to zero.

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        forces[self.index] = 0.0

    def __repr__(self) -> str:
        """String representation of the constraint."""
        if len(self.index) <= 10:
            indices_str = self.index.tolist()
        else:
            indices_str = f"{self.index[:5].tolist()}...{self.index[-5:].tolist()}"
        return f"FixAtoms(indices={indices_str})"

    def todict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the constraint
        """
        return {"name": "FixAtoms", "kwargs": {"indices": self.index.tolist()}}


class FixCom(FixConstraint):
    """Constraint that fixes the center of mass of all atoms per system.

    This constraint prevents the center of mass from moving by:
    - Adjusting positions to maintain center of mass position
    - Removing center of mass velocity from momenta
    - Adjusting forces to remove net force
    - Removing 3 degrees of freedom (center of mass translation)

    The constraint is applied to all atoms in the system.
    """

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        Fixing center of mass removes 3 degrees of freedom (x, y, z translation).

        Args:
            state: Simulation state

        Returns:
            Always returns 3 (center of mass translation degrees of freedom)
        """
        # if self.index.numel() == 0:
        #     return 3 * torch.ones(state.n_systems, dtype=torch.long)
        # removed_dof = torch.zeros(state.n_systems, dtype=torch.long)
        # removed_dof[self.index] = 1
        # return 3 * removed_dof
        return 3 * torch.ones(state.n_systems, dtype=torch.long)

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Adjust positions to maintain center of mass position.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        dtype = state.positions.dtype
        system_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0, state.system_idx, state.masses
        )
        self.coms = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx.unsqueeze(-1).expand(-1, 3),
            state.masses.unsqueeze(-1) * state.positions,
        )
        self.coms /= system_mass.unsqueeze(-1)

        new_com = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx.unsqueeze(-1).expand(-1, 3),
            state.masses.unsqueeze(-1) * state.positions,
        )
        new_com /= system_mass.unsqueeze(-1)
        displacement = -new_com + self.coms
        new_positions += displacement[state.system_idx]

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Remove center of mass velocity from momenta.

        Args:
            state: Current simulation state
            momenta: Momenta to be adjusted in-place
        """
        # Compute center of mass momenta
        dtype = momenta.dtype
        com_momenta = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx.unsqueeze(-1).expand(-1, 3),
            momenta,
        )
        system_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0, state.system_idx, state.masses
        )
        velocity_com = com_momenta / system_mass.unsqueeze(-1)
        momenta -= velocity_com[state.system_idx] * state.masses.unsqueeze(-1)

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Remove net force to prevent center of mass acceleration.

        This implements the constraint from Eq. (3) and (7) in
        https://doi.org/10.1021/jp9722824

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        dtype = state.forces.dtype
        system_square_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0, state.system_idx, torch.square(state.masses)
        )
        lmd = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx.unsqueeze(-1).expand(-1, 3),
            forces * state.masses.unsqueeze(-1),
        )
        lmd /= system_square_mass.unsqueeze(-1)
        forces -= lmd[state.system_idx] * state.masses.unsqueeze(-1)

    def __repr__(self) -> str:
        """String representation of the constraint."""
        return "FixCom()"

    def todict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the constraint
        """
        return {"name": "FixCom", "kwargs": {}}


def count_degrees_of_freedom(
    state: SimState, constraints: list[FixConstraint] | None = None
) -> int:
    """Count the total degrees of freedom in a system with constraints.

    This function calculates the total number of degrees of freedom by starting
    with the unconstrained count (n_atoms * 3) and subtracting the degrees of
    freedom removed by each constraint.

    Args:
        state: Simulation state
        constraints: List of active constraints (optional)

    Returns:
        Total number of degrees of freedom
    """
    # Start with unconstrained DOF
    total_dof = state.n_atoms * 3

    # Subtract DOF removed by constraints
    if constraints is not None:
        for constraint in constraints:
            total_dof -= constraint.get_removed_dof(state)

    return max(0, total_dof)  # Ensure non-negative


# WIP
def warn_if_overlapping_constraints(constraints: list[FixConstraint]) -> None:
    """Issue warnings if constraints might overlap in problematic ways.

    This function checks for potential issues like multiple constraints
    acting on the same atoms, which could lead to unexpected behavior.

    Args:
        constraints: List of constraints to check
    """
    indexed_constraints = []
    has_com_constraint = False

    for constraint in constraints:
        if isinstance(constraint, IndexedConstraint):
            indexed_constraints.append(constraint)
        elif isinstance(constraint, FixCom):
            has_com_constraint = True

    # Check for overlapping atom indices
    if len(indexed_constraints) > 1:
        all_indices = torch.cat([c.index for c in indexed_constraints])
        unique_indices = torch.unique(all_indices)
        if len(unique_indices) < len(all_indices):
            warnings.warn(
                "Multiple constraints are acting on the same atoms. "
                "This may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    # Warn about COM constraint with fixed atoms
    if has_com_constraint and indexed_constraints:
        warnings.warn(
            "Using FixCom together with other constraints may lead to "
            "unexpected behavior. The center of mass constraint is applied "
            "to all atoms, including those that may be constrained by other means.",
            UserWarning,
            stacklevel=2,
        )
