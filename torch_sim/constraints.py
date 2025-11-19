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
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch_sim.state import SimState


class Constraint(ABC):
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


class AtomIndexedConstraint(Constraint):
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
            self.indices = torch.empty(0, dtype=torch.long)
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

        self.indices = indices.long()

    def get_indices(self) -> torch.Tensor:
        """Get the constrained atom indices.

        Returns:
            Tensor of atom indices affected by this constraint
        """
        return self.indices.clone()


class SystemConstraint(Constraint):
    """Base class for constraints that act on specific system indices.

    This class provides common functionality for constraints that operate
    on a subset of systems, identified by their indices.
    """

    def __init__(self, system_idx: torch.Tensor | list[int] | None = None) -> None:
        """Initialize indexed constraint.

        Args:
            system_idx: Indices of systems to constrain. Can be a tensor or
                        list of integers.

        Raises:
            ValueError: If both indices and mask are provided, or if indices have
                        wrong shape/type
        """
        self.initialized = True
        if system_idx is None:
            # Empty constraint
            self.system_idx = torch.empty(0, dtype=torch.long)
            self.initialized = False
            return

        # Convert to tensor if needed
        system_idx = torch.as_tensor(system_idx)

        # Ensure we have the right shape and type
        system_idx = torch.atleast_1d(system_idx)
        if system_idx.ndim > 1:
            raise ValueError(
                "system_idx has wrong number of dimensions. "
                f"Got {system_idx.ndim}, expected ndim <= 1"
            )


def update_constraint(
    constraint: AtomIndexedConstraint | SystemConstraint,
    atom_mask: torch.Tensor,
    system_mask: torch.Tensor,
) -> Constraint:
    """Update a constraint to account for atom and system masks.

    Args:
        constraint: Constraint to update
        atom_mask: Boolean mask for atoms to keep
        system_mask: Boolean mask for systems to keep
    """
    # n_atoms_per_system = system_idx.bincount()

    def update_indices(idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cumsum_atom_mask = torch.cumsum(~mask, dim=0)
        new_indices = idx - cumsum_atom_mask[idx]
        mask_indices = torch.where(mask)[0]
        drop_indices = ~torch.isin(idx, mask_indices)
        return new_indices[~drop_indices]

    if isinstance(constraint, AtomIndexedConstraint):
        constraint.indices = update_indices(constraint.indices, atom_mask)

    elif isinstance(constraint, SystemConstraint):
        constraint.system_idx = update_indices(constraint.system_idx, system_mask)

    else:
        raise NotImplementedError(
            f"Constraint type {type(constraint)} is not implemented"
        )

    return constraint


def select_sub_constraint(
    constraint: AtomIndexedConstraint | SystemConstraint,
    atom_idx: torch.Tensor,
    sys_idx: int,
) -> AtomIndexedConstraint | SystemConstraint | None:
    """Select a constraint for a given atom and system index.

    Args:
        constraint: Constraint to select
        atom_idx: Atom indices for a single system
        sys_idx: System index for a single system
    """
    """Split a constraint into a list of constraints."""

    # TODO: finish this
    # TODO: we can probably eliminate the for loop and make this a split
    #       out constraint function that just bumps out a single constraint
    #       then embed this function in the split_state function
    if isinstance(constraint, AtomIndexedConstraint):
        mask = torch.isin(constraint.indices, atom_idx)
        masked_indices = constraint.indices[mask]
        new_atom_idx = masked_indices - atom_idx.min()
        if len(new_atom_idx) == 0:
            return None
        return type(constraint)(new_atom_idx)

    if isinstance(constraint, SystemConstraint):
        mask = torch.isin(constraint.system_idx, sys_idx)
        masked_system_idx = constraint.system_idx[mask]
        new_system_idx = masked_system_idx - sys_idx
        if len(new_system_idx) == 0:
            return None
        return type(constraint)(new_system_idx)

    raise NotImplementedError(f"Constraint type {type(constraint)} is not implemented")


# def merge_constraint_into_list(
#     constraints: list[AtomIndexedConstraint | SystemConstraint],
#     constraint: AtomIndexedConstraint | SystemConstraint,
# ) -> list[Constraint]:
#     """Merge a constraint into a list of constraints.

#     Args:
#         constraints: List of constraints
#         constraint: Constraint to merge

#     Returns:
#         List of constraints
#     """
#     return constraints


class FixAtoms(AtomIndexedConstraint):
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
            state.system_idx[self.indices], minlength=state.n_systems
        )
        return 3 * fixed_atoms_system_idx

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Reset positions of fixed atoms to their current values.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        new_positions[self.indices] = state.positions[self.indices]

    def adjust_forces(
        self,
        state: SimState,  # noqa: ARG002
        forces: torch.Tensor,
    ) -> None:
        """Set forces on fixed atoms to zero.

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        forces[self.indices] = 0.0

    def __repr__(self) -> str:
        """String representation of the constraint."""
        if len(self.indices) <= 10:
            indices_str = self.indices.tolist()
        else:
            indices_str = f"{self.indices[:5].tolist()}...{self.indices[-5:].tolist()}"
        return f"FixAtoms(indices={indices_str})"


class FixCom(SystemConstraint):
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
        if self.system_idx != slice(None):
            affected_systems = torch.zeros(state.n_systems, dtype=torch.long)
            affected_systems[self.system_idx] = 1
            return 3 * affected_systems
        return 3 * torch.ones(state.n_systems, dtype=torch.long)

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Adjust positions to maintain center of mass position.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        dtype = state.positions.dtype
        n_systems = (
            state.n_systems if self.system_idx == slice(None) else len(self.system_idx)
        )
        index_to_consider = (
            torch.isin(state.system_idx, self.system_idx)
            if self.system_idx != slice(None)
            else torch.ones(state.n_atoms, dtype=torch.bool)
        )
        system_mass = torch.zeros(n_systems, dtype=dtype).scatter_add_(
            0, state.system_idx[index_to_consider], state.masses[index_to_consider]
        )
        if not hasattr(self, "coms"):
            self.coms = torch.zeros((n_systems, 3), dtype=dtype).scatter_add_(
                0,
                state.system_idx[index_to_consider].unsqueeze(-1).expand(-1, 3),
                state.masses[index_to_consider].unsqueeze(-1)
                * state.positions[index_to_consider],
            )
            self.coms /= system_mass.unsqueeze(-1)

        new_com = torch.zeros((n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx[index_to_consider].unsqueeze(-1).expand(-1, 3),
            state.masses[index_to_consider].unsqueeze(-1)
            * new_positions[index_to_consider],
        )
        new_com /= system_mass.unsqueeze(-1)
        displacement = torch.zeros(state.n_systems, 3, dtype=dtype)
        displacement[self.system_idx] = -new_com + self.coms
        new_positions += displacement[state.system_idx]

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Remove center of mass velocity from momenta.

        Args:
            state: Current simulation state
            momenta: Momenta to be adjusted in-place
        """
        # Compute center of mass momenta
        dtype = momenta.dtype
        n_systems = (
            state.n_systems if self.system_idx == slice(None) else len(self.system_idx)
        )
        index_to_consider = (
            torch.isin(state.system_idx, self.system_idx)
            if self.system_idx != slice(None)
            else torch.ones(state.n_atoms, dtype=torch.bool)
        )
        com_momenta = torch.zeros((n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx[index_to_consider].unsqueeze(-1).expand(-1, 3),
            momenta[index_to_consider],
        )
        system_mass = torch.zeros(n_systems, dtype=dtype).scatter_add_(
            0, state.system_idx[index_to_consider], state.masses[index_to_consider]
        )
        velocity_com = com_momenta / system_mass.unsqueeze(-1)
        velocity_change = torch.zeros(state.n_systems, 3, dtype=dtype)
        velocity_change[self.system_idx] = velocity_com
        momenta -= velocity_change[state.system_idx] * state.masses.unsqueeze(-1)

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Remove net force to prevent center of mass acceleration.

        This implements the constraint from Eq. (3) and (7) in
        https://doi.org/10.1021/jp9722824

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        dtype = state.positions.dtype
        n_systems = (
            state.n_systems if self.system_idx == slice(None) else len(self.system_idx)
        )
        index_to_consider = (
            torch.isin(state.system_idx, self.system_idx)
            if self.system_idx != slice(None)
            else torch.ones(state.n_atoms, dtype=torch.bool)
        )
        system_square_mass = torch.zeros(n_systems, dtype=dtype).scatter_add_(
            0,
            state.system_idx[index_to_consider],
            torch.square(state.masses[index_to_consider]),
        )
        lmd = torch.zeros((n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx[index_to_consider].unsqueeze(-1).expand(-1, 3),
            forces[index_to_consider] * state.masses[index_to_consider].unsqueeze(-1),
        )
        lmd /= system_square_mass.unsqueeze(-1)
        forces_change = torch.zeros(state.n_systems, 3, dtype=dtype)
        forces_change[self.system_idx] = lmd
        forces -= forces_change[state.system_idx] * state.masses.unsqueeze(-1)

    def __repr__(self) -> str:
        """String representation of the constraint."""
        return "FixCom()"


def count_degrees_of_freedom(
    state: SimState, constraints: list[Constraint] | None = None
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


def validate_constraints(
    constraints: list[Constraint], state: SimState | None = None
) -> None:
    """Validate constraints for potential issues and incompatibilities.

    This function checks for:
    1. Overlapping atom indices across multiple constraints
    2. AtomIndexedConstraints spanning multiple systems (requires state)
    3. Mixing FixCom with other constraints (warning only)

    Args:
        constraints: List of constraints to validate
        state: Optional SimState for validating atom indices belong to same system

    Raises:
        ValueError: If constraints are invalid or span multiple systems

    Warns:
        UserWarning: If constraints may lead to unexpected behavior
    """
    if not constraints:
        return

    indexed_constraints = []
    has_com_constraint = False

    for constraint in constraints:
        if isinstance(constraint, AtomIndexedConstraint):
            indexed_constraints.append(constraint)

            # Validate that atom indices exist in state if provided
            if (
                (state is not None)
                and (len(constraint.indices) > 0)
                and (constraint.indices.max() >= state.n_atoms)
            ):
                raise ValueError(
                    f"Constraint {type(constraint).__name__} has indices up to "
                    f"{constraint.indices.max()}, but state only has {state.n_atoms} "
                    "atoms"
                )

        elif isinstance(constraint, FixCom):
            has_com_constraint = True

    # Check for overlapping atom indices
    if len(indexed_constraints) > 1:
        all_indices = torch.cat([c.indices for c in indexed_constraints])
        unique_indices = torch.unique(all_indices)
        if len(unique_indices) < len(all_indices):
            warnings.warn(
                "Multiple constraints are acting on the same atoms. "
                "This may lead to unexpected behavior.",
                UserWarning,
                stacklevel=3,
            )

    # Warn about COM constraint with fixed atoms
    if has_com_constraint and indexed_constraints:
        warnings.warn(
            "Using FixCom together with other constraints may lead to "
            "unexpected behavior. The center of mass constraint is applied "
            "to all atoms, including those that may be constrained by other means.",
            UserWarning,
            stacklevel=3,
        )
