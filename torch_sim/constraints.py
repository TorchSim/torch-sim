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
from typing import TYPE_CHECKING, Self

import torch

from torch_sim.symmetrize import (
    _prep_symmetry,
    refine_symmetry,
    symmetrize_rank1,
    symmetrize_rank2,
)


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

    def adjust_stress(  # noqa: B027
        self, state: SimState, stress: torch.Tensor
    ) -> None:
        """Adjust stress tensor to satisfy the constraint.

        Default is a no-op. Override in subclasses that need stress symmetrization.

        Args:
            state: Current simulation state
            stress: Stress tensor to be adjusted in-place
        """

    def adjust_cell(  # noqa: B027
        self, state: SimState, cell: torch.Tensor
    ) -> None:
        """Adjust cell to satisfy the constraint.

        Default is a no-op. Override in subclasses that need cell symmetrization.

        Args:
            state: Current simulation state
            cell: Cell tensor to be adjusted in-place (column vector convention)
        """

    @abstractmethod
    def select_constraint(
        self, atom_mask: torch.Tensor, system_mask: torch.Tensor
    ) -> None | Self:
        """Update the constraint to account for atom and system masks.

        Args:
            atom_mask: Boolean mask for atoms to keep
            system_mask: Boolean mask for systems to keep
        """

    @abstractmethod
    def select_sub_constraint(self, atom_idx: torch.Tensor, sys_idx: int) -> None | Self:
        """Select a constraint for a given atom and system index.

        Args:
            atom_idx: Atom indices for a single system
            sys_idx: System index for a single system

        Returns:
            Constraint for the given atom and system index
        """

    @classmethod
    def merge(
        cls,
        constraints: list[Self],
        state_indices: list[int],
        atom_offsets: torch.Tensor,
    ) -> Self:
        """Merge multiple constraints of the same type into one.

        This method is called during state concatenation to combine constraints
        from multiple states. Subclasses can override this for custom merge logic.

        Args:
            constraints: List of constraints to merge (all of the same type)
            state_indices: Index of the source state for each constraint
            atom_offsets: Cumulative atom counts for offset calculation

        Returns:
            A single merged constraint

        Raises:
            NotImplementedError: If the constraint type doesn't support merging
        """
        raise NotImplementedError(
            f"Constraint type {cls.__name__} does not implement merge. "
            "Override this method to support state concatenation."
        )


def _mask_constraint_indices(idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cumsum_atom_mask = torch.cumsum(~mask, dim=0)
    new_indices = idx - cumsum_atom_mask[idx]
    mask_indices = torch.where(mask)[0]
    drop_indices = ~torch.isin(idx, mask_indices)
    return new_indices[~drop_indices]


class AtomConstraint(Constraint):
    """Base class for constraints that act on specific atom indices.

    This class provides common functionality for constraints that operate
    on a subset of atoms, identified by their indices.
    """

    def __init__(
        self,
        atom_idx: torch.Tensor | list[int] | None = None,
        atom_mask: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Initialize indexed constraint.

        Args:
            atom_idx: Indices of atoms to constrain. Can be a tensor or list of integers.
            atom_mask: Boolean mask for atoms to constrain.

        Raises:
            ValueError: If both indices and mask are provided, or if indices have
                       wrong shape/type
        """
        if atom_idx is not None and atom_mask is not None:
            raise ValueError("Provide either atom_idx or atom_mask, not both.")
        if atom_mask is not None:
            atom_mask = torch.as_tensor(atom_mask)
            atom_idx = torch.where(atom_mask)[0]

        # Convert to tensor if needed
        atom_idx = torch.as_tensor(atom_idx)

        # Ensure we have the right shape and type
        atom_idx = torch.atleast_1d(atom_idx)
        if atom_idx.ndim != 1:
            raise ValueError(
                "atom_idx has wrong number of dimensions. "
                f"Got {atom_idx.ndim}, expected ndim <= 1"
            )

        if torch.is_floating_point(atom_idx):
            raise ValueError(
                f"Indices must be integers or boolean mask, not dtype={atom_idx.dtype}"
            )

        self.atom_idx = atom_idx.long()

    def get_indices(self) -> torch.Tensor:
        """Get the constrained atom indices.

        Returns:
            Tensor of atom indices affected by this constraint
        """
        return self.atom_idx.clone()

    def select_constraint(
        self,
        atom_mask: torch.Tensor,
        system_mask: torch.Tensor,  # noqa: ARG002
    ) -> None | Self:
        """Update the constraint to account for atom and system masks.

        Args:
            atom_mask: Boolean mask for atoms to keep
            system_mask: Boolean mask for systems to keep
        """
        indices = self.atom_idx.clone()
        indices = _mask_constraint_indices(indices, atom_mask)
        if len(indices) == 0:
            return None
        return type(self)(indices)

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,
        sys_idx: int,  # noqa: ARG002
    ) -> None | Self:
        """Select a constraint for a given atom and system index.

        Args:
            atom_idx: Atom indices for a single system
            sys_idx: System index for a single system
        """
        mask = torch.isin(self.atom_idx, atom_idx)
        masked_indices = self.atom_idx[mask]
        new_atom_idx = masked_indices - atom_idx.min()
        if len(new_atom_idx) == 0:
            return None
        return type(self)(new_atom_idx)

    @classmethod
    def merge(
        cls,
        constraints: list[Self],
        state_indices: list[int],
        atom_offsets: torch.Tensor,
    ) -> Self:
        """Merge multiple AtomConstraints by concatenating indices with offsets.

        Args:
            constraints: List of constraints to merge
            state_indices: Index of the source state for each constraint
            atom_offsets: Cumulative atom counts for offset calculation

        Returns:
            A single merged constraint with adjusted atom indices
        """
        all_indices = []
        for constraint, state_idx in zip(constraints, state_indices, strict=False):
            offset = atom_offsets[state_idx]
            all_indices.append(constraint.atom_idx + offset)
        return cls(torch.cat(all_indices))


class SystemConstraint(Constraint):
    """Base class for constraints that act on specific system indices.

    This class provides common functionality for constraints that operate
    on a subset of systems, identified by their indices.
    """

    def __init__(
        self,
        system_idx: torch.Tensor | list[int] | None = None,
        system_mask: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Initialize indexed constraint.

        Args:
            system_idx: Indices of systems to constrain.
                Can be a tensor or list of integers.
            system_mask: Boolean mask for systems to constrain.

        Raises:
            ValueError: If both indices and mask are provided, or if indices have
                       wrong shape/type
        """
        if system_idx is not None and system_mask is not None:
            raise ValueError("Provide either system_idx or system_mask, not both.")
        if system_mask is not None:
            system_idx = torch.as_tensor(system_idx)
            system_idx = torch.where(system_mask)[0]

        # Convert to tensor if needed
        system_idx = torch.as_tensor(system_idx)

        # Ensure we have the right shape and type
        system_idx = torch.atleast_1d(system_idx)
        if system_idx.ndim != 1:
            raise ValueError(
                "system_idx has wrong number of dimensions. "
                f"Got {system_idx.ndim}, expected ndim <= 1"
            )

        # Check for duplicates
        if len(system_idx) != len(torch.unique(system_idx)):
            raise ValueError("Duplicate system indices found in SystemConstraint.")

        if torch.is_floating_point(system_idx):
            raise ValueError(
                f"Indices must be integers or boolean mask, not dtype={system_idx.dtype}"
            )

        self.system_idx = system_idx.long()

    def select_constraint(
        self,
        atom_mask: torch.Tensor,  # noqa: ARG002
        system_mask: torch.Tensor,
    ) -> None | Self:
        """Update the constraint to account for atom and system masks.

        Args:
            atom_mask: Boolean mask for atoms to keep
            system_mask: Boolean mask for systems to keep
        """
        system_idx = self.system_idx.clone()
        system_idx = _mask_constraint_indices(system_idx, system_mask)
        if len(system_idx) == 0:
            return None
        return type(self)(system_idx)

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,  # noqa: ARG002
        sys_idx: int,
    ) -> None | Self:
        """Select a constraint for a given atom and system index.

        Args:
            atom_idx: Atom indices for a single system
            sys_idx: System index for a single system
        """
        return type(self)(torch.tensor([0])) if sys_idx in self.system_idx else None

    @classmethod
    def merge(
        cls,
        constraints: list[Self],
        state_indices: list[int],
        atom_offsets: torch.Tensor,  # noqa: ARG003
    ) -> Self:
        """Merge multiple SystemConstraints by concatenating indices with offsets.

        Args:
            constraints: List of constraints to merge
            state_indices: Index of the source state for each constraint
            atom_offsets: Cumulative atom counts (unused for SystemConstraint)

        Returns:
            A single merged constraint with adjusted system indices
        """
        all_indices = []
        for constraint, state_idx in zip(constraints, state_indices, strict=False):
            # For SystemConstraint, the offset is the state index itself
            all_indices.append(constraint.system_idx + state_idx)
        return cls(torch.cat(all_indices))


def merge_constraints(
    constraint_lists: list[list[AtomConstraint | SystemConstraint]],
    num_atoms_per_state: torch.Tensor,
) -> list[Constraint]:
    """Merge constraints from multiple systems into a single list of constraints.

    Args:
        constraint_lists: List of lists of constraints
        num_atoms_per_state: Number of atoms per system

    Returns:
        List of merged constraints
    """
    from collections import defaultdict

    # Calculate atom offsets: for state i, offset = sum of atoms in states 0 to i-1
    device, dtype = num_atoms_per_state.device, num_atoms_per_state.dtype
    atom_offsets = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            torch.cumsum(num_atoms_per_state[:-1], dim=0),
        ]
    )

    # Group constraints by type, tracking their source state index
    constraints_by_type: dict[type[Constraint], tuple[list, list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for state_idx, constraint_list in enumerate(constraint_lists):
        for constraint in constraint_list:
            constraints, indices = constraints_by_type[type(constraint)]
            constraints.append(constraint)
            indices.append(state_idx)

    # Merge each group using the constraint's merge method
    result = []
    for constraint_type, (constraints, state_indices) in constraints_by_type.items():
        merged = constraint_type.merge(constraints, state_indices, atom_offsets)
        result.append(merged)

    return result


class FixAtoms(AtomConstraint):
    """Constraint that fixes specified atoms in place.

    This constraint prevents the specified atoms from moving by:
    - Resetting their positions to original values
    - Setting their forces to zero
    - Removing 3 degrees of freedom per fixed atom

    Examples:
        Fix atoms with indices [0, 1, 2]:
        >>> constraint = FixAtoms(atom_idx=[0, 1, 2])

        Fix atoms using a boolean mask:
        >>> mask = torch.tensor([True, True, True, False, False])
        >>> constraint = FixAtoms(mask=mask)
    """

    def __init__(
        self,
        atom_idx: torch.Tensor | list[int] | None = None,
        atom_mask: torch.Tensor | list[int] | None = None,
    ) -> None:
        """Initialize FixAtoms constraint and check for duplicate indices."""
        super().__init__(atom_idx=atom_idx, atom_mask=atom_mask)
        # Check duplicates
        if len(self.atom_idx) != len(torch.unique(self.atom_idx)):
            raise ValueError("Duplicate atom indices found in FixAtoms constraint.")

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        Each fixed atom removes 3 degrees of freedom (x, y, z motion).

        Args:
            state: Simulation state

        Returns:
            Number of degrees of freedom removed (3 * number of fixed atoms)
        """
        fixed_atoms_system_idx = torch.bincount(
            state.system_idx[self.atom_idx], minlength=state.n_systems
        )
        return 3 * fixed_atoms_system_idx

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Reset positions of fixed atoms to their current values.

        Args:
            state: Current simulation state
            new_positions: Proposed positions to be adjusted in-place
        """
        new_positions[self.atom_idx] = state.positions[self.atom_idx]

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
        forces[self.atom_idx] = 0.0

    def __repr__(self) -> str:
        """String representation of the constraint."""
        if len(self.atom_idx) <= 10:
            indices_str = self.atom_idx.tolist()
        else:
            indices_str = f"{self.atom_idx[:5].tolist()}...{self.atom_idx[-5:].tolist()}"
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

    coms: torch.Tensor | None = None

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        Fixing center of mass removes 3 degrees of freedom (x, y, z translation).

        Args:
            state: Simulation state

        Returns:
            Always returns 3 (center of mass translation degrees of freedom)
        """
        affected_systems = torch.zeros(state.n_systems, dtype=torch.long)
        affected_systems[self.system_idx] = 1
        return 3 * affected_systems

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
        if self.coms is None:
            self.coms = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
                0,
                state.system_idx.unsqueeze(-1).expand(-1, 3),
                state.masses.unsqueeze(-1) * state.positions,
            )
            self.coms /= system_mass.unsqueeze(-1)

        new_com = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx.unsqueeze(-1).expand(-1, 3),
            state.masses.unsqueeze(-1) * new_positions,
        )
        new_com /= system_mass.unsqueeze(-1)
        displacement = torch.zeros(state.n_systems, 3, dtype=dtype)
        displacement[self.system_idx] = (
            -new_com[self.system_idx] + self.coms[self.system_idx]
        )
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
        velocity_change = torch.zeros(state.n_systems, 3, dtype=dtype)
        velocity_change[self.system_idx] = velocity_com[self.system_idx]
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
        system_square_mass = torch.zeros(state.n_systems, dtype=dtype).scatter_add_(
            0,
            state.system_idx,
            torch.square(state.masses),
        )
        lmd = torch.zeros((state.n_systems, 3), dtype=dtype).scatter_add_(
            0,
            state.system_idx.unsqueeze(-1).expand(-1, 3),
            forces * state.masses.unsqueeze(-1),
        )
        lmd /= system_square_mass.unsqueeze(-1)
        forces_change = torch.zeros(state.n_systems, 3, dtype=dtype)
        forces_change[self.system_idx] = lmd[self.system_idx]
        forces -= forces_change[state.system_idx] * state.masses.unsqueeze(-1)

    def __repr__(self) -> str:
        """String representation of the constraint."""
        return f"FixCom(system_idx={self.system_idx})"


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


def check_no_index_out_of_bounds(
    indices: torch.Tensor, max_state_indices: int, constraint_name: str
) -> None:
    """Check that constraint indices are within bounds of the state."""
    if (len(indices) > 0) and (indices.max() >= max_state_indices):
        raise ValueError(
            f"Constraint {constraint_name} has indices up to "
            f"{indices.max()}, but state only has {max_state_indices} "
            "atoms"
        )


def validate_constraints(constraints: list[Constraint], state: SimState) -> None:
    """Validate constraints for potential issues and incompatibilities.

    This function checks for:
    1. Overlapping atom indices across multiple constraints
    2. AtomConstraints spanning multiple systems (requires state)
    3. Mixing FixCom with other constraints (warning only)

    Args:
        constraints: List of constraints to validate
        state: SimState to check against

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
        if isinstance(constraint, AtomConstraint):
            indexed_constraints.append(constraint)

            # Validate that atom indices exist in state if provided
            check_no_index_out_of_bounds(
                constraint.atom_idx, state.n_atoms, type(constraint).__name__
            )
        elif isinstance(constraint, SystemConstraint):
            check_no_index_out_of_bounds(
                constraint.system_idx, state.n_systems, type(constraint).__name__
            )

        if isinstance(constraint, FixCom):
            has_com_constraint = True

    # Check for overlapping atom indices
    if len(indexed_constraints) > 1:
        all_indices = torch.cat([c.atom_idx for c in indexed_constraints])
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


class FixSymmetry(SystemConstraint):
    """Constraint to preserve spacegroup symmetry during optimization.

    This constraint symmetrizes forces, positions, and cell/stress
    according to the crystal symmetry operations. Each system in a batch can
    have different symmetry operations.

    Requires the spglib package to be available for automatic symmetry detection.

    The constraint works by:
    - Symmetrizing forces/momenta as rank-1 tensors using all symmetry operations
    - Symmetrizing position displacements similarly for position adjustments
    - Symmetrizing stress/cell deformation as rank-2 tensors

    Attributes:
        rotations: List of rotation matrices for each system,
            shape (n_ops, 3, 3) per system.
        symm_maps: List of symmetry atom mappings for each system,
            shape (n_ops, n_atoms) per system.
        do_adjust_positions: Whether to symmetrize position adjustments.
        do_adjust_cell: Whether to symmetrize cell/stress adjustments.

    Examples:
        Create from SimState:
        >>> constraint = FixSymmetry.from_state(state, symprec=0.01)
    """

    # Type annotations
    rotations: list[torch.Tensor]
    symm_maps: list[torch.Tensor]
    do_adjust_positions: bool
    do_adjust_cell: bool

    def __init__(
        self,
        rotations: list[torch.Tensor],
        symm_maps: list[torch.Tensor],
        system_idx: torch.Tensor | None = None,
        *,
        adjust_positions: bool = True,
        adjust_cell: bool = True,
    ) -> None:
        """Initialize FixSymmetry constraint.

        Args:
            rotations: List of rotation tensors, one per system.
                Each tensor has shape (n_ops, 3, 3).
            symm_maps: List of symmetry mapping tensors, one per system.
                Each tensor has shape (n_ops, n_atoms_in_system).
            system_idx: Indices of systems this constraint applies to.
                If None, defaults to [0, 1, ..., n_systems-1].
            adjust_positions: Whether to symmetrize position adjustments.
            adjust_cell: Whether to symmetrize cell/stress adjustments.

        Raises:
            ValueError: If lists have mismatched lengths or system_idx is wrong length.
        """
        n_systems = len(rotations)

        # Validate list lengths
        if len(symm_maps) != n_systems:
            raise ValueError(
                "rotations and symm_maps must have the same length. "
                f"Got {len(rotations)}, {len(symm_maps)}."
            )

        if system_idx is None:
            # Infer device from rotations tensors
            device = rotations[0].device if rotations else torch.device("cpu")
            system_idx = torch.arange(n_systems, device=device)

        if len(system_idx) != n_systems:
            raise ValueError(
                f"system_idx length ({len(system_idx)}) must match "
                f"number of systems ({n_systems})"
            )

        super().__init__(system_idx=system_idx)

        self.rotations = rotations
        self.symm_maps = symm_maps
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell

    @classmethod
    def from_state(
        cls,
        state: SimState,
        symprec: float = 0.01,
        *,
        adjust_positions: bool = True,
        adjust_cell: bool = True,
        refine_symmetry_state: bool = True,
    ) -> Self:
        """Create FixSymmetry constraint from a SimState.

        Directly uses tensor data from the state to determine symmetry.

        Warning:
            By default, this method **mutates the input state** in-place to refine
            the atomic positions and cell vectors to ideal symmetric values.
            Set ``refine_symmetry_state=False`` to skip this refinement if you
            want to preserve the original state (though this may lead to
            symmetry detection issues if the structure is not already ideal).

        Args:
            state: SimState containing one or more systems.
            symprec: Symmetry precision for spglib.
            adjust_positions: Whether to symmetrize position adjustments.
            adjust_cell: Whether to symmetrize cell/stress adjustments.
            refine_symmetry_state: Whether to refine the state's positions and cell
                to ideal symmetric values. When True (default), the input state
                is modified in-place. When False, the state is not modified but
                the constraint may not work correctly if the structure deviates
                from ideal symmetry.

        Returns:
            FixSymmetry constraint configured for the state's structures.
        """
        try:
            import spglib  # noqa: F401
        except ImportError:
            raise ImportError("spglib is required for FixSymmetry.from_state") from None

        rotations = []
        symm_maps = []

        # Get atom counts per system for slicing
        atoms_per_system = state.n_atoms_per_system
        cumsum = torch.cat(
            [
                torch.zeros(1, device=state.device, dtype=torch.long),
                torch.cumsum(atoms_per_system, dim=0),
            ]
        )

        for sys_idx in range(state.n_systems):
            start = cumsum[sys_idx].item()
            end = cumsum[sys_idx + 1].item()

            # Extract data for this system
            cell = state.row_vector_cell[sys_idx]
            positions = state.positions[start:end]
            atomic_numbers = state.atomic_numbers[start:end]

            if refine_symmetry_state:
                # Refine symmetry of the structure first
                refined_cell, refined_positions = refine_symmetry(
                    cell, positions, atomic_numbers, symprec=symprec
                )

                # Apply refined cell and positions back to state
                state.cell[sys_idx] = refined_cell.mT  # row→column vector convention
                state.positions[start:end] = refined_positions

                # Get symmetry operations using refined structure
                rots, symm_map = _prep_symmetry(
                    refined_cell, refined_positions, atomic_numbers, symprec=symprec
                )
            else:
                # Use structure as-is without refinement
                rots, symm_map = _prep_symmetry(
                    cell, positions, atomic_numbers, symprec=symprec
                )

            rotations.append(rots)
            symm_maps.append(symm_map)

        system_idx = torch.arange(state.n_systems, device=state.device)

        return cls(
            rotations=rotations,
            symm_maps=symm_maps,
            system_idx=system_idx,
            adjust_positions=adjust_positions,
            adjust_cell=adjust_cell,
        )

    @classmethod
    def merge(
        cls,
        constraints: list[Self],
        state_indices: list[int],  # noqa: ARG003
        atom_offsets: torch.Tensor,  # noqa: ARG003
    ) -> Self:
        """Merge multiple FixSymmetry constraints into one.

        Args:
            constraints: List of FixSymmetry constraints to merge.
            state_indices: Index of the source state for each constraint (unused).
            atom_offsets: Cumulative atom counts (unused for FixSymmetry).

        Returns:
            Merged FixSymmetry constraint.

        Raises:
            ValueError: If constraints list is empty or if constraints have
                mismatched adjust_positions or adjust_cell settings.
        """
        if not constraints:
            raise ValueError("Cannot merge empty list of constraints")

        # Validate that all constraints have matching settings
        first_adjust_positions = constraints[0].do_adjust_positions
        first_adjust_cell = constraints[0].do_adjust_cell

        for i, constraint in enumerate(constraints[1:], start=1):
            if constraint.do_adjust_positions != first_adjust_positions:
                raise ValueError(
                    f"Cannot merge FixSymmetry constraints with different "
                    f"adjust_positions settings: constraint 0 has "
                    f"adjust_positions={first_adjust_positions}, but constraint "
                    f"{i} has adjust_positions={constraint.do_adjust_positions}"
                )
            if constraint.do_adjust_cell != first_adjust_cell:
                raise ValueError(
                    f"Cannot merge FixSymmetry constraints with different "
                    f"adjust_cell settings: constraint 0 has "
                    f"adjust_cell={first_adjust_cell}, but constraint "
                    f"{i} has adjust_cell={constraint.do_adjust_cell}"
                )

        rotations = []
        symm_maps = []
        system_indices = []

        # Use cumulative offset (not state_indices) to handle multi-system constraints
        cumulative_offset = 0
        for constraint in constraints:
            for idx in range(len(constraint.rotations)):
                rotations.append(constraint.rotations[idx])
                symm_maps.append(constraint.symm_maps[idx])
                system_indices.append(cumulative_offset + idx)
            cumulative_offset += len(constraint.rotations)

        device = rotations[0].device

        return cls(
            rotations=rotations,
            symm_maps=symm_maps,
            system_idx=torch.tensor(system_indices, device=device),
            adjust_positions=first_adjust_positions,
            adjust_cell=first_adjust_cell,
        )

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Get number of removed degrees of freedom.

        FixSymmetry doesn't explicitly remove DOF in the same way as FixAtoms.
        This matches ASE's FixSymmetry behavior which also raises NotImplementedError.

        Args:
            state: Simulation state

        Raises:
            NotImplementedError: FixSymmetry does not support DOF counting.
        """
        raise NotImplementedError("FixSymmetry does not implement get_removed_dof.")

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Symmetrize position displacements.

        Args:
            state: Current simulation state
            new_positions: Proposed new positions to be adjusted in-place
        """
        if not self.do_adjust_positions:
            return

        # Compute displacement from current positions
        displacement = new_positions - state.positions

        # Symmetrize the displacement
        self._symmetrize_rank1(state, displacement)

        # Apply symmetrized displacement
        new_positions[:] = state.positions + displacement

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Symmetrize forces according to crystal symmetry.

        Args:
            state: Current simulation state
            forces: Forces to be adjusted in-place
        """
        self._symmetrize_rank1(state, forces)

    def adjust_momenta(self, state: SimState, momenta: torch.Tensor) -> None:
        """Symmetrize momenta according to crystal symmetry.

        Args:
            state: Current simulation state
            momenta: Momenta to be adjusted in-place
        """
        self._symmetrize_rank1(state, momenta)

    def adjust_cell(self, state: SimState, new_cell: torch.Tensor) -> None:
        """Symmetrize cell deformation in-place.

        Computes the deformation gradient as ``(cell_inv @ new_cell).T - I``
        and symmetrizes it as a rank-2 tensor.

        Args:
            state: Current simulation state
            new_cell: Proposed new cell tensor of shape (n_systems, 3, 3)
                in column vector convention, modified in-place.

        Raises:
            RuntimeError: If the deformation gradient step is too large (> 0.25),
                which can cause incorrect symmetrization.

        Warns:
            UserWarning: If the deformation gradient step is large (> 0.15),
                symmetrization may be ill-behaved.
        """
        if not self.do_adjust_cell:
            return

        device = state.device
        dtype = state.dtype
        identity = torch.eye(3, device=device, dtype=dtype)

        for sys_idx_local, sys_idx_global in enumerate(self.system_idx):
            # Get current and new cells in row vector convention
            cur_cell = state.row_vector_cell[sys_idx_global]
            new_cell_row = new_cell[sys_idx_global].mT

            # Calculate deformation gradient
            cur_cell_inv = torch.linalg.inv(cur_cell)
            delta_deform_grad = (cur_cell_inv @ new_cell_row).mT - identity

            # Check for large deformation gradient (following ASE)
            max_delta = torch.abs(delta_deform_grad).max().item()
            if max_delta > 0.25:
                raise RuntimeError(
                    f"FixSymmetry adjust_cell does not work properly with large "
                    f"deformation gradient step {max_delta:.4f} > 0.25. "
                    f"Consider using smaller optimization steps."
                )
            if max_delta > 0.15:
                warnings.warn(
                    f"FixSymmetry adjust_cell may be ill-behaved with large "
                    f"deformation gradient step {max_delta:.4f} > 0.15",
                    UserWarning,
                    stacklevel=2,
                )

            # Symmetrize deformation gradient directly
            symmetrized_delta = symmetrize_rank2(
                cur_cell, delta_deform_grad, self.rotations[sys_idx_local].to(dtype=dtype)
            )

            # Reconstruct cell and update in-place
            new_cell_row_sym = cur_cell @ (symmetrized_delta + identity).mT
            new_cell[sys_idx_global] = new_cell_row_sym.mT  # Back to column convention

    def adjust_stress(self, state: SimState, stress: torch.Tensor) -> None:
        """Symmetrize stress tensor in-place.

        Args:
            state: Current simulation state
            stress: Stress tensor of shape (n_systems, 3, 3), modified in-place.
        """
        dtype = stress.dtype

        for sys_idx_local, sys_idx_global in enumerate(self.system_idx):
            # Get current cell and symmetrize stress directly
            cur_cell = state.row_vector_cell[sys_idx_global]
            sys_stress = stress[sys_idx_global]
            symmetrized = symmetrize_rank2(
                cur_cell, sys_stress, self.rotations[sys_idx_local].to(dtype=dtype)
            )
            stress[sys_idx_global] = symmetrized

    def _symmetrize_rank1(self, state: SimState, vectors: torch.Tensor) -> None:
        """Symmetrize rank-1 tensors (forces, momenta, displacements) in-place.

        Uses fractional-coordinate rotations from spglib together with the current
        cell to transform vectors. The cell is fetched at runtime to ensure
        correctness during variable-cell relaxation.

        Args:
            state: Current simulation state (used for cell and atom indexing)
            vectors: Tensor of shape (n_atoms, 3) to be symmetrized in-place
        """
        # Get atom counts per system
        atoms_per_system = state.n_atoms_per_system
        cumsum = torch.cat(
            [
                torch.zeros(1, device=state.device, dtype=torch.long),
                torch.cumsum(atoms_per_system, dim=0),
            ]
        )

        dtype = vectors.dtype
        for sys_idx_local, sys_idx_global in enumerate(self.system_idx):
            start = cumsum[sys_idx_global].item()
            end = cumsum[sys_idx_global + 1].item()

            # Extract vectors for this system
            sys_vectors = vectors[start:end]

            # Get current cell for this system
            cell = state.row_vector_cell[sys_idx_global]

            # Symmetrize directly
            symmetrized = symmetrize_rank1(
                cell,
                sys_vectors,
                self.rotations[sys_idx_local].to(dtype=dtype),
                self.symm_maps[sys_idx_local],
            )

            # Update in place
            vectors[start:end] = symmetrized

    def select_constraint(
        self,
        atom_mask: torch.Tensor,  # noqa: ARG002
        system_mask: torch.Tensor,
    ) -> Self | None:
        """Select constraint for systems matching the mask.

        Args:
            atom_mask: Boolean mask for atoms (not used for SystemConstraint)
            system_mask: Boolean mask for systems to keep

        Returns:
            New FixSymmetry for selected systems, or None if no systems match.
        """
        # Get indices of systems that are in both system_mask and self.system_idx
        keep_global_indices = torch.where(system_mask)[0]
        mask = torch.isin(self.system_idx, keep_global_indices)

        if not mask.any():
            return None

        new_rotations = [self.rotations[i] for i in range(len(mask)) if mask[i]]
        new_symm_maps = [self.symm_maps[i] for i in range(len(mask)) if mask[i]]

        # Remap system indices
        new_system_idx = _mask_constraint_indices(self.system_idx[mask], system_mask)

        return type(self)(
            rotations=new_rotations,
            symm_maps=new_symm_maps,
            system_idx=new_system_idx,
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
        )

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,  # noqa: ARG002
        sys_idx: int,
    ) -> Self | None:
        """Select constraint for a single system.

        Args:
            atom_idx: Atom indices (not used, kept for interface compatibility)
            sys_idx: System index to select

        Returns:
            New FixSymmetry for the selected system, or None if not found.
        """
        if sys_idx not in self.system_idx:
            return None

        local_idx = (self.system_idx == sys_idx).nonzero(as_tuple=True)[0].item()

        return type(self)(
            rotations=[self.rotations[local_idx]],
            symm_maps=[self.symm_maps[local_idx]],
            system_idx=torch.tensor([0], device=self.system_idx.device),
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
        )

    def __repr__(self) -> str:
        """String representation of the constraint."""
        n_systems = len(self.rotations)
        n_ops_list = [r.shape[0] for r in self.rotations]
        if len(n_ops_list) <= 3:
            ops_str = str(n_ops_list)
        else:
            ops_str = f"[{n_ops_list[0]}, ..., {n_ops_list[-1]}]"
        return (
            f"FixSymmetry(n_systems={n_systems}, "
            f"n_ops={ops_str}, "
            f"adjust_positions={self.do_adjust_positions}, "
            f"adjust_cell={self.do_adjust_cell})"
        )
