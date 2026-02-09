"""Constraints for molecular dynamics simulations.

This module implements constraints inspired by ASE's constraint system,
adapted for the torch-sim framework with support for batched operations
and PyTorch tensors.

The constraints affect degrees of freedom counting and modify forces, momenta,
and positions during MD simulations.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

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

    @abstractmethod
    def reindex(self, atom_offset: int, system_offset: int) -> Self:
        """Return a copy with indices shifted to global coordinates.

        Called during state concatenation to adjust indices before merging.

        Args:
            atom_offset: Offset to add to atom indices
            system_offset: Offset to add to system indices
        """

    @classmethod
    @abstractmethod
    def merge(cls, constraints: list[Self]) -> Self:
        """Merge multiple already-reindexed constraints into one.

        Constraints must have global (absolute) indices — call ``reindex``
        first. Subclasses override this to handle type-specific data.

        Args:
            constraints: Constraints to merge (all same type, already reindexed)
        """


def _cumsum_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Cumulative sum with a leading zero, e.g. [3, 2, 4] -> [0, 3, 5, 9]."""
    return torch.cat(
        [torch.zeros(1, device=tensor.device, dtype=tensor.dtype), tensor.cumsum(dim=0)]
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

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with atom indices shifted by atom_offset."""
        return type(self)(self.atom_idx + atom_offset)

    @classmethod
    def merge(cls, constraints: list[Self]) -> Self:
        """Merge by concatenating already-reindexed atom indices."""
        return cls(torch.cat([c.atom_idx for c in constraints]))


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

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with system indices shifted by system_offset."""
        return type(self)(self.system_idx + system_offset)

    @classmethod
    def merge(cls, constraints: list[Self]) -> Self:
        """Merge by concatenating already-reindexed system indices."""
        return cls(torch.cat([c.system_idx for c in constraints]))


def merge_constraints(
    constraint_lists: list[list[AtomConstraint | SystemConstraint]],
    num_atoms_per_state: torch.Tensor,
    num_systems_per_state: torch.Tensor | None = None,
) -> list[Constraint]:
    """Merge constraints from multiple states into a single list.

    Each constraint is first reindexed to global coordinates (via ``reindex``),
    then constraints of the same type are merged (via ``merge``).

    Args:
        constraint_lists: List of lists of constraints, one list per state
        num_atoms_per_state: Number of atoms per state
        num_systems_per_state: Number of systems per state. Falls back to 1
            per state if not provided.

    Returns:
        List of merged constraints
    """
    from collections import defaultdict

    # Calculate cumulative offsets for atoms and systems
    device, dtype = num_atoms_per_state.device, num_atoms_per_state.dtype
    atom_offsets = _cumsum_with_zero(num_atoms_per_state[:-1])
    if num_systems_per_state is None:
        num_systems_per_state = torch.ones(
            len(constraint_lists), device=device, dtype=dtype
        )
    system_offsets = _cumsum_with_zero(num_systems_per_state[:-1])

    # Reindex each constraint to global coordinates, then group by type
    grouped: dict[type[Constraint], list[Constraint]] = defaultdict(list)
    for state_idx, constraint_list in enumerate(constraint_lists):
        a_off = int(atom_offsets[state_idx].item())
        s_off = int(system_offsets[state_idx].item())
        for constraint in constraint_list:
            grouped[type(constraint)].append(constraint.reindex(a_off, s_off))

    return [ctype.merge(cs) for ctype, cs in grouped.items()]


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
    """Preserve spacegroup symmetry during optimization.

    Symmetrizes forces/momenta as rank-1 tensors and stress/cell deformation
    as rank-2 tensors using the crystal's symmetry operations. Each system in
    a batch can have different symmetry operations.

    Forces and stress are always symmetrized. Position and cell symmetrization
    can be toggled via ``adjust_positions`` and ``adjust_cell``.
    """

    rotations: list[torch.Tensor]
    symm_maps: list[torch.Tensor]
    reference_cells: list[torch.Tensor] | None
    do_adjust_positions: bool
    do_adjust_cell: bool
    max_cumulative_strain: float

    def __init__(
        self,
        rotations: list[torch.Tensor],
        symm_maps: list[torch.Tensor],
        system_idx: torch.Tensor | None = None,
        *,
        adjust_positions: bool = True,
        adjust_cell: bool = True,
        reference_cells: list[torch.Tensor] | None = None,
        max_cumulative_strain: float = 0.5,
    ) -> None:
        """Initialize FixSymmetry constraint.

        Args:
            rotations: Rotation tensors per system, each (n_ops, 3, 3).
            symm_maps: Atom mapping tensors per system, each (n_ops, n_atoms).
            system_idx: System indices (defaults to 0..n_systems-1).
            adjust_positions: Whether to symmetrize position displacements.
            adjust_cell: Whether to symmetrize cell/stress adjustments.
            reference_cells: Initial refined cells (row vectors) per system for
                cumulative strain tracking. If None, cumulative check is skipped.
            max_cumulative_strain: Maximum allowed cumulative strain from the
                reference cell. If exceeded, the cell update is clamped to
                keep the structure within this strain envelope.
        """
        n_systems = len(rotations)
        if len(symm_maps) != n_systems:
            raise ValueError(
                f"rotations and symm_maps length mismatch: "
                f"{n_systems} vs {len(symm_maps)}"
            )
        if system_idx is None:
            device = rotations[0].device if rotations else torch.device("cpu")
            system_idx = torch.arange(n_systems, device=device)
        if len(system_idx) != n_systems:
            raise ValueError(
                f"system_idx length ({len(system_idx)}) != n_systems ({n_systems})"
            )
        if reference_cells is not None and len(reference_cells) != n_systems:
            raise ValueError(
                f"reference_cells length ({len(reference_cells)}) "
                f"!= n_systems ({n_systems})"
            )

        super().__init__(system_idx=system_idx)
        self.rotations = rotations
        self.symm_maps = symm_maps
        self.reference_cells = reference_cells
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell
        self.max_cumulative_strain = max_cumulative_strain

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
        """Create from SimState, optionally refining to ideal symmetry first.

        Warning:
            When ``refine_symmetry_state=True`` (default), the input state is
            **mutated in-place** to have ideal symmetric positions and cell.

        Args:
            state: SimState containing one or more systems.
            symprec: Symmetry precision for moyopy.
            adjust_positions: Whether to symmetrize position displacements.
            adjust_cell: Whether to symmetrize cell/stress adjustments.
            refine_symmetry_state: Whether to refine positions/cell to ideal values.
        """
        try:
            import moyopy  # noqa: F401
        except ImportError:
            raise ImportError(
                "moyopy required for FixSymmetry: pip install moyopy"
            ) from None

        from torch_sim.symmetrize import prep_symmetry, refine_and_prep_symmetry

        rotations, symm_maps, reference_cells = [], [], []
        cumsum = _cumsum_with_zero(state.n_atoms_per_system)

        for sys_idx in range(state.n_systems):
            start, end = cumsum[sys_idx].item(), cumsum[sys_idx + 1].item()
            cell = state.row_vector_cell[sys_idx]
            pos, nums = state.positions[start:end], state.atomic_numbers[start:end]

            if refine_symmetry_state:
                # Single moyopy call: refine + get symmetry ops in one pass
                cell, pos, rots, smap = refine_and_prep_symmetry(
                    cell,
                    pos,
                    nums,
                    symprec=symprec,
                )
                state.cell[sys_idx] = cell.mT  # row→column vector convention
                state.positions[start:end] = pos
            else:
                rots, smap = prep_symmetry(cell, pos, nums, symprec=symprec)

            rotations.append(rots)
            symm_maps.append(smap)
            # Store the refined cell as the reference for cumulative strain tracking
            reference_cells.append(state.row_vector_cell[sys_idx].clone())

        return cls(
            rotations,
            symm_maps,
            system_idx=torch.arange(state.n_systems, device=state.device),
            adjust_positions=adjust_positions,
            adjust_cell=adjust_cell,
            reference_cells=reference_cells,
        )

    # === Symmetrization hooks ===

    def adjust_forces(self, state: SimState, forces: torch.Tensor) -> None:
        """Symmetrize forces according to crystal symmetry."""
        self._symmetrize_rank1(state, forces)

    def adjust_positions(self, state: SimState, new_positions: torch.Tensor) -> None:
        """Symmetrize position displacements (skipped if do_adjust_positions=False)."""
        if not self.do_adjust_positions:
            return
        displacement = new_positions - state.positions
        self._symmetrize_rank1(state, displacement)
        new_positions[:] = state.positions + displacement

    def adjust_stress(self, state: SimState, stress: torch.Tensor) -> None:
        """Symmetrize stress tensor in-place.

        Always runs (like adjust_forces), independent of do_adjust_cell.
        """
        from torch_sim.symmetrize import symmetrize_rank2

        dtype = stress.dtype
        for ci, si in enumerate(self.system_idx):
            rots = self.rotations[ci].to(dtype=dtype)
            stress[si] = symmetrize_rank2(state.row_vector_cell[si], stress[si], rots)

    def adjust_cell(self, state: SimState, new_cell: torch.Tensor) -> None:
        """Symmetrize cell deformation gradient in-place.

        Computes ``F = inv(cell) @ new_cell_row``, symmetrizes ``F - I`` as a
        rank-2 tensor, then reconstructs ``cell @ (sym(F-I) + I)``.

        Also checks cumulative strain from the initial reference cell. If the
        total deformation exceeds ``max_cumulative_strain``, the update is
        clamped to prevent phase transitions that would break the symmetry
        constraint (e.g. hexagonal → tetragonal cell collapse).

        Args:
            state: Current simulation state.
            new_cell: Cell tensor (n_systems, 3, 3) in column vector convention.

        Raises:
            RuntimeError: If deformation gradient contains NaN or Inf.
        """
        if not self.do_adjust_cell:
            return

        from torch_sim.symmetrize import symmetrize_rank2

        identity = torch.eye(3, device=state.device, dtype=state.dtype)
        for ci, si in enumerate(self.system_idx):
            cur_cell = state.row_vector_cell[si]
            new_row = new_cell[si].mT  # column → row convention

            # Per-step deformation: clamp large steps to avoid ill-conditioned
            # symmetrization while still making progress. The cumulative strain
            # guard below is the real safety net against phase transitions.
            deform_delta = torch.linalg.solve(cur_cell, new_row) - identity
            max_delta = torch.abs(deform_delta).max().item()
            if math.isnan(max_delta) or math.isinf(max_delta):
                raise RuntimeError(
                    f"FixSymmetry: deformation gradient is {max_delta}, "
                    f"cell may be singular or ill-conditioned."
                )
            if max_delta > 0.25:
                deform_delta = deform_delta * (0.25 / max_delta)

            # Symmetrize the per-step deformation
            rots = self.rotations[ci].to(dtype=state.dtype)
            sym_delta = symmetrize_rank2(cur_cell, deform_delta, rots)
            proposed_cell = cur_cell @ (sym_delta + identity)

            # Cumulative strain check against reference cell
            if self.reference_cells is not None:
                ref_cell = self.reference_cells[ci].to(
                    device=state.device, dtype=state.dtype
                )
                cumulative_strain = torch.linalg.solve(ref_cell, proposed_cell) - identity
                max_cumulative = torch.abs(cumulative_strain).max().item()
                if max_cumulative > self.max_cumulative_strain:
                    # Clamp: scale the cumulative strain to stay within the envelope
                    scale = self.max_cumulative_strain / max_cumulative
                    clamped_strain = cumulative_strain * scale
                    proposed_cell = ref_cell @ (clamped_strain + identity)

            new_cell[si] = proposed_cell.mT  # back to column convention

    def _symmetrize_rank1(self, state: SimState, vectors: torch.Tensor) -> None:
        """Symmetrize a rank-1 tensor in-place for each constrained system."""
        from torch_sim.symmetrize import symmetrize_rank1

        cumsum = _cumsum_with_zero(state.n_atoms_per_system)
        dtype = vectors.dtype
        for ci, si in enumerate(self.system_idx):
            start, end = cumsum[si].item(), cumsum[si + 1].item()
            vectors[start:end] = symmetrize_rank1(
                state.row_vector_cell[si],
                vectors[start:end],
                self.rotations[ci].to(dtype=dtype),
                self.symm_maps[ci],
            )

    # === Constraint interface ===

    def get_removed_dof(self, state: SimState) -> torch.Tensor:
        """Returns zero - constrains direction, not DOF count."""
        return torch.zeros(state.n_systems, dtype=torch.long, device=state.device)

    def reindex(self, atom_offset: int, system_offset: int) -> Self:  # noqa: ARG002
        """Return copy with system indices shifted by system_offset."""
        return type(self)(
            list(self.rotations),
            list(self.symm_maps),
            self.system_idx + system_offset,
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=list(self.reference_cells) if self.reference_cells else None,
            max_cumulative_strain=self.max_cumulative_strain,
        )

    @classmethod
    def merge(cls, constraints: list[Self]) -> Self:
        """Merge by concatenating rotations, symm_maps, and system indices."""
        if not constraints:
            raise ValueError("Cannot merge empty constraint list")
        if any(
            c.do_adjust_positions != constraints[0].do_adjust_positions
            or c.do_adjust_cell != constraints[0].do_adjust_cell
            or c.max_cumulative_strain != constraints[0].max_cumulative_strain
            for c in constraints[1:]
        ):
            raise ValueError(
                "Cannot merge FixSymmetry constraints with different "
                "adjust_positions/adjust_cell/max_cumulative_strain settings"
            )
        rotations = [r for c in constraints for r in c.rotations]
        symm_maps = [s for c in constraints for s in c.symm_maps]
        system_idx = torch.cat([c.system_idx for c in constraints])
        # Merge reference cells if all constraints have them
        ref_cells = None
        if all(c.reference_cells is not None for c in constraints):
            ref_cells = [rc for c in constraints for rc in c.reference_cells]
        return cls(
            rotations,
            symm_maps,
            system_idx=system_idx,
            adjust_positions=constraints[0].do_adjust_positions,
            adjust_cell=constraints[0].do_adjust_cell,
            reference_cells=ref_cells,
            max_cumulative_strain=constraints[0].max_cumulative_strain,
        )

    def select_constraint(
        self,
        atom_mask: torch.Tensor,  # noqa: ARG002
        system_mask: torch.Tensor,
    ) -> Self | None:
        """Select constraint for systems matching the mask."""
        keep = torch.where(system_mask)[0]
        mask = torch.isin(self.system_idx, keep)
        if not mask.any():
            return None
        local_idx = mask.nonzero(as_tuple=False).flatten().tolist()
        ref_cells = (
            [self.reference_cells[idx] for idx in local_idx]
            if self.reference_cells
            else None
        )
        return type(self)(
            [self.rotations[idx] for idx in local_idx],
            [self.symm_maps[idx] for idx in local_idx],
            _mask_constraint_indices(self.system_idx[mask], system_mask),
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=ref_cells,
            max_cumulative_strain=self.max_cumulative_strain,
        )

    def select_sub_constraint(
        self,
        atom_idx: torch.Tensor,  # noqa: ARG002
        sys_idx: int,
    ) -> Self | None:
        """Select constraint for a single system."""
        if sys_idx not in self.system_idx:
            return None
        local = (self.system_idx == sys_idx).nonzero(as_tuple=True)[0].item()
        ref_cells = [self.reference_cells[local]] if self.reference_cells else None
        return type(self)(
            [self.rotations[local]],
            [self.symm_maps[local]],
            torch.tensor([0], device=self.system_idx.device),
            adjust_positions=self.do_adjust_positions,
            adjust_cell=self.do_adjust_cell,
            reference_cells=ref_cells,
            max_cumulative_strain=self.max_cumulative_strain,
        )

    def __repr__(self) -> str:
        """String representation."""
        n_ops = [r.shape[0] for r in self.rotations]
        ops = str(n_ops) if len(n_ops) <= 3 else f"[{n_ops[0]}, ..., {n_ops[-1]}]"
        return (
            f"FixSymmetry(n_systems={len(self.rotations)}, n_ops={ops}, "
            f"adjust_positions={self.do_adjust_positions}, "
            f"adjust_cell={self.do_adjust_cell})"
        )
