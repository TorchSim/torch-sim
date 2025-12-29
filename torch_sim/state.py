"""The core state representation.

The main SimState class represents atomistic systems with support for batched
operations and conversion to/from various atomistic formats.
"""

import copy
import importlib
import typing
from collections import defaultdict
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import torch

import torch_sim as ts
from torch_sim.typing import StateLike


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

from torch_sim.constraints import Constraint, merge_constraints, validate_constraints


@dataclass
class SimState:
    """State representation for atomistic systems with batched operations support.

    Contains the fundamental properties needed to describe an atomistic system:
    positions, masses, unit cell, periodic boundary conditions, and atomic numbers.
    Supports batched operations where multiple atomistic systems can be processed
    simultaneously, managed through system indices.

    States support slicing, cloning, splitting, popping, and movement to other
    data structures or devices. Slicing is supported through fancy indexing,
    e.g. `state[[0, 1, 2]]` will return a new state containing only the first three
    systems. The other operations are available through the `pop`, `split`, `clone`,
    and `to` methods.

    Attributes:
        positions (torch.Tensor): Atomic positions with shape (n_atoms, 3)
        masses (torch.Tensor): Atomic masses with shape (n_atoms,)
        cell (torch.Tensor): Unit cell vectors with shape (n_systems, 3, 3).
            Note that we use a column vector convention, i.e. the cell vectors are
            stored as `[[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]` as opposed to
            the row vector convention `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`
            used by ASE.
        pbc (bool | list[bool] | torch.Tensor): indicates periodic boundary
            conditions in each axis. If a boolean is provided, all axes are
            assumed to have the same periodic boundary conditions.
        atomic_numbers (torch.Tensor): Atomic numbers with shape (n_atoms,)
        system_idx (torch.Tensor): Maps each atom index to its system index.
            Has shape (n_atoms,), must be unique consecutive integers starting from 0.
        constraints (list["Constraint"] | None): List of constraints applied to the
            system. Constraints affect degrees of freedom and modify positions.

    Properties:
        wrap_positions (torch.Tensor): Positions wrapped according to periodic boundary
            conditions
        device (torch.device): Device of the positions tensor
        dtype (torch.dtype): Data type of the positions tensor
        n_atoms (int): Total number of atoms across all systems
        n_systems (int): Number of unique systems in the system

    Notes:
        - positions, masses, and atomic_numbers must have shape (n_atoms, 3).
        - cell must be in the conventional matrix form.
        - system indices must be unique consecutive integers starting from 0.

    Examples:
        >>> state = initialize_state(
        ...     [ase_atoms_1, ase_atoms_2, ase_atoms_3], device, dtype
        ... )
        >>> state.n_systems
        3
        >>> new_state = state[[0, 1]]
        >>> new_state.n_systems
        2
        >>> cloned_state = state.clone()
    """

    positions: torch.Tensor
    masses: torch.Tensor
    cell: torch.Tensor
    pbc: torch.Tensor | list[bool] | bool
    atomic_numbers: torch.Tensor
    charge: torch.Tensor | None = field(default=None)
    spin: torch.Tensor | None = field(default=None)
    system_idx: torch.Tensor | None = field(default=None)
    _constraints: list["Constraint"] = field(default_factory=lambda: [])  # noqa: PIE807

    if TYPE_CHECKING:

        @property
        def system_idx(self) -> torch.Tensor:
            """A getter for system_idx that tells type checkers it's always defined."""
            return self.system_idx

        @property
        def pbc(self) -> torch.Tensor:
            """A getter for pbc that tells type checkers it's always defined."""
            return self.pbc

    _atom_attributes: ClassVar[set[str]] = {
        "positions",
        "masses",
        "atomic_numbers",
        "system_idx",
    }
    _system_attributes: ClassVar[set[str]] = {"cell", "charge", "spin"}
    _global_attributes: ClassVar[set[str]] = {"pbc"}

    def __post_init__(self) -> None:  # noqa: C901
        """Initialize the SimState and validate the arguments."""
        # Check that positions, masses and atomic numbers have compatible shapes
        shapes = [
            getattr(self, attr).shape[0]
            for attr in ("positions", "masses", "atomic_numbers")
        ]

        if len(set(shapes)) > 1:
            raise ValueError(
                f"Incompatible shapes: positions {shapes[0]}, "
                f"masses {shapes[1]}, atomic_numbers {shapes[2]}"
            )

        if isinstance(self.pbc, bool):
            self.pbc = [self.pbc] * 3
        if not isinstance(self.pbc, torch.Tensor):
            self.pbc = torch.tensor(self.pbc, dtype=torch.bool, device=self.device)

        initial_system_idx = self.system_idx
        if initial_system_idx is None:
            self.system_idx = torch.zeros(
                self.n_atoms, device=self.device, dtype=torch.int64
            )
        else:  # assert that system indices are unique consecutive integers
            _, counts = torch.unique_consecutive(initial_system_idx, return_counts=True)
            if not torch.all(counts == torch.bincount(initial_system_idx)):
                raise ValueError("System indices must be unique consecutive integers")

        if self.constraints:
            validate_constraints(self.constraints, state=self)

        if self.charge is None:
            self.charge = torch.zeros(
                self.n_systems, device=self.device, dtype=self.dtype
            )
        elif self.charge.shape[0] != self.n_systems:
            raise ValueError(f"Charge must have shape (n_systems={self.n_systems},)")
        if self.spin is None:
            self.spin = torch.zeros(self.n_systems, device=self.device, dtype=self.dtype)
        elif self.spin.shape[0] != self.n_systems:
            raise ValueError(f"Spin must have shape (n_systems={self.n_systems},)")

        if self.cell.ndim != 3 and initial_system_idx is None:
            self.cell = self.cell.unsqueeze(0)

        if self.cell.shape[-2:] != (3, 3):
            raise ValueError("Cell must have shape (n_systems, 3, 3)")

        if self.cell.shape[0] != self.n_systems:
            raise ValueError(
                f"Cell must have shape (n_systems, 3, 3), got {self.cell.shape}"
            )

        # if devices aren't all the same, raise an error, in a clean way
        devices = {
            attr: getattr(self, attr).device
            for attr in (
                "positions",
                "masses",
                "cell",
                "atomic_numbers",
                "pbc",
                "system_idx",
            )
        }
        if len(set(devices.values())) > 1:
            raise ValueError("All tensors must be on the same device")

    @property
    def wrap_positions(self) -> torch.Tensor:
        """Atomic positions wrapped according to periodic boundary conditions if pbc=True,
        otherwise returns unwrapped positions with shape (n_atoms, 3).
        """
        # TODO: implement a wrapping method
        return self.positions

    @property
    def device(self) -> torch.device:
        """The device where the tensor data is located."""
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the positions tensor."""
        return self.positions.dtype

    @property
    def n_atoms(self) -> int:
        """Total number of atoms in the system across all systems."""
        return self.positions.shape[0]

    @property
    def n_atoms_per_system(self) -> torch.Tensor:
        """Number of atoms per system."""
        return self.system_idx.bincount()

    @property
    def n_systems(self) -> int:
        """Number of systems in the system."""
        return torch.unique(self.system_idx).shape[0]

    @property
    def volume(self) -> torch.Tensor:
        """Volume of the system."""
        return torch.det(self.cell)

    @property
    def attributes(self) -> dict[str, torch.Tensor]:
        """Get all public attributes of the state."""
        return {
            attr: getattr(self, attr)
            for attr in self._atom_attributes
            | self._system_attributes
            | self._global_attributes
            | {"_constraints"}
        }

    @property
    def column_vector_cell(self) -> torch.Tensor:
        """Unit cell following the column vector convention."""
        return self.cell

    @column_vector_cell.setter
    def column_vector_cell(self, value: torch.Tensor) -> None:
        """Set the unit cell from value following the column vector convention.

        Args:
            value: The unit cell as a column vector
        """
        self.cell = value

    @property
    def row_vector_cell(self) -> torch.Tensor:
        """Unit cell following the row vector convention."""
        return self.cell.mT

    @row_vector_cell.setter
    def row_vector_cell(self, value: torch.Tensor) -> None:
        """Set the unit cell from value following the row vector convention.

        Args:
            value: The unit cell as a row vector
        """
        self.cell = value.mT

    def constrain_positions(self) -> None:
        """Set the positions and apply constraints if they exist.

        Args:
            new_positions: New positions tensor with shape (n_atoms, 3)
        """
        # Apply constraints if they exist
        for constraint in self.constraints:
            constraint.adjust_positions(self, self.positions)

    @property
    def constraints(self) -> list[Constraint]:
        """Get the constraints for the SimState.

        Returns:
            list["Constraint"]: List of constraints applied to the system.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: list[Constraint] | Constraint) -> None:
        """Set the constraints for the SimState.

        Args:
            constraints (list["Constraint"] | None): List of constraints to apply.
                If None, no constraints are applied.

        Raises:
            ValueError: If constraints are invalid or span multiple systems
        """
        # check it is a list
        if isinstance(constraints, Constraint):
            constraints = [constraints]

        # Validate new constraints before adding
        validate_constraints(constraints, state=self)

        self._constraints = constraints

    def get_number_of_degrees_of_freedom(self) -> torch.Tensor:
        """Calculate degrees of freedom accounting for constraints.

        Returns:
            torch.Tensor: Number of degrees of freedom per system, with shape
                (n_systems,). Each system starts with 3 * n_atoms_per_system degrees
                of freedom, minus any degrees removed by constraints.
        """
        # Start with unconstrained DOF: 3 degrees per atom
        dof_per_system = 3 * self.n_atoms_per_system

        # Subtract DOF removed by constraints
        if self.constraints is not None:
            for constraint in self.constraints:
                removed_dof = constraint.get_removed_dof(self)
                dof_per_system -= removed_dof

        # Ensure non-negative DOF
        if (dof_per_system <= 0).any():
            raise ValueError("Degrees of freedom cannot be zero or negative")
        return dof_per_system

    def clone(self) -> Self:
        """Create a deep copy of the SimState.

        Creates a new SimState object with identical but independent tensors,
        allowing modification without affecting the original.

        Returns:
            SimState: A new SimState object with the same properties as the original
        """
        attrs = {}
        for attr_name, attr_value in self.attributes.items():
            if isinstance(attr_value, torch.Tensor):
                attrs[attr_name] = attr_value.clone()
            else:
                attrs[attr_name] = copy.deepcopy(attr_value)

        return type(self)(**attrs)

    @classmethod
    def from_state(cls, state: "SimState", **additional_attrs: Any) -> Self:
        """Create a new state from an existing state with additional attributes.

        This method copies all attributes from the source state and adds any additional
        attributes needed for the target state class. It's useful for converting between
        different state types (e.g., SimState to MDState).

        Args:
            state: Source state to copy base attributes from
            **additional_attrs: Additional attributes required by the target state class

        Returns:
            New state of the target class with copied and additional attributes

        Example:
            >>> from torch_sim.integrators.md import MDState
            >>> md_state = MDState.from_state(
            ...     sim_state,
            ...     energy=model_output["energy"],
            ...     forces=model_output["forces"],
            ...     momenta=torch.zeros_like(sim_state.positions),
            ... )
        """
        # Copy all attributes from the source state
        attrs = {}
        for attr_name, attr_value in state.attributes.items():
            if isinstance(attr_value, torch.Tensor):
                attrs[attr_name] = attr_value.clone()
            else:
                attrs[attr_name] = copy.deepcopy(attr_value)

        # Add/override with additional attributes
        attrs.update(additional_attrs)

        return cls(**attrs)

    def to_atoms(self) -> list["Atoms"]:
        """Convert the SimState to a list of ASE Atoms objects.

        Returns:
            list[Atoms]: A list of ASE Atoms objects, one per system
        """
        return ts.io.state_to_atoms(self)

    def to_structures(self) -> list["Structure"]:
        """Convert the SimState to a list of pymatgen Structure objects.

        Returns:
            list[Structure]: A list of pymatgen Structure objects, one per system
        """
        return ts.io.state_to_structures(self)

    def to_phonopy(self) -> list["PhonopyAtoms"]:
        """Convert the SimState to a list of PhonopyAtoms objects.

        Returns:
            list[PhonopyAtoms]: A list of PhonopyAtoms objects, one per system
        """
        return ts.io.state_to_phonopy(self)

    def split(self) -> list[Self]:
        """Split the SimState into a list of single-system SimStates.

        Divides the current state into separate states, each containing a single system,
        preserving all properties appropriately for each system.

        Returns:
            list[SimState]: A list of SimState objects, one per system
        """
        return _split_state(self)

    def pop(self, system_indices: int | list[int] | slice | torch.Tensor) -> list[Self]:
        """Pop off states with the specified system indices.

        This method modifies the original state object by removing the specified
        systems and returns the removed systems as separate SimState objects.

        Args:
            system_indices (int | list[int] | slice | torch.Tensor): The system indices
                to pop

        Returns:
            list[SimState]: Popped SimState objects, one per system index

        Notes:
            This method modifies the original SimState in-place.
        """
        system_indices = _normalize_system_indices(
            system_indices, self.n_systems, self.device
        )

        # Get the modified state and popped states
        modified_state, popped_states = _pop_states(self, system_indices)

        # Update all attributes of self with the modified state's attributes
        for attr_name, attr_value in modified_state.attributes.items():
            setattr(self, attr_name, attr_value)

        return popped_states

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Self:
        """Convert the SimState to a new device and/or data type.

        Args:
            device (torch.device, optional): The target device.
                Defaults to current device.
            dtype (torch.dtype, optional): The target data type.
                Defaults to current dtype.

        Returns:
            SimState: A new SimState with tensors on the specified device and dtype
        """
        return _state_to_device(self, device, dtype)

    def __getitem__(self, system_indices: int | list[int] | slice | torch.Tensor) -> Self:
        """Enable standard Python indexing syntax for slicing batches.

        Args:
            system_indices (int | list[int] | slice | torch.Tensor): The system indices
                to include

        Returns:
            SimState: A new SimState containing only the specified systems
        """
        # TODO: need to document that slicing is supported
        # Reuse the existing slice method
        system_indices = _normalize_system_indices(
            system_indices, self.n_systems, self.device
        )

        return _slice_state(self, system_indices)

    def __init_subclass__(cls, **kwargs) -> None:
        """Enforce that all derived states cannot have tensor attributes that can also be
        None. This is because torch.concatenate cannot concat between a tensor and a None.

        Also enforce all of child classes's attributes are specified in _atom_attributes,
        _system_attributes, or _global_attributes.
        """
        cls._assert_no_tensor_attributes_can_be_none()
        cls._assert_all_attributes_have_defined_scope()
        super().__init_subclass__(**kwargs)

    @classmethod
    def _assert_no_tensor_attributes_can_be_none(cls) -> None:
        # We need to use get_type_hints to correctly inspect the types

        # exceptions exist because the type hint doesn't actually reflect the real type
        # (since we change their type in the post_init)
        exceptions = {"system_idx", "charge", "spin"}

        type_hints = typing.get_type_hints(cls)
        for attr_name, attr_type_hint in type_hints.items():
            origin = typing.get_origin(attr_type_hint)
            if attr_name in exceptions:
                continue
            is_union = origin is typing.Union
            if not is_union and origin is not None:
                # For Python 3.10+ `|` syntax, origin is types.UnionType
                # We check by name to be robust against module reloading/patching issues
                is_union = origin.__module__ == "types" and origin.__name__ == "UnionType"
            if is_union:
                args = typing.get_args(attr_type_hint)
                if torch.Tensor in args and type(None) in args:
                    raise TypeError(
                        f"Attribute '{attr_name}' in class '{cls.__name__}' is not "
                        "allowed to be of type 'torch.Tensor | None' because torch.cat "
                        "cannot concatenate between a tensor and a None. Please default "
                        "the tensor with dummy values and track the 'None' case."
                    )

    @classmethod
    def _assert_all_attributes_have_defined_scope(cls) -> None:
        all_defined_attributes = (
            cls._atom_attributes | cls._system_attributes | cls._global_attributes
        )
        # 1) assert that no attribute is defined twice in all_defined_attributes
        duplicates = (
            (cls._atom_attributes & cls._system_attributes)
            | (cls._atom_attributes & cls._global_attributes)
            | (cls._system_attributes & cls._global_attributes)
        )
        if duplicates:
            raise TypeError(
                f"Attributes {duplicates} are declared multiple times in {cls.__name__} "
                "in _atom_attributes, _system_attributes, or _global_attributes"
            )

        # 2) assert that all attributes are defined in all_defined_attributes
        all_annotations = {}
        for parent_cls in cls.mro():
            if hasattr(parent_cls, "__annotations__"):
                all_annotations.update(parent_cls.__annotations__)

        # Get class namespace attributes (methods, properties, class vars with values)
        class_namespace = vars(cls)
        attributes_to_check = set(class_namespace.keys()) | set(all_annotations.keys())

        for attr_name in attributes_to_check:
            is_special_attribute = attr_name.startswith("__")
            is_private_attribute = attr_name.startswith("_") and not is_special_attribute
            is_property = attr_name in class_namespace and isinstance(
                class_namespace.get(attr_name), property
            )
            is_method = hasattr(cls, attr_name) and callable(getattr(cls, attr_name))
            is_class_variable = (
                # Note: _atom_attributes, _system_attributes, and _global_attributes
                # are all class variables
                typing.get_origin(all_annotations.get(attr_name)) is typing.ClassVar
            )

            if (
                is_special_attribute
                or is_private_attribute
                or is_property
                or is_method
                or is_class_variable
            ):
                continue

            if attr_name not in all_defined_attributes:
                raise TypeError(
                    f"Attribute '{attr_name}' is not defined in {cls.__name__} in any "
                    "of _atom_attributes, _system_attributes, or _global_attributes"
                )


@dataclass(kw_only=True)
class DeformGradMixin:
    """Mixin for states that support deformation gradients."""

    reference_cell: torch.Tensor

    _system_attributes: ClassVar[set[str]] = {"reference_cell"}

    if TYPE_CHECKING:
        # define this under a TYPE_CHECKING block to avoid it being included in the
        # dataclass __init__ during runtime
        row_vector_cell: torch.Tensor

    @property
    def reference_row_vector_cell(self) -> torch.Tensor:
        """Get the original unit cell in terms of row vectors."""
        return self.reference_cell.mT

    @reference_row_vector_cell.setter
    def reference_row_vector_cell(self, value: torch.Tensor) -> None:
        """Set the original unit cell in terms of row vectors."""
        self.reference_cell = value.mT

    @staticmethod
    def _deform_grad(
        reference_row_vector_cell: torch.Tensor, row_vector_cell: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the deformation gradient from original cell to current cell.

        Returns:
            The deformation gradient
        """
        return torch.linalg.solve(reference_row_vector_cell, row_vector_cell).transpose(
            -2, -1
        )

    def deform_grad(self) -> torch.Tensor:
        """Calculate the deformation gradient from original cell to current cell.

        Returns:
            The deformation gradient
        """
        return self._deform_grad(self.reference_row_vector_cell, self.row_vector_cell)


def _normalize_system_indices(
    system_indices: int | Sequence[int] | slice | torch.Tensor,
    n_systems: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize system indices to handle negative indices and different input types.

    Converts various system index representations to a consistent tensor format,
    handling negative indices in the Python style (counting from the end).

    Args:
        system_indices (int | list[int] | slice | torch.Tensor): The system indices to
            normalize
        n_systems (int): Total number of systems in the system
        device (torch.device): Device to place the output tensor on

    Returns:
        torch.Tensor: Normalized system indices as a tensor

    Raises:
        TypeError: If system_indices is of an unsupported type
    """
    if isinstance(system_indices, int):
        # Handle negative integer indexing
        if system_indices < 0:
            system_indices = n_systems + system_indices
        return torch.tensor([system_indices], device=device)
    if isinstance(system_indices, list):
        # Handle negative indices in lists
        normalized = [idx if idx >= 0 else n_systems + idx for idx in system_indices]
        return torch.tensor(normalized, device=device)
    if isinstance(system_indices, slice):
        # Let PyTorch handle the slice conversion with negative indices
        return torch.arange(n_systems, device=device)[system_indices]
    if isinstance(system_indices, torch.Tensor):
        # Handle negative indices in tensors
        return torch.where(system_indices < 0, n_systems + system_indices, system_indices)
    raise TypeError(f"Unsupported index type: {type(system_indices)}")


def _state_to_device[T: SimState](
    state: T, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> T:
    """Convert the SimState to a new device and dtype.

    Creates a new SimState with all tensors moved to the specified device and
    with the specified data type.

    Args:
        state (SimState): The state to convert
        device (torch.device, optional): The target device. Defaults to current device.
        dtype (torch.dtype, optional): The target data type. Defaults to current dtype.

    Returns:
        SimState: A new SimState with tensors on the specified device and dtype
    """
    if device is None:
        device = state.device
    if dtype is None:
        dtype = state.dtype

    attrs = state.attributes
    for attr_name, attr_value in attrs.items():
        if isinstance(attr_value, torch.Tensor):
            attrs[attr_name] = attr_value.to(device=device)

    if dtype is not None:
        attrs["positions"] = attrs["positions"].to(dtype=dtype)
        attrs["masses"] = attrs["masses"].to(dtype=dtype)
        attrs["cell"] = attrs["cell"].to(dtype=dtype)
        attrs["atomic_numbers"] = attrs["atomic_numbers"].to(dtype=torch.int)
    return type(state)(**attrs)


def get_attrs_for_scope(
    state: SimState, scope: Literal["per-atom", "per-system", "global"]
) -> Generator[tuple[str, Any], None, None]:
    """Get attributes for a given scope.

    Args:
        state (SimState): The state to get attributes for
        scope (Literal["per-atom", "per-system", "global"]): The scope to get
            attributes for

    Returns:
        Generator[tuple[str, Any], None, None]: A generator of attribute names and values
    """
    match scope:
        case "per-atom":
            attr_names = state._atom_attributes  # noqa: SLF001
        case "per-system":
            attr_names = state._system_attributes  # noqa: SLF001
        case "global":
            attr_names = state._global_attributes  # noqa: SLF001
        case _:
            raise ValueError(f"Unknown scope: {scope!r}")
    for attr_name in attr_names:
        yield attr_name, getattr(state, attr_name)


def _filter_attrs_by_mask(
    state: SimState,
    atom_mask: torch.Tensor,
    system_mask: torch.Tensor,
) -> dict:
    """Filter attributes by atom and system masks.

    Selects subsets of attributes based on boolean masks for atoms and systems.

    Args:
        state (SimState): The state to filter
        atom_mask (torch.Tensor): Boolean mask for atoms to include with shape
            (n_atoms,)
        system_mask (torch.Tensor): Boolean mask for systems to include with shape
            (n_systems,)

    Returns:
        dict: Filtered attributes with appropriate handling for each scope
    """
    # Copy global attributes directly
    filtered_attrs = dict(get_attrs_for_scope(state, "global"))

    # take into account constraints that are AtomConstraint
    filtered_attrs["_constraints"] = [
        constraint.select_constraint(atom_mask, system_mask)
        for constraint in copy.deepcopy(state.constraints)
    ]
    # Remove any None constraints resulting from selection
    filtered_attrs["_constraints"] = [
        constraint
        for constraint in filtered_attrs["_constraints"]
        if constraint is not None
    ]

    # Filter per-atom attributes
    for attr_name, attr_value in get_attrs_for_scope(state, "per-atom"):
        if attr_name == "system_idx":
            # Get the old system indices for the selected atoms
            old_system_indices = attr_value[atom_mask]

            # Get the system indices that are kept
            kept_indices = torch.arange(attr_value.max() + 1, device=attr_value.device)[
                system_mask
            ]

            # Create a mapping from old system indices to new consecutive indices
            system_idx_map = {idx.item(): i for i, idx in enumerate(kept_indices)}

            # Create new system tensor with remapped indices
            new_system_idxs = torch.tensor(
                [system_idx_map[b.item()] for b in old_system_indices],
                device=attr_value.device,
                dtype=attr_value.dtype,
            )
            filtered_attrs[attr_name] = new_system_idxs

        else:
            filtered_attrs[attr_name] = attr_value[atom_mask]

    # Filter per-system attributes
    for attr_name, attr_value in get_attrs_for_scope(state, "per-system"):
        if isinstance(attr_value, torch.Tensor):
            filtered_attrs[attr_name] = attr_value[system_mask]
        else:  # Non-tensor attributes (e.g. cell filter) are copied as-is
            filtered_attrs[attr_name] = attr_value

    return filtered_attrs


def _split_state[T: SimState](state: T) -> list[T]:
    """Split a SimState into a list of states, each containing a single system.

    Divides a multi-system state into individual single-system states, preserving
    appropriate properties for each system.

    Args:
        state (SimState): The SimState to split

    Returns:
        list[SimState]: A list of SimState objects, each containing a single
            system
    """
    system_sizes = state.n_atoms_per_system.tolist()

    split_per_atom = {}
    for attr_name, attr_value in get_attrs_for_scope(state, "per-atom"):
        if attr_name != "system_idx":
            split_per_atom[attr_name] = torch.split(attr_value, system_sizes, dim=0)

    split_per_system = {}
    for attr_name, attr_value in get_attrs_for_scope(state, "per-system"):
        if isinstance(attr_value, torch.Tensor):
            split_per_system[attr_name] = torch.split(attr_value, 1, dim=0)
        else:  # Non-tensor attributes are replicated for each split
            split_per_system[attr_name] = [attr_value] * state.n_systems

    global_attrs = dict(get_attrs_for_scope(state, "global"))

    # Create a state for each system
    states: list[T] = []
    n_systems = len(system_sizes)
    zero_tensor = torch.tensor([0], device=state.device, dtype=torch.int64)
    cumsum_atoms = torch.cat((zero_tensor, torch.cumsum(state.n_atoms_per_system, dim=0)))
    for sys_idx in range(n_systems):
        system_attrs = {
            # Create a system tensor with all zeros for this system
            "system_idx": torch.zeros(
                system_sizes[sys_idx], device=state.device, dtype=torch.int64
            ),
            # Add the split per-atom attributes
            **{
                attr_name: split_per_atom[attr_name][sys_idx]
                for attr_name in split_per_atom
            },
            # Add the split per-system attributes
            **{
                attr_name: split_per_system[attr_name][sys_idx]
                for attr_name in split_per_system
            },
            # Add the global attributes
            **global_attrs,
        }

        atom_idx = torch.arange(cumsum_atoms[sys_idx], cumsum_atoms[sys_idx + 1])
        new_constraints = [
            new_constraint
            for constraint in state.constraints
            if (new_constraint := constraint.select_sub_constraint(atom_idx, sys_idx))
        ]

        system_attrs["_constraints"] = new_constraints
        states.append(type(state)(**system_attrs))  # type: ignore[invalid-argument-type]

    return states


def _pop_states[T: SimState](
    state: T, pop_indices: list[int] | torch.Tensor
) -> tuple[T, list[T]]:
    """Pop off the states with the specified indices.

    Extracts and removes the specified system indices from the state.

    Args:
        state (SimState): The SimState to modify
        pop_indices (list[int] | torch.Tensor): The system indices to extract and remove

    Returns:
        tuple[SimState, list[SimState]]: A tuple containing:
            - The modified original state with specified systems removed
            - A list of the extracted SimStates, one per popped system

    Notes:
        Unlike the pop method, this function does not modify the input state.
    """
    if len(pop_indices) == 0:
        return state, []

    if isinstance(pop_indices, list):
        pop_indices = torch.tensor(pop_indices, device=state.device, dtype=torch.int64)

    # Create masks for the atoms and systems to keep and pop
    system_range = torch.arange(state.n_systems, device=state.device)
    pop_system_mask = torch.isin(system_range, pop_indices)
    keep_system_mask = ~pop_system_mask

    pop_atom_mask = torch.isin(state.system_idx, pop_indices)
    keep_atom_mask = ~pop_atom_mask

    # Filter attributes for keep and pop states
    keep_attrs = _filter_attrs_by_mask(state, keep_atom_mask, keep_system_mask)
    pop_attrs = _filter_attrs_by_mask(state, pop_atom_mask, pop_system_mask)

    # Create the keep state
    keep_state: T = type(state)(**keep_attrs)  # type: ignore[assignment]

    # Create and split the pop state
    pop_state: T = type(state)(**pop_attrs)  # type: ignore[assignment]
    pop_states = _split_state(pop_state)

    return keep_state, pop_states


def _slice_state[T: SimState](state: T, system_indices: list[int] | torch.Tensor) -> T:
    """Slice a substate from the SimState containing only the specified system indices.

    Creates a new SimState containing only the specified systems, preserving
    all relevant properties.

    Args:
        state (SimState): The state to slice
        system_indices (list[int] | torch.Tensor): System indices to include in the
            sliced state

    Returns:
        SimState: A new SimState object containing only the specified systems

    Raises:
        ValueError: If system_indices is empty
    """
    if isinstance(system_indices, list):
        system_indices = torch.tensor(
            system_indices, device=state.device, dtype=torch.int64
        )

    if len(system_indices) == 0:
        raise ValueError("system_indices cannot be empty")

    # Create masks for the atoms and systems to include
    system_range = torch.arange(state.n_systems, device=state.device)
    system_mask = torch.isin(system_range, system_indices)
    atom_mask = torch.isin(state.system_idx, system_indices)

    # Filter attributes
    filtered_attrs = _filter_attrs_by_mask(state, atom_mask, system_mask)

    # Create the sliced state
    return type(state)(**filtered_attrs)  # type: ignore[invalid-return-type]


def concatenate_states[T: SimState](  # noqa: C901
    states: Sequence[T], device: torch.device | None = None
) -> T:
    """Concatenate a list of SimStates into a single SimState.

    Combines multiple states into a single state with multiple systems.
    Global properties are taken from the first state, and per-atom and per-system
    properties are concatenated.

    Args:
        states (Sequence[SimState]): A list of SimState objects to concatenate
        device (torch.device, optional): The device to place the concatenated state on.
            Defaults to the device of the first state.

    Returns:
        SimState: A new SimState containing all input states as separate systems

    Raises:
        ValueError: If states is empty
        TypeError: If not all states are of the same type
    """
    if not states:
        raise ValueError("Cannot concatenate an empty list of states")

    # Get the first state to determine properties
    first_state = states[0]

    # Ensure all states are of the same class
    state_class = type(first_state)
    if not all(isinstance(state, state_class) for state in states):
        raise TypeError("All states must be of the same type")

    # Use the target device or default to the first state's device
    target_device = device or first_state.device

    # Initialize result with global properties from first state
    concatenated = dict(get_attrs_for_scope(first_state, "global"))

    # Pre-allocate lists for tensors to concatenate
    per_atom_tensors = defaultdict(list)
    per_system_tensors = defaultdict(list)
    new_system_indices = []
    system_offset = 0
    num_atoms_per_state = []

    # Process all states in a single pass
    for state in states:
        # Move state to target device if needed
        if state.device != target_device:
            state = state.to(target_device)

        # Collect per-atom properties
        for prop, val in get_attrs_for_scope(state, "per-atom"):
            if prop == "system_idx":
                # skip system_idx, it will be handled below
                continue
            per_atom_tensors[prop].append(val)

        # Collect per-system properties
        for prop, val in get_attrs_for_scope(state, "per-system"):
            per_system_tensors[prop].append(val)

        # Update system indices
        num_systems = state.n_systems
        new_indices = state.system_idx + system_offset
        new_system_indices.append(new_indices)
        num_atoms_per_state.append(state.n_atoms)

        system_offset += num_systems

    # Concatenate collected tensors
    for prop, tensors in per_atom_tensors.items():
        # if tensors:
        concatenated[prop] = torch.cat(tensors, dim=0)

    for prop, tensors in per_system_tensors.items():
        # if tensors:
        if isinstance(tensors[0], torch.Tensor):
            concatenated[prop] = torch.cat(tensors, dim=0)
        else:  # Non-tensor attributes, take first one (they should all be identical)
            concatenated[prop] = tensors[0]

    # Concatenate system indices
    concatenated["system_idx"] = torch.cat(new_system_indices)

    # Merge constraints
    constraint_lists = [state.constraints for state in states]
    constraints = merge_constraints(
        constraint_lists, torch.tensor(num_atoms_per_state, device=target_device)
    )

    # Create a new instance of the same class
    return state_class(**concatenated, _constraints=constraints)


def initialize_state(
    system: StateLike,
    device: torch.device,
    dtype: torch.dtype,
) -> SimState:
    """Initialize state tensors from a atomistic system representation.

    Converts various atomistic system representations (ASE Atoms, pymatgen Structure,
    PhonopyAtoms, or existing SimState) to a SimState object.

    Args:
        system (StateLike): Input system to convert to state tensors
        device (torch.device): Device to create tensors on
        dtype (torch.dtype): Data type for tensor values

    Returns:
        SimState: State representation initialized from input system

    Raises:
        ValueError: If system type is not supported or if list items have inconsistent
        types
    """
    # TODO: create a way to pass velocities from pmg and ase

    if isinstance(system, SimState):
        return system.clone().to(device, dtype)

    if isinstance(system, list | tuple) and all(isinstance(s, SimState) for s in system):
        if not all(state.n_systems == 1 for state in system):
            raise ValueError(
                "When providing a list of states, to the initialize_state function, "
                "all states must have n_systems == 1. To fix this, you can split the "
                "states into individual states with the split_state function."
            )
        return ts.concatenate_states(system)

    converters = [
        ("pymatgen.core", "Structure", ts.io.structures_to_state),
        ("ase", "Atoms", ts.io.atoms_to_state),
        ("phonopy.structure.atoms", "PhonopyAtoms", ts.io.phonopy_to_state),
    ]

    # Try each converter
    for module_path, class_name, converter_func in converters:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            if isinstance(system, cls) or (
                isinstance(system, list | tuple)
                and all(isinstance(s, cls) for s in system)
            ):
                return converter_func(system, device, dtype)
        except ImportError:
            continue

    # remaining code just for informative error
    all_same_type = (
        isinstance(system, list | tuple)
        and all(isinstance(s, type(system[0])) for s in system)
        and system
    )
    if isinstance(system, list | tuple) and not all_same_type:
        raise ValueError(
            f"All items in list must be of the same type, "
            f"found {type(system[0])} and {type(system[1])}"
        )

    system_type = (
        f"list[{type(system[0])}]" if isinstance(system, list | tuple) else type(system)
    )

    raise ValueError(f"Unsupported {system_type=}")
