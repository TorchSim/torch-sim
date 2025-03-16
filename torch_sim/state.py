"""TorchScript-compatible Dataclass Module.

This module provides a dataclass decorator that extends Python's built-in dataclass
functionality to work seamlessly with TorchScript. It enables the creation of
strongly-typed, immutable data structures that can be used in both Python and
TorchScript contexts.

Example:
    ```python
    from torchsim.dataclass import dataclass
    import torch


    @dataclass
    class ParticleState:
        position: torch.Tensor
        velocity: torch.Tensor
        mass: torch.Tensor
    ```
"""

import copy
import importlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

import torch

from torch_sim.io import (
    atoms_to_state,
    phonopy_to_state,
    state_to_atoms,
    state_to_phonopy,
    state_to_structures,
    structures_to_state,
)


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


from typing import TypeVar, Union


T = TypeVar("T", bound="BaseState")
StateLike = Union[
    "Atoms",
    "Structure",
    "PhonopyAtoms",
    list["Atoms"],
    list["Structure"],
    list["PhonopyAtoms"],
    T,
    list[T],
]


# TODO: change later on
@dataclass
class BaseState:
    """Base state class for molecular systems.

    Contains the fundamental properties needed to describe a molecular system:
    positions, masses, unit cell, periodic boundary conditions, and atomic numbers.

    positions, masses, and atomic numbers must have shape (n_atoms, 3), where
    n_atoms is the total number of atoms summed over all batches.

    Batch indices must have shape (n_atoms,) and must be unique consecutive integers.

    cell must have shape (n_batches, 3, 3) and must be in the conventional matrix form.

    pbc must be a boolean. Currently, all batches must have the same pbc.

    Attributes:
        positions: Tensor containing atomic positions with shape (n_atoms, 3)
        masses: Tensor containing atomic masses with shape (n_atoms,)
        cell: Tensor containing unit cell vectors with shape (n_batches, 3, 3)
        pbc: Boolean indicating whether to use periodic boundary conditions
        atomic_numbers: Tensor containing atomic numbers with shape (n_atoms,)
    """

    positions: torch.Tensor
    masses: torch.Tensor
    cell: torch.Tensor
    pbc: bool  # TODO: do all calculators support mixed pbc?
    atomic_numbers: torch.Tensor
    batch: torch.Tensor | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate and process the state after initialization."""
        # data validation and fill batch
        # should make pbc a tensor here
        # if devices aren't all the same, raise an error, in a clean way
        devices = {
            attr: getattr(self, attr).device
            for attr in ("positions", "masses", "cell", "atomic_numbers")
        }
        if len(set(devices.values())) > 1:
            raise ValueError("All tensors must be on the same device")

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

        if self.batch is None:
            self.batch = torch.zeros(self.n_atoms, device=self.device, dtype=torch.int64)
        else:
            # assert that batch indices are unique consecutive integers
            _, counts = torch.unique_consecutive(self.batch, return_counts=True)
            if not torch.all(counts == torch.bincount(self.batch)):
                raise ValueError("Batch indices must be unique consecutive integers")

    @property
    def wrap_positions(self) -> torch.Tensor:
        """Get positions wrapped into the primary unit cell.

        Returns:
            torch.Tensor: Atomic positions wrapped according to periodic boundary
                conditions if pbc=True, otherwise returns unwrapped positions.
        """
        # TODO: implement a wrapping method
        return self.positions

    @property
    def device(self) -> torch.device:
        """Get the device of the positions tensor.

        Returns:
            torch.device: The device of the positions tensor
        """
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the positions tensor.

        Returns:
            torch.dtype: The dtype of the positions tensor
        """
        return self.positions.dtype

    @property
    def n_atoms(self) -> int:
        """Get the number of atoms in the system.

        Returns:
            int: The number of atoms in the system
        """
        return self.positions.shape[0]

    @property
    def n_batches(self) -> int:
        """Get the number of batches in the system.

        Returns:
            int: The number of batches in the system
        """
        return torch.unique(self.batch).shape[0]

    def clone(self) -> Self:
        """Create a deep copy of the BaseState.

        Returns:
            A new BaseState object with the same properties as the original
        """
        attrs = {}
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                attrs[attr_name] = attr_value.clone()
            else:
                attrs[attr_name] = copy.deepcopy(attr_value)

        return self.__class__(**attrs)

    def to_atoms(self) -> list["Atoms"]:
        """Convert the BaseState to a list of Atoms.

        Returns:
            A list of Atoms
        """
        return state_to_atoms(self)

    def to_structures(self) -> list["Structure"]:
        """Convert the BaseState to a list of Structures.

        Returns:
            A list of Structures
        """
        return state_to_structures(self)

    def to_phonopy(self) -> list["PhonopyAtoms"]:
        """Convert the BaseState to a list of PhonopyAtoms.

        Returns:
            A list of PhonopyAtoms
        """
        return state_to_phonopy(self)

    def split(self) -> list[Self]:
        """Split the BaseState into a list of BaseStates.

        Returns:
            A list of BaseStates
        """
        return split_state(self)

    def pop(self, batch_indices: int | list[int] | slice | torch.Tensor) -> list[Self]:
        """Pop off states with the specified batch indices.

        This method modifies the original state object by removing the specified batches.

        Args:
            batch_indices: The batch indices to pop

        Returns:
            List of popped states
        """
        batch_indices = _normalize_batch_indices(
            batch_indices, self.n_batches, self.device
        )

        # Get the modified state and popped states
        modified_state, popped_states = pop_states(self, batch_indices)

        # Update all attributes of self with the modified state's attributes
        for attr_name, attr_value in vars(modified_state).items():
            setattr(self, attr_name, attr_value)

        return popped_states

    def __getitem__(self, batch_indices: int | list[int] | slice | torch.Tensor) -> Self:
        """Enable standard Python indexing syntax for slicing batches.

        Args:
            batch_indices: The batch indices to include in the sliced state

        Returns:
            A new BaseState containing only the specified batches
        """
        # Reuse the existing slice method
        batch_indices = _normalize_batch_indices(
            batch_indices, self.n_batches, self.device
        )

        return slice_state(self, batch_indices)


def _normalize_batch_indices(
    batch_indices: int | list[int] | slice | torch.Tensor,
    n_batches: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize batch indices to handle negative indices and different input types.

    Args:
        batch_indices: The batch indices to normalize
        n_batches: Total number of batches
        device: Device to place the tensor on

    Returns:
        Normalized batch indices as a tensor
    """
    if isinstance(batch_indices, int):
        # Handle negative integer indexing
        if batch_indices < 0:
            batch_indices = n_batches + batch_indices
        return torch.tensor([batch_indices], device=device)
    if isinstance(batch_indices, list):
        # Handle negative indices in lists
        normalized = [idx if idx >= 0 else n_batches + idx for idx in batch_indices]
        return torch.tensor(normalized, device=device)
    if isinstance(batch_indices, slice):
        # Let PyTorch handle the slice conversion with negative indices
        return torch.arange(n_batches, device=device)[batch_indices]
    if isinstance(batch_indices, torch.Tensor):
        # Handle negative indices in tensors
        return torch.where(batch_indices < 0, n_batches + batch_indices, batch_indices)
    raise TypeError(f"Unsupported index type: {type(batch_indices)}")


def state_to_device(
    state: BaseState, device: torch.device, dtype: torch.dtype | None = None
) -> Self:
    """Convert the BaseState to a new device and dtype.

    Args:
        state: The state to convert
        device: The device to convert to
        dtype: The dtype to convert to

    Returns:
        A new BaseState object with the converted device and dtype
    """
    attrs = vars(state)
    for attr_name, attr_value in attrs.items():
        if isinstance(attr_value, torch.Tensor):
            attrs[attr_name] = attr_value.to(device=device)

    if dtype is not None:
        attrs["positions"] = attrs["positions"].to(dtype=dtype)
        attrs["masses"] = attrs["masses"].to(dtype=dtype)
        attrs["cell"] = attrs["cell"].to(dtype=dtype)
        attrs["atomic_numbers"] = attrs["atomic_numbers"].to(dtype=torch.int)
    return type(state)(**attrs)


def infer_property_scope(
    state: BaseState,
    ambiguous_handling: Literal["error", "globalize", "globalize_warn"] = "error",
) -> dict[Literal["global", "per_atom", "per_batch"], list[str]]:
    """Infer whether a property is global, per-atom, or per-batch.

    Args:
        state: The state to infer the property scope of
        ambiguous_handling: How to handle ambiguous properties

    Returns:
        tuple[tuple[str], tuple[str], tuple[str]]: Tuple of three tuples,
            containing the names of properties that are global, per-atom, and
            per-batch, respectively.
    """
    # TODO: this cannot effectively resolve global properties with
    # length of n_atoms or n_batches, they will be classified incorrectly,
    # no clear fix

    if state.n_atoms == state.n_batches:
        raise ValueError(
            f"n_atoms ({state.n_atoms}) and n_batches ({state.n_batches}) are equal, "
            "which means shapes cannot be inferred unambiguously."
        )

    scope = {
        "global": [],
        "per_atom": [],
        "per_batch": [],
    }

    # Iterate through all attributes
    for attr_name, attr_value in vars(state).items():
        # Handle scalar values (global properties)
        if not isinstance(attr_value, torch.Tensor):
            scope["global"].append(attr_name)
            continue

        # Handle tensor properties based on shape
        shape = attr_value.shape

        # Empty tensor case
        if len(shape) == 0:
            scope["global"].append(attr_name)
        # Vector/matrix with first dimension matching number of atoms
        elif shape[0] == state.n_atoms:
            scope["per_atom"].append(attr_name)
        # Tensor with first dimension matching number of batches
        elif shape[0] == state.n_batches:
            scope["per_batch"].append(attr_name)
        # Any other shape is ambiguous
        elif ambiguous_handling == "error":
            raise ValueError(
                f"Cannot categorize property '{attr_name}' with shape {shape}. "
                f"Expected first dimension to be either {state.n_atoms} (per-atom) or "
                f"{state.n_batches} (per-batch), or a scalar (global)."
            )
        elif ambiguous_handling in ("globalize", "globalize_warn"):
            scope["global"].append(attr_name)

            if ambiguous_handling == "globalize_warn":
                warnings.warn(
                    f"Property '{attr_name}' with shape {shape} is ambiguous, "
                    "treating as global. This may lead to unexpected behavior "
                    "and suggests the State is not being used as intended.",
                    stacklevel=1,
                )

    return scope


def _get_property_attrs(
    state: BaseState, ambiguous_handling: Literal["error", "globalize"] = "error"
) -> dict[str, dict]:
    """Get global, per-atom, and per-batch attributes from a state.

    Args:
        state: The state to extract attributes from
        ambiguous_handling: How to handle ambiguous properties

    Returns:
        Dictionary with 'global', 'per_atom', and 'per_batch' attribute names and values
    """
    scope = infer_property_scope(state, ambiguous_handling=ambiguous_handling)

    attrs = {"global": {}, "per_atom": {}, "per_batch": {}}

    # Process global properties
    for attr_name in scope["global"]:
        attrs["global"][attr_name] = getattr(state, attr_name)

    # Process per-atom properties
    for attr_name in scope["per_atom"]:
        attrs["per_atom"][attr_name] = getattr(state, attr_name)

    # Process per-batch properties
    for attr_name in scope["per_batch"]:
        attrs["per_batch"][attr_name] = getattr(state, attr_name)

    return attrs


def _filter_attrs_by_mask(
    attrs: dict[str, dict],
    atom_mask: torch.Tensor,
    batch_mask: torch.Tensor,
) -> dict:
    """Filter attributes by atom and batch masks.

    Args:
        attrs: Dictionary with 'global', 'per_atom', and 'per_batch' attributes
        atom_mask: Boolean mask for atoms to include
        batch_mask: Boolean mask for batches to include

    Returns:
        Dictionary of filtered attributes
    """
    filtered_attrs = {}

    # Copy global attributes directly
    filtered_attrs.update(attrs["global"])

    # Filter per-atom attributes
    for attr_name, attr_value in attrs["per_atom"].items():
        if attr_name == "batch":
            # Get the old batch indices for the selected atoms
            old_batch = attr_value[atom_mask]

            # Get the batch indices that are kept
            kept_indices = torch.arange(attr_value.max() + 1, device=attr_value.device)[
                batch_mask
            ]

            # Create a mapping from old batch indices to new consecutive indices
            batch_map = {idx.item(): i for i, idx in enumerate(kept_indices)}

            # Create new batch tensor with remapped indices
            new_batch = torch.tensor(
                [batch_map[b.item()] for b in old_batch],
                device=attr_value.device,
                dtype=attr_value.dtype,
            )
            filtered_attrs[attr_name] = new_batch
        else:
            filtered_attrs[attr_name] = attr_value[atom_mask]

    # Filter per-batch attributes
    for attr_name, attr_value in attrs["per_batch"].items():
        filtered_attrs[attr_name] = attr_value[batch_mask]

    return filtered_attrs


def split_state(
    state: BaseState,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> list[BaseState]:
    """Split a state into a list of states, each containing a single batch element."""
    attrs = _get_property_attrs(state, ambiguous_handling)
    batch_sizes = torch.bincount(state.batch).tolist()

    # Split per-atom attributes by batch
    split_per_atom = {}
    for attr_name, attr_value in attrs["per_atom"].items():
        if attr_name == "batch":
            continue
        split_per_atom[attr_name] = torch.split(attr_value, batch_sizes, dim=0)

    # Split per-batch attributes into individual elements
    split_per_batch = {}
    for attr_name, attr_value in attrs["per_batch"].items():
        split_per_batch[attr_name] = torch.split(attr_value, 1, dim=0)

    # Create a state for each batch
    states = []
    for i in range(state.n_batches):
        batch_attrs = {
            # Create a batch tensor with all zeros for this batch
            "batch": torch.zeros(batch_sizes[i], device=state.device, dtype=torch.int64),
            # Add the split per-atom attributes
            **{attr_name: split_per_atom[attr_name][i] for attr_name in split_per_atom},
            # Add the split per-batch attributes
            **{attr_name: split_per_batch[attr_name][i] for attr_name in split_per_batch},
            # Add the global attributes
            **attrs["global"],
        }
        states.append(type(state)(**batch_attrs))

    return states


def pop_states(
    state: BaseState,
    pop_indices: list[int] | torch.Tensor,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> tuple[BaseState, list[BaseState]]:
    """Pop off the states with the specified indices."""
    if len(pop_indices) == 0:
        return state, []

    if isinstance(pop_indices, list):
        pop_indices = torch.tensor(pop_indices, device=state.device, dtype=torch.int64)

    attrs = _get_property_attrs(state, ambiguous_handling)

    # Create masks for the atoms and batches to keep and pop
    batch_range = torch.arange(state.n_batches, device=state.device)
    pop_batch_mask = torch.isin(batch_range, pop_indices)
    keep_batch_mask = ~pop_batch_mask

    pop_atom_mask = torch.isin(state.batch, pop_indices)
    keep_atom_mask = ~pop_atom_mask

    # Filter attributes for keep and pop states
    keep_attrs = _filter_attrs_by_mask(attrs, keep_atom_mask, keep_batch_mask)
    pop_attrs = _filter_attrs_by_mask(attrs, pop_atom_mask, pop_batch_mask)

    # Create the keep state
    keep_state = type(state)(**keep_attrs)

    # Create and split the pop state
    pop_state = type(state)(**pop_attrs)
    pop_states = split_state(pop_state, ambiguous_handling)

    return keep_state, pop_states


def slice_state(
    state: BaseState,
    batch_indices: list[int] | torch.Tensor,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> BaseState:
    """Slice a substate from the BaseState containing only the specified batch indices.

    Args:
        state: The state to slice
        batch_indices: List or tensor of batch indices to include in the sliced state
        ambiguous_handling: How to handle ambiguous properties

    Returns:
        A BaseState object containing only the specified batches
    """
    if isinstance(batch_indices, list):
        batch_indices = torch.tensor(
            batch_indices, device=state.device, dtype=torch.int64
        )

    if len(batch_indices) == 0:
        raise ValueError("batch_indices cannot be empty")

    attrs = _get_property_attrs(state, ambiguous_handling)

    # Create masks for the atoms and batches to include
    batch_range = torch.arange(state.n_batches, device=state.device)
    batch_mask = torch.isin(batch_range, batch_indices)
    atom_mask = torch.isin(state.batch, batch_indices)

    # Filter attributes
    filtered_attrs = _filter_attrs_by_mask(attrs, atom_mask, batch_mask)

    # Create the sliced state
    return type(state)(**filtered_attrs)


def concatenate_states(
    states: list[BaseState], device: torch.device | None = None
) -> BaseState:
    """Concatenate a list of BaseStates into a single BaseState.

    Global properties are taken from the first state, and per-atom and per-batch
    properties are concatenated.

    Args:
        states: A list of BaseState objects to concatenate
        device: The device to concatenate on

    Returns:
        BaseState: A BaseState object initialized from the input states
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

    # Get property scopes from the first state to identify
    # global/per-atom/per-batch properties
    first_scope = infer_property_scope(first_state)
    global_props = set(first_scope["global"])
    per_atom_props = set(first_scope["per_atom"])
    per_batch_props = set(first_scope["per_batch"])

    # Initialize result with global properties from first state
    concatenated = {prop: getattr(first_state, prop) for prop in global_props}

    # Pre-allocate lists for tensors to concatenate
    per_atom_tensors = {prop: [] for prop in per_atom_props}
    per_batch_tensors = {prop: [] for prop in per_batch_props}
    new_batch_indices = []
    batch_offset = 0

    # Process all states in a single pass
    for state in states:
        # Move state to target device if needed
        if state.device != target_device:
            state = state_to_device(state, target_device)

        # Collect per-atom properties
        for prop in per_atom_props:
            # if hasattr(state, prop):
            per_atom_tensors[prop].append(getattr(state, prop))

        # Collect per-batch properties
        for prop in per_batch_props:
            # if hasattr(state, prop):
            per_batch_tensors[prop].append(getattr(state, prop))

        # Update batch indices
        num_batches = state.n_batches
        new_indices = state.batch + batch_offset
        new_batch_indices.append(new_indices)
        batch_offset += num_batches

    # Concatenate collected tensors
    for prop, tensors in per_atom_tensors.items():
        # if tensors:
        concatenated[prop] = torch.cat(tensors, dim=0)

    for prop, tensors in per_batch_tensors.items():
        # if tensors:
        concatenated[prop] = torch.cat(tensors, dim=0)

    # Concatenate batch indices
    concatenated["batch"] = torch.cat(new_batch_indices)

    # Create a new instance of the same class
    return state_class(**concatenated)


def initialize_state(
    system: StateLike,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Initialize state tensors from a system.

    Args:
        system: Input system to convert to state tensors
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: State tensors initialized from input system

    Raises:
        ValueError: If system type is not supported
    """
    # TODO: create a way to pass velocities from pmg and ase

    if isinstance(system, BaseState):
        return state_to_device(system, device, dtype)

    if isinstance(system, list) and all(isinstance(s, BaseState) for s in system):
        if not all(state.n_batches == 1 for state in system):
            raise ValueError(
                "When providing a list of states, to the initialize_state function, "
                "all states must have n_batches == 1. To fix this, you can split the "
                "states into individual states with the split_state function."
            )
        return concatenate_states(system)

    converters = [
        ("pymatgen.core", "Structure", structures_to_state),
        ("ase", "Atoms", atoms_to_state),
        ("phonopy.structure.atoms", "PhonopyAtoms", phonopy_to_state),
    ]

    # Try each converter
    for module_path, class_name, converter_func in converters:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            if isinstance(system, cls) or (
                isinstance(system, list) and all(isinstance(s, cls) for s in system)
            ):
                return converter_func(system, device, dtype)
        except ImportError:
            continue

    # remaining code just for informative error
    is_list = isinstance(system, list)
    all_same_type = (
        is_list and all(isinstance(s, type(system[0])) for s in system) and system
    )
    if is_list and not all_same_type:
        raise ValueError(
            f"All items in list must be of the same type, "
            f"found {type(system[0])} and {type(system[1])}"
        )

    system_type = f"list[{type(system[0])}]" if is_list else type(system)

    raise ValueError(f"Unsupported system type, {system_type}")
