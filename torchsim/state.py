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
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

import torch


if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure


from typing import TypeVar, Union


T = TypeVar("T", bound="BaseState")
StateLike = Union["Atoms", "Structure", list["Atoms"], list["Structure"], T, list[T]]


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


def slice_substate(
    state: BaseState,
    batch_index: int,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> Self:
    """Slice a substate from the BaseState.

    Args:
        state: The state to slice
        batch_index: The index of the batch to slice
        ambiguous_handling: How to handle ambiguous properties

    Returns:
        A BaseState object containing the sliced substate
    """
    scope = infer_property_scope(state, ambiguous_handling=ambiguous_handling)

    # Create a mask for the atoms in the specified batch
    batch_mask = state.batch == batch_index

    # Initialize a dictionary to hold the sliced attributes
    sliced_attrs = {}

    # Process global properties (unchanged)
    for attr_name in scope["global"]:
        sliced_attrs[attr_name] = getattr(state, attr_name)

    # Process per-atom properties (filter by batch mask)
    for attr_name in scope["per_atom"]:
        attr_value = getattr(state, attr_name)
        sliced_attrs[attr_name] = attr_value[batch_mask]

    # Process per-batch properties (select the specific batch)
    for attr_name in scope["per_batch"]:
        attr_value = getattr(state, attr_name)
        sliced_attrs[attr_name] = attr_value[batch_index : batch_index + 1]

    # Create a new batch tensor with all zeros (single batch)
    n_sliced_atoms = sliced_attrs.get("positions").shape[0]
    sliced_attrs["batch"] = torch.zeros(
        n_sliced_atoms, device=state.device, dtype=torch.int64
    )

    # Create a new instance of the same class
    return type(state)(**sliced_attrs)


def concatenate_states(  # noqa: C901
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

    # Categorize properties by scope for each state
    property_scopes = [infer_property_scope(state) for state in states]

    # Collect all property names across all states
    all_props = set()
    for scope in property_scopes:
        for scope_type in scope.values():
            all_props.update(scope_type)

    # Initialize dictionaries to hold concatenated properties
    concatenated = {}

    # Process global properties (take from first state)
    for prop_name in property_scopes[0]["global"]:
        concatenated[prop_name] = getattr(first_state, prop_name)

    # Process per-atom properties (concatenate)
    for prop_name in set().union(*[scope["per_atom"] for scope in property_scopes]):
        tensors = [
            getattr(state, prop_name) for state in states if hasattr(state, prop_name)
        ]
        if tensors:
            concatenated[prop_name] = torch.cat(tensors, dim=0)

    # Process per-batch properties (concatenate)
    for prop_name in set().union(*[scope["per_batch"] for scope in property_scopes]):
        tensors = [
            getattr(state, prop_name) for state in states if hasattr(state, prop_name)
        ]
        if tensors:
            concatenated[prop_name] = torch.cat(tensors, dim=0)

    # Create new batch indices that account for existing batch structure
    new_batch_indices = []
    batch_offset = 0

    device = device or states[0].device
    for state in states:
        state = state_to_device(state, device)

        # Get the number of unique batches in this state
        num_batches = len(torch.unique(state.batch))

        # For each atom, map its current batch index to a new index with the offset
        new_indices = state.batch + batch_offset
        new_batch_indices.append(new_indices)

        # Update the offset for the next state
        batch_offset += num_batches

    # Concatenate all batch indices
    concatenated["batch"] = torch.cat(new_batch_indices)

    # Create a new instance of the same class
    return state_class(**concatenated)
