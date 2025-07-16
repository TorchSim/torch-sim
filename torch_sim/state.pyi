from typing import Literal, TypeVar, overload

import torch

from torch_sim.typing import StateLike

class SimState:
    positions: torch.Tensor
    masses: torch.Tensor
    cell: torch.Tensor
    pbc: bool
    atomic_numbers: torch.Tensor
    batch: torch.Tensor

    @overload
    def __init__(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        cell: torch.Tensor,
        pbc: bool,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        cell: torch.Tensor,
        pbc: bool,
        atomic_numbers: torch.Tensor,
        system_idx: torch.Tensor | None = None,
    ) -> None: ...

class MDSimState(SimState):
    velocities: torch.Tensor
    masses: torch.Tensor

class DeformGradMixin(MDSimState):
    @property
    def momenta(self) -> torch.Tensor: ...
    @property
    def reference_row_vector_cell(self) -> torch.Tensor: ...
    @reference_row_vector_cell.setter
    def reference_row_vector_cell(self, value: torch.Tensor) -> None: ...
    @staticmethod
    def _deform_grad(
        reference_row_vector_cell: torch.Tensor, row_vector_cell: torch.Tensor
    ) -> torch.Tensor: ...
    def deform_grad(self) -> torch.Tensor: ...

def _normalize_batch_indices(
    batch_indices: int | list[int] | slice | torch.Tensor,
    n_batches: int,
    device: torch.device,
) -> torch.Tensor: ...

SimStateT = TypeVar("SimStateT", bound=SimState)

def state_to_device(
    state: SimStateT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SimStateT: ...
def infer_property_scope(
    state: SimState,
    ambiguous_handling: Literal["error", "globalize", "globalize_warn"] = "error",
) -> dict[Literal["global", "per_atom", "per_batch"], list[str]]: ...
def _get_property_attrs(
    state: SimState, ambiguous_handling: Literal["error", "globalize"] = "error"
) -> dict[str, dict]: ...
def _filter_attrs_by_mask(
    attrs: dict[str, dict],
    atom_mask: torch.Tensor,
    batch_mask: torch.Tensor,
) -> dict: ...
def _split_state(
    state: SimStateT,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> list[SimStateT]: ...
def _pop_states(
    state: SimState,
    pop_indices: list[int] | torch.Tensor,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> tuple[SimState, list[SimState]]: ...
def _slice_state(
    state: SimStateT,
    batch_indices: list[int] | torch.Tensor,
    ambiguous_handling: Literal["error", "globalize"] = "error",
) -> SimStateT: ...
def concatenate_states(
    states: list[SimState], device: torch.device | None = None
) -> SimState: ...
def initialize_state(
    system: StateLike,
    device: torch.device,
    dtype: torch.dtype,
) -> SimState: ...
