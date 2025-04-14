"""TorchSim wrapper for SevenNet models."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from torch_sim.elastic import voigt_6_to_full_3x3_stress
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict


if TYPE_CHECKING:
    from collections.abc import Callable

    from sevenn.nn.sequential import AtomGraphSequential


try:
    import sevenn._keys as key
    import torch
    from sevenn.atom_graph_data import AtomGraphData
    from sevenn.calculator import torch_script_type
    from torch_geometric.loader.dataloader import Collater

except ImportError:

    class SevenNetModel(torch.nn.Module, ModelInterface):
        """SevenNet model wrapper for torch_sim.

        This class is a placeholder for the SevenNetModel class.
        It raises an ImportError if sevenn is not installed.
        """

        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            """Dummy constructor to raise ImportError."""
            raise ImportError("sevenn must be installed to use this model.")


class SevenNetModel(torch.nn.Module, ModelInterface):
    """Computes atomistic energies, forces and stresses using an SevenNet model.

    This class wraps an SevenNet model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, configuration, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    Examples:
        >>> model = SevenNetModel(model=loaded_sevenn_model)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: AtomGraphSequential,
        *,  # force remaining arguments to be keyword-only
        modal: str | None = None,
        neighbor_list_fn: Callable = vesin_nl_ts,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the SevenNetModel with specified configuration.

        Loads an SevenNet model from either a model object or a model path.
        Sets up the model parameters for subsequent use in energy and force calculations.

        Args:
            model (AtomGraphSequential): The SevenNet model to wrap.
            modal (str | None): modal (fidelity) if given model is multi-modal model.
                for 7net-mf-ompa, it should be one of 'mpa' (MPtrj + sAlex) or 'omat24'
                (OMat24).
            neighbor_list_fn (Callable): Neighbor list function to use.
                Default is vesin_nl_ts.
            device (torch.device | str | None): Device to run the model on
            dtype (torch.dtype | None): Data type for computation

        Raises:
            ValueError: the model doesn't have a cutoff
            ValueError: the model has a modal_map but modal is not given
            ValueError: the modal given is not in the modal_map
        """
        super().__init__()

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        if torch.dtype is not torch.float32:
            warnings.warn(
                "SevenNetModel currently only supports"
                "float32, but received different dtype",
                UserWarning,
                stacklevel=2,
            )

        self._dtype = dtype
        self._memory_scales_with = "n_atoms_x_density"
        self._compute_stress = True
        self._compute_forces = True

        if model.cutoff == 0.0:
            raise ValueError("Model cutoff seems not initialized")

        model.eval_type_map = torch.tensor(
            data=True,
        )  # TODO: from sevenn not sure if needed
        model.set_is_batch_data(True)
        model_loaded = model
        self.cutoff = torch.tensor(model.cutoff)
        self.neighbor_list_fn = neighbor_list_fn

        self.model = model_loaded

        self.modal = None
        modal_map = self.model.modal_map
        if modal_map:
            modal_ava = list(modal_map.keys())
            if not modal:
                raise ValueError(f"modal argument missing (avail: {modal_ava})")
            if modal not in modal_ava:
                raise ValueError(f"unknown modal {modal} (not in {modal_ava})")
            self.modal = modal
        elif not self.model.modal_map and modal:
            warnings.warn(
                f"modal={modal} is ignored as model has no modal_map",
                stacklevel=2,
            )

        self.model = model.to(self._device)
        self.model = self.model.eval()

        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Perform forward pass to compute energies, forces, and other properties.

        Takes a simulation state and computes the properties implemented by the model,
        such as energy, forces, and stresses.

        Args:
            state (SimState | StateDict): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3],
                    if compute_stress is True

        Notes:
            The state is automatically transferred to the model's device if needed.
            All output tensors are detached from the computation graph.
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        if state.device != self._device:
            state = state.to(self._device)

        # TODO: is this clone necessary?
        state = state.clone()

        data_list = []
        for b in range(state.batch.max().item() + 1):
            batch_mask = state.batch == b

            pos = state.positions[batch_mask]
            # SevenNet uses row vector cell convention for neighbor list
            row_vector_cell = state.row_vector_cell[b]
            pbc = state.pbc
            atomic_numbers = state.atomic_numbers[batch_mask]

            edge_idx, shifts_idx = self.neighbor_list_fn(
                positions=pos,
                cell=row_vector_cell,
                pbc=pbc,
                cutoff=self.cutoff,
            )

            shifts = torch.mm(shifts_idx, row_vector_cell)
            edge_vec = pos[edge_idx[1]] - pos[edge_idx[0]] + shifts

            data = {
                key.NODE_FEATURE: atomic_numbers,
                key.ATOMIC_NUMBERS: atomic_numbers.to(
                    dtype=torch.int64, device=self.device
                ),
                key.POS: pos,
                key.EDGE_IDX: edge_idx,
                key.EDGE_VEC: edge_vec,
                key.CELL: row_vector_cell,
                key.CELL_SHIFT: shifts_idx,
                key.CELL_VOLUME: torch.det(row_vector_cell),
                key.NUM_ATOMS: torch.tensor(len(atomic_numbers), device=self.device),
                key.DATA_MODALITY: self.modal,
            }
            data[key.INFO] = {}

            data = AtomGraphData(**data)
            data_list.append(data)

        batched_data = Collater([], follow_batch=None, exclude_keys=None)(data_list)
        batched_data.to(self.device)

        if isinstance(self.model, torch_script_type):
            batched_data[key.NODE_FEATURE] = torch.tensor(
                [self.type_map[z.item()] for z in data[key.NODE_FEATURE]],
                dtype=torch.int64,
                device=self.device,
            )
            batched_data[key.POS].requires_grad_(
                requires_grad=True
            )  # backward compatibility
            batched_data[key.EDGE_VEC].requires_grad_(requires_grad=True)
            batched_data = batched_data.to_dict()
            del batched_data["data_info"]

        output = self.model(batched_data)

        results = {}
        energy = output[key.PRED_TOTAL_ENERGY]
        if energy is not None:
            results["energy"] = energy.detach()
        else:
            results["energy"] = torch.zeros(
                state.batch.max().item() + 1, device=self.device
            )

        forces = output[key.PRED_FORCE]
        if forces is not None:
            results["forces"] = forces.detach()

        stress = output[key.PRED_STRESS]
        if stress is not None:
            results["stress"] = voigt_6_to_full_3x3_stress(stress.detach())

        return results
