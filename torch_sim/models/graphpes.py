"""An interface for using arbitrary GraphPESModels in torch_sim.

This module provides a TorchSim wrapper of the GraphPES models for computing
energies, forces, and stresses of atomistic systems. It serves as a wrapper around
the graph_pes library, integrating it with the torch_sim framework to enable seamless
simulation of atomistic systems with machine learning potentials.

The GraphPESWrapper class adapts GraphPESModels to the ModelInterface protocol,
allowing them to be used within the broader torch_sim simulation framework.

Notes:
    This implementation requires graph_pes to be installed and accessible.
    It supports various model configurations through model instances or model paths.
"""

import typing
from pathlib import Path

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict


try:
    from graph_pes import AtomicGraph, GraphPESModel
    from graph_pes.atomic_graph import PropertyKey, to_batch
    from graph_pes.models import load_model

except ImportError:
    PropertyKey = str

    class GraphPESWrapper(torch.nn.Module, ModelInterface):  # type: ignore[reportRedeclaration]
        """GraphPESModel wrapper for torch_sim.

        This class is a placeholder for the GraphPESWrapper class.
        It raises an ImportError if graph_pes is not installed.
        """

        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:  # noqa: D107
            raise ImportError("graph_pes must be installed to use this model.")

    class AtomicGraph:  # type: ignore[reportRedeclaration]  # noqa: D101
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:  # noqa: D107,ARG002
            raise ImportError("graph_pes must be installed to use this model.")

    class GraphPESModel(torch.nn.Module):  # type: ignore[reportRedeclaration]  # noqa: D101
        pass


def state_to_atomic_graph(state: SimState, cutoff: torch.Tensor) -> AtomicGraph:
    """Convert a SimState object into an AtomicGraph object.

    Args:
        state: SimState object containing atomic positions, cell, and atomic numbers
        cutoff: Cutoff radius for the neighbor list

    Returns:
        AtomicGraph object representing the batched structures
    """
    graphs = []

    for i in range(state.n_batches):
        batch_mask = state.batch == i
        R = state.positions[batch_mask]
        Z = state.atomic_numbers[batch_mask]
        cell = state.row_vector_cell[i]
        nl, shifts = vesin_nl_ts(
            R,
            cell,
            state.pbc,
            # graph-pes models internally trim the neighbour list to the
            # model's cutoff value. To ensure no strange edge effects whereby
            # edges that are exactly `cutoff` long are included/excluded,
            # we bump this up slightly here
            cutoff + 1e-5,
        )

        graphs.append(
            AtomicGraph(
                Z=Z.long(),
                R=R,
                cell=cell,
                neighbour_list=nl.long(),
                neighbour_cell_offsets=shifts,
                properties={},
                cutoff=cutoff.item(),
                other={},
            )
        )

    return to_batch(graphs)


class GraphPESWrapper(torch.nn.Module, ModelInterface):
    """Wrapper for GraphPESModel in TorchSim.

    This class provides a TorchSim wrapper around GraphPESModel instances,
    allowing them to be used within the broader torch_sim simulation framework.

    The graph-pes package allows for the training of existing model architectures,
    including SchNet, PaiNN, MACE, NequIP, TensorNet, EDDP and more.
    You can use any of these, as well as your own custom architectures, with this wrapper.
    See the the graph-pes repo for more details: https://github.com/jla-gardner/graph-pes

    Args:
        model: GraphPESModel instance, or a path to a model file
        device: Device to run the model on
        dtype: Data type for the model
        compute_forces: Whether to compute forces
        compute_stress: Whether to compute stress

    Example:
        >>> from torch_sim.models import GraphPESWrapper
        >>> from graph_pes.models import load_model
        >>> model = load_model("path/to/model.pt")
        >>> wrapper = GraphPESWrapper(model)
        >>> state = SimState(
        ...     positions=torch.randn(10, 3),
        ...     cell=torch.eye(3),
        ...     atomic_numbers=torch.randint(1, 104, (10,)),
        ... )
        >>> wrapper(state)
    """

    def __init__(
        self,
        model: GraphPESModel | str | Path,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_forces: bool = True,
        compute_stress: bool = True,
    ) -> None:
        """Initialize the GraphPESWrapper.

        Args:
            model: GraphPESModel instance, or a path to a model file
            device: Device to run the model on
            dtype: Data type for the model
            compute_forces: Whether to compute forces
            compute_stress: Whether to compute stress
        """
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype

        _model = typing.cast(
            "GraphPESModel",
            (
                model if isinstance(model, GraphPESModel) else load_model(model)  # type: ignore[arg-type]
            ),
        )
        self._gp_model = _model.to(device=self._device, dtype=self._dtype)

        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        self._properties: list[PropertyKey] = ["energy"]
        if self._compute_forces:
            self._properties.append("forces")
        if self._compute_stress:
            self._properties.append("stress")

        if self._gp_model.cutoff.item() < 0.5:
            self._memory_scales_with = "n_atoms"

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Forward pass for the GraphPESWrapper.

        Args:
            state: SimState object containing atomic positions, cell, and atomic numbers

        Returns:
            Dictionary containing the computed energies, forces, and stresses
            (where applicable)
        """
        if not isinstance(state, SimState):
            state = SimState(**state)  # type: ignore[arg-type]

        atomic_graph = state_to_atomic_graph(state, self._gp_model.cutoff)
        return self._gp_model.predict(atomic_graph, self._properties)  # type: ignore[return-value]
