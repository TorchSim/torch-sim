"""ORB: PyTorch implementation of ORB models for atomistic simulations.

This module provides a TorchSim wrapper of the ORB models for computing
energies, forces, and stresses of atomistic systems. It serves as a wrapper around
the ORB models library, integrating it with the torch_sim framework to enable seamless
simulation of atomistic systems with machine learning potentials.

The OrbModel class adapts ORB models to the ModelInterface protocol,
allowing them to be used within the broader torch_sim simulation framework.

Notes:
    This implementation requires orb_models to be installed and accessible.
    It supports various model configurations through model instances or model paths.
"""

from __future__ import annotations

import typing
from pathlib import Path

import torch

from torch_sim.elastic import voigt_6_to_full_3x3_stress
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState, StateDict


try:
    from ase.geometry import cell_to_cellpar
    from orb_models.forcefield import featurization_utilities as feat_util
    from orb_models.forcefield.atomic_system import SystemConfig
    from orb_models.forcefield.base import AtomGraphs, _map_concat
    from orb_models.forcefield.graph_regressor import GraphRegressor

    try:
        from orb_models.forcefield.featurization_utilities import EdgeCreationMethod
    except ImportError as exp:
        raise ImportError(
            "Orb model version is too old, interface requires >v0.4.2. If release is "
            "not yet available, install from github."
        ) from exp
except ImportError:

    class OrbModel(torch.nn.Module, ModelInterface):
        """ORB model wrapper for torch_sim.

        This class is a placeholder for the OrbModel class.
        It raises an ImportError if orb_models is not installed.
        """

        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            """Dummy constructor to raise ImportError."""
            raise ImportError("orb_models must be installed to use this model.")


if typing.TYPE_CHECKING:
    from orb_models.forcefield.conservative_regressor import (
        ConservativeForcefieldRegressor,
    )
    from orb_models.forcefield.featurization_utilities import EdgeCreationMethod
    from orb_models.forcefield.graph_regressor import GraphRegressor


def state_to_atom_graphs(  # noqa: PLR0915
    state: SimState,
    *,
    wrap: bool = True,
    edge_method: EdgeCreationMethod | None = None,
    system_config: SystemConfig | None = None,
    max_num_neighbors: int | None = None,
    system_id: int | None = None,  # noqa: ARG001
    half_supercell: bool = False,
    device: torch.device | None = None,
    output_dtype: torch.dtype | None = None,
    graph_construction_dtype: torch.dtype | None = None,
) -> AtomGraphs:
    """Convert a SimState object into AtomGraphs format, ready for use in an ORB model.

    Args:
        state: SimState object containing atomic positions, cell, and atomic numbers
        wrap: Whether to wrap atomic positions into the central unit cell (if there is
        one).
        edge_method (EdgeCreationMethod, optional): The method to use for graph edge
            construction. If None, the edge method is chosen automatically based on
            device and system size.
        system_config: The system configuration to use for graph construction.
        max_num_neighbors: Maximum number of neighbors each node can send messages to.
            If None, will use system_config.max_num_neighbors.
        system_id: Optional index that is relative to a particular dataset.
        half_supercell (bool): Whether to use half the supercell for graph construction.
            This can improve performance for large systems.
        device: The device to put the tensors on.
        output_dtype: The dtype to use for all floating point tensors stored on the
            AtomGraphs object.
        graph_construction_dtype: The dtype to use for floating point tensors in the
            graph construction.

    Returns:
        AtomGraphs object containing the graph representation of the atomic system
    """
    if system_config is None:
        system_config = SystemConfig(radius=6.0, max_num_neighbors=20)

    # Handle batch information if present
    if state.batch is not None:
        n_node = torch.bincount(state.batch)
    else:
        n_node = torch.tensor([len(state.positions)])

    # Set default dtype if not provided
    output_dtype = torch.get_default_dtype() if output_dtype is None else output_dtype
    graph_construction_dtype = (
        torch.get_default_dtype()
        if graph_construction_dtype is None
        else graph_construction_dtype
    )

    # Extract data from SimState
    positions = state.positions
    row_vector_cell = (
        state.row_vector_cell
    )  # Orb uses row vector cell convention for neighbor list
    atomic_numbers = state.atomic_numbers.long()

    # Create PBC tensor based on state.pbc
    pbc = torch.tensor([state.pbc, state.pbc, state.pbc], dtype=torch.bool)

    max_num_neighbors = max_num_neighbors or system_config.max_num_neighbors

    # Get atom embeddings for the model
    n_atoms = len(atomic_numbers)
    k_hot = (
        system_config.diffuse_atom_types
        if hasattr(system_config, "diffuse_atom_types")
        else False
    )

    if k_hot:
        atom_type_embedding = torch.ones(n_atoms, 118) * -feat_util.ATOM_TYPE_K
        atom_type_embedding[torch.arange(n_atoms), atomic_numbers] = feat_util.ATOM_TYPE_K
    else:
        atom_type_embedding = torch.nn.functional.one_hot(atomic_numbers, num_classes=118)
    atomic_numbers_embedding = atom_type_embedding.to(output_dtype)

    # Wrap positions into the central cell if needed
    if wrap and (torch.any(row_vector_cell != 0) and torch.any(pbc)):
        positions = feat_util.batch_map_to_pbc_cell(positions, row_vector_cell, n_node)

    # Compute edges of the graph
    edge_index, edge_vectors, unit_shifts, batch_num_edges = (
        feat_util.batch_compute_pbc_radius_graph(
            positions=positions,
            cells=row_vector_cell,
            pbc=pbc.unsqueeze(0).repeat(len(n_node), 1),
            radius=system_config.radius,
            n_node=n_node,
            max_number_neighbors=torch.tensor([max_num_neighbors] * len(n_node)),
            edge_method=edge_method,
            half_supercell=half_supercell,
            device=device,
        )
    )
    senders, receivers = edge_index[0], edge_index[1]

    n_systems = state.batch.max().item() + 1
    node_feats_list = []
    edge_feats_list = []
    graph_feats_list = []
    system_edges = torch.repeat_interleave(
        torch.arange(n_systems, device=state.device), batch_num_edges
    )
    for i in range(n_systems):
        batch_mask = state.batch == i
        system_edge_mask = system_edges == i
        try:
            positions_per_system = positions[batch_mask]
            atomic_numbers_per_system = atomic_numbers[batch_mask]
            atomic_numbers_embedding_per_system = atomic_numbers_embedding[batch_mask]
            edge_vectors_per_system = edge_vectors[system_edge_mask]
            unit_shifts_per_system = unit_shifts[system_edge_mask]
        except Exception:  # noqa: BLE001
            import pdb  # noqa: T100

            pdb.set_trace()  # noqa: T100

        cell_per_system = row_vector_cell[i]
        pbc_per_system = pbc
        lattice_per_system = torch.from_numpy(
            cell_to_cellpar(cell_per_system.squeeze(0).cpu().numpy())
        )

        # Create features dictionaries
        node_feats = {
            "positions": positions_per_system,
            "atomic_numbers": atomic_numbers_per_system.to(torch.long),
            "atomic_numbers_embedding": atomic_numbers_embedding_per_system,
            "atom_identity": torch.arange(
                len(positions_per_system), device=positions_per_system.device
            ).to(torch.long),
        }

        edge_feats = {
            "vectors": edge_vectors_per_system,
            "unit_shifts": unit_shifts_per_system,
        }

        graph_feats = {
            "cell": cell_per_system,
            "pbc": pbc_per_system,
            "lattice": lattice_per_system.to(device=positions_per_system.device),
        }

        # Add batch dimension to non-scalar graph features
        graph_feats = {
            k: v.unsqueeze(0) if v.numel() > 1 else v for k, v in graph_feats.items()
        }

        node_feats_list.append(node_feats)
        edge_feats_list.append(edge_feats)
        graph_feats_list.append(graph_feats)

    # Create and return AtomGraphs object
    return AtomGraphs(
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=batch_num_edges,
        node_features=_map_concat(node_feats_list),
        edge_features=_map_concat(edge_feats_list),
        system_features=_map_concat(graph_feats_list),
        node_targets={},  # No targets since we're using for inference
        edge_targets={},
        system_targets={},
        fix_atoms=None,  # No fixed atoms in SimState
        tags=None,  # No tags in SimState
        radius=system_config.radius,
        max_num_neighbors=torch.tensor([max_num_neighbors] * len(n_node)),
        system_id=None,
    ).to(device=device, dtype=output_dtype)


class OrbModel(torch.nn.Module, ModelInterface):
    """Computes atomistic energies, forces and stresses using an ORB model.

    This class wraps an ORB model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, configuration, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    Attributes:
        model (Union[GraphRegressor, ConservativeForcefieldRegressor]): The ORB model
        system_config (SystemConfig): Configuration for the atomic system
        conservative (bool): Whether to use conservative forces/stresses calculation
        implemented_properties (list): Properties the model can compute
        _dtype (torch.dtype): Data type used for computation
        _device (torch.device): Device where computation is performed
        _edge_method (EdgeCreationMethod): Method for creating edges in the graph
        _max_num_neighbors (int): Maximum number of neighbors for each atom
        _half_supercell (bool): Whether to use half supercell optimization
        _memory_scales_with (str): What the memory usage scales with

    Examples:
        >>> model = OrbModel(model=loaded_orb_model, compute_stress=True)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: GraphRegressor | ConservativeForcefieldRegressor | str | Path,
        *,  # force remaining arguments to be keyword-only
        conservative: bool | None = None,
        compute_stress: bool = True,
        compute_forces: bool = True,
        system_config: SystemConfig | None = None,
        max_num_neighbors: int | None = None,
        edge_method: EdgeCreationMethod | None = None,
        half_supercell: bool | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the OrbModel with specified configuration.

        Loads an ORB model from either a model object or a model path.
        Sets up the model parameters for subsequent use in energy and force calculations.

        Args:
            model (Union[GraphRegressor, ConservativeForcefieldRegressor, str, Path]):
                Either a model object or a path to a saved model
            conservative (bool | None): Whether to use conservative forces/stresses
                If None, determined based on model type
            compute_stress (bool): Whether to compute stress tensor
            compute_forces (bool): Whether to compute forces
            system_config (SystemConfig | None): Configuration for the atomic system
                If None, defaults to SystemConfig(radius=6.0, max_num_neighbors=20)
            max_num_neighbors (int | None): Maximum number of neighbors for each atom
            edge_method (EdgeCreationMethod | None): Method for creating edges
            half_supercell (bool | None): Whether to use half supercell optimization
                If None, determined based on system size
            device (torch.device | str | None): Device to run the model on
            dtype (torch.dtype | None): Data type for computation

        Raises:
            ValueError: If conservative mode is requested but model doesn't support it
            ImportError: If orb_models is not installed
        """
        super().__init__()

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._dtype = dtype
        self._memory_scales_with = "n_atoms"
        self._compute_stress = compute_stress
        self._compute_forces = compute_forces

        # Set up system configuration
        self.system_config = system_config or SystemConfig(
            radius=6.0, max_num_neighbors=20
        )
        self._max_num_neighbors = max_num_neighbors
        self._edge_method = edge_method
        self._half_supercell = half_supercell

        # Load model if path is provided
        if isinstance(model, str | Path):
            model = torch.load(model, map_location=self._device)

        self.model = model.to(self._device)
        self.model = self.model.eval()

        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        # Determine if the model is conservative
        model_is_conservative = hasattr(self.model, "grad_forces_name")
        self.conservative = (
            conservative if conservative is not None else model_is_conservative
        )

        if self.conservative and not model_is_conservative:
            raise ValueError(
                "Conservative mode requested, but model is not a "
                "ConservativeForcefieldRegressor."
            )

        # Set up implemented properties
        self.implemented_properties = self.model.properties

        # Add forces and stress to implemented properties if conservative model
        if self.conservative:
            if "forces" not in self.implemented_properties:
                self.implemented_properties.append("forces")
            if compute_stress and "stress" not in self.implemented_properties:
                self.implemented_properties.append("stress")

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

        half_supercell = (
            torch.max(torch.det(state.cell)) > 1000
            if self._half_supercell is None
            else self._half_supercell
        )

        # Convert state to atom graphs
        batch = state_to_atom_graphs(
            state,
            system_config=self.system_config,
            max_num_neighbors=self._max_num_neighbors,
            edge_method=self._edge_method,
            half_supercell=half_supercell,
            device=self.device,
        )

        # Run forward pass
        predictions = self.model.predict(batch)

        results = {}
        model_has_direct_heads = (
            "forces" in self.model.heads and "stress" in self.model.heads
        )
        for prop in self.implemented_properties:
            # The model has no direct heads for forces/stress, so we skip these props.
            if not model_has_direct_heads and prop == "forces":
                continue
            if not model_has_direct_heads and prop == "stress":
                continue
            _property = "energy" if prop == "free_energy" else prop
            results[prop] = predictions[_property].squeeze()

        if self.conservative:
            results["direct_forces"] = results["forces"]
            results["direct_stress"] = results["stress"]
            results["forces"] = results[self.model.grad_forces_name]
            results["stress"] = results[self.model.grad_stress_name]

        if "stress" in results and results["stress"].shape[-1] == 6:
            # TODO: is there a point to converting the direct stress if conservative?
            results["stress"] = voigt_6_to_full_3x3_stress(results["stress"])

        return results
