"""Utilities for building dataloaders and processing batches."""

from typing import Any

import numpy as np
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader as DataLoader_pyg

from torch_sim.models.mattersim.utils.converter import GraphConverter


def batch_to_dict(
    graph_batch: Any,
    model_type: str = "m3gnet",
) -> dict[str, torch.Tensor]:
    """Convert a graph batch to a dictionary of tensors.

    Args:
        graph_batch: The graph batch to convert
        model_type: The type of model being used

    Returns:
        Dictionary containing the graph tensors
    """
    if model_type == "m3gnet":
        # TODO: key_list
        atom_pos = graph_batch.atom_pos
        cell = graph_batch.cell
        pbc_offsets = graph_batch.pbc_offsets
        atom_attr = graph_batch.atom_attr
        edge_index = graph_batch.edge_index
        three_body_indices = graph_batch.three_body_indices
        num_three_body = graph_batch.num_three_body
        num_bonds = graph_batch.num_bonds
        num_triple_ij = graph_batch.num_triple_ij
        num_atoms = graph_batch.num_atoms
        num_graphs = graph_batch.num_graphs
        num_graphs = torch.tensor(num_graphs)
        batch = graph_batch.batch

        # Resemble input dictionary
        graph_dict = {}
        graph_dict["atom_pos"] = atom_pos
        graph_dict["cell"] = cell
        graph_dict["pbc_offsets"] = pbc_offsets
        graph_dict["atom_attr"] = atom_attr
        graph_dict["edge_index"] = edge_index
        graph_dict["three_body_indices"] = three_body_indices
        graph_dict["num_three_body"] = num_three_body
        graph_dict["num_bonds"] = num_bonds
        graph_dict["num_triple_ij"] = num_triple_ij
        graph_dict["num_atoms"] = num_atoms
        graph_dict["num_graphs"] = num_graphs
        graph_dict["batch"] = batch
    else:
        raise NotImplementedError

    return graph_dict


def build_dataloader(
    *,
    atoms: list[Atoms] | None = None,
    energies: list[float] | None = None,
    forces: list[np.ndarray] | None = None,
    stresses: list[np.ndarray] | None = None,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    batch_size: int = 64,
    model_type: str = "m3gnet",
    shuffle: bool = False,
    only_inference: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset: Any | None = None,
    **kwargs,
) -> DataLoader_pyg:
    """Build a dataloader given a list of atoms.

    Args:
        atoms: A list of atoms in ase format
        energies: A list of energy (float) with unit eV
        forces: A list of nx3 force matrix (np.ndarray) with unit eV/Å,
            where n is the number of atom in each structure
        stresses: A list of 3x3 stress matrix (np.ndarray) with unit GPa
        cutoff: Cutoff radius for neighbor finding
        threebody_cutoff: Cutoff radius for three-body interactions
        batch_size: Number of graphs in each batch
        model_type: Type of model to build dataloader for
        shuffle: Whether to shuffle the dataset
        only_inference: If True, energies, forces and stresses will be ignored
        num_workers: Number of workers for dataloader
        pin_memory: If True, the datasets will be stored in GPU or CPU memory
        dataset: The dataset object for the dataloader. Only used with
            graphormer and geomformer.
        **kwargs: Additional arguments passed to the converter

    Returns:
        A PyTorch Geometric DataLoader object
    """
    converter = GraphConverter(
        model_type, cutoff, has_threebody=True, threebody_cutoff=threebody_cutoff
    )

    preprocessed_data = []

    if dataset is None:
        if not only_inference:
            assert energies is not None, (
                "energies must be provided if only_inference is False"
            )
        if stresses is not None:
            assert np.array(stresses[0]).shape == (
                3,
                3,
            ), "stresses must be a list of 3x3 matrices"

        length = len(atoms)
        if energies is None:
            energies = [None] * length
        if forces is None:
            forces = [None] * length
        if stresses is None:
            stresses = [None] * length

    if model_type == "m3gnet":
        for graph, energy, force, stress in zip(
            atoms, energies, forces, stresses, strict=True
        ):
            graph = converter.convert(graph.copy(), energy, force, stress, **kwargs)
            if graph is not None:
                preprocessed_data.append(graph)
    else:
        raise NotImplementedError

    return DataLoader_pyg(
        preprocessed_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
