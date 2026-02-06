"""Symmetry utilities for crystal structures using moyopy.

Functions operate on single (unbatched) systems. The ``n_ops`` dimension
refers to the number of symmetry operations of the space group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from moyopy import MoyoDataset

    from torch_sim.state import SimState


def _moyo_dataset(
    cell: torch.Tensor,
    frac_pos: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1e-4,
) -> MoyoDataset:
    """Get MoyoDataset from cell, fractional positions, and atomic numbers."""
    from moyopy import Cell, MoyoDataset

    moyo_cell = Cell(
        basis=cell.detach().cpu().tolist(),
        positions=frac_pos.detach().cpu().tolist(),
        numbers=atomic_numbers.detach().cpu().int().tolist(),
    )
    return MoyoDataset(moyo_cell, symprec=symprec)


def _extract_symmetry_ops(
    dataset: MoyoDataset, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract rotation and translation tensors from a MoyoDataset.

    Returns:
        (rotations, translations) with shapes (n_ops, 3, 3) and (n_ops, 3).
    """
    rotations = torch.as_tensor(
        dataset.operations.rotations, dtype=dtype, device=device
    ).round()
    translations = torch.as_tensor(
        dataset.operations.translations, dtype=dtype, device=device
    )
    return rotations, translations


def get_symmetry_datasets(state: SimState, symprec: float = 1e-4) -> list[MoyoDataset]:
    """Get MoyoDataset for each system in a SimState."""
    datasets = []
    for single in state.split():
        cell = single.row_vector_cell[0]
        frac = single.positions @ torch.linalg.inv(cell)
        datasets.append(_moyo_dataset(cell, frac, single.atomic_numbers, symprec))
    return datasets


# Above this threshold, build_symmetry_map falls back to a per-operation loop
# to avoid allocating an O(n_ops * n_atoms^2) tensor that can OOM on supercells.
_SYMM_MAP_CHUNK_THRESHOLD = 200


def build_symmetry_map(
    rotations: torch.Tensor,
    translations: torch.Tensor,
    frac_pos: torch.Tensor,
) -> torch.Tensor:
    """Build atom mapping for each symmetry operation.

    For each (R, t), maps atom i to atom j where R @ frac_i + t ≈ frac_j (mod 1).

    Returns:
        Symmetry mapping tensor, shape (n_ops, n_atoms).
    """
    n_ops = rotations.shape[0]
    n_atoms = frac_pos.shape[0]

    if n_atoms <= _SYMM_MAP_CHUNK_THRESHOLD:
        # Vectorized: allocates (n_ops, n_atoms, n_atoms, 3) — fast for small systems
        new_pos = torch.einsum("oij,nj->oni", rotations, frac_pos) + translations[:, None]
        delta = frac_pos[None, None] - new_pos[:, :, None]
        delta -= delta.round()
        return torch.argmin(torch.linalg.norm(delta, dim=-1), dim=-1).long()

    # Per-op loop: allocates only (n_atoms, n_atoms, 3) at a time
    result = torch.empty(n_ops, n_atoms, dtype=torch.long, device=frac_pos.device)
    for op_idx in range(n_ops):
        new_pos_op = frac_pos @ rotations[op_idx].T + translations[op_idx]
        delta = frac_pos[None, :, :] - new_pos_op[:, None, :]
        delta -= delta.round()
        result[op_idx] = torch.argmin(torch.linalg.norm(delta, dim=-1), dim=-1)
    return result


def prep_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get symmetry rotations and atom mappings for a structure.

    Returns:
        (rotations, symm_map) with shapes (n_ops, 3, 3) and (n_ops, n_atoms).
    """
    frac_pos = positions @ torch.linalg.inv(cell)
    dataset = _moyo_dataset(cell, frac_pos, atomic_numbers, symprec)
    rotations, translations = _extract_symmetry_ops(dataset, cell.dtype, cell.device)
    return rotations, build_symmetry_map(rotations, translations, frac_pos)


def _refine_symmetry_impl(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core refinement returning all intermediate data for reuse.

    Returns:
        (refined_cell, refined_positions, rotations, translations)
    """
    dtype, device = cell.dtype, cell.device
    frac_pos = positions @ torch.linalg.inv(cell)
    dataset = _moyo_dataset(cell, frac_pos, atomic_numbers, symprec)
    rotations, translations = _extract_symmetry_ops(dataset, dtype, device)
    n_ops, n_atoms = rotations.shape[0], positions.shape[0]

    # Symmetrize cell metric: g_sym = avg(R^T @ g @ R), then polar decomposition
    metric = cell @ cell.T
    metric_sym = torch.einsum("nji,jk,nkl->il", rotations, metric, rotations) / n_ops

    def _mat_sqrt(mat: torch.Tensor) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(mat)
        return evecs @ torch.diag(evals.sqrt()) @ evecs.T

    new_cell = _mat_sqrt(metric_sym) @ torch.linalg.inv(_mat_sqrt(metric)) @ cell

    # Symmetrize positions via displacement averaging over symmetry orbits
    new_frac = positions @ torch.linalg.inv(new_cell)
    symm_map = build_symmetry_map(rotations, translations, new_frac)

    transformed = torch.einsum("oij,nj->oni", rotations, new_frac) + translations[:, None]
    disp = transformed - new_frac[symm_map]
    disp -= disp.round()  # wrap into [-0.5, 0.5]

    target = symm_map.reshape(-1).unsqueeze(-1).expand(-1, 3)
    accum = torch.zeros(n_atoms, 3, dtype=dtype, device=device)
    accum.scatter_add_(0, target, disp.reshape(-1, 3))

    new_positions = (new_frac + accum / n_ops) @ new_cell
    return new_cell, new_positions, rotations, translations


def refine_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize cell and positions according to the detected space group.

    Uses polar decomposition for the cell metric tensor and scatter-add
    averaging over symmetry orbits for atomic positions.

    Returns:
        (symmetrized_cell, symmetrized_positions) as row vectors.
    """
    new_cell, new_positions, _rotations, _translations = _refine_symmetry_impl(
        cell, positions, atomic_numbers, symprec
    )
    return new_cell, new_positions


def refine_and_prep_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine symmetry and get ops/mappings in a single moyopy call.

    Combines ``refine_symmetry`` and ``prep_symmetry`` to avoid redundant
    symmetry detection. Used by ``FixSymmetry.from_state``.

    Returns:
        (refined_cell, refined_positions, rotations, symm_map)
    """
    new_cell, new_positions, rotations, translations = _refine_symmetry_impl(
        cell, positions, atomic_numbers, symprec
    )
    # Build symm_map on the final refined fractional coordinates
    refined_frac = new_positions @ torch.linalg.inv(new_cell)
    symm_map = build_symmetry_map(rotations, translations, refined_frac)
    return new_cell, new_positions, rotations, symm_map


def symmetrize_rank1(
    lattice: torch.Tensor,
    vectors: torch.Tensor,
    rotations: torch.Tensor,
    symm_map: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize a rank-1 per-atom tensor (forces, velocities, displacements).

    Works in fractional coordinates internally. Returns symmetrized Cartesian tensor.
    """
    n_ops, n_atoms = rotations.shape[0], vectors.shape[0]
    scaled = vectors @ torch.linalg.inv(lattice)
    # Rotate each vector by each symmetry op: scaled @ R^T
    rotated = torch.einsum("ij,nkj->nik", scaled, rotations).reshape(-1, 3)
    # Scatter-add to target atoms and average
    target = symm_map.reshape(-1).unsqueeze(-1).expand(-1, 3)
    accum = torch.zeros(n_atoms, 3, dtype=vectors.dtype, device=vectors.device)
    accum.scatter_add_(0, target, rotated)
    return (accum / n_ops) @ lattice


def symmetrize_rank2(
    lattice: torch.Tensor,
    tensor: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize a rank-2 tensor (stress, strain) over all symmetry operations."""
    n_ops = rotations.shape[0]
    inv_lat = torch.linalg.inv(lattice)
    scaled = lattice @ tensor @ lattice.T
    sym_scaled = torch.einsum("nji,jk,nkl->il", rotations, scaled, rotations) / n_ops
    return inv_lat @ sym_scaled @ inv_lat.T
