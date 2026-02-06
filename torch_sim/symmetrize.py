"""Symmetry refinement utilities for crystal structures.

This module provides functions for refining and symmetrizing atomic structures
using moyopy (Python bindings for the moyo crystal symmetry library).

The main entry point is `refine_symmetry` which symmetrizes both the cell
and atomic positions according to the detected space group symmetry.

Note: Functions in this module operate on single (unbatched) systems.
The `n_ops` dimension refers to the number of symmetry operations
(rotations + translations) of the space group.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from moyopy import MoyoDataset

    from torch_sim.state import SimState


def _get_moyo_dataset(
    cell: torch.Tensor,
    scaled_positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1.0e-4,
) -> MoyoDataset:
    """Get symmetry dataset from moyopy.

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        scaled_positions: Fractional coordinates, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        symprec: Symmetry precision in units of cell basis vectors

    Returns:
        MoyoDataset with symmetry information
    """
    from moyopy import Cell, MoyoDataset

    cell_list = cell.detach().cpu().tolist()
    positions_list = scaled_positions.detach().cpu().tolist()
    numbers_list = atomic_numbers.detach().cpu().int().tolist()

    moyo_cell = Cell(basis=cell_list, positions=positions_list, numbers=numbers_list)
    return MoyoDataset(moyo_cell, symprec=symprec)


def get_symmetry_datasets(
    state: SimState,
    symprec: float = 1.0e-4,
) -> list[MoyoDataset]:
    """Get symmetry datasets for all systems in a SimState.

    Args:
        state: SimState containing one or more systems
        symprec: Symmetry precision for moyopy

    Returns:
        List of MoyoDataset objects, one per system in the state.
    """
    datasets = []

    for single_state in state.split():
        cell = single_state.row_vector_cell[0]
        positions = single_state.positions

        scaled_positions = _get_scaled_positions(positions, cell)

        dataset = _get_moyo_dataset(
            cell=cell,
            scaled_positions=scaled_positions,
            atomic_numbers=single_state.atomic_numbers,
            symprec=symprec,
        )
        datasets.append(dataset)

    return datasets


def _get_scaled_positions(
    positions: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Convert Cartesian positions to fractional coordinates (unbatched).

    See also ``transforms.get_fractional_coordinates`` for the batched version.

    Args:
        positions: Cartesian positions, shape (n_atoms, 3)
        cell: Unit cell as row vectors, shape (3, 3)

    Returns:
        Fractional coordinates, shape (n_atoms, 3)
    """
    return positions @ torch.linalg.inv(cell)


def refine_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
    *,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine symmetry of a structure.

    Symmetrizes both cell vectors and atomic positions by averaging
    over the detected symmetry operations using polar decomposition
    for the cell metric and scatter-add averaging for positions.

    The refinement process:
    1. Detect symmetry operations of the input structure
    2. Symmetrize the cell metric tensor (preserving cell orientation)
    3. Symmetrize atomic positions by averaging over symmetry orbits

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        positions: Cartesian positions, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        symprec: Symmetry precision for moyopy
        verbose: If True, log symmetry information before and after

    Returns:
        Tuple of (symmetrized_cell, symmetrized_positions):
        - symmetrized_cell: Symmetrized cell as row vectors, shape (3, 3)
        - symmetrized_positions: Symmetrized Cartesian positions, shape (n_atoms, 3)
    """
    device = cell.device
    dtype = cell.dtype

    # Step 1: Detect symmetry
    scaled_positions = _get_scaled_positions(positions, cell)
    dataset = _get_moyo_dataset(cell, scaled_positions, atomic_numbers, symprec)

    if verbose:
        logger.info(
            "symmetrize: prec %s got space group number %s",
            symprec,
            dataset.number,
        )

    rotations = torch.as_tensor(
        dataset.operations.rotations, dtype=dtype, device=device
    ).round()
    translations = torch.as_tensor(
        dataset.operations.translations, dtype=dtype, device=device
    )
    n_ops = rotations.shape[0]

    # Step 2: Symmetrize cell via metric tensor + polar decomposition
    # Row-vector metric: g[i,j] = a_i · a_j = (cell @ cell.T)[i,j]
    # Symmetry invariance: R.T @ g @ R = g for all rotations R
    metric = cell @ cell.T
    metric_sym = torch.einsum("nji,jk,nkl->il", rotations, metric, rotations) / n_ops

    # Left polar decomposition: cell = P @ V where P = sqrt(metric)
    # Keep same orientation V but with symmetrized metric P_sym
    sqrt_metric = _matrix_sqrt(metric)
    sqrt_metric_sym = _matrix_sqrt(metric_sym)
    new_cell = sqrt_metric_sym @ torch.linalg.inv(sqrt_metric) @ cell

    # Step 3: Symmetrize positions by averaging displacements over symmetry orbits
    # Recompute fractional coords in the symmetrized cell
    new_frac = positions @ torch.linalg.inv(new_cell)
    symm_map = build_symmetry_map(rotations, translations, new_frac)

    # For each op, transform fractional positions: R @ frac + t
    new_frac_all = (
        torch.einsum("oij,nj->oni", rotations, new_frac) + translations[:, None, :]
    )  # (n_ops, n_atoms, 3)
    # Compute displacement from target atom's current position, wrapped for periodicity
    n_atoms = positions.shape[0]
    target_frac = new_frac[symm_map]  # (n_ops, n_atoms, 3)
    displacement = new_frac_all - target_frac
    displacement -= displacement.round()  # wrap into [-0.5, 0.5]

    # Scatter-add wrapped displacements to target atoms and average
    target = symm_map.reshape(-1).unsqueeze(-1).expand(-1, 3)
    accum = torch.zeros(n_atoms, 3, dtype=dtype, device=device)
    accum.scatter_add_(0, target, displacement.reshape(-1, 3))
    sym_frac = new_frac + accum / n_ops

    new_positions = sym_frac @ new_cell

    if verbose:
        final_scaled = _get_scaled_positions(new_positions, new_cell)
        final_dataset = _get_moyo_dataset(new_cell, final_scaled, atomic_numbers, 1e-4)
        logger.info(
            "symmetrize: prec 1e-4 got space group number %s",
            final_dataset.number,
        )

    return new_cell, new_positions


def _matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """Compute matrix square root of a symmetric positive-definite matrix.

    Uses eigendecomposition: sqrt(A) = Q @ diag(sqrt(eigenvalues)) @ Q.T

    Args:
        mat: Symmetric positive-definite matrix, shape (3, 3)

    Returns:
        Matrix square root, shape (3, 3)
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    return eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T


def _prep_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1.0e-4,
    *,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare structure for symmetry-preserving minimization.

    Determines the symmetry operations (rotations in fractional coordinates)
    and atom mappings needed for symmetry-constrained optimization.

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        positions: Cartesian positions, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        symprec: Symmetry precision for moyopy
        verbose: If True, log symmetry information

    Returns:
        Tuple of (rotations, symm_map):
        - rotations: Rotation matrices in fractional coords, shape (n_ops, 3, 3)
        - symm_map: Atom mapping tensor, shape (n_ops, n_atoms)
    """
    device = cell.device
    dtype = cell.dtype

    scaled_positions = _get_scaled_positions(positions, cell)
    dataset = _get_moyo_dataset(cell, scaled_positions, atomic_numbers, symprec)

    if verbose:
        logger.info(
            "symmetrize: prec %s got space group number %s, n_ops %d",
            symprec,
            dataset.number,
            len(dataset.operations),
        )

    rotations = torch.as_tensor(
        dataset.operations.rotations, dtype=dtype, device=device
    ).round()
    translations = torch.as_tensor(
        dataset.operations.translations, dtype=dtype, device=device
    )

    # Build symmetry mapping
    symm_map = build_symmetry_map(rotations, translations, scaled_positions)

    return rotations, symm_map


def build_symmetry_map(
    rotations: torch.Tensor,
    translations: torch.Tensor,
    scaled_positions: torch.Tensor,
) -> torch.Tensor:
    """Build symmetry atom mapping for each symmetry operation.

    For each symmetry operation (R, t), determines which atom each atom
    maps to: atom i → atom j where R @ frac_i + t ≈ frac_j (mod 1).

    Args:
        rotations: Rotation matrices in fractional coords, shape (n_ops, 3, 3)
        translations: Translation vectors in fractional coords, shape (n_ops, 3)
        scaled_positions: Fractional coordinates, shape (n_atoms, 3)

    Returns:
        Symmetry mapping tensor, shape (n_ops, n_atoms)
    """
    # Transform all atoms by all symmetry operations at once
    # new_pos: (n_ops, n_atoms, 3)
    new_pos = (
        torch.einsum("oij,nj->oni", rotations, scaled_positions)
        + translations[:, None, :]
    )

    # Compute wrapped deltas to account for periodicity
    # delta: (n_ops, n_atoms, n_atoms, 3)
    delta = scaled_positions[None, None, :, :] - new_pos[:, :, None, :]
    delta -= delta.round()  # wrap into [-0.5, 0.5]

    # Distances to all candidate atoms, then choose nearest
    distances = torch.linalg.norm(delta, dim=-1)  # (n_ops, n_atoms, n_atoms)
    return torch.argmin(distances, dim=-1).to(dtype=torch.long)  # (n_ops, n_atoms)


def symmetrize_rank1(
    lattice: torch.Tensor,
    forces: torch.Tensor,
    rotations: torch.Tensor,
    symm_map: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize rank-1 tensor (forces, velocities, etc).

    Averages the tensor over all symmetry operations, respecting atom
    permutations. Works in fractional coordinates internally.

    Args:
        lattice: Cell vectors as row vectors, shape (3, 3)
        forces: Forces array, shape (n_atoms, 3)
        rotations: Rotation matrices in fractional coords, shape (n_ops, 3, 3)
        symm_map: Atom mapping for each symmetry operation, shape (n_ops, n_atoms)

    Returns:
        Symmetrized forces, shape (n_atoms, 3)
    """
    n_ops = rotations.shape[0]
    n_atoms = forces.shape[0]

    # Transform to scaled (fractional) coordinates: (n_atoms, 3)
    scaled_forces = forces @ lattice.inverse()

    # Apply all rotations at once: (n_ops, n_atoms, 3)
    # rotations: (n_ops, 3, 3), scaled_forces: (n_atoms, 3)
    # For each op: scaled_forces @ rot.T (rotate the vectors)
    # Note: we use rotations.mT to get the transpose of each rotation matrix
    transformed_forces = torch.einsum("ij,nkj->nik", scaled_forces, rotations)

    # Flatten for scatter: (n_ops * n_atoms, 3)
    transformed_flat = transformed_forces.reshape(-1, 3)

    # Flatten symm_map to get target indices: (n_ops * n_atoms,)
    target_indices = symm_map.reshape(-1)

    # Expand target indices to match 3D coordinates: (n_ops * n_atoms, 3)
    target_indices_expanded = target_indices.unsqueeze(-1).expand(-1, 3)

    # Scatter add to accumulate forces at target atoms
    # Result shape: (n_atoms, 3)
    accumulated = torch.zeros(n_atoms, 3, dtype=forces.dtype, device=forces.device)
    accumulated.scatter_add_(0, target_indices_expanded, transformed_flat)

    # Average over symmetry operations
    symmetrized_scaled = accumulated / n_ops

    # Transform back to Cartesian
    return symmetrized_scaled @ lattice


def symmetrize_rank2(
    lattice: torch.Tensor,
    stress: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize rank-2 tensor (stress, strain, etc).

    Averages the tensor over all symmetry operations in scaled coordinates.

    Args:
        lattice: Cell vectors as row vectors, shape (3, 3)
        stress: Stress tensor, shape (3, 3)
        rotations: Rotation matrices in fractional coords, shape (n_ops, 3, 3)

    Returns:
        Symmetrized stress tensor, shape (3, 3)
    """
    n_ops = rotations.shape[0]
    inv_lattice = lattice.inverse()

    # Scale stress: lattice @ stress @ lattice.T
    scaled_stress = lattice @ stress @ lattice.T

    # Symmetrize in scaled coordinates using vectorized operations
    # r.T @ scaled_stress @ r for all rotations at once
    # For r.T @ A @ r: result[i,l] = sum_j,k r[j,i] * A[j,k] * r[k,l]
    # With batched rotations: einsum "nji,jk,nkl->il"
    symmetrized_scaled_stress = (
        torch.einsum("nji,jk,nkl->il", rotations, scaled_stress, rotations) / n_ops
    )

    # Transform back: inv_lattice @ symmetrized_scaled_stress @ inv_lattice.T
    return inv_lattice @ symmetrized_scaled_stress @ inv_lattice.T
