"""Symmetry refinement utilities for crystal structures.

This module provides functions for refining and symmetrizing atomic structures
using spglib. It is adapted from ASE's spacegroup.symmetrize module but
reimplemented to work with torch tensors directly.

The main entry point is `refine_symmetry` which symmetrizes both the cell
and atomic positions according to the detected space group symmetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from spglib import SpglibDataset

    from torch_sim.state import SimState


__all__ = [
    "refine_symmetry",
    "build_symmetry_map",
    "symmetrize_rank1",
    "symmetrize_rank2",
    "get_symmetry_datasets",
]


def _get_symmetry_dataset(
    cell: torch.Tensor,
    scaled_positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1.0e-6,
) -> SpglibDataset | None:
    """Get symmetry dataset from spglib.

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        scaled_positions: Fractional coordinates, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        symprec: Symmetry precision

    Returns:
        Symmetry dataset with attribute access
    """
    import spglib

    # Convert tensors to numpy for spglib
    cell_np = cell.detach().cpu().numpy()
    positions_np = scaled_positions.detach().cpu().numpy()
    numbers_np = atomic_numbers.detach().cpu().numpy()

    cell_tuple = (cell_np, positions_np, numbers_np)
    return spglib.get_symmetry_dataset(cell_tuple, symprec=symprec)


def get_symmetry_datasets(
    state: SimState,
    symprec: float = 1.0e-6,
) -> list[SpglibDataset | None]:
    """Get symmetry datasets for all systems in a SimState.

    Args:
        state: SimState containing one or more systems
        symprec: Symmetry precision for spglib

    Returns:
        List of spglib symmetry datasets, one per system in the state.
        Returns None for systems where symmetry detection fails.
    """
    datasets = []

    for single_state in state.split():
        cell = single_state.row_vector_cell[0]
        positions = single_state.positions

        # Compute scaled (fractional) positions for this system
        scaled_positions = _get_scaled_positions(positions, cell)

        dataset = _get_symmetry_dataset(
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
    """Convert Cartesian positions to fractional coordinates.

    Args:
        positions: Cartesian positions, shape (n_atoms, 3)
        cell: Unit cell as row vectors, shape (3, 3)

    Returns:
        Fractional coordinates, shape (n_atoms, 3)
    """
    inv_cell = torch.linalg.inv(cell)
    return positions @ inv_cell


def _symmetrize_cell(
    cell: torch.Tensor,
    dataset: SpglibDataset,
) -> torch.Tensor:
    """Symmetrize the cell based on the symmetry dataset.

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        dataset: spglib symmetry dataset

    Returns:
        Symmetrized cell as row vectors, shape (3, 3)
    """
    device = cell.device
    dtype = cell.dtype

    # Get standardized cell and apply transformations
    std_cell = torch.as_tensor(dataset.std_lattice, dtype=dtype, device=device)
    trans_matrix = torch.as_tensor(
        dataset.transformation_matrix, dtype=dtype, device=device
    )
    rot_matrix = torch.as_tensor(
        dataset.std_rotation_matrix, dtype=dtype, device=device
    )

    trans_std_cell = trans_matrix.T @ std_cell
    rot_trans_std_cell = trans_std_cell @ rot_matrix

    return rot_trans_std_cell


def _symmetrize_positions(
    positions: torch.Tensor,
    dataset: SpglibDataset,
    primitive_cell: tuple,
) -> torch.Tensor:
    """Symmetrize atomic positions.

    Args:
        positions: Cartesian positions, shape (n_atoms, 3)
        dataset: spglib symmetry dataset
        primitive_cell: Result from spglib.find_primitive (cell, positions, numbers)

    Returns:
        Symmetrized Cartesian positions, shape (n_atoms, 3)
    """
    device = positions.device
    dtype = positions.dtype

    prim_cell_np, _prim_scaled_pos, _prim_types = primitive_cell
    prim_cell = torch.as_tensor(prim_cell_np, dtype=dtype, device=device)

    # Calculate offset between standard cell and actual cell
    std_cell = torch.as_tensor(dataset.std_lattice, dtype=dtype, device=device)
    rot_matrix = torch.as_tensor(
        dataset.std_rotation_matrix, dtype=dtype, device=device
    )
    std_positions = torch.as_tensor(dataset.std_positions, dtype=dtype, device=device)

    rot_std_cell = std_cell @ rot_matrix
    rot_std_pos = std_positions @ rot_std_cell

    # Get mapping indices
    mapping_to_primitive = list(dataset.mapping_to_primitive)
    std_mapping_to_primitive = list(dataset.std_mapping_to_primitive)

    dp0 = positions[mapping_to_primitive.index(0)] - rot_std_pos[
        std_mapping_to_primitive.index(0)
    ]

    # Create aligned set of standard cell positions
    rot_prim_cell = prim_cell @ rot_matrix
    inv_rot_prim_cell = torch.linalg.inv(rot_prim_cell)
    aligned_std_pos = rot_std_pos + dp0

    # Find ideal positions
    new_positions = positions.clone()
    n_atoms = positions.shape[0]

    for i_at in range(n_atoms):
        std_i_at = std_mapping_to_primitive.index(mapping_to_primitive[i_at])
        dp = aligned_std_pos[std_i_at] - positions[i_at]
        dp_s = dp @ inv_rot_prim_cell
        new_positions[i_at] = aligned_std_pos[std_i_at] - torch.round(dp_s) @ rot_prim_cell

    return new_positions


def refine_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 0.01,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine symmetry of a structure.

    This function symmetrizes both the cell and atomic positions according
    to the detected space group symmetry.

    The refinement process:
    1. Detect symmetry of the input structure
    2. Symmetrize the cell vectors to match the ideal lattice
    3. Symmetrize atomic positions to ideal Wyckoff positions

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        positions: Cartesian positions, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        symprec: Symmetry precision for spglib
        verbose: If True, print symmetry information before and after

    Returns:
        Tuple of (symmetrized_cell, symmetrized_positions):
        - symmetrized_cell: Symmetrized cell as row vectors, shape (3, 3)
        - symmetrized_positions: Symmetrized Cartesian positions, shape (n_atoms, 3)
    """
    import spglib

    # Step 1: Check and symmetrize cell
    scaled_positions = _get_scaled_positions(positions, cell)
    dataset = _get_symmetry_dataset(cell, scaled_positions, atomic_numbers, symprec)

    if dataset is None:
        raise RuntimeError("spglib could not determine symmetry for structure")

    if verbose:
        print(
            f"symmetrize: prec {symprec} got symmetry group number {dataset.number}, "
            f"international (Hermann-Mauguin) {dataset.international}, "
            f"Hall {dataset.hall}"
        )

    new_cell = _symmetrize_cell(cell, dataset)

    # Scale positions to new cell
    new_positions = scaled_positions @ new_cell

    # Step 2: Check and symmetrize positions with the new cell
    new_scaled_positions = _get_scaled_positions(new_positions, new_cell)
    dataset = _get_symmetry_dataset(
        new_cell, new_scaled_positions, atomic_numbers, symprec
    )

    if dataset is None:
        raise RuntimeError("spglib could not determine symmetry after cell refinement")

    # Find primitive cell
    cell_np = new_cell.detach().cpu().numpy()
    positions_np = new_scaled_positions.detach().cpu().numpy()
    numbers_np = atomic_numbers.detach().cpu().numpy()

    primitive_result = spglib.find_primitive(
        (cell_np, positions_np, numbers_np), symprec=symprec
    )
    if primitive_result is None:
        raise RuntimeError("spglib could not find primitive cell")

    new_positions = _symmetrize_positions(
        new_positions, dataset, primitive_result
    )

    # Final check
    if verbose:
        final_scaled = _get_scaled_positions(new_positions, new_cell)
        final_dataset = _get_symmetry_dataset(
            new_cell, final_scaled, atomic_numbers, 1e-4
        )
        if final_dataset is not None:
            print(
                f"symmetrize: prec 1e-4 got symmetry group number "
                f"{final_dataset.number}, "
                f"international (Hermann-Mauguin) {final_dataset.international}, "
                f"Hall {final_dataset.hall}"
            )

    return new_cell, new_positions


def _prep_symmetry(
    cell: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    symprec: float = 1.0e-6,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare structure for symmetry-preserving minimization.

    This function determines the symmetry operations and atom mappings
    needed for symmetry-constrained optimization.

    Args:
        cell: Unit cell as row vectors, shape (3, 3)
        positions: Cartesian positions, shape (n_atoms, 3)
        atomic_numbers: Atomic numbers, shape (n_atoms,)
        symprec: Symmetry precision for spglib
        verbose: If True, print symmetry information

    Returns:
        Tuple of (rotations, symm_map):
        - rotations: Rotation matrices, shape (n_ops, 3, 3)
        - symm_map: Atom mapping tensor, shape (n_ops, n_atoms)
    """
    device = cell.device
    dtype = cell.dtype

    scaled_positions = _get_scaled_positions(positions, cell)
    dataset = _get_symmetry_dataset(cell, scaled_positions, atomic_numbers, symprec)

    if dataset is None:
        raise RuntimeError("spglib could not determine symmetry for structure")

    if verbose:
        print(
            f"symmetrize: prec {symprec} got symmetry group number {dataset.number}, "
            f"international (Hermann-Mauguin) {dataset.international}, "
            f"Hall {dataset.hall}"
        )

    rotations = torch.as_tensor(dataset.rotations.copy(), dtype=dtype, device=device)
    translations = torch.as_tensor(
        dataset.translations.copy(), dtype=dtype, device=device
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

    For each symmetry operation, determines which atom each atom maps to.

    Args:
        rotations: Rotation matrices, shape (n_ops, 3, 3)
        translations: Translation vectors, shape (n_ops, 3)
        scaled_positions: Fractional coordinates, shape (n_atoms, 3)

    Returns:
        Symmetry mapping tensor, shape (n_ops, n_atoms)
    """
    # Transform all atoms by all symmetry operations at once
    # new_pos: (n_ops, n_atoms, 3)
    new_pos = torch.einsum("oij,nj->oni", rotations, scaled_positions) + translations[:, None, :]

    # Compute wrapped deltas to account for periodicity
    # delta: (n_ops, n_atoms, n_atoms, 3)
    delta = scaled_positions[None, None, :, :] - new_pos[:, :, None, :]
    delta -= delta.round()  # wrap into [-0.5, 0.5]

    # Distances to all candidate atoms, then choose nearest
    distances = torch.linalg.norm(delta, dim=-1)  # (n_ops, n_atoms, n_atoms)
    symm_map = torch.argmin(distances, dim=-1).to(dtype=torch.long)  # (n_ops, n_atoms)

    return symm_map


def symmetrize_rank1(
    lattice: torch.Tensor,
    forces: torch.Tensor,
    rotations: torch.Tensor,
    symm_map: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize rank-1 tensor (forces, velocities, etc).

    Args:
        lattice: Cell vectors as row vectors, shape (3, 3)
        forces: Forces array, shape (n_atoms, 3)
        rotations: Rotation matrices, shape (n_ops, 3, 3)
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
    symmetrized_forces = symmetrized_scaled @ lattice

    return symmetrized_forces


def symmetrize_rank2(
    lattice: torch.Tensor,
    stress: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize rank-2 tensor (stress, strain, etc).

    Args:
        lattice: Cell vectors as row vectors, shape (3, 3)
        stress: Stress tensor, shape (3, 3)
        rotations: Rotation matrices, shape (n_ops, 3, 3)

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
    symmetrized_scaled_stress = torch.einsum(
        "nji,jk,nkl->il", rotations, scaled_stress, rotations
    ) / n_ops

    # Transform back: inv_lattice @ symmetrized_scaled_stress @ inv_lattice.T
    return inv_lattice @ symmetrized_scaled_stress @ inv_lattice.T
