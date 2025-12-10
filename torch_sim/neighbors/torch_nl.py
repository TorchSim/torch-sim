"""Batched neighbor list implementations for multiple systems.

This module provides neighbor list calculations optimized for batched processing
of multiple atomic systems simultaneously. These implementations are designed for
use with multiple systems that may have different numbers of atoms.

The API follows the batched convention used in MACE and other models:
- Requires system_idx to identify which system each atom belongs to
- Returns (mapping, system_mapping, shifts_idx) tuples
- mapping: [2, n_neighbors] - pairs of atom indices
- system_mapping: [n_neighbors] - which system each neighbor pair belongs to
- shifts_idx: [n_neighbors, 3] - periodic shift indices
"""

import torch

from torch_sim import transforms


def strict_nl(
    cutoff: float,
    positions: torch.Tensor,
    cell: torch.Tensor,
    mapping: torch.Tensor,
    system_mapping: torch.Tensor,
    shifts_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a strict cutoff to the neighbor list defined in the mapping.

    This function filters the neighbor list based on a specified cutoff distance.
    It computes the squared distances between pairs of positions and retains only
    those pairs that are within the cutoff distance. The function also accounts
    for periodic boundary conditions by applying cell shifts when necessary.

    Args:
        cutoff (float):
            The maximum distance for considering two atoms as neighbors. This value
            is used to filter the neighbor pairs based on their distances.
        positions (torch.Tensor): A tensor of shape (n_atoms, 3) representing
            the positions of the atoms.
        cell (torch.Tensor): Unit cell vectors according to the row vector convention,
            i.e. `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        mapping (torch.Tensor):
            A tensor of shape (2, n_pairs) that specifies pairs of indices in `positions`
            for which to compute distances.
        system_mapping (torch.Tensor):
            A tensor that maps the shifts to the corresponding cells, used in conjunction
            with `shifts_idx` to compute the correct periodic shifts.
        shifts_idx (torch.Tensor):
            A tensor of shape (n_shifts, 3) representing the indices for shifts to apply
            to the distances for periodic boundary conditions.

    Returns:
        tuple:
            A tuple containing:
                - mapping (torch.Tensor): A filtered tensor of shape (2, n_filtered_pairs)
                  with pairs of indices that are within the cutoff distance.
                - mapping_system (torch.Tensor): A tensor of shape (n_filtered_pairs,)
                  that maps the filtered pairs to their corresponding systems.
                - shifts_idx (torch.Tensor): A tensor of shape (n_filtered_pairs, 3)
                  containing the periodic shift indices for the filtered pairs.

    Notes:
        - The function computes the squared distances to avoid the computational cost
          of taking square roots, which is unnecessary for comparison.
        - If no cell shifts are needed (i.e., for non-periodic systems), the function
          directly computes the squared distances between the positions.

    References:
        - https://github.com/felixmusil/torch_nl
    """
    cell_shifts = transforms.compute_cell_shifts(cell, shifts_idx, system_mapping)
    if cell_shifts is None:
        d2 = (positions[mapping[0]] - positions[mapping[1]]).square().sum(dim=1)
    else:
        d2 = (
            (positions[mapping[0]] - positions[mapping[1]] - cell_shifts)
            .square()
            .sum(dim=1)
        )

    mask = d2 < cutoff * cutoff
    mapping = mapping[:, mask]
    mapping_system = system_mapping[mask]
    shifts_idx = shifts_idx[mask]
    return mapping, mapping_system, shifts_idx


@torch.jit.script
def torch_nl_n2(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    system_idx: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a set of atomic structures using a
    naive neighbor search before applying a strict `cutoff`.

    The atomic positions `pos` should be wrapped inside their respective unit cells.

    This implementation uses a naive O(N²) neighbor search which can be slow for
    large systems but is simple and works reliably for small to medium systems.

    Args:
        positions (torch.Tensor [n_atom, 3]): A tensor containing the positions
            of atoms wrapped inside their respective unit cells.
        cell (torch.Tensor [3*n_structure, 3]): Unit cell vectors according to
            the row vector convention, i.e. `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc (torch.Tensor [n_structure, 3] bool):
            A tensor indicating the periodic boundary conditions to apply.
            Partial PBC are not supported yet.
        cutoff (torch.Tensor):
            The cutoff radius used for the neighbor search.
        system_idx (torch.Tensor [n_atom,] torch.long):
            A tensor containing the index of the structure to which each atom belongs.
        self_interaction (bool, optional):
            A flag to indicate whether to keep the center atoms as their own neighbors.
            Default is False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mapping (torch.Tensor [2, n_neighbors]):
                A tensor containing the indices of the neighbor list for the given
                positions array. `mapping[0]` corresponds to the central atom indices,
                and `mapping[1]` corresponds to the neighbor atom indices.
            system_mapping (torch.Tensor [n_neighbors]):
                A tensor mapping the neighbor atoms to their respective structures.
            shifts_idx (torch.Tensor [n_neighbors, 3]):
                A tensor containing the cell shift indices used to reconstruct the
                neighbor atom positions.

    Example:
        >>> # Create a batched system with 2 structures
        >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        >>> cell = torch.eye(3).repeat(2, 1) * 10.0  # Two cells
        >>> pbc = torch.tensor([[True, True, True], [True, True, True]])
        >>> cutoff = torch.tensor(2.0)
        >>> # First 2 atoms in system 0, last in system 1
        >>> system_idx = torch.tensor([0, 0, 1])
        >>> mapping, sys_map, shifts = torch_nl_n2(
        ...     positions, cell, pbc, cutoff, system_idx
        ... )

    References:
        - https://github.com/felixmusil/torch_nl
    """
    n_atoms = torch.bincount(system_idx)
    mapping, system_mapping, shifts_idx = transforms.build_naive_neighborhood(
        positions, cell, pbc, cutoff.item(), n_atoms, self_interaction
    )
    mapping, mapping_system, shifts_idx = strict_nl(
        cutoff.item(), positions, cell, mapping, system_mapping, shifts_idx
    )
    return mapping, mapping_system, shifts_idx


@torch.jit.script
def torch_nl_linked_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    system_idx: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002 (*, not compatible with torch.jit.script)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a set of atomic structures using the linked
    cell algorithm before applying a strict `cutoff`.

    The atoms positions `pos` should be wrapped inside their respective unit cells.

    This is the recommended default for batched neighbor list calculations as it
    provides good performance for systems of various sizes using the linked cell
    algorithm which has O(N) complexity.

    Args:
        positions (torch.Tensor [n_atom, 3]):
            A tensor containing the positions of atoms wrapped inside
            their respective unit cells.
        cell (torch.Tensor [3*n_systems, 3]): Unit cell vectors according to
            the row vector convention, i.e. `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc (torch.Tensor [n_systems, 3] bool):
            A tensor indicating the periodic boundary conditions to apply.
            Partial PBC are not supported yet.
        cutoff (torch.Tensor):
            The cutoff radius used for the neighbor search.
        system_idx (torch.Tensor [n_atom,] torch.long):
            A tensor containing the index of the structure to which each atom belongs.
        self_interaction (bool, optional):
            A flag to indicate whether to keep the center atoms as their own neighbors.
            Default is False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
                - mapping (torch.Tensor [2, n_neighbors]):
                    A tensor containing the indices of the neighbor list for the given
                    positions array. `mapping[0]` corresponds to the central atom
                    indices, and `mapping[1]` corresponds to the neighbor atom indices.
                - system_mapping (torch.Tensor [n_neighbors]):
                    A tensor mapping the neighbor atoms to their respective structures.
                - shifts_idx (torch.Tensor [n_neighbors, 3]):
                    A tensor containing the cell shift indices used to reconstruct the
                    neighbor atom positions.

    Example:
        >>> # Create a batched system with 2 structures
        >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        >>> cell = torch.eye(3).repeat(2, 1) * 10.0  # Two cells
        >>> pbc = torch.tensor([[True, True, True], [True, True, True]])
        >>> cutoff = torch.tensor(2.0)
        >>> # First 2 atoms in system 0, last in system 1
        >>> system_idx = torch.tensor([0, 0, 1])
        >>> mapping, sys_map, shifts = torch_nl_linked_cell(
        ...     positions, cell, pbc, cutoff, system_idx
        ... )

    References:
        - https://github.com/felixmusil/torch_nl
    """
    n_atoms = torch.bincount(system_idx)
    mapping, system_mapping, shifts_idx = transforms.build_linked_cell_neighborhood(
        positions, cell, pbc, cutoff.item(), n_atoms, self_interaction
    )

    mapping, mapping_system, shifts_idx = strict_nl(
        cutoff.item(), positions, cell, mapping, system_mapping, shifts_idx
    )
    return mapping, mapping_system, shifts_idx
