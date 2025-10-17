"""Coordinate transformations and periodic boundary conditions.

This module provides functions for handling coordinate transformations and periodic
boundary conditions in molecular simulations, including matrix inversions and
general PBC wrapping.
"""

from collections.abc import Callable, Iterable
from functools import wraps

import torch
from torch.types import _dtype


def get_fractional_coordinates(
    positions: torch.Tensor, cell: torch.Tensor
) -> torch.Tensor:
    """Convert Cartesian coordinates to fractional coordinates.

    This function transforms atomic positions from Cartesian coordinates to fractional
    coordinates using the provided unit cell matrix. The fractional coordinates represent
    the position of each atom relative to the unit cell vectors.

    Args:
        positions (torch.Tensor): Atomic positions in Cartesian coordinates.
            Shape: [..., 3] where ... represents optional system dimensions.
        cell (torch.Tensor): Unit cell matrix with lattice vectors as rows.
            Shape: [..., 3, 3] where ... matches positions' system dimensions.

    Returns:
        torch.Tensor: Atomic positions in fractional coordinates with same shape as input
            positions. Each component will be in range [0,1) for positions
            inside the cell.

    Example:
        >>> pos = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
        >>> cell = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        >>> frac = get_fractional_coordinates(pos, cell)
        >>> print(frac)
        tensor([[0.25, 0.25, 0.25],
                [0.50, 0.00, 0.00]])
    """
    if cell.ndim == 3:  # Handle batched cell tensors
        # For batched cells, we need to determine if this is:
        # 1. A single system (n_systems=1) - can be squeezed and handled normally
        # 2. Multiple systems - need proper system handling

        if cell.shape[0] == 1:
            # Single system case - squeeze and use the 2D implementation
            cell_2d = cell.squeeze(0)  # Remove batch dimension
            return torch.linalg.solve(cell_2d.mT, positions.mT).mT
        # Multiple systems case - this would require system indices to know which
        # atoms belong to which system. For now, this is not implemented.
        raise NotImplementedError(
            f"Multiple system cell tensors with shape {cell.shape} are not yet "
            "supported in get_fractional_coordinates. For multiple system systems, "
            "you need to provide system indices to determine which atoms belong to "
            "which system. For single system systems, consider squeezing the batch "
            "dimension or using individual calls per system."
        )

    # Original case for 2D cell matrix
    return torch.linalg.solve(cell.mT, positions.mT).mT


def inverse_box(box: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of an affine transformation.

    Computes the multiplicative inverse of a transformation, handling three cases:
    1. Scalars: returns reciprocal (1/x)
    2. Vectors: returns element-wise reciprocal
    3. Matrices: returns matrix inverse using torch.linalg.inv

    Args:
        box (torch.Tensor): A PyTorch tensor representing either:
            - scalar: A single number (0-dim tensor or 1-element tensor)
            - vector: 1D tensor of scaling factors
            - matrix: 2D tensor representing linear transformation

    Returns:
        torch.Tensor: The inverse of the input transformation with the
            same shape as input:
            - scalar -> scalar: 1/x
            - vector -> vector: element-wise 1/x
            - matrix -> matrix: matrix inverse

    Raises:
        ValueError: If the input tensor has more than 2 dimensions.
        torch.linalg.LinAlgError: If matrix is singular (non-invertible).

    Examples:
        >>> # Scalar inverse
        >>> inverse_box(torch.tensor(2.0))
        tensor(0.5000)

        >>> # Vector inverse (element-wise)
        >>> inverse_box(torch.tensor([2.0, 4.0]))
        tensor([0.5000, 0.2500])

        >>> # Matrix inverse
        >>> mat = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        >>> inverse_box(mat)
        tensor([[ 1, -2],
                [ 0,  1]])
    """
    if (torch.is_tensor(box) and box.ndim == 0) or box.numel() == 1 or box.ndim == 1:
        return 1 / box
    if box.ndim == 2:
        return torch.linalg.inv(box)
    raise ValueError(f"Box must be either: a scalar, a vector, or a matrix. Found {box}.")


def pbc_wrap_general(
    positions: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """Apply periodic boundary conditions using lattice
        vector transformation method.

    This implementation follows the general matrix-based approach for
    periodic boundary conditions in arbitrary triclinic cells:
    1. Transform positions to fractional coordinates using B = A^(-1)
    2. Wrap fractional coordinates to [0,1) using modulo
    3. Transform back to real space using A

    Args:
        positions (torch.Tensor): Tensor of shape (..., d)
            containing particle positions in real space.
        lattice_vectors (torch.Tensor): Tensor of shape (d, d) containing
            lattice vectors as columns (A matrix in the equations).

    Returns:
        torch.Tensor: Wrapped positions in real space with same shape as input positions.
    """
    # Validate inputs
    if not torch.is_floating_point(positions) or not torch.is_floating_point(
        lattice_vectors
    ):
        raise TypeError("Positions and lattice vectors must be floating point tensors.")

    if lattice_vectors.ndim != 2 or lattice_vectors.shape[0] != lattice_vectors.shape[1]:
        raise ValueError("Lattice vectors must be a square matrix.")

    if positions.shape[-1] != lattice_vectors.shape[0]:
        raise ValueError("Position dimensionality must match lattice vectors.")

    # Transform to fractional coordinates: f = Br
    frac_coords = positions @ torch.linalg.inv(lattice_vectors).T

    # Wrap to reference cell [0,1) using modulo
    wrapped_frac = frac_coords % 1.0

    # Transform back to real space: r_row_wrapped = wrapped_f_row @ M_row
    return wrapped_frac @ lattice_vectors.T


def pbc_wrap_batched(
    positions: torch.Tensor, cell: torch.Tensor, system_idx: torch.Tensor
) -> torch.Tensor:
    """Apply periodic boundary conditions to batched systems.

    This function handles wrapping positions for multiple atomistic systems
    (systems) in one operation. It uses the system indices to determine which
    atoms belong to which system and applies the appropriate cell vectors.

    Args:
        positions (torch.Tensor): Tensor of shape (n_atoms, 3) containing
            particle positions in real space.
        cell (torch.Tensor): Tensor of shape (n_systems, 3, 3) containing
            lattice vectors as column vectors.
        system_idx (torch.Tensor): Tensor of shape (n_atoms,) containing system
            indices for each atom.

    Returns:
        torch.Tensor: Wrapped positions in real space with same shape as input positions.
    """
    # Validate inputs
    if not torch.is_floating_point(positions) or not torch.is_floating_point(cell):
        raise TypeError("Positions and lattice vectors must be floating point tensors.")

    if positions.shape[-1] != cell.shape[-1]:
        raise ValueError("Position dimensionality must match lattice vectors.")

    # Get unique system indices and counts
    uniq_systems = torch.unique(system_idx)
    n_systems = len(uniq_systems)

    if n_systems != cell.shape[0]:
        raise ValueError(
            f"Number of unique systems ({n_systems}) doesn't "
            f"match number of cells ({cell.shape[0]})"
        )

    # Efficient approach without explicit loops
    # Get the cell for each atom based on its system index
    B = torch.linalg.inv(cell)  # Shape: (n_systems, 3, 3)
    B_per_atom = B[system_idx]  # Shape: (n_atoms, 3, 3)

    # Transform to fractional coordinates: f = B·r
    # For each atom, multiply its position by its system's inverse cell matrix
    frac_coords = torch.bmm(B_per_atom, positions.unsqueeze(2)).squeeze(2)

    # Wrap to reference cell [0,1) using modulo
    wrapped_frac = frac_coords % 1.0

    # Transform back to real space: r = A·f
    # Get the cell for each atom based on its system index
    cell_per_atom = cell[system_idx]  # Shape: (n_atoms, 3, 3)

    # For each atom, multiply its wrapped fractional coords by its system's cell matrix
    return torch.bmm(cell_per_atom, wrapped_frac.unsqueeze(2)).squeeze(2)


def minimum_image_displacement(
    *,
    dr: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: bool = True,
) -> torch.Tensor:
    """Apply minimum image convention to displacement vectors.

    Args:
        dr (torch.Tensor): Displacement vectors [N, 3] or [N, N, 3].
        cell (Optional[torch.Tensor]): Unit cell matrix [3, 3].
        pbc (bool): Whether to apply periodic boundary conditions.

    Returns:
        torch.Tensor: Minimum image displacement vectors with same shape as input.
    """
    if cell is None or not pbc:
        return dr

    # Convert to fractional coordinates
    cell_inv = torch.linalg.inv(cell)
    dr_frac = torch.einsum("ij,...j->...i", cell_inv, dr)

    # Apply minimum image convention
    dr_frac -= torch.round(dr_frac)

    # Convert back to cartesian
    return torch.einsum("ij,...j->...i", cell, dr_frac)


def get_pair_displacements(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: bool = True,
    pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    shifts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute displacement vectors and distances between atom pairs.

    Args:
        positions (torch.Tensor): Atomic positions [N, 3].
        cell (Optional[torch.Tensor]): Unit cell matrix [3, 3].
        pbc (bool): Whether to apply periodic boundary conditions.
        pairs (Optional[Tuple[torch.Tensor, torch.Tensor]]):
            (i, j) indices for specific pairs to compute.
        shifts (Optional[torch.Tensor]): Shift vectors for periodic images [n_pairs, 3].

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Displacement vectors [n_pairs, 3].
            - Distances [n_pairs].
    """
    if pairs is None:
        # Create full distance matrix
        ri = positions.unsqueeze(0)  # [1, N, 3]
        rj = positions.unsqueeze(1)  # [N, 1, 3]
        dr = rj - ri  # [N, N, 3]

        if cell is not None and pbc:
            dr = minimum_image_displacement(dr=dr, cell=cell, pbc=pbc)

        # Calculate distances
        distances = torch.norm(dr, dim=-1)  # [N, N]

        # Mask out self-interactions
        mask = torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float("inf"))

        return dr, distances

    # Compute displacements for specific pairs
    i, j = pairs
    dr = positions[j] - positions[i]  # [n_pairs, 3]

    if cell is not None and pbc:
        if shifts is not None:
            # Apply provided shifts
            dr = dr + torch.einsum("ij,kj->ki", cell, shifts)
        else:
            dr = minimum_image_displacement(dr=dr, cell=cell, pbc=pbc)

    distances = torch.norm(dr, dim=-1)
    return dr, distances


def translate_pretty(
    fractional: torch.Tensor, pbc: torch.Tensor | list[bool]
) -> torch.Tensor:
    """ASE pretty translation in pytorch.

    Translates atoms such that fractional positions are minimized.

    Args:
        fractional (torch.Tensor): Tensor of shape (n_atoms, 3)
            containing fractional coordinates.
        pbc (Union[torch.Tensor, list[bool]]): Boolean tensor or list of
            shape (3,) indicating periodic boundary conditions.

    Returns:
        torch.Tensor: Translated fractional coordinates of shape (n_atoms, 3).

    Example:
        >>> coords = torch.tensor([[0.1, 1.2, -0.3], [0.7, 0.8, 0.9]])
        >>> pbc = [True, True, True]
        >>> translate_pretty(coords, pbc)
        tensor([[0.1000, 0.2000, 0.7000],
                [0.7000, 0.8000, 0.9000]])
    """
    if not isinstance(pbc, torch.Tensor):
        pbc = torch.tensor(pbc, dtype=torch.bool, device=fractional.device)

    fractional = fractional.clone()
    for dim in range(3):
        if not pbc[dim]:
            continue

        # Sort positions along this dimension
        indices = torch.argsort(fractional[:, dim])
        sp = fractional[indices, dim]

        # Calculate wrapped differences between consecutive positions
        widths = (torch.roll(sp, 1) - sp) % 1.0

        # Find the position that minimizes the differences and subtract it
        min_idx = torch.argmin(widths)
        fractional[:, dim] -= sp[min_idx]
        fractional[:, dim] %= 1.0

    return fractional


def wrap_positions(
    positions: torch.Tensor,
    cell: torch.Tensor,
    *,
    pbc: bool | list[bool] | torch.Tensor = True,
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    pretty_translation: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """ASE wrap_positions in pytorch.

    Wrap atomic positions to unit cell.

    Args:
        positions (torch.Tensor): Atomic positions [N, 3].
        cell (torch.Tensor): Unit cell matrix [3, 3].
        pbc (Union[bool, list[bool], torch.Tensor]): Whether to apply
            periodic boundary conditions.
        center (Tuple[float, float, float]): Center of the cell as
            (x,y,z) tuple, defaults to (0.5, 0.5, 0.5).
        pretty_translation (bool): Whether to minimize the spread of
            fractional coordinates.
        eps (float): Small number to handle edge cases in wrapping.

    Returns:
        torch.Tensor: Wrapped positions in Cartesian coordinates [N, 3].
    """
    device = positions.device

    # Convert center to tensor
    center_tensor = torch.tensor(center, dtype=positions.dtype, device=device)

    # Handle PBC input
    if isinstance(pbc, bool):
        pbc = [pbc] * 3
    if not isinstance(pbc, torch.Tensor):
        pbc = torch.tensor(pbc, dtype=torch.bool, device=device)

    # Calculate shift based on center
    shift = center_tensor - 0.5 - eps
    shift[~pbc] = 0.0

    # Convert positions to fractional coordinates
    fractional = torch.linalg.solve(cell.T, positions.T).T - shift

    if pretty_translation:
        fractional = translate_pretty(fractional, pbc)
        shift = center_tensor - 0.5
        shift[~pbc] = 0.0
        fractional += shift
    else:
        # Apply PBC wrapping - keep mask as boolean
        # Remove the problematic conversion: mask = pbc.to(positions.dtype)
        fractional = torch.where(
            pbc.unsqueeze(0),  # Keep as boolean tensor
            (fractional % 1.0) + shift.unsqueeze(0),
            fractional,
        )

    # Convert back to Cartesian coordinates
    return torch.matmul(fractional, cell)


def strides_of(v: torch.Tensor) -> torch.Tensor:
    """Calculate the cumulative strides of a flattened tensor.

    This function computes the cumulative sum of the input tensor `v` after flattening it.
    The resulting tensor contains the cumulative strides, which can be useful for indexing
    or iterating over elements in a flattened representation.

    Args:
        v (torch.Tensor): A tensor of any shape to be flattened and processed.

    Returns:
        torch.Tensor: A tensor of shape (n + 1,) where n is the number of elements in `v`,
        containing the cumulative strides.
    """
    v = v.flatten()
    stride = v.new_empty(v.shape[0] + 1)
    stride[0] = 0
    torch.cumsum(v, dim=0, dtype=stride.dtype, out=stride[1:])
    return stride


def get_number_of_cell_repeats(
    cutoff: float, cell: torch.Tensor, pbc: torch.Tensor
) -> torch.Tensor:
    """Determine the number of cell repeats required for a given
        cutoff distance.

    This function calculates how many times the unit cell needs to
    be repeated in each dimension to ensure that all interactions
    within the specified cutoff distance are accounted for,
    considering periodic boundary conditions (PBC).

    Args:
        cutoff (float): The cutoff distance for interactions.
        cell (torch.Tensor): A tensor of shape (n_cells, 3, 3)
            representing the unit cell matrices.
        pbc (torch.Tensor): A tensor of shape (n_cells, 3)
            indicating whether periodic boundary conditions are
            applied in each dimension.

    Returns:
        torch.Tensor: A tensor of shape (n_cells, 3)
            containing the number of repeats for each dimension,
            where non-PBC dimensions are set to zero.
    """
    cell = cell.view((-1, 3, 3))
    pbc = pbc.view((-1, 3))

    has_pbc = pbc.prod(dim=1, dtype=torch.bool)
    reciprocal_cell = torch.zeros_like(cell)
    reciprocal_cell[has_pbc, :, :] = torch.linalg.inv(cell[has_pbc, :, :]).transpose(2, 1)
    inv_distances = reciprocal_cell.norm(2, dim=-1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    return torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))


def get_cell_shift_idx(num_repeats: torch.Tensor, dtype: _dtype) -> torch.Tensor:
    """Generate the indices for cell shifts based on the number of repeats.

    This function creates a tensor of indices that represent the shifts in
    each dimension based on the specified number of repeats. The shifts are
    generated for all combinations of repeats in the three spatial dimensions.

    Args:
        num_repeats (torch.Tensor): A tensor of shape (3,)
            indicating the number of repeats in each dimension.
        dtype (_dtype): The desired data type for the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (n_shifts, 3) containing the
            Cartesian product of the shift indices for each dimension.
    """
    reps = []
    for ii in range(3):
        r1 = torch.arange(
            -num_repeats[ii],
            num_repeats[ii] + 1,
            device=num_repeats.device,
            dtype=dtype,
        )
        _, indices = torch.sort(torch.abs(r1))
        reps.append(r1[indices])
    return torch.cartesian_prod(reps[0], reps[1], reps[2])


def compute_distances_with_cell_shifts(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: torch.Tensor,
) -> torch.Tensor:
    """Compute distances between pairs of positions, optionally
        including cell shifts.

    This function calculates the Euclidean distances between pairs
    of positions specified by the mapping tensor. If cell shifts are
    provided, they are added to the distance calculation to account
    for periodic boundary conditions.

    Args:
        pos (torch.Tensor): A tensor of shape (n_atoms, 3)
            representing the positions of atoms.
        mapping (torch.Tensor): A tensor of shape (2, n_pairs) that
            specifies pairs of indices in `pos` for which to compute
            distances.
        cell_shifts (Optional[torch.Tensor]): A tensor of shape (n_pairs, 3)
            representing the shifts to apply to the distances for
            periodic boundary conditions. If None, no shifts are applied.

    Returns:
        torch.Tensor: A tensor of shape (n_pairs,) containing the
            computed distances for each pair.
    """
    if mapping.dim() != 2:
        raise ValueError(f"Mapping must be a 2D tensor, got {mapping.shape}")
    if mapping.shape[0] != 2:
        raise ValueError(f"Mapping must have 2 rows, got {mapping.shape[0]}")

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    return dr.norm(p=2, dim=1)


def compute_cell_shifts(
    cell: torch.Tensor, shifts_idx: torch.Tensor, system_mapping: torch.Tensor
) -> torch.Tensor:
    """Compute the cell shifts based on the provided indices and cell matrix.

    This function calculates the shifts to apply to positions based on the specified
    indices and the unit cell matrix. If the cell is None, it returns None.

    Args:
        cell (torch.Tensor): A tensor of shape (n_cells, 3, 3)
            representing the unit cell matrices.
        shifts_idx (torch.Tensor): A tensor of shape (n_shifts, 3)
            representing the indices for shifts.
        system_mapping (torch.Tensor): A tensor of shape (n_systems,)
            that maps the shifts to the corresponding cells.

    Returns:
        torch.Tensor: A tensor of shape (n_systems, 3) containing
            the computed cell shifts.
    """
    if cell is None:
        cell_shifts = None
    else:
        cell_shifts = torch.einsum(
            "jn,jnm->jm", shifts_idx, cell.view(-1, 3, 3)[system_mapping]
        )
    return cell_shifts


def get_fully_connected_mapping(
    *,
    i_ids: torch.Tensor,
    shifts_idx: torch.Tensor,
    self_interaction: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a fully connected mapping of atom indices with optional cell shifts.

    This function computes a mapping of atom indices for a fully connected graph,
    considering periodic boundary conditions through cell shifts. It can also exclude
    self-interactions based on the provided flag.

    Args:
        i_ids (torch.Tensor): A tensor of shape (n_atoms,)
            containing the indices of the atoms.
        shifts_idx (torch.Tensor): A tensor of shape (n_shifts, 3)
            representing the shifts to apply for periodic boundary
            conditions.
        self_interaction (bool): A flag indicating whether to include
            self-interactions in the mapping.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mapping (torch.Tensor): A tensor of shape (n_pairs, 2)
                representing the pairs of indices for which distances
                will be computed.
            - shifts_idx (torch.Tensor): A tensor of shape (n_pairs, 3)
                representing the corresponding shifts for the computed pairs.
    """
    n_atom = i_ids.shape[0]
    n_atom2 = n_atom * n_atom
    n_cell_image = shifts_idx.shape[0]
    j_ids = torch.repeat_interleave(
        i_ids, n_cell_image, dim=0, output_size=n_cell_image * n_atom
    )
    mapping = torch.cartesian_prod(i_ids, j_ids)
    shifts_idx = shifts_idx.repeat((n_atom2, 1))
    if not self_interaction:
        mask = torch.ones(mapping.shape[0], dtype=torch.bool, device=i_ids.device)
        ids = n_cell_image * torch.arange(n_atom, device=i_ids.device) + torch.arange(
            0, mapping.shape[0], n_atom * n_cell_image, device=i_ids.device
        )
        mask[ids] = False
        mapping = mapping[mask, :]
        shifts_idx = shifts_idx[mask]
    return mapping, shifts_idx


def build_naive_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool,  # noqa: FBT001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a naive neighborhood list for atoms based on positions
        and periodic boundary conditions.

    This function computes a neighborhood list of atoms within a
    specified cutoff distance, considering periodic boundary conditions
    defined by the unit cell. It returns the mapping of atom pairs,
    the system mapping for each structure, and the corresponding shifts.

    Args:
        positions (torch.Tensor): A tensor of shape (n_atoms, 3)
            representing the positions of atoms.
        cell (torch.Tensor): A tensor of shape (n_cells, 3, 3)
            representing the unit cell matrices.
        pbc (torch.Tensor): A tensor indicating whether
            periodic boundary conditions are applied.
        cutoff (float): The cutoff distance beyond which atoms are not
            considered neighbors.
        n_atoms (torch.Tensor): A tensor containing the number of atoms
            in each structure.
        self_interaction (bool): A flag indicating whether to include
            self-interactions.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - mapping (torch.Tensor): A tensor of shape (n_pairs, 2)
                representing the pairs of indices for neighboring atoms.
            - system_mapping (torch.Tensor): A tensor of shape (n_pairs,)
                indicating the structure index for each pair.
            - shifts_idx (torch.Tensor): A tensor of shape (n_pairs, 3)
                representing the shifts applied for periodic boundary
                conditions.
    """
    device = positions.device
    dtype = positions.dtype

    num_repeats_ = get_number_of_cell_repeats(cutoff, cell, pbc)

    stride = strides_of(n_atoms)
    ids = torch.arange(positions.shape[0], device=device, dtype=torch.long)

    mapping, system_mapping, shifts_idx_ = [], [], []
    for struct_idx in range(n_atoms.shape[0]):
        num_repeats = num_repeats_[struct_idx]
        shifts_idx = get_cell_shift_idx(num_repeats, dtype)
        i_ids = ids[stride[struct_idx] : stride[struct_idx + 1]]

        s_mapping, shifts_idx = get_fully_connected_mapping(
            i_ids=i_ids, shifts_idx=shifts_idx, self_interaction=self_interaction
        )
        mapping.append(s_mapping)
        system_mapping.append(
            torch.full((s_mapping.shape[0],), struct_idx, dtype=torch.long, device=device)
        )
        shifts_idx_.append(shifts_idx)
    return (
        torch.cat(mapping, dim=0).t(),
        torch.cat(system_mapping, dim=0),
        torch.cat(shifts_idx_, dim=0),
    )


def ravel_3d(idx_3d: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert 3D indices into linear indices for an array of given shape.

    This function takes 3D indices, which are typically used to
    reference elements in a 3D array, and converts them into
    linear indices. The linear index is calculated based on the
    provided shape of the array, allowing for easy access to
    elements in a flattened representation.

    Args:
        idx_3d (torch.Tensor): A tensor of shape [-1, 3]
            representing the 3D indices to be converted.
        shape (torch.Tensor): A tensor of shape [3]
            representing the dimensions of the array.

    Returns:
        torch.Tensor: A tensor containing the linear indices
            corresponding to the input 3D indices.
    """
    return idx_3d[:, 2] + shape[2] * (idx_3d[:, 1] + shape[1] * idx_3d[:, 0])


def unravel_3d(idx_linear: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert linear indices back into 3D indices for an array of given shape.

    This function takes linear indices, which are used to reference
    elements in a flattened array, and converts them back into 3D indices.
    The conversion is based on the provided shape of the array.

    Args:
        idx_linear (torch.Tensor): A tensor of shape [-1]
            representing the linear indices to be converted.
        shape (torch.Tensor): A tensor of shape [3]
            representing the dimensions of the array.

    Returns:
        torch.Tensor: A tensor of shape [-1, 3]
            containing the 3D indices corresponding to the input linear indices.
    """
    idx_3d = idx_linear.new_empty((idx_linear.shape[0], 3))
    idx_3d[:, 2] = torch.remainder(idx_linear, shape[2])
    idx_3d[:, 1] = torch.remainder(
        torch.div(idx_linear, shape[2], rounding_mode="floor"), shape[1]
    )
    idx_3d[:, 0] = torch.div(idx_linear, shape[1] * shape[2], rounding_mode="floor")
    return idx_3d


def get_linear_bin_idx(
    cell: torch.Tensor, pos: torch.Tensor, n_bins_s: torch.Tensor
) -> torch.Tensor:
    """Calculate the linear bin index for each position within a defined box.

    This function computes the linear bin index for each position
    based on the provided cell vectors and the number of bins in
    each direction. The positions are first scaled according to the
    cell dimensions, and then the corresponding bin indices are determined.

    Args:
        cell (torch.Tensor): A tensor of shape [3, 3]
            representing the cell vectors defining the box.
        pos (torch.Tensor): A tensor of shape [-1, 3]
            representing the set of positions to be binned.
        n_bins_s (torch.Tensor): A tensor of shape [3]
            representing the number of bins in each direction.

    Returns:
        torch.Tensor: A tensor containing the linear bin indices for each position.
    """
    scaled_pos = torch.linalg.solve(cell.t(), pos.t()).t()
    bin_index_s = torch.floor(scaled_pos * n_bins_s).to(torch.long)
    return ravel_3d(bin_index_s, n_bins_s)


def scatter_bin_index(
    n_bins: int,
    max_n_atom_per_bin: int,
    n_images: int,
    bin_index: torch.Tensor,
) -> torch.Tensor:
    """Convert a linear table of bin indices into a structured bin ID table.

    This function takes a linear table of bin indices and organizes
    it into a 2D table where each row corresponds to a bin and
    each column corresponds to an atom index. Empty entries in the
    resulting table are filled with a placeholder value (n_images)
    to facilitate later removal.

    Args:
        n_bins (int): The total number of bins.
        max_n_atom_per_bin (int): The maximum number of atoms that can be
            stored in each bin.
        n_images (int): The total number of atoms, including periodic
            boundary condition replicas.
        bin_index (torch.Tensor): A tensor mapping each atom index to
            its corresponding bin index.

    Returns:
        torch.Tensor: A tensor of shape [n_bins, max_n_atom_per_bin]
        relating bin indices (rows) to atom indices (columns).
    """
    device = bin_index.device
    sorted_bin_index, sorted_id = torch.sort(bin_index)
    bin_id = torch.full(
        (n_bins * max_n_atom_per_bin,), n_images, device=device, dtype=torch.long
    )
    sorted_bin_id = torch.remainder(
        torch.arange(bin_index.shape[0], device=device), max_n_atom_per_bin
    )
    sorted_bin_id = sorted_bin_index * max_n_atom_per_bin + sorted_bin_id
    bin_id.scatter_(dim=0, index=sorted_bin_id, src=sorted_id)
    return bin_id.view((n_bins, max_n_atom_per_bin))


def linked_cell(  # noqa: PLR0915
    pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    num_repeats: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor]:
    """Determine the atomic neighborhood of the atoms of a given structure
    for a particular cutoff using the linked cell algorithm.

    This function identifies neighboring atoms within a specified cutoff
    distance by utilizing the linked cell method. It accounts for
    periodic boundary conditions (PBC) by replicating the unit cell
    in all directions as necessary.

    Args:
        pos (torch.Tensor): A tensor of shape [n_atom, 3] representing
            atomic positions in the unit cell.
        cell (torch.Tensor): A tensor of shape [3, 3] representing
            the unit cell vectors.
        cutoff (float): The distance threshold used to determine which
            atoms are considered neighbors.
        num_repeats (torch.Tensor): A tensor indicating the number of
            unit cell repetitions required in each direction to account
            for periodic boundary conditions.
        self_interaction (bool, optional): If set to True, the original
            atoms will be included as their own neighbors. Default is False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - neigh_atom (torch.Tensor): A tensor containing pairs of indices
              where neigh_atom[0] represents the original atom indices
              and neigh_atom[1] represents their corresponding neighbor
              indices.
            - neigh_shift_idx (torch.Tensor): A tensor containing the cell
              shift indices for each neighbor atom, which are necessary for
              reconstructing the positions of the neighboring atoms.
    """
    device = pos.device
    dtype = pos.dtype
    n_atom = pos.shape[0]

    # Find all the integer shifts of the unit cell given the cutoff and periodicity
    shifts_idx = get_cell_shift_idx(num_repeats, dtype)
    n_cell_image = shifts_idx.shape[0]
    shifts_idx = torch.repeat_interleave(
        shifts_idx, n_atom, dim=0, output_size=n_atom * n_cell_image
    )
    batch_image = torch.zeros((shifts_idx.shape[0]), dtype=torch.long)
    cell_shifts = compute_cell_shifts(cell.view(-1, 3, 3), shifts_idx, batch_image)

    i_ids = torch.arange(n_atom, device=device, dtype=torch.long)
    i_ids = i_ids.repeat(n_cell_image)
    # Compute the positions of the replicated unit cell (including the original)
    # they are organized such that: 1st n_atom are the non-shifted atom,
    # 2nd n_atom are moved by the same translation, ...
    images = pos[i_ids] + cell_shifts
    n_images = images.shape[0]
    # Create a rectangular box at [0,0,0] that encompasses all the atoms
    # (hence shifting the atoms so that they lie inside the box)
    b_min = images.min(dim=0).values
    b_max = images.max(dim=0).values
    images -= b_min - 1e-5
    box_length = b_max - b_min + 1e-3

    # Divide the box into square bins of size cutoff in 3D
    n_bins_s = torch.maximum(torch.ceil(box_length / cutoff), pos.new_ones(3))
    # Adapt the box lengths so that it encompasses
    box_vec = torch.diag_embed(n_bins_s * cutoff)
    n_bins_s = n_bins_s.to(torch.long)
    n_bins = int(torch.prod(n_bins_s))
    # Determine which bins the original atoms and the images belong to following
    # a linear indexing of the 3D bins
    bin_index_j = get_linear_bin_idx(box_vec, images, n_bins_s)
    n_atom_j_per_bin = torch.bincount(bin_index_j, minlength=n_bins)
    max_n_atom_per_bin = int(n_atom_j_per_bin.max())
    # Convert the linear map bin_index_j into a 2D map. This allows for
    # Fully vectorized neighbor assignment
    bin_id_j = scatter_bin_index(n_bins, max_n_atom_per_bin, n_images, bin_index_j)

    # Find which bins the original atoms belong to
    bin_index_i = bin_index_j[:n_atom]
    i_bins_l = torch.unique(bin_index_i)
    i_bins_s = unravel_3d(i_bins_l, n_bins_s)

    # Find the bin indices in the neighborhood of i_bins_l. Since the bins have
    # a side length of cutoff only 27 bins are in the neighborhood
    # (including itself)
    dd = torch.tensor([0, 1, -1], dtype=torch.long, device=device)
    bin_shifts = torch.cartesian_prod(dd, dd, dd)
    n_neigh_bins = bin_shifts.shape[0]
    bin_shifts = bin_shifts.repeat((i_bins_s.shape[0], 1))
    neigh_bins_s = (
        torch.repeat_interleave(
            i_bins_s,
            n_neigh_bins,
            dim=0,
            output_size=n_neigh_bins * i_bins_s.shape[0],
        )
        + bin_shifts
    )

    # Some of the generated bin indices might not be valid
    mask = torch.all(
        torch.logical_and(neigh_bins_s < n_bins_s.view(1, 3), neigh_bins_s >= 0),
        dim=1,
    )

    # Remove the bins that are outside of the search range, i.e. beyond
    # the borders of the box in the case of non-periodic directions.
    neigh_j_bins_l = ravel_3d(neigh_bins_s[mask], n_bins_s)

    max_neigh_per_atom = max_n_atom_per_bin * n_neigh_bins
    # The i_bin related to neigh_j_bins_l
    repeats = mask.view(-1, n_neigh_bins).sum(dim=1)
    neigh_i_bins_l = torch.cat(
        [
            torch.arange(rr, device=device) + i_bins_l[ii] * n_neigh_bins
            for ii, rr in enumerate(repeats)
        ],
        dim=0,
    )
    # linear neighbor list. make it at large as necessary
    neigh_atom = torch.empty(
        (2, n_atom * max_neigh_per_atom), dtype=torch.long, device=device
    )
    # Fill the i_atom index
    neigh_atom[0] = (
        torch.arange(n_atom).view(-1, 1).repeat(1, max_neigh_per_atom).view(-1)
    )
    # Relate `bin_index` (row) with the `neighbor_atom_index` (stored in the columns).
    # empty entries are set to `n_images`
    bin_id_ij = torch.full(
        (n_bins * n_neigh_bins, max_n_atom_per_bin),
        n_images,
        dtype=torch.long,
        device=device,
    )
    # Fill the bins with neighbor atom indices
    bin_id_ij[neigh_i_bins_l] = bin_id_j[neigh_j_bins_l]
    bin_id_ij = bin_id_ij.view((n_bins, max_neigh_per_atom))

    # Map the neighbors in the bins to the central atoms
    neigh_atom[1] = bin_id_ij[bin_index_i].view(-1)

    # Remove empty entries
    neigh_atom = neigh_atom[:, neigh_atom[1] != n_images]

    if not self_interaction:
        # Neighbor atoms are still indexed from 0 to n_atom*n_cell_image
        neigh_atom = neigh_atom[:, neigh_atom[0] != neigh_atom[1]]

    # Sort neighbor list so that the i_atom indices increase
    sorted_ids = torch.argsort(neigh_atom[0])
    neigh_atom = neigh_atom[:, sorted_ids]

    # Get the cell shift indices for each neighbor atom
    neigh_shift_idx = shifts_idx[neigh_atom[1]]
    # make sure the j_atom indices access the original positions
    neigh_atom[1] = torch.remainder(neigh_atom[1], n_atom)
    return neigh_atom, neigh_shift_idx


def build_linked_cell_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the neighbor list of a given set of atomic structures using
    the linked cell algorithm.

    This function constructs a neighbor list for multiple atomic structures
    by applying the linked cell method. It handles periodic boundary conditions
    and returns the indices of neighboring atoms along with their corresponding
    structure information.

    Args:
        positions (torch.Tensor): A tensor containing the atomic positions
            for each structure, where each row corresponds to an atom's position
            in 3D space.
        cell (torch.Tensor): A tensor containing the unit cell vectors for
            each structure, formatted as a 3D array.
        pbc (torch.Tensor): A boolean tensor indicating the periodic boundary
            conditions to apply for each structure.
        cutoff (float): The distance threshold used to determine which atoms are
            considered neighbors.
        n_atoms (torch.Tensor): A tensor containing the number of atoms in each
            structure.
        self_interaction (bool): If set to True, the original atoms will be included as
            their own neighbors. Default is False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - mapping (torch.Tensor): A tensor containing pairs of indices where
              mapping[0] represents the central atom indices and mapping[1]
              represents their corresponding neighbor indices.
            - system_mapping (torch.Tensor): A tensor containing the structure indices
              corresponding to each neighbor atom.
            - cell_shifts_idx (torch.Tensor): A tensor containing the cell
              shift indices for each neighbor atom, which are necessary for
              reconstructing the positions of the neighboring atoms.
    """
    n_structure = n_atoms.shape[0]
    device = positions.device
    cell = cell.view((-1, 3, 3))
    pbc = pbc.view((-1, 3))
    # Compute the number of cell replicas necessary so that all the
    # unit cell's atoms have a complete neighborhood (no MIC assumed here)
    num_repeats = get_number_of_cell_repeats(cutoff, cell, pbc)

    stride = strides_of(n_atoms)

    mapping, system_mapping, cell_shifts_idx = [], [], []
    for struct_idx in range(n_structure):
        # Compute the neighborhood with the linked cell algorithm
        neigh_atom, neigh_shift_idx = linked_cell(
            positions[stride[struct_idx] : stride[struct_idx + 1]],
            cell[struct_idx],
            cutoff,
            num_repeats[struct_idx],
            self_interaction,
        )

        system_mapping.append(
            struct_idx * torch.ones(neigh_atom.shape[1], dtype=torch.long, device=device)
        )
        # Shift the mapping indices to access positions
        mapping.append(neigh_atom + stride[struct_idx])
        cell_shifts_idx.append(neigh_shift_idx)

    return (
        torch.cat(mapping, dim=1),
        torch.cat(system_mapping, dim=0),
        torch.cat(cell_shifts_idx, dim=0),
    )


def multiplicative_isotropic_cutoff(
    fn: Callable[..., torch.Tensor],
    r_onset: float | torch.Tensor,
    r_cutoff: float | torch.Tensor,
) -> Callable[..., torch.Tensor]:
    """Creates a smoothly truncated version of an isotropic function.

    Takes an isotropic function f(r) and constructs a new function f'(r) that smoothly
    transitions to zero between r_onset and r_cutoff. The resulting function is C¹
    continuous (continuous in both value and first derivative).

    The truncation is achieved by multiplying the original function by a smooth
    switching function S(r) where:
    - S(r) = 1 for r < r_onset
    - S(r) = 0 for r > r_cutoff
    - S(r) smoothly transitions between 1 and 0 for r_onset < r < r_cutoff

    The switching function follows the form used in HOOMD-blue:
    S(r) = (rc² - r²)² * (rc² + 2r² - 3ro²) / (rc² - ro²)³
    where rc = r_cutoff and ro = r_onset

    Args:
        fn: Function to be truncated. Should take a tensor of distances [n, m]
            as first argument, plus optional additional arguments.
        r_onset: Distance at which the function begins to be modified.
        r_cutoff: Distance at which the function becomes zero.

    Returns:
        A new function with the same signature as fn that smoothly goes to zero
        between r_onset and r_cutoff.

    References:
        HOOMD-blue documentation:
        https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/module-pair.html
    """
    r_c = torch.square(torch.tensor(r_cutoff))
    r_o = torch.square(torch.tensor(r_onset))

    def smooth_fn(dr: torch.Tensor) -> torch.Tensor:
        """Compute the smooth switching function."""
        r = torch.square(dr)

        # Compute switching function for intermediate region
        numerator = torch.square(r_c - r) * (r_c + 2 * r - 3 * r_o)
        denominator = torch.pow(r_c - r_o, 3)
        intermediate = torch.where(
            dr < r_cutoff, numerator / denominator, torch.zeros_like(dr)
        )

        # Return 1 for r < r_onset, switching function for r_onset < r < r_cutoff
        return torch.where(dr < r_onset, torch.ones_like(dr), intermediate)

    @wraps(fn)
    def cutoff_fn(dr: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the switching function to the original function."""
        return smooth_fn(dr) * fn(dr, *args, **kwargs)

    return cutoff_fn


def high_precision_sum(
    x: torch.Tensor,
    dim: int | Iterable[int] | None = None,
    *,
    keepdim: bool = False,
) -> torch.Tensor:
    """Sums tensor elements over specified dimensions at 64-bit precision.

    This function casts the input tensor to a higher precision type (64-bit),
    performs the summation, and then casts back to the original dtype. This helps
    prevent numerical instability issues that can occur when summing many numbers,
    especially with floating point values.

    Args:
        x: Input tensor to sum
        dim: Dimension(s) along which to sum. If None, sum over all dimensions
        keepdim: If True, retains reduced dimensions with length 1

    Returns:
        torch.Tensor: Sum of elements cast back to original dtype

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        >>> high_precision_sum(x)
        tensor(6., dtype=torch.float32)
    """
    if torch.is_complex(x):
        high_precision_dtype = torch.complex128
    elif torch.is_floating_point(x):
        high_precision_dtype = torch.float64
    else:  # integer types
        high_precision_dtype = torch.int64

    # Cast to high precision, sum, and cast back to original dtype
    return torch.sum(x.to(high_precision_dtype), dim=dim, keepdim=keepdim).to(x.dtype)


def safe_mask(
    mask: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    operand: torch.Tensor,
    placeholder: float = 0.0,
) -> torch.Tensor:
    """Safely applies a function to masked values in a tensor.

    This function applies the given function only to elements where the mask is True,
    avoiding potential numerical issues with masked-out values. Masked-out positions
    are filled with the placeholder value.

    Args:
        mask: Boolean tensor indicating which elements to process (True) or mask (False)
        fn: TorchScript function to apply to the masked elements
        operand: Input tensor to apply the function to
        placeholder: Value to use for masked-out positions (default: 0.0)

    Returns:
        torch.Tensor: Result tensor where fn is applied to masked elements and
            placeholder value is used for masked-out elements

    Example:
        >>> x = torch.tensor([1.0, 2.0, -1.0])
        >>> mask = torch.tensor([True, True, False])
        >>> safe_mask(mask, torch.log, x)
        tensor([0, 0.6931, 0])
    """
    masked = torch.where(mask, operand, torch.zeros_like(operand))
    return torch.where(mask, fn(masked), torch.full_like(operand, placeholder))


def unwrap_positions(
    pos: torch.Tensor, box: torch.Tensor, system_idx: torch.Tensor
) -> torch.Tensor:
    """Vectorized unwrapping for multiple systems without explicit loops.

    Parameters
    ----------
    pos : (T, N_tot, 3)
        Wrapped cartesian positions for all systems concatenated.
    box : (n_systems, 3, 3) or (T, n_systems, 3, 3)
        Box matrices, constant or time-dependent.
    system_idx : (N_tot,)
        For each atom, which system it belongs to (0..n_systems-1).

    Returns:
    -------
    unwrapped_pos : (T, N_tot, 3)
        Unwrapped cartesian positions.
    """
    # -- Constant boxes per system
    if box.ndim == 3:
        inv_box = torch.inverse(box)  # (n_systems, 3, 3)

        # Map each atom to its system's box
        inv_box_atoms = inv_box[system_idx]  # (N, 3, 3)
        box_atoms = box[system_idx]  # (N, 3, 3)

        # Compute fractional coordinates
        frac = torch.einsum("tni,nij->tnj", pos, inv_box_atoms)

        # Fractional displacements and unwrap
        dfrac = frac[1:] - frac[:-1]
        dfrac -= torch.round(dfrac)

        # Back to Cartesian
        dcart = torch.einsum("tni,nij->tnj", dfrac, box_atoms)

    # -- Time-dependent boxes per system
    elif box.ndim == 4:
        inv_box = torch.inverse(box)  # (T, n_systems, 3, 3)

        # Gather each atom's box per frame efficiently
        inv_box_atoms = inv_box[:, system_idx]  # (T, N, 3, 3)
        box_atoms = box[:, system_idx]  # (T, N, 3, 3)

        # Compute fractional coordinates per frame
        frac = torch.einsum("tni,tnij->tnj", pos, inv_box_atoms)

        dfrac = frac[1:] - frac[:-1]
        dfrac -= torch.round(dfrac)

        dcart = torch.einsum("tni,tnij->tnj", dfrac, box_atoms[:-1])

    else:
        raise ValueError("box must have shape (n_systems,3,3) or (T,n_systems,3,3)")

    # Cumulative reconstruction
    unwrapped = torch.empty_like(pos)
    unwrapped[0] = pos[0]
    unwrapped[1:] = torch.cumsum(dcart, dim=0) + unwrapped[0]

    return unwrapped
