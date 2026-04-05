"""Coordinate transformations and periodic boundary conditions.

This module provides functions for handling coordinate transformations and periodic
boundary conditions in molecular simulations, including matrix inversions and
general PBC wrapping.
"""

from collections.abc import Callable
from functools import wraps

import torch
from torch.nn.utils.rnn import pad_sequence
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


def pbc_wrap_batched(
    positions: torch.Tensor,
    cell: torch.Tensor,
    system_idx: torch.Tensor,
    pbc: torch.Tensor | bool = True,  # noqa: FBT001, FBT002
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
        pbc (torch.Tensor | bool): Tensor of shape (3,) containing boolean values
            indicating whether periodic boundary conditions are applied in each dimension.
            Can also be a bool. Defaults to True.

    Returns:
        torch.Tensor: Wrapped positions in real space with same shape as input positions.
    """
    if isinstance(pbc, bool):
        pbc = torch.tensor([pbc, pbc, pbc], dtype=torch.bool, device=positions.device)
    if not torch.is_floating_point(positions) or not torch.is_floating_point(cell):
        raise TypeError("Positions and lattice vectors must be floating point tensors.")
    if positions.shape[-1] != cell.shape[-1]:
        raise ValueError("Position dimensionality must match lattice vectors.")
    uniq_systems = torch.unique(system_idx)
    n_systems = len(uniq_systems)
    if n_systems != cell.shape[0]:
        raise ValueError(
            f"Number of unique systems ({n_systems}) doesn't "
            f"match number of cells ({cell.shape[0]})"
        )
    pbc_batched = pbc.unsqueeze(0).expand(n_systems, -1)
    wrapped, _ = pbc_wrap_batched_and_get_lattice_shifts(
        positions, cell.mT, system_idx, pbc_batched
    )
    return wrapped


def pbc_wrap_batched_and_get_lattice_shifts(
    positions: torch.Tensor,
    cell: torch.Tensor,
    system_idx: torch.Tensor,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap Cartesian positions into the primary cell and return the applied shifts.

    ``cell`` rows are lattice vectors (row-vector convention matching
    ``compute_cell_shifts`` and the batched neighbor-list APIs). Fractional coordinates
    use ``cell_col = cell[s].T`` so ``r = f @ cell[s]`` with ``f`` in ``[0, 1)`` on
    periodic axes. Atoms in non-periodic systems or systems with singular cells are
    left unchanged.

    Returns ``(wrapped_positions, lattice_shifts)`` where ``lattice_shifts[i]`` is the
    integer vector ``floor(frac_i)`` on periodic axes (zero elsewhere). The neighbor-list
    code uses these to correct ``shifts_idx`` so they remain valid for the original
    (unwrapped) input positions.
    """
    cell_T = cell.transpose(1, 2)
    dets = torch.linalg.det(cell_T)
    invertible = torch.isfinite(dets) & (dets.abs() > 1e-12)
    active = pbc.any(dim=1) & invertible

    if not active.any():
        return positions.clone(), torch.zeros_like(positions, dtype=cell.dtype)

    # Get the inverse cell for each atom based on its system index
    B = torch.zeros_like(cell)  # Shape: (n_systems, 3, 3)
    B[active] = torch.linalg.inv(cell_T[active])
    B_per_atom = B[system_idx]  # Shape: (n_atoms, 3, 3)

    # Transform to fractional coordinates: f = B·r
    # For each atom, multiply its position by its system's inverse cell matrix
    frac = torch.bmm(B_per_atom, positions.unsqueeze(2)).squeeze(2)

    pbc_per_atom = pbc[system_idx]
    active_per_atom = active[system_idx].unsqueeze(1)
    pbc_mask = pbc_per_atom & active_per_atom

    # Wrap to reference cell [0,1) using floor
    int_shifts = torch.where(pbc_mask, torch.floor(frac), torch.zeros_like(frac))
    wrapped_frac = frac - int_shifts

    # Transform back to real space: r = A·f
    # For each atom, multiply its wrapped fractional coords by its system's cell matrix
    cell_per_atom = cell_T[system_idx]  # Shape: (n_atoms, 3, 3)
    wrapped_pos = torch.bmm(cell_per_atom, wrapped_frac.unsqueeze(2)).squeeze(2)
    out = torch.where(active_per_atom, wrapped_pos, positions)
    shifts = torch.where(active_per_atom, int_shifts, torch.zeros_like(int_shifts))
    return out, shifts


def minimum_image_displacement(
    *,
    dr: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | bool = True,
) -> torch.Tensor:
    """Apply minimum image convention to displacement vectors.

    Args:
        dr (torch.Tensor): Displacement vectors [N, 3] or [N, N, 3].
        cell (Optional[torch.Tensor]): Unit cell matrix [3, 3].
        pbc (Optional[torch.Tensor]): Boolean tensor of shape (3,) indicating
            periodic boundary conditions in each dimension.

    Returns:
        torch.Tensor: Minimum image displacement vectors with same shape as input.
    """
    if isinstance(pbc, bool):
        pbc = torch.tensor([pbc] * 3, dtype=torch.bool, device=dr.device)
    if cell is None or not pbc.any():
        return dr

    # Convert to fractional coordinates
    cell_inv = torch.linalg.inv(cell)
    dr_frac = torch.einsum("ij,...j->...i", cell_inv, dr)

    # Apply minimum image convention
    dr_frac -= torch.where(pbc, torch.round(dr_frac), torch.zeros_like(dr_frac))

    # Convert back to cartesian
    return torch.einsum("ij,...j->...i", cell, dr_frac)


def get_pair_displacements(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | bool = True,
    pairs: tuple[torch.Tensor, torch.Tensor] | None = None,
    shifts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute displacement vectors and distances between atom pairs.

    Args:
        positions (torch.Tensor): Atomic positions [N, 3].
        cell (Optional[torch.Tensor]): Unit cell matrix [3, 3].
        pbc (Optional[torch.Tensor]): Boolean tensor of shape (3,) indicating
            periodic boundary conditions in each dimension.
        pairs (Optional[Tuple[torch.Tensor, torch.Tensor]]):
            (i, j) indices for specific pairs to compute.
        shifts (Optional[torch.Tensor]): Shift vectors for periodic images [n_pairs, 3].

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Displacement vectors [n_pairs, 3].
            - Distances [n_pairs].
    """
    if isinstance(pbc, bool):
        pbc = torch.tensor([pbc] * 3, dtype=torch.bool, device=positions.device)
    if pairs is None:
        # Create full distance matrix
        ri = positions.unsqueeze(0)  # [1, N, 3]
        rj = positions.unsqueeze(1)  # [N, 1, 3]
        dr = rj - ri  # [N, N, 3]

        if cell is not None and pbc.any():
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

    if cell is not None and pbc.any():
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

    has_pbc = pbc.any(dim=1)
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
        n_rep = int(num_repeats[ii].item())
        r1 = torch.arange(
            -n_rep,
            n_rep + 1,
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


def _calculate_n2_lattice_shifts(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    """Compute the superset of integer lattice shift vectors needed across all systems.

    For periodic axes, computes the number of images needed based on
    face-to-face distances. Non-periodic axes get zero repeats.

    Args:
        cell: Cell matrices [n_systems, 3, 3].
        pbc: PBC flags [n_systems, 3].
        cutoff: Cutoff radius.

    Returns:
        Integer lattice shift vectors [n_shifts, 3].
    """
    num_repeats = get_number_of_cell_repeats(cutoff, cell, pbc)  # (n_systems, 3)
    # take the max across all systems so a single shift set covers everything
    S_max = num_repeats.max(dim=0).values  # (3,)
    repeat_x = int(S_max[0].item())
    repeat_y = int(S_max[1].item())
    repeat_z = int(S_max[2].item())

    return torch.cartesian_prod(
        torch.arange(-repeat_x, repeat_x + 1, device=cell.device, dtype=torch.long),
        torch.arange(-repeat_y, repeat_y + 1, device=cell.device, dtype=torch.long),
        torch.arange(-repeat_z, repeat_z + 1, device=cell.device, dtype=torch.long),
    )  # (n_shifts, 3)


def _pad_batched_positions(
    positions: torch.Tensor,
    n_atoms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad flat per-system positions and return the atom mask and offsets."""
    device = positions.device
    n_systems = n_atoms.shape[0]
    n_max = int(n_atoms.max().item())
    offsets = torch.zeros(n_systems, dtype=torch.long, device=device)
    if n_systems > 1:
        offsets[1:] = torch.cumsum(n_atoms[:-1], dim=0)
    pos_list = [positions[offsets[i] : offsets[i] + n_atoms[i]] for i in range(n_systems)]
    batch_positions = pad_sequence(pos_list, batch_first=True, padding_value=0.0)
    atom_mask = torch.arange(n_max, device=device).unsqueeze(0) < n_atoms.unsqueeze(1)
    return batch_positions, atom_mask, offsets


def build_naive_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool,  # noqa: FBT001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a vectorized O(N^2) neighborhood list for batched atomic systems.

    All systems are padded to a common size and processed simultaneously
    using batched tensor operations. Pairs within the cutoff are returned
    with global atom indices.

    NOTE: due to the use of `pad_sequence`, this function is best used when
    all the systems being batched have a similar number of atoms as this
    reduces the memory overhead of the padding.

    Args:
        positions (torch.Tensor): A tensor of shape (n_total_atoms, 3)
            representing the positions of atoms.
        cell (torch.Tensor): A tensor of shape (n_systems, 3, 3)
            representing the unit cell matrices.
        pbc (torch.Tensor): A tensor of shape (n_systems, 3) indicating
            whether periodic boundary conditions are applied.
        cutoff (float): The cutoff distance beyond which atoms are not
            considered neighbors.
        n_atoms (torch.Tensor): A tensor containing the number of atoms
            in each structure.
        self_interaction (bool): A flag indicating whether to include
            self-interactions.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - mapping (torch.Tensor): A tensor of shape (2, n_pairs)
                representing the pairs of global indices for neighboring atoms.
            - system_mapping (torch.Tensor): A tensor of shape (n_pairs,)
                indicating the structure index for each pair.
            - shifts_idx (torch.Tensor): A tensor of shape (n_pairs, 3)
                representing the integer lattice shifts for each pair.

    References:
        - https://github.com/venkatkapil24/batch_nl: inspired the use of `pad_sequence`
          to vectorize a previous implementation that used a loop to iterate over systems.
    """
    device = positions.device
    dtype = positions.dtype

    cell = cell.view(-1, 3, 3)
    pbc = pbc.view(-1, 3).to(torch.bool)
    batch_positions, batch_mask, offsets = _pad_batched_positions(positions, n_atoms)

    # --- compute lattice shifts ---
    lattice_shifts = _calculate_n2_lattice_shifts(cell, pbc, cutoff)  # (n_shifts, 3)

    # Cartesian shifts per system: (n_systems, n_shifts, 3)
    cart_shifts = torch.matmul(lattice_shifts.to(dtype), cell)

    # shifted positions: (n_systems, n_shifts, n_max, 3)
    shifted = cart_shifts.unsqueeze(-2) + batch_positions.unsqueeze(1)

    # pairwise distances: (n_systems, n_shifts, n_max, n_max)
    diff = batch_positions.unsqueeze(1).unsqueeze(3) - shifted.unsqueeze(2)
    dist = torch.sqrt((diff * diff).sum(dim=-1))

    # --- build criterion mask ---
    criterion = dist < cutoff
    if not self_interaction:
        criterion = criterion & (dist >= 1e-6)

    # mask out shifts along non-periodic axes per system
    # pbc: (n_systems, 3), lattice_shifts: (n_shifts, 3)
    # a shift is valid only if non-zero components are along periodic axes
    # shift_ok: (n_systems, n_shifts) — True if the shift is allowed for that system
    shift_is_zero = lattice_shifts == 0  # (n_shifts, 3)
    shift_ok = (shift_is_zero.unsqueeze(0) | pbc.unsqueeze(1)).all(
        dim=-1
    )  # (n_systems, n_shifts)
    criterion = criterion & shift_ok[:, :, None, None]

    # mask out padded atoms
    pair_mask = (batch_mask.unsqueeze(-2) & batch_mask.unsqueeze(-1)).unsqueeze(
        1
    )  # (n_systems, 1, n_max, n_max)
    criterion = criterion & pair_mask

    # --- extract edges ---
    config_idx, shift_idx, atom_idx, neighbor_idx = torch.nonzero(
        criterion,
        as_tuple=True,
    )

    if config_idx.numel() == 0:
        mapping = torch.zeros((2, 0), dtype=torch.long, device=device)
        system_mapping = torch.zeros(0, dtype=torch.long, device=device)
        shifts_out = torch.zeros((0, 3), dtype=dtype, device=device)
        return mapping, system_mapping, shifts_out

    # convert local indices to global atom indices
    mapping = torch.stack(
        [
            atom_idx + offsets[config_idx],
            neighbor_idx + offsets[config_idx],
        ],
        dim=0,
    ).to(torch.long)

    system_mapping = config_idx.to(torch.long)
    shifts_out = lattice_shifts[shift_idx].to(dtype)

    return mapping, system_mapping, shifts_out


def ravel_3d(idx_3d: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert 3D indices into linear indices for an array of given shape.

    This function takes 3D indices, which are typically used to
    reference elements in a 3D array, and converts them into
    linear indices. The linear index is calculated based on the
    provided shape of the array, allowing for easy access to
    elements in a flattened representation.

    Args:
        idx_3d (torch.Tensor): A tensor of shape [..., 3]
            representing the 3D indices to be converted.
        shape (torch.Tensor): A tensor of shape [3]
            representing the dimensions of the array.

    Returns:
        torch.Tensor: A tensor of shape [...]
            containing the linear indices
            corresponding to the input 3D indices.
    """
    return idx_3d[..., 2] + shape[2] * (idx_3d[..., 1] + shape[1] * idx_3d[..., 0])


def unravel_3d(idx_linear: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert linear indices back into 3D indices for an array of given shape.

    This function takes linear indices, which are used to reference
    elements in a flattened array, and converts them back into 3D indices.
    The conversion is based on the provided shape of the array.

    Args:
        idx_linear (torch.Tensor): A tensor of shape [...]
            representing the linear indices to be converted.
        shape (torch.Tensor): A tensor of shape [3]
            representing the dimensions of the array.

    Returns:
        torch.Tensor: A tensor of shape [..., 3]
            containing the 3D indices corresponding to the input linear indices.
    """
    z = torch.remainder(idx_linear, shape[2])
    y = torch.remainder(torch.div(idx_linear, shape[2], rounding_mode="floor"), shape[1])
    x = torch.div(idx_linear, shape[1] * shape[2], rounding_mode="floor")
    return torch.stack([x, y, z], dim=-1)


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


def build_linked_cell_neighborhood_serial(
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
    # TODO: can we vectorize this for loop?
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


def _neighbor_bin_shifts_3d(device: torch.device) -> torch.Tensor:
    """Return the 27 neighboring 3D bin offsets."""
    dd = torch.tensor([0, 1, -1], dtype=torch.long, device=device)
    return torch.cartesian_prod(dd, dd, dd)


def _within_bin_position(sorted_bin: torch.Tensor) -> torch.Tensor:
    """Compute within-bin position for a sorted bin index tensor.

    Given (S, N) of sorted bin indices, returns (S, N) where each element
    is the 0-based position of that atom within its bin (0 for first atom
    in the bin, 1 for second, etc.).

    Uses a vectorized segment-cumsum: cumsum of ones, minus the cumulative
    count at each group boundary broadcast to the group.
    """
    S, N = sorted_bin.shape
    device = sorted_bin.device
    same = torch.ones(S, N, dtype=torch.long, device=device)
    same[:, 1:] = (sorted_bin[:, 1:] == sorted_bin[:, :-1]).long()
    cum = torch.cumsum(same, dim=1)  # (S, N) — running count within each row
    boundary = same == 0  # True at group starts (except position 0)
    boundary_val = torch.where(boundary, cum - 1, torch.zeros_like(cum))
    correction = torch.cummax(boundary_val, dim=1).values
    return cum - correction - 1


def _build_linked_cell_images_batched(
    batch_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    shifts_idx_unique: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shifted images and validity mask for the batched linked-cell path."""
    n_structure, n_max = batch_pos.shape[:2]
    n_shifts = shifts_idx_unique.shape[0]
    cart_shifts = torch.matmul(
        shifts_idx_unique.to(batch_pos.dtype), cell
    )  # (n_systems, n_shifts, 3)
    shift_ok = ((shifts_idx_unique == 0).unsqueeze(0) | pbc.unsqueeze(1)).all(dim=-1)
    # `shift_ok` is (n_systems, n_shifts): valid shared lattice shifts per system.
    # Flatten the (shift, atom) grid into one image axis for later sorting/binning.
    images_flat = (batch_pos.unsqueeze(1) + cart_shifts.unsqueeze(2)).reshape(
        n_structure, n_shifts * n_max, 3
    )  # (n_systems, n_shifts * max_atoms, 3)
    image_valid = (
        (atom_mask.unsqueeze(1) & shift_ok.unsqueeze(-1))
        .expand(-1, n_shifts, -1)
        .reshape(n_structure, n_shifts * n_max)
    )  # (n_systems, n_shifts * max_atoms)
    return cart_shifts, images_flat, image_valid, shift_ok


def _bin_linked_cell_images_batched(
    batch_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    images_flat: torch.Tensor,
    image_valid: torch.Tensor,
    cart_shifts: torch.Tensor,
    shift_ok: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign shifted images to bins and build the per-bin image lookup table."""
    device = batch_pos.device
    dtype = batch_pos.dtype
    n_structure = batch_pos.shape[0]
    n_img = images_flat.shape[1]
    big_val = torch.finfo(dtype).max / 2
    pos_for_min = torch.where(
        atom_mask.unsqueeze(-1), batch_pos, torch.full_like(batch_pos, big_val)
    )
    pos_for_max = torch.where(
        atom_mask.unsqueeze(-1), batch_pos, torch.full_like(batch_pos, -big_val)
    )
    shift_for_min = torch.where(
        shift_ok.unsqueeze(-1), cart_shifts, torch.full_like(cart_shifts, big_val)
    )
    shift_for_max = torch.where(
        shift_ok.unsqueeze(-1), cart_shifts, torch.full_like(cart_shifts, -big_val)
    )
    b_min = pos_for_min.min(dim=1).values + shift_for_min.min(dim=1).values
    b_max = pos_for_max.max(dim=1).values + shift_for_max.max(dim=1).values
    # Rebuild the same cutoff-sized box construction used in the single-structure path.
    images_shifted = images_flat - b_min.unsqueeze(1) + 1e-5
    box_length = b_max - b_min + 1e-3
    n_bins_s_per_sys = torch.maximum(
        torch.ceil(box_length / cutoff),
        torch.ones(n_structure, 3, device=device, dtype=dtype),
    ).to(torch.long)  # (n_systems, 3)
    n_bins_s = n_bins_s_per_sys.max(dim=0).values  # (3,)
    n_bins = int(n_bins_s.prod().item())
    box_diag_per_sys = n_bins_s_per_sys.to(dtype) * cutoff
    scaled_pos = images_shifted / box_diag_per_sys.unsqueeze(1)
    bin_3d = torch.floor(scaled_pos * n_bins_s_per_sys.to(dtype).unsqueeze(1)).to(
        torch.long
    )
    bin_3d = bin_3d.clamp(
        min=torch.zeros(3, device=device, dtype=torch.long),
        max=(n_bins_s - 1),
    )
    bin_linear = ravel_3d(bin_3d, n_bins_s)  # (n_systems, n_shifts * max_atoms)
    # Sort-by-bin lets us scatter images into a dense (bin, slot) lookup table.
    safe_bin = torch.where(image_valid, bin_linear, torch.full_like(bin_linear, n_bins))
    sorted_bin, sorted_order = torch.sort(safe_bin, dim=1)
    sorted_valid = image_valid.gather(1, sorted_order)
    sorted_bin_clamped = torch.where(
        sorted_valid, sorted_bin, torch.zeros_like(sorted_bin)
    )
    within_pos = _within_bin_position(sorted_bin_clamped)
    within_pos = torch.where(sorted_valid, within_pos, torch.zeros_like(within_pos))
    counts = torch.zeros(n_structure, n_bins, device=device, dtype=torch.long)
    counts.scatter_add_(1, sorted_bin_clamped, sorted_valid.long())
    max_apb = max(int(counts.max().item()), 1)
    sentinel = n_img
    bin_id_j = torch.full(
        (n_structure, n_bins * max_apb + 1), sentinel, dtype=torch.long, device=device
    )
    flat_target = sorted_bin_clamped * max_apb + within_pos
    scatter_mask = sorted_valid & (within_pos < max_apb)
    trash_idx = n_bins * max_apb
    safe_target = torch.where(
        scatter_mask, flat_target, torch.full_like(flat_target, trash_idx)
    )
    src_vals = torch.where(
        scatter_mask, sorted_order, torch.full_like(sorted_order, sentinel)
    )
    bin_id_j.scatter_(1, safe_target, src_vals)
    bin_id_j = bin_id_j[:, : n_bins * max_apb].view(
        n_structure, n_bins, max_apb
    )  # (n_systems, n_bins, max_atoms_per_bin)
    return bin_linear, bin_id_j, n_bins_s


def _gather_linked_cell_candidates_batched(
    bin_linear: torch.Tensor,
    bin_id_j: torch.Tensor,
    atom_mask: torch.Tensor,
    n_bins_s: torch.Tensor,
    shifts_idx_unique: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Gather per-atom candidate neighbor images from surrounding bins."""
    device = bin_linear.device
    n_structure, n_max = atom_mask.shape
    max_apb = bin_id_j.shape[2]
    sentinel = bin_linear.shape[1]
    zero_shift_idx = (shifts_idx_unique == 0).all(dim=1).nonzero(as_tuple=True)[0][0]
    orig_start = int(zero_shift_idx.item()) * n_max
    bin_index_i = bin_linear[:, orig_start : orig_start + n_max]
    bin_shifts_27 = _neighbor_bin_shifts_3d(device)
    n_nb = bin_shifts_27.shape[0]
    # Each central atom only needs images from its own bin and the 26 adjacent bins.
    i_bins_3d = unravel_3d(bin_index_i, n_bins_s)
    neigh_bins_3d = i_bins_3d.unsqueeze(2) + bin_shifts_27.view(
        1, 1, n_nb, 3
    )  # (n_systems, max_atoms, 27, 3)
    neigh_ok = ((neigh_bins_3d >= 0) & (neigh_bins_3d < n_bins_s.view(1, 1, 1, 3))).all(
        dim=-1
    ) & atom_mask.unsqueeze(2)
    neigh_bins_lin = ravel_3d(
        neigh_bins_3d.clamp(
            min=torch.zeros(3, device=device, dtype=torch.long),
            max=(n_bins_s - 1),
        ),
        n_bins_s,
    )
    gather_idx = (
        neigh_bins_lin.reshape(n_structure, -1).unsqueeze(-1).expand(-1, -1, max_apb)
    )  # (n_systems, max_atoms * 27, max_atoms_per_bin)
    candidates = bin_id_j.gather(1, gather_idx).reshape(
        n_structure, n_max, n_nb, max_apb
    )  # (n_systems, max_atoms, 27, max_atoms_per_bin)
    candidate_valid = neigh_ok.unsqueeze(-1) & (
        candidates != sentinel
    )  # (n_systems, max_atoms, 27, max_atoms_per_bin)
    return (
        candidates.reshape(n_structure, n_max, n_nb * max_apb),
        candidate_valid.reshape(n_structure, n_max, n_nb * max_apb),
        orig_start,
    )


def _finalize_linked_cell_pairs_batched(
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    offsets: torch.Tensor,
    shifts_idx_unique: torch.Tensor,
    orig_start: int,
    *,
    self_interaction: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert candidate image indices into the final neighbor list outputs."""
    device = candidates.device
    _, n_max, _ = candidates.shape
    pair_valid = candidate_valid
    if not self_interaction:
        local_image_idx = torch.arange(n_max, device=device).view(1, -1, 1) + orig_start
        pair_valid = pair_valid & (candidates != local_image_idx)
    # Compact the valid (system, i, candidate-slot) triples once, then index directly.
    s_flat, i_flat, k_flat = pair_valid.nonzero(as_tuple=True)
    j_flat = candidates[s_flat, i_flat, k_flat]
    j_atom = j_flat % n_max
    shift_out = shifts_idx_unique[j_flat // n_max]
    mapping = torch.stack([i_flat + offsets[s_flat], j_atom + offsets[s_flat]], dim=0)
    sort_idx = torch.argsort(mapping[0])
    return mapping[:, sort_idx], s_flat[sort_idx], shift_out[sort_idx]


def build_linked_cell_neighborhood_batched(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully batched linked-cell neighbor list construction.

    Drop-in replacement for build_linked_cell_neighborhood that processes all
    structures simultaneously using padded tensors, eliminating Python for-loops
    over structures. The algorithm mirrors the single-structure ``linked_cell``
    but operates on all structures at once via padded 3D/4D tensors.

    Args:
        positions: Flat atomic positions (N_total, 3).
        cell: Unit cell matrices, broadcastable to (n_structure, 3, 3).
        pbc: PBC flags, broadcastable to (n_structure, 3).
        cutoff: Neighbor cutoff distance.
        n_atoms: Number of atoms per structure (n_structure,).
        self_interaction: Whether to include self-pairs.

    Returns:
        (mapping, system_mapping, cell_shifts_idx) — same format as
        build_linked_cell_neighborhood.
    """
    shift_dtype = cell.dtype
    cell = cell.view(-1, 3, 3)
    pbc = pbc.view(-1, 3).to(torch.bool)
    shifts_idx_unique = _calculate_n2_lattice_shifts(cell, pbc, cutoff)
    batch_pos, atom_mask, offsets = _pad_batched_positions(positions, n_atoms)
    # Mirror the main linked-cell stages: images -> bins -> candidates -> pairs.
    cart_shifts, images_flat, image_valid, shift_ok = _build_linked_cell_images_batched(
        batch_pos, atom_mask, shifts_idx_unique, cell, pbc
    )
    bin_linear, bin_id_j, n_bins_s = _bin_linked_cell_images_batched(
        batch_pos, atom_mask, images_flat, image_valid, cart_shifts, shift_ok, cutoff
    )
    candidates, candidate_valid, orig_start = _gather_linked_cell_candidates_batched(
        bin_linear, bin_id_j, atom_mask, n_bins_s, shifts_idx_unique
    )
    mapping, system_mapping, shift_out = _finalize_linked_cell_pairs_batched(
        candidates,
        candidate_valid,
        offsets,
        shifts_idx_unique,
        orig_start,
        self_interaction=self_interaction,
    )
    return mapping, system_mapping, shift_out.to(shift_dtype)


def build_linked_cell_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for the batched linked-cell implementation."""
    if cell.shape[0] == 1:
        return build_linked_cell_neighborhood_serial(
            positions,
            cell,
            pbc,
            cutoff,
            n_atoms,
            self_interaction,
        )
    return build_linked_cell_neighborhood_batched(
        positions,
        cell,
        pbc,
        cutoff,
        n_atoms,
        self_interaction,
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
    dim: int | tuple[int, ...] | list[int] | None = None,
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
    x_high = x.to(high_precision_dtype)
    if dim is None:
        return torch.sum(x_high).to(x.dtype)
    return torch.sum(x_high, dim=dim, keepdim=keepdim).to(x.dtype)


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
    positions: torch.Tensor, cells: torch.Tensor, system_idx: torch.Tensor
) -> torch.Tensor:
    """Vectorized unwrapping for multiple systems without explicit loops.

    Parameters
    ----------
    positions : (T, N_tot, 3)
        Wrapped cartesian positions for all systems concatenated.
    cells : (n_systems, 3, 3) or (T, n_systems, 3, 3)
        Box matrices, constant or time-dependent.
    system_idx : (N_tot,)
        For each atom, which system it belongs to (0..n_systems-1).

    Returns:
    -------
    unwrapped_pos : (T, N_tot, 3)
        Unwrapped cartesian positions.
    """
    # -- Constant boxes per system
    if cells.ndim == 3:
        inv_box = torch.inverse(cells)  # (n_systems, 3, 3)

        # Map each atom to its system's box
        inv_box_atoms = inv_box[system_idx]  # (N, 3, 3)
        box_atoms = cells[system_idx]  # (N, 3, 3)

        # Compute fractional coordinates
        frac = torch.einsum("tni,nij->tnj", positions, inv_box_atoms)

        # Fractional displacements and unwrap
        dfrac = frac[1:] - frac[:-1]
        dfrac -= torch.round(dfrac)

        # Back to Cartesian
        dcart = torch.einsum("tni,nij->tnj", dfrac, box_atoms)

    # -- Time-dependent boxes per system
    elif cells.ndim == 4:
        inv_box = torch.inverse(cells)  # (T, n_systems, 3, 3)

        # Gather each atom's box per frame efficiently
        inv_box_atoms = inv_box[:, system_idx]  # (T, N, 3, 3)
        box_atoms = cells[:, system_idx]  # (T, N, 3, 3)

        # Compute fractional coordinates per frame
        frac = torch.einsum("tni,tnij->tnj", positions, inv_box_atoms)

        dfrac = frac[1:] - frac[:-1]
        dfrac -= torch.round(dfrac)

        # Reconstruct unwrapped fractional trajectory
        unwrapped_frac = torch.empty_like(frac)
        unwrapped_frac[0] = frac[0]
        unwrapped_frac[1:] = torch.cumsum(dfrac, dim=0) + frac[0]

        # Convert back to Cartesian using each frame's cell
        return torch.einsum("tni,tnij->tnj", unwrapped_frac, box_atoms)

    else:
        raise ValueError("box must have shape (n_systems,3,3) or (T,n_systems,3,3)")

    # Cumulative reconstruction (constant cell path)
    unwrapped = torch.empty_like(positions)
    unwrapped[0] = positions[0]
    unwrapped[1:] = torch.cumsum(dcart, dim=0) + unwrapped[0]

    return unwrapped


def get_centers_of_mass(
    positions: torch.Tensor,
    masses: torch.Tensor,
    system_idx: torch.Tensor,
    n_systems: int,
) -> torch.Tensor:
    """Compute the centers of mass for each structure in the simulation state.s.

    Args:
        positions (torch.Tensor): Atomic positions of shape (N, 3).
        masses (torch.Tensor): Atomic masses of shape (N,).
        system_idx (torch.Tensor): System indices for each atom of shape (N,).
        n_systems (int): Total number of systems.

    Returns:
        torch.Tensor: A tensor of shape (n_structures, 3) containing
            the center of mass coordinates for each structure.
    """
    coms = torch.zeros((n_systems, 3), dtype=positions.dtype).scatter_add_(
        0,
        system_idx.unsqueeze(-1).expand(-1, 3),
        masses.unsqueeze(-1) * positions,
    )
    system_masses = torch.zeros((n_systems,), dtype=positions.dtype).scatter_add_(
        0, system_idx, masses
    )
    coms /= system_masses.unsqueeze(-1)
    return coms
