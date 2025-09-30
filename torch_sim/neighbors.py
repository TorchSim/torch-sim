"""Utilities for neighbor list calculations."""

import torch
from vesin import NeighborList as VesinNeighborList
from vesin.torch import NeighborList as VesinNeighborList_ts

import torch_sim.math as tsm
from torch_sim import transforms


@torch.jit.script
def primitive_neighbor_list(  # noqa: C901, PLR0915
    quantities: str,
    pbc: tuple[bool, bool, bool],
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
    use_scaled_positions: bool = False,  # noqa: FBT001, FBT002
    max_n_bins: int = int(1e6),
) -> list[torch.Tensor]:
    """Compute a neighbor list for an atomic configuration.

    ASE periodic neighbor list implementation
    Atoms outside periodic boundaries are mapped into the unit cell. Atoms
    outside non-periodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.
    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Args:
        quantities: Quantities to compute by the neighbor list algorithm. Each character
            in this string defines a quantity. They are returned in a tuple of
            the same order. Possible quantities are
                * 'i' : first atom index
                * 'j' : second atom index
                * 'd' : absolute distance
                * 'D' : distance vector
                * 'S' : shift vector (number of cell boundaries crossed by the bond
                  between atom i and j). With the shift vector S, the
                  distances D between atoms can be computed from:
                  D = positions[j]-positions[i]+S.dot(cell)
        pbc: 3-tuple indicating giving periodic boundaries in the three Cartesian
            directions.
        cell: Unit cell vectors according to the row vector convention, i.e.
            `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        positions: Atomic positions. Anything that can be converted to an ndarray of
            shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
            use_scaled_positions is set to true, this must be scaled positions.
        cutoff: Cutoff for neighbor search. It can be:
            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood.
              See :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
        device: PyTorch device to use for computations
        dtype: PyTorch data type to use
        self_interaction: Return the atom itself as its own neighbor if set to true.
            Default: False
        use_scaled_positions: If set to true, positions are expected to be
            scaled positions.
        max_n_bins: Maximum number of bins used in neighbor search. This is used to limit
            the maximum amount of memory required by the neighbor list.

    Returns:
        list[torch.Tensor]: One tensor for each item in `quantities`. Indices in `i`
            are returned in ascending order 0..len(a)-1, but the order of (i,j)
            pairs is not guaranteed.

    References:
        - This code is modified version of the github gist
        https://gist.github.com/Linux-cpp-lisp/692018c74b3906b63529e60619f5a207
    """
    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    # c: Cartesian index, can have values 0, 1, 2
    # i: Global atom index, can have values 0..len(a)-1
    # xyz: Bin index, three values identifying x-, y- and z-component of a
    #         spatial bin that is used to make neighbor search O(n)
    # b: Linearized version of the 'xyz' bin index
    # a: Bin-local atom index, i.e. index identifying an atom *within* a
    #     bin
    # p: Pair index, can have value 0 or 1
    # n: (Linear) neighbor index

    if len(positions) == 0:
        raise RuntimeError("No atoms provided")

    # Compute reciprocal lattice vectors.
    recip_cell = torch.linalg.pinv(cell).T
    b1_c, b2_c, b3_c = recip_cell[0], recip_cell[1], recip_cell[2]

    # Compute distances of cell faces.
    l1 = torch.linalg.norm(b1_c)
    l2 = torch.linalg.norm(b2_c)
    l3 = torch.linalg.norm(b3_c)
    pytorch_scalar_1 = torch.as_tensor(1.0, device=device, dtype=dtype)
    face_dist_c = torch.hstack(
        [
            1 / l1 if l1 > 0 else pytorch_scalar_1,
            1 / l2 if l2 > 0 else pytorch_scalar_1,
            1 / l3 if l3 > 0 else pytorch_scalar_1,
        ]
    )
    if face_dist_c.shape != (3,):
        raise ValueError(f"face_dist_c.shape={face_dist_c.shape} != (3,)")

    # we don't handle other fancier cutoffs
    max_cutoff: torch.Tensor = cutoff

    # We use a minimum bin size of 3 A
    bin_size = torch.maximum(max_cutoff, torch.tensor(3.0, device=device, dtype=dtype))
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    n_bins_c = torch.maximum(
        (face_dist_c / bin_size).to(dtype=torch.long, device=device),
        torch.ones(3, dtype=torch.long, device=device),
    )
    n_bins = torch.prod(n_bins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while n_bins > max_n_bins:
        n_bins_c = torch.maximum(
            n_bins_c // 2, torch.ones(3, dtype=torch.long, device=device)
        )
        n_bins = torch.prod(n_bins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search = torch.ceil(bin_size * n_bins_c / face_dist_c).to(
        dtype=torch.long, device=device
    )
    neigh_search_x, neigh_search_y, neigh_search_z = (
        neigh_search[0],
        neigh_search[1],
        neigh_search[2],
    )

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    pytorch_scalar_int_0 = torch.as_tensor(0, dtype=torch.long, device=device)
    neigh_search_x = (
        pytorch_scalar_int_0 if n_bins_c[0] == 1 and not pbc[0] else neigh_search_x
    )
    neigh_search_y = (
        pytorch_scalar_int_0 if n_bins_c[1] == 1 and not pbc[1] else neigh_search_y
    )
    neigh_search_z = (
        pytorch_scalar_int_0 if n_bins_c[2] == 1 and not pbc[2] else neigh_search_z
    )

    # Sort atoms into bins.
    if not any(pbc):
        scaled_positions_ic = positions
    elif use_scaled_positions:
        scaled_positions_ic = positions
        positions = torch.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = torch.linalg.solve(cell.T, positions.T).T

    bin_index_ic = torch.floor(scaled_positions_ic * n_bins_c).to(
        dtype=torch.long, device=device
    )
    cell_shift_ic = torch.zeros_like(bin_index_ic, device=device)

    for c in range(3):
        if pbc[c]:
            # (Note: torch.divmod does not exist in older numpy versions)
            cell_shift_ic[:, c], bin_index_ic[:, c] = tsm.torch_divmod(
                bin_index_ic[:, c], n_bins_c[c]
            )
        else:
            bin_index_ic[:, c] = torch.clip(bin_index_ic[:, c], 0, n_bins_c[c] - 1)  # type: ignore[call-overload]

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = bin_index_ic[:, 0] + n_bins_c[0] * (
        bin_index_ic[:, 1] + n_bins_c[1] * bin_index_ic[:, 2]
    )

    # atom_i contains atom index in new sort order.
    atom_i = torch.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_n_atoms_per_bin = torch.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_n_atoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -torch.ones(
        n_bins, max_n_atoms_per_bin, dtype=torch.long, device=device
    )  # type: ignore[call-overload]
    for bin_cnt in range(int(max_n_atoms_per_bin.item())):
        # Create a mask array that identifies the first atom of each bin.
        mask = torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=device),
                bin_index_i[:-1] != bin_index_i[1:],
            ),
            dim=0,
        )
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], bin_cnt] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = torch.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    if len(atom_i) != 0:
        raise ValueError(f"len(atom_i)={len(atom_i)} != 0")
    if len(bin_index_i) != 0:
        raise ValueError(f"len(bin_index_i)={len(bin_index_i)} != 0")

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_n_atoms_per_bin**2.
    # atom_pairs_pn_np = np.indices(
    #     (max_n_atoms_per_bin, max_n_atoms_per_bin), dtype=int
    # ).reshape(2, -1)
    atom_pairs_pn = torch.cartesian_prod(
        torch.arange(max_n_atoms_per_bin, device=device),  # type: ignore[call-overload]
        torch.arange(max_n_atoms_per_bin, device=device),  # type: ignore[call-overload]
    )
    atom_pairs_pn = atom_pairs_pn.T.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neigh_tuple_nn = []
    second_at_neigh_tuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_n_atoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = torch.meshgrid(
        torch.arange(n_bins_c[2], device=device),  # type: ignore[call-overload]
        torch.arange(n_bins_c[1], device=device),  # type: ignore[call-overload]
        torch.arange(n_bins_c[0], device=device),  # type: ignore[call-overload]
        indexing="ij",
    )
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + n_bins_c[0] * (biny_xyz + n_bins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == torch.arange(torch.prod(n_bins_c))).all()

    # First atoms in pair.
    _first_at_neigh_tuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-int(neigh_search_z.item()), int(neigh_search_z.item()) + 1):
        for dy in range(-int(neigh_search_y.item()), int(neigh_search_y.item()) + 1):
            for dx in range(-int(neigh_search_x.item()), int(neigh_search_x.item()) + 1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = tsm.torch_divmod(binx_xyz + dx, n_bins_c[0])
                shifty_xyz, neighbiny_xyz = tsm.torch_divmod(biny_xyz + dy, n_bins_c[1])
                shiftz_xyz, neighbinz_xyz = tsm.torch_divmod(binz_xyz + dz, n_bins_c[2])
                neighbin_b = (
                    neighbinx_xyz
                    + n_bins_c[0] * (neighbiny_xyz + n_bins_c[1] * neighbinz_xyz)
                ).ravel()

                # Second atom in pair.
                _second_at_neigh_tuple_n = atoms_in_bin_ba[neighbin_b][
                    :, atom_pairs_pn[1]
                ]

                # Shift vectors.
                # TODO: was np.resize:
                # _cell_shift_vector_x_n_np = np.resize(
                #     shiftx_xyz.reshape(-1, 1).numpy(),
                #     (int(max_n_atoms_per_bin.item() ** 2), shiftx_xyz.numel()),
                # ).T
                # _cell_shift_vector_y_n_np = np.resize(
                #     shifty_xyz.reshape(-1, 1).numpy(),
                #     (int(max_n_atoms_per_bin.item() ** 2), shifty_xyz.numel()),
                # ).T
                # _cell_shift_vector_z_n_np = np.resize(
                #     shiftz_xyz.reshape(-1, 1).numpy(),
                #     (int(max_n_atoms_per_bin.item() ** 2), shiftz_xyz.numel()),
                # ).T
                # this basically just tiles shiftx_xyz.reshape(-1, 1) n times
                _cell_shift_vector_x_n = shiftx_xyz.reshape(-1, 1).repeat(
                    (1, int(max_n_atoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_x_n.shape == _cell_shift_vector_x_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_x_n.numpy(), _cell_shift_vector_x_n_np
                # )
                _cell_shift_vector_y_n = shifty_xyz.reshape(-1, 1).repeat(
                    (1, int(max_n_atoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_y_n.shape == _cell_shift_vector_y_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_y_n.numpy(), _cell_shift_vector_y_n_np
                # )
                _cell_shift_vector_z_n = shiftz_xyz.reshape(-1, 1).repeat(
                    (1, int(max_n_atoms_per_bin.item() ** 2))
                )
                # assert _cell_shift_vector_z_n.shape == _cell_shift_vector_z_n_np.shape
                # assert np.allclose(
                #     _cell_shift_vector_z_n.numpy(), _cell_shift_vector_z_n_np
                # )

                # We have created too many pairs because we assumed each bin
                # has exactly max_n_atoms_per_bin atoms. Remove all superfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = torch.logical_and(
                    _first_at_neigh_tuple_n != -1, _second_at_neigh_tuple_n != -1
                )
                if mask.sum() > 0:
                    first_at_neigh_tuple_nn += [_first_at_neigh_tuple_n[mask]]
                    second_at_neigh_tuple_nn += [_second_at_neigh_tuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neigh_tuple_n = torch.cat(first_at_neigh_tuple_nn)
    second_at_neigh_tuple_n = torch.cat(second_at_neigh_tuple_nn)
    cell_shift_vector_n = torch.vstack(
        [
            torch.cat(cell_shift_vector_x_n),
            torch.cat(cell_shift_vector_y_n),
            torch.cat(cell_shift_vector_z_n),
        ]
    ).T

    # Add global cell shift to shift vectors
    cell_shift_vector_n += (
        cell_shift_ic[first_at_neigh_tuple_n] - cell_shift_ic[second_at_neigh_tuple_n]
    )

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = torch.logical_not(
            torch.logical_and(
                first_at_neigh_tuple_n == second_at_neigh_tuple_n,
                (cell_shift_vector_n == 0).all(dim=1),
            )
        )
        first_at_neigh_tuple_n = first_at_neigh_tuple_n[m]
        second_at_neigh_tuple_n = second_at_neigh_tuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For non-periodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neigh_tuple_n = first_at_neigh_tuple_n[m]
            second_at_neigh_tuple_n = second_at_neigh_tuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    bin_cnt_sort_idx = torch.argsort(first_at_neigh_tuple_n)
    first_at_neigh_tuple_n = first_at_neigh_tuple_n[bin_cnt_sort_idx]
    second_at_neigh_tuple_n = second_at_neigh_tuple_n[bin_cnt_sort_idx]
    cell_shift_vector_n = cell_shift_vector_n[bin_cnt_sort_idx]

    # Compute distance vectors.
    # TODO: Use .T?
    distance_vector_nc = (
        positions[second_at_neigh_tuple_n]
        - positions[first_at_neigh_tuple_n]
        + cell_shift_vector_n.to(cell.dtype).matmul(cell)
    )
    abs_distance_vector_n = torch.sqrt(
        torch.sum(distance_vector_nc * distance_vector_nc, dim=1)
    )

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neigh_tuple_n = first_at_neigh_tuple_n[mask]
    second_at_neigh_tuple_n = second_at_neigh_tuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    ret_vals = []
    for quant in quantities:
        if quant == "i":
            ret_vals += [first_at_neigh_tuple_n]
        elif quant == "j":
            ret_vals += [second_at_neigh_tuple_n]
        elif quant == "D":
            ret_vals += [distance_vector_nc]
        elif quant == "d":
            ret_vals += [abs_distance_vector_n]
        elif quant == "S":
            ret_vals += [cell_shift_vector_n]
        else:
            raise ValueError("Unsupported quantity specified.")

    return ret_vals


@torch.jit.script
def standard_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,  # noqa: FBT001
    cutoff: torch.Tensor,
    sort_id: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor lists using primitive neighbor list algorithm.

    This function provides a standardized interface for computing neighbor lists
    in atomic systems, wrapping the more general primitive_neighbor_list implementation.
    It handles both periodic and non-periodic boundary conditions and returns
    neighbor pairs along with their periodic shifts.

    The function follows ASE's neighbor list conventions (see ASE:
    https://gitlab.com/ase/ase/-/blob/master/ase/neighborlist.py?ref_type=heads#L152
    but provides a simplified interface focused on the most common use case of
    getting neighbor pairs and shifts.

    Key Features:
    - Handles both periodic and non-periodic systems
    - Returns both neighbor indices and shift vectors for periodic systems
    - Optional sorting of neighbors by first index for better memory access patterns
    - Fully compatible with PyTorch's automatic differentiation

    Args:
        positions: Atomic positions tensor of shape (num_atoms, 3)
        cell: Unit cell vectors according to the row vector convention, i.e.
            `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc: Whether to use periodic boundary conditions (applied to all directions)
        cutoff: Maximum distance for considering atoms as neighbors
        sort_id: If True, sort neighbors by first atom index for better memory
            access patterns

    Returns:
        tuple containing:
            - mapping: Tensor of shape (2, num_neighbors) containing pairs of
              atom indices that are neighbors. Each column (i,j) represents a
              neighbor pair.
            - shifts: Tensor of shape (num_neighbors, 3) containing the periodic
              shift vectors needed to get the correct periodic image for each
              neighbor pair.

    Example:
        >>> # Get neighbors for a periodic system
        >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> cell = torch.eye(3) * 10.0
        >>> mapping, shifts = standard_nl(positions, cell, True, 1.5)
        >>> print(mapping)  # Shows pairs of neighboring atoms
        >>> print(shifts)  # Shows corresponding periodic shifts

    Notes:
        - The function uses primitive_neighbor_list internally but provides a simpler
          interface
        - For non-periodic systems (pbc=False), shifts will be zero vectors
        - The neighbor list includes both (i,j) and (j,i) pairs for complete force
          computation
        - Memory usage scales with system size and number of neighbors per atom

    References:
        - https://gist.github.com/Linux-cpp-lisp/692018c74b3906b63529e60619f5a207
    """
    device = positions.device
    dtype = positions.dtype
    i, j, S = primitive_neighbor_list(
        quantities="ijS",
        positions=positions,
        cell=cell,
        pbc=(pbc, pbc, pbc),
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        self_interaction=False,
        use_scaled_positions=False,
        max_n_bins=torch.tensor(1e6, dtype=torch.int64, device=device),
    )

    mapping = torch.stack((i, j), dim=0)
    mapping = mapping.to(dtype=torch.long)
    shifts = S.to(dtype=dtype)

    if sort_id:
        idx = torch.argsort(mapping[0])
        mapping = mapping[:, idx]
        shifts = shifts[idx, :]

    return mapping, shifts


@torch.jit.script
def vesin_nl_ts(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,  # noqa: FBT001
    cutoff: torch.Tensor,
    sort_id: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor lists using TorchScript-compatible Vesin implementation.

    This function provides a TorchScript-compatible interface to the Vesin neighbor
    list algorithm using VesinNeighborList_ts. It handles both periodic and non-periodic
    systems and returns neighbor pairs along with their periodic shifts.

    Args:
        positions: Atomic positions tensor of shape (num_atoms, 3)
        cell: Unit cell vectors according to the row vector convention, i.e.
            `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc: Whether to use periodic boundary conditions (applied to all directions)
        cutoff: Maximum distance (scalar tensor) for considering atoms as neighbors
        sort_id: If True, sort neighbors by first atom index for better memory
            access patterns

    Returns:
        tuple containing:
            - mapping: Tensor of shape (2, num_neighbors) containing pairs of
              atom indices that are neighbors. Each column (i,j) represents a
              neighbor pair.
            - shifts: Tensor of shape (num_neighbors, 3) containing the periodic
              shift vectors needed to get the correct periodic image for each
              neighbor pair.

    Notes:
        - Uses VesinNeighborList_ts for TorchScript compatibility
        - Requires CPU tensors in float64 precision internally
        - Returns tensors on the same device as input with original precision
        - For non-periodic systems (pbc=False), shifts will be zero vectors
        - The neighbor list includes both (i,j) and (j,i) pairs

    References:
          https://github.com/Luthaf/vesin
    """
    device = positions.device
    dtype = positions.dtype

    neighbor_list_fn = VesinNeighborList_ts(cutoff.item(), full_list=True)

    # Convert tensors to CPU and float64 properly
    positions_cpu = positions.cpu().to(dtype=torch.float64)
    cell_cpu = cell.cpu().to(dtype=torch.float64)

    # Only works on CPU and requires float64
    i, j, S = neighbor_list_fn.compute(
        points=positions_cpu,
        box=cell_cpu,
        periodic=pbc,
        quantities="ijS",
    )

    mapping = torch.stack((i, j), dim=0)
    mapping = mapping.to(dtype=torch.long, device=device)
    shifts = S.to(dtype=dtype, device=device)

    if sort_id:
        idx = torch.argsort(mapping[0])
        mapping = mapping[:, idx]
        shifts = shifts[idx, :]

    return mapping, shifts


def vesin_nl(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    cutoff: torch.Tensor,
    sort_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor lists using the standard Vesin implementation.

    This function provides an interface to the standard Vesin neighbor list
    algorithm using VesinNeighborList. It handles both periodic and non-periodic
    systems and returns neighbor pairs along with their periodic shifts.

    Args:
        positions: Atomic positions tensor of shape (num_atoms, 3)
        cell: Unit cell vectors according to the row vector convention, i.e.
            `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc: Whether to use periodic boundary conditions (applied to all directions)
        cutoff: Maximum distance (scalar tensor) for considering atoms as neighbors
        sort_id: If True, sort neighbors by first atom index for better memory
            access patterns

    Returns:
        tuple containing:
            - mapping: Tensor of shape (2, num_neighbors) containing pairs of
              atom indices that are neighbors. Each column (i,j) represents a
              neighbor pair.
            - shifts: Tensor of shape (num_neighbors, 3) containing the periodic
              shift vectors needed to get the correct periodic image for each
              neighbor pair.

    Notes:
        - Uses standard VesinNeighborList implementation
        - Requires CPU tensors in float64 precision internally
        - Returns tensors on the same device as input with original precision
        - For non-periodic systems (pbc=False), shifts will be zero vectors
        - The neighbor list includes both (i,j) and (j,i) pairs
        - Supports pre-sorting through the VesinNeighborList constructor

    References:
        - https://github.com/Luthaf/vesin
    """
    device = positions.device
    dtype = positions.dtype

    neighbor_list_fn = VesinNeighborList(cutoff, full_list=True, sorted=sort_id)

    # Convert tensors to CPU and float64 without gradients
    positions_cpu = positions.detach().cpu().to(dtype=torch.float64)
    cell_cpu = cell.detach().cpu().to(dtype=torch.float64)

    # Only works on CPU and returns numpy arrays
    i, j, S = neighbor_list_fn.compute(
        points=positions_cpu,
        box=cell_cpu,
        periodic=pbc,
        quantities="ijS",
    )
    i, j = (
        torch.tensor(i, dtype=torch.long, device=device),
        torch.tensor(j, dtype=torch.long, device=device),
    )
    mapping = torch.stack((i, j), dim=0)
    shifts = torch.tensor(S, dtype=dtype, device=device)

    return mapping, shifts


def strict_nl(
    cutoff: float,
    positions: torch.Tensor,
    cell: torch.Tensor | None,
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
        cell (torch.Tensor | None): Unit cell vectors according to the row vector
            convention. i.e. `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
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
    if cell is None:
        d2 = (positions[mapping[0]] - positions[mapping[1]]).square().sum(dim=1)
    else:
        cell_shifts = transforms.compute_cell_shifts_strict(
            cell, shifts_idx, system_mapping
        )
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
    cutoff: float,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    system_idx: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a set of atomic structures using a
    naive neighbor search before applying a strict `cutoff`.
    The atomic positions `pos` should be wrapped inside their respective unit cells.

    Args:
        cutoff (float):
            The cutoff radius used for the neighbor search.
        positions (torch.Tensor [n_atom, 3]): A tensor containing the positions
            of atoms wrapped inside their respective unit cells.
        cell (torch.Tensor [3*n_structure, 3]): Unit cell vectors according to
            the row vector convention, i.e. `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc (torch.Tensor [n_structure, 3] bool):
            A tensor indicating the periodic boundary conditions to apply.
            Partial PBC are not supported yet.
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

    References:
        - https://github.com/felixmusil/torch_nl
    """
    n_atoms = torch.bincount(system_idx)
    mapping, system_mapping, shifts_idx = transforms.build_naive_neighborhood(
        positions, cell, pbc, cutoff, n_atoms, self_interaction
    )
    mapping, mapping_system, shifts_idx = strict_nl(
        cutoff, positions, cell, mapping, system_mapping, shifts_idx
    )
    return mapping, mapping_system, shifts_idx


@torch.jit.script
def torch_nl_linked_cell(
    cutoff: float,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    system_idx: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a set of atomic structures using the linked
    cell algorithm before applying a strict `cutoff`. The atoms positions `pos`
    should be wrapped inside their respective unit cells.

    Args:
        cutoff (float):
            The cutoff radius used for the neighbor search.
        positions (torch.Tensor [n_atom, 3]):
            A tensor containing the positions of atoms wrapped inside
            their respective unit cells.
        cell (torch.Tensor [3*n_structure, 3]): Unit cell vectors according to
            the row vector convention, i.e. `[[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]`.
        pbc (torch.Tensor [n_structure, 3] bool):
            A tensor indicating the periodic boundary conditions to apply.
            Partial PBC are not supported yet.
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

    References:
        - https://github.com/felixmusil/torch_nl
    """
    n_atoms = torch.bincount(system_idx)
    mapping, system_mapping, shifts_idx = transforms.build_linked_cell_neighborhood(
        positions, cell, pbc, cutoff, n_atoms, self_interaction
    )

    mapping, mapping_system, shifts_idx = strict_nl(
        cutoff, positions, cell, mapping, system_mapping, shifts_idx
    )
    return mapping, mapping_system, shifts_idx
