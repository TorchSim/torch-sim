"""Pure-PyTorch triplet and quadruplet interaction index builders.

All functions use only standard PyTorch ops (argsort, bincount, repeat_interleave,
boolean masking, etc.) and are compatible with ``torch.jit.script``.
No ``torch_scatter`` or ``torch_sparse`` dependencies.

Typical usage::

    from torch_sim.neighbors import torchsim_nl
    from torch_sim.neighbors.nbody import build_triplets, build_quadruplets

    mapping, system_mapping, shifts_idx = torchsim_nl(
        positions, cell, pbc, cutoff, system_idx
    )
    trip = build_triplets(mapping, n_atoms)
    # trip["trip_in"], trip["trip_out"] index into edges
"""

from __future__ import annotations

import torch


def _inner_idx(sorted_idx: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Local enumeration within sorted contiguous segments.

    For a sorted index tensor ``[0,0,0,1,1,2,2,2,2]`` returns ``[0,1,2,0,1,0,1,2,3]``.

    Args:
        sorted_idx: 1-D tensor of segment ids, **must be sorted**.
        dim_size: Total number of segments (>= max(sorted_idx)+1).

    Returns:
        1-D tensor same length as *sorted_idx* with per-segment local indices.
    """
    counts = torch.bincount(sorted_idx, minlength=dim_size)
    offsets = counts.cumsum(0) - counts
    return (
        torch.arange(sorted_idx.size(0), device=sorted_idx.device) - offsets[sorted_idx]
    )


def build_triplets(
    edge_index: torch.Tensor,
    n_atoms: int,
) -> dict[str, torch.Tensor]:
    """Build triplet interaction indices from an edge list.

    For every pair of edges ``(b→a)`` and ``(c→a)`` that share the same target
    atom ``a`` with ``edge_ba ≠ edge_ca``, produces a triplet ``b→a←c``.

    Uses only ops that are JIT/AOTInductor safe: ``argsort``, ``bincount``,
    ``repeat_interleave``, and boolean indexing.

    Args:
        edge_index: ``[2, n_edges]`` tensor where ``edge_index[0]`` are sources
            and ``edge_index[1]`` are targets.
        n_atoms: Total number of atoms (used for bincount sizing).

    Returns:
        Dict with keys:

        - ``"trip_in"`` — edge indices of the *incoming* edge ``b→a``, shape
          ``[n_triplets]``.
        - ``"trip_out"`` — edge indices of the *outgoing* edge ``c→a``, shape
          ``[n_triplets]``.
        - ``"trip_out_agg"`` — per-segment local index for aggregation, shape
          ``[n_triplets]``.
        - ``"center_atom"`` — atom index ``a`` for each triplet, shape
          ``[n_triplets]``.
    """
    targets = edge_index[1]  # target atoms
    n_edges = targets.size(0)
    device = targets.device

    # Sort edges by target atom to get contiguous groups
    order = torch.argsort(targets, stable=True)
    sorted_targets = targets[order]

    # Degree per atom and CSR-style offsets
    deg = torch.bincount(sorted_targets, minlength=n_atoms)
    offsets = torch.zeros(n_atoms + 1, dtype=torch.long, device=device)
    offsets[1:] = deg.cumsum(0)

    # Number of ordered triplets per atom: deg*(deg-1)
    n_trip_per_atom = deg * (deg - 1)
    total_triplets = int(n_trip_per_atom.sum().item())

    if total_triplets == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return {
            "trip_in": empty,
            "trip_out": empty,
            "trip_out_agg": empty,
            "center_atom": empty,
        }

    # Atom ids that have at least 2 edges
    active = deg >= 2
    active_atoms = torch.where(active)[0]
    active_deg = deg[active]
    active_offsets = offsets[:-1][active]
    active_n_trip = n_trip_per_atom[active]

    # Expand: for each active atom, enumerate deg*(deg-1) triplets
    atom_rep = torch.repeat_interleave(
        torch.arange(active_atoms.size(0), device=device), active_n_trip
    )
    base_off = torch.repeat_interleave(active_offsets, active_n_trip)
    d = torch.repeat_interleave(active_deg, active_n_trip)

    # Local triplet index within each atom's group
    local = _inner_idx(atom_rep, active_atoms.size(0))

    # Map local index to (row, col) within the deg x (deg-1) grid
    # row = local // (deg-1),  col = local % (deg-1)
    dm1 = d - 1
    row = local // dm1
    col = local % dm1
    # Skip diagonal: if col >= row, shift col by 1
    col = col + (col >= row).long()

    trip_in = order[base_off + row]
    trip_out = order[base_off + col]

    # Center atom for each triplet
    center = torch.repeat_interleave(active_atoms, active_n_trip)

    # Aggregation index: local enumeration by trip_out
    trip_out_agg = _inner_idx(trip_out, n_edges) if total_triplets > 0 else trip_out

    return {
        "trip_in": trip_in,
        "trip_out": trip_out,
        "trip_out_agg": trip_out_agg,
        "center_atom": center,
    }


def build_mixed_triplets(
    edge_index_in: torch.Tensor,
    edge_index_out: torch.Tensor,
    n_atoms: int,
    to_outedge: bool = False,  # noqa: FBT001, FBT002
    cell_offsets_in: torch.Tensor | None = None,
    cell_offsets_out: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build triplet indices across two different edge sets sharing the same atoms.

    For each edge in ``edge_index_out``, finds all edges in ``edge_index_in``
    that share the same atom (target or source depending on *to_outedge*),
    filtering self-loops via cell offsets when provided.

    This is the pure-PyTorch equivalent of GemNet-OC ``get_mixed_triplets``.

    Args:
        edge_index_in: ``[2, n_edges_in]`` — input graph edges.
        edge_index_out: ``[2, n_edges_out]`` — output graph edges.
        n_atoms: Total number of atoms.
        to_outedge: If True, match on the *source* atom of out-edges (``a→c``
            style); otherwise match on the *target* atom (``c→a`` style).
        cell_offsets_in: ``[n_edges_in, 3]`` periodic offsets for input graph.
        cell_offsets_out: ``[n_edges_out, 3]`` periodic offsets for output graph.

    Returns:
        Dict with keys ``"trip_in"``, ``"trip_out"``, ``"trip_out_agg"``.
    """
    src_in, tgt_in = edge_index_in[0], edge_index_in[1]
    src_out, tgt_out = edge_index_out[0], edge_index_out[1]
    n_edges_out = src_out.size(0)
    device = src_in.device

    # Build CSR of input edges grouped by target atom
    order_in = torch.argsort(tgt_in, stable=True)
    sorted_tgt_in = tgt_in[order_in]
    deg_in = torch.bincount(sorted_tgt_in, minlength=n_atoms)
    csr_in = torch.zeros(n_atoms + 1, dtype=torch.long, device=device)
    csr_in[1:] = deg_in.cumsum(0)

    # For each output edge, pick the shared atom
    shared_atom = src_out if to_outedge else tgt_out

    # Degree of each output edge's shared atom in the input graph
    deg_per_out = deg_in[shared_atom]  # [n_edges_out]

    # Expand: repeat each output edge index by degree of its shared atom
    trip_out = torch.repeat_interleave(
        torch.arange(n_edges_out, device=device), deg_per_out
    )
    # For each expanded entry, the corresponding input edge
    base_off = csr_in[shared_atom]  # start offset into sorted input edges
    base_off_exp = torch.repeat_interleave(base_off, deg_per_out)

    # Local index within the group
    local = _inner_idx(trip_out, n_edges_out)
    trip_in = order_in[base_off_exp + local]

    # Filter self-loops: atom-index check + cell offset check
    if to_outedge:
        idx_atom_in = src_in[trip_in]
        idx_atom_out = tgt_out[trip_out]
    else:
        idx_atom_in = src_in[trip_in]
        idx_atom_out = src_out[trip_out]

    mask = idx_atom_in != idx_atom_out
    if cell_offsets_in is not None and cell_offsets_out is not None:
        if to_outedge:
            cell_sum = cell_offsets_out[trip_out] + cell_offsets_in[trip_in]
        else:
            cell_sum = cell_offsets_out[trip_out] - cell_offsets_in[trip_in]
        mask = mask | torch.any(cell_sum != 0, dim=-1)

    trip_in = trip_in[mask]
    trip_out = trip_out[mask]

    trip_out_agg = _inner_idx(trip_out, n_edges_out)

    return {
        "trip_in": trip_in,
        "trip_out": trip_out,
        "trip_out_agg": trip_out_agg,
    }


def build_quadruplets(
    main_edge_index: torch.Tensor,
    qint_edge_index: torch.Tensor,
    n_atoms: int,
    main_cell_offsets: torch.Tensor,
    qint_cell_offsets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build quadruplet interaction indices ``d→b→a←c`` from two edge sets.

    Combines an input triplet set ``d→b→a`` (edges from ``main_graph``
    arriving at ``b`` for each ``qint_graph`` edge ``b→a``) with an output
    triplet set ``c→a←b`` (``qint_graph`` edges arriving at ``a`` for each
    ``main_graph`` edge ``c→a``), then takes their Cartesian product over the
    shared intermediate edge ``b→a``.

    Pure-PyTorch equivalent of GemNet-OC ``get_quadruplets``.

    Args:
        main_edge_index: ``[2, n_main_edges]`` — main graph edges.
        qint_edge_index: ``[2, n_qint_edges]`` — quadruplet interaction graph edges.
        n_atoms: Total number of atoms.
        main_cell_offsets: ``[n_main_edges, 3]`` cell offsets for main graph.
        qint_cell_offsets: ``[n_qint_edges, 3]`` cell offsets for qint graph.

    Returns:
        Dict with keys (for quadruplet ``d→b→a←c``):

        - ``"main_edge_d_to_b"`` — main graph edge indices for ``d→b`` edges,
          shape ``[n_trip_in]``. Indices into ``main_edge_index``.
        - ``"qint_edge_b_to_a"`` — qint graph edge indices for ``b→a`` edges
          (intermediate), shape ``[n_trip_in]``. Indices into ``qint_edge_index``.
        - ``"qint_edge_b_to_a_agg"`` — aggregation indices for ``b→a`` edges,
          shape ``[n_trip_in]``.
        - ``"main_edge_c_to_a"`` — main graph edge indices for ``c→a`` edges,
          shape ``[n_trip_out]``. Indices into ``main_edge_index``.
        - ``"main_edge_c_to_a_agg"`` — aggregation indices for ``c→a`` edges,
          shape ``[n_trip_out]``.
        - ``"quad_main_edge_c_to_a"`` — main graph edge indices for ``c→a`` edges
          for each quadruplet, shape ``[n_quads]``. Indices into
          ``main_edge_index``.
        - ``"trip_in_to_quad"`` — maps input-triplet index → quadruplet index,
          shape ``[n_quads]``.
        - ``"trip_out_to_quad"`` — maps output-triplet index → quadruplet index,
          shape ``[n_quads]``.
        - ``"quad_main_edge_agg"`` — per-segment local index for aggregation,
          shape ``[n_quads]``.
    """
    src_main = main_edge_index[0]
    n_main_edges = src_main.size(0)
    n_qint_edges = qint_edge_index.size(1)
    device = src_main.device

    # Input triplets: d→b→a  (input=main, output=qint, to_outedge=True)
    triplet_in = build_mixed_triplets(
        main_edge_index,
        qint_edge_index,
        n_atoms,
        to_outedge=True,
        cell_offsets_in=main_cell_offsets,
        cell_offsets_out=qint_cell_offsets,
    )

    # Output triplets: c→a←b  (input=qint, output=main, to_outedge=False)
    triplet_out = build_mixed_triplets(
        qint_edge_index,
        main_edge_index,
        n_atoms,
        to_outedge=False,
        cell_offsets_in=qint_cell_offsets,
        cell_offsets_out=main_cell_offsets,
    )

    # Count input triplets per intermediate (qint) edge
    ones_in = torch.ones_like(triplet_in["trip_out"])
    n_trip_in_per_inter = torch.zeros(n_qint_edges, dtype=torch.long, device=device)
    n_trip_in_per_inter.index_add_(0, triplet_in["trip_out"], ones_in)

    # Build CSR of input triplets grouped by intermediate (qint) edge.
    # Sort input triplets by qint edge so CSR lookup is contiguous.
    order_ti = torch.argsort(triplet_in["trip_out"], stable=True)
    sorted_trip_in_by_inter = triplet_in["trip_in"][order_ti]

    csr_ti = torch.zeros(n_qint_edges + 1, dtype=torch.long, device=device)
    csr_ti[1:] = n_trip_in_per_inter.cumsum(0)

    # For each output triplet, count how many input triplets share its intermediate edge.
    # Only output triplets with ≥1 match can form quadruplets.
    n_in_for_out = n_trip_in_per_inter[triplet_out["trip_in"]]
    valid_out = n_in_for_out > 0
    trip_out_main = triplet_out["trip_out"][valid_out]  # c→a main edge indices
    trip_out_inter = triplet_out["trip_in"][valid_out]  # b→a qint edge indices
    n_in_for_valid = n_in_for_out[valid_out]

    # Cartesian product: each valid output triplet paired with each of its input triplets.
    quad_out = torch.repeat_interleave(trip_out_main, n_in_for_valid)
    inter_edge = torch.repeat_interleave(trip_out_inter, n_in_for_valid)
    trip_out_to_quad = torch.repeat_interleave(
        torch.arange(trip_out_main.size(0), device=device), n_in_for_valid
    )

    # Local index cycling 0..n_in[e]-1 within each output-triplet block.
    # cumsum gives the start of each block in the expanded array; subtracting it
    # from the global position gives the within-block offset.
    n_quads_pre = int(n_in_for_valid.sum().item())
    cum_starts = torch.zeros(n_quads_pre, dtype=torch.long, device=device)
    if trip_out_main.size(0) > 0:
        starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                n_in_for_valid.cumsum(0)[:-1],
            ]
        )
        cum_starts = torch.repeat_interleave(starts, n_in_for_valid)
    local = torch.arange(n_quads_pre, dtype=torch.long, device=device) - cum_starts

    base = csr_ti[inter_edge]
    ti_idx = base + local
    d_to_b_edge = sorted_trip_in_by_inter[ti_idx]

    # Filter: c ≠ d (with cell offsets)
    idx_atom_c = src_main[quad_out]
    idx_atom_d = src_main[d_to_b_edge]

    cell_offset_cd = (
        main_cell_offsets[d_to_b_edge]
        + qint_cell_offsets[inter_edge]
        - main_cell_offsets[quad_out]
    )
    mask = (idx_atom_c != idx_atom_d) | torch.any(cell_offset_cd != 0, dim=-1)

    quad_out = quad_out[mask]
    trip_out_to_quad = trip_out_to_quad[mask]
    trip_in_to_quad = order_ti[ti_idx[mask]]

    quad_out_agg = _inner_idx(quad_out, n_main_edges)

    return {
        # d→b edge indices into main_edge_index
        "main_edge_d_to_b": triplet_in["trip_in"],
        # b→a edge indices into qint_edge_index (intermediate)
        "qint_edge_b_to_a": triplet_in["trip_out"],
        # aggregation index for b→a edges
        "qint_edge_b_to_a_agg": triplet_in["trip_out_agg"],
        # c→a edge indices into main_edge_index
        "main_edge_c_to_a": triplet_out["trip_out"],
        # aggregation index for c→a edges
        "main_edge_c_to_a_agg": triplet_out["trip_out_agg"],
        # c→a edge indices for each quadruplet
        "quad_main_edge_c_to_a": quad_out,
        # maps input-triplet index → quadruplet index
        "trip_in_to_quad": trip_in_to_quad,
        # maps output-triplet index → quadruplet index
        "trip_out_to_quad": trip_out_to_quad,
        # per-segment local index for aggregation
        "quad_main_edge_agg": quad_out_agg,
    }
