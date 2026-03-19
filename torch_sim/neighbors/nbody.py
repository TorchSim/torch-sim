"""Pure-PyTorch triplet and quadruplet interaction index builders.

Uses only standard PyTorch ops (argsort, bincount, repeat_interleave, boolean
masking) and is compatible with ``torch.jit.script``. No ``torch_scatter`` or
``torch_sparse`` dependencies.

``build_triplets`` finds every ordered pair of edges ``(b→a, c→a)`` sharing a
target atom ``a`` — the angle environment used by three-body potentials (Tersoff,
SW) and message-passing networks (DimeNet).

``build_mixed_triplets`` does the same across two *different* edge sets (different
cutoffs or connectivity rules). Used internally by ``build_quadruplets`` and
directly for architectures with separate embedding and interaction graphs.

``build_quadruplets`` builds four-body interactions ``d→b→a←c`` from two neighbour
lists at different cutoffs. The *central* bond ``b→a`` comes from the "internal"
graph (shorter cutoff), while the *outer* bonds ``d→b`` and ``c→a`` come from the
**main** graph (longer cutoff)::

    d ——(main, long)——> b ===(internal, short)===> a <——(main, long)—— c

For each short central bond, all long-range neighbours of its endpoints are paired
(excluding ``c == d`` in the same image). This biases the model toward interactions
where the central bond is strongest, which is the opposite of a uniform-cutoff
torsion. Pure-PyTorch equivalent of GemNet-OC ``get_quadruplets``::

    mapping, _, shifts = torch_nl_linked_cell(pos, cell, pbc, tensor(5.0), sys_idx)
    qmapping, _, qshifts = torch_nl_linked_cell(pos, cell, pbc, tensor(3.0), sys_idx)
    trip = build_triplets(mapping, n_atoms)
    quad = build_quadruplets(mapping, qmapping, n_atoms, shifts.float(), qshifts.float())
    # quad["quad_c_to_a_edge"]      — c→a main-edge index per quadruplet
    # quad["quad_d_to_b_trip_idx"]  — index into d_to_b_edge/b_to_a_edge per quadruplet
    # quad["quad_c_to_a_trip_idx"]  — index into c_to_a_edge per quadruplet
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
    internal_edge_index: torch.Tensor,
    n_atoms: int,
    main_cell_offsets: torch.Tensor,
    internal_cell_offsets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build quadruplet interaction indices ``d→b→a←c`` from two edge sets.

    For each internal (short-cutoff) bond ``b→a``, pairs every main-graph
    neighbour ``d`` of ``b`` with every main-graph neighbour ``c`` of ``a``,
    excluding ``c == d`` in the same periodic image.  The resulting four-atom
    chains have a short central bond flanked by longer outer bonds::

        d ——(main)——> b ===(internal)===> a <——(main)—— c

    Pure-PyTorch equivalent of GemNet-OC ``get_quadruplets``.

    Args:
        main_edge_index: ``[2, n_main]`` — long-range (outer) graph edges.
        internal_edge_index: ``[2, n_internal]`` — short-range (central) graph edges.
        n_atoms: Total number of atoms.
        main_cell_offsets: ``[n_main, 3]`` periodic cell offsets for main graph.
        internal_cell_offsets: ``[n_internal, 3]`` periodic cell offsets for
            internal graph.

    Returns:
        Dict with keys describing the quadruplet ``d→b→a←c``:

        - ``"d_to_b_edge"`` — main-edge indices for ``d→b``, shape ``[n_trip_in]``.
        - ``"b_to_a_edge"`` — internal-edge indices for the central bond ``b→a``,
          shape ``[n_trip_in]``.
        - ``"b_to_a_edge_agg"`` — local aggregation index within each ``b→a`` edge,
          shape ``[n_trip_in]``.
        - ``"c_to_a_edge"`` — main-edge indices for ``c→a``, shape ``[n_trip_out]``.
        - ``"c_to_a_edge_agg"`` — local aggregation index within each ``c→a`` edge,
          shape ``[n_trip_out]``.
        - ``"quad_c_to_a_edge"`` — main-edge index of the ``c→a`` bond for each
          quadruplet, shape ``[n_quads]``.
        - ``"quad_d_to_b_trip_idx"`` — index into ``d_to_b_edge`` / ``b_to_a_edge``
          for each quadruplet, shape ``[n_quads]``.
        - ``"quad_c_to_a_trip_idx"`` — index into ``c_to_a_edge`` for each
          quadruplet, shape ``[n_quads]``.
        - ``"quad_c_to_a_agg"`` — local aggregation index within each ``c→a`` main
          edge across quadruplets, shape ``[n_quads]``.
    """
    src_main = main_edge_index[0]
    n_main_edges = src_main.size(0)
    n_internal_edges = internal_edge_index.size(1)
    device = src_main.device

    # Input triplets d→b→a: main edges arriving at b, paired with internal edge b→a.
    triplet_in = build_mixed_triplets(
        main_edge_index,
        internal_edge_index,
        n_atoms,
        to_outedge=True,
        cell_offsets_in=main_cell_offsets,
        cell_offsets_out=internal_cell_offsets,
    )

    # Output triplets c→a←b: internal edge b→a paired with main edges arriving at a.
    triplet_out = build_mixed_triplets(
        internal_edge_index,
        main_edge_index,
        n_atoms,
        to_outedge=False,
        cell_offsets_in=internal_cell_offsets,
        cell_offsets_out=main_cell_offsets,
    )

    # Count input triplets per internal edge
    ones_in = torch.ones_like(triplet_in["trip_out"])
    n_trip_in_per_inter = torch.zeros(n_internal_edges, dtype=torch.long, device=device)
    n_trip_in_per_inter.index_add_(0, triplet_in["trip_out"], ones_in)

    # Build CSR of input triplets grouped by internal edge.
    # Sort input triplets by internal edge so CSR lookup is contiguous.
    order_ti = torch.argsort(triplet_in["trip_out"], stable=True)
    sorted_trip_in_by_inter = triplet_in["trip_in"][order_ti]

    csr_ti = torch.zeros(n_internal_edges + 1, dtype=torch.long, device=device)
    csr_ti[1:] = n_trip_in_per_inter.cumsum(0)

    # Only output triplets with ≥1 matching input triplet can form quadruplets.
    n_in_for_out = n_trip_in_per_inter[triplet_out["trip_in"]]
    valid_out = n_in_for_out > 0
    trip_out_main = triplet_out["trip_out"][valid_out]  # c→a main edge indices
    trip_out_inter = triplet_out["trip_in"][valid_out]  # b→a internal edge indices
    n_in_for_valid = n_in_for_out[valid_out]

    # Cartesian product: each valid output triplet paired with each input triplet
    # that shares its central b→a internal edge.
    quad_c_to_a = torch.repeat_interleave(trip_out_main, n_in_for_valid)
    central_edge = torch.repeat_interleave(trip_out_inter, n_in_for_valid)
    quad_c_to_a_trip_idx = torch.repeat_interleave(
        torch.arange(trip_out_main.size(0), device=device), n_in_for_valid
    )

    # Local index cycling 0..n_in[e]-1 within each output-triplet block.
    # cumsum gives the start of each block; subtracting it gives the within-block offset.
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

    ti_idx = csr_ti[central_edge] + local
    d_to_b = sorted_trip_in_by_inter[ti_idx]

    # Filter: c ≠ d (same atom in same periodic image is not a valid quadruplet)
    cell_offset_cd = (
        main_cell_offsets[d_to_b]
        + internal_cell_offsets[central_edge]
        - main_cell_offsets[quad_c_to_a]
    )
    mask = (src_main[quad_c_to_a] != src_main[d_to_b]) | torch.any(
        cell_offset_cd != 0, dim=-1
    )

    quad_c_to_a = quad_c_to_a[mask]
    quad_c_to_a_trip_idx = quad_c_to_a_trip_idx[mask]
    quad_d_to_b_trip_idx = order_ti[ti_idx[mask]]

    return {
        "d_to_b_edge": triplet_in["trip_in"],
        "b_to_a_edge": triplet_in["trip_out"],
        "b_to_a_edge_agg": triplet_in["trip_out_agg"],
        "c_to_a_edge": triplet_out["trip_out"],
        "c_to_a_edge_agg": triplet_out["trip_out_agg"],
        "quad_c_to_a_edge": quad_c_to_a,
        "quad_d_to_b_trip_idx": quad_d_to_b_trip_idx,
        "quad_c_to_a_trip_idx": quad_c_to_a_trip_idx,
        "quad_c_to_a_agg": _inner_idx(quad_c_to_a, n_main_edges),
    }
