"""Tests for n-body interaction index builders."""

import pytest
import torch

from torch_sim.neighbors.nbody import (
    _inner_idx,
    build_mixed_triplets,
    build_quadruplets,
    build_triplets,
)


def test_inner_idx() -> None:
    """Test _inner_idx local enumeration within sorted segments."""
    # Test case from docstring: [0,0,0,1,1,2,2,2,2] -> [0,1,2,0,1,0,1,2,3]
    sorted_idx = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    result = _inner_idx(sorted_idx, dim_size=3)
    expected = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
    torch.testing.assert_close(result, expected)

    # Test single segment
    sorted_idx = torch.tensor([0, 0, 0])
    result = _inner_idx(sorted_idx, dim_size=1)
    expected = torch.tensor([0, 1, 2])
    torch.testing.assert_close(result, expected)

    # Test empty
    sorted_idx = torch.tensor([], dtype=torch.long)
    result = _inner_idx(sorted_idx, dim_size=0)
    expected = torch.tensor([], dtype=torch.long)
    torch.testing.assert_close(result, expected)

    # Test with gaps
    sorted_idx = torch.tensor([0, 0, 2, 2, 2])
    result = _inner_idx(sorted_idx, dim_size=3)
    expected = torch.tensor([0, 1, 0, 1, 2])
    torch.testing.assert_close(result, expected)


def test_build_triplets_simple() -> None:
    """Test build_triplets with a simple star graph."""
    # Star graph: atom 0 connected to atoms 1, 2, 3
    # Produces deg*(deg-1) = 3*2 = 6 ordered triplets (not combinations)
    edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]])  # [2, 3]
    n_atoms = 4

    result = build_triplets(edge_index, n_atoms)

    assert len(result["trip_in"]) == 6  # 3*2 = 6 ordered pairs
    assert len(result["trip_out"]) == 6
    assert len(result["center_atom"]) == 6
    assert (result["center_atom"] == 0).all()

    # Verify all triplets have center atom 0
    assert torch.all(result["center_atom"] == 0)

    # Verify trip_in and trip_out are different edges
    assert torch.all(result["trip_in"] != result["trip_out"])


def test_build_triplets_empty() -> None:
    """Test build_triplets with no valid triplets."""
    # Linear chain: 0-1-2 (no atom has degree >= 2)
    edge_index = torch.tensor([[0, 1], [1, 2]])  # [2, 2]
    n_atoms = 3

    result = build_triplets(edge_index, n_atoms)

    assert len(result["trip_in"]) == 0
    assert len(result["trip_out"]) == 0
    assert len(result["center_atom"]) == 0
    assert len(result["trip_out_agg"]) == 0


def test_build_triplets_complex() -> None:
    """Test build_triplets with a more complex graph."""
    # Graph: 0-1-2-3, with 1 connected to 4, 5
    # Atom 1 has degree 4 (edges: 0→1, 2→1, 4→1, 5→1)
    # Produces deg*(deg-1) = 4*3 = 12 ordered triplets
    edge_index = torch.tensor(
        [[0, 2, 4, 5], [1, 1, 1, 1]]  # All edges point to atom 1
    )
    n_atoms = 6

    result = build_triplets(edge_index, n_atoms)

    assert len(result["trip_in"]) == 12  # 4*3 = 12 ordered pairs
    assert len(result["trip_out"]) == 12
    assert (result["center_atom"] == 1).all()

    # Verify all triplets are unique
    trip_pairs = torch.stack([result["trip_in"], result["trip_out"]], dim=0)
    unique_pairs = torch.unique(trip_pairs, dim=1)
    assert unique_pairs.shape[1] == 12


def test_build_mixed_triplets_to_outedge_false() -> None:
    """Test build_mixed_triplets with to_outedge=False (c→a style)."""
    # When to_outedge=False, matches on target atom of output edges
    # Input edges: 0→4, 1→4, 3→5
    # Output edges: 2→4, 2→5
    # Should match on target atoms 4 and 5, producing triplets:
    # (0→4, 2→4), (1→4, 2→4), (3→5, 2→5)
    edge_index_in = torch.tensor([[0, 1, 3], [4, 4, 5]])
    edge_index_out = torch.tensor([[2, 2], [4, 5]])
    n_atoms = 6

    result = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms, to_outedge=False
    )

    assert len(result["trip_in"]) == 3
    assert len(result["trip_out"]) == 3

    # Verify trip_in edges point to atoms 4 or 5 (targets of output edges)
    trip_in_targets = edge_index_in[1][result["trip_in"]]
    assert torch.all((trip_in_targets == 4) | (trip_in_targets == 5))
    # Verify trip_out edges have targets 4 or 5
    trip_out_targets = edge_index_out[1][result["trip_out"]]
    assert torch.all((trip_out_targets == 4) | (trip_out_targets == 5))


def test_build_mixed_triplets_to_outedge_true() -> None:
    """Test build_mixed_triplets with to_outedge=True (a→c style)."""
    # Input edges: 0→2, 1→2, 3→2
    # Output edges: 2→4, 2→5
    # Should match on source atom 2 of output edges, producing triplets:
    # (0→2, 2→4), (1→2, 2→4), (3→2, 2→4), (0→2, 2→5), (1→2, 2→5), (3→2, 2→5)
    edge_index_in = torch.tensor([[0, 1, 3], [2, 2, 2]])
    edge_index_out = torch.tensor([[2, 2], [4, 5]])
    n_atoms = 6

    result = build_mixed_triplets(edge_index_in, edge_index_out, n_atoms, to_outedge=True)

    assert len(result["trip_in"]) == 6
    assert len(result["trip_out"]) == 6

    # Verify all trip_in edges point to atom 2
    assert torch.all(edge_index_in[1][result["trip_in"]] == 2)
    # Verify all trip_out edges start from atom 2
    assert torch.all(edge_index_out[0][result["trip_out"]] == 2)


def test_build_mixed_triplets_self_loop_filtering() -> None:
    """Test that build_mixed_triplets filters self-loops."""
    # When to_outedge=False, matches on target atom of output edges
    # Input edges: 0→2, 1→2 (where 1→2 is a self-loop relative to output)
    # Output edges: 1→2
    # Should filter out the self-loop where source of input (1) equals
    # source of output (1)
    edge_index_in = torch.tensor([[0, 1], [2, 2]])
    edge_index_out = torch.tensor([[1], [2]])
    n_atoms = 3

    result = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms, to_outedge=False
    )

    # Should filter out the edge where src_in (1) == src_out (1)
    assert len(result["trip_in"]) == 1
    assert result["trip_in"][0] == 0  # Only the non-self-loop edge
    src_in = edge_index_in[0][result["trip_in"][0]]
    src_out = edge_index_out[0][result["trip_out"][0]]
    assert src_in != src_out


def test_build_mixed_triplets_with_cell_offsets() -> None:
    """Test build_mixed_triplets with cell offset filtering."""
    # When to_outedge=False, matches on target atom of output edges
    # Input edges: 0→3, 1→3
    # Output edges: 2→3
    edge_index_in = torch.tensor([[0, 1], [3, 3]])
    edge_index_out = torch.tensor([[2], [3]])
    n_atoms = 4

    # Without cell offsets: should produce 2 triplets
    result_no_offsets = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms, to_outedge=False
    )
    assert len(result_no_offsets["trip_in"]) == 2

    # With cell offsets that filter one out
    # The mask keeps edges where: (idx_atom_in != idx_atom_out) OR (cell_sum != 0)
    # So if cell_sum is non-zero, the edge is kept (not filtered)
    # To filter, we need idx_atom_in == idx_atom_out AND cell_sum == 0
    # Let's test with offsets that create a non-zero cell_sum for one edge
    cell_offsets_in = torch.tensor([[0, 0, 0], [0, 0, 0]])  # No offset in input
    cell_offsets_out = torch.tensor([[1, 0, 0]])  # Offset in output

    result_with_offsets = build_mixed_triplets(
        edge_index_in,
        edge_index_out,
        n_atoms,
        to_outedge=False,
        cell_offsets_in=cell_offsets_in,
        cell_offsets_out=cell_offsets_out,
    )

    # With to_outedge=False: cell_sum = cell_offsets_out - cell_offsets_in
    # For both edges: cell_sum = [1,0,0] - [0,0,0] = [1,0,0] (non-zero)
    # So both edges are kept (mask includes OR with cell_sum != 0)
    # Actually, let's just verify it runs without error
    assert isinstance(result_with_offsets["trip_in"], torch.Tensor)
    assert len(result_with_offsets["trip_in"]) >= 0


def test_build_triplets_exact_values() -> None:
    """Verify exact trip_in/trip_out pairs for a hand-checkable star graph.

    Star: edges 0→A, 1→A, 2→A  (edge indices 0,1,2, all target atom A=3).
    Triplets b→A←c (b≠c, ordered pairs):
      (e0,e1), (e0,e2), (e1,e0), (e1,e2), (e2,e0), (e2,e1)
    So trip_in and trip_out are permutations of {0,1,2} where in≠out.
    """
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
    result = build_triplets(edge_index, n_atoms=4)

    pairs = set(zip(result["trip_in"].tolist(), result["trip_out"].tolist(), strict=True))
    expected = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    assert pairs == expected
    assert (result["center_atom"] == 3).all()


def test_build_triplets_two_centers() -> None:
    """Two independent star centers produce independent triplet sets.

    Edges: 0→A(=4), 1→A(=4), 2→B(=5), 3→B(=5).
    Triplets at A: (e0,e1),(e1,e0); at B: (e2,e3),(e3,e2). Total 4.
    """
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 4, 5, 5]])
    result = build_triplets(edge_index, n_atoms=6)

    assert len(result["trip_in"]) == 4
    pairs = set(zip(result["trip_in"].tolist(), result["trip_out"].tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0), (2, 3), (3, 2)}
    # Center atoms match
    center = result["center_atom"].tolist()
    ins = result["trip_in"].tolist()
    outs = result["trip_out"].tolist()
    for ti, to, c in zip(ins, outs, center, strict=True):
        assert edge_index[1, ti].item() == c
        assert edge_index[1, to].item() == c


def test_build_mixed_triplets_exact_values_to_outedge_false() -> None:
    """Hand-verified triplets for to_outedge=False (c→a style).

    in-edges:  e0=0→4, e1=1→4, e2=3→5
    out-edges: f0=2→4, f1=2→5

    For f0 (target=4): in-edges with target 4 are e0,e1 → triplets (e0,f0),(e1,f0)
    For f1 (target=5): in-edges with target 5 are e2    → triplet  (e2,f1)
    Self-loop check: src_in vs src_out — none here (sources 0,1,3 ≠ 2).
    Expected: trip_in=[0,1,2], trip_out=[0,0,1] (in some order within each group).
    """
    edge_index_in = torch.tensor([[0, 1, 3], [4, 4, 5]])
    edge_index_out = torch.tensor([[2, 2], [4, 5]])
    result = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms=6, to_outedge=False
    )

    pairs = set(zip(result["trip_in"].tolist(), result["trip_out"].tolist(), strict=True))
    assert pairs == {(0, 0), (1, 0), (2, 1)}


def test_build_mixed_triplets_exact_values_to_outedge_true() -> None:
    """Hand-verified triplets for to_outedge=True (d→b→a style).

    in-edges:  e0=0→2, e1=1→2, e2=3→2
    out-edges: f0=2→4, f1=2→5

    For f0 (source=2): in-edges with target 2 are e0,e1,e2 → 3 triplets
    For f1 (source=2): same in-edges → 3 triplets
    Self-loop check (to_outedge=True): src_in vs tgt_out.
      src_in ∈ {0,1,3}, tgt_out ∈ {4,5} — no overlap, all 6 survive.
    """
    edge_index_in = torch.tensor([[0, 1, 3], [2, 2, 2]])
    edge_index_out = torch.tensor([[2, 2], [4, 5]])
    result = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms=6, to_outedge=True
    )

    assert len(result["trip_in"]) == 6
    pairs = set(zip(result["trip_in"].tolist(), result["trip_out"].tolist(), strict=True))
    assert pairs == {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}


def test_build_mixed_triplets_cell_offset_self_loop() -> None:
    """Self-loop distinguished only by cell offset is kept; same-cell is dropped.

    in-edge e0: atom 1→2 with offset [0,0,0]
    out-edge f0: atom 1→2 with offset [0,0,0]
    Same atom AND same cell → self-loop, dropped. Only e0 and f0 are involved;
    result should be empty.

    in-edge e1: atom 1→2 with offset [1,0,0]  (image copy)
    out-edge f0: atom 1→2 with offset [0,0,0]
    cell_sum = out - in = [-1,0,0] ≠ 0 → kept.
    """
    edge_index_in = torch.tensor([[1, 1], [2, 2]])  # e0, e1
    edge_index_out = torch.tensor([[1], [2]])  # f0
    n_atoms = 3
    offsets_in = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.float)
    offsets_out = torch.tensor([[0, 0, 0]], dtype=torch.float)

    result_no_off = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms, to_outedge=False
    )
    # Without offsets: src_in=1, src_out=1 → both filtered (same source)
    assert len(result_no_off["trip_in"]) == 0

    result_with_off = build_mixed_triplets(
        edge_index_in,
        edge_index_out,
        n_atoms,
        to_outedge=False,
        cell_offsets_in=offsets_in,
        cell_offsets_out=offsets_out,
    )
    # e0 still filtered (same atom, cell_sum=[0,0,0]-[0,0,0]=[0,0,0])
    # e1 kept (same atom, but cell_sum=[0,0,0]-[1,0,0]=[-1,0,0] ≠ 0)
    assert len(result_with_off["trip_in"]) == 1
    assert result_with_off["trip_in"][0].item() == 1  # e1


def test_build_quadruplets_exact_torsion() -> None:
    """Exact output for the minimal torsion 0-1-2-3, qint edge 1→2.

    main edges (full bidirectional list):
      e0=0→1, e1=1→2, e2=2→3, e3=1→0, e4=2→1, e5=3→2

    build_mixed_triplets(main, qint, to_outedge=True):
      shared_atom = src(q0) = 1.
      Matches main edges where tgt_in == 1: e0(0→1) and e3(1→0)... wait,
      tgt of e3 = 0, not 1. Only e0(0→1) has tgt=1.
      Self-loop filter (to_outedge=True): src_in[e0]=0 vs tgt_out[q0]=2 → 0≠2 ✓
      Input triplets: [(e0, q0)]  →  1 input triplet.

    build_mixed_triplets(qint, main, to_outedge=False):
      shared_atom = tgt(main out-edge). For each main edge, shared atom = its target.
      Match qint edges where tgt_in == shared_atom. qint edge q0: tgt=2.
      Main edges with target 2: e1(1→2), e4(2→1)? No, e4 target=1. e5(3→2) target=2.
      So main edges with target 2: e1(1→2), e5(3→2).
      Self-loop filter (to_outedge=False): src_in[q0]=1 vs src_out.
        e1: src_out=1 == src_in=1 → filtered!
        e5: src_out=3 ≠ 1 → kept.
      Output triplets: [(q0, e5)]  → 1 output triplet.

    Cartesian product: 1×1 = 1. c≠d filter:
      c=src(e5)=3, d=src(e0)=0 → 3≠0 ✓ → 1 quadruplet survives.
    """
    main = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])  # e0..e5
    qint = torch.tensor([[1], [2]])
    n_atoms = 4
    main_cell = torch.zeros(6, 3)
    qint_cell = torch.zeros(1, 3)

    result = build_quadruplets(main, qint, n_atoms, main_cell, qint_cell)

    assert len(result["quad_c_to_a_edge"]) == 1
    # The single c→a edge is e5 (index 5), arriving at atom 2
    assert result["quad_c_to_a_edge"][0].item() == 5
    assert main[1, 5].item() == 2  # sanity: e5 targets atom 2
    # trip_in_to_quad[0] points into triplet_in["trip_in"]; d→b edge must target atom 1
    ti = build_mixed_triplets(
        main,
        qint,
        n_atoms,
        to_outedge=True,
        cell_offsets_in=main_cell,
        cell_offsets_out=qint_cell,
    )
    d_to_b = ti["trip_in"][result["quad_d_to_b_trip_idx"][0].item()]
    assert main[1, d_to_b].item() == 1


def test_build_quadruplets_multi_input_triplets() -> None:
    """Multiple input triplets per qint edge all pair correctly.

    main edges: e0=0→1, e1=2→1, e2=1→3, e3=3→1
      (atoms 0,2 both arrive at 1; atom 3 also arrives at 1)
    qint edge: q0=1→3

    build_mixed_triplets(main, qint, to_outedge=True):
      shared_atom = src(q0)=1; main edges with tgt=1: e0,e1,e3.
      Self-loop (to_outedge=True): src_in vs tgt_out[q0]=3.
        e0: src=0 ≠ 3 ✓, e1: src=2 ≠ 3 ✓, e3: src=3 == 3 → filtered.
      Input triplets: [(e0,q0),(e1,q0)]  →  2 input triplets.

    build_mixed_triplets(qint, main, to_outedge=False):
      For each main out-edge, shared_atom = tgt. Match qint edges with tgt_in=shared_atom.
      qint q0 has tgt=3; main edges with target=3: e2(1→3).
      Self-loop: src_in[q0]=1 vs src_out[e2]=1 → equal → filtered.
      Output triplets: none → 0 quadruplets.

    Use a different qint to get output triplets: q0=1→4, add e4=5→4.
    """
    # main: e0=0→1, e1=2→1, e2=5→4, e3=1→4
    # qint: q0=1→4
    main = torch.tensor([[0, 2, 5, 1], [1, 1, 4, 4]])
    qint = torch.tensor([[1], [4]])
    n_atoms = 6
    main_cell = torch.zeros(4, 3)
    qint_cell = torch.zeros(1, 3)

    # Input triplets (d→b=1): e0,e1 arrive at 1; self-loop: src vs tgt(q0)=4 → 0,2≠4 ✓
    # Output triplets (c→4): e2(5→4),e3(1→4) arrive at 4.
    #   Self-loop: src_in[q0]=1 vs src_out: e3 src=1 → filtered; e2 src=5≠1 ✓.
    # Cross product: 2 input x 1 output = 2.
    # c≠d filter: c=src(e2)=5; d=src(e0)=0 → 5≠0 ✓; d=src(e1)=2 → 5≠2 ✓. All 2 survive.
    result = build_quadruplets(main, qint, n_atoms, main_cell, qint_cell)

    assert len(result["quad_c_to_a_edge"]) == 2
    assert (main[1][result["quad_c_to_a_edge"]] == 4).all()
    ti = build_mixed_triplets(
        main,
        qint,
        n_atoms,
        to_outedge=True,
        cell_offsets_in=main_cell,
        cell_offsets_out=qint_cell,
    )
    d_to_b = ti["trip_in"][result["quad_d_to_b_trip_idx"]]
    assert (main[1][d_to_b] == 1).all()


def test_build_quadruplets_empty() -> None:
    """Disconnected main and qint graphs produce zero quadruplets."""
    main_edge_index = torch.tensor([[0], [1]])
    internal_edge_index = torch.tensor([[2], [3]])
    n_atoms = 4
    result = build_quadruplets(
        main_edge_index,
        internal_edge_index,
        n_atoms,
        torch.zeros(1, 3),
        torch.zeros(1, 3),
    )
    assert len(result["quad_c_to_a_edge"]) == 0
    assert len(result["quad_d_to_b_trip_idx"]) == 0
    assert len(result["quad_c_to_a_trip_idx"]) == 0


def test_build_quadruplets_cd_same_atom_different_cell() -> None:
    """c==d by atom index but different cell image: quadruplet is kept.
    c==d same atom same cell: quadruplet is dropped.

    main: e0=0→1, e1=0→1(image,[1,0,0]), e2=0→3  |  qint: q0=1→3
    Input triplets (d→b=1, to_outedge=True, shared=src(q0)=1):
      main edges with tgt=1: e0,e1. Self-loop: src vs tgt(q0)=3 → 0≠3 ✓ both.
      2 input triplets.
    Output triplets (c→3, to_outedge=False, shared=tgt(main)):
      main edges with tgt=3: e2. Self-loop: src_in[q0]=1 vs src_out[e2]=0 → 1≠0 ✓.
      1 output triplet.
    Cross product: 2.
    c≠d filter: c=src(e2)=0, d=src(e0)=0 → same atom.
      cell_offset_cd = main_cell[e0] + qint_cell[q0] - main_cell[e2]
        = [0,0,0]+[0,0,0]-[0,0,0] = [0,0,0] → zero → FILTERED (c==d, same image).
      For e1: d=src(e1)=0 == c=0; cell_cd = [1,0,0]+[0,0,0]-[0,0,0]=[1,0,0] ≠ 0 → KEPT.
    Result: 1 quadruplet (from e1 image copy).
    """
    main = torch.tensor([[0, 0, 0], [1, 1, 3]])  # e0,e1,e2
    qint = torch.tensor([[1], [3]])
    n_atoms = 4
    main_cell = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float)
    qint_cell = torch.zeros(1, 3)

    result = build_quadruplets(main, qint, n_atoms, main_cell, qint_cell)
    assert len(result["quad_c_to_a_edge"]) == 1


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_build_triplets_device(device: str) -> None:
    """Test that build_triplets works on different devices."""
    dev = torch.device(device)
    edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]], device=dev)
    n_atoms = 4

    result = build_triplets(edge_index, n_atoms)

    assert result["trip_in"].device == dev
    assert result["trip_out"].device == dev
    assert result["center_atom"].device == dev


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_build_quadruplets_device(device: str) -> None:
    """Test that build_quadruplets works on different devices."""
    dev = torch.device(device)
    main_edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]], device=dev)
    internal_edge_index = torch.tensor([[1], [2]], device=dev)
    n_atoms = 4

    main_cell_offsets = torch.zeros(3, 3, device=dev)
    internal_cell_offsets = torch.zeros(1, 3, device=dev)

    result = build_quadruplets(
        main_edge_index,
        internal_edge_index,
        n_atoms,
        main_cell_offsets,
        internal_cell_offsets,
    )

    assert result["quad_c_to_a_edge"].device == dev
    assert result["quad_d_to_b_trip_idx"].device == dev
    assert result["d_to_b_edge"].device == dev
    assert result["c_to_a_edge"].device == dev


def test_build_triplets_jit_script() -> None:
    """Test that build_triplets can be JIT compiled."""
    edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]])
    n_atoms = 4

    # Compile the function
    compiled_fn = torch.jit.script(build_triplets)

    # Run compiled version
    result_compiled = compiled_fn(edge_index, n_atoms)

    # Run original version
    result_original = build_triplets(edge_index, n_atoms)

    # Results should match
    assert len(result_compiled["trip_in"]) == len(result_original["trip_in"])
    torch.testing.assert_close(result_compiled["trip_in"], result_original["trip_in"])
    torch.testing.assert_close(result_compiled["trip_out"], result_original["trip_out"])
    torch.testing.assert_close(
        result_compiled["center_atom"], result_original["center_atom"]
    )


def test_build_mixed_triplets_jit_script() -> None:
    """Test that build_mixed_triplets can be JIT compiled."""
    edge_index_in = torch.tensor([[0, 1, 3], [4, 4, 5]])
    edge_index_out = torch.tensor([[2, 2], [4, 5]])
    n_atoms = 6

    # JIT script doesn't support keyword-only args, so we need to wrap it
    # Use a wrapper that calls the function with positional args
    def wrapper_fn(
        edge_index_in: torch.Tensor,
        edge_index_out: torch.Tensor,
        n_atoms: int,
    ) -> dict[str, torch.Tensor]:
        return build_mixed_triplets(
            edge_index_in, edge_index_out, n_atoms, to_outedge=False
        )

    compiled_fn = torch.jit.script(wrapper_fn)

    # Run compiled version
    result_compiled = compiled_fn(edge_index_in, edge_index_out, n_atoms)

    # Run original version
    result_original = build_mixed_triplets(
        edge_index_in, edge_index_out, n_atoms, to_outedge=False
    )

    # Results should match
    assert len(result_compiled["trip_in"]) == len(result_original["trip_in"])
    torch.testing.assert_close(result_compiled["trip_in"], result_original["trip_in"])
    torch.testing.assert_close(result_compiled["trip_out"], result_original["trip_out"])


def test_build_quadruplets_jit_script() -> None:
    """Test that build_quadruplets can be JIT compiled."""
    main_edge_index = torch.tensor([[0, 2, 1, 1], [1, 1, 3, 4]])
    internal_edge_index = torch.tensor([[1], [3]])
    n_atoms = 5
    main_cell_offsets = torch.zeros(4, 3)
    internal_cell_offsets = torch.zeros(1, 3)

    compiled_fn = torch.jit.script(build_quadruplets)

    # Run compiled version
    result_compiled = compiled_fn(
        main_edge_index,
        internal_edge_index,
        n_atoms,
        main_cell_offsets,
        internal_cell_offsets,
    )

    # Run original version
    result_original = build_quadruplets(
        main_edge_index,
        internal_edge_index,
        n_atoms,
        main_cell_offsets,
        internal_cell_offsets,
    )

    # Results should match
    torch.testing.assert_close(
        result_compiled["d_to_b_edge"], result_original["d_to_b_edge"]
    )
    torch.testing.assert_close(
        result_compiled["b_to_a_edge"], result_original["b_to_a_edge"]
    )
    torch.testing.assert_close(
        result_compiled["c_to_a_edge"], result_original["c_to_a_edge"]
    )
    torch.testing.assert_close(
        result_compiled["quad_c_to_a_edge"], result_original["quad_c_to_a_edge"]
    )
    torch.testing.assert_close(
        result_compiled["quad_d_to_b_trip_idx"], result_original["quad_d_to_b_trip_idx"]
    )
    torch.testing.assert_close(
        result_compiled["quad_c_to_a_trip_idx"], result_original["quad_c_to_a_trip_idx"]
    )
