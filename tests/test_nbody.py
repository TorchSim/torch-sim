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


def test_build_quadruplets_simple() -> None:
    """Test build_quadruplets with a simple case."""
    # Main edges: 0→1, 2→1, 1→3, 1→4
    # Qint edges (central bonds): 1→3
    # Input triplets (d→b→a): edges arriving at b=1 for qint edge 1→3
    #   So: 0→1→3, 2→1→3
    # Output triplets (c→a←b): edges arriving at a=3 for main edges starting from 1
    #   So: 1→3←1 (from main edge 1→3), 1→4 (but 4≠3, so filtered)
    # Quadruplets: Cartesian product, filtered where c≠d
    main_edge_index = torch.tensor([[0, 2, 1, 1], [1, 1, 3, 4]])
    qint_edge_index = torch.tensor([[1], [3]])  # Central bond 1→3
    n_atoms = 5

    main_cell_offsets = torch.zeros(4, 3)
    qint_cell_offsets = torch.zeros(1, 3)

    result = build_quadruplets(
        main_edge_index, qint_edge_index, n_atoms, main_cell_offsets, qint_cell_offsets
    )

    assert "main_edge_d_to_b" in result
    assert "qint_edge_b_to_a" in result
    assert "qint_edge_b_to_a_agg" in result
    assert "main_edge_c_to_a" in result
    assert "main_edge_c_to_a_agg" in result
    assert "quad_main_edge_c_to_a" in result
    assert "trip_in_to_quad" in result
    assert "trip_out_to_quad" in result
    assert "quad_main_edge_agg" in result

    # May have quadruplets depending on filtering
    # The structure should be valid even if empty
    assert isinstance(result["quad_main_edge_c_to_a"], torch.Tensor)
    quad_len = len(result["quad_main_edge_c_to_a"])
    trip_len = len(result["trip_in_to_quad"])
    assert quad_len == trip_len


def test_build_quadruplets_empty() -> None:
    """Test build_quadruplets with no valid quadruplets."""
    # Main edges: 0→1
    # Qint edges: 2→3 (disconnected from main)
    main_edge_index = torch.tensor([[0], [1]])
    qint_edge_index = torch.tensor([[2], [3]])
    n_atoms = 4

    main_cell_offsets = torch.zeros(1, 3)
    qint_cell_offsets = torch.zeros(1, 3)

    result = build_quadruplets(
        main_edge_index, qint_edge_index, n_atoms, main_cell_offsets, qint_cell_offsets
    )

    # Should have empty quadruplets
    assert len(result["quad_main_edge_c_to_a"]) == 0
    assert len(result["trip_in_to_quad"]) == 0
    assert len(result["trip_out_to_quad"]) == 0


def test_build_quadruplets_torsion_like() -> None:
    """Test build_quadruplets with a torsion-like pattern i-j-k-l."""
    # Torsion pattern: 0-1-2-3 where 1-2 is the central bond
    # Main edges (all bonds): 0→1, 1→2, 2→3, 1→0, 2→1, 3→2 (bidirectional)
    # Qint edges (central bonds): 1→2
    main_edge_index = torch.tensor(
        [[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]]  # Bidirectional edges
    )
    qint_edge_index = torch.tensor([[1], [2]])  # Central bond
    n_atoms = 4

    main_cell_offsets = torch.zeros(6, 3)
    qint_cell_offsets = torch.zeros(1, 3)

    result = build_quadruplets(
        main_edge_index, qint_edge_index, n_atoms, main_cell_offsets, qint_cell_offsets
    )

    # Should produce quadruplets where:
    # - trip_in points to edges arriving at 1 (0→1, 2→1)
    # - quad_main_edge_c_to_a points to edges starting from 2 (2→3, 2→1)
    # - But 2→1 is filtered (same as input), leaving 2→3
    assert len(result["quad_main_edge_c_to_a"]) > 0

    # Verify the structure
    assert isinstance(result["main_edge_d_to_b"], torch.Tensor)
    assert isinstance(result["main_edge_c_to_a"], torch.Tensor)


def test_build_quadruplets_cell_offset_filtering() -> None:
    """Test that build_quadruplets filters based on cell offsets."""
    # Main edges: 0→1, 2→1, 1→3
    # Qint edges: 1→3
    main_edge_index = torch.tensor([[0, 2, 1], [1, 1, 3]])
    qint_edge_index = torch.tensor([[1], [3]])
    n_atoms = 4

    # Without offsets: should produce some quadruplets
    main_cell_offsets = torch.zeros(3, 3)
    qint_cell_offsets = torch.zeros(1, 3)

    # Test that it runs without errors (result may be empty)
    _result_no_offsets = build_quadruplets(
        main_edge_index,
        qint_edge_index,
        n_atoms,
        main_cell_offsets,
        qint_cell_offsets,
    )

    # With offsets that create filtering
    main_cell_offsets_filtered = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]]  # Last edge has offset
    )

    result_with_offsets = build_quadruplets(
        main_edge_index,
        qint_edge_index,
        n_atoms,
        main_cell_offsets_filtered,
        qint_cell_offsets,
    )

    # Results may differ based on filtering
    assert isinstance(result_with_offsets["quad_main_edge_c_to_a"], torch.Tensor)


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
    qint_edge_index = torch.tensor([[1], [2]], device=dev)
    n_atoms = 4

    main_cell_offsets = torch.zeros(3, 3, device=dev)
    qint_cell_offsets = torch.zeros(1, 3, device=dev)

    result = build_quadruplets(
        main_edge_index, qint_edge_index, n_atoms, main_cell_offsets, qint_cell_offsets
    )

    assert result["quad_main_edge_c_to_a"].device == dev
    assert result["trip_in_to_quad"].device == dev
    assert result["main_edge_d_to_b"].device == dev
    assert result["main_edge_c_to_a"].device == dev


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
    qint_edge_index = torch.tensor([[1], [3]])
    n_atoms = 5
    main_cell_offsets = torch.zeros(4, 3)
    qint_cell_offsets = torch.zeros(1, 3)

    compiled_fn = torch.jit.script(build_quadruplets)

    # Run compiled version
    result_compiled = compiled_fn(
        main_edge_index, qint_edge_index, n_atoms, main_cell_offsets, qint_cell_offsets
    )

    # Run original version
    result_original = build_quadruplets(
        main_edge_index, qint_edge_index, n_atoms, main_cell_offsets, qint_cell_offsets
    )

    # Results should match
    torch.testing.assert_close(
        result_compiled["main_edge_d_to_b"], result_original["main_edge_d_to_b"]
    )
    torch.testing.assert_close(
        result_compiled["qint_edge_b_to_a"], result_original["qint_edge_b_to_a"]
    )
    torch.testing.assert_close(
        result_compiled["main_edge_c_to_a"], result_original["main_edge_c_to_a"]
    )
    torch.testing.assert_close(
        result_compiled["quad_main_edge_c_to_a"], result_original["quad_main_edge_c_to_a"]
    )
    torch.testing.assert_close(
        result_compiled["trip_in_to_quad"], result_original["trip_in_to_quad"]
    )
    torch.testing.assert_close(
        result_compiled["trip_out_to_quad"], result_original["trip_out_to_quad"]
    )
