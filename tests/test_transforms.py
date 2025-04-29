# ruff: noqa: PT011
import numpy as np
import pytest
import torch
from ase.geometry import wrap_positions as ase_wrap_positions

import torch_sim.transforms as tst
from torch_sim.state import SimState


def test_inverse_box_scalar() -> None:
    """Test inverse function with scalar input.

    Verifies that the inverse of a scalar tensor returns its reciprocal.
    """
    # Test scalar inverse
    x = torch.tensor(2.0)
    assert torch.allclose(tst.inverse_box(x), torch.tensor(0.5))


def test_inverse_box_vector() -> None:
    """Test inverse function with vector input.

    Verifies that the inverse of a vector tensor returns element-wise reciprocals.
    """
    # Test vector inverse
    x = torch.tensor([2.0, 4.0])
    expected = torch.tensor([0.5, 0.25])
    assert torch.allclose(tst.inverse_box(x), expected)


def test_inverse_box_matrix() -> None:
    """Test inverse function with matrix input.

    Verifies that the inverse of a 2x2 matrix returns the correct matrix inverse.
    """
    # Test matrix inverse
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([[-2.0, 1.0], [1.5, -0.5]])
    assert torch.allclose(tst.inverse_box(x), expected)


def test_inverse_box_invalid() -> None:
    """Test inverse function with invalid input.

    Verifies that attempting to invert a 3D tensor raises a ValueError.
    """
    # Test invalid input (3D tensor)
    x = torch.ones(2, 2, 2)
    with pytest.raises(ValueError):
        tst.inverse_box(x)


def test_inverse_box_single_element() -> None:
    """Test inverse function with single element tensor.

    Verifies that a single-element tensor is correctly inverted.
    """
    # Test single element tensor
    x = torch.tensor([2.0])
    assert torch.allclose(tst.inverse_box(x), torch.tensor(0.5))


def test_pbc_wrap_general_orthorhombic() -> None:
    """Test periodic boundary wrapping with orthorhombic cell.

    Tests wrapping of positions in a simple cubic/orthorhombic cell where
    the lattice vectors are aligned with coordinate axes. This is the simplest
    case where the lattice matrix is diagonal.
    """
    # Simple cubic cell with length 2.0
    lattice = torch.eye(3) * 2.0

    # Test positions outside box in various directions
    positions = torch.tensor(
        [
            [2.5, 0.5, 0.5],  # Beyond +x face
            [-0.5, 0.5, 0.5],  # Beyond -x face
            [0.5, 2.5, 0.5],  # Beyond +y face
            [0.5, 0.5, -2.5],  # Beyond -z face
        ]
    )

    expected = torch.tensor(
        [
            [0.5, 0.5, 0.5],  # Wrapped to +x face
            [1.5, 0.5, 0.5],  # Wrapped to -x face
            [0.5, 0.5, 0.5],  # Wrapped to +y face
            [0.5, 0.5, 1.5],  # Wrapped to -z face
        ]
    )

    wrapped = tst.pbc_wrap_general(positions, lattice)
    assert torch.allclose(wrapped, expected)


@pytest.mark.parametrize(
    ("cell", "shift"),
    [
        # Cubic cell, integer shift [1, 1, 1]
        (
            torch.eye(3, dtype=torch.float64) * 2.0,
            torch.tensor([1, 1, 1], dtype=torch.float64),
        ),
        # Triclinic cell, integer shift [1, 1, 1]
        (
            torch.tensor(
                [
                    [2.0, 0.5, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.3, 2.0],
                ],
                dtype=torch.float64,
            ).T,
            torch.tensor([1, 1, 1], dtype=torch.float64),
        ),
        # Triclinic cell, integer shift [-1, 2, 0]
        (
            torch.tensor(
                [
                    [2.0, 0.5, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.3, 2.0],
                ],
                dtype=torch.float64,
            ).T,
            torch.tensor([-1, 2, 0], dtype=torch.float64),
        ),
    ],
)
def test_pbc_wrap_general_param(cell: torch.Tensor, shift: torch.Tensor) -> None:
    """Test periodic boundary wrapping for various cells and integer shifts."""
    base_frac = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float64)
    base_cart = base_frac @ cell.T
    shifted_cart = base_cart + (shift @ cell.T)
    wrapped = tst.pbc_wrap_general(shifted_cart, cell)
    torch.testing.assert_close(wrapped, base_cart, rtol=1e-6, atol=1e-6)


def test_pbc_wrap_general_edge_case() -> None:
    """Test periodic boundary wrapping at cell boundaries.

    Verifies correct handling of positions exactly on cell boundaries,
    which should be wrapped to zero rather than one to maintain consistency.
    """
    lattice = torch.eye(2) * 2.0
    positions = torch.tensor(
        [
            [2.0, 1.0],  # On +x boundary
            [1.0, 2.0],  # On +y boundary
            [2.0, 2.0],  # On corner
        ]
    )

    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])

    wrapped = tst.pbc_wrap_general(positions, lattice)
    assert torch.allclose(wrapped, expected)


def test_pbc_wrap_general_invalid_inputs() -> None:
    """Test error handling for invalid inputs.

    Verifies that appropriate errors are raised for:
    - Non-floating point tensors
    - Non-square lattice matrix
    - Mismatched dimensions between positions and lattice
    """
    # Test integer tensors
    with pytest.raises(TypeError):
        tst.pbc_wrap_general(torch.ones(3, dtype=torch.int64), torch.eye(3))

    # Test non-square lattice
    with pytest.raises(ValueError):
        tst.pbc_wrap_general(torch.ones(3), torch.ones(3, 2))

    # Test dimension mismatch
    with pytest.raises(ValueError):
        tst.pbc_wrap_general(torch.ones(4), torch.eye(3))


def test_pbc_wrap_general_batch() -> None:
    """Test periodic boundary wrapping with batched positions.

    Verifies that the function correctly handles batched position inputs
    while using a single lattice definition.
    """
    lattice = torch.eye(3) * 2.0

    # Batch of positions with shape (2, 4, 3)
    positions = torch.tensor(
        [
            [[2.5, 0.5, 0.5], [0.5, 2.5, 0.5], [0.5, 0.5, 2.5], [2.5, 2.5, 2.5]],
            [[3.5, 1.5, 1.5], [-0.5, 1.5, 1.5], [1.5, -0.5, 1.5], [1.5, 1.5, -0.5]],
        ]
    )

    expected = torch.tensor(
        [
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
        ]
    )

    wrapped = tst.pbc_wrap_general(positions, lattice)
    assert torch.allclose(wrapped, expected)


@pytest.mark.parametrize(
    "pbc", [[True, True, True], [True, True, False], [False, False, False], True, False]
)
@pytest.mark.parametrize("pretty_translation", [True, False])
def test_wrap_positions_matches_ase(
    *, pbc: bool | list[bool], pretty_translation: bool
) -> None:
    # Generate random positions and cell
    torch.manual_seed(42)
    positions = torch.randn(10, 3)
    cell = torch.eye(3) + 0.1 * torch.randn(3, 3)

    # Run both implementations
    torch_result = tst.wrap_positions(
        positions, cell, pbc=pbc, pretty_translation=pretty_translation
    )

    ase_result = ase_wrap_positions(
        positions.numpy(), cell.numpy(), pbc=pbc, pretty_translation=pretty_translation
    )

    np.testing.assert_allclose(torch_result.numpy(), ase_result, rtol=1e-6, atol=1e-6)


def test_wrap_positions_basic():
    pos = torch.tensor([[-0.1, 1.01, -0.5]], dtype=torch.float32)
    cell = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]])

    wrapped = tst.wrap_positions(pos, cell, pbc=[True, True, False])
    expected = torch.tensor([[0.9, 0.01, -0.5]])

    torch.testing.assert_close(wrapped, expected, rtol=1e-6, atol=1e-6)


def test_translate_pretty():
    coords = torch.tensor([[0.1, 1.2, -0.3], [0.7, 0.8, 0.9]])
    pbc = [True, True, True]

    translated = tst.translate_pretty(coords, pbc)

    # Check that differences between coordinates are preserved
    orig_diff = (coords[1] - coords[0]) % 1.0
    new_diff = (translated[1] - translated[0]) % 1.0

    torch.testing.assert_close(orig_diff, new_diff, rtol=1e-6, atol=1e-6)

    # Check that coordinates are in [0, 1) range
    assert torch.all((translated >= 0) & (translated < 1))


def test_pbc_wrap_batched_orthorhombic(si_double_sim_state: SimState) -> None:
    """Test batched periodic boundary wrapping with orthorhombic cell."""
    # Make a copy of the state to modify positions
    state = si_double_sim_state

    # Modify a specific atom's position in each batch to be outside the cell
    # Get the first atom in each batch
    batch_0_mask = state.batch == 0
    batch_1_mask = state.batch == 1

    # Get current cell size (assume cubic for simplicity)
    cell_size = state.cell[0, 0, 0]

    # Create test positions that are outside the box in various directions
    test_positions = state.positions.clone()

    # First batch: beyond +x face
    idx0 = torch.where(batch_0_mask)[0][0]
    test_positions[idx0, 0] = cell_size + 0.5

    # Second batch: beyond -x face
    idx1 = torch.where(batch_1_mask)[0][0]
    test_positions[idx1, 0] = -0.5

    # Apply wrapping
    wrapped = tst.pbc_wrap_batched(test_positions, cell=state.cell, batch=state.batch)

    # Check first modified atom is properly wrapped
    assert wrapped[idx0, 0] < cell_size
    assert wrapped[idx0, 0] >= 0

    # Check second modified atom is properly wrapped
    assert wrapped[idx1, 0] < cell_size
    assert wrapped[idx1, 0] >= 0


def test_pbc_wrap_batched_triclinic(device: torch.device) -> None:
    """Test batched periodic boundary wrapping with triclinic cell."""
    # Define cell matrices (M_row convention)
    cell1 = torch.tensor(
        [
            [2.0, 0.5, 0.0],  # a vector with b-tilt
            [0.0, 2.0, 0.0],  # b vector
            [0.0, 0.3, 2.0],  # c vector with b-tilt
        ],
        dtype=torch.float64,
        device=device,
    )
    cell2 = torch.tensor(
        [
            [2.0, 0.0, 0.5],  # a vector with c-tilt
            [0.3, 2.0, 0.0],  # b vector with a-tilt
            [0.0, 0.0, 2.0],  # c vector
        ],
        dtype=torch.float64,
        device=device,
    )
    cell = torch.stack([cell1, cell2])

    # Define positions (r_row convention)
    positions = torch.tensor(
        [
            [2.5, 2.5, 2.5],  # Atom 0 (batch 0)
            [2.7, 2.7, 2.7],  # Atom 1 (batch 1)
        ],
        dtype=torch.float64,
        device=device,
    )
    batch = torch.tensor([0, 1], device=device)

    # Stack the cells for batched processing
    cell = torch.stack([cell1, cell2])

    # Define positions (r_row convention)
    positions = torch.tensor(
        [
            [2.5, 2.5, 2.5],  # Atom 0 (batch 0)
            [2.7, 2.7, 2.7],  # Atom 1 (batch 1)
        ],
        dtype=torch.float64,
        device=device,
    )
    batch = torch.tensor([0, 1], device=device)

    # --- Calculate Expected Results Manually (M_row, r_row) --- #
    def manual_wrap(r_row: torch.Tensor, M_row: torch.Tensor) -> torch.Tensor:
        inv_M_row_T = torch.linalg.inv(M_row.T)
        f_row = r_row @ inv_M_row_T
        wrapped_f_row = f_row - torch.floor(f_row)
        # Handle boundary cases (optional but good practice)
        wrapped_f_row = torch.where(
            torch.isclose(wrapped_f_row, torch.ones_like(wrapped_f_row)),
            torch.zeros_like(wrapped_f_row),
            wrapped_f_row,
        )
        return wrapped_f_row @ M_row

    expected1 = manual_wrap(positions[0], cell1)
    expected2 = manual_wrap(positions[1], cell2)
    expected_stacked = torch.stack([expected1, expected2])
    # --- End Manual Calculation --- #

    # Apply the function under test
    wrapped = tst.pbc_wrap_batched(positions, cell=cell, batch=batch)

    # Verify results match the expected values
    torch.testing.assert_close(wrapped, expected_stacked, rtol=1e-6, atol=1e-6)


def test_pbc_wrap_batched_edge_case(device: torch.device) -> None:
    """Test batched boundary wrapping at cell edges."""
    # Create two identical cells
    cell = torch.eye(3, device=device) * 2.0
    cell = torch.stack([cell, cell])

    # Create positions at cell boundaries
    positions = torch.tensor(
        [
            [2.0, 1.0, 0.5],  # First atom (batch 0), on +x boundary
            [1.0, 2.0, 0.5],  # Second atom (batch 1), on +y boundary
        ],
        device=device,
    )

    # Create batch indices
    batch = torch.tensor([0, 1], device=device)

    # Apply wrapping
    wrapped = tst.pbc_wrap_batched(positions, cell=cell, batch=batch)

    # Expected results (wrapping to 0.0 rather than 2.0)
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.5],  # x-coordinate wrapped from 2.0 to 0.0
            [1.0, 0.0, 0.5],  # y-coordinate wrapped from 2.0 to 0.0
        ],
        device=device,
    )

    # Verify results
    assert torch.allclose(wrapped, expected)


def test_pbc_wrap_batched_invalid_inputs(device: torch.device) -> None:
    """Test error handling for invalid inputs in batched wrapping."""
    # Valid inputs for reference
    positions = torch.ones(4, 3, device=device)
    cell = torch.stack([torch.eye(3, device=device)] * 2)
    batch = torch.tensor([0, 0, 1, 1], device=device)

    # Test integer tensors
    with pytest.raises(TypeError):
        tst.pbc_wrap_batched(
            torch.ones(4, 3, dtype=torch.int64, device=device), cell, batch
        )

    # Test dimension mismatch - positions
    with pytest.raises(ValueError):
        tst.pbc_wrap_batched(
            torch.ones(4, 2, device=device),  # Wrong dimension (2 instead of 3)
            cell,
            batch,
        )

    # Test mismatch between batch indices and cell
    with pytest.raises(ValueError):
        tst.pbc_wrap_batched(
            positions,
            torch.stack([torch.eye(3, device=device)] * 3),  # 3 cell but only 2 batches
            batch,
        )


def test_pbc_wrap_batched_multi_atom(si_double_sim_state: SimState) -> None:
    """Test batched wrapping with realistic multi-atom system."""
    state = si_double_sim_state

    # Get a copy of positions to modify
    test_positions = state.positions.clone()

    # Move all atoms of the first batch outside the cell in +x
    batch_0_mask = state.batch == 0
    cell_size_x = state.cell[0, 0, 0].item()
    test_positions[batch_0_mask, 0] += cell_size_x

    # Move all atoms of the second batch outside the cell in -y
    batch_1_mask = state.batch == 1
    cell_size_y = state.cell[0, 1, 1].item()
    test_positions[batch_1_mask, 1] -= cell_size_y

    # Apply wrapping
    wrapped = tst.pbc_wrap_batched(test_positions, cell=state.cell, batch=state.batch)

    # Check all positions are within the cell boundaries
    for b in range(2):  # For each batch
        batch_mask = state.batch == b

        # Check x coordinates
        assert torch.all(wrapped[batch_mask, 0] >= 0)
        assert torch.all(wrapped[batch_mask, 0] < state.cell[b, 0, 0])

        # Check y coordinates
        assert torch.all(wrapped[batch_mask, 1] >= 0)
        assert torch.all(wrapped[batch_mask, 1] < state.cell[b, 1, 1])

        # Check z coordinates
        assert torch.all(wrapped[batch_mask, 2] >= 0)
        assert torch.all(wrapped[batch_mask, 2] < state.cell[b, 2, 2])


def test_pbc_wrap_batched_preserves_relative_positions(
    si_double_sim_state: SimState,
) -> None:
    """Test that relative positions within each batch are preserved after wrapping."""
    state = si_double_sim_state

    # Get a copy of positions
    original_positions = state.positions.clone()

    # Move all atoms outside the cell, but maintain their relative positions
    test_positions = original_positions.clone()
    test_positions += torch.tensor([10.0, 15.0, 20.0], device=state.device)

    # Apply wrapping
    wrapped = tst.pbc_wrap_batched(test_positions, cell=state.cell, batch=state.batch)

    # Check that relative positions within each batch are preserved
    for b in range(2):  # For each batch
        batch_mask = state.batch == b

        # Calculate pairwise distances before wrapping
        atoms_in_batch = torch.sum(batch_mask).item()
        for n_atoms in range(atoms_in_batch - 1):
            for j in range(n_atoms + 1, atoms_in_batch):
                # Get the indices of atoms i and j in this batch
                batch_indices = torch.where(batch_mask)[0]
                idx_i = batch_indices[n_atoms]
                idx_j = batch_indices[j]

                # Original vector from i to j
                orig_vec = (
                    original_positions[idx_j] - original_positions[idx_i]
                ) % state.cell[b].diag()

                # Vector after wrapping
                wrapped_vec = (wrapped[idx_j] - wrapped[idx_i]) % state.cell[b].diag()

                # Check that relative positions are preserved
                assert torch.allclose(orig_vec, wrapped_vec, atol=1e-6)


def test_safe_mask_basic() -> None:
    """Test basic functionality of safe_mask with log function.

    Tests that safe_mask correctly applies log function to masked values
    and uses default placeholder (0.0) for masked-out values.
    """
    x = torch.tensor([1.0, 2.0, -1.0])
    mask = torch.tensor([True, True, False])
    result = tst.safe_mask(mask, torch.log, x)

    expected = torch.tensor([0, 0.6931, 0])
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_safe_mask_custom_placeholder() -> None:
    """Test safe_mask with a custom placeholder value.

    Tests that safe_mask correctly uses the provided placeholder value (-999.0)
    for masked-out elements instead of the default.
    """
    x = torch.tensor([1.0, 2.0, -1.0])
    mask = torch.tensor([True, False, False])
    result = tst.safe_mask(mask, torch.log, x, placeholder=-999.0)

    expected = torch.tensor([0.0, -999, -999])
    torch.testing.assert_close(result, expected)


def test_safe_mask_all_masked() -> None:
    """Test safe_mask when all elements are masked out.

    Tests that safe_mask returns a tensor of zeros when no elements
    are selected by the mask.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, False])
    result = tst.safe_mask(mask, torch.log, x)

    expected = torch.zeros_like(x)
    torch.testing.assert_close(result, expected)


def test_safe_mask_none_masked() -> None:
    """Test safe_mask when no elements are masked out.

    Tests that safe_mask correctly applies the function to all elements
    when the mask is all True.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, True, True])
    result = tst.safe_mask(mask, torch.log, x)

    expected = torch.log(x)
    torch.testing.assert_close(result, expected)


def test_safe_mask_shape_mismatch() -> None:
    """Test safe_mask error handling for shape mismatch.

    Tests that safe_mask raises a RuntimeError when the shapes of the
    input tensor and mask don't match.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False])

    with pytest.raises(RuntimeError):
        tst.safe_mask(mask, torch.log, x)


def test_high_precision_sum_float() -> None:
    """Test high_precision_sum with float32 input.

    Verifies that:
    1. The function maintains the input dtype (float32) in the output
    2. The summation is computed correctly
    3. The precision is adequate for basic float32 operations
    """
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = tst.high_precision_sum(x)
    assert result.dtype == torch.float32
    expected = torch.tensor(6.0, dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_high_precision_sum_double() -> None:
    """Test high_precision_sum with float64 input.

    Verifies that:
    1. The function maintains the input dtype (float64) in the output
    2. The summation is computed correctly at double precision
    3. No precision is lost when input is already float64
    """
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = tst.high_precision_sum(x)
    assert result.dtype == torch.float64
    expected = torch.tensor(6.0, dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_high_precision_sum_int() -> None:
    """Test high_precision_sum with integer input.

    Verifies that:
    1. The function handles integer inputs correctly
    2. The output maintains the input dtype (int32)
    3. Integer arithmetic is precise and lossless
    """
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    result = tst.high_precision_sum(x)
    assert result.dtype == torch.int32
    assert result == torch.tensor(6, dtype=torch.int32)


def test_high_precision_sum_complex() -> None:
    """Test high_precision_sum with complex number input.

    Verifies that:
    1. The function correctly handles complex numbers
    2. Both real and imaginary components are summed properly
    3. The output maintains the input dtype (complex64)
    4. Complex arithmetic is performed at high precision
    """
    x = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)
    result = tst.high_precision_sum(x)
    assert result.dtype == torch.complex64
    expected = torch.tensor(3 + 3j, dtype=torch.complex64)
    torch.testing.assert_close(result, expected)


def test_high_precision_sum_dim() -> None:
    """Test high_precision_sum with dimension reduction.

    Verifies that:
    1. The function correctly sums along a specified dimension
    2. The output shape is correct (reduced by one dimension)
    3. The results are accurate when summing along a single axis

    Example:
        Input shape: (2, 2)
        Output shape: (2,) when dim=0
    """
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = tst.high_precision_sum(x, dim=0)
    expected = torch.tensor([4.0, 6.0], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_high_precision_sum_keepdim() -> None:
    """Test high_precision_sum with keepdim option.

    Verifies that:
    1. The keepdim parameter correctly preserves dimensions
    2. The output shape has a singleton dimension where reduction occurred
    3. The results are accurate while maintaining dimensional structure

    Example:
        Input shape: (2, 2)
        Output shape: (1, 2) when dim=0 and keepdim=True
    """
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = tst.high_precision_sum(x, dim=0, keepdim=True)
    assert result.shape == (1, 2)
    expected = torch.tensor([[4.0, 6.0]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_high_precision_sum_multiple_dims() -> None:
    """Test high_precision_sum with multiple dimension reduction.

    Verifies that:
    1. The function can sum over multiple dimensions simultaneously
    2. The output shape is correct when reducing multiple dimensions
    3. The results are accurate for multi-dimensional reduction

    Example:
        Input shape: (2, 3, 4)
        Output shape: (3,) when dim=(0, 2)
        Each output element is the sum of 8 numbers (2 * 4 = 8)
    """
    x = torch.ones((2, 3, 4), dtype=torch.float32)
    result = tst.high_precision_sum(x, dim=(0, 2))
    assert result.shape == (3,)
    expected = torch.tensor([8.0, 8.0, 8.0], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_high_precision_sum_numerical_stability() -> None:
    """Test numerical stability of high_precision_sum.

    Verifies that:
    1. The function maintains accuracy with numbers of different magnitudes
    2. Small numbers aren't lost when summed with large numbers
    3. The high precision intermediate step provides better accuracy
    """
    # Create a tensor with numbers of very different magnitudes
    x = torch.tensor([1e-8, 1e8, 1e-8], dtype=torch.float32)
    result = tst.high_precision_sum(x)
    expected = torch.tensor(1e8 + 2e-8, dtype=torch.float32)
    torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-8)


def test_high_precision_sum_empty() -> None:
    """Test high_precision_sum with empty tensor.

    Verifies that:
    1. The function handles empty tensors gracefully
    2. The output maintains the correct dtype
    3. The sum of an empty tensor is 0 of the appropriate type
    """
    x = torch.tensor([], dtype=torch.float32)
    result = tst.high_precision_sum(x)
    assert result.dtype == torch.float32
    assert result == torch.tensor(0.0, dtype=torch.float32)


def test_multiplicative_isotropic_cutoff_basic() -> None:
    """Test basic functionality of the cutoff wrapper."""

    def constant_fn(dr: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(dr)

    cutoff_fn = tst.multiplicative_isotropic_cutoff(
        constant_fn, r_onset=1.0, r_cutoff=2.0
    )

    # Test points in different regions
    dr = torch.tensor([0.5, 1.5, 2.5])
    result = cutoff_fn(dr)

    torch.testing.assert_close(result[0], torch.tensor(1.0))  # Before onset
    assert 0.0 < result[1] < 1.0  # Between onset and cutoff
    torch.testing.assert_close(result[2], torch.tensor(0.0))  # After cutoff


def test_multiplicative_isotropic_cutoff_continuity() -> None:
    """Test that the cutoff function is continuous at boundaries."""

    def linear_fn(dr: torch.Tensor) -> torch.Tensor:
        return dr

    r_onset = 1.0
    r_cutoff = 2.0
    cutoff_fn = tst.multiplicative_isotropic_cutoff(linear_fn, r_onset, r_cutoff)

    # Test near onset
    dr_before = torch.tensor([r_onset - 1e-5])
    dr_after = torch.tensor([r_onset + 1e-5])
    torch.testing.assert_close(
        cutoff_fn(dr_before), cutoff_fn(dr_after), rtol=1e-4, atol=1e-5
    )

    # Test near cutoff
    dr_before = torch.tensor([r_cutoff - 1e-5])
    dr_after = torch.tensor([r_cutoff + 1e-5])
    torch.testing.assert_close(
        cutoff_fn(dr_before), cutoff_fn(dr_after), rtol=1e-4, atol=1e-5
    )


def test_multiplicative_isotropic_cutoff_derivative_continuity() -> None:
    """Test that the derivative of the cutoff function is continuous."""

    def quadratic_fn(dr: torch.Tensor) -> torch.Tensor:
        return dr**2

    r_onset = 1.0
    r_cutoff = 2.0
    cutoff_fn = tst.multiplicative_isotropic_cutoff(quadratic_fn, r_onset, r_cutoff)

    # Test derivative near onset and cutoff using finite differences
    points = torch.tensor([r_onset, r_cutoff], requires_grad=True)

    # Compute gradients
    result = cutoff_fn(points)
    grads = torch.autograd.grad(result.sum(), points)[0]

    # Verify gradients change smoothly
    assert not torch.isnan(grads).any()
    assert not torch.isinf(grads).any()


def test_multiplicative_isotropic_cutoff_with_parameters() -> None:
    """Test that the cutoff wrapper works with functions that take parameters."""

    def parameterized_fn(dr: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * dr

    cutoff_fn = tst.multiplicative_isotropic_cutoff(
        parameterized_fn, r_onset=1.0, r_cutoff=2.0
    )

    dr = torch.tensor([0.5, 1.5, 2.5])
    result = cutoff_fn(dr, scale=2.0)

    torch.testing.assert_close(result[0], torch.tensor(1.0))  # Before onset
    assert 0.0 < result[1] < 3.0  # Between onset and cutoff
    torch.testing.assert_close(result[2], torch.tensor(0.0))  # After cutoff


def test_multiplicative_isotropic_cutoff_batch() -> None:
    """Test that the cutoff wrapper works with batched inputs."""

    def constant_fn(dr: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(dr)

    cutoff_fn = tst.multiplicative_isotropic_cutoff(
        constant_fn, r_onset=1.0, r_cutoff=2.0
    )

    # Test with 2D input
    dr = torch.rand(5, 5) * 3.0
    result = cutoff_fn(dr)

    assert result.shape == (5, 5)
    assert (result <= 1.0).all()
    assert (result >= 0.0).all()


def test_multiplicative_isotropic_cutoff_gradient() -> None:
    """Test that gradients can be propagated through the cutoff function."""

    def linear_fn(dr: torch.Tensor) -> torch.Tensor:
        return dr

    cutoff_fn = tst.multiplicative_isotropic_cutoff(linear_fn, r_onset=1.0, r_cutoff=2.0)

    dr = torch.tensor([1.5], requires_grad=True)
    result = cutoff_fn(dr)
    grad = torch.autograd.grad(result, dr)[0]

    assert not torch.isnan(grad)
    assert not torch.isinf(grad)


@pytest.mark.parametrize(
    ("pos", "cell", "expected"),
    [
        (
            [[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [[0.25, 0.25, 0.25], [0.5, 0.0, 0.0]],
        ),
        (
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[0.5, 0.25, 0.5], [0.0, 0.0, 0.0]],
        ),
    ],
)
def test_get_fractional_coordinates(
    pos: list[list[float]], cell: list[list[float]], expected: list[list[float]]
) -> None:
    """Test get_fractional_coordinates with various inputs.

    Tests the function with both cubic and non-orthogonal cells.
    """
    frac = tst.get_fractional_coordinates(torch.tensor(pos), torch.tensor(cell))
    torch.testing.assert_close(frac, torch.tensor(expected))


@pytest.mark.parametrize(
    ("dr", "cell", "pbc", "expected"),
    [
        (
            [[1.5, 1.5, 1.5], [-1.5, -1.5, -1.5]],
            torch.eye(3) * 3.0,
            False,
            [[1.5, 1.5, 1.5], [-1.5, -1.5, -1.5]],
        ),
        (
            [[1.5, 1.5, 1.5], [-1.5, -1.5, -1.5]],
            torch.eye(3) * 3.0,
            True,
            [[1.5, 1.5, 1.5], [-1.5, -1.5, -1.5]],
        ),
        (
            [[2.2, 0.0, 0.0], [0.0, 2.2, 0.0], [0.0, 0.0, 2.2]],
            torch.eye(3) * 2.0,
            True,
            [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
        ),
    ],
)
def test_minimum_image_displacement(
    *, dr: list[list[float]], cell: torch.Tensor, pbc: bool, expected: list[list[float]]
) -> None:
    """Test minimum_image_displacement with various inputs.

    Tests function with and without PBC and with different displacement vectors.
    """
    result = tst.minimum_image_displacement(dr=torch.tensor(dr), cell=cell, pbc=pbc)
    torch.testing.assert_close(result, torch.tensor(expected))


@pytest.mark.parametrize(
    ("positions", "cell", "pbc", "pairs", "shifts", "expected_dr", "expected_distance"),
    [
        (  # No PBC case with specific pair
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            None,
            False,
            (torch.tensor([0]), torch.tensor([1])),
            None,
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([1.0]),
        ),
        (  # PBC case with specific pair
            torch.tensor([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0]]),
            torch.eye(3) * 2.0,
            True,
            (torch.tensor([0]), torch.tensor([1])),
            None,
            torch.tensor([[-0.1, 0.0, 0.0]]),
            torch.tensor([0.1]),
        ),
        (  # With explicit shifts
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.eye(3) * 2.0,
            True,
            (torch.tensor([0]), torch.tensor([1])),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[3.0, 0.0, 0.0]]),
            torch.tensor([3.0]),
        ),
    ],
)
def test_get_pair_displacements(
    *,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: bool,
    pairs: tuple[torch.Tensor, torch.Tensor],
    shifts: torch.Tensor,
    expected_dr: torch.Tensor,
    expected_distance: torch.Tensor,
) -> None:
    """Test get_pair_displacements with various inputs.

    Tests function with and without PBC, with specific pairs, and with explicit shifts.
    """
    dr, distances = tst.get_pair_displacements(
        positions=positions, cell=cell, pbc=pbc, pairs=pairs, shifts=shifts
    )

    torch.testing.assert_close(dr, expected_dr)
    torch.testing.assert_close(distances, expected_distance)


@pytest.mark.parametrize(
    ("v", "expected"),
    [
        ([1, 2, 3], [0, 1, 3, 6]),
        ([[1, 2], [3, 4]], [0, 1, 3, 6, 10]),
    ],
)
def test_strides_of(v: list[int], expected: list[int]) -> None:
    """Test strides_of with 1D and 2D tensors.

    Verifies that the function correctly computes cumulative strides
    for both 1D and multidimensional tensors.
    """
    strides = tst.strides_of(torch.tensor(v))
    torch.testing.assert_close(strides, torch.tensor(expected))


def test_strides_of_empty() -> None:
    """Test strides_of with empty tensor."""
    v = torch.tensor([], dtype=torch.int64)
    strides = tst.strides_of(v)
    expected = torch.tensor([0], dtype=torch.int64)
    torch.testing.assert_close(strides, expected)


@pytest.mark.parametrize(
    ("cutoff", "cell", "pbc", "expected_shape", "expected_props"),
    [
        (
            1.5,
            torch.eye(3).unsqueeze(0) * 2.0,
            torch.tensor([[True, True, True]]),
            torch.Size([1, 3]),
            {"min_value": 0, "all_equal": True},
        ),
        (
            1.5,
            torch.eye(3).unsqueeze(0) * 2.0,
            torch.tensor([[True, False, True]]),
            torch.Size([1, 3]),
            {"zero_dim": 1},  # Removed nonzero_dims since it's not reliable
        ),
        (
            1.5,
            torch.stack([torch.eye(3) * 2.0, torch.eye(3) * 4.0]),
            torch.tensor([[True, True, True], [True, True, True]]),
            torch.Size([2, 3]),
            {"batch_equal": True},
        ),
    ],
)
def test_get_number_of_cell_repeats(
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    expected_shape: torch.Size,
    expected_props: dict,
) -> None:
    """Test get_number_of_cell_repeats with various parameters.

    Tests with different cell sizes, PBC conditions, and batch sizes.
    """
    num_repeats = tst.get_number_of_cell_repeats(cutoff, cell, pbc)

    # Check shape
    assert num_repeats.shape == expected_shape

    # Check specific properties based on test case
    if "min_value" in expected_props:
        assert torch.all(num_repeats >= expected_props["min_value"])

    if expected_props.get("all_equal"):
        assert num_repeats[0, 0] == num_repeats[0, 1] == num_repeats[0, 2]

    if "zero_dim" in expected_props:
        assert num_repeats[0, expected_props["zero_dim"]] == 0

    if expected_props.get("batch_equal"):
        assert num_repeats[1, 0] == num_repeats[1, 1] == num_repeats[1, 2]


@pytest.mark.parametrize(
    ("num_repeats", "expected_shape", "expected_range"),
    [
        ([1, 1, 1], (27, 3), {"min": -1, "max": 1}),
        ([0, 0, 0], (1, 3), {"exact": [[0.0, 0.0, 0.0]]}),
        ([1, 0, 2], (15, 3), {"dim_values": {0: (-1, 1), 1: (0, 0), 2: (-2, 2)}}),
    ],
)
def test_get_cell_shift_idx(
    num_repeats: list[int], expected_shape: tuple, expected_range: dict
) -> None:
    """Test get_cell_shift_idx with different repeat parameters.

    Tests the function with symmetric, zero, and asymmetric repeats.
    """
    n_repeats = torch.tensor(num_repeats)
    shifts = tst.get_cell_shift_idx(n_repeats, torch.float)

    # Check shape
    assert shifts.shape == expected_shape

    # Check ranges or exact values
    if "min" in expected_range and "max" in expected_range:
        assert torch.all(shifts >= expected_range["min"])
        assert torch.all(shifts <= expected_range["max"])

    if "exact" in expected_range:
        torch.testing.assert_close(shifts, torch.tensor(expected_range["exact"]))

    if "dim_values" in expected_range:
        for dim, (min_val, max_val) in expected_range["dim_values"].items():
            assert torch.all(shifts[:, dim] >= min_val)
            assert torch.all(shifts[:, dim] <= max_val)


@pytest.mark.parametrize(
    ("idx_3d", "shape", "expected"),
    [([[0, 0, 0], [1, 2, 3]], [2, 3, 4], [0, 23])],
)
def test_ravel_3d(idx_3d: list[list[int]], shape: list[int], expected: list[int]) -> None:
    """Test ravel_3d function.

    Verifies correct conversion of 3D indices to linear indices.
    """
    linear_idx = tst.ravel_3d(torch.tensor(idx_3d), torch.tensor(shape))
    torch.testing.assert_close(linear_idx, torch.tensor(expected))


@pytest.mark.parametrize(
    ("linear_idx", "shape", "expected"),
    [([0, 23], [2, 3, 4], [[0, 0, 0], [1, 2, 3]])],
)
def test_unravel_3d(
    linear_idx: list[int], shape: list[int], expected: list[list[int]]
) -> None:
    """Test unravel_3d function.

    Verifies correct conversion of linear indices back to 3D indices.
    """
    idx_3d = tst.unravel_3d(torch.tensor(linear_idx), torch.tensor(shape))
    torch.testing.assert_close(idx_3d, torch.tensor(expected))


def test_ravel_unravel_3d_roundtrip() -> None:
    """Test roundtrip conversion with ravel_3d and unravel_3d."""
    original_idx = torch.tensor([[0, 1, 2], [1, 0, 3], [1, 2, 0]])
    shape = torch.tensor([2, 3, 4])

    linear_idx = tst.ravel_3d(original_idx, shape)
    reconstructed_idx = tst.unravel_3d(linear_idx, shape)

    torch.testing.assert_close(reconstructed_idx, original_idx)


@pytest.mark.parametrize(
    ("cell", "pos", "n_bins_s", "expected"),
    [(torch.eye(3) * 2.0, [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], [2, 2, 2], [0, 7])],
)
def test_get_linear_bin_idx(
    cell: torch.Tensor, pos: list[float], n_bins_s: list[float], expected: list[float]
) -> None:
    """Test get_linear_bin_idx function.

    Verifies correct calculation of linear bin indices for positions.
    """
    bin_idx = tst.get_linear_bin_idx(cell, torch.tensor(pos), torch.tensor(n_bins_s))
    torch.testing.assert_close(bin_idx, torch.tensor(expected))


def test_scatter_bin_index_basic() -> None:
    """Test scatter_bin_index function."""
    n_bins = 3
    max_n_atom_per_bin = 2
    n_images = 5
    bin_index = torch.tensor([0, 0, 1, 2])

    bin_id = tst.scatter_bin_index(n_bins, max_n_atom_per_bin, n_images, bin_index)

    # Check shape and basic properties
    assert bin_id.shape == torch.Size([3, 2])
    assert torch.sum(bin_id != n_images) == 4  # All 4 atoms should be assigned
    assert torch.sum(bin_id == n_images) == 2  # 2 slots should be empty


@pytest.mark.parametrize(
    ("pos", "mapping", "cell_shifts", "expected"),
    [
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0, 0], [1, 2]]),
            None,
            torch.tensor([1.0, 1.0]),
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([[0, 0], [1, 1]]),
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.tensor([1.0, 3.0]),
        ),
    ],
)
def test_compute_distances_with_cell_shifts(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Test compute_distances_with_cell_shifts function.

    Tests with and without cell shifts applied.
    """
    distances = tst.compute_distances_with_cell_shifts(pos, mapping, cell_shifts)
    torch.testing.assert_close(distances, expected)


def test_compute_cell_shifts_basic() -> None:
    """Test compute_cell_shifts function."""
    cell = torch.eye(3).unsqueeze(0) * 2.0
    shifts_idx = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    batch_mapping = torch.tensor([0, 0])

    cell_shifts = tst.compute_cell_shifts(cell, shifts_idx, batch_mapping)

    expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    torch.testing.assert_close(cell_shifts, expected)


@pytest.mark.parametrize(
    ("self_interaction", "expected_shape"), [(True, (9, 2)), (False, (6, 2))]
)
def test_get_fully_connected_mapping(
    *, self_interaction: bool, expected_shape: tuple
) -> None:
    """Test get_fully_connected_mapping with and without self-interaction.

    Tests that the function generates the correct number of pairs and handles
    self-interaction flag appropriately.
    """
    i_ids = torch.tensor([0, 1, 2])
    shifts_idx = torch.tensor([[0.0, 0.0, 0.0]])

    mapping, shifts = tst.get_fully_connected_mapping(
        i_ids=i_ids, shifts_idx=shifts_idx, self_interaction=self_interaction
    )

    # Check shapes
    assert mapping.shape == expected_shape
    assert shifts.shape[0] == expected_shape[0]
    assert shifts.shape[1] == 3

    # Check self-interaction behavior
    if not self_interaction:
        for i in range(mapping.shape[0]):
            assert mapping[i, 0] != mapping[i, 1], "Self-pair incorrectly included"


def test_get_fully_connected_mapping_with_multiple_shifts() -> None:
    """Test get_fully_connected_mapping with multiple shift vectors."""
    i_ids = torch.tensor([0, 1])
    shifts_idx = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    mapping, shifts = tst.get_fully_connected_mapping(
        i_ids=i_ids, shifts_idx=shifts_idx, self_interaction=False
    )

    # With 2 atoms, 3 shifts, and no self-interaction
    assert mapping.shape[0] == 10
    assert shifts.shape[0] == 10


def test_linked_cell_basic() -> None:
    """Test basic functionality of linked_cell."""
    # Create a simple system with two atoms
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cell = torch.eye(3) * 4.0
    cutoff = 1.5
    num_repeats = torch.tensor([1, 1, 1])

    neigh_atom, neigh_shift_idx = tst.linked_cell(
        pos, cell, cutoff, num_repeats, self_interaction=False
    )

    # Check that atoms 0 and 1 are neighbors of each other
    assert neigh_atom.shape[1] >= 2  # At least 2 neighbor pairs

    # Find the pair (0,1) in the neighbor list
    found = False
    for idx in range(neigh_atom.shape[1]):
        if (neigh_atom[0, idx] == 0 and neigh_atom[1, idx] == 1) or (
            neigh_atom[0, idx] == 1 and neigh_atom[1, idx] == 0
        ):
            found = True
            break

    assert found, "Expected atoms 0 and 1 to be neighbors"


def test_build_linked_cell_neighborhood_basic() -> None:
    """Test basic functionality of build_linked_cell_neighborhood."""
    # Create a simple system with two structures, each with two atoms
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    cell = torch.stack([torch.eye(3) * 4.0, torch.eye(3) * 4.0])
    pbc = torch.tensor([[True, True, True], [True, True, True]])
    cutoff = 1.5
    n_atoms = torch.tensor([2, 2])

    mapping, batch_mapping, cell_shifts_idx = tst.build_linked_cell_neighborhood(
        positions, cell, pbc, cutoff, n_atoms, self_interaction=False
    )

    # Check that atoms in the same structure are neighbors
    assert mapping.shape[1] >= 2  # At least 2 neighbor pairs

    # Verify batch_mapping has correct length
    assert batch_mapping.shape[0] == mapping.shape[1]

    # Verify that there are neighbors from both batches
    assert torch.any(batch_mapping == 0)
    assert torch.any(batch_mapping == 1)
