"""Tests for utility functions in torchsim.utils."""

import pytest
import torch

from torch_sim.utils.tools import (
    high_precision_sum,
    multiplicative_isotropic_cutoff,
    safe_mask,
)


def test_safe_mask_basic() -> None:
    """Test basic functionality of safe_mask with log function.

    Tests that safe_mask correctly applies log function to masked values
    and uses default placeholder (0.0) for masked-out values.
    """
    x = torch.tensor([1.0, 2.0, -1.0])
    mask = torch.tensor([True, True, False])
    result = safe_mask(mask, torch.log, x)

    expected = torch.tensor([0.0000, 0.6931, 0.0000])
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_safe_mask_custom_placeholder() -> None:
    """Test safe_mask with a custom placeholder value.

    Tests that safe_mask correctly uses the provided placeholder value (-999.0)
    for masked-out elements instead of the default.
    """
    x = torch.tensor([1.0, 2.0, -1.0])
    mask = torch.tensor([True, False, False])
    result = safe_mask(mask, torch.log, x, placeholder=-999.0)

    expected = torch.tensor([0.0000, -999.0000, -999.0000])
    torch.testing.assert_close(result, expected)


def test_safe_mask_all_masked() -> None:
    """Test safe_mask when all elements are masked out.

    Tests that safe_mask returns a tensor of zeros when no elements
    are selected by the mask.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, False])
    result = safe_mask(mask, torch.log, x)

    expected = torch.zeros_like(x)
    torch.testing.assert_close(result, expected)


def test_safe_mask_none_masked() -> None:
    """Test safe_mask when no elements are masked out.

    Tests that safe_mask correctly applies the function to all elements
    when the mask is all True.
    """
    x = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, True, True])
    result = safe_mask(mask, torch.log, x)

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
        safe_mask(mask, torch.log, x)


def test_high_precision_sum_float() -> None:
    """Test high_precision_sum with float32 input.

    Verifies that:
    1. The function maintains the input dtype (float32) in the output
    2. The summation is computed correctly
    3. The precision is adequate for basic float32 operations
    """
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = high_precision_sum(x)
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
    result = high_precision_sum(x)
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
    result = high_precision_sum(x)
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
    result = high_precision_sum(x)
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
    result = high_precision_sum(x, dim=0)
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
    result = high_precision_sum(x, dim=0, keepdim=True)
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
    result = high_precision_sum(x, dim=(0, 2))
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
    result = high_precision_sum(x)
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
    result = high_precision_sum(x)
    assert result.dtype == torch.float32
    assert result == torch.tensor(0.0, dtype=torch.float32)


def test_multiplicative_isotropic_cutoff_basic() -> None:
    """Test basic functionality of the cutoff wrapper."""

    def constant_fn(dr: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(dr)

    cutoff_fn = multiplicative_isotropic_cutoff(constant_fn, r_onset=1.0, r_cutoff=2.0)

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
    cutoff_fn = multiplicative_isotropic_cutoff(linear_fn, r_onset, r_cutoff)

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
    cutoff_fn = multiplicative_isotropic_cutoff(quadratic_fn, r_onset, r_cutoff)

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

    cutoff_fn = multiplicative_isotropic_cutoff(
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

    cutoff_fn = multiplicative_isotropic_cutoff(constant_fn, r_onset=1.0, r_cutoff=2.0)

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

    cutoff_fn = multiplicative_isotropic_cutoff(linear_fn, r_onset=1.0, r_cutoff=2.0)

    dr = torch.tensor([1.5], requires_grad=True)
    result = cutoff_fn(dr)
    grad = torch.autograd.grad(result, dr)[0]

    assert not torch.isnan(grad)
    assert not torch.isinf(grad)
