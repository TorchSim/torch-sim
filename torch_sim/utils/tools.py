"""Utility functions for the torchsim package."""

from collections.abc import Callable, Iterable
from functools import wraps

import torch


def safe_mask(
    mask: torch.Tensor,
    fn: torch.jit.ScriptFunction,
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
        tensor([0.0000, 0.6931, 0.0000])
    """
    masked = torch.where(mask, operand, torch.zeros_like(operand))
    return torch.where(mask, fn(masked), torch.full_like(operand, placeholder))


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


def multiplicative_isotropic_cutoff(
    fn: Callable[..., torch.Tensor],
    r_onset: torch.Tensor,
    r_cutoff: torch.Tensor,
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
        https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
    """
    r_c = r_cutoff**2
    r_o = r_onset**2

    def smooth_fn(dr: torch.Tensor) -> torch.Tensor:
        """Compute the smooth switching function."""
        r = dr**2

        # Compute switching function for intermediate region
        numerator = (r_c - r) ** 2 * (r_c + 2 * r - 3 * r_o)
        denominator = (r_c - r_o) ** 3
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


def calculate_momenta(
    positions: torch.Tensor,
    masses: torch.Tensor,
    kT: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    """Calculate momenta from positions and masses."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Generate random momenta from normal distribution
    momenta = torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    ) * torch.sqrt(masses * kT).unsqueeze(-1)

    # Center the momentum if more than one particle
    if positions.shape[0] > 1:
        momenta = momenta - torch.mean(momenta, dim=0, keepdim=True)

    return momenta
