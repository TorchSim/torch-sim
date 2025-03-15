"""Math utilities for torch_sim."""

import torch


@torch.jit.script
def torch_divmod(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute division and modulo operations for tensors.

    Args:
        a: Dividend tensor
        b: Divisor tensor

    Returns:
        tuple containing:
            - Quotient tensor
            - Remainder tensor
    """
    d = torch.div(a, b, rounding_mode="floor")
    m = a % b
    return d, m
