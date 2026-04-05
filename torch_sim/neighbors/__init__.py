"""Neighbor list implementations for torch-sim.

This module provides multiple neighbor list implementations with automatic
fallback based on available dependencies. The API supports both single-system
and batched (multi-system) calculations.

Available Implementations:
    - Alchemiops: Warp-accelerated neighbor list implementation
    - Vesin: High-performance neighbor list implementation
    - TorchNL: Pure PyTorch implementation

Default Neighbor Lists:
    The module automatically selects the best available implementation:
    - Priority: alchemiops_nl_n2 > vesin_nl_ts > torch_nl_linked_cell
"""

import torch

from torch_sim.neighbors.alchemiops import (
    ALCHEMIOPS_AVAILABLE,
    alchemiops_nl_cell_list,
    alchemiops_nl_n2,
)
from torch_sim.neighbors.torch_nl import strict_nl, torch_nl_linked_cell, torch_nl_n2
from torch_sim.neighbors.vesin import (
    VESIN_AVAILABLE,
    VESIN_TORCHSCRIPT_AVAILABLE,
    vesin_nl,
    vesin_nl_ts,
)


# Set default neighbor list based on what's available (priority order)
if ALCHEMIOPS_AVAILABLE:
    # Alchemiops is fastest on NVIDIA GPUs
    # TODO: why default to n2? we should document the cross-over point
    default_batched_nl = alchemiops_nl_n2
elif VESIN_TORCHSCRIPT_AVAILABLE:
    default_batched_nl = vesin_nl_ts
elif VESIN_AVAILABLE:
    default_batched_nl = vesin_nl
else:
    default_batched_nl = torch_nl_linked_cell


def torchsim_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    system_idx: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute neighbor lists with automatic selection of best available implementation.

    This function automatically selects the best available neighbor list implementation
    based on what's installed. Priority order:
    1. Alchemiops (NVIDIA CUDA optimized) if available
    2. Vesin (fast, cross-platform) if available
    3. torch_nl_linked_cell (pure PyTorch fallback)

    Args:
        positions: Atomic positions tensor [n_atoms, 3]
        cell: Unit cell vectors [n_systems, 3, 3] or [3, 3]
        pbc: Boolean tensor [n_systems, 3] or [3]
        cutoff: Maximum distance (scalar tensor) for considering atoms as neighbors
        system_idx: Tensor [n_atoms] indicating which system each atom belongs to
        self_interaction: If True, include self-pairs. Default: False

    Returns:
        tuple containing:
            - mapping: Tensor [2, num_neighbors] - pairs of atom indices
            - system_mapping: Tensor [num_neighbors] - system assignment for each pair
            - shifts_idx: Tensor [num_neighbors, 3] - periodic shift indices

    Notes:
        - Automatically uses best available implementation
        - Priority: Alchemiops > Vesin > torch_nl_linked_cell
        - Fallback works on NVIDIA CUDA, AMD ROCm, and CPU
        - For non-periodic systems (pbc=False), shifts will be zero vectors
        - The neighbor list includes both (i,j) and (j,i) pairs
        - Accepts both single-system [3, 3] or batched [n_systems, 3, 3] cell formats
        - Accepts both single [3] or batched [n_systems, 3] PBC formats
    """
    if ALCHEMIOPS_AVAILABLE:
        return alchemiops_nl_n2(
            positions, cell, pbc, cutoff, system_idx, self_interaction
        )

    if VESIN_TORCHSCRIPT_AVAILABLE:
        return vesin_nl_ts(positions, cell, pbc, cutoff, system_idx, self_interaction)

    if VESIN_AVAILABLE:
        return vesin_nl(positions, cell, pbc, cutoff, system_idx, self_interaction)

    return torch_nl_linked_cell(
        positions, cell, pbc, cutoff, system_idx, self_interaction
    )


__all__ = [
    "ALCHEMIOPS_AVAILABLE",
    "ALCHEMIOPS_TORCH_AVAILABLE",
    "VESIN_AVAILABLE",
    "VESIN_TORCHSCRIPT_AVAILABLE",
    "alchemiops_nl_cell_list",
    "alchemiops_nl_n2",
    "strict_nl",
    "torch_nl_linked_cell",
    "torch_nl_n2",
    "vesin_nl",
    "vesin_nl_ts",
]
