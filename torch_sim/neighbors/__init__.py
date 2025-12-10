"""Neighbor list implementations for torch-sim.

This module provides multiple neighbor list implementations with automatic
fallback based on available dependencies. The API supports both single-system
and batched (multi-system) calculations.

Available Implementations:
    - Primitive: Pure PyTorch implementation (always available)
    - Vesin: High-performance neighbor lists (optional, requires vesin package)
    - Batched: Optimized for multiple systems (torch_nl_n2, torch_nl_linked_cell)

Default Neighbor Lists:
    The module automatically selects the best available implementation:
    - For single systems: vesin_nl (if available) or standard_nl (fallback)
    - For batched systems: torch_nl_linked_cell (always available)
"""

import torch

from torch_sim.neighbors.standard import primitive_neighbor_list, standard_nl
from torch_sim.neighbors.torch_nl import strict_nl, torch_nl_linked_cell, torch_nl_n2


# Try to import Vesin implementations
try:
    from torch_sim.neighbors.vesin import (
        VESIN_AVAILABLE,
        VesinNeighborList,
        VesinNeighborListTorch,
        vesin_nl,
        vesin_nl_ts,
    )
except ImportError:
    VESIN_AVAILABLE = False
    VesinNeighborList = None  # type: ignore[assignment,misc]
    VesinNeighborListTorch = None  # type: ignore[assignment,misc]
    vesin_nl = None  # type: ignore[assignment]
    vesin_nl_ts = None  # type: ignore[assignment]

# Set default neighbor list based on what's available
if VESIN_AVAILABLE:
    default_nl = vesin_nl
    default_nl_ts = vesin_nl_ts
else:
    default_nl = standard_nl
    default_nl_ts = standard_nl

# For batched calculations, always use linked cell as default
default_batched_nl = torch_nl_linked_cell


def torchsim_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    system_idx: torch.Tensor,
    self_interaction: bool = False,  # noqa: FBT001, FBT002
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute neighbor lists with automatic fallback for AMD ROCm compatibility.

    This function automatically selects the best available neighbor list implementation.
    When vesin is available, it uses vesin_nl_ts for optimal performance. When vesin
    is not available (e.g., on AMD ROCm systems), it falls back to standard_nl.

    Args:
        positions: Atomic positions tensor [n_atoms, 3]
        cell: Unit cell vectors [3*n_systems, 3] (row vector convention)
        pbc: Boolean tensor [3] or [n_systems, 3] for periodic boundary conditions
        cutoff: Maximum distance (scalar tensor) for considering atoms as neighbors
        system_idx: Tensor [n_atoms] indicating which system each atom belongs to.
            For single system, use torch.zeros(n_atoms, dtype=torch.long)
        self_interaction: If True, include self-pairs. Default: False

    Returns:
        tuple containing:
            - mapping: Tensor [2, num_neighbors] - pairs of atom indices
            - system_mapping: Tensor [num_neighbors] - system assignment for each pair
            - shifts_idx: Tensor [num_neighbors, 3] - periodic shift indices

    Notes:
        - Automatically uses vesin_nl_ts when vesin is available
        - Falls back to standard_nl when vesin is unavailable (AMD ROCm)
        - Fallback works on NVIDIA CUDA, AMD ROCm, and CPU
        - For non-periodic systems (pbc=False), shifts will be zero vectors
        - The neighbor list includes both (i,j) and (j,i) pairs
    """
    if not VESIN_AVAILABLE:
        return torch_nl_linked_cell(
            positions, cell, pbc, cutoff, system_idx, self_interaction
        )

    return vesin_nl_ts(positions, cell, pbc, cutoff, system_idx, self_interaction)


__all__ = [
    # Availability
    "VESIN_AVAILABLE",
    "VesinNeighborList",
    "VesinNeighborListTorch",
    # Defaults
    "default_batched_nl",
    "default_nl",
    "default_nl_ts",
    # Core implementations
    "primitive_neighbor_list",
    "standard_nl",
    # Utilities
    "strict_nl",
    # Batched implementations
    "torch_nl_linked_cell",
    "torch_nl_n2",
    "torchsim_nl",
    # Vesin implementations
    "vesin_nl",
    "vesin_nl_ts",
]
