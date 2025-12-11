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


def _normalize_inputs(
    cell: torch.Tensor, pbc: torch.Tensor, n_systems: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize cell and PBC tensors to standard batch format.

    Handles multiple input formats:
    - cell: [3, 3], [n_systems, 3, 3], or [n_systems*3, 3]
    - pbc: [3], [n_systems, 3], or [n_systems*3]

    Returns:
        (cell, pbc) normalized to ([n_systems, 3, 3], [n_systems, 3])
        Both tensors are guaranteed to be contiguous.
    """
    # Normalize cell
    if cell.ndim == 2:
        if cell.shape[0] == 3:
            cell = cell.unsqueeze(0).expand(n_systems, -1, -1).contiguous()
        else:
            cell = cell.reshape(n_systems, 3, 3)

    # Normalize PBC
    if pbc.ndim == 1:
        if pbc.shape[0] == 3:
            pbc = pbc.unsqueeze(0).expand(n_systems, -1).contiguous()
        else:
            pbc = pbc.reshape(n_systems, 3)

    return cell, pbc


# Try to import Alchemiops implementations (NVIDIA CUDA acceleration)
try:
    from torch_sim.neighbors.alchemiops import ALCHEMIOPS_AVAILABLE, alchemiops_nl_n2
except ImportError:
    ALCHEMIOPS_AVAILABLE = False
    alchemiops_nl_n2 = None  # type: ignore[assignment]

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

# Set default neighbor list based on what's available (priority order)
if ALCHEMIOPS_AVAILABLE:
    # Alchemiops is fastest on NVIDIA GPUs
    default_batched_nl = alchemiops_nl_n2
elif VESIN_AVAILABLE:
    # Vesin is good fallback
    default_batched_nl = vesin_nl_ts  # Still use native for batched
else:
    # Pure PyTorch fallback
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
    if VESIN_AVAILABLE:
        return vesin_nl_ts(positions, cell, pbc, cutoff, system_idx, self_interaction)
    return torch_nl_linked_cell(
        positions, cell, pbc, cutoff, system_idx, self_interaction
    )


__all__ = [
    "ALCHEMIOPS_AVAILABLE",
    "VESIN_AVAILABLE",
    "VesinNeighborList",
    "VesinNeighborListTorch",
    "alchemiops_nl_n2",
    "default_batched_nl",
    "primitive_neighbor_list",
    "standard_nl",
    "strict_nl",
    "torch_nl_linked_cell",
    "torch_nl_n2",
    "torchsim_nl",
    "vesin_nl",
    "vesin_nl_ts",
]
