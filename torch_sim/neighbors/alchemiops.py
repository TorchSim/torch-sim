"""Alchemiops-based neighbor list implementations.

This module provides high-performance CUDA-accelerated neighbor list calculations
using the nvalchemiops library. Uses the naive N^2 implementation for reliability.

nvalchemiops is available at: https://github.com/NVIDIA/nvalchemiops
"""

import warnings

import torch


try:
    from nvalchemiops.neighborlist import (
        batch_naive_neighbor_list as _batch_naive_neighbor_list,
    )
    from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors

    ALCHEMIOPS_AVAILABLE = True
except ImportError:
    ALCHEMIOPS_AVAILABLE = False
    _batch_naive_neighbor_list = None  # type: ignore[assignment]
    estimate_max_neighbors = None  # type: ignore[assignment]

__all__ = [
    "ALCHEMIOPS_AVAILABLE",
    "alchemiops_nl_n2",
]


def _prepare_inputs(cell: torch.Tensor, pbc: torch.Tensor, system_idx: torch.Tensor):  # noqa: ANN202
    """Prepare cell and PBC tensors for alchemiops functions.

    Ensures tensors are properly shaped and contiguous for Warp backend.
    """
    n_systems = system_idx.max().item() + 1

    # Reshape cell: [3*n_systems, 3] or [3, 3] -> [n_systems, 3, 3]
    if cell.ndim == 2:
        cell_reshaped = (
            cell.unsqueeze(0) if cell.shape[0] == 3 else cell.reshape(n_systems, 3, 3)
        )
    else:
        cell_reshaped = cell

    # Reshape PBC: various formats -> [n_systems, 3]
    if pbc.ndim == 1:
        pbc_reshaped = (
            pbc.unsqueeze(0).expand(n_systems, -1)
            if pbc.shape[0] == 3
            else pbc.reshape(n_systems, 3)
        )
    else:
        pbc_reshaped = pbc

    # Ensure tensors are contiguous for Warp backend
    cell_reshaped = cell_reshaped.contiguous()
    pbc_reshaped = pbc_reshaped.contiguous()

    return cell_reshaped, pbc_reshaped.to(torch.bool), n_systems


if ALCHEMIOPS_AVAILABLE:

    def alchemiops_nl_n2(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: torch.Tensor,
        system_idx: torch.Tensor,
        self_interaction: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute neighbor lists using Alchemiops naive N^2 algorithm.

        Args:
            positions: Atomic positions tensor [n_atoms, 3]
            cell: Unit cell vectors [3*n_systems, 3] (row vector convention)
            pbc: Boolean tensor [3] or [n_systems, 3] for PBC
            cutoff: Maximum distance (scalar tensor)
            system_idx: Tensor [n_atoms] indicating system assignment
            self_interaction: If True, include self-pairs

        Returns:
            (mapping, system_mapping, shifts_idx)
        """
        r_max = cutoff.item() if isinstance(cutoff, torch.Tensor) else cutoff
        cell_reshaped, pbc_bool, _n_systems = _prepare_inputs(cell, pbc, system_idx)

        # Call alchemiops neighbor list
        res = _batch_naive_neighbor_list(
            positions=positions,
            cutoff=r_max,
            batch_idx=system_idx.to(torch.int32),
            cell=cell_reshaped,
            pbc=pbc_bool,
            return_neighbor_list=True,
        )

        # Parse results: (neighbor_list, neighbor_ptr[, neighbor_list_shifts])
        if len(res) == 3:  # type: ignore[arg-type]
            mapping, _, shifts_idx = res  # type: ignore[misc]
        else:
            mapping, _ = res  # type: ignore[misc]
            shifts_idx = torch.zeros(
                (mapping.shape[1], 3), dtype=positions.dtype, device=positions.device
            )

        # Convert dtypes
        mapping = mapping.to(dtype=torch.long)
        # Convert shifts_idx to floating point to match cell dtype (for einsum)
        shifts_idx = shifts_idx.to(dtype=cell.dtype)

        # Create system_mapping
        system_mapping = system_idx[mapping[0]]

        # Alchemiops does NOT include self-interactions by default
        # Add them only if requested
        if self_interaction:
            n_atoms = positions.shape[0]
            self_pairs = torch.arange(n_atoms, device=positions.device, dtype=torch.long)
            self_mapping = torch.stack([self_pairs, self_pairs], dim=0)
            # Self-shifts should match shifts_idx dtype
            self_shifts = torch.zeros(
                (n_atoms, 3), dtype=cell.dtype, device=positions.device
            )

            mapping = torch.cat([mapping, self_mapping], dim=1)
            shifts_idx = torch.cat([shifts_idx, self_shifts], dim=0)
            system_mapping = torch.cat([system_mapping, system_idx], dim=0)

        # Check if neighbors exceed estimate
        max_neighbors_estimate = estimate_max_neighbors(r_max)
        if mapping.shape[1] > max_neighbors_estimate:
            warnings.warn(
                f"Number of neighbors {mapping.shape[1]} exceeds estimated max "
                f"{max_neighbors_estimate} for cutoff {r_max}.",
                UserWarning,
                stacklevel=2,
            )
        return mapping, system_mapping, shifts_idx

else:
    # Provide stub function that raises informative error
    def alchemiops_nl_n2(  # type: ignore[misc]
        *args,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stub function when nvalchemiops is not available."""
        raise ImportError(
            "nvalchemiops is not installed. Install it with: pip install nvalchemiops"
        )
