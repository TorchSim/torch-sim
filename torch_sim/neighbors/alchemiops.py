"""Alchemiops-based neighbor list implementations.

This module provides high-performance CUDA-accelerated neighbor list calculations
using the nvalchemiops library. Supports both naive N^2 and cell list algorithms.

nvalchemiops is available at: https://github.com/NVIDIA/nvalchemiops
"""

import torch


try:
    from nvalchemiops.neighborlist import batch_cell_list, batch_naive_neighbor_list
    from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors

    ALCHEMIOPS_AVAILABLE = True
except ImportError:
    ALCHEMIOPS_AVAILABLE = False
    batch_naive_neighbor_list = None  # type: ignore[assignment]
    batch_cell_list = None  # type: ignore[assignment]
    estimate_max_neighbors = None  # type: ignore[assignment, name-defined]

__all__ = [
    "ALCHEMIOPS_AVAILABLE",
    "alchemiops_nl_cell_list",
    "alchemiops_nl_n2",
]


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
            cell: Unit cell vectors [n_systems, 3, 3] or [3, 3]
            pbc: Boolean tensor [n_systems, 3] or [3]
            cutoff: Maximum distance (scalar tensor)
            system_idx: Tensor [n_atoms] indicating system assignment
            self_interaction: If True, include self-pairs

        Returns:
            (mapping, system_mapping, shifts_idx)
        """
        from torch_sim.neighbors import _normalize_inputs

        r_max = cutoff.item() if isinstance(cutoff, torch.Tensor) else cutoff
        n_systems = system_idx.max().item() + 1
        cell, pbc = _normalize_inputs(cell, pbc, n_systems)

        # Call alchemiops neighbor list
        res = batch_naive_neighbor_list(
            positions=positions,
            cutoff=r_max,
            batch_idx=system_idx.to(torch.int32),
            cell=cell,
            pbc=pbc.to(torch.bool),
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

        return mapping, system_mapping, shifts_idx

    def alchemiops_nl_cell_list(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: torch.Tensor,
        system_idx: torch.Tensor,
        self_interaction: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute neighbor lists using Alchemiops cell list algorithm.

        Args:
            positions: Atomic positions tensor [n_atoms, 3]
            cell: Unit cell vectors [n_systems, 3, 3] or [3, 3]
            pbc: Boolean tensor [n_systems, 3] or [3]
            cutoff: Maximum distance (scalar tensor)
            system_idx: Tensor [n_atoms] indicating system assignment
            self_interaction: If True, include self-pairs

        Returns:
            (mapping, system_mapping, shifts_idx)
        """
        from torch_sim.neighbors import _normalize_inputs

        r_max = cutoff.item() if isinstance(cutoff, torch.Tensor) else cutoff
        n_systems = system_idx.max().item() + 1
        cell, pbc = _normalize_inputs(cell, pbc, n_systems)

        # For non-periodic systems with zero cells, use a nominal identity cell
        # to avoid division by zero in alchemiops warp kernels
        # See https://github.com/NVIDIA/nvalchemi-toolkit-ops/issues/4
        is_non_periodic = ~pbc.any(dim=1)  # [n_systems]
        is_zero_cell = cell.abs().sum(dim=(1, 2)) == 0  # [n_systems]
        needs_nominal_cell = is_non_periodic & is_zero_cell
        if needs_nominal_cell.any():
            identity = torch.eye(3, dtype=cell.dtype, device=cell.device)
            cell = cell.clone()  # Avoid modifying the original
            cell[needs_nominal_cell] = identity

        # Call alchemiops cell list
        res = batch_cell_list(
            positions=positions,
            cutoff=r_max,
            batch_idx=system_idx.to(torch.int32),
            cell=cell,
            pbc=pbc.to(torch.bool),
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

        return mapping, system_mapping, shifts_idx

else:
    # Provide stub functions that raise informative errors
    def alchemiops_nl_n2(  # type: ignore[misc]
        *args,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stub function when nvalchemiops is not available."""
        raise ImportError(
            "nvalchemiops is not installed. Install it with: pip install nvalchemiops"
        )

    def alchemiops_nl_cell_list(  # type: ignore[misc]
        *args,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stub function when nvalchemiops is not available."""
        raise ImportError(
            "nvalchemiops is not installed. Install it with: pip install nvalchemiops"
        )
