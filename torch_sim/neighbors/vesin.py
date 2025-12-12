"""Vesin-based neighbor list implementations.

This module provides high-performance neighbor list calculations using the
Vesin library. It includes both TorchScript-compatible and standard implementations.

Vesin is available at: https://github.com/Luthaf/vesin
"""

import torch


try:
    from vesin import NeighborList as VesinNeighborList
    from vesin.torch import NeighborList as VesinNeighborListTorch

    VESIN_AVAILABLE = True
except ImportError:
    VESIN_AVAILABLE = False
    VesinNeighborList = None  # type: ignore[assignment, misc]
    VesinNeighborListTorch = None  # type: ignore[assignment, misc]

__all__ = [
    "VESIN_AVAILABLE",
    "VesinNeighborList",
    "VesinNeighborListTorch",
    "vesin_nl",
    "vesin_nl_ts",
]


if VESIN_AVAILABLE:

    def vesin_nl_ts(  # noqa: PLR0915
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: torch.Tensor,
        system_idx: torch.Tensor,
        self_interaction: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute neighbor lists using TorchScript-compatible Vesin.

        This function provides a TorchScript-compatible interface to the Vesin
        neighbor list algorithm using VesinNeighborListTorch.

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

        Example:
            >>> # Single system
            >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            >>> system_idx = torch.zeros(2, dtype=torch.long)
            >>> mapping, sys_map, shifts = vesin_nl_ts(
            ...     positions, cell, pbc, cutoff, system_idx
            ... )

        Notes:
            - Uses VesinNeighborListTorch for TorchScript compatibility
            - Requires CPU tensors in float64 precision internally
            - Returns tensors on the same device as input with original precision
            - For non-periodic systems, shifts will be zero vectors
            - The neighbor list includes both (i,j) and (j,i) pairs

        References:
              https://github.com/Luthaf/vesin
        """
        device = positions.device
        dtype = positions.dtype
        n_systems = system_idx.max().item() + 1

        # Normalize inputs to batch format
        if cell.ndim == 2:
            if cell.shape[0] == 3:
                cell = cell.unsqueeze(0).expand(n_systems, -1, -1)
            else:
                cell = cell.reshape(n_systems, 3, 3)
        if pbc.ndim == 1:
            if pbc.shape[0] == 3:
                pbc = pbc.unsqueeze(0).expand(n_systems, -1)
            else:
                pbc = pbc.reshape(n_systems, 3)

        # Process each system's neighbor list separately
        edge_indices = []
        shifts_idx_list = []
        system_mapping_list = []
        offset = 0

        for sys_idx in range(n_systems):
            system_mask = system_idx == sys_idx
            n_atoms_in_system = system_mask.sum().item()

            if n_atoms_in_system == 0:
                continue

            # Calculate neighbor list for this system
            neighbor_list_fn = VesinNeighborListTorch(cutoff.item(), full_list=True)

            # Get the cell for this system
            cell_sys = cell[sys_idx]

            # Convert tensors to CPU and float64 properly
            positions_cpu = positions[system_mask].cpu().to(dtype=torch.float64)
            cell_cpu = cell_sys.cpu().to(dtype=torch.float64)
            periodic_cpu = pbc[sys_idx].to(dtype=torch.bool).cpu()

            # Only works on CPU and requires float64
            i, j, S = neighbor_list_fn.compute(
                points=positions_cpu,
                box=cell_cpu,
                periodic=periodic_cpu,
                quantities="ijS",
            )

            edge_idx = torch.stack((i, j), dim=0).to(dtype=torch.long, device=device)
            shifts = S.to(dtype=dtype, device=device)

            # Adjust indices for the global atom indexing
            edge_idx = edge_idx + offset

            edge_indices.append(edge_idx)
            shifts_idx_list.append(shifts)
            system_mapping_list.append(
                torch.full((edge_idx.shape[1],), sys_idx, dtype=torch.long, device=device)
            )

            offset += n_atoms_in_system

        # Combine all neighbor lists
        if len(edge_indices) == 0:
            # No neighbors found
            mapping = torch.zeros((2, 0), dtype=torch.long, device=device)
            system_mapping = torch.zeros(0, dtype=torch.long, device=device)
            shifts_idx = torch.zeros((0, 3), dtype=dtype, device=device)
        else:
            mapping = torch.cat(edge_indices, dim=1)
            shifts_idx = torch.cat(shifts_idx_list, dim=0)
            system_mapping = torch.cat(system_mapping_list, dim=0)

        # Add self-interactions if requested
        if self_interaction:
            n_atoms = positions.shape[0]
            self_pairs = torch.arange(n_atoms, device=device, dtype=torch.long)
            self_mapping = torch.stack([self_pairs, self_pairs], dim=0)
            self_shifts = torch.zeros((n_atoms, 3), dtype=dtype, device=device)
            self_sys_mapping = system_idx

            mapping = torch.cat([mapping, self_mapping], dim=1)
            shifts_idx = torch.cat([shifts_idx, self_shifts], dim=0)
            system_mapping = torch.cat([system_mapping, self_sys_mapping], dim=0)

        return mapping, system_mapping, shifts_idx

    def vesin_nl(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float | torch.Tensor,
        system_idx: torch.Tensor,
        self_interaction: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute neighbor lists using the standard Vesin implementation.

        This function provides an interface to the standard Vesin neighbor list
        algorithm using VesinNeighborList.

        Args:
            positions: Atomic positions tensor [n_atoms, 3]
            cell: Unit cell vectors [n_systems, 3, 3] or [3, 3]
            pbc: Boolean tensor [n_systems, 3] or [3]
            cutoff: Maximum distance for considering atoms as neighbors
            system_idx: Tensor [n_atoms] indicating which system each atom belongs to
            self_interaction: If True, include self-pairs. Default: False

        Returns:
            tuple containing:
                - mapping: Tensor [2, num_neighbors] - pairs of atom indices
                - system_mapping: Tensor [num_neighbors] - system assignment for each pair
                - shifts_idx: Tensor [num_neighbors, 3] - periodic shift indices

        Example:
            >>> # Single system
            >>> positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            >>> system_idx = torch.zeros(2, dtype=torch.long)
            >>> mapping, sys_map, shifts = vesin_nl(
            ...     positions, cell, pbc, cutoff, system_idx
            ... )

        Notes:
            - Uses standard VesinNeighborList implementation
            - Requires CPU tensors in float64 precision internally
            - Returns tensors on the same device as input with original precision
            - For non-periodic systems, shifts will be zero vectors
            - The neighbor list includes both (i,j) and (j,i) pairs

        References:
            - https://github.com/Luthaf/vesin
        """
        from torch_sim.neighbors import _normalize_inputs

        device = positions.device
        dtype = positions.dtype
        n_systems = system_idx.max().item() + 1
        cell, pbc = _normalize_inputs(cell, pbc, n_systems)

        # Process each system's neighbor list separately
        edge_indices = []
        shifts_idx_list = []
        system_mapping_list = []
        offset = 0

        for sys_idx in range(n_systems):
            system_mask = system_idx == sys_idx
            n_atoms_in_system = system_mask.sum().item()

            if n_atoms_in_system == 0:
                continue

            # Get the cell for this system
            cell_sys = cell[sys_idx]

            # Calculate neighbor list for this system
            neighbor_list_fn = VesinNeighborList(
                (float(cutoff)), full_list=True, sorted=False
            )

            # Convert tensors to CPU and float64 without gradients
            positions_cpu = positions[system_mask].detach().cpu().to(dtype=torch.float64)
            cell_cpu = cell_sys.detach().cpu().to(dtype=torch.float64)
            periodic_cpu = pbc[sys_idx].detach().to(dtype=torch.bool).cpu()

            # Only works on CPU and returns numpy arrays
            i, j, S = neighbor_list_fn.compute(
                points=positions_cpu,
                box=cell_cpu,
                periodic=periodic_cpu,
                quantities="ijS",
            )
            i, j = (
                torch.tensor(i, dtype=torch.long, device=device),
                torch.tensor(j, dtype=torch.long, device=device),
            )
            edge_idx = torch.stack((i, j), dim=0)
            shifts = torch.tensor(S, dtype=dtype, device=device)

            # Adjust indices for the global atom indexing
            edge_idx = edge_idx + offset

            edge_indices.append(edge_idx)
            shifts_idx_list.append(shifts)
            system_mapping_list.append(
                torch.full((edge_idx.shape[1],), sys_idx, dtype=torch.long, device=device)
            )

            offset += n_atoms_in_system

        # Combine all neighbor lists
        if len(edge_indices) == 0:
            # No neighbors found
            mapping = torch.zeros((2, 0), dtype=torch.long, device=device)
            system_mapping = torch.zeros(0, dtype=torch.long, device=device)
            shifts_idx = torch.zeros((0, 3), dtype=dtype, device=device)
        else:
            mapping = torch.cat(edge_indices, dim=1)
            shifts_idx = torch.cat(shifts_idx_list, dim=0)
            system_mapping = torch.cat(system_mapping_list, dim=0)

        # Add self-interactions if requested
        if self_interaction:
            n_atoms = positions.shape[0]
            self_pairs = torch.arange(n_atoms, device=device, dtype=torch.long)
            self_mapping = torch.stack([self_pairs, self_pairs], dim=0)
            self_shifts = torch.zeros((n_atoms, 3), dtype=dtype, device=device)
            self_sys_mapping = system_idx

            mapping = torch.cat([mapping, self_mapping], dim=1)
            shifts_idx = torch.cat([shifts_idx, self_shifts], dim=0)
            system_mapping = torch.cat([system_mapping, self_sys_mapping], dim=0)

        return mapping, system_mapping, shifts_idx

else:
    # Provide stub functions that raise informative errors
    def vesin_nl_ts(  # type: ignore[misc]
        *args,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stub function when Vesin is not available."""
        raise ImportError("Vesin is not installed. Install it with: pip install vesin")

    def vesin_nl(  # type: ignore[misc]
        *args,  # noqa: ARG001
        **kwargs,  # noqa: ARG001
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stub function when Vesin is not available."""
        raise ImportError("Vesin is not installed. Install it with: pip install vesin")
