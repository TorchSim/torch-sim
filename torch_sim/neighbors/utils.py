"""Utilities for neighbor list calculations."""

import torch


def complete_cell(cell: torch.Tensor, pbc: torch.Tensor) -> torch.Tensor:
    """Fill zero-length lattice vectors on non-periodic axes, like ASE's complete_cell.

    Slabs, wires and molecules built by ASE (e.g. ``graphene()``, ``nanotube()``,
    ``molecule()``) leave the vectors of their non-periodic axes at zero. These are
    replaced by unit vectors orthogonal to the others, keeping the cell right-handed.
    Image shifts along non-periodic axes are always zero, so the filled vectors never
    enter interatomic displacements.

    Args:
        cell: Row-vector cells [n_systems, 3, 3]
        pbc: Periodic boundary conditions [n_systems, 3]

    Returns:
        Cells [n_systems, 3, 3] with every non-periodic zero vector filled in.
    """
    missing = (cell == 0).all(dim=-1) & ~pbc.to(torch.bool)
    if not missing.any():
        return cell
    n_missing = missing.sum(dim=-1)[:, None, None]
    eye = torch.eye(3, dtype=cell.dtype, device=cell.device)

    # One missing vector: the unit normal of the other two
    normal = torch.linalg.cross(cell.roll(-1, dims=1), cell.roll(-2, dims=1), dim=-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(
        1e-30
    )

    # Two missing vectors: an orthonormal pair (u, v x u) orthogonal to the present v
    present = (cell * (~missing).unsqueeze(-1)).sum(dim=1)
    v = present / torch.linalg.vector_norm(present, dim=-1, keepdim=True).clamp_min(1e-30)
    helper = torch.where((v[:, 0].abs() > 0.9).unsqueeze(-1), eye[1], eye[0])
    u = helper - (helper * v).sum(dim=-1, keepdim=True) * v
    u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(1e-30)
    w = torch.linalg.cross(v, u, dim=-1)
    present_idx = (~missing).to(torch.int64).argmax(dim=-1, keepdim=True)
    offset = (torch.arange(3, device=cell.device) - present_idx) % 3
    pair = torch.where((offset == 1).unsqueeze(-1), u.unsqueeze(1), w.unsqueeze(1))

    filled = torch.where(n_missing == 1, normal, pair)
    filled = torch.where(n_missing == 3, eye.expand_as(cell), filled)
    return torch.where(missing.unsqueeze(-1), filled, cell)


def normalize_inputs(
    cell: torch.Tensor, pbc: torch.Tensor, n_systems: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize cell and PBC tensors to standard batch format.

    Handles multiple input formats:
    - cell: [3, 3], [n_systems, 3, 3], or [n_systems*3, 3]
    - pbc: [3], [n_systems, 3], or [n_systems*3]

    Zero-length cell vectors on non-periodic axes are filled in with
    :func:`complete_cell`.

    Returns:
        (cell, pbc) normalized to ([n_systems, 3, 3], [n_systems, 3])
        Both tensors are guaranteed to be contiguous.
    """
    if cell.ndim == 2:
        if cell.shape[0] == 3:
            cell = cell.unsqueeze(0).expand(n_systems, -1, -1).contiguous()
        else:
            cell = cell.reshape(n_systems, 3, 3).contiguous()
    else:
        cell = cell.contiguous()
    if pbc.ndim == 1:
        if pbc.shape[0] == 3:
            pbc = pbc.unsqueeze(0).expand(n_systems, -1).contiguous()
        else:
            pbc = pbc.reshape(n_systems, 3).contiguous()
    else:
        pbc = pbc.contiguous()
    return complete_cell(cell, pbc), pbc
