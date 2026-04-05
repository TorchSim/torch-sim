import torch


@torch.jit.script
def normalize_inputs(
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
    return cell, pbc


normalize_inputs_jit = normalize_inputs
