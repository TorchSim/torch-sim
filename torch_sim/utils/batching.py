"""Utility functions for batched structure optimization workflows."""

from typing import TextIO

import torch


def write_log_line(
    file: TextIO,
    step: int,
    properties: dict[str, list[float]],
    converged: list[bool],
    batch_idx: list[int],
) -> None:
    """Write a formatted log line to the given file.

    Writes a nicely formatted line containing the current optimization step,
    property values for each structure, convergence status, and batch indices.
    On first step (step=0), writes a header row.

    Args:
        file: File handle to write to
        step: Current optimization step
        properties: Dictionary mapping property names to lists of values for each
            structure (e.g. {"energy": [...], "pressure": [...], "max_force": [...]})
        converged: List of convergence status for each structure
        batch_idx: List of batch indices being processed
    """
    if step == 0:
        # Write header row with property names
        header_parts = ["Step"]
        header_parts.extend(properties.keys())
        header_parts.extend(["Force mask", "Batch Indices"])
        header = " | ".join(header_parts)
        file.write(header + "\n")

        # Write separator line
        file.write("-" * len(header) + "\n")

    # Build property string by joining each property list
    prop_strings = [f"{step:4d}"]
    for values in properties.values():
        formatted_values = [f"{val:.4f}" for val in values]
        prop_strings.append(str(formatted_values))

    # Add convergence and batch info
    prop_strings.extend([str(converged), f"Batch indices: {batch_idx}"])

    # Join with separator and write
    struct_line = " | ".join(prop_strings)
    file.write(struct_line + "\n")


@torch.jit.script
def calculate_force_convergence_mask(
    forces: torch.Tensor, batch: torch.Tensor, batch_size: int, fmax: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized implementation of force convergence check.

    Uses torch.scatter_reduce for efficient calculation when possible,
    falls back to loop implementation if batch indices are out of bounds.

    Args:
        forces: Tensor of shape [n_atoms, 3] containing all forces
        batch: Tensor of shape [n_atoms] mapping atoms to their batch index
        batch_size: Number of structures in batch
        fmax: Maximum force threshold for convergence

    Returns:
        force_norms: Tensor of shape [batch_size] with max force norm per structure
        force_mask: Boolean tensor of shape [batch_size] indicating converged structures
    """
    # Initialize tensor for max forces per structure
    max_forces = torch.zeros(batch_size, device=forces.device)

    # Compute force norms for all atoms at once
    force_norms = torch.norm(forces, dim=1)  # [n_atoms]

    # Handle out-of-bounds batch indices
    if batch.max() >= batch_size:
        # Fall back to safer loop implementation
        for b in range(batch_size):
            mask = batch == b
            if mask.any():
                max_forces[b] = force_norms[mask].max()
    else:
        # Use efficient scatter_reduce operation
        max_forces.scatter_reduce_(0, batch, force_norms, reduce="amax")

    # Check convergence against threshold
    force_mask = max_forces < fmax

    return max_forces, force_mask
