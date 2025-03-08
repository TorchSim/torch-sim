"""Utility functions for batched structure optimization workflows."""

from collections.abc import Callable
from typing import Any, TextIO

import torch
from ase import Atoms
from pymatgen.core import Structure

from torchsim.runners import atoms_to_state, structures_to_state


def write_log_line(
    file: TextIO,
    step: int,
    properties: dict[str, list[float]],
    converged: list[bool],
    batch_idx: list[int],
) -> None:
    """Write a formatted log line to the given file.

    Args:
        file: File handle to write to
        step: Current optimization step
        properties: Dictionary mapping property names to lists of values for each
            structure
            e.g. {"energy": [...], "pressure": [...], "max_force": [...]}
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


def check_max_atoms_in_batch(
    current_struct: Atoms | Structure,
    next_struct: Atoms | Structure,
    struct_list: list[Atoms | Structure],
    max_atoms: int,
) -> bool:
    """Check if swapping structures would exceed max batch size.

    Args:
        current_struct: Structure to be replaced (Atoms or pymatgen Structure)
        next_struct: Structure to add (Atoms or pymatgen Structure)
        struct_list (list): Current list of structures
        max_atoms (int): Maximum allowed atoms in batch

    Returns:
        bool: True if swap is allowed, False otherwise
    """

    def _get_natoms(struct: Atoms | Structure) -> int:
        if isinstance(struct, Atoms):
            return len(struct)
        # pymatgen Structure
        return len(struct.sites)

    total_batch_atoms = (
        sum(_get_natoms(struct) for struct in struct_list)
        - _get_natoms(current_struct)
        + _get_natoms(next_struct)
    )
    return total_batch_atoms <= max_atoms


def swap_structure(
    idx: int,
    current_idx: int,
    struct_list: list[Atoms | Structure],
    all_struct_list: list[Atoms | Structure],
    device: torch.device,
    dtype: torch.dtype,
    optimizer_init: Callable,
) -> tuple[Any, int]:
    """Swap a converged structure with the next one in the queue.

    Args:
        idx: Index of structure to replace
        current_idx: Index of next structure to add
        struct_list: Current list of structures (Atoms or pymatgen Structure)
        all_struct_list: Full list of structures to process
        device: Torch device
        dtype: Torch dtype
        optimizer_init: Optimizer initialization function

    Returns:
        tuple containing:
            - Any: Updated state after swapping structure
            - int: Incremented current_idx
    """
    struct_list[idx] = all_struct_list[current_idx]

    # Convert structures based on type
    if isinstance(struct_list[0], Atoms):
        base_state = atoms_to_state(struct_list, device=device, dtype=dtype)
    else:  # pymatgen Structure
        base_state = structures_to_state(struct_list, device=device, dtype=dtype)

    state = optimizer_init(base_state)
    torch.cuda.empty_cache()
    print(f"Replaced structure at index {idx} with next structure (index {current_idx})")
    return state, current_idx + 1


def calculate_force_convergence(
    state: Any,
    batch_size: int,
    fmax: float,
) -> tuple[list[float], list[bool]]:
    """Calculate force norms and check convergence for each structure in batch.

    Args:
        state: Current state containing forces and batch indices
        batch_size: Number of structures in batch
        fmax: Force convergence threshold

    Returns:
        tuple containing:
            - list[float]: Maximum force norm for each structure
            - list[bool]: Whether each structure is converged
    """
    force_norms = torch.stack(
        [state.forces[state.batch == i].norm(dim=-1).max() for i in range(batch_size)]
    ).tolist()
    force_converged = [norm < fmax for norm in force_norms]
    return force_norms, force_converged
