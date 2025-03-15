"""Simulation runners for molecular dynamics and optimization.

This module provides functions for running molecular dynamics simulations and geometry
optimizations using various calculators and integrators. It includes utilities for
converting between different molecular representations and handling simulation state.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from torch_sim.state import BaseState, StateLike, concatenate_states, state_to_device


if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

try:
    from pymatgen.core import Structure
except ImportError:

    class Structure:
        """Stub class for pymatgen Structure when not installed."""


try:
    from ase import Atoms
except ImportError:

    class Atoms:
        """Stub class for ASE Atoms when not installed."""


try:
    from phonopy.structure.atoms import PhonopyAtoms
except ImportError:

    class PhonopyAtoms:
        """Stub class for PhonopyAtoms when not installed."""


def initialize_state(
    system: StateLike,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Initialize state tensors from a system.

    Args:
        system: Input system to convert to state tensors
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: State tensors initialized from input system

    Raises:
        ValueError: If system type is not supported
    """
    # TODO: create a way to pass velocities from pmg and ase

    if isinstance(system, BaseState):
        return state_to_device(system, device, dtype)

    if isinstance(system, list) and all(isinstance(s, BaseState) for s in system):
        if not all(state.n_batches == 1 for state in system):
            raise ValueError(
                "When providing a list of states, to the initialize_state function, "
                "all states must have n_batches == 1. To fix this, you can split the "
                "states into individual states with the split_state function."
            )
        return concatenate_states(system)

    try:
        from pymatgen.core import Structure

        if isinstance(system, Structure) or (
            isinstance(system, list) and all(isinstance(s, Structure) for s in system)
        ):
            return structures_to_state(system, device, dtype)
    except ImportError:
        pass

    try:
        from ase import Atoms

        if isinstance(system, Atoms) or (
            isinstance(system, list) and all(isinstance(s, Atoms) for s in system)
        ):
            return atoms_to_state(system, device, dtype)
    except ImportError:
        pass

    # remaining code just for informative error
    is_list = isinstance(system, list)
    all_same_type = (
        is_list and all(isinstance(s, type(system[0])) for s in system) and system
    )
    if is_list and not all_same_type:
        raise ValueError(
            f"All items in list must be of the same type, "
            f"found {type(system[0])} and {type(system[1])}"
        )

    system_type = f"list[{type(system[0])}]" if is_list else type(system)

    raise ValueError(f"Unsupported system type, {system_type}")


def state_to_atoms(state: BaseState) -> list["Atoms"]:
    """Convert a BaseState to a list of ASE Atoms objects.

    Args:
        state (BaseState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Atoms]: List of ASE Atoms objects, one per batch

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase import Atoms
        from ase.data import chemical_symbols
    except ImportError as err:
        raise ImportError("ASE is required for state_to_atoms conversion") from err

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_batches, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    batch = state.batch.detach().cpu().numpy()

    atoms_list = []
    for batch_idx in np.unique(batch):
        mask = batch == batch_idx
        batch_positions = positions[mask]
        batch_numbers = atomic_numbers[mask]
        batch_cell = cell[batch_idx].T  # Transpose for ASE convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in batch_numbers]

        atoms_list.append(
            Atoms(
                symbols=symbols,
                positions=batch_positions,
                cell=batch_cell,
                pbc=state.pbc,
            )
        )

    return atoms_list


def state_to_structures(state: BaseState) -> list["Structure"]:
    """Convert a BaseState to a list of Pymatgen Structure objects.

    Args:
        state (BaseState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Structure]: List of Pymatgen Structure objects, one per batch

    Notes:
        - Output positions and cell will be in Å
        - Assumes periodic boundary conditions
    """
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.periodic_table import Element
    except ImportError as err:
        raise ImportError(
            "Pymatgen is required for state_to_structure conversion"
        ) from err

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_batches, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    batch = state.batch.detach().cpu().numpy()

    # Get unique batch indices and counts
    unique_batches = np.unique(batch)
    structures = []

    for batch_idx in unique_batches:
        # Get mask for current batch
        mask = batch == batch_idx
        batch_positions = positions[mask]
        batch_numbers = atomic_numbers[mask]
        batch_cell = cell[batch_idx].T  # Transpose for conventional form

        # Create species list from atomic numbers
        species = [Element.from_Z(z) for z in batch_numbers]

        # Create structure for this batch
        structures.append(
            Structure(
                lattice=Lattice(batch_cell),
                species=species,
                coords=batch_positions,
                coords_are_cartesian=True,
            )
        )

    return structures


def state_to_phonopy(state: BaseState) -> list["PhonopyAtoms"]:
    """Convert a BaseState to a list of PhonopyAtoms objects.

    Args:
        state (BaseState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[PhonopyAtoms]: List of PhonopyAtoms objects, one per batch

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase.data import chemical_symbols
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as err:
        raise ImportError(
            "Phonopy is required for state_to_phonopy_atoms conversion"
        ) from err

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_batches, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    batch = state.batch.detach().cpu().numpy()

    phonopy_atoms_list = []
    for batch_idx in np.unique(batch):
        mask = batch == batch_idx
        batch_positions = positions[mask]
        batch_numbers = atomic_numbers[mask]
        batch_cell = cell[batch_idx].T  # Transpose for Phonopy convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in batch_numbers]
        phonopy_atoms_list.append(
            PhonopyAtoms(
                symbols=symbols,
                positions=batch_positions,
                cell=batch_cell,
                pbc=state.pbc,
            )
        )

    return phonopy_atoms_list


def atoms_to_state(
    atoms: "Atoms | list[Atoms]",
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Create state tensors from an ASE Atoms object or list of Atoms objects.

    Args:
        atoms: Single ASE Atoms object or list of Atoms objects
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: Batched state tensors in internal units
    """
    try:
        from ase import Atoms
    except ImportError as err:
        raise ImportError("ASE is required for state_to_atoms conversion") from err

    atoms_list = [atoms] if isinstance(atoms, Atoms) else atoms

    # Stack all properties in one go
    positions = torch.tensor(
        np.concatenate([a.positions for a in atoms_list]), dtype=dtype, device=device
    )
    masses = torch.tensor(
        np.concatenate([a.get_masses() for a in atoms_list]), dtype=dtype, device=device
    )
    atomic_numbers = torch.tensor(
        np.concatenate([a.get_atomic_numbers() for a in atoms_list]),
        dtype=torch.int,
        device=device,
    )
    cell = torch.tensor(
        np.stack([a.cell.array.T for a in atoms_list]), dtype=dtype, device=device
    )

    # Create batch indices using repeat_interleave
    atoms_per_batch = torch.tensor([len(a) for a in atoms_list], device=device)
    batch = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_batch
    )

    # Verify consistent pbc
    if not all(all(a.pbc) == all(atoms_list[0].pbc) for a in atoms_list):
        raise ValueError("All systems must have the same periodic boundary conditions")

    return BaseState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=all(atoms_list[0].pbc),
        atomic_numbers=atomic_numbers,
        batch=batch,
    )


def structures_to_state(
    structure: "Structure | list[Structure]",
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Create a BaseState from pymatgen Structure(s).

    Args:
        structure: Single Structure or list of Structure objects
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: Batched state tensors in internal units

    Notes:
        - Cell matrix follows ASE convention: [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        - Assumes periodic boundary conditions from Structure
    """
    try:
        from pymatgen.core import Structure
    except ImportError as err:
        raise ImportError(
            "Pymatgen is required for state_to_structure conversion"
        ) from err

    struct_list = [structure] if isinstance(structure, Structure) else structure

    # Stack all properties
    cell = torch.tensor(
        np.stack([s.lattice.matrix.T for s in struct_list]), dtype=dtype, device=device
    )
    positions = torch.tensor(
        np.concatenate([s.cart_coords for s in struct_list]), dtype=dtype, device=device
    )
    masses = torch.tensor(
        np.concatenate([[site.specie.atomic_mass for site in s] for s in struct_list]),
        dtype=dtype,
        device=device,
    )
    atomic_numbers = torch.tensor(
        np.concatenate([[site.specie.number for site in s] for s in struct_list]),
        dtype=torch.int,
        device=device,
    )

    # Create batch indices
    atoms_per_batch = torch.tensor([len(s) for s in struct_list], device=device)
    batch = torch.repeat_interleave(
        torch.arange(len(struct_list), device=device), atoms_per_batch
    )

    return BaseState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,  # Structures are always periodic
        atomic_numbers=atomic_numbers,
        batch=batch,
    )


def phonopy_to_state(
    phonopy_atoms: "PhonopyAtoms | list[PhonopyAtoms]",
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Create state tensors from a PhonopyAtoms object or list of PhonopyAtoms objects.

    Args:
        phonopy_atoms: Single PhonopyAtoms object or list of PhonopyAtoms objects
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: Batched state tensors in internal units
    """
    try:
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as err:
        raise ImportError(
            "Phonopy is required for phonopy_to_state conversion"
        ) from err

    phonopy_atoms_list = (
        [phonopy_atoms] if isinstance(phonopy_atoms, PhonopyAtoms) else phonopy_atoms
    )

    # Stack all properties in one go
    positions = torch.tensor(
        np.concatenate([a.positions for a in phonopy_atoms_list]),
        dtype=dtype,
        device=device,
    )
    masses = torch.tensor(
        np.concatenate([a.masses for a in phonopy_atoms_list]),
        dtype=dtype,
        device=device,
    )
    atomic_numbers = torch.tensor(
        np.concatenate([a.numbers for a in phonopy_atoms_list]),
        dtype=torch.int,
        device=device,
    )
    cell = torch.tensor(
        np.stack([a.cell.T for a in phonopy_atoms_list]), dtype=dtype, device=device
    )

    # Create batch indices using repeat_interleave
    atoms_per_batch = torch.tensor([len(a) for a in phonopy_atoms_list], device=device)
    batch = torch.repeat_interleave(
        torch.arange(len(phonopy_atoms_list), device=device), atoms_per_batch
    )

    """
    NOTE: PhonopyAtoms does not have pbc attribute for Supercells assume True
    Verify consistent pbc
    if not all(all(a.pbc) == all(phonopy_atoms_list[0].pbc) for a in phonopy_atoms_list):
        raise ValueError("All systems must have the same periodic boundary conditions")
    """

    return BaseState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,
        atomic_numbers=atomic_numbers,
        batch=batch,
    )
