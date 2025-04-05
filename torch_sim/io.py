"""Input/output utilities for atomistic systems.

This module provides functions for converting between different structural
representations. It includes utilities for converting ASE Atoms objects,
Pymatgen Structures, and PhonopyAtoms objects to SimState objects and vice versa.

The module handles:

* Converting between ASE Atoms and SimState
* Converting between Pymatgen Structure and SimState
* Converting between PhonopyAtoms and SimState
* Batched conversions for multiple structures
"""

from typing import TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

    from torch_sim.state import SimState


def state_to_atoms(state: "SimState") -> list["Atoms"]:
    """Convert a SimState to a list of ASE Atoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Atoms]: ASE Atoms objects, one per batch

    Raises:
        ImportError: If ASE is not installed

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase import Atoms
        from ase.data import chemical_symbols
    except ImportError:
        raise ImportError("ASE is required for state_to_atoms conversion") from None

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

        atoms = Atoms(
            symbols=symbols, positions=batch_positions, cell=batch_cell, pbc=state.pbc
        )
        atoms_list.append(atoms)

    return atoms_list


def state_to_structures(state: "SimState") -> list["Structure"]:
    """Convert a SimState to a list of Pymatgen Structure objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Structure]: Pymatgen Structure objects, one per batch

    Raises:
        ImportError: If Pymatgen is not installed

    Notes:
        - Output positions and cell will be in Å
        - Assumes periodic boundary conditions
    """
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.periodic_table import Element
    except ImportError:
        raise ImportError(
            "Pymatgen is required for state_to_structure conversion"
        ) from None

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
        struct = Structure(
            lattice=Lattice(batch_cell),
            species=species,
            coords=batch_positions,
            coords_are_cartesian=True,
        )
        structures.append(struct)

    return structures


def state_to_phonopy(state: "SimState") -> list["PhonopyAtoms"]:
    """Convert a SimState to a list of PhonopyAtoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[PhonopyAtoms]: PhonopyAtoms objects, one per batch

    Raises:
        ImportError: If Phonopy is not installed

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase.data import chemical_symbols
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        raise ImportError(
            "Phonopy is required for state_to_phonopy_atoms conversion"
        ) from None

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
) -> "SimState":
    """Convert an ASE Atoms object or list of Atoms objects to a SimState.

    Args:
        atoms (Atoms | list[Atoms]): Single ASE Atoms object or list of Atoms objects
        device (torch.device): Device to create tensors on
        dtype (torch.dtype): Data type for tensors (typically torch.float32 or
            torch.float64)

    Returns:
        SimState: TorchSim SimState object.

    Raises:
        ImportError: If ASE is not installed
        ValueError: If systems have inconsistent periodic boundary conditions

    Notes:
        - Input positions and cell should be in Å
        - Input masses should be in amu
        - All systems must have consistent periodic boundary conditions
    """
    from torch_sim.state import SimState

    try:
        from ase import Atoms
    except ImportError:
        raise ImportError("ASE is required for state_to_atoms conversion") from None

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
    cell = torch.tensor(  # Transpose cell from ASE convention to torchsim convention
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

    return SimState(
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
) -> "SimState":
    """Create a SimState from pymatgen Structure(s).

    Args:
        structure (Structure | list[Structure]): Single Structure or list of
            Structure objects
        device (torch.device): Device to create tensors on
        dtype (torch.dtype): Data type for tensors (typically torch.float32 or
            torch.float64)

    Returns:
        SimState: TorchSim SimState object.

    Raises:
        ImportError: If Pymatgen is not installed

    Notes:
        - Input positions and cell should be in Å
        - Cell matrix follows ASE convention: [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        - Assumes periodic boundary conditions from Structure
    """
    from torch_sim.state import SimState

    try:
        from pymatgen.core import Structure
    except ImportError:
        raise ImportError(
            "Pymatgen is required for state_to_structure conversion"
        ) from None

    struct_list = [structure] if isinstance(structure, Structure) else structure

    # Stack all properties
    cell = torch.tensor(
        np.stack([s.lattice.matrix for s in struct_list]), dtype=dtype, device=device
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

    return SimState(
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
) -> "SimState":
    """Create state tensors from a PhonopyAtoms object or list of PhonopyAtoms objects.

    Args:
        phonopy_atoms (PhonopyAtoms | list[PhonopyAtoms]): Single PhonopyAtoms object
            or list of PhonopyAtoms objects
        device (torch.device): Device to create tensors on
        dtype (torch.dtype): Data type for tensors (typically torch.float32 or
            torch.float64)

    Returns:
        SimState: TorchSim SimState object.

    Raises:
        ImportError: If Phonopy is not installed

    Notes:
        - Input positions and cell should be in Å
        - Input masses should be in amu
        - PhonopyAtoms does not have pbc attribute for Supercells, assumes True
        - Cell matrix follows ASE convention: [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
    """
    from torch_sim.state import SimState

    try:
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        raise ImportError("Phonopy is required for phonopy_to_state conversion") from None

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
        np.stack([a.cell for a in phonopy_atoms_list]), dtype=dtype, device=device
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

    return SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,
        atomic_numbers=atomic_numbers,
        batch=batch,
    )
