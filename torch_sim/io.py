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

import torch_sim as ts


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure


def state_to_atoms(state: "ts.SimState") -> list["Atoms"]:
    """Convert a SimState to a list of ASE Atoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Atoms]: ASE Atoms objects, one per system

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
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_systems, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    system_indices = state.system_idx.detach().cpu().numpy()
    pbc = state.pbc.detach().cpu().numpy()

    atoms_list = []
    for sys_idx in np.unique(system_indices):
        mask = system_indices == sys_idx
        system_positions = positions[mask]
        system_numbers = atomic_numbers[mask]
        system_cell = cell[sys_idx].T  # Transpose for ASE convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in system_numbers]

        atoms = Atoms(
            symbols=symbols, positions=system_positions, cell=system_cell, pbc=pbc
        )
        atoms_list.append(atoms)

    return atoms_list


def state_to_structures(state: "ts.SimState") -> list["Structure"]:
    """Convert a SimState to a list of Pymatgen Structure objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Structure]: Pymatgen Structure objects, one per system

    Raises:
        ImportError: If Pymatgen is not installed

    Notes:
        - Output positions and cell will be in Å
        - Assumes periodic boundary conditions
    """
    try:
        from pymatgen.core import Element, Lattice, Structure
    except ImportError:
        raise ImportError(
            "Pymatgen is required for state_to_structures conversion"
        ) from None

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_systems, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    system_indices = state.system_idx.detach().cpu().numpy()

    # Get unique system indices and counts
    uniq_systems = np.unique(system_indices)
    structures: list[Structure] = []

    for uniq_sys_idx in uniq_systems:
        # Get mask for current system
        mask = system_indices == uniq_sys_idx
        system_positions = positions[mask]
        system_numbers = atomic_numbers[mask]
        system_cell = cell[uniq_sys_idx].T  # Transpose for conventional form

        # Create species list from atomic numbers
        species = [Element.from_Z(z) for z in system_numbers]

        # Create structure for this system
        struct = Structure(
            lattice=Lattice(system_cell, pbc=(state.pbc.tolist())),
            species=species,
            coords=system_positions,
            coords_are_cartesian=True,
        )
        structures.append(struct)

    return structures


def state_to_phonopy(state: "ts.SimState") -> list["PhonopyAtoms"]:
    """Convert a SimState to a list of PhonopyAtoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[PhonopyAtoms]: PhonopyAtoms objects, one per system

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
        raise ImportError("Phonopy is required for state_to_phonopy conversion") from None

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_systems, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    system_indices = state.system_idx.detach().cpu().numpy()

    phonopy_atoms_list: list[PhonopyAtoms] = []
    for sys_idx in np.unique(system_indices):
        mask = system_indices == sys_idx
        system_positions = positions[mask]
        system_numbers = atomic_numbers[mask]
        system_cell = cell[sys_idx].T  # Transpose for Phonopy convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in system_numbers]

        # Note: pbc is not used in the init since it's always assumed to be true
        # https://github.com/phonopy/phonopy/blob/develop/phonopy/structure/atoms.py#L140
        phonopy_atoms = PhonopyAtoms(
            symbols=symbols, positions=system_positions, cell=system_cell
        )
        phonopy_atoms_list.append(phonopy_atoms)

    return phonopy_atoms_list


def atoms_to_state(
    atoms: "Atoms | list[Atoms]", device: torch.device, dtype: torch.dtype
) -> "ts.SimState":
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
    try:
        from ase import Atoms
    except ImportError:
        raise ImportError("ASE is required for atoms_to_state conversion") from None

    atoms_list = [atoms] if isinstance(atoms, Atoms) else atoms

    # Stack all properties in one go
    positions = torch.tensor(
        np.concatenate([at.positions for at in atoms_list]), dtype=dtype, device=device
    )
    masses = torch.tensor(
        np.concatenate([at.get_masses() for at in atoms_list]), dtype=dtype, device=device
    )
    atomic_numbers = torch.tensor(
        np.concatenate([at.get_atomic_numbers() for at in atoms_list]),
        dtype=torch.int,
        device=device,
    )
    cell = torch.tensor(  # Transpose cell from ASE convention to TorchSim convention
        np.stack([at.cell.array.T for at in atoms_list]), dtype=dtype, device=device
    )

    # Create system indices using repeat_interleave
    atoms_per_system = torch.tensor([len(at) for at in atoms_list], device=device)
    system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_system
    )

    # Verify consistent pbc
    if not all(np.all(np.equal(at.pbc, atoms_list[0].pbc)) for at in atoms_list[1:]):
        raise ValueError("All systems must have the same periodic boundary conditions")

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=atoms_list[0].pbc,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
    )


def structures_to_state(
    structure: "Structure | list[Structure]", device: torch.device, dtype: torch.dtype
) -> "ts.SimState":
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
    try:
        from pymatgen.core import Structure
    except ImportError:
        raise ImportError(
            "Pymatgen is required for structures_to_state conversion"
        ) from None

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

    # Create system indices
    atoms_per_system = torch.tensor([len(s) for s in struct_list], device=device)
    system_idx = torch.repeat_interleave(
        torch.arange(len(struct_list), device=device), atoms_per_system
    )

    # Verify consistent pbc
    if not all(tuple(s.pbc) == tuple(struct_list[0].pbc) for s in struct_list[1:]):
        raise ValueError("All systems must have the same periodic boundary conditions")

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=struct_list[0].pbc,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
    )


def phonopy_to_state(
    phonopy_atoms: "PhonopyAtoms | list[PhonopyAtoms]",
    device: torch.device,
    dtype: torch.dtype,
) -> "ts.SimState":
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
    try:
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError:
        raise ImportError("Phonopy is required for phonopy_to_state conversion") from None

    phonopy_atoms_list = (
        [phonopy_atoms] if isinstance(phonopy_atoms, PhonopyAtoms) else phonopy_atoms
    )

    # Stack all properties in one go
    kwargs = {"dtype": dtype, "device": device}
    positions = torch.tensor(
        np.concatenate([at.positions for at in phonopy_atoms_list]), **kwargs
    )
    masses = torch.tensor(
        np.concatenate([at.masses for at in phonopy_atoms_list]), **kwargs
    )
    atomic_numbers = torch.tensor(
        np.concatenate([a.numbers for a in phonopy_atoms_list]),
        dtype=torch.int,
        device=device,
    )
    cell = torch.tensor(
        np.stack([at.cell.T for at in phonopy_atoms_list]), dtype=dtype, device=device
    )

    # Create system indices using repeat_interleave
    atoms_per_system = torch.tensor([len(at) for at in phonopy_atoms_list], device=device)
    system_idx = torch.repeat_interleave(
        torch.arange(len(phonopy_atoms_list), device=device), atoms_per_system
    )

    """
    NOTE: PhonopyAtoms does not have pbc attribute for Supercells assume True
    Verify consistent pbc
    if not all(all(at.pbc) == all(phonopy_atoms_lst[0].pbc) for at in phonopy_atoms_lst):
        raise ValueError("All systems must have the same periodic boundary conditions")
    """

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
    )
