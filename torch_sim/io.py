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
        list[Atoms]: ASE Atoms objects, one per graph

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
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_graphs, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    graph_idx = state.graph_idx.detach().cpu().numpy()

    atoms_list = []
    for idx in np.unique(graph_idx):
        mask = graph_idx == idx
        graph_positions = positions[mask]
        graph_numbers = atomic_numbers[mask]
        graph_cell = cell[idx].T  # Transpose for ASE convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in graph_numbers]

        atoms = Atoms(
            symbols=symbols, positions=graph_positions, cell=graph_cell, pbc=state.pbc
        )
        atoms_list.append(atoms)

    return atoms_list


def state_to_structures(state: "ts.SimState") -> list["Structure"]:
    """Convert a SimState to a list of Pymatgen Structure objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Structure]: Pymatgen Structure objects, one per graph

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
            "Pymatgen is required for state_to_structures conversion"
        ) from None

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_graphs, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    graph_idx = state.graph_idx.detach().cpu().numpy()

    # Get unique graph indices and counts
    unique_graphs = np.unique(graph_idx)
    structures = []

    for unique_graph_idx in unique_graphs:
        # Get mask for current graph
        mask = graph_idx == unique_graph_idx
        graph_positions = positions[mask]
        graph_numbers = atomic_numbers[mask]
        graph_cell = cell[unique_graph_idx].T  # Transpose for conventional form

        # Create species list from atomic numbers
        species = [Element.from_Z(z) for z in graph_numbers]

        # Create structure for this graph
        struct = Structure(
            lattice=Lattice(graph_cell),
            species=species,
            coords=graph_positions,
            coords_are_cartesian=True,
        )
        structures.append(struct)

    return structures


def state_to_phonopy(state: "ts.SimState") -> list["PhonopyAtoms"]:
    """Convert a SimState to a list of PhonopyAtoms objects.

    Args:
        state (SimState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[PhonopyAtoms]: PhonopyAtoms objects, one per graph

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
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_graphs, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    graph_idx = state.graph_idx.detach().cpu().numpy()

    phonopy_atoms_list = []
    for idx in np.unique(graph_idx):
        mask = graph_idx == idx
        graph_positions = positions[mask]
        graph_numbers = atomic_numbers[mask]
        graph_cell = cell[idx].T  # Transpose for Phonopy convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in graph_numbers]
        phonopy_atoms_list.append(
            PhonopyAtoms(
                symbols=symbols,
                positions=graph_positions,
                cell=graph_cell,
                pbc=state.pbc,
            )
        )

    return phonopy_atoms_list


def atoms_to_state(
    atoms: "Atoms | list[Atoms]",
    device: torch.device,
    dtype: torch.dtype,
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

    # Create graph indices using repeat_interleave
    atoms_per_graph = torch.tensor([len(a) for a in atoms_list], device=device)
    graph_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_graph
    )

    # Verify consistent pbc
    if not all(all(a.pbc) == all(atoms_list[0].pbc) for a in atoms_list):
        raise ValueError("All systems must have the same periodic boundary conditions")

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=all(atoms_list[0].pbc),
        atomic_numbers=atomic_numbers,
        graph_idx=graph_idx,
    )


def structures_to_state(
    structure: "Structure | list[Structure]",
    device: torch.device,
    dtype: torch.dtype,
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

    # Create graph indices
    atoms_per_graph = torch.tensor([len(s) for s in struct_list], device=device)
    graph_idx = torch.repeat_interleave(
        torch.arange(len(struct_list), device=device), atoms_per_graph
    )

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,  # Structures are always periodic
        atomic_numbers=atomic_numbers,
        graph_idx=graph_idx,
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

    # Create graph indices using repeat_interleave
    atoms_per_graph = torch.tensor([len(a) for a in phonopy_atoms_list], device=device)
    graph_idx = torch.repeat_interleave(
        torch.arange(len(phonopy_atoms_list), device=device), atoms_per_graph
    )

    """
    NOTE: PhonopyAtoms does not have pbc attribute for Supercells assume True
    Verify consistent pbc
    if not all(all(a.pbc) == all(phonopy_atoms_list[0].pbc) for a in phonopy_atoms_list):
        raise ValueError("All systems must have the same periodic boundary conditions")
    """

    return ts.SimState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,
        atomic_numbers=atomic_numbers,
        graph_idx=graph_idx,
    )
