from typing import Any

import torch
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

from torch_sim.io import (
    atoms_to_state,
    phonopy_to_state,
    state_to_atoms,
    state_to_phonopy,
    state_to_structures,
    structures_to_state,
)
from torch_sim.state import BaseState


def test_single_structure_to_state(si_structure: Structure, device: torch.device) -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = structures_to_state(si_structure, device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert all(
        t.device.type == device.type for t in [state.positions, state.masses, state.cell]
    )
    assert all(
        t.dtype == torch.float64 for t in [state.positions, state.masses, state.cell]
    )
    assert state.atomic_numbers.dtype == torch.int

    # Check shapes and values
    assert state.positions.shape == (8, 3)
    assert torch.allclose(state.masses, torch.full_like(state.masses, 28.0855))  # Si
    assert torch.all(state.atomic_numbers == 14)  # Si atomic number
    assert torch.allclose(
        state.cell,
        torch.diag(torch.full((3,), 5.43, device=device, dtype=torch.float64)),
    )


def test_multiple_structures_to_state(
    si_structure: Structure, device: torch.device
) -> None:
    """Test conversion from list of pymatgen Structure to state tensors."""
    state = structures_to_state([si_structure, si_structure], device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.batch.shape == (16,)
    assert torch.all(
        state.batch == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8)
    )


def test_single_atoms_to_state(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to state tensors."""
    state = atoms_to_state(si_atoms, device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (8, 3)
    assert state.masses.shape == (8,)
    assert state.cell.shape == (1, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (8,)
    assert state.batch.shape == (8,)
    assert torch.all(state.batch == 0)


def test_multiple_atoms_to_state(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to state tensors."""
    state = atoms_to_state([si_atoms, si_atoms], device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.batch.shape == (16,)
    assert torch.all(
        state.batch == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8),
    )


def test_state_to_structure(ar_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of pymatgen Structure."""
    structures = state_to_structures(ar_base_state)
    assert len(structures) == 1
    assert isinstance(structures[0], Structure)
    assert len(structures[0]) == 32


def test_state_to_multiple_structures(ar_double_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of pymatgen Structure."""
    structures = state_to_structures(ar_double_base_state)
    assert len(structures) == 2
    assert isinstance(structures[0], Structure)
    assert isinstance(structures[1], Structure)
    assert len(structures[0]) == 32
    assert len(structures[1]) == 32


def test_state_to_atoms(ar_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of ASE Atoms."""
    atoms = state_to_atoms(ar_base_state)
    assert len(atoms) == 1
    assert isinstance(atoms[0], Atoms)
    assert len(atoms[0]) == 32


def test_state_to_multiple_atoms(ar_double_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of ASE Atoms."""
    atoms = state_to_atoms(ar_double_base_state)
    assert len(atoms) == 2
    assert isinstance(atoms[0], Atoms)
    assert isinstance(atoms[1], Atoms)
    assert len(atoms[0]) == 32
    assert len(atoms[1]) == 32


def test_to_atoms(ar_base_state: BaseState) -> None:
    """Test conversion from BaseState to list of ASE Atoms."""
    atoms = state_to_atoms(ar_base_state)
    assert isinstance(atoms[0], Atoms)


def test_to_structures(ar_base_state: BaseState) -> None:
    """Test conversion from BaseState to list of Pymatgen Structure."""
    structures = state_to_structures(ar_base_state)
    assert isinstance(structures[0], Structure)


def test_single_phonopy_to_state(si_phonopy_atoms: Any, device: torch.device) -> None:
    """Test conversion from PhonopyAtoms to state tensors."""
    state = phonopy_to_state(si_phonopy_atoms, device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert all(
        t.device.type == device.type for t in [state.positions, state.masses, state.cell]
    )
    assert all(
        t.dtype == torch.float64 for t in [state.positions, state.masses, state.cell]
    )
    assert state.atomic_numbers.dtype == torch.int

    # Check shapes and values
    assert state.positions.shape == (8, 3)
    assert torch.allclose(state.masses, torch.full_like(state.masses, 28.0855))  # Si
    assert torch.all(state.atomic_numbers == 14)  # Si atomic number
    assert torch.allclose(
        state.cell,
        torch.diag(torch.full((3,), 5.43, device=device, dtype=torch.float64)),
    )


def test_multiple_phonopy_to_state(si_phonopy_atoms: Any, device: torch.device) -> None:
    """Test conversion from multiple PhonopyAtoms to state tensors."""
    state = phonopy_to_state([si_phonopy_atoms, si_phonopy_atoms], device, torch.float64)

    # Check basic properties
    assert isinstance(state, BaseState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.batch.shape == (16,)
    assert torch.all(
        state.batch == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8),
    )


def test_state_to_phonopy(ar_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of PhonopyAtoms."""
    phonopy_atoms = state_to_phonopy(ar_base_state)
    assert len(phonopy_atoms) == 1
    assert isinstance(phonopy_atoms[0], PhonopyAtoms)
    assert len(phonopy_atoms[0]) == 32


def test_state_to_multiple_phonopy(ar_double_base_state: BaseState) -> None:
    """Test conversion from state tensors to list of PhonopyAtoms."""
    phonopy_atoms = state_to_phonopy(ar_double_base_state)
    assert len(phonopy_atoms) == 2
    assert isinstance(phonopy_atoms[0], PhonopyAtoms)
    assert isinstance(phonopy_atoms[1], PhonopyAtoms)
    assert len(phonopy_atoms[0]) == 32
    assert len(phonopy_atoms[1]) == 32
