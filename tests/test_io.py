import itertools
import sys
from typing import Any

import pytest
import torch
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

import torch_sim as ts


def test_single_structure_to_state(si_structure: Structure, device: torch.device) -> None:
    """Test conversion from pymatgen Structure to state tensors."""
    state = ts.io.structures_to_state(si_structure, device, torch.float64)

    # Check basic properties
    assert isinstance(state, ts.SimState)
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
    state = ts.io.structures_to_state([si_structure, si_structure], device, torch.float64)

    # Check basic properties
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.system_idx.shape == (16,)
    assert torch.all(
        state.system_idx
        == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8)
    )


def test_single_atoms_to_state(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to state tensors."""
    state = ts.io.atoms_to_state(si_atoms, device, torch.float64)

    # Check basic properties
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == (8, 3)
    assert state.masses.shape == (8,)
    assert state.cell.shape == (1, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (8,)
    assert state.system_idx.shape == (8,)
    assert torch.all(state.system_idx == 0)


def test_multiple_atoms_to_state(si_atoms: Atoms, device: torch.device) -> None:
    """Test conversion from ASE Atoms to state tensors."""
    state = ts.io.atoms_to_state([si_atoms, si_atoms], device, torch.float64)

    # Check basic properties
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.system_idx.shape == (16,)
    assert torch.all(
        state.system_idx
        == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8),
    )


def test_state_to_structure(ar_supercell_sim_state: ts.SimState) -> None:
    """Test conversion from state tensors to list of pymatgen Structure."""
    structures = ts.io.state_to_structures(ar_supercell_sim_state)
    assert len(structures) == 1
    assert isinstance(structures[0], Structure)
    assert len(structures[0]) == 32


def test_state_to_multiple_structures(ar_double_sim_state: ts.SimState) -> None:
    """Test conversion from state tensors to list of pymatgen Structure."""
    structures = ts.io.state_to_structures(ar_double_sim_state)
    assert len(structures) == 2
    assert isinstance(structures[0], Structure)
    assert isinstance(structures[1], Structure)
    assert len(structures[0]) == 32
    assert len(structures[1]) == 32


def test_state_to_atoms(ar_supercell_sim_state: ts.SimState) -> None:
    """Test conversion from state tensors to list of ASE Atoms."""
    atoms = ts.io.state_to_atoms(ar_supercell_sim_state)
    assert len(atoms) == 1
    assert isinstance(atoms[0], Atoms)
    assert len(atoms[0]) == 32


def test_state_to_multiple_atoms(ar_double_sim_state: ts.SimState) -> None:
    """Test conversion from state tensors to list of ASE Atoms."""
    atoms = ts.io.state_to_atoms(ar_double_sim_state)
    assert len(atoms) == 2
    assert isinstance(atoms[0], Atoms)
    assert isinstance(atoms[1], Atoms)
    assert len(atoms[0]) == 32
    assert len(atoms[1]) == 32


def test_to_atoms(ar_supercell_sim_state: ts.SimState) -> None:
    """Test conversion from SimState to list of ASE Atoms."""
    atoms = ts.io.state_to_atoms(ar_supercell_sim_state)
    assert isinstance(atoms[0], Atoms)


def test_to_structures(ar_supercell_sim_state: ts.SimState) -> None:
    """Test conversion from SimState to list of Pymatgen Structure."""
    structures = ts.io.state_to_structures(ar_supercell_sim_state)
    assert isinstance(structures[0], Structure)


def test_single_phonopy_to_state(si_phonopy_atoms: Any, device: torch.device) -> None:
    """Test conversion from PhonopyAtoms to state tensors."""
    state = ts.io.phonopy_to_state(si_phonopy_atoms, device, torch.float64)

    # Check basic properties
    assert isinstance(state, ts.SimState)
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
    state = ts.io.phonopy_to_state(
        [si_phonopy_atoms, si_phonopy_atoms], device, torch.float64
    )

    # Check basic properties
    assert isinstance(state, ts.SimState)
    assert state.positions.shape == (16, 3)
    assert state.masses.shape == (16,)
    assert state.cell.shape == (2, 3, 3)
    assert state.pbc
    assert state.atomic_numbers.shape == (16,)
    assert state.system_idx.shape == (16,)
    assert torch.all(
        state.system_idx
        == torch.repeat_interleave(torch.tensor([0, 1], device=device), 8),
    )


def test_state_to_phonopy(ar_supercell_sim_state: ts.SimState) -> None:
    """Test conversion from state tensors to list of PhonopyAtoms."""
    phonopy_atoms = ts.io.state_to_phonopy(ar_supercell_sim_state)
    assert len(phonopy_atoms) == 1
    assert isinstance(phonopy_atoms[0], PhonopyAtoms)
    assert len(phonopy_atoms[0]) == 32


def test_state_to_multiple_phonopy(ar_double_sim_state: ts.SimState) -> None:
    """Test conversion from state tensors to list of PhonopyAtoms."""
    phonopy_atoms = ts.io.state_to_phonopy(ar_double_sim_state)
    assert len(phonopy_atoms) == 2
    assert isinstance(phonopy_atoms[0], PhonopyAtoms)
    assert isinstance(phonopy_atoms[1], PhonopyAtoms)
    assert len(phonopy_atoms[0]) == 32
    assert len(phonopy_atoms[1]) == 32


@pytest.mark.parametrize(
    ("sim_state_name", "conversion_functions"),
    itertools.product(
        [
            "ar_supercell_sim_state",
            "si_sim_state",
            "ti_sim_state",
            "sio2_sim_state",
            "fe_supercell_sim_state",
            "cu_sim_state",
            "ar_double_sim_state",
            "mixed_double_sim_state",
            # TODO: round trip benzene/non-pbc systems
        ],
        [
            (ts.io.state_to_atoms, ts.io.atoms_to_state),
            (ts.io.state_to_structures, ts.io.structures_to_state),
            (ts.io.state_to_phonopy, ts.io.phonopy_to_state),
        ],
    ),
)
def test_state_round_trip(
    sim_state_name: str,
    conversion_functions: tuple,
    request: pytest.FixtureRequest,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Test round-trip conversion from SimState through various formats and back.

    Args:
        sim_state_name: Name of the sim_state fixture to test
        conversion_functions: Tuple of (to_format, from_format) conversion functions
        request: Pytest fixture request object to get dynamic fixtures
        device: Device to run tests on
        dtype: Data type to use
    """
    # Get the sim_state fixture dynamically using the name
    sim_state: ts.SimState = request.getfixturevalue(sim_state_name)
    to_format_fn, from_format_fn = conversion_functions
    unique_systems = torch.unique(sim_state.system_idx)

    # Convert to intermediate format
    intermediate_format = to_format_fn(sim_state)
    assert len(intermediate_format) == len(unique_systems)

    # Convert back to state
    round_trip_state: ts.SimState = from_format_fn(intermediate_format, device, dtype)

    # Check that all properties match
    assert torch.allclose(sim_state.positions, round_trip_state.positions)
    assert torch.allclose(sim_state.cell, round_trip_state.cell)
    assert torch.all(sim_state.atomic_numbers == round_trip_state.atomic_numbers)
    assert torch.all(sim_state.system_idx == round_trip_state.system_idx)
    assert sim_state.pbc == round_trip_state.pbc

    if isinstance(intermediate_format[0], Atoms):
        # TODO: masses round trip for pmg and phonopy masses is not exact
        # since both use their own isotope masses based on species,
        # not the ones in the state
        assert torch.allclose(sim_state.masses, round_trip_state.masses)


def test_state_to_atoms_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ase", None)
    monkeypatch.setitem(sys.modules, "ase.data", None)

    with pytest.raises(
        ImportError, match="ASE is required for state_to_atoms conversion"
    ):
        ts.io.state_to_atoms(None)


def test_state_to_phonopy_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "phonopy", None)
    monkeypatch.setitem(sys.modules, "phonopy.structure", None)
    monkeypatch.setitem(sys.modules, "phonopy.structure.atoms", None)

    with pytest.raises(
        ImportError, match="Phonopy is required for state_to_phonopy conversion"
    ):
        ts.io.state_to_phonopy(None)


def test_state_to_structures_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pymatgen", None)
    monkeypatch.setitem(sys.modules, "pymatgen.core", None)
    monkeypatch.setitem(sys.modules, "pymatgen.core.structure", None)

    with pytest.raises(
        ImportError, match="Pymatgen is required for state_to_structures conversion"
    ):
        ts.io.state_to_structures(None)


def test_atoms_to_state_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ase", None)
    monkeypatch.setitem(sys.modules, "ase.data", None)

    with pytest.raises(
        ImportError, match="ASE is required for atoms_to_state conversion"
    ):
        ts.io.atoms_to_state(None, None, None)


def test_phonopy_to_state_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "phonopy", None)
    monkeypatch.setitem(sys.modules, "phonopy.structure", None)
    monkeypatch.setitem(sys.modules, "phonopy.structure.atoms", None)

    with pytest.raises(
        ImportError, match="Phonopy is required for phonopy_to_state conversion"
    ):
        ts.io.phonopy_to_state(None, None, None)


def test_structures_to_state_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pymatgen", None)
    monkeypatch.setitem(sys.modules, "pymatgen.core", None)
    monkeypatch.setitem(sys.modules, "pymatgen.core.structure", None)

    with pytest.raises(
        ImportError, match="Pymatgen is required for structures_to_state conversion"
    ):
        ts.io.structures_to_state(None, None, None)
