import itertools
import sys
from typing import Any

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim.state import SimState
from torch_sim.typing import AtomExtras, SystemExtras


def test_single_atoms_to_state(si_atoms: Atoms) -> None:
    """Test basic shape/dtype/device properties of atoms_to_state."""
    state = ts.io.atoms_to_state(si_atoms, DEVICE, torch.float64)
    assert isinstance(state, SimState)
    assert state.positions.shape == (8, 3)
    assert state.masses.shape == (8,)
    assert state.cell.shape == (1, 3, 3)
    assert torch.all(state.pbc)
    assert state.atomic_numbers.shape == (8,)
    assert state.system_idx is not None
    assert torch.all(state.system_idx == 0)
    assert all(
        t.device.type == DEVICE.type for t in (state.positions, state.masses, state.cell)
    )
    assert all(
        t.dtype == torch.float64 for t in (state.positions, state.masses, state.cell)
    )
    assert state.atomic_numbers.dtype == torch.int


@pytest.mark.parametrize(
    ("system_extras_map", "atom_extras_map", "expected_sys", "expected_atom"),
    [
        pytest.param(None, None, {}, {}, id="no-extras-by-default"),
        pytest.param(
            {"charge": "charge", "spin": "spin"},
            None,
            {"charge": 3.0, "spin": 2.0},
            {},
            id="system-extras-identity-map",
        ),
        pytest.param(
            {"total_charge": "charge"},
            None,
            {"total_charge": 3.0},
            {},
            id="system-extras-rename",
        ),
        pytest.param(
            None,
            {"site_tags": "my_tags"},
            {},
            {"site_tags": [1.0, 2.0, 3.0]},
            id="atom-extras-rename",
        ),
    ],
)
def test_extras_map_import(
    system_extras_map: dict[SystemExtras, str] | None,
    atom_extras_map: dict[AtomExtras, str] | None,
    expected_sys: dict[str, float],
    expected_atom: dict[str, list[float]],
) -> None:
    """test how system_extras_map and atom_extras_map control which keys are
    read and how they are renamed on import.
    """
    mol = molecule("H2O")
    mol.info["charge"] = 3.0
    mol.info["spin"] = 2.0
    mol.arrays["my_tags"] = np.array([1.0, 2.0, 3.0])
    state = ts.io.atoms_to_state(
        [mol],
        DEVICE,
        DTYPE,
        system_extras_map=system_extras_map,
        atom_extras_map=atom_extras_map,
    )
    if not expected_sys and not expected_atom:
        assert not state.system_extras
        assert not state.atom_extras
    for key, val in expected_sys.items():
        assert getattr(state, key)[0].item() == val
    for key, vals in expected_atom.items():
        assert getattr(state, key).shape == (len(vals),)


def test_extras_map_missing_key_skipped() -> None:
    """Missing ASE keys are silently skipped rather than defaulting to zero."""
    mol = molecule("H2O")
    state = ts.io.atoms_to_state(
        [mol], DEVICE, DTYPE, system_extras_map={"charge": "charge"}
    )
    assert not state.system_extras


def test_extras_map_multi_system() -> None:
    """System extras work across multiple systems with correct per-system values."""
    mol1, mol2 = molecule("H2O"), molecule("CH4")
    mol1.info["charge"] = 1.0
    mol2.info["charge"] = -1.0
    state = ts.io.atoms_to_state(
        [mol1, mol2], DEVICE, DTYPE, system_extras_map={"charge": "charge"}
    )
    assert state.charge.shape == (2,)
    assert state.charge[0].item() == 1.0
    assert state.charge[1].item() == -1.0


def test_extras_map_export_roundtrip() -> None:
    """System and atom extras round-trip through state_to_atoms with rename."""
    mol = molecule("H2O")
    mol.info["charge"] = 5.0
    mol.arrays["my_tags"] = np.array([1.0, 2.0, 3.0])
    sys_map = {"total_charge": "charge"}
    atom_map = {"site_tags": "my_tags"}
    state = ts.io.atoms_to_state(
        [mol], DEVICE, DTYPE, system_extras_map=sys_map, atom_extras_map=atom_map
    )
    atoms = ts.io.state_to_atoms(
        state, system_extras_map=sys_map, atom_extras_map=atom_map
    )
    assert atoms[0].info["charge"] == 5.0
    np.testing.assert_allclose(atoms[0].arrays["my_tags"], [1.0, 2.0, 3.0])
    atoms_no_map = ts.io.state_to_atoms(state)
    assert "charge" not in atoms_no_map[0].info


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
        ],
        [
            (ts.io.state_to_atoms, ts.io.atoms_to_state),
            (ts.io.state_to_structures, ts.io.structures_to_state),
            (ts.io.state_to_phonopy, ts.io.phonopy_to_state),
        ],
    ),
)
def test_state_round_trip(
    sim_state_name: str, conversion_functions: tuple, request: pytest.FixtureRequest
) -> None:
    """Test round-trip conversion from SimState through various formats and back."""
    sim_state: SimState = request.getfixturevalue(sim_state_name)
    to_format_fn, from_format_fn = conversion_functions
    assert sim_state.system_idx is not None
    uniq_systems = torch.unique(sim_state.system_idx)
    intermediate_format = to_format_fn(sim_state)
    assert len(intermediate_format) == len(uniq_systems)
    round_trip_state: SimState = from_format_fn(intermediate_format, DEVICE, DTYPE)
    assert round_trip_state.system_idx is not None
    assert torch.allclose(sim_state.positions, round_trip_state.positions)
    assert torch.allclose(sim_state.cell, round_trip_state.cell)
    assert torch.all(sim_state.atomic_numbers == round_trip_state.atomic_numbers)
    assert torch.all(sim_state.system_idx == round_trip_state.system_idx)
    assert torch.equal(sim_state.pbc, round_trip_state.pbc)
    if isinstance(intermediate_format[0], Atoms):
        assert torch.allclose(sim_state.masses, round_trip_state.masses)


@pytest.mark.parametrize(
    ("monkeypatch_modules", "func", "args", "match"),
    [
        (
            ["ase", "ase.data"],
            ts.io.state_to_atoms,
            (None,),
            "ASE is required for state_to_atoms",
        ),
        (
            ["ase", "ase.data"],
            ts.io.atoms_to_state,
            (None, None, None),
            "ASE is required for atoms_to_state",
        ),
        (
            ["phonopy", "phonopy.structure", "phonopy.structure.atoms"],
            ts.io.state_to_phonopy,
            (None,),
            "Phonopy is required for state_to_phonopy",
        ),
        (
            ["phonopy", "phonopy.structure", "phonopy.structure.atoms"],
            ts.io.phonopy_to_state,
            (None, None, None),
            "Phonopy is required for phonopy_to_state",
        ),
        (
            ["pymatgen", "pymatgen.core", "pymatgen.core.structure"],
            ts.io.state_to_structures,
            (None,),
            "Pymatgen is required for state_to_structures",
        ),
        (
            ["pymatgen", "pymatgen.core", "pymatgen.core.structure"],
            ts.io.structures_to_state,
            (None, None, None),
            "Pymatgen is required for structures_to_state",
        ),
    ],
)
def test_import_errors(
    monkeypatch: pytest.MonkeyPatch,
    monkeypatch_modules: list[str],
    func: Any,
    args: tuple,
    match: str,
) -> None:
    """All IO functions raise ImportError when their backend is unavailable."""
    for mod in monkeypatch_modules:
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(ImportError, match=match):
        func(*args)
