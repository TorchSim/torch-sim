from typing import Any

import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.testing import SIMSTATE_GENERATORS


DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _make_simstate_fixture(name: str) -> pytest.fixture:
    """Create a pytest fixture for a sim_state generator."""

    @pytest.fixture(name=name)
    def _fixture() -> ts.SimState:
        return SIMSTATE_GENERATORS[name](DEVICE, DTYPE)

    return _fixture


# Programmatically generate fixtures for all sim_state generators
for _name in SIMSTATE_GENERATORS:
    globals()[_name] = _make_simstate_fixture(_name)


@pytest.fixture
def lj_model() -> LennardJonesModel:
    """Create a Lennard-Jones model with reasonable parameters for Ar."""
    return LennardJonesModel(
        use_neighbor_list=True,
        sigma=3.405,
        epsilon=0.0104,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=True,
        cutoff=2.5 * 3.405,
    )


@pytest.fixture
def ar_atoms() -> Atoms:
    """Create a face-centered cubic (FCC) Argon structure."""
    return bulk("Ar", "fcc", a=5.26, cubic=True)


@pytest.fixture
def fe_atoms() -> Atoms:
    """Create crystalline iron using ASE."""
    return bulk("Fe", "fcc", a=5.26, cubic=True)


@pytest.fixture
def si_atoms() -> Atoms:
    """Create crystalline silicon using ASE."""
    return bulk("Si", "diamond", a=5.43, cubic=True)


@pytest.fixture
def benzene_atoms() -> Atoms:
    """Create benzene using ASE."""
    return molecule("C6H6")


@pytest.fixture
def si_structure() -> Structure:
    """Create crystalline silicon using pymatgen."""
    lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
    species = ["Si"] * 8
    coords = [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def si_phonopy_atoms() -> Any:
    """Create crystalline silicon using PhonopyAtoms."""
    lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
    species = ["Si"] * 8
    coords = [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ]
    return PhonopyAtoms(
        cell=lattice,
        scaled_positions=coords,
        symbols=species,
        pbc=True,
    )


@pytest.fixture
def ar_double_sim_state(ar_supercell_sim_state: ts.SimState) -> ts.SimState:
    """Create a batched state from ar_fcc_sim_state."""
    return ts.concatenate_states(
        [ar_supercell_sim_state, ar_supercell_sim_state],
        device=ar_supercell_sim_state.device,
    )


@pytest.fixture
def si_double_sim_state(si_sim_state: ts.SimState) -> ts.SimState:
    """Create a basic state from si_structure."""
    return ts.concatenate_states(
        [si_sim_state, si_sim_state],
        device=si_sim_state.device,
    )


@pytest.fixture
def mixed_double_sim_state(
    ar_supercell_sim_state: ts.SimState, si_sim_state: ts.SimState
) -> ts.SimState:
    """Create a batched state from ar_fcc_sim_state."""
    return ts.concatenate_states(
        [ar_supercell_sim_state, si_sim_state],
        device=ar_supercell_sim_state.device,
    )
