from typing import Any

import numpy as np
import pytest
import torch
import torch.distributions.weibull
from ase import Atoms
from ase.build import bulk, molecule
from ase.spacegroup import crystal
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.mace import MaceModel


DEVICE = torch.device("cpu")
DTYPE = torch.float64


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
def si_sim_state(si_atoms: Any) -> Any:
    """Create a basic state from si_structure."""
    return ts.io.atoms_to_state(si_atoms, DEVICE, DTYPE)


@pytest.fixture
def cu_sim_state() -> ts.SimState:
    """Create crystalline copper using ASE."""
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def mg_sim_state() -> ts.SimState:
    """Create crystalline magnesium using ASE."""
    atoms = bulk("Mg", "hcp", a=3.17, c=5.14)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def sb_sim_state() -> ts.SimState:
    """Create crystalline antimony using ASE."""
    atoms = bulk("Sb", "rhombohedral", a=4.58, alpha=60)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def ti_sim_state() -> ts.SimState:
    """Create crystalline titanium using ASE."""
    atoms = bulk("Ti", "hcp", a=2.94, c=4.64)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def tio2_sim_state() -> ts.SimState:
    """Create crystalline TiO2 using ASE."""
    a, c = 4.60, 2.96
    basis = [("Ti", 0.5, 0.5, 0), ("O", 0.695679, 0.695679, 0.5)]
    atoms = crystal(
        symbols=[b[0] for b in basis],
        basis=[b[1:] for b in basis],
        spacegroup=136,  # P4_2/mnm
        cellpar=[a, a, c, 90, 90, 90],
    )
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def ga_sim_state() -> ts.SimState:
    """Create crystalline Ga using ASE."""
    a, b, c = 4.43, 7.60, 4.56
    basis = [("Ga", 0, 0.344304, 0.415401)]
    atoms = crystal(
        symbols=[b[0] for b in basis],
        basis=[b[1:] for b in basis],
        spacegroup=64,  # Cmce
        cellpar=[a, b, c, 90, 90, 90],
    )
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def niti_sim_state() -> ts.SimState:
    """Create crystalline NiTi using ASE."""
    a, b, c = 2.89, 3.97, 4.83
    alpha, beta, gamma = 90.00, 105.23, 90.00
    basis = [
        ("Ni", 0.369548, 0.25, 0.217074),
        ("Ti", 0.076622, 0.25, 0.671102),
    ]
    atoms = crystal(
        symbols=[b[0] for b in basis],
        basis=[b[1:] for b in basis],
        spacegroup=11,
        cellpar=[a, b, c, alpha, beta, gamma],
    )
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def sio2_sim_state() -> ts.SimState:
    """Create an alpha-quartz SiO2 system for testing."""
    atoms = crystal(
        symbols=["O", "Si"],
        basis=[[0.413, 0.2711, 0.2172], [0.4673, 0, 0.3333]],
        spacegroup=152,
        cellpar=[4.9019, 4.9019, 5.3988, 90, 90, 120],
    )
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def rattled_sio2_sim_state(sio2_sim_state: ts.SimState) -> ts.SimState:
    """Create a rattled SiO2 system for testing."""
    sim_state = sio2_sim_state.clone()

    # Store the current RNG state
    rng_state = torch.random.get_rng_state()
    try:
        # Temporarily set a fixed seed
        torch.manual_seed(3)
        weibull = torch.distributions.weibull.Weibull(scale=0.1, concentration=1)
        rnd = torch.randn_like(sim_state.positions, device=DEVICE, dtype=DTYPE)
        rnd = rnd / torch.norm(rnd, dim=-1, keepdim=True).to(device=DEVICE)
        shifts = weibull.sample(rnd.shape).to(device=DEVICE) * rnd
        sim_state.positions = sim_state.positions + shifts
    finally:
        # Restore the original RNG state
        torch.random.set_rng_state(rng_state)

    return sim_state


@pytest.fixture
def rattled_si_sim_state(si_sim_state: ts.SimState) -> ts.SimState:
    """Create a rattled Si system for testing."""
    sim_state = si_sim_state.clone()

    # Store the current RNG state
    rng_state = torch.random.get_rng_state()
    try:
        # Temporarily set a fixed seed
        torch.manual_seed(3)
        weibull = torch.distributions.weibull.Weibull(scale=0.1, concentration=1)
        rnd = torch.randn_like(sim_state.positions, device=DEVICE, dtype=DTYPE)
        rnd = rnd / torch.norm(rnd, dim=-1, keepdim=True).to(device=DEVICE)
        shifts = weibull.sample(rnd.shape).to(device=DEVICE) * rnd
        sim_state.positions = sim_state.positions + shifts
    finally:
        # Restore the original RNG state
        torch.random.set_rng_state(rng_state)

    return sim_state


@pytest.fixture
def casio3_sim_state() -> ts.SimState:
    a, b, c = 7.9258, 7.3202, 7.0653
    alpha, beta, gamma = 90.055, 95.217, 103.426
    basis = [
        ("Ca", 0.19831, 0.42266, 0.76060),
        ("Ca", 0.20241, 0.92919, 0.76401),
        ("Ca", 0.50333, 0.75040, 0.52691),
        ("Si", 0.1851, 0.3875, 0.2684),
        ("Si", 0.1849, 0.9542, 0.2691),
        ("Si", 0.3973, 0.7236, 0.0561),
        ("O", 0.3034, 0.4616, 0.4628),
        ("O", 0.3014, 0.9385, 0.4641),
        ("O", 0.5705, 0.7688, 0.1988),
        ("O", 0.9832, 0.3739, 0.2655),
        ("O", 0.9819, 0.8677, 0.2648),
        ("O", 0.4018, 0.7266, 0.8296),
        ("O", 0.2183, 0.1785, 0.2254),
        ("O", 0.2713, 0.8704, 0.0938),
        ("O", 0.2735, 0.5126, 0.0931),
    ]
    atoms = crystal(
        symbols=[b[0] for b in basis],
        basis=[b[1:] for b in basis],
        spacegroup=2,
        cellpar=[a, b, c, alpha, beta, gamma],
    )
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


@pytest.fixture
def benzene_sim_state(benzene_atoms: Any) -> Any:
    """Create a basic state from benzene_atoms."""
    return ts.io.atoms_to_state(benzene_atoms, DEVICE, DTYPE)


@pytest.fixture
def fe_supercell_sim_state(fe_atoms: Atoms) -> Any:
    """Create a face-centered cubic (FCC) iron structure with 4x4x4 supercell."""
    return ts.io.atoms_to_state(fe_atoms.repeat([4, 4, 4]), DEVICE, DTYPE)


@pytest.fixture
def ar_supercell_sim_state(ar_atoms: Atoms) -> ts.SimState:
    """Create a face-centered cubic (FCC) Argon structure with 2x2x2 supercell."""
    return ts.io.atoms_to_state(ar_atoms.repeat([2, 2, 2]), DEVICE, DTYPE)


@pytest.fixture
def ar_double_sim_state(ar_supercell_sim_state: ts.SimState) -> ts.SimState:
    """Create a batched state from ar_fcc_sim_state."""
    return ts.concatenate_states(
        [ar_supercell_sim_state, ar_supercell_sim_state],
        device=ar_supercell_sim_state.device,
    )


@pytest.fixture
def si_double_sim_state(si_atoms: Atoms) -> Any:
    """Create a basic state from si_structure."""
    return ts.io.atoms_to_state([si_atoms, si_atoms], DEVICE, DTYPE)


@pytest.fixture
def mixed_double_sim_state(
    ar_supercell_sim_state: ts.SimState, si_sim_state: ts.SimState
) -> ts.SimState:
    """Create a batched state from ar_fcc_sim_state."""
    return ts.concatenate_states(
        [ar_supercell_sim_state, si_sim_state],
        device=ar_supercell_sim_state.device,
    )


@pytest.fixture
def osn2_sim_state(ts_mace_mpa: MaceModel) -> ts.SimState:
    """Provides an initial SimState for rhombohedral OsN2."""
    # For pymatgen Structure initialization
    from pymatgen.core import Lattice, Structure

    a = 3.211996
    lattice = Lattice.from_parameters(a, a, a, 60, 60, 60)
    species = ["Os", "N"]
    frac_coords = [[0.75, 0.7501, -0.25], [0, 0, 0]]  # Slightly perturbed
    structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    return ts.initialize_state(
        structure, dtype=ts_mace_mpa.dtype, device=ts_mace_mpa.device
    )


@pytest.fixture
def distorted_fcc_al_conventional_sim_state(ts_mace_mpa: MaceModel) -> ts.SimState:
    """Initial SimState for a slightly distorted FCC Al conventional cell (4 atoms)."""
    # Create a standard 4-atom conventional FCC Al cell
    atoms_fcc = bulk("Al", crystalstructure="fcc", a=4.05, cubic=True)

    # Define a small triclinic strain matrix (deviations from identity)
    strain_matrix = np.array([[1.0, 0.05, -0.03], [0.04, 1.0, 0.06], [-0.02, 0.03, 1.0]])

    original_cell = atoms_fcc.get_cell()
    new_cell = original_cell @ strain_matrix.T  # Apply strain
    atoms_fcc.set_cell(new_cell, scale_atoms=True)

    # Slightly perturb atomic positions to break perfect symmetry after strain
    positions = atoms_fcc.get_positions()
    np_rng = np.random.default_rng(seed=42)
    positions += np_rng.normal(scale=0.01, size=positions.shape)
    atoms_fcc.set_positions(positions)

    dtype = ts_mace_mpa.dtype
    device = ts_mace_mpa.device
    # Convert the ASE Atoms object to SimState (will be a single batch with 4 atoms)
    return ts.io.atoms_to_state(atoms_fcc, device=device, dtype=dtype)
