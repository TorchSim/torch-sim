from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from ase import Atoms
from ase.build import bulk
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

from torch_sim.models.lennard_jones import LennardJonesModel, UnbatchedLennardJonesModel
from torch_sim.runners import atoms_to_state, concatenate_states
from torch_sim.state import BaseState
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.unbatched.unbatched_integrators import nve


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


@pytest.fixture
def si_atoms() -> Any:
    """Create crystalline silicon using ASE."""
    return bulk("Si", "diamond", a=5.43, cubic=True)


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
def si_base_state(si_atoms: Any, device: torch.device) -> Any:
    """Create a basic state from si_structure."""
    return atoms_to_state(si_atoms, device, torch.float64)


@pytest.fixture
def fe_fcc_state(device: torch.device) -> Any:
    fe_atoms = bulk("Fe", "fcc", a=5.26, cubic=True).repeat([4, 4, 4])
    return atoms_to_state(fe_atoms, device, torch.float64)


@pytest.fixture
def si_double_base_state(si_atoms: Atoms, device: torch.device) -> Any:
    """Create a basic state from si_structure."""
    return atoms_to_state([si_atoms, si_atoms], device, torch.float64)


@pytest.fixture
def ar_base_state(device: torch.device) -> BaseState:
    """Create a face-centered cubic (FCC) Argon structure."""
    # Create FCC Ar using ASE, with 4x4x4 supercell
    ar_atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([2, 2, 2])
    return atoms_to_state(ar_atoms, device, torch.float64)


@pytest.fixture
def ar_double_base_state(ar_base_state: BaseState) -> BaseState:
    """Create a batched state from ar_fcc_base_state."""
    return concatenate_states(
        [ar_base_state, ar_base_state], device=ar_base_state.device
    )


@pytest.fixture
def unbatched_lj_calculator(device: torch.device) -> UnbatchedLennardJonesModel:
    """Create a Lennard-Jones calculator with reasonable parameters for Ar."""
    return UnbatchedLennardJonesModel(
        use_neighbor_list=True,
        sigma=3.405,
        epsilon=0.0104,
        device=device,
        dtype=torch.float64,
        compute_force=True,
        compute_stress=True,
        cutoff=2.5 * 3.405,
    )


@pytest.fixture
def lj_calculator(device: torch.device) -> LennardJonesModel:
    """Create a Lennard-Jones calculator with reasonable parameters for Ar."""
    return LennardJonesModel(
        use_neighbor_list=True,
        sigma=3.405,
        epsilon=0.0104,
        device=device,
        dtype=torch.float64,
        compute_force=True,
        compute_stress=True,
        cutoff=2.5 * 3.405,
    )


@pytest.fixture
def torchsim_trajectory(si_base_state: BaseState, lj_calculator: Any, tmp_path: Path):
    """Test NVE integration conserves energy."""
    # Initialize integrator
    kT = torch.tensor(300.0)  # Temperature in K
    dt = torch.tensor(0.001)  # Small timestep for stability

    state, update_fn = nve(
        **asdict(si_base_state),
        model=lj_calculator,
        dt=dt,
        kT=kT,
    )

    reporter = TrajectoryReporter(tmp_path / "test.hdf5", state_frequency=1)

    # Run several steps
    for step in range(10):
        state = update_fn(state, dt)
        reporter.report(state, step)

    yield reporter.trajectory

    reporter.close()
