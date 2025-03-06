from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from ase import Atoms
from ase.build import bulk
from pymatgen.core import Structure

from torchsim.models.lennard_jones import LennardJonesModel, UnbatchedLennardJonesModel
from torchsim.runners import atoms_to_state, concatenate_states
from torchsim.state import BaseState
from torchsim.trajectory import TrajectoryReporter
from torchsim.unbatched_integrators import nve


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


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
def ar_fcc_base_state(device: torch.device) -> BaseState:
    """Create a face-centered cubic (FCC) Argon structure."""
    # 5.26 Å is a typical lattice constant for Ar
    a = 5.26  # Lattice constant
    N = 4  # Supercell size
    n_atoms = 4 * N * N * N  # Total number of atoms (4 atoms per unit cell)
    dtype = torch.float64

    # Create positions tensor directly
    positions = torch.zeros((n_atoms, 3), device=device, dtype=dtype)
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Add base FCC positions with offset
                positions[idx] = torch.tensor([i, j, k], device=device, dtype=dtype) * a
                positions[idx + 1] = (
                    torch.tensor([i, j + 0.5, k + 0.5], device=device, dtype=dtype) * a
                )
                positions[idx + 2] = (
                    torch.tensor([i + 0.5, j, k + 0.5], device=device, dtype=dtype) * a
                )
                positions[idx + 3] = (
                    torch.tensor([i + 0.5, j + 0.5, k], device=device, dtype=dtype) * a
                )
                idx += 4

    # Create cell tensor with shape (1, 3, 3) to match atoms_to_state format
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * (N * a)

    # Create batch indices
    batch = torch.zeros(n_atoms, device=device, dtype=torch.long)

    return BaseState(
        positions=positions,
        masses=torch.full((n_atoms,), 39.95, device=device, dtype=dtype),  # Ar mass
        cell=cell,  # Cubic cell
        pbc=True,
        atomic_numbers=torch.full(
            (n_atoms,), 18, device=device, dtype=torch.long
        ),  # Ar atomic number
        batch=batch,
    )


@pytest.fixture
def ar_fcc_batched_base_state(ar_fcc_base_state: BaseState) -> BaseState:
    """Create a batched state from ar_fcc_base_state."""
    return concatenate_states(
        [ar_fcc_base_state, ar_fcc_base_state], device=ar_fcc_base_state.device
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
