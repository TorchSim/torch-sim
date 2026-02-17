"""Unit tests for optimizer state classes."""

import pytest
import torch

from torch_sim.optimizers.state import BFGSState, FireState, LBFGSState, OptimState
from torch_sim.state import SimState


@pytest.fixture
def sim_state() -> SimState:
    """Basic SimState for testing."""
    return SimState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64),
        masses=torch.tensor([1.0, 2.0], dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 6], dtype=torch.int64),
        system_idx=torch.zeros(2, dtype=torch.int64),
    )


@pytest.fixture
def optim_data() -> dict:
    """Optimizer state data."""
    return {
        "forces": torch.tensor(
            [[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=torch.float64
        ),
        "energy": torch.tensor([1.5], dtype=torch.float64),
        "stress": torch.zeros(1, 3, 3, dtype=torch.float64),
    }


def test_optim_state_init(sim_state: SimState, optim_data: dict) -> None:
    """Test OptimState initialization."""
    state = OptimState(**sim_state.attributes, **optim_data)
    assert torch.equal(state.forces, optim_data["forces"])
    assert torch.equal(state.energy, optim_data["energy"])
    assert torch.equal(state.stress, optim_data["stress"])


def test_fire_state_custom_values(sim_state: SimState, optim_data: dict) -> None:
    """Test FireState with custom values."""
    fire_data = {
        "velocities": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64
        ),
        "dt": torch.tensor([0.01], dtype=torch.float64),
        "alpha": torch.tensor([0.1], dtype=torch.float64),
        "n_pos": torch.tensor([5], dtype=torch.int32),
    }

    state = FireState(**sim_state.attributes, **optim_data, **fire_data)

    assert torch.equal(state.velocities, fire_data["velocities"])
    assert torch.equal(state.dt, fire_data["dt"])
    assert torch.equal(state.alpha, fire_data["alpha"])
    assert torch.equal(state.n_pos, fire_data["n_pos"])


def test_bfgs_state_custom_values(sim_state: SimState, optim_data: dict) -> None:
    """Test BFGSState with custom values."""
    bfgs_data = {
        "hessian": torch.eye(6, dtype=torch.float64).unsqueeze(0),  # [1, 6, 6]
        "prev_forces": optim_data["forces"].clone(),
        "prev_positions": sim_state.positions.clone(),
        "alpha": torch.tensor([70.0], dtype=torch.float64),
        "max_step": torch.tensor([0.2], dtype=torch.float64),
        "n_iter": torch.tensor([0], dtype=torch.int32),
        "atom_idx_in_system": torch.arange(2, dtype=torch.int64),
        "max_atoms": torch.tensor([2], dtype=torch.int64),
    }

    state = BFGSState(**sim_state.attributes, **optim_data, **bfgs_data)

    assert torch.equal(state.hessian, bfgs_data["hessian"])
    assert torch.equal(state.prev_forces, bfgs_data["prev_forces"])
    assert torch.equal(state.prev_positions, bfgs_data["prev_positions"])
    assert torch.equal(state.alpha, bfgs_data["alpha"])
    assert torch.equal(state.max_step, bfgs_data["max_step"])
    assert torch.equal(state.n_iter, bfgs_data["n_iter"])
    assert torch.equal(state.atom_idx_in_system, bfgs_data["atom_idx_in_system"])
    assert torch.equal(state.max_atoms, bfgs_data["max_atoms"])


def test_lbfgs_state_custom_values(sim_state: SimState, optim_data: dict) -> None:
    """Test LBFGSState with custom values."""
    lbfgs_data = {
        "prev_forces": optim_data["forces"].clone(),
        "prev_positions": sim_state.positions.clone(),
        "s_history": torch.zeros((1, 0, 2, 3), dtype=torch.float64),  # [S, H, M, 3]
        "y_history": torch.zeros((1, 0, 2, 3), dtype=torch.float64),
        "step_size": torch.tensor([1.0], dtype=torch.float64),
        "alpha": torch.tensor([70.0], dtype=torch.float64),
        "n_iter": torch.tensor([0], dtype=torch.int32),
        "max_atoms": torch.tensor([2], dtype=torch.int64),
    }

    state = LBFGSState(**sim_state.attributes, **optim_data, **lbfgs_data)

    assert torch.equal(state.prev_forces, lbfgs_data["prev_forces"])
    assert torch.equal(state.prev_positions, lbfgs_data["prev_positions"])
    assert torch.equal(state.s_history, lbfgs_data["s_history"])
    assert torch.equal(state.y_history, lbfgs_data["y_history"])
    assert torch.equal(state.step_size, lbfgs_data["step_size"])
    assert torch.equal(state.alpha, lbfgs_data["alpha"])
    assert torch.equal(state.n_iter, lbfgs_data["n_iter"])
    assert torch.equal(state.max_atoms, lbfgs_data["max_atoms"])
