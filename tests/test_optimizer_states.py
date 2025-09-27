"""Unit tests for optimizer state classes."""

import pytest
import torch

from torch_sim.optimizers.state import FireState, OptimState
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
    state = OptimState(**vars(sim_state), **optim_data)
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

    state = FireState(**vars(sim_state), **optim_data, **fire_data)

    assert torch.equal(state.velocities, fire_data["velocities"])
    assert torch.equal(state.dt, fire_data["dt"])
    assert torch.equal(state.alpha, fire_data["alpha"])
    assert torch.equal(state.n_pos, fire_data["n_pos"])
