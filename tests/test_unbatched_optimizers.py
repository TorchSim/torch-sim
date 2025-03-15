from typing import Any

import torch

from torch_sim.state import BaseState
from torch_sim.unbatched_optimizers import fire, gradient_descent, unit_cell_fire


def test_fire_optimizer(ar_base_state: BaseState, unbatched_lj_calculator: Any) -> None:
    """Test FIRE optimization of Ar FCC structure."""
    # perturb the structure
    ar_base_state.positions[1][0] = 1.0
    ar_base_state.cell = ar_base_state.cell.squeeze(0)

    # Initialize optimizer
    init_fn, update_fn = fire(
        model=unbatched_lj_calculator,
        dt_start=0.01,
        dt_max=0.1,
        n_min=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha_start=0.1,
    )

    state = init_fn(state=ar_base_state)

    # Run a few optimization steps
    initial_force_norm = torch.norm(state.forces)
    for _step in range(10):
        state = update_fn(state)

    # Check that forces are being minimized
    assert torch.norm(state.forces) < initial_force_norm, (
        "Forces should decrease during optimization"
    )

    # Check state properties
    assert hasattr(state, "dt"), "FIRE state should have timestep"
    assert hasattr(state, "alpha"), "FIRE state should have mixing parameter"
    assert hasattr(state, "n_pos"), "FIRE state should have step counter"
    assert state.positions.shape == ar_base_state.positions.shape


def test_gradient_descent_optimizer(
    ar_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test gradient descent optimization of Ar FCC structure."""
    # perturb the structure
    ar_base_state.positions[1][0] = 1.0
    ar_base_state.cell = ar_base_state.cell.squeeze(0)
    # Initialize optimizer
    init_fn, update_fn = gradient_descent(
        model=unbatched_lj_calculator,
        lr=0.01,
    )

    state = init_fn(state=ar_base_state)

    # Run a few optimization steps
    initial_energy = state.energy
    for _step in range(10):
        state = update_fn(state)

    # Check that energy is decreasing
    assert state.energy < initial_energy, "Energy should decrease during optimization"

    # Check state properties
    assert state.positions.shape == ar_base_state.positions.shape


def test_unit_cell_fire_optimizer(
    ar_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test FIRE optimization of Ar FCC structure."""
    # perturb the structure
    ar_base_state.positions[1][0] = 1.0
    ar_base_state.cell = ar_base_state.cell.squeeze(0)

    # Initialize optimizer
    init_fn, update_fn = unit_cell_fire(
        model=unbatched_lj_calculator,
        dt_start=0.01,
        dt_max=0.1,
        n_min=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha_start=0.1,
    )

    state = init_fn(state=ar_base_state)

    # Run a few optimization steps
    initial_force_norm = torch.norm(state.forces)
    initial_pressure = torch.trace(state.stress) / 3.0
    for _step in range(10):
        state = update_fn(state)

    # Check that forces are being minimized
    assert torch.norm(state.forces) < initial_force_norm, (
        "Forces should decrease during optimization"
    )

    # Check that pressure is being minimized
    assert torch.abs(torch.trace(state.stress) / 3.0 - initial_pressure) < 0.01, (
        "Pressure should decrease during optimization"
    )

    # Check state properties
    assert hasattr(state, "dt"), "FIRE state should have timestep"
    assert hasattr(state, "alpha"), "FIRE state should have mixing parameter"
    assert hasattr(state, "n_pos"), "FIRE state should have step counter"
    assert state.positions.shape == ar_base_state.positions.shape


def test_optimizer_convergence(
    ar_base_state: BaseState, unbatched_lj_calculator: Any
) -> None:
    """Test that both optimizers can reach similar final states."""
    # perturb the structure
    ar_base_state.positions[1][0] = 1.0
    ar_base_state.cell = ar_base_state.cell.squeeze(0)
    # Run FIRE
    fire_init, fire_update = fire(
        model=unbatched_lj_calculator,
        dt_start=0.01,
    )

    # Run GD
    gd_init, gd_update = gradient_descent(
        model=unbatched_lj_calculator,
        lr=0.01,
    )

    fire_state = fire_init(state=ar_base_state)
    gd_state = gd_init(state=ar_base_state)

    # Optimize both for more steps
    for _step in range(50):
        fire_state = fire_update(fire_state)
        gd_state = gd_update(gd_state)

    # Check that both methods reach similar energies
    assert torch.allclose(
        fire_state.energy,
        gd_state.energy,
        rtol=1e-1,  # 10% tolerance
    ), "Both optimizers should converge to similar energies"
