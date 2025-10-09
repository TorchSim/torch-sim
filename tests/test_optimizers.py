import copy
from collections.abc import Callable
from dataclasses import fields
from functools import partial
from typing import Any, get_args

import pytest
import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import FireState, MdFlavor, OptimState
from torch_sim.state import SimState


def test_gradient_descent_optimization(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface
) -> None:
    """Test that the Gradient Descent optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state

    # Initialize Gradient Descent optimizer
    state = ts.gradient_descent_init(
        state=ar_supercell_sim_state, model=lj_model, lr=0.01
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = ts.gradient_descent_step(state=state, model=lj_model, pos_lr=0.01)
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Gradient Descent optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert max_force < 0.2, f"Forces should be small after optimization, got {max_force=}"

    assert not torch.allclose(state.positions, initial_state.positions)


def test_unit_cell_gradient_descent_optimization(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface
) -> None:
    """Test that the Gradient Descent optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_supercell_sim_state.positions
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    ar_supercell_sim_state.positions = perturbed_positions
    initial_state = ar_supercell_sim_state

    # Initialize Gradient Descent optimizer with unit cell filter
    state = ts.gradient_descent_init(
        state=ar_supercell_sim_state, model=lj_model, cell_filter=ts.CellFilter.unit
    )

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = ts.gradient_descent_step(
            state=state, model=lj_model, pos_lr=0.01, cell_lr=0.1
        )
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Gradient Descent optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization, got {pressure=}"
    )
    assert max_force < 0.2, f"Forces should be small after optimization, got {max_force=}"

    assert not torch.allclose(state.positions, initial_state.positions)
    assert not torch.allclose(state.cell, initial_state.cell)


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface, md_flavor: MdFlavor
) -> None:
    """Test that the FIRE optimizer actually minimizes energy."""
    # Add some random displacement to positions
    # Create a fresh copy for each test run to avoid interference

    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )

    current_sim_state = SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=ar_supercell_sim_state.cell.clone(),
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )

    initial_state_positions = current_sim_state.positions.clone()

    # Initialize FIRE optimizer
    state = ts.fire_init(current_sim_state, lj_model, md_flavor=md_flavor, dt_start=0.1)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    assert steps_taken < max_steps, (
        f"FIRE optimization for {md_flavor=} did not converge in {max_steps=}"
    )

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"FIRE optimization for {md_flavor=} should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    # bumped up the tolerance to 0.3 to account for the fact that ase_fire is more lenient
    # in beginning steps
    assert max_force < 0.3, (
        f"{md_flavor=} forces should be small after optimization, got {max_force=}"
    )

    assert not torch.allclose(state.positions, initial_state_positions), (
        f"{md_flavor=} positions should have changed after optimization."
    )


@pytest.mark.parametrize(
    ("optimizer_fn", "expected_state_type"),
    [
        (ts.OptimFlavor.fire, FireState),
        (ts.OptimFlavor.gradient_descent, OptimState),
    ],
)
def test_simple_optimizer_init_with_dict(
    optimizer_fn: ts.OptimFlavor,
    expected_state_type: FireState | OptimState,
    ar_supercell_sim_state: SimState,
    lj_model: ModelInterface,
) -> None:
    """Test simple optimizer init_fn with a SimState dictionary."""
    state_dict = {
        field.name: getattr(ar_supercell_sim_state, field.name)
        for field in fields(ar_supercell_sim_state)
    }
    init_fn, _ = ts.OPTIM_REGISTRY[optimizer_fn]
    opt_state = init_fn(model=lj_model, state=state_dict)
    assert isinstance(opt_state, expected_state_type)
    assert opt_state.energy is not None
    assert opt_state.forces is not None


@pytest.mark.parametrize(
    "optim_func",
    [ts.fire_init, partial(ts.fire_init, cell_filter=ts.CellFilter.unit)],
)
def test_optimizer_invalid_md_flavor(
    optim_func: Callable[..., Any],
    lj_model: ModelInterface,
    ar_supercell_sim_state: SimState,
) -> None:
    """Test optimizer with an invalid md_flavor raises ValueError."""
    with pytest.raises(ValueError, match="Unknown md_flavor"):
        optim_func(
            model=lj_model, state=ar_supercell_sim_state, md_flavor="invalid_flavor"
        )


def test_fire_ase_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface
) -> None:
    """Test that the ASE FIRE P<0 branch behaves as expected."""
    f_dec = 0.5  # Default from fire optimizer
    alpha_start = 0.1  # Default from fire optimizer
    dt_start_val = 0.1

    state = ts.fire_init(
        state=ar_supercell_sim_state,
        model=lj_model,
        md_flavor="ase_fire",
        alpha_start=alpha_start,
        dt_start=dt_start_val,
    )

    # Save parameters from initial state
    initial_dt_batch = state.dt.clone()  # per-system dt

    # Manipulate state to ensure P < 0 for the step_fn
    # Ensure forces are non-trivial
    state.forces += torch.sign(state.forces + 1e-6) * 1e-2
    state.forces[torch.abs(state.forces) < 1e-3] = 1e-3
    # Set velocities directly opposite to current forces
    state.velocities = -state.forces * 0.1  # v = -k * F

    # Store forces that will be used in the power calculation and v += dt*F step
    forces_at_power_calc = state.forces.clone()

    # Deepcopy state as step_fn modifies it in-place
    state_to_update = copy.deepcopy(state)
    updated_state = ts.fire_step(
        state=state_to_update,
        model=lj_model,
        f_dec=f_dec,
        dt_max=1.0,
        max_step=10.0,  # Large max_step to not interfere with velocity check
    )

    # Assertions for P < 0 branch being taken
    # Check for a single-batch state (ar_supercell_sim_state is single batch)
    expected_dt_val = initial_dt_batch[0] * f_dec
    assert torch.allclose(updated_state.dt[0], expected_dt_val)
    assert torch.allclose(
        updated_state.alpha[0],
        torch.tensor(
            alpha_start,
            dtype=updated_state.alpha.dtype,
            device=updated_state.alpha.device,
        ),
    )
    assert updated_state.n_pos[0] == 0

    # Assertions for velocity update in ASE P < 0 case:
    # v_after_mixing_is_0, then v_final = dt_new * F_at_power_calc
    expected_final_velocities = (
        expected_dt_val * forces_at_power_calc[updated_state.system_idx == 0]
    )
    assert torch.allclose(
        updated_state.velocities[updated_state.system_idx == 0],
        expected_final_velocities,
        atol=1e-6,
    )


def test_fire_vv_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface
) -> None:
    """Attempt to trigger and test the VV FIRE P<0 branch."""
    f_dec = 0.5
    alpha_start = 0.1
    # Use a very large dt_start to encourage overshooting and P<0 inside _vv_fire_step
    dt_start_val = 2.0
    dt_max_val = 2.0

    state = ts.fire_init(
        state=ar_supercell_sim_state,
        model=lj_model,
        md_flavor="vv_fire",
        alpha_start=alpha_start,
        dt_start=dt_start_val,
    )

    initial_dt_batch = state.dt.clone()
    initial_alpha_batch = state.alpha.clone()  # Already alpha_start

    state_to_update = copy.deepcopy(state)
    updated_state = ts.fire_step(
        state=state_to_update,
        model=lj_model,
        f_dec=f_dec,
        dt_max=dt_max_val,
        n_min=0,  # Allow dt to change immediately
    )

    # Check if the P<0 branch was likely hit (params changed accordingly for batch 0)
    expected_dt_val = initial_dt_batch[0] * f_dec
    expected_alpha_val = torch.tensor(
        alpha_start,
        dtype=initial_alpha_batch.dtype,
        device=initial_alpha_batch.device,
    )

    p_lt_0_branch_taken = (
        torch.allclose(updated_state.dt[0], expected_dt_val)
        and torch.allclose(updated_state.alpha[0], expected_alpha_val)
        and updated_state.n_pos[0] == 0
    )

    if not p_lt_0_branch_taken:
        return

    # If P<0 branch was taken, velocities should be zeroed
    assert torch.allclose(
        updated_state.velocities[updated_state.system_idx == 0],
        torch.zeros_like(updated_state.velocities[updated_state.system_idx == 0]),
        atol=1e-7,
    )


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_unit_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface, md_flavor: MdFlavor
) -> None:
    """Test that the Unit Cell FIRE optimizer actually minimizes energy."""

    # Add random displacement to positions and cell
    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )
    current_cell = (
        ar_supercell_sim_state.cell.clone()
        + torch.randn_like(ar_supercell_sim_state.cell) * 0.01
    )

    current_sim_state = SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=current_cell,
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    # Initialize FIRE optimizer with unit cell filter
    state = ts.fire_init(
        state=current_sim_state,
        model=lj_model,
        dt_start=0.1,
        md_flavor=md_flavor,
        cell_filter=ts.CellFilter.unit,
    )

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]
    max_steps = 1000
    steps_taken = 0

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    assert steps_taken < max_steps, (
        f"Unit Cell FIRE {md_flavor=} optimization did not converge in {max_steps=}"
    )

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Unit Cell FIRE {md_flavor=} optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization, got {pressure=}"
    )
    assert max_force < 0.3, (
        f"{md_flavor=} forces should be small after optimization, got {max_force}"
    )

    assert not torch.allclose(state.positions, initial_state_positions), (
        f"{md_flavor=} positions should have changed after optimization."
    )
    assert not torch.allclose(state.cell, initial_state_cell), (
        f"{md_flavor=} cell should have changed after optimization."
    )


@pytest.mark.parametrize(
    ("optimizer_fn", "cell_filter", "expected_state_type", "cell_factor_val"),
    [
        (ts.OptimFlavor.fire, ts.CellFilter.unit, ts.CellFireState, 100),
        (
            ts.OptimFlavor.gradient_descent,
            ts.CellFilter.unit,
            ts.CellOptimState,
            50.0,
        ),
        (ts.OptimFlavor.fire, ts.CellFilter.frechet, ts.CellFireState, 75.0),
    ],
)
def test_cell_optimizer_init_with_dict_and_cell_factor(
    optimizer_fn: ts.OptimFlavor,
    expected_state_type: OptimState,
    cell_filter: ts.CellFilter,
    cell_factor_val: float,
    ar_supercell_sim_state: SimState,
    lj_model: ModelInterface,
) -> None:
    """Test cell optimizer init_fn with dict state and explicit cell_factor."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    init_fn, _ = ts.OPTIM_REGISTRY[optimizer_fn]
    opt_state = init_fn(
        model=lj_model,
        state=state_dict,
        cell_factor=cell_factor_val,
        cell_filter=cell_filter,
    )

    assert isinstance(opt_state, expected_state_type)
    assert opt_state.energy is not None
    assert opt_state.forces is not None
    assert opt_state.stress is not None
    # Check cell_factor is stored in cell_state
    expected_cf_tensor = torch.full(
        (opt_state.n_systems, 1, 1),
        float(cell_factor_val),  # Ensure float for comparison if int is passed
        device=lj_model.device,
        dtype=lj_model.dtype,
    )
    assert torch.allclose(opt_state.cell_factor, expected_cf_tensor)


@pytest.mark.parametrize(
    ("optimizer_fn", "cell_filter", "expected_state_type"),
    [
        (ts.OptimFlavor.fire, ts.CellFilter.unit, ts.CellFireState),
        (ts.OptimFlavor.fire, ts.CellFilter.frechet, ts.CellFireState),
        (
            ts.OptimFlavor.gradient_descent,
            ts.CellFilter.unit,
            ts.CellOptimState,
        ),
        (
            ts.OptimFlavor.gradient_descent,
            ts.CellFilter.frechet,
            ts.CellOptimState,
        ),
    ],
)
def test_cell_optimizer_init_cell_factor_none(
    optimizer_fn: ts.OptimFlavor,
    cell_filter: ts.CellFilter,
    expected_state_type: OptimState,
    ar_supercell_sim_state: SimState,
    lj_model: ModelInterface,
) -> None:
    """Test cell optimizer init_fn with cell_factor=None."""
    init_fn, _ = ts.OPTIM_REGISTRY[optimizer_fn]
    opt_state = init_fn(
        model=lj_model,
        state=ar_supercell_sim_state,
        cell_factor=None,
        cell_filter=cell_filter,
    )
    # Ensure n_systems > 0 for cell_factor calculation from counts
    assert ar_supercell_sim_state.n_systems > 0
    assert isinstance(opt_state, expected_state_type)
    _, counts = torch.unique(ar_supercell_sim_state.system_idx, return_counts=True)
    expected_cf_tensor = counts.to(dtype=lj_model.dtype).view(-1, 1, 1)

    # Check cell_factor is stored in cell_state for new API
    assert torch.allclose(opt_state.cell_factor, expected_cf_tensor)

    assert opt_state.energy is not None
    assert opt_state.forces is not None
    assert opt_state.stress is not None


@pytest.mark.filterwarnings("ignore:WARNING: Non-positive volume detected")
def test_unit_cell_fire_ase_non_positive_volume_warning(
    ar_supercell_sim_state: SimState,
    lj_model: ModelInterface,
    capsys: pytest.CaptureFixture,
) -> None:
    """Attempt to trigger non-positive volume warning in ASE unit cell fire."""
    # Use a state that might lead to cell inversion with aggressive steps
    # Make a copy and slightly perturb the cell to make it prone to issues
    perturbed_state = ar_supercell_sim_state.clone()
    perturbed_state.cell += (
        torch.randn_like(perturbed_state.cell) * 0.5
    )  # Large perturbation
    # Also ensure no PBC issues by slightly expanding cell if it got too small
    if torch.linalg.det(perturbed_state.cell[0]) < 1.0:
        perturbed_state.cell[0] *= 2.0

    state = ts.fire_init(
        state=perturbed_state,
        model=lj_model,
        md_flavor="ase_fire",
        dt_start=1.0,
        alpha_start=0.99,  # Aggressive alpha
        cell_filter=ts.CellFilter.unit,
    )

    # Run a few steps hoping to trigger the warning
    for _ in range(5):
        state = ts.fire_step(
            state=state,
            model=lj_model,
            dt_max=5.0,  # Large dt
            max_step=2.0,  # Large max_step
            f_dec=0.99,  # Slow down dt decrease
        )
        if "WARNING: Non-positive volume detected" in capsys.readouterr().err:
            break  # Warning captured

    assert state is not None  # Ensure optimizer ran


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_frechet_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface, md_flavor: MdFlavor
) -> None:
    """Test that the Frechet Cell FIRE optimizer actually minimizes energy for different
    md_flavors."""

    # Add random displacement to positions and cell
    # Create a fresh copy for each test run to avoid interference
    current_positions = (
        ar_supercell_sim_state.positions.clone()
        + torch.randn_like(ar_supercell_sim_state.positions) * 0.1
    )
    current_cell = (
        ar_supercell_sim_state.cell.clone()
        + torch.randn_like(ar_supercell_sim_state.cell) * 0.01
    )

    current_sim_state = SimState(
        positions=current_positions,
        masses=ar_supercell_sim_state.masses.clone(),
        cell=current_cell,
        pbc=ar_supercell_sim_state.pbc,
        atomic_numbers=ar_supercell_sim_state.atomic_numbers.clone(),
        system_idx=ar_supercell_sim_state.system_idx.clone(),
    )

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    state = ts.fire_init(
        state=current_sim_state,
        model=lj_model,
        dt_start=0.1,
        md_flavor=md_flavor,
        cell_filter=ts.CellFilter.frechet,
    )

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]  # Ensure float for comparison
    max_steps = 1000
    steps_taken = 0

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        energies.append(state.energy.item())
        steps_taken += 1

    assert steps_taken < max_steps, (
        f"Frechet FIRE {md_flavor=} optimization did not converge in {max_steps=}"
    )

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"Frechet FIRE {md_flavor=} optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    # Assumes single batch for this state stress access
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0

    # Adjust tolerances if needed, Frechet might behave slightly differently
    pressure_tol, force_tol = 0.01, 0.2

    assert torch.abs(pressure) < pressure_tol, (
        f"{md_flavor=} pressure should be below {pressure_tol=} after Frechet "
        f"optimization, got {pressure.item()}"
    )
    assert max_force < force_tol, (
        f"{md_flavor=} forces should be below {force_tol=} after Frechet optimization, "
        f"got {max_force}"
    )

    assert not torch.allclose(state.positions, initial_state_positions, atol=1e-5), (
        f"{md_flavor=} positions should have changed after Frechet optimization."
    )
    assert not torch.allclose(state.cell, initial_state_cell, atol=1e-5), (
        f"{md_flavor=} cell should have changed after Frechet optimization."
    )


@pytest.mark.parametrize(
    "filter_func",
    [None, ts.CellFilter.unit, ts.CellFilter.frechet],
)
def test_optimizer_batch_consistency(
    filter_func: ts.CellFilter | None,
    ar_supercell_sim_state: SimState,
    lj_model: ModelInterface,
) -> None:
    """Test batched optimizer is consistent with individual optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    # Create two distinct initial states by cloning and perturbing
    state1_orig = ar_supercell_sim_state.clone()

    # Apply identical perturbations to state1_orig
    # for state_item in [state1_orig, state2_orig]: # Old loop structure
    generator.manual_seed(43)  # Reset seed for positions
    state1_orig.positions += (
        torch.randn(
            state1_orig.positions.shape, device=state1_orig.device, generator=generator
        )
        * 0.1
    )
    if filter_func:
        generator.manual_seed(44)  # Reset seed for cell
        state1_orig.cell += (
            torch.randn(
                state1_orig.cell.shape, device=state1_orig.device, generator=generator
            )
            * 0.01
        )

    # Ensure state2_orig is identical to perturbed state1_orig
    state2_orig = state1_orig.clone()

    final_individual_states = []

    def energy_converged(e_current: torch.Tensor, e_prev: torch.Tensor) -> bool:
        """Check for energy convergence (scalar energies)."""
        return not torch.allclose(e_current, e_prev, atol=1e-6)

    for state_for_indiv_opt in [state1_orig.clone(), state2_orig.clone()]:
        init_fn_indiv, step_fn_indiv = ts.OPTIM_REGISTRY[ts.OptimFlavor.fire]
        opt_state_indiv = init_fn_indiv(
            model=lj_model,
            state=state_for_indiv_opt,
            dt_start=0.1,
            cell_filter=filter_func,
        )

        current_e_indiv = opt_state_indiv.energy
        # Ensure prev_e_indiv is different to start the loop
        e_prev_indiv = current_e_indiv + torch.tensor(
            1.0, device=current_e_indiv.device, dtype=current_e_indiv.dtype
        )

        steps_indiv = 0
        while energy_converged(current_e_indiv, e_prev_indiv):
            e_prev_indiv = current_e_indiv
            opt_state_indiv = step_fn_indiv(
                model=lj_model, state=opt_state_indiv, dt_max=0.3
            )
            current_e_indiv = opt_state_indiv.energy
            steps_indiv += 1
            if steps_indiv > 1000:
                raise ValueError(
                    f"Individual opt for {filter_func.name} did not converge"
                )
        final_individual_states.append(opt_state_indiv)

    # Batched optimization
    multi_state_initial = ts.concatenate_states(
        [state1_orig.clone(), state2_orig.clone()],
        device=ar_supercell_sim_state.device,
    )

    init_fn_batch, step_fn_batch = ts.OPTIM_REGISTRY[ts.OptimFlavor.fire]
    batch_opt_state = init_fn_batch(
        model=lj_model, state=multi_state_initial, cell_filter=filter_func
    )

    e_current_batch = batch_opt_state.energy.clone()
    # Ensure e_prev_batch requires update and has same shape
    e_prev_batch = e_current_batch + torch.tensor(
        1.0, device=e_current_batch.device, dtype=e_current_batch.dtype
    )

    steps_batch = 0
    # Converge when all batch energies have converged
    while not torch.allclose(e_current_batch, e_prev_batch, atol=1e-6):
        e_prev_batch = e_current_batch.clone()
        batch_opt_state = step_fn_batch(model=lj_model, state=batch_opt_state)
        e_current_batch = batch_opt_state.energy.clone()
        steps_batch += 1
        if steps_batch > 1000:
            raise ValueError(f"Batched opt for {filter_func.name} did not converge")

    individual_final_energies = [s.energy.item() for s in final_individual_states]
    for idx, indiv_energy in enumerate(individual_final_energies):
        assert abs(e_current_batch[idx].item() - indiv_energy) < 1e-4, (
            f"Energy batch {idx} ({filter_func=}): "
            f"{e_current_batch[idx].item()} vs indiv {indiv_energy}"
        )

    # Check positions changed for both parts of the batch
    n_atoms_first_state = state1_orig.positions.shape[0]
    assert not torch.allclose(
        batch_opt_state.positions[:n_atoms_first_state],
        multi_state_initial.positions[:n_atoms_first_state],
        atol=1e-5,  # Added tolerance as in original frechet test
    ), f"{filter_func=} positions batch 0 did not change."
    assert not torch.allclose(
        batch_opt_state.positions[n_atoms_first_state:],
        multi_state_initial.positions[n_atoms_first_state:],
        atol=1e-5,
    ), f"{filter_func=} positions batch 1 did not change."

    if filter_func:
        assert not torch.allclose(
            batch_opt_state.cell, multi_state_initial.cell, atol=1e-5
        ), f"{filter_func.name} cell did not change."


def test_unit_cell_fire_multi_batch(
    ar_supercell_sim_state: SimState, lj_model: ModelInterface
) -> None:
    """Test FIRE optimization with multiple batches."""
    # Create a multi-batch system by duplicating ar_fcc_state

    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    for state in (ar_supercell_sim_state_1, ar_supercell_sim_state_2):
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    multi_state = ts.concatenate_states(
        [ar_supercell_sim_state_1, ar_supercell_sim_state_2],
        device=ar_supercell_sim_state.device,
    )

    # Initialize FIRE optimizer with unit cell filter
    state = ts.fire_init(
        state=multi_state, model=lj_model, dt_start=0.1, cell_filter=ts.CellFilter.unit
    )
    initial_state = copy.deepcopy(state)

    # Run optimization for a few steps
    prev_energy = torch.ones(2, device=state.device, dtype=state.energy.dtype) * 1000
    current_energy = initial_state.energy
    step = 0
    while not torch.allclose(current_energy, prev_energy, atol=1e-9):
        prev_energy = current_energy
        state = ts.fire_step(state=state, model=lj_model, dt_max=0.3)
        current_energy = state.energy

        step += 1
        if step > 500:
            raise ValueError("Optimization did not converge")

    # check that we actually optimized
    assert step > 10

    # Check that energy decreased for both batches
    assert torch.all(state.energy < initial_state.energy), (
        "FIRE optimization should reduce energy for all batches"
    )

    # transfer the energy and force checks to the batched optimizer
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert torch.all(max_force < 0.1), (
        f"Forces should be small after optimization, got {max_force=}"
    )

    n_ar_atoms = ar_supercell_sim_state.n_atoms
    assert not torch.allclose(
        state.positions[:n_ar_atoms], multi_state.positions[:n_ar_atoms]
    )
    assert not torch.allclose(
        state.positions[n_ar_atoms:], multi_state.positions[n_ar_atoms:]
    )

    # we are evolving identical systems
    assert current_energy[0] == current_energy[1]


def test_fire_fixed_cell_unit_cell_consistency(  # noqa: C901
    ar_supercell_sim_state: SimState, lj_model: ModelInterface
) -> None:
    """Test batched Frechet Fixed cell FIRE optimization is
    consistent with FIRE (position only) optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in (ar_supercell_sim_state_1, ar_supercell_sim_state_2):
        generator.manual_seed(43)
        state.positions += (
            torch.randn(state.positions.shape, device=state.device, generator=generator)
            * 0.1
        )

    # Optimize each state individually
    final_individual_states_unit_cell = []
    total_steps_unit_cell = []

    def energy_converged(current_energy: torch.Tensor, prev_energy: torch.Tensor) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in (ar_supercell_sim_state_1, ar_supercell_sim_state_2):
        state_opt = ts.fire_init(
            state,
            lj_model,
            dt_start=0.1,
            cell_filter=ts.CellFilter.unit,
            hydrostatic_strain=True,
            constant_volume=True,
        )

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = ts.fire_step(state=state_opt, model=lj_model, dt_max=0.3)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states_unit_cell.append(state_opt)
        total_steps_unit_cell.append(step)

    # Optimize each state individually
    final_individual_states_fire = []
    total_steps_fire = []

    def energy_converged(current_energy: torch.Tensor, prev_energy: torch.Tensor) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in (ar_supercell_sim_state_1, ar_supercell_sim_state_2):
        state_opt = ts.fire_init(state=state, model=lj_model, dt_start=0.1)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = ts.fire_step(state=state_opt, model=lj_model, dt_max=0.3)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError(f"Optimization did not converge in {step=}")

        final_individual_states_fire.append(state_opt)
        total_steps_fire.append(step)

    individual_energies_unit_cell = [
        state.energy.item() for state in final_individual_states_unit_cell
    ]
    individual_energies_fire = [
        state.energy.item() for state in final_individual_states_fire
    ]
    # Check that final energies from fixed cell optimization match
    # position only optimizations
    for step, energy_unit_cell in enumerate(individual_energies_unit_cell):
        assert abs(energy_unit_cell - individual_energies_fire[step]) < 1e-4, (
            f"Energy for system {step} doesn't match position only optimization: "
            f"system={energy_unit_cell}, individual={individual_energies_fire[step]}"
        )
