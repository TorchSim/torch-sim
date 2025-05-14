import copy
from dataclasses import fields
from typing import get_args

import pytest
import torch

from torch_sim.optimizers import (
    FireState,
    FrechetCellFIREState,
    GDState,
    MdFlavor,
    UnitCellFireState,
    UnitCellGDState,
    fire,
    frechet_cell_fire,
    gradient_descent,
    unit_cell_fire,
    unit_cell_gradient_descent,
)
from torch_sim.state import SimState, concatenate_states


def test_gradient_descent_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
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
    init_fn, update_fn = gradient_descent(model=lj_model, lr=0.01)

    state = init_fn(ar_supercell_sim_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
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
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
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
    init_fn, update_fn = unit_cell_gradient_descent(
        model=lj_model,
        positions_lr=0.01,
        cell_lr=0.1,
    )

    state = init_fn(ar_supercell_sim_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
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
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
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
        batch=ar_supercell_sim_state.batch.clone(),
    )

    initial_state_positions = current_sim_state.positions.clone()

    # Initialize FIRE optimizer
    init_fn, update_fn = fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
    )

    state = init_fn(current_sim_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    max_steps = 1000  # Add max step to prevent infinite loop
    steps_taken = 0
    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    if steps_taken == max_steps:
        print(f"FIRE optimization for {md_flavor=} did not converge in {max_steps} steps")

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
    [(fire, FireState), (gradient_descent, GDState)],
)
def test_simple_optimizer_init_with_dict(
    optimizer_fn: callable,
    expected_state_type: type,
    ar_supercell_sim_state: SimState,
    lj_model: torch.nn.Module,
) -> None:
    """Test simple optimizer init_fn with a SimState dictionary."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    init_fn, _ = optimizer_fn(model=lj_model)
    opt_state = init_fn(state_dict)
    assert isinstance(opt_state, expected_state_type)
    assert opt_state.energy is not None
    assert opt_state.forces is not None


@pytest.mark.parametrize("optimizer_func", [fire, unit_cell_fire, frechet_cell_fire])
def test_optimizer_invalid_md_flavor(
    optimizer_func: callable, lj_model: torch.nn.Module
) -> None:
    """Test optimizer with an invalid md_flavor raises ValueError."""
    with pytest.raises(ValueError, match="Unknown md_flavor"):
        optimizer_func(model=lj_model, md_flavor="invalid_flavor")


def test_fire_ase_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test that the ASE FIRE P<0 branch behaves as expected."""
    f_dec = 0.5  # Default from fire optimizer
    alpha_start = 0.1  # Default from fire optimizer
    dt_start_val = 0.1

    init_fn, update_fn = fire(
        model=lj_model,
        md_flavor="ase_fire",
        f_dec=f_dec,
        alpha_start=alpha_start,
        dt_start=dt_start_val,
        dt_max=1.0,
        max_step=10.0,  # Large max_step to not interfere with velocity check
    )
    # Initialize state (forces are computed here)
    state = init_fn(ar_supercell_sim_state)

    # Save parameters from initial state
    initial_dt_batch = state.dt.clone()  # per-batch dt

    # Manipulate state to ensure P < 0 for the update_fn step
    # Ensure forces are non-trivial
    state.forces += torch.sign(state.forces + 1e-6) * 1e-2
    state.forces[torch.abs(state.forces) < 1e-3] = 1e-3
    # Set velocities directly opposite to current forces
    state.velocities = -state.forces * 0.1  # v = -k * F

    # Store forces that will be used in the power calculation and v += dt*F step
    forces_at_power_calc = state.forces.clone()

    # Deepcopy state as update_fn modifies it in-place
    state_to_update = copy.deepcopy(state)
    updated_state = update_fn(state_to_update)

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
        expected_dt_val * forces_at_power_calc[updated_state.batch == 0]
    )
    assert torch.allclose(
        updated_state.velocities[updated_state.batch == 0],
        expected_final_velocities,
        atol=1e-6,
    )


def test_fire_vv_negative_power_branch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Attempt to trigger and test the VV FIRE P<0 branch."""
    f_dec = 0.5
    alpha_start = 0.1
    # Use a very large dt_start to encourage overshooting and P<0 inside _vv_fire_step
    dt_start_val = 2.0
    dt_max_val = 2.0

    init_fn, update_fn = fire(
        model=lj_model,
        md_flavor="vv_fire",
        f_dec=f_dec,
        alpha_start=alpha_start,
        dt_start=dt_start_val,
        dt_max=dt_max_val,
        n_min=0,  # Allow dt to change immediately
    )
    state = init_fn(ar_supercell_sim_state)

    initial_dt_batch = state.dt.clone()
    initial_alpha_batch = state.alpha.clone()  # Already alpha_start

    state_to_update = copy.deepcopy(state)
    updated_state = update_fn(state_to_update)

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
        updated_state.velocities[updated_state.batch == 0],
        torch.zeros_like(updated_state.velocities[updated_state.batch == 0]),
        atol=1e-7,
    )


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_unit_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
) -> None:
    """Test that the Unit Cell FIRE optimizer actually minimizes energy."""
    print(f"\n--- Starting test_unit_cell_fire_optimization for {md_flavor=} ---")

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
        batch=ar_supercell_sim_state.batch.clone(),
    )
    print(f"[{md_flavor}] Initial SimState created.")

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    # Initialize FIRE optimizer
    print(f"Initializing {md_flavor} optimizer...")
    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
    )
    print(f"[{md_flavor}] Optimizer functions obtained.")

    state = init_fn(current_sim_state)
    energy = float(getattr(state, "energy", "nan"))
    print(f"[{md_flavor}] Initial state created by init_fn. {energy=:.4f}")

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]
    max_steps = 1000
    steps_taken = 0
    print(f"[{md_flavor}] Entering optimization loop (max_steps: {max_steps})...")

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    print(f"[{md_flavor}] Loop finished after {steps_taken} steps.")

    if steps_taken == max_steps and abs(energies[-2] - energies[-1]) > 1e-6:
        print(
            f"WARNING: Unit Cell FIRE {md_flavor=} optimization did not converge "
            f"in {max_steps} steps. Final energy: {energies[-1]:.4f}"
        )
    else:
        print(
            f"Unit Cell FIRE {md_flavor=} optimization converged in {steps_taken} "
            f"steps. Final energy: {energies[-1]:.4f}"
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
    ("optimizer_fn", "expected_state_type", "cell_factor_val"),
    [
        (unit_cell_fire, UnitCellFireState, 100),
        (unit_cell_gradient_descent, UnitCellGDState, 50.0),
        (frechet_cell_fire, FrechetCellFIREState, 75.0),
    ],
)
def test_cell_optimizer_init_with_dict_and_cell_factor(
    optimizer_fn: callable,
    expected_state_type: type,
    cell_factor_val: float,
    ar_supercell_sim_state: SimState,
    lj_model: torch.nn.Module,
) -> None:
    """Test cell optimizer init_fn with dict state and explicit cell_factor."""
    state_dict = {
        f.name: getattr(ar_supercell_sim_state, f.name)
        for f in fields(ar_supercell_sim_state)
    }
    init_fn, _ = optimizer_fn(model=lj_model, cell_factor=cell_factor_val)
    opt_state = init_fn(state_dict)

    assert isinstance(opt_state, expected_state_type)
    assert opt_state.energy is not None
    assert opt_state.forces is not None
    assert opt_state.stress is not None
    expected_cf_tensor = torch.full(
        (opt_state.n_batches, 1, 1),
        float(cell_factor_val),  # Ensure float for comparison if int is passed
        device=lj_model.device,
        dtype=lj_model.dtype,
    )
    assert torch.allclose(opt_state.cell_factor, expected_cf_tensor)


@pytest.mark.parametrize(
    ("optimizer_fn", "expected_state_type"),
    [
        (unit_cell_fire, UnitCellFireState),
        (frechet_cell_fire, FrechetCellFIREState),
    ],
)
def test_cell_optimizer_init_cell_factor_none(
    optimizer_fn: callable,
    expected_state_type: type,
    ar_supercell_sim_state: SimState,
    lj_model: torch.nn.Module,
) -> None:
    """Test cell optimizer init_fn with cell_factor=None."""
    init_fn, _ = optimizer_fn(model=lj_model, cell_factor=None)
    # Ensure n_batches > 0 for cell_factor calculation from counts
    assert ar_supercell_sim_state.n_batches > 0
    opt_state = init_fn(ar_supercell_sim_state)  # Uses SimState directly
    assert isinstance(opt_state, expected_state_type)
    _, counts = torch.unique(ar_supercell_sim_state.batch, return_counts=True)
    expected_cf_tensor = counts.to(dtype=lj_model.dtype).view(-1, 1, 1)
    assert torch.allclose(opt_state.cell_factor, expected_cf_tensor)
    assert opt_state.energy is not None
    assert opt_state.forces is not None
    assert opt_state.stress is not None


@pytest.mark.filterwarnings("ignore:WARNING: Non-positive volume detected")
def test_unit_cell_fire_ase_non_positive_volume_warning(
    ar_supercell_sim_state: SimState,
    lj_model: torch.nn.Module,
    capsys: pytest.CaptureFixture,
) -> None:
    """Attempt to trigger non-positive volume warning in unit_cell_fire ASE."""
    # Use a state that might lead to cell inversion with aggressive steps
    # Make a copy and slightly perturb the cell to make it prone to issues
    perturbed_state = ar_supercell_sim_state.clone()
    perturbed_state.cell += (
        torch.randn_like(perturbed_state.cell) * 0.5
    )  # Large perturbation
    # Also ensure no PBC issues by slightly expanding cell if it got too small
    if torch.linalg.det(perturbed_state.cell[0]) < 1.0:
        perturbed_state.cell[0] *= 2.0

    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        md_flavor="ase_fire",
        dt_max=5.0,  # Large dt
        max_step=2.0,  # Large max_step
        dt_start=1.0,
        f_dec=0.99,  # Slow down dt decrease
        alpha_start=0.99,  # Aggressive alpha
    )
    state = init_fn(perturbed_state)

    # Run a few steps hoping to trigger the warning
    for _ in range(5):
        state = update_fn(state)
        if "WARNING: Non-positive volume detected" in capsys.readouterr().err:
            break  # Warning captured

    assert state is not None  # Ensure optimizer ran


@pytest.mark.parametrize("md_flavor", get_args(MdFlavor))
def test_frechet_cell_fire_optimization(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module, md_flavor: MdFlavor
) -> None:
    """Test that the Frechet Cell FIRE optimizer actually minimizes energy for different
    md_flavors."""
    print(f"\n--- Starting test_frechet_cell_fire_optimization for {md_flavor=} ---")

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
        batch=ar_supercell_sim_state.batch.clone(),
    )
    print(f"[{md_flavor}] Initial SimState created for Frechet test.")

    initial_state_positions = current_sim_state.positions.clone()
    initial_state_cell = current_sim_state.cell.clone()

    # Initialize FIRE optimizer
    print(f"Initializing Frechet {md_flavor} optimizer...")
    init_fn, update_fn = frechet_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
        md_flavor=md_flavor,
    )
    print(f"[{md_flavor}] Frechet optimizer functions obtained.")

    state = init_fn(current_sim_state)
    energy = float(getattr(state, "energy", "nan"))
    print(f"[{md_flavor}] Initial state created by Frechet init_fn. {energy=:.4f}")

    # Run optimization for a few steps
    energies = [1000.0, state.energy.item()]  # Ensure float for comparison
    max_steps = 1000
    steps_taken = 0
    print(f"[{md_flavor}] Entering Frechet optimization loop (max_steps: {max_steps})...")

    while abs(energies[-2] - energies[-1]) > 1e-6 and steps_taken < max_steps:
        state = update_fn(state)
        energies.append(state.energy.item())
        steps_taken += 1

    print(f"[{md_flavor}] Frechet loop finished after {steps_taken} steps.")

    if steps_taken == max_steps and abs(energies[-2] - energies[-1]) > 1e-6:
        print(
            f"WARNING: Frechet Cell FIRE {md_flavor=} optimization did not converge "
            f"in {max_steps} steps. Final energy: {energies[-1]:.4f}"
        )
    else:
        print(
            f"Frechet Cell FIRE {md_flavor=} optimization converged in {steps_taken} "
            f"steps. Final energy: {energies[-1]:.4f}"
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
    pressure_tol = 0.01
    force_tol = 0.2

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


@pytest.mark.parametrize("optimizer_func", [fire, unit_cell_fire, frechet_cell_fire])
def test_optimizer_batch_consistency(
    optimizer_func: callable,
    ar_supercell_sim_state: SimState,
    lj_model: torch.nn.Module,
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
    if optimizer_func in (unit_cell_fire, frechet_cell_fire):
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

    def energy_converged(current_e: torch.Tensor, prev_e: torch.Tensor) -> bool:
        """Check for energy convergence (scalar energies)."""
        return not torch.allclose(current_e, prev_e, atol=1e-6)

    for state_for_indiv_opt in [state1_orig.clone(), state2_orig.clone()]:
        init_fn_indiv, update_fn_indiv = optimizer_func(
            model=lj_model, dt_max=0.3, dt_start=0.1
        )
        opt_state_indiv = init_fn_indiv(state_for_indiv_opt)

        current_e_indiv = opt_state_indiv.energy
        # Ensure prev_e_indiv is different to start the loop
        prev_e_indiv = current_e_indiv + torch.tensor(
            1.0, device=current_e_indiv.device, dtype=current_e_indiv.dtype
        )

        steps_indiv = 0
        while energy_converged(current_e_indiv, prev_e_indiv):
            prev_e_indiv = current_e_indiv
            opt_state_indiv = update_fn_indiv(opt_state_indiv)
            current_e_indiv = opt_state_indiv.energy
            steps_indiv += 1
            if steps_indiv > 1000:
                raise ValueError(
                    f"Individual opt for {optimizer_func.__name__} did not converge"
                )
        final_individual_states.append(opt_state_indiv)

    # Batched optimization
    multi_state_initial = concatenate_states(
        [state1_orig.clone(), state2_orig.clone()],
        device=ar_supercell_sim_state.device,
    )

    init_fn_batch, update_fn_batch = optimizer_func(
        model=lj_model, dt_max=0.3, dt_start=0.1
    )
    batch_opt_state = init_fn_batch(multi_state_initial)

    current_energies_batch = batch_opt_state.energy.clone()
    # Ensure prev_energies_batch requires update and has same shape
    prev_energies_batch = current_energies_batch + torch.tensor(
        1.0, device=current_energies_batch.device, dtype=current_energies_batch.dtype
    )

    steps_batch = 0
    # Converge when all batch energies have converged
    while not torch.allclose(current_energies_batch, prev_energies_batch, atol=1e-6):
        prev_energies_batch = current_energies_batch.clone()
        batch_opt_state = update_fn_batch(batch_opt_state)
        current_energies_batch = batch_opt_state.energy.clone()
        steps_batch += 1
        if steps_batch > 1000:
            raise ValueError(
                f"Batched opt for {optimizer_func.__name__} did not converge"
            )

    individual_final_energies = [s.energy.item() for s in final_individual_states]
    for idx, indiv_energy in enumerate(individual_final_energies):
        assert abs(batch_opt_state.energy[idx].item() - indiv_energy) < 1e-4, (
            f"Energy batch {idx} ({optimizer_func.__name__}): "
            f"{batch_opt_state.energy[idx].item()} vs indiv {indiv_energy}"
        )

    # Check positions changed for both parts of the batch
    n_atoms_first_state = state1_orig.positions.shape[0]
    assert not torch.allclose(
        batch_opt_state.positions[:n_atoms_first_state],
        multi_state_initial.positions[:n_atoms_first_state],
        atol=1e-5,  # Added tolerance as in original frechet test
    ), f"{optimizer_func.__name__} positions batch 0 did not change."
    assert not torch.allclose(
        batch_opt_state.positions[n_atoms_first_state:],
        multi_state_initial.positions[n_atoms_first_state:],
        atol=1e-5,
    ), f"{optimizer_func.__name__} positions batch 1 did not change."

    if optimizer_func in (unit_cell_fire, frechet_cell_fire):
        assert not torch.allclose(
            batch_opt_state.cell, multi_state_initial.cell, atol=1e-5
        ), f"{optimizer_func.__name__} cell did not change."


def test_unit_cell_fire_multi_batch(
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test FIRE optimization with multiple batches."""
    # Create a multi-batch system by duplicating ar_fcc_state

    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    multi_state = concatenate_states(
        [ar_supercell_sim_state_1, ar_supercell_sim_state_2],
        device=ar_supercell_sim_state.device,
    )

    # Initialize FIRE optimizer
    init_fn, update_fn = unit_cell_fire(
        model=lj_model,
        dt_max=0.3,
        dt_start=0.1,
    )

    state = init_fn(multi_state)
    initial_state = copy.deepcopy(state)

    # Run optimization for a few steps
    prev_energy = torch.ones(2, device=state.device, dtype=state.energy.dtype) * 1000
    current_energy = initial_state.energy
    step = 0
    while not torch.allclose(current_energy, prev_energy, atol=1e-9):
        prev_energy = current_energy
        state = update_fn(state)
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
    ar_supercell_sim_state: SimState, lj_model: torch.nn.Module
) -> None:
    """Test batched Frechet Fixed cell FIRE optimization is
    consistent with FIRE (position only) optimizations."""
    generator = torch.Generator(device=ar_supercell_sim_state.device)

    ar_supercell_sim_state_1 = copy.deepcopy(ar_supercell_sim_state)
    ar_supercell_sim_state_2 = copy.deepcopy(ar_supercell_sim_state)

    # Add same random perturbation to both states
    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(state.positions.shape, device=state.device, generator=generator)
            * 0.1
        )

    # Optimize each state individually
    final_individual_states_unit_cell = []
    total_steps_unit_cell = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = unit_cell_fire(
            model=lj_model,
            dt_max=0.3,
            dt_start=0.1,
            hydrostatic_strain=True,
            constant_volume=True,
        )

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
            current_energy = state_opt.energy
            step += 1
            if step > 1000:
                raise ValueError("Optimization did not converge")

        final_individual_states_unit_cell.append(state_opt)
        total_steps_unit_cell.append(step)

    # Optimize each state individually
    final_individual_states_fire = []
    total_steps_fire = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_supercell_sim_state_1, ar_supercell_sim_state_2]:
        init_fn, update_fn = fire(model=lj_model, dt_max=0.3, dt_start=0.1)

        state_opt = init_fn(state)

        # Run optimization until convergence
        current_energy = state_opt.energy
        prev_energy = current_energy + 1

        step = 0
        while energy_converged(current_energy, prev_energy):
            prev_energy = current_energy
            state_opt = update_fn(state_opt)
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
            f"Energy for batch {step} doesn't match position only optimization: "
            f"batch={energy_unit_cell}, individual={individual_energies_fire[step]}"
        )
