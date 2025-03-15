import copy

import torch

from torch_sim.optimizers import (
    gradient_descent,
    unit_cell_fire,
    unit_cell_gradient_descent,
)
from torch_sim.state import BaseState, concatenate_states


def test_gradient_descent_optimization(
    ar_base_state: BaseState, lj_calculator: torch.nn.Module
) -> None:
    """Test that the Gradient Descent optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_base_state.positions + torch.randn_like(ar_base_state.positions) * 0.1
    )

    ar_base_state.positions = perturbed_positions
    initial_state = ar_base_state

    # Initialize Gradient Descent optimizer
    init_fn, update_fn = gradient_descent(
        model=lj_calculator,
        lr=0.01,
    )

    state = init_fn(ar_base_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"FIRE optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    assert max_force < 0.2, f"Forces should be small after optimization (got {max_force})"

    assert not torch.allclose(state.positions, initial_state.positions)


def test_unit_cell_gradient_descent_optimization(
    ar_base_state: BaseState, lj_calculator: torch.nn.Module
) -> None:
    """Test that the Gradient Descent optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_base_state.positions + torch.randn_like(ar_base_state.positions) * 0.1
    )

    ar_base_state.positions = perturbed_positions
    initial_state = ar_base_state

    # Initialize Gradient Descent optimizer
    init_fn, update_fn = unit_cell_gradient_descent(
        model=lj_calculator,
        positions_lr=0.01,
        cell_lr=0.1,
    )

    state = init_fn(ar_base_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"FIRE optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization (got {pressure})"
    )
    assert max_force < 0.2, f"Forces should be small after optimization (got {max_force})"

    assert not torch.allclose(state.positions, initial_state.positions)
    assert not torch.allclose(state.cell, initial_state.cell)


def test_unit_cell_fire_optimization(
    ar_base_state: BaseState, lj_calculator: torch.nn.Module
) -> None:
    """Test that the FIRE optimizer actually minimizes energy."""
    # Add some random displacement to positions
    perturbed_positions = (
        ar_base_state.positions + torch.randn_like(ar_base_state.positions) * 0.1
    )

    ar_base_state.positions = perturbed_positions
    initial_state = ar_base_state

    # Initialize FIRE optimizer
    init_fn, update_fn = unit_cell_fire(
        model=lj_calculator,
        dt_max=0.3,
        dt_start=0.1,
    )

    state = init_fn(ar_base_state)

    # Run optimization for a few steps
    energies = [1000, state.energy.item()]
    while abs(energies[-2] - energies[-1]) > 1e-6:
        state = update_fn(state)
        energies.append(state.energy.item())

    energies = energies[1:]

    # Check that energy decreased
    assert energies[-1] < energies[0], (
        f"FIRE optimization should reduce energy "
        f"(initial: {energies[0]}, final: {energies[-1]})"
    )

    # Check force convergence
    max_force = torch.max(torch.norm(state.forces, dim=1))
    pressure = torch.trace(state.stress.squeeze(0)) / 3.0
    assert pressure < 0.01, (
        f"Pressure should be small after optimization (got {pressure})"
    )
    assert max_force < 0.2, f"Forces should be small after optimization (got {max_force})"

    assert not torch.allclose(state.positions, initial_state.positions)
    assert not torch.allclose(state.cell, initial_state.cell)


def test_unit_cell_fire_multi_batch(
    ar_base_state: BaseState, lj_calculator: torch.nn.Module
) -> None:
    """Test FIRE optimization with multiple batches."""
    # Create a multi-batch system by duplicating ar_fcc_state

    generator = torch.Generator(device=ar_base_state.device)

    ar_base_state_1 = copy.deepcopy(ar_base_state)
    ar_base_state_2 = copy.deepcopy(ar_base_state)

    for state in [ar_base_state_1, ar_base_state_2]:
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
        [ar_base_state_1, ar_base_state_2],
        device=ar_base_state.device,
    )

    # Initialize FIRE optimizer
    init_fn, update_fn = unit_cell_fire(
        model=lj_calculator,
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
        f"Forces should be small after optimization (got {max_force})"
    )

    pressure_0 = torch.trace(state.stress[0]) / 3.0
    pressure_1 = torch.trace(state.stress[1]) / 3.0
    assert torch.allclose(pressure_0, pressure_1), (
        f"Pressure should be the same for all batches (got {pressure_0} and {pressure_1})"
    )
    assert pressure_0 < 0.01, (
        f"Pressure should be small after optimization (got {pressure_0})"
    )
    assert pressure_1 < 0.01, (
        f"Pressure should be small after optimization (got {pressure_1})"
    )

    n_ar_atoms = ar_base_state.n_atoms
    assert not torch.allclose(
        state.positions[:n_ar_atoms], multi_state.positions[:n_ar_atoms]
    )
    assert not torch.allclose(
        state.positions[n_ar_atoms:], multi_state.positions[n_ar_atoms:]
    )
    assert not torch.allclose(state.cell, multi_state.cell)

    # we are evolving identical systems
    assert current_energy[0] == current_energy[1]


def test_unit_cell_fire_batch_consistency(
    ar_base_state: BaseState, lj_calculator: torch.nn.Module
) -> None:
    """Test batched FIRE optimization is consistent with individual optimizations."""
    generator = torch.Generator(device=ar_base_state.device)

    ar_base_state_1 = copy.deepcopy(ar_base_state)
    ar_base_state_2 = copy.deepcopy(ar_base_state)

    # Add same random perturbation to both states
    for state in [ar_base_state_1, ar_base_state_2]:
        generator.manual_seed(43)
        state.positions += (
            torch.randn(
                state.positions.shape,
                device=state.device,
                generator=generator,
            )
            * 0.1
        )

    # Optimize each state individually
    final_individual_states = []
    total_steps = []

    def energy_converged(current_energy: float, prev_energy: float) -> bool:
        """Check if optimization should continue based on energy convergence."""
        return not torch.allclose(current_energy, prev_energy, atol=1e-6)

    for state in [ar_base_state_1, ar_base_state_2]:
        init_fn, update_fn = unit_cell_fire(
            model=lj_calculator,
            dt_max=0.3,
            dt_start=0.1,
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

        final_individual_states.append(state_opt)
        total_steps.append(step)

    # Now optimize both states together in a batch
    multi_state = concatenate_states(
        [copy.deepcopy(ar_base_state_1), copy.deepcopy(ar_base_state_2)],
        device=ar_base_state.device,
    )

    init_fn, batch_update_fn = unit_cell_fire(
        model=lj_calculator,
        dt_max=0.3,
        dt_start=0.1,
    )

    batch_state = init_fn(multi_state)

    # Run optimization until convergence for both batches
    current_energies = batch_state.energy.clone()
    prev_energies = current_energies + 1

    step = 0
    while energy_converged(current_energies[0], prev_energies[0]) and energy_converged(
        current_energies[1], prev_energies[1]
    ):
        prev_energies = current_energies.clone()
        batch_state = batch_update_fn(batch_state)
        current_energies = batch_state.energy.clone()
        step += 1
        if step > 1000:
            raise ValueError("Optimization did not converge")

    individual_energies = [state.energy.item() for state in final_individual_states]
    # Check that final energies from batched optimization match individual optimizations
    for step, individual_energy in enumerate(individual_energies):
        assert abs(batch_state.energy[step].item() - individual_energy) < 1e-4, (
            f"Energy for batch {step} doesn't match individual optimization: "
            f"batch={batch_state.energy[step].item()}, individual={individual_energy}"
        )
