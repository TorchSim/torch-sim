from typing import Any

import pytest
import torch

from torch_sim.autobatching import (
    ChunkingAutoBatcher,
    HotSwappingAutoBatcher,
    calculate_memory_scaler,
    determine_max_batch_size,
    to_constant_volume_bins,
)
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.state import SimState


def test_exact_fit():
    values = [1, 2, 1]
    bins = to_constant_volume_bins(values, 2)
    assert len(bins) == 2


def test_weight_pos():
    values = [[1, "x"], [2, "y"], [1, "z"]]
    bins = to_constant_volume_bins(values, 2, weight_pos=0)
    for bin_ in bins:
        for item in bin_:
            assert isinstance(item[0], int)
            assert isinstance(item[1], str)


def test_key_func():
    values = [{"x": "a", "y": 1}, {"x": "b", "y": 5}, {"x": "b", "y": 3}]
    bins = to_constant_volume_bins(values, 2, key=lambda x: x["y"])

    for bin_ in bins:
        for item in bin_:
            assert "x" in item
            assert "y" in item


def test_no_fit():
    values = [42, 24]
    bins = to_constant_volume_bins(values, 20)
    assert bins == [[42], [24]]


def test_bounds_and_tuples():
    c = [
        ("a", 10, "foo"),
        ("b", 10, "log"),
        ("c", 11),
        ("d", 1, "bar"),
        ("e", 2, "bommel"),
        ("f", 7, "floggo"),
    ]
    V_max = 11

    bins = to_constant_volume_bins(c, V_max, weight_pos=1, upper_bound=11)
    bins = [sorted(_bin, key=lambda x: x[0]) for _bin in bins]
    assert bins == [
        [("a", 10, "foo"), ("d", 1, "bar")],
        [("b", 10, "log")],
        [
            ("e", 2, "bommel"),
            ("f", 7, "floggo"),
        ],
    ]

    bins = to_constant_volume_bins(c, V_max, weight_pos=1, lower_bound=1)
    bins = [sorted(_bin, key=lambda x: x[0]) for _bin in bins]
    assert bins == [
        [("c", 11)],
        [("a", 10, "foo")],
        [("b", 10, "log")],
        [
            ("e", 2, "bommel"),
            ("f", 7, "floggo"),
        ],
    ]

    bins = to_constant_volume_bins(c, V_max, weight_pos=1, lower_bound=1, upper_bound=11)
    bins = [sorted(_bin, key=lambda x: x[0]) for _bin in bins]
    assert bins == [
        [("a", 10, "foo")],
        [("b", 10, "log")],
        [("e", 2, "bommel"), ("f", 7, "floggo")],
    ]


def test_calculate_scaling_metric(si_sim_state: SimState) -> None:
    """Test calculation of scaling metrics for a state."""
    # Test n_atoms metric
    n_atoms_metric = calculate_memory_scaler(si_sim_state, "n_atoms")
    assert n_atoms_metric == si_sim_state.n_atoms

    # Test n_atoms_x_density metric
    density_metric = calculate_memory_scaler(si_sim_state, "n_atoms_x_density")
    volume = torch.abs(torch.linalg.det(si_sim_state.cell[0])) / 1000
    expected = si_sim_state.n_atoms * (si_sim_state.n_atoms / volume.item())
    assert pytest.approx(density_metric, rel=1e-5) == expected

    # Test invalid metric
    with pytest.raises(ValueError, match="Invalid metric"):
        calculate_memory_scaler(si_sim_state, "invalid_metric")


def test_split_state(si_double_sim_state: SimState) -> None:
    """Test splitting a batched state into individual states."""
    split_states = si_double_sim_state.split()

    # Check we get the right number of states
    assert len(split_states) == 2

    # Check each state has the correct properties
    for state in enumerate(split_states):
        assert state[1].n_batches == 1
        assert torch.all(
            state[1].batch == 0
        )  # Each split state should have batch indices reset to 0
        assert state[1].n_atoms == si_double_sim_state.n_atoms // 2
        assert state[1].positions.shape[0] == si_double_sim_state.n_atoms // 2
        assert state[1].cell.shape[0] == 1


def test_chunking_auto_batcher(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    """Test ChunkingAutoBatcher with different states."""
    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = ChunkingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # Check that the batcher correctly identified the metrics
    assert len(batcher.memory_scalers) == 2
    assert batcher.memory_scalers[0] == si_sim_state.n_atoms
    assert batcher.memory_scalers[1] == fe_supercell_sim_state.n_atoms

    # Get batches until None is returned
    batches = list(batcher)

    # Check we got the expected number of batches
    assert len(batches) == len(batcher.batched_states)

    # Test restore_original_order
    restored_states = batcher.restore_original_order(batches)
    assert len(restored_states) == len(states)

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)


def test_chunking_auto_batcher_with_indices(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    """Test ChunkingAutoBatcher with return_indices=True."""
    states = [si_sim_state, fe_supercell_sim_state]

    batcher = ChunkingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,
        return_indices=True,
    )
    batcher.load_states(states)

    # Get batches with indices
    batches_with_indices = []
    for batch, indices in batcher:
        batches_with_indices.append((batch, indices))

    # Check we got the expected number of batches
    assert len(batches_with_indices) == len(batcher.batched_states)

    # Check that the indices match the expected bin indices
    for i, (_, indices) in enumerate(batches_with_indices):
        assert indices == batcher.index_bins[i]


def test_chunking_auto_batcher_restore_order_with_split_states(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    """Test ChunkingAutoBatcher's restore_original_order method with split states."""
    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = ChunkingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # Get batches until None is returned
    batches = []
    while True:
        batch = batcher.next_batch()
        if batch is None:
            break
        # Split each batch into individual states to simulate processing
        # split_batch = split_state(batch)
        batches.append(batch)

    # Test restore_original_order with split states
    # This tests the chain.from_iterable functionality
    restored_states = batcher.restore_original_order(batches)

    # Check we got the right number of states back
    assert len(restored_states) == len(states)

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)


def test_hot_swapping_max_metric_too_small(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    """Test HotSwappingAutoBatcher with different states."""
    # Create a list of states
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric
    batcher = HotSwappingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=1.0,  # Set a small value to force multiple batches
    )
    # Get the first batch
    with pytest.raises(ValueError, match="is greater than max_metric"):
        batcher.load_states(states)


def test_hot_swapping_auto_batcher(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    """Test HotSwappingAutoBatcher with different states."""
    # Create a list of states
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric
    batcher = HotSwappingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,  # Set a small value to force multiple batches
        return_indices=True,
    )
    batcher.load_states(states)

    # Get the first batch
    first_batch, [], _ = batcher.next_batch(states, None)
    assert isinstance(first_batch, SimState)

    # Create a convergence tensor where the first state has converged
    convergence = torch.tensor([True])

    # Get the next batch
    next_batch, popped_batch, idx = batcher.next_batch(first_batch, convergence)
    assert isinstance(next_batch, SimState)
    assert isinstance(popped_batch, list)
    assert isinstance(popped_batch[0], SimState)
    assert idx == [1]

    # Check that the converged state was removed
    assert len(batcher.current_scalers) == 1
    assert len(batcher.current_idx) == 1
    assert len(batcher.completed_idx_og_order) == 1

    # Create a convergence tensor where the remaining state has converged
    convergence = torch.tensor([True])

    # Get the next batch, which should be None since all states have converged
    final_batch, popped_batch, _ = batcher.next_batch(next_batch, convergence)
    assert final_batch is None

    # Check that all states are marked as completed
    assert len(batcher.completed_idx_og_order) == 2


def test_determine_max_batch_size_fibonacci(
    si_sim_state: SimState, lj_model: LennardJonesModel, monkeypatch: Any
) -> None:
    """Test that determine_max_batch_size uses Fibonacci sequence correctly."""

    # Mock measure_model_memory_forward to avoid actual GPU memory testing
    def mock_measure(*_args: Any, **_kwargs: Any) -> float:
        return 0.1  # Return a small constant memory usage

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", mock_measure
    )

    # Test with a small max_atoms value to limit the sequence
    max_size = determine_max_batch_size(si_sim_state, lj_model, max_atoms=10)

    # The Fibonacci sequence up to 10 is [1, 2, 3, 5, 8, 13]
    # Since we're not triggering OOM errors with our mock, it should
    # return the largest value < max_atoms
    assert max_size == 8


def test_hot_swapping_auto_batcher_restore_order(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    """Test HotSwappingAutoBatcher's restore_original_order method."""
    states = [si_sim_state, fe_supercell_sim_state]

    batcher = HotSwappingAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=260.0
    )
    batcher.load_states(states)

    # Get the first batch
    first_batch, [] = batcher.next_batch(states, None)

    # Simulate convergence of all states
    completed_states_list = []
    convergence = torch.tensor([True])
    next_batch, completed_states = batcher.next_batch(first_batch, convergence)
    completed_states_list.extend(completed_states)

    # sample batch a second time
    # sample batch a second time
    next_batch, completed_states = batcher.next_batch(next_batch, convergence)
    completed_states_list.extend(completed_states)

    # Test restore_original_order
    restored_states = batcher.restore_original_order(completed_states_list)
    assert len(restored_states) == 2

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)

    # # Test error when number of states doesn't match
    # with pytest.raises(
    #     ValueError, match="Number of completed states .* does not match"
    # ):
    #     batcher.restore_original_order([si_sim_state])


def test_hot_swapping_with_fire(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    fire_init, fire_update = unit_cell_fire(lj_model)

    si_fire_state = fire_init(si_sim_state)
    fe_fire_state = fire_init(fe_supercell_sim_state)

    fire_states = [si_fire_state, fe_fire_state] * 5
    fire_states = [state.clone() for state in fire_states]
    for state in fire_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = HotSwappingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        # max_metric=400_000,
        max_memory_scaler=600,
    )
    batcher.load_states(fire_states)

    def convergence_fn(state: SimState) -> bool:
        batch_wise_max_force = torch.zeros(
            state.n_batches, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        batch_wise_max_force = batch_wise_max_force.scatter_reduce(
            dim=0, index=state.batch, src=max_forces, reduce="amax"
        )
        return batch_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        print(f"Starting new batch of {state.n_batches} states.")

        state, completed_states = batcher.next_batch(state, convergence_tensor)
        print("Number of completed states", len(completed_states))

        all_completed_states.extend(completed_states)
        if state is None:
            break

        # run 10 steps, arbitrary number
        for _ in range(10):
            state = fire_update(state)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(fire_states)


def test_chunking_auto_batcher_with_fire(
    si_sim_state: SimState, fe_supercell_sim_state: SimState, lj_model: LennardJonesModel
) -> None:
    fire_init, fire_update = unit_cell_fire(lj_model)

    si_fire_state = fire_init(si_sim_state)
    fe_fire_state = fire_init(fe_supercell_sim_state)

    fire_states = [si_fire_state, fe_fire_state] * 5
    fire_states = [state.clone() for state in fire_states]
    for state in fire_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batch_lengths = [state.n_atoms for state in fire_states]
    optimal_batches = to_constant_volume_bins(batch_lengths, 400)
    optimal_n_batches = len(optimal_batches)

    batcher = ChunkingAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=400
    )
    batcher.load_states(fire_states)

    finished_states = []
    n_batches = 0
    for batch in batcher:
        n_batches += 1
        for _ in range(5):
            batch = fire_update(batch)

        finished_states.extend(batch.split())

    restored_states = batcher.restore_original_order(finished_states)
    assert len(restored_states) == len(fire_states)
    for restored, original in zip(restored_states, fire_states, strict=True):
        assert torch.all(restored.atomic_numbers == original.atomic_numbers)
    # analytically determined to be optimal
    assert n_batches == optimal_n_batches


def test_hot_swapping_max_iterations(
    si_sim_state: SimState,
    fe_supercell_sim_state: SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test HotSwappingAutoBatcher with max_iterations limit."""
    # Create states that won't naturally converge
    states = [si_sim_state.clone(), fe_supercell_sim_state.clone()]

    # Set max_attempts to a small value to ensure quick termination
    max_attempts = 3
    batcher = HotSwappingAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=800.0,
        max_iterations=max_attempts,
    )
    batcher.load_states(states)

    # Get the first batch
    state, [] = batcher.next_batch(None, None)

    # Create a convergence tensor that never converges
    convergence_tensor = torch.zeros(state.n_batches, dtype=torch.bool)

    all_completed_states = []
    iteration_count = 0

    # Process batches until complete
    while state is not None:
        iteration_count += 1
        state, completed_states = batcher.next_batch(state, convergence_tensor)
        all_completed_states.extend(completed_states)

        # Update convergence tensor for next iteration (still all False)
        if state is not None:
            convergence_tensor = torch.zeros(state.n_batches, dtype=torch.bool)

        if iteration_count > max_attempts + 4:
            raise ValueError("Should have terminated by now")

    # Verify all states were processed
    assert len(all_completed_states) == len(states)

    # Verify we didn't exceed max_attempts + 1 iterations (first call doesn't count)
    assert iteration_count == 3

    # Verify swap_attempts tracking
    for i in range(len(states)):
        assert batcher.swap_attempts[i] == max_attempts
