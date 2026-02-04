from typing import Any

import pytest
import torch

import torch_sim as ts
from torch_sim.autobatching import (
    BinningAutoBatcher,
    InFlightAutoBatcher,
    calculate_memory_scaler,
    determine_max_batch_size,
    to_constant_volume_bins,
)
from torch_sim.models.lennard_jones import LennardJonesModel


def test_exact_fit():
    values = [1, 2, 1]
    bins = to_constant_volume_bins(values, 2)
    assert len(bins) == 2


def test_weight_pos():
    values = [[1, "x"], [2, "y"], [1, "z"]]
    bins = to_constant_volume_bins(values, 2, weight_pos=0)
    for vol_bin in bins:
        for item in vol_bin:
            assert isinstance(item, list)
            assert isinstance(item[0], int)
            assert isinstance(item[1], str)


def test_key_func():
    values = [{"x": "a", "y": 1}, {"x": "b", "y": 5}, {"x": "b", "y": 3}]
    bins = to_constant_volume_bins(values, 2, key=lambda x: x["y"])

    for vol_bin in bins:
        for item in vol_bin:
            assert isinstance(item, dict)
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


def test_calculate_scaling_metric(si_sim_state: ts.SimState) -> None:
    """Test calculation of scaling metrics for a state."""
    # Test n_atoms metric
    n_atoms_metric = calculate_memory_scaler(si_sim_state, "n_atoms")
    assert n_atoms_metric == [si_sim_state.n_atoms]
    assert n_atoms_metric == [8]

    # Test n_atoms_x_density metric
    density_metric = calculate_memory_scaler(si_sim_state, "n_atoms_x_density")
    volume = torch.abs(torch.linalg.det(si_sim_state.cell[0])) / 1000
    expected = si_sim_state.n_atoms * (si_sim_state.n_atoms / volume.item())
    assert pytest.approx(density_metric[0], rel=1e-5) == expected
    assert pytest.approx(density_metric[0], rel=1e-5) == (8**2) * 1000 / (5.43**3)

    # Test invalid metric
    with pytest.raises(ValueError, match="Invalid metric"):
        calculate_memory_scaler(si_sim_state, "invalid_metric")


def test_calculate_scaling_metric_non_periodic(benzene_sim_state: ts.SimState) -> None:
    """Test calculation of scaling metrics for a non-periodic state."""
    n_atoms_metric = calculate_memory_scaler(benzene_sim_state, "n_atoms")
    assert n_atoms_metric == [benzene_sim_state.n_atoms]
    assert n_atoms_metric == [12]

    n_atoms_x_density_metric = calculate_memory_scaler(
        benzene_sim_state, "n_atoms_x_density"
    )
    assert n_atoms_x_density_metric[0] > 0
    bbox = (
        benzene_sim_state.positions.max(dim=0).values
        - benzene_sim_state.positions.min(dim=0).values
    ).clone()
    for i, p in enumerate(benzene_sim_state.pbc):
        if not p:
            bbox[i] += 2.0
    assert pytest.approx(n_atoms_x_density_metric[0], rel=1e-5) == (
        benzene_sim_state.n_atoms**2 / (bbox.prod().item() / 1000)
    )


def test_split_state(si_double_sim_state: ts.SimState) -> None:
    """Test splitting a batched state into individual states."""
    split_states = si_double_sim_state.split()

    # Check we get the right number of states
    assert len(split_states) == 2

    # Check each state has the correct properties
    for state in enumerate(split_states):
        assert state[1].n_systems == 1
        assert torch.all(
            state[1].system_idx == 0
        )  # Each split state should have system indices reset to 0
        assert state[1].n_atoms == si_double_sim_state.n_atoms // 2
        assert state[1].positions.shape[0] == si_double_sim_state.n_atoms // 2
        assert state[1].cell.shape[0] == 1


def test_binning_auto_batcher(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with different states."""
    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = BinningAutoBatcher(
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
    batches = [batch for batch, _ in batcher]

    # Check we got the expected number of systems
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


def test_binning_auto_batcher_auto_metric(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test BinningAutoBatcher with different states."""
    # monkeypatch determine max memory scaler
    monkeypatch.setattr(
        "torch_sim.autobatching.determine_max_batch_size",
        lambda *args, **kwargs: 50,  # noqa: ARG005
    )

    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
    )
    batcher.load_states(states)

    # Check that the batcher correctly identified the metrics
    assert len(batcher.memory_scalers) == 2
    assert batcher.memory_scalers[0] == si_sim_state.n_atoms
    assert batcher.memory_scalers[1] == fe_supercell_sim_state.n_atoms

    # Get batches until None is returned
    batches = [batch for batch, _ in batcher]

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


def test_binning_auto_batcher_with_indices(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with indices tracking."""
    states = [si_sim_state, fe_supercell_sim_state]

    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,
    )
    batcher.load_states(states)

    # Get batches and track indices manually
    batches_with_indices = []
    for batch, indices in batcher:
        batches_with_indices.append((batch, indices))

    # Check we got the expected number of batches
    assert len(batches_with_indices) == len(batcher.batched_states)

    # Check that the indices match the expected bin indices
    for idx, (_, indices) in enumerate(batches_with_indices):
        assert indices == batcher.index_bins[idx]


def test_binning_auto_batcher_restore_order_with_split_states(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher's restore_original_order method with split states."""
    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # loop through all batches to test we're restore order correctly
    batches = []
    for batch, _indices in batcher:
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


def test_in_flight_max_metric_too_small(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher with different states."""
    # Create a list of states
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric
    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=1.0,  # Set a small value to force multiple batches
    )
    # Get the first batch
    with pytest.raises(ValueError, match="is greater than max_metric"):
        batcher.load_states(states)


def test_in_flight_auto_batcher(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher with different states."""
    # Create a list of states
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric
    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # Get the first batch
    first_batch, [] = batcher.next_batch(states, None)
    assert isinstance(first_batch, ts.SimState)

    # Create a convergence tensor where the first state has converged
    convergence = torch.tensor([True])

    # Get the next batch
    next_batch, popped_batch = batcher.next_batch(first_batch, convergence)
    assert isinstance(next_batch, ts.SimState)
    assert isinstance(popped_batch, list)
    assert isinstance(popped_batch[0], ts.SimState)

    # Check that the converged state was removed
    assert len(batcher.current_scalers) == 1
    assert len(batcher.current_idx) == 1
    assert len(batcher.completed_idx_og_order) == 1

    # Create a convergence tensor where the remaining state has converged
    convergence = torch.tensor([True])

    # Get the next batch, which should be None since all states have converged
    final_batch, popped_batch = batcher.next_batch(next_batch, convergence)
    assert final_batch is None

    # Check that all states are marked as completed
    assert len(batcher.completed_idx_og_order) == 2


def test_determine_max_batch_size_fibonacci(
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that determine_max_batch_size uses Fibonacci sequence correctly."""

    # Mock measure_model_memory_forward to avoid actual GPU memory testing
    def mock_measure(*_args: Any, **_kwargs: Any) -> float:
        return 0.1  # Return a small constant memory usage

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", mock_measure
    )

    # Test with a small max_atoms value to limit the sequence
    max_size = determine_max_batch_size(si_sim_state, lj_model, max_atoms=16)
    # The Fibonacci sequence up to 10 is [1, 2, 3, 5, 8, 13]
    # Since we're not triggering OOM errors with our mock, it should return the
    # largest value that fits within max_atoms (simstate has 8 atoms, so 2 batches)
    assert max_size == 2


@pytest.mark.parametrize("scale_factor", [1.1, 1.4])
def test_determine_max_batch_size_small_scale_factor_no_infinite_loop(
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
    scale_factor: float,
) -> None:
    """Test determine_max_batch_size doesn't infinite loop with small scale factors."""
    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", lambda *_: 0.1
    )

    max_size = determine_max_batch_size(
        si_sim_state, lj_model, max_atoms=20, scale_factor=scale_factor
    )
    assert 0 < max_size <= 20

    # Verify sequence is strictly increasing (prevents infinite loop)
    sizes = [1]
    while (
        next_size := max(round(sizes[-1] * scale_factor), sizes[-1] + 1)
    ) * si_sim_state.n_atoms <= 20:
        sizes.append(next_size)

    assert all(sizes[idx] > sizes[idx - 1] for idx in range(1, len(sizes)))
    assert max_size == sizes[-1]


def test_in_flight_auto_batcher_restore_order(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher's restore_original_order method."""
    states = [si_sim_state, fe_supercell_sim_state]

    batcher = InFlightAutoBatcher(
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


@pytest.mark.parametrize(
    "num_steps_per_batch",
    [
        5,  # At 5 steps, not every state will converge before the next batch.
        # This tests the merging of partially converged states with new states
        # which has been a bug in the past.
        10,  # At 10 steps, all states will converge before the next batch
    ],
)
def test_in_flight_with_fire(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    num_steps_per_batch: int,
) -> None:
    si_fire_state = ts.fire_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_fire_state = ts.fire_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    fire_states = [si_fire_state, fe_fire_state] * 5
    fire_states = [state.clone() for state in fire_states]
    for state in fire_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        # max_metric=400_000,
        max_memory_scaler=600,
    )
    batcher.load_states(fire_states)

    def convergence_fn(state: ts.FireState) -> torch.Tensor:
        system_wise_max_force = torch.zeros(
            state.n_systems, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        system_wise_max_force = system_wise_max_force.scatter_reduce(
            dim=0, index=state.system_idx, src=max_forces, reduce="amax"
        )
        return system_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        state, completed_states = batcher.next_batch(state, convergence_tensor)

        all_completed_states.extend(completed_states)
        if state is None:
            break

        for _ in range(num_steps_per_batch):
            state = ts.fire_step(state=state, model=lj_model)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(fire_states)


def test_binning_auto_batcher_with_fire(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    si_fire_state = ts.fire_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_fire_state = ts.fire_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    fire_states = [si_fire_state, fe_fire_state] * 5
    fire_states = [state.clone() for state in fire_states]
    for state in fire_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batch_lengths = [state.n_atoms for state in fire_states]
    optimal_batches = to_constant_volume_bins(batch_lengths, 400)
    optimal_n_systems = len(optimal_batches)

    batcher = BinningAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=400
    )
    batcher.load_states(fire_states)

    finished_states = []
    n_systems = 0
    for batch, _ in batcher:
        n_systems += 1
        for _ in range(5):
            batch = ts.fire_step(state=batch, model=lj_model)

        finished_states.extend(batch.split())

    restored_states = batcher.restore_original_order(finished_states)
    assert len(restored_states) == len(fire_states)
    for restored, original in zip(restored_states, fire_states, strict=True):
        assert torch.all(restored.atomic_numbers == original.atomic_numbers)
    # analytically determined to be optimal
    assert n_systems == optimal_n_systems


def test_in_flight_max_iterations(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher with max_iterations limit."""
    # Create states that won't naturally converge
    states = [si_sim_state.clone(), fe_supercell_sim_state.clone()]

    # Set max_iterations to a small value to ensure quick termination
    max_iterations = 3
    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=800.0,
        max_iterations=max_iterations,
    )
    batcher.load_states(states)

    # Get the first batch
    state, [] = batcher.next_batch(None, None)
    assert state is not None

    # Create a convergence tensor that never converges
    convergence_tensor = torch.zeros(state.n_systems, dtype=torch.bool)

    all_completed_states = []
    iteration_count = 0

    # Process batches until complete
    while state is not None:
        iteration_count += 1
        state, completed_states = batcher.next_batch(state, convergence_tensor)
        all_completed_states.extend(completed_states)

        # Update convergence tensor for next iteration (still all False)
        if state is not None:
            convergence_tensor = torch.zeros(state.n_systems, dtype=torch.bool)

        if iteration_count > max_iterations + 4:
            raise ValueError("Should have terminated by now")

    # Verify all states were processed
    assert len(all_completed_states) == len(states)

    # Verify we didn't exceed max_iterations + 1 iterations (first call doesn't count)
    assert iteration_count == 3

    # Verify iteration_count tracking
    for idx in range(len(states)):
        assert batcher.iteration_count[idx] == max_iterations


@pytest.mark.parametrize(
    "num_steps_per_batch",
    [
        5,  # At 5 steps, not every state will converge before the next batch.
        10,  # At 10 steps, all states will converge before the next batch
    ],
)
def test_in_flight_with_bfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    num_steps_per_batch: int,
) -> None:
    """Test InFlightAutoBatcher with BFGS optimizer."""
    si_bfgs_state = ts.bfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_bfgs_state = ts.bfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    bfgs_states = [si_bfgs_state, fe_bfgs_state] * 5
    bfgs_states = [state.clone() for state in bfgs_states]
    for state in bfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=6000,
    )
    batcher.load_states(bfgs_states)

    def convergence_fn(state: ts.BFGSState) -> torch.Tensor:
        system_wise_max_force = torch.zeros(
            state.n_systems, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        system_wise_max_force = system_wise_max_force.scatter_reduce(
            dim=0, index=state.system_idx, src=max_forces, reduce="amax"
        )
        return system_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        state, completed_states = batcher.next_batch(state, convergence_tensor)

        all_completed_states.extend(completed_states)
        if state is None:
            break

        for _ in range(num_steps_per_batch):
            state = ts.bfgs_step(state=state, model=lj_model)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(bfgs_states)


def test_binning_auto_batcher_with_bfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with BFGS optimizer."""
    si_bfgs_state = ts.bfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_bfgs_state = ts.bfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    bfgs_states = [si_bfgs_state, fe_bfgs_state] * 5
    bfgs_states = [state.clone() for state in bfgs_states]
    for state in bfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = BinningAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=6000
    )
    batcher.load_states(bfgs_states)

    all_finished_states: list[ts.SimState] = []
    total_batches = 0
    for batch, _ in batcher:
        total_batches += 1  # noqa: SIM113
        for _ in range(5):
            batch = ts.bfgs_step(state=batch, model=lj_model)
        all_finished_states.extend(batch.split())

    assert len(all_finished_states) == len(bfgs_states)


@pytest.mark.parametrize(
    "num_steps_per_batch",
    [
        5,  # At 5 steps, not every state will converge before the next batch.
        10,  # At 10 steps, all states will converge before the next batch
    ],
)
def test_in_flight_with_lbfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    num_steps_per_batch: int,
) -> None:
    """Test InFlightAutoBatcher with L-BFGS optimizer."""
    si_lbfgs_state = ts.lbfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_lbfgs_state = ts.lbfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    lbfgs_states = [si_lbfgs_state, fe_lbfgs_state] * 5
    lbfgs_states = [state.clone() for state in lbfgs_states]
    for state in lbfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=6000,
    )
    batcher.load_states(lbfgs_states)

    def convergence_fn(state: ts.LBFGSState) -> torch.Tensor:
        system_wise_max_force = torch.zeros(
            state.n_systems, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        system_wise_max_force = system_wise_max_force.scatter_reduce(
            dim=0, index=state.system_idx, src=max_forces, reduce="amax"
        )
        return system_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        state, completed_states = batcher.next_batch(state, convergence_tensor)

        all_completed_states.extend(completed_states)
        if state is None:
            break

        for _ in range(num_steps_per_batch):
            state = ts.lbfgs_step(state=state, model=lj_model)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(lbfgs_states)


def test_binning_auto_batcher_with_lbfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with L-BFGS optimizer."""
    si_lbfgs_state = ts.lbfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_lbfgs_state = ts.lbfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    lbfgs_states = [si_lbfgs_state, fe_lbfgs_state] * 5
    lbfgs_states = [state.clone() for state in lbfgs_states]
    for state in lbfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = BinningAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=6000
    )
    batcher.load_states(lbfgs_states)

    all_finished_states: list[ts.SimState] = []
    total_batches = 0
    for batch, _ in batcher:
        total_batches += 1  # noqa: SIM113
        for _ in range(5):
            batch = ts.lbfgs_step(state=batch, model=lj_model)
        all_finished_states.extend(batch.split())

    assert len(all_finished_states) == len(lbfgs_states)
