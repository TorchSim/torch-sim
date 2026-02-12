"""Automatic batching and GPU memory management.

This module provides utilities for efficient batch processing of simulation states
by dynamically determining optimal batch sizes based on GPU memory constraints.
It includes tools for memory usage estimation, batch size determination, and
two complementary strategies for batching: binning and hot-swapping.

Example:
    Using BinningAutoBatcher with a model::

        batcher = BinningAutoBatcher(model, memory_scales_with="n_atoms")
        batcher.load_states(states)
        final_states = []
        for batch, _indices in batcher:
            final_states.append(evolve_batch(batch))
        final_states = batcher.restore_original_order(final_states)

Notes:
    Memory scaling estimates are approximate and may need tuning for specific
    model architectures and GPU configurations.
"""

from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from typing import Any, get_args

import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.typing import MemoryScaling


def to_constant_volume_bins[  # noqa: C901, PLR0915
    T: dict[int, float] | list[float] | list[tuple[T, ...]]
](
    items: T,
    max_volume: float,
    *,
    weight_pos: int | None = None,
    key: Callable[[T], float] | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> list[T]:
    """Distribute items into bins of fixed maximum volume.

    Groups items into the minimum number of bins possible while ensuring each bin's
    total weight does not exceed max_volume. Items are sorted by weight in descending
    order before binning to improve packing efficiency.

    Ported here from binpacking by @benmaier. https://pypi.org/project/binpacking.

    Args:
        items (dict[int, float] | list[float] | list[tuple]): Items to distribute,
            provided as either:
            - Dictionary with numeric weights as values
            - List of numeric weights
            - List of tuples containing weights (requires weight_pos or key)
        max_volume (float): Maximum allowed weight sum per bin.
        weight_pos (int | None): For tuple lists, index of weight in each tuple.
            Defaults to None.
        key (callable | None): Function to extract weight from list items.
            Defaults to None.
        lower_bound (float | None): Exclude items with weights below this value.
            Defaults to None.
        upper_bound (float | None): Exclude items with weights above this value.
            Defaults to None.

    Returns:
        list[dict[int, float]] | list[list[float]] | list[list[tuple]]:
            List of bins, where each bin contains items of the same type as input:
            - List of dictionaries if input was a dictionary
            - List of lists if input was a list of numbers
            - List of lists of tuples if input was a list of tuples

    Raises:
        TypeError: If input is not iterable.
        ValueError: If weight_pos or key is not provided for tuple list input,
            or if lower_bound >= upper_bound.
    """

    def _get_bins(lst: list[float], ndx: list[int]) -> list[float]:
        return [lst[n] for n in ndx]

    def _argmax_bins(lst: list[float]) -> int:
        return max(range(len(lst)), key=lst.__getitem__)

    def _rev_argsort_bins(lst: list[float]) -> list[int]:
        return sorted(range(len(lst)), key=lambda i: -lst[i])

    if not hasattr(items, "__len__"):
        raise TypeError("d must be iterable")

    if not isinstance(items, dict) and hasattr(items[0], "__len__"):
        if weight_pos is not None:
            key = lambda x: x[weight_pos]  # noqa: E731
        if key is None:
            raise ValueError("Must provide weight_pos or key for tuple list")

    if not isinstance(items, dict) and key:
        new_dict = dict(enumerate(items))
        items = {idx: key(val) for idx, val in enumerate(items)}  # type: ignore[invalid-assignment]
        is_tuple_list = True
    else:
        is_tuple_list = False

    if isinstance(items, dict):
        # get keys and values (weights)
        keys_vals = items.items()
        keys = [k for k, v in keys_vals]
        vals = [v for k, v in keys_vals]

        # sort weights decreasingly
        n_dcs = _rev_argsort_bins(vals)

        weights = _get_bins(vals, n_dcs)
        keys = _get_bins(keys, n_dcs)

        bins = [{}]
    else:
        weights = sorted(items, key=lambda x: -x)
        bins = [[]]

    # find the valid indices
    if lower_bound is not None and upper_bound is not None and lower_bound < upper_bound:
        valid_ndcs = filter(
            lambda i: lower_bound < weights[i] < upper_bound, range(len(weights))
        )
    elif lower_bound is not None:
        valid_ndcs = filter(lambda i: lower_bound < weights[i], range(len(weights)))
    elif upper_bound is not None:
        valid_ndcs = filter(lambda i: weights[i] < upper_bound, range(len(weights)))
    elif lower_bound is None and upper_bound is None:
        valid_ndcs = range(len(weights))
    elif lower_bound >= upper_bound:
        raise ValueError("lower_bound is greater or equal to upper_bound")

    valid_ndcs = list(valid_ndcs)

    weights = _get_bins(weights, valid_ndcs)

    if isinstance(items, dict):
        keys = _get_bins(keys, valid_ndcs)

    # prepare array containing the current weight of the bins
    weight_sum = [0.0]

    # iterate through the weight list, starting with heaviest
    for item, weight in enumerate(weights):
        if isinstance(items, dict):
            key = keys[item]

        # find candidate bins where the weight might fit
        candidate_bins = list(
            filter(lambda i: weight_sum[i] + weight <= max_volume, range(len(weight_sum)))
        )

        # if there are candidates where it fits
        if len(candidate_bins) > 0:
            # find the fullest bin where this item fits and assign it
            candidate_index = _argmax_bins(_get_bins(weight_sum, candidate_bins))
            b = candidate_bins[candidate_index]

        # if this weight doesn't fit in any existent bin
        elif item > 0:
            # note! if this is the very first item then there is already an
            # empty bin open so we don't need to open another one.

            # open a new bin
            b = len(weight_sum)
            weight_sum.append(0.0)
            if isinstance(items, dict):
                bins.append({})
            else:
                bins.append([])

        # if we are at the very first item, use the empty bin already open
        else:
            b = 0

        # put it in
        if isinstance(items, dict):
            bins[b][key] = weight
        else:
            bins[b].append(weight)

        # increase weight sum of the bin and continue with
        # next item
        weight_sum[b] += weight

    if not is_tuple_list:
        return bins
    new_bins = []
    for bin_idx in range(len(bins)):
        new_bins.append([])
        for _key in bins[bin_idx]:
            new_bins[bin_idx].append(new_dict[_key])
    return new_bins


def measure_model_memory_forward(state: SimState, model: ModelInterface) -> float:
    """Measure peak GPU memory usage during a model's forward pass.

    Clears GPU cache, runs a forward pass with the provided state, and measures
    the maximum memory allocated during execution. This function helps determine
    the actual GPU memory requirements for processing a simulation state.

    Args:
        state (SimState): Input state to pass to the model, with shape information
            determined by the specific SimState instance.
        model (ModelInterface): Model to measure memory usage for, implementing
            the ModelInterface protocol.

    Returns:
        float: Peak memory usage in gigabytes.

    Raises:
        ValueError: If the model device is CPU, as memory estimation is only
            meaningful for GPU-based models.

    Notes:
        This function performs a synchronization and cache clearing operation
        before measurement, which may impact performance if called frequently.
    """
    # TODO: Make it cleaner
    # assert model device is not cpu
    if (isinstance(model.device, str) and model.device == "cpu") or (
        isinstance(model.device, torch.device) and model.device.type == "cpu"
    ):
        raise ValueError(
            "Memory estimation does not make sense on CPU and is unsupported."
        )

    print(  # noqa: T201
        "Model Memory Estimation: Running forward pass on state with "
        f"{state.n_atoms} atoms and {state.n_systems} systems.",
    )
    # Clear GPU memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    model(state)

    return torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB


def determine_max_batch_size(
    state: SimState,
    model: ModelInterface,
    max_atoms: int = 500_000,
    start_size: int = 1,
    scale_factor: float = 1.6,
    oom_error_message: str | list[str] = "CUDA out of memory",
) -> int:
    """Determine maximum batch size that fits in GPU memory.

    Uses a geometric sequence to efficiently search for the largest number of
    batches that can be processed without running out of GPU memory. This function
    incrementally tests larger batch sizes until it encounters an out-of-memory
    error or reaches the specified maximum atom count.

    Args:
        state (SimState): State to replicate for testing.
        model (ModelInterface): Model to test with.
        max_atoms (int): Upper limit on number of atoms to try (for safety).
            Defaults to 500,000.
        start_size (int): Initial batch size to test. Defaults to 1.
        scale_factor (float): Factor to multiply batch size by in each iteration.
            Defaults to 1.6.
        oom_error_message (str | list[str]): String or list of strings to match in
            RuntimeError messages to identify out-of-memory errors. Defaults to
            "CUDA out of memory".

    Returns:
        int: Maximum number of batches that fit in GPU memory.

    Raises:
        RuntimeError: If a RuntimeError occurs that doesn't match any of the
            specified OOM error messages.

    Example::

        # Find the maximum batch size for a Lennard-Jones model
        max_batches = determine_max_batch_size(
            state=sample_state, model=lj_model, max_atoms=100_000
        )

    Notes:
        The function returns a batch size slightly smaller than the actual maximum
        (with a safety margin) to avoid operating too close to memory limits.
    """
    # Convert oom_error_message to list if it's a string
    if isinstance(oom_error_message, str):
        oom_error_message = [oom_error_message]

    # Create a geometric sequence of batch sizes
    sizes = [start_size]
    while (
        next_size := max(round(sizes[-1] * scale_factor), sizes[-1] + 1)
    ) * state.n_atoms <= max_atoms:
        sizes.append(next_size)

    for sys_idx in range(len(sizes)):
        n_systems = sizes[sys_idx]
        concat_state = ts.concatenate_states([state] * n_systems)

        try:
            measure_model_memory_forward(concat_state, model)
        except Exception as exc:
            exc_str = str(exc)
            # Check if any of the OOM error messages match
            for msg in oom_error_message:
                if msg in exc_str:
                    return sizes[max(0, sys_idx - 2)]

            # No OOM message matched - re-raise the error
            raise

    return sizes[-1]


def calculate_memory_scalers(
    state: SimState,
    memory_scales_with: MemoryScaling = "n_atoms_x_density",
) -> list[float]:
    """Calculate a metric that estimates memory requirements for each system in a state.

    Provides different scaling metrics that correlate with memory usage.
    Models with radial neighbor cutoffs generally scale with "n_atoms_x_density",
    while models with a fixed number of neighbors scale with "n_atoms".
    The choice of metric can significantly impact the accuracy of memory requirement
    estimations for different types of simulation systems.

    Uses vectorized operations for batched periodic states and ``state[i]``
    indexing for non-periodic systems so no eager split is needed.

    Args:
        state (SimState): State to calculate metric for, with shape information
            specific to the SimState instance.
        memory_scales_with ("n_atoms_x_density" | "n_atoms"): Type of metric
            to use. "n_atoms" uses only atom count and is suitable for models that
            have a fixed number of neighbors. "n_atoms_x_density" uses atom count
            multiplied by number density and is better for models with radial cutoffs
            Defaults to "n_atoms_x_density".

    Returns:
        list[float]: Calculated metric value for each system.

    Raises:
        ValueError: If an invalid metric type is provided.

    Example::

        # Calculate memory scaling factor based on atom count
        metrics = calculate_memory_scalers(state, memory_scales_with="n_atoms")

        # Calculate memory scaling factor based on atom count and density
        metrics = calculate_memory_scalers(state, memory_scales_with="n_atoms_x_density")
    """
    if memory_scales_with == "n_atoms":
        return state.n_atoms_per_system.tolist()
    if memory_scales_with == "n_atoms_x_density":
        if state.n_systems > 1 and state.pbc.all().item():  # vectorized path
            n_atoms = state.n_atoms_per_system.to(state.volume.dtype)
            volume = torch.abs(state.volume) / 1000  # A^3 -> nm^3
            return torch.where(volume > 0, n_atoms * n_atoms / volume, n_atoms).tolist()
        # per-system path (non-periodic or single system)
        scalers = []
        for i in range(state.n_systems):
            s = state[i]
            if all(s.pbc):
                volume = torch.abs(torch.linalg.det(s.cell[0])) / 1000
            else:
                bbox = s.positions.max(dim=0).values - s.positions.min(dim=0).values
                for j, periodic in enumerate(s.pbc):
                    if not periodic:
                        bbox[j] += 2.0
                volume = bbox.prod() / 1000
            scalers.append(s.n_atoms * (s.n_atoms / volume.item()))
        return scalers
    raise ValueError(
        f"Invalid metric: {memory_scales_with}, must be one of {get_args(MemoryScaling)}"
    )


def estimate_max_memory_scaler(
    state: SimState,
    model: ModelInterface,
    metric_values: list[float] | torch.Tensor,
    **kwargs: Any,
) -> float:
    """Estimate maximum memory scaling metric that fits in GPU memory.

    Tests both minimum and maximum metric states to determine a safe upper bound
    for the memory scaling metric. This approach ensures the estimated value works
    for both small, dense systems and large, sparse systems.

    Args:
        state (SimState): Batched state containing all systems. Individual
            systems are accessed via ``state[idx]`` (integer indexing), so only
            the two extreme states are materialized.
        model (ModelInterface): Model to test with, implementing the ModelInterface
            protocol.
        metric_values (list[float]): Corresponding metric values for each state,
            as calculated by calculate_memory_scalers().
        **kwargs: Additional keyword arguments passed to determine_max_batch_size.

    Returns:
        float: Maximum safe metric value that fits in GPU memory.

    Example::

        # Calculate metrics for a set of states
        metrics = calculate_memory_scalers(state, memory_scales_with="n_atoms")

        # Estimate maximum safe metric value
        max_metric = estimate_max_memory_scaler(state, model, metrics)

    Notes:
        This function tests batch sizes with both the smallest and largest systems
        to find a conservative estimate that works across varying system sizes.
        The returned value will be the minimum of the two estimates.
    """
    metric_values = torch.tensor(metric_values)

    # select one state with the min n_atoms
    min_metric = metric_values.min()
    max_metric = metric_values.max()

    min_state = state[int(metric_values.argmin())]
    max_state = state[int(metric_values.argmax())]

    print(  # noqa: T201
        "Model Memory Estimation: Estimating memory from worst case of "
        f"largest and smallest system. Largest system has {max_state.n_atoms} atoms "
        f"and {max_state.n_systems} batches, and smallest system has "
        f"{min_state.n_atoms} atoms and {min_state.n_systems} batches.",
    )
    min_state_max_batches = determine_max_batch_size(min_state, model, **kwargs)
    max_state_max_batches = determine_max_batch_size(max_state, model, **kwargs)

    return min(
        min_state_max_batches * min_metric.item(),
        max_state_max_batches * max_metric.item(),
    )


class BinningAutoBatcher[T: SimState]:
    """Batcher that groups states into bins of similar computational cost.

    Divides a collection of states into batches that can be processed efficiently
    without exceeding GPU memory. States are grouped based on a memory scaling
    metric to maximize GPU utilization. This approach is ideal for scenarios where
    all states need to be evolved the same number of steps.

    To avoid a slow memory estimation step, set the `max_memory_scaler` to a
    known value.

    Attributes:
        model (ModelInterface): Model used for memory estimation and processing.
        memory_scales_with (str): Metric type used for memory estimation.
        max_memory_scaler (float): Maximum memory metric allowed per system.
        max_atoms_to_try (int): Maximum number of atoms to try when estimating memory.
        memory_scalers (list[float]): Memory scaling metrics for each state.
        index_to_scaler (dict): Mapping from state index to its scaling metric.
        index_bins (list[list[int]]): Groups of state indices that can be batched
            together.
        batched_states (list[list[SimState]]): Grouped states ready for batching.
        current_state_bin (int): Index of the current batch being processed.

    Example::

        # Create a batcher with a Lennard-Jones model
        batcher = BinningAutoBatcher(
            model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=1000.0
        )

        # Load states and process them in batches
        batcher.load_states(states)
        final_states = []
        for batch, _indices in batcher:
            final_states.append(evolve_batch(batch))

        # Restore original order
        ordered_final_states = batcher.restore_original_order(final_states)
    """

    def __init__(
        self,
        model: ModelInterface,
        *,
        memory_scales_with: MemoryScaling = "n_atoms_x_density",
        max_memory_scaler: float | None = None,
        max_atoms_to_try: int = 500_000,
        memory_scaling_factor: float = 1.6,
        max_memory_padding: float = 1.0,
        oom_error_message: str | list[str] = "CUDA out of memory",
    ) -> None:
        """Initialize the binning auto-batcher.

        Args:
            model (ModelInterface): Model to batch for, used to estimate memory
                requirements.
            memory_scales_with ("n_atoms" | "n_atoms_x_density"): Metric to use
                for estimating memory requirements:
                - "n_atoms": Uses only atom count
                - "n_atoms_x_density": Uses atom count multiplied by number density
                Defaults to "n_atoms_x_density".
            max_memory_scaler (float | None): Maximum metric value allowed per system. If
                None, will be automatically estimated. Defaults to None.
            max_atoms_to_try (int): Maximum number of atoms to try when estimating
                max_memory_scaler. Defaults to 500,000.
            memory_scaling_factor (float): Factor to multiply batch size by in each
                iteration. Larger values will get a batch size more quickly, smaller
                values will get a more accurate limit. Must be greater than 1. Defaults
                to 1.6.
            max_memory_padding (float): Multiply the auto-determined max_memory_scaler
                by this value to account for fluctuations in max memory. Defaults to 1.0.
            oom_error_message (str | list[str]): String or list of strings to match in
                RuntimeError messages to identify out-of-memory errors. Defaults to
                "CUDA out of memory".
        """
        self.max_memory_scaler = max_memory_scaler
        self.max_atoms_to_try = max_atoms_to_try
        self.memory_scales_with = memory_scales_with
        self.model = model
        self.memory_scaling_factor = memory_scaling_factor
        self.max_memory_padding = max_memory_padding
        self.oom_error_message = oom_error_message

    def load_states(self, states: T | Sequence[T]) -> float:
        """Load new states into the batcher.

        Processes the input states, computes memory scaling metrics for each,
        and organizes them into optimal batches using a bin-packing algorithm
        to maximize GPU utilization.

        Args:
            states (SimState | list[SimState]): Collection of states to batch. Either a
                list of individual SimState objects or a single batched SimState that
                will be split into individual states. Each SimState has shape
                information specific to its instance.

        Returns:
            float: Maximum memory scaling metric that fits in GPU memory.

        Raises:
            ValueError: If any individual state has a memory scaling metric greater
                than the maximum allowed value.

        Example::

            # Load individual states
            batcher.load_states([state1, state2, state3])

            # Or load a batched state that will be split
            batcher.load_states(batched_state)

        Notes:
            This method resets the current state bin index, so any ongoing iteration
            will be restarted when this method is called.
        """
        batched = (
            states if isinstance(states, SimState) else ts.concatenate_states(states)
        )
        self.memory_scalers = calculate_memory_scalers(batched, self.memory_scales_with)
        if not self.max_memory_scaler:
            self.max_memory_scaler = estimate_max_memory_scaler(
                batched,
                self.model,
                self.memory_scalers,
                max_atoms=self.max_atoms_to_try,
                scale_factor=self.memory_scaling_factor,
                oom_error_message=self.oom_error_message,
            )
            self.max_memory_scaler = self.max_memory_scaler * self.max_memory_padding

        # verify that no systems are too large
        max_metric_value = max(self.memory_scalers)
        max_metric_idx = self.memory_scalers.index(max_metric_value)
        if max_metric_value > self.max_memory_scaler:
            raise ValueError(
                f"Max metric of system with index {max_metric_idx} in states: "
                f"{max(self.memory_scalers)} is greater than max_metric "
                f"{self.max_memory_scaler}, please set a larger max_metric "
                f"or run smaller systems metric."
            )

        self.index_to_scaler = dict(enumerate(self.memory_scalers))
        self.index_bins = to_constant_volume_bins(
            self.index_to_scaler, max_volume=self.max_memory_scaler
        )  # list[dict[original_index: int, memory_scale:float]]
        # Convert to list of lists of indices
        self.index_bins = [list(batch.keys()) for batch in self.index_bins]
        self.batched_states = [[batched[index_bin]] for index_bin in self.index_bins]
        self.current_state_bin = 0

        return self.max_memory_scaler

    def next_batch(self) -> tuple[T | None, list[int]]:
        """Get the next batch of states.

        Returns batches sequentially until all states have been processed. Each batch
        contains states grouped together to maximize GPU utilization without exceeding
        memory constraints.

        Returns:
            tuple[T | None, list[int]]: A tuple containing:
                - A concatenated SimState containing the next batch of states,
                  or None if no more batches
                - List of indices of states in the current batch

        Example::

            # Get batches one by one
            for batch, indices in batcher:
                process_batch(batch)

        """
        # TODO: need to think about how this intersects with reporting too
        # TODO: definitely a clever treatment to be done with iterators here
        if self.current_state_bin < len(self.batched_states):
            state_bin = self.batched_states[self.current_state_bin]
            state = ts.concatenate_states(state_bin)
            indices = (
                self.index_bins[self.current_state_bin]
                if self.current_state_bin < len(self.index_bins)
                else []
            )
            self.current_state_bin += 1
            return state, indices
        return None, []

    def __iter__(self) -> Iterator[tuple[T, list[int]]]:
        """Return self as an iterator.

        Allows using the batcher in a for loop to iterate through all batches.
        Resets the current state bin index to start iteration from the beginning.

        Returns:
            Iterator[tuple[T, list[int]]]: Self as an iterator.

        Example::

            # Iterate through all batches
            for batch, indices in batcher:
                process_batch(batch)
        """
        return self

    def __next__(self) -> tuple[T, list[int]]:
        """Get the next batch for iteration.

        Implements the iterator protocol to allow using the batcher in a for loop.

        Returns:
            tuple[T, list[int]]: The next batch of states and their indices.

        Raises:
            StopIteration: When there are no more batches.
        """
        next_batch, indices = self.next_batch()
        if next_batch is None:
            raise StopIteration
        return next_batch, indices

    def restore_original_order(self, batched_states: Sequence[T]) -> list[T]:
        """Reorder processed states back to their original sequence.

        Takes states that were processed in batches and restores them to the
        original order they were provided in. This is essential after batch
        processing to ensure results correspond to the input states.

        Args:
            batched_states (Sequence[SimState]): State batches to reorder. These can be
                either concatenated batch states that will be split, or already
                split individual states.

        Returns:
            list[SimState]: States in their original order, with shape information
                matching the original input states.

        Raises:
            ValueError: If the number of states doesn't match the number of
                original indices.

        Example::

            # Process batches and restore original order
            results = []
            for batch, _indices in batcher:
                results.append(process_batch(batch))
            ordered_results = batcher.restore_original_order(results)

        """
        all_states = [
            state[i] for state in batched_states for i in range(state.n_systems)
        ]
        original_indices = list(chain.from_iterable(self.index_bins))

        if len(all_states) != len(original_indices):
            raise ValueError(
                f"Number of states ({len(all_states)}) does not match "
                f"number of original indices ({len(original_indices)})"
            )

        # sort states by original indices
        indexed_states = list(zip(original_indices, all_states, strict=True))
        return [state for _, state in sorted(indexed_states, key=lambda x: x[0])]


class InFlightAutoBatcher[T: SimState]:
    """Batcher that dynamically swaps states based on convergence.

    Optimizes GPU utilization by removing converged states from the batch and
    adding new states to process. This approach is ideal for iterative processes
    where different states may converge at different rates, such as geometry
    optimization.

    To avoid a slow memory estimation step, set the `max_memory_scaler` to a
    known value.

    .. image:: https://github.com/janosh/diagrams/raw/main/assets/in-flight-auto-batcher/in-flight-auto-batcher.svg
       :alt: In-flight auto-batcher diagram

    Attributes:
        model (ModelInterface): Model used for memory estimation and processing.
        memory_scales_with (str): Metric type used for memory estimation.
        max_memory_scaler (float): Maximum memory metric allowed per system.
        max_atoms_to_try (int): Maximum number of atoms to try when estimating memory.
        max_iterations (int | None): Maximum number of iterations per state.
        memory_scalers (list[float]): Memory scaling metrics for each state.
        current_idx (list[int]): Indices of states in the current batch.
        completed_idx (list[int]): Indices of states that have been processed.
        completed_idx_og_order (list[int]): Original indices of completed states.
        current_scalers (list[float]): Memory metrics for states in current batch.
        iteration_count (dict[int, int]): Number of iterations for each state.

    Example::

        # Create a hot-swapping batcher
        batcher = InFlightAutoBatcher(
            model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=1000.0
        )

        # Load states and process them with convergence checking
        batcher.load_states(states)
        batch, completed_states = batcher.next_batch(None, None)

        while batch is not None:
            # Process the batch
            batch = process_batch(batch)

            # Check convergence
            convergence = check_convergence(batch)

            # Get next batch, with converged states swapped out
            batch, new_completed = batcher.next_batch(batch, convergence)
            completed_states.extend(new_completed)

        # Restore original order
        ordered_results = batcher.restore_original_order(completed_states)
    """

    def __init__(
        self,
        model: ModelInterface,
        *,
        memory_scales_with: MemoryScaling = "n_atoms_x_density",
        max_memory_scaler: float | None = None,
        max_atoms_to_try: int = 500_000,
        memory_scaling_factor: float = 1.6,
        max_iterations: int | None = None,
        max_memory_padding: float = 1.0,
        oom_error_message: str | list[str] = "CUDA out of memory",
    ) -> None:
        """Initialize the hot-swapping auto-batcher.

        Args:
            model (ModelInterface): Model to batch for, used to estimate memory
                requirements.
            memory_scales_with ("n_atoms" | "n_atoms_x_density"): Metric to use
                for estimating memory requirements:
                - "n_atoms": Uses only atom count
                - "n_atoms_x_density": Uses atom count multiplied by number density
                Defaults to "n_atoms_x_density".
            max_memory_scaler (float | None): Maximum metric value allowed per system.
                If None, will be automatically estimated. Defaults to None.
            max_atoms_to_try (int): Maximum number of atoms to try when estimating
                max_memory_scaler. Defaults to 500,000.
            memory_scaling_factor (float): Factor to multiply batch size by in each
                iteration. Larger values will get a batch size more quickly, smaller
                values will get a more accurate limit. Must be greater than 1. Defaults
                to 1.6.
            max_iterations (int | None): Maximum number of iterations to process a state
                before considering it complete, regardless of convergence. Used to prevent
                infinite loops. Defaults to None (no limit).
            max_memory_padding (float): Multiply the auto-determined max_memory_scaler
                by this value to account for fluctuations in max memory. Defaults to 1.0.
            oom_error_message (str | list[str]): String or list of strings to match in
                RuntimeError messages to identify out-of-memory errors. Defaults to
                "CUDA out of memory".
        """
        self.model = model
        self.memory_scales_with = memory_scales_with
        self.max_memory_scaler = max_memory_scaler or None
        self.max_atoms_to_try = max_atoms_to_try
        self.memory_scaling_factor = memory_scaling_factor
        self.max_iterations = max_iterations
        self.max_memory_padding = max_memory_padding
        self.oom_error_message = oom_error_message

    def load_states(self, states: Sequence[T] | Iterator[T] | T) -> None:
        """Load new states into the batcher.

        Processes the input states, computes memory scaling metrics for each,
        and prepares them for dynamic batching based on convergence criteria.
        Unlike BinningAutoBatcher, this doesn't create fixed batches upfront.

        Args:
            states (list[SimState] | Iterator[SimState] | SimState): Collection of
                states to batch. Can be a list of individual SimState objects, an
                iterator yielding SimState objects, or a single batched SimState
                that will be split into individual states. Each SimState has shape
                information specific to its instance.

        Raises:
            ValueError: If any individual state has a memory scaling metric greater
                than the maximum allowed value.

        Example::

            # Load individual states
            batcher.load_states([state1, state2, state3])

            # Or load a batched state that will be split
            batcher.load_states(batched_state)

            # Or load states from an iterator
            batcher.load_states(state_generator())

        Notes:
            This method resets the current state indices and completed state tracking,
            so any ongoing processing will be restarted when this method is called.
        """
        if isinstance(states, SimState):
            states = states.split()
        if isinstance(states, list | tuple):
            states = iter(states)

        self.states_iterator = states

        self.current_scalers = []
        self.current_idx = []
        self.iterator_idx = 0
        self.iteration_count = []  # Track attempts for each state

        self.completed_idx_og_order = []

        self.first_batch_returned = False
        self._first_batch = self._get_first_batch()
        return self.max_memory_scaler

    def _get_next_states(self) -> list[T]:
        """Add states from the iterator until max_memory_scaler is reached.

        Pulls states from the iterator and adds them to the current batch until
        adding another would exceed the maximum memory scaling metric.

        Returns:
            list[SimState]: new states added to the batch.
        """
        new_metrics: list[float] = []
        new_idx: list[int] = []
        new_states: list[T] = []
        for state in self.states_iterator:
            metric = calculate_memory_scalers(state, self.memory_scales_with)[0]
            if metric > self.max_memory_scaler:
                raise ValueError(
                    f"State {metric=} is greater than max_metric {self.max_memory_scaler}"
                    ", please set a larger max_metric or run smaller systems metric."
                )
            if (
                sum(self.current_scalers) + sum(new_metrics) + metric
                > self.max_memory_scaler
            ):
                # put the state back in the iterator
                self.states_iterator = chain([state], self.states_iterator)
                break

            new_metrics.append(metric)
            new_idx.append(self.iterator_idx)
            new_states.append(state)
            # Initialize attempt counter for new state
            self.iteration_count.append(0)
            self.iterator_idx += 1

        self.current_scalers.extend(new_metrics)
        self.current_idx.extend(new_idx)

        return new_states

    def _delete_old_states(self, completed_idx: list[int]) -> None:
        """Remove completed states from tracking lists.

        Updates internal tracking of states and their metrics when states are
        completed and removed from processing.

        Args:
            completed_idx: Indices of completed states to remove.
        """
        # Sort in descending order to avoid index shifting problems
        completed_idx.sort(reverse=True)

        # update state tracking lists
        for idx in completed_idx:
            og_idx = self.current_idx.pop(idx)
            self.current_scalers.pop(idx)
            self.completed_idx_og_order.append(og_idx)

    def _get_first_batch(self) -> T:
        """Create and return the first batch of states.

        Initializes the batcher by estimating memory requirements if needed
        and creating the first batch of states to process.

        Returns:
            T: first batch of states.
        """
        # we need to sample a state and use it to estimate the max metric
        # for the first batch
        first_state = next(self.states_iterator)
        first_metric = calculate_memory_scalers(first_state, self.memory_scales_with)[0]
        self.current_scalers += [first_metric]
        self.current_idx += [0]
        self.iteration_count.append(0)  # Initialize attempt counter for first state
        self.iterator_idx += 1

        # if max_metric is not set, estimate it
        has_max_metric = bool(self.max_memory_scaler)
        if not has_max_metric:
            n_systems = determine_max_batch_size(
                first_state,
                self.model,
                max_atoms=self.max_atoms_to_try,
                scale_factor=self.memory_scaling_factor,
                oom_error_message=self.oom_error_message,
            )
            self.max_memory_scaler = n_systems * first_metric * 0.8

        states = self._get_next_states()
        all_states = [first_state, *states]

        if not has_max_metric:
            concatenated_state = ts.concatenate_states(all_states)
            self.max_memory_scaler = estimate_max_memory_scaler(
                concatenated_state,
                self.model,
                self.current_scalers,
                max_atoms=self.max_atoms_to_try,
                scale_factor=self.memory_scaling_factor,
                oom_error_message=self.oom_error_message,
            )
            self.max_memory_scaler = self.max_memory_scaler * self.max_memory_padding
            newer_states = self._get_next_states()
            all_states.extend(newer_states)

        return ts.concatenate_states(all_states)

    def next_batch(  # noqa: C901
        self, updated_state: T | None, convergence_tensor: torch.Tensor | None
    ) -> tuple[T, list[T]]:
        """Get the next batch of states based on convergence.

        Removes converged states from the batch, adds new states if possible,
        and returns both the updated batch and the completed states. This method
        implements the core dynamic batching strategy of the InFlightAutoBatcher.

        Args:
            updated_state (SimState | None): Current state after processing, or None
                for the first call. Contains shape information specific to the SimState
                instance.
            convergence_tensor (torch.Tensor | None): Boolean tensor with shape
                [n_systems] indicating which states have converged (True) or not
                (False). Should be None only for the first call.

        Returns:
            tuple[SimState | None, list[SimState]]: (next_batch, completed_states)
                where next_batch is a SimState or None if all states are processed,
                and completed_states is a list of SimState objects.

        Raises:
            AssertionError: If convergence_tensor doesn't match the expected shape or
                if other validation checks fail.

        Example::

            # Initial call
            batch, completed = batcher.next_batch(None, None)

            # Process batch and check for convergence
            batch = process_batch(batch)
            convergence = check_convergence(batch)

            # Get next batch with converged states removed and new states added
            batch, completed = batcher.next_batch(batch, convergence)

        Notes:
            When max_iterations is set, states that exceed this limit will be
            forcibly marked as converged regardless of their actual convergence state.
        """
        if not self.first_batch_returned:
            self.first_batch_returned = True
            return self._first_batch, []

        if (
            convergence_tensor is None or updated_state is None
        ) and self.first_batch_returned:
            raise ValueError(
                "A convergence tensor must be provided after the "
                "first batch has been run."
            )

        # assert statements helpful for debugging, should be moved to validate fn
        # the first two are most important
        if updated_state is None:
            raise ValueError("updated_state cannot be None")
        if convergence_tensor is None:
            raise ValueError("convergence_tensor cannot be None")
        if len(convergence_tensor) != updated_state.n_systems:
            raise ValueError(f"{len(convergence_tensor)=} != {updated_state.n_systems=}")
        if len(self.current_idx) != len(self.current_scalers):
            raise ValueError(f"{len(self.current_idx)=} != {len(self.current_scalers)=}")
        if len(convergence_tensor.shape) != 1:
            raise ValueError(f"{len(convergence_tensor.shape)=} != 1")
        if updated_state.n_systems <= 0:
            raise ValueError(f"{updated_state.n_systems=} <= 0")

        # Increment attempt counters and check for max attempts in a single loop
        for cur_idx, abs_idx in enumerate(self.current_idx):
            self.iteration_count[abs_idx] += 1
            if self.max_iterations is not None and (
                self.iteration_count[abs_idx] >= self.max_iterations
            ):
                # Force convergence for states that have reached max attempts
                convergence_tensor[cur_idx] = torch.tensor(True)  # noqa: FBT003

        completed_idx = torch.where(convergence_tensor)[0].tolist()

        completed_states = updated_state.pop(completed_idx)

        # necessary to ensure states that finish at the same time are ordered properly
        completed_states.reverse()
        completed_idx.sort(reverse=True)

        self._delete_old_states(completed_idx)
        next_states = self._get_next_states()

        # there are no states left to run, return the completed states
        if not self.current_idx:
            return None, completed_states  # type: ignore[invalid-return-type]

        # concatenate remaining state with next states
        if updated_state.n_systems > 0:
            next_states = [updated_state, *next_states]
        next_batch = ts.concatenate_states(next_states)

        return next_batch, completed_states

    def restore_original_order(self, completed_states: Sequence[T]) -> list[T]:
        """Reorder completed states back to their original sequence.

        Takes states that were completed in arbitrary order and restores them
        to the original order they were provided in. This is essential after using
        the hot-swapping strategy to ensure results correspond to input states.

        Args:
            completed_states (Sequence[SimState]): Completed states to reorder. Each
                SimState contains simulation data with shape specific to its instance.

        Returns:
            list[SimState]: States in their original order, with shape information
                matching the original input states.

        Raises:
            ValueError: If the number of completed states doesn't match the
                number of completed indices.

        Example::

            # After processing with next_batch
            all_completed_states = []

            # Process all states
            while batch is not None:
                batch = process_batch(batch)
                convergence = check_convergence(batch)
                batch, new_completed = batcher.next_batch(batch, convergence)
                all_completed_states.extend(new_completed)

            # Restore original order
            ordered_results = batcher.restore_original_order(all_completed_states)

        Notes:
            This method should only be called after all states have been processed,
            or you will only get the subset of states that have completed so far.
        """
        # TODO: should act on full states, not state slices
        if len(completed_states) != len(self.completed_idx_og_order):
            raise ValueError(
                f"Number of completed states ({len(completed_states)}) does not match "
                f"number of completed indices ({len(self.completed_idx_og_order)})"
            )

        # Create pairs of (original_index, state)
        indexed_states = list(
            zip(self.completed_idx_og_order, completed_states, strict=True)
        )

        # Sort by original index
        return [state for _, state in sorted(indexed_states, key=lambda x: x[0])]
