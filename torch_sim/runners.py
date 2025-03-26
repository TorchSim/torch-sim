"""Simulation runners for molecular dynamics and optimization.

This module provides functions for running molecular dynamics simulations and geometry
optimizations using various models and integrators. It includes utilities for
converting between different atomistic representations and handling simulation state.
"""

import warnings
from collections.abc import Callable
from itertools import chain

import torch
from numpy.typing import ArrayLike

from torch_sim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher
from torch_sim.models.interface import ModelInterface
from torch_sim.quantities import batchwise_max_force
from torch_sim.state import SimState, StateLike, concatenate_states, initialize_state
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.units import UnitSystem


def _configure_batches_iterator(
    model: ModelInterface,
    state: SimState,
    autobatcher: ChunkingAutoBatcher | bool,
) -> ChunkingAutoBatcher:
    """Create a batches iterator for the integrate function.

    Args:
        model (ModelInterface): The model to use for the integration
        state (SimState): The state to use for the integration
        autobatcher (ChunkingAutoBatcher | bool): The autobatcher to use for integration

    Returns:
        A batches iterator
    """
    # load and properly configure the autobatcher
    if autobatcher is True:
        autobatcher = ChunkingAutoBatcher(
            model=model,
            return_indices=True,
        )
        autobatcher.load_states(state)
        batches = autobatcher
    elif isinstance(autobatcher, ChunkingAutoBatcher):
        autobatcher.load_states(state)
        autobatcher.return_indices = True
        batches = autobatcher
    elif autobatcher is False:
        batches = [(state, [])]
    else:
        raise ValueError(
            f"Invalid autobatcher type: {type(autobatcher).__name__}, "
            "must be bool or ChunkingAutoBatcher."
        )
    return batches


def integrate(
    system: StateLike,
    model: ModelInterface,
    *,
    integrator: Callable,
    n_steps: int,
    temperature: float | ArrayLike,
    timestep: float,
    unit_system: UnitSystem = UnitSystem.metal,
    trajectory_reporter: TrajectoryReporter | None = None,
    autobatcher: ChunkingAutoBatcher | bool = False,
    **integrator_kwargs: dict,
) -> SimState:
    """Simulate a system using a model and integrator.

    Args:
        system (StateLike): Input system to simulate
        model (ModelInterface): Neural network model module
        integrator (Callable): Integration algorithm function
        n_steps (int): Number of integration steps
        temperature (float | ArrayLike): Temperature or array of temperatures for each
            step
        timestep (float): Integration time step
        unit_system (UnitSystem): Unit system for temperature and time
        integrator_kwargs: Additional keyword arguments for integrator
        trajectory_reporter (TrajectoryReporter | None): Optional reporter for tracking
            trajectory.
        autobatcher (ChunkingAutoBatcher | bool): Optional autobatcher to use
        **integrator_kwargs: Additional keyword arguments for integrator init function

    Returns:
        SimState: Final state after integration
    """
    # create a list of temperatures
    temps = temperature if hasattr(temperature, "__iter__") else [temperature] * n_steps
    if len(temps) != n_steps:
        raise ValueError(
            f"len(temperature) = {len(temps)}. It must equal n_steps = {n_steps}"
        )

    # initialize the state
    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    init_fn, update_fn = integrator(
        model=model,
        kT=torch.tensor(temps[0] * unit_system.temperature, dtype=dtype, device=device),
        dt=torch.tensor(timestep * unit_system.time, dtype=dtype, device=device),
        **integrator_kwargs,
    )
    state = init_fn(state)

    batch_iterator = _configure_batches_iterator(model, state, autobatcher)

    final_states: list[SimState] = []
    og_filenames = trajectory_reporter.filenames if trajectory_reporter else None
    for state, batch_indices in batch_iterator:
        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[og_filenames[i] for i in batch_indices]
            )

        # run the simulation
        for step in range(1, n_steps + 1):
            state = update_fn(state, kT=temps[step - 1] * unit_system.temperature)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)

        # finish the trajectory reporter
        final_states.append(state)

    if trajectory_reporter:
        trajectory_reporter.finish()

    if isinstance(batch_iterator, ChunkingAutoBatcher):
        reordered_states = batch_iterator.restore_original_order(final_states)
        return concatenate_states(reordered_states)

    return state


def _configure_hot_swapping_autobatcher(
    model: ModelInterface,
    state: SimState,
    autobatcher: HotSwappingAutoBatcher | bool,
    max_attempts: int,  # TODO: change name to max_iterations
) -> HotSwappingAutoBatcher:
    """Configure the hot swapping autobatcher for the optimize function.

    Args:
        model (ModelInterface): The model to use for the autobatcher
        state (SimState): The state to use for the autobatcher
        autobatcher (HotSwappingAutoBatcher | bool): The autobatcher to use for the
            autobatcher
        max_attempts (int): The maximum number of attempts for the autobatcher

    Returns:
        A hot swapping autobatcher
    """
    # load and properly configure the autobatcher
    if isinstance(autobatcher, HotSwappingAutoBatcher):
        autobatcher.return_indices = True
        autobatcher.max_attempts = max_attempts
        autobatcher.load_states(state)
    else:
        if autobatcher:
            memory_scales_with = model.memory_scales_with
            max_memory_scaler = None
        else:
            memory_scales_with = "n_atoms"
            max_memory_scaler = state.n_atoms + 1
        autobatcher = HotSwappingAutoBatcher(
            model=model,
            return_indices=True,
            max_memory_scaler=max_memory_scaler,
            memory_scales_with=memory_scales_with,
            max_iterations=max_attempts,
        )
        autobatcher.load_states(state)
    return autobatcher


def generate_force_convergence_fn(force_tol: float = 1e-1) -> Callable:
    """Generate a convergence function for the convergence_fn argument
    of the optimize function.

    Args:
        force_tol (float): Force tolerance for convergence

    Returns:
        Convergence function that takes a state and last energy and
        returns a batchwise boolean function
    """

    def convergence_fn(
        state: SimState,
        last_energy: torch.Tensor,  # noqa: ARG001
    ) -> bool:
        """Check if the system has converged."""
        return batchwise_max_force(state) < force_tol

    return convergence_fn


def optimize(
    system: StateLike,
    model: ModelInterface,
    *,
    optimizer: Callable,
    convergence_fn: Callable | None = None,
    unit_system: UnitSystem = UnitSystem.metal,
    trajectory_reporter: TrajectoryReporter | None = None,
    autobatcher: HotSwappingAutoBatcher | bool = False,
    max_steps: int = 10_000,
    steps_between_swaps: int = 5,
    **optimizer_kwargs: dict,
) -> SimState:
    """Optimize a system using a model and optimizer.

    Args:
        system (StateLike): Input system to optimize (ASE Atoms, Pymatgen Structure, or
            SimState)
        model (ModelInterface): Neural network model module
        optimizer (Callable): Optimization algorithm function
        convergence_fn (Callable | None): Condition for convergence, should return a
            boolean tensor of length n_batches
        unit_system (UnitSystem): Unit system for energy tolerance
        optimizer_kwargs: Additional keyword arguments for optimizer init function
        trajectory_reporter (TrajectoryReporter | None): Optional reporter for tracking
            optimization trajectory
        autobatcher (HotSwappingAutoBatcher | bool): Optional autobatcher to use. If
            False, the system will assume
            infinite memory and will not batch, but will still remove converged
            structures from the batch. If True, the system will estimate the memory
            available and batch accordingly. If a HotSwappingAutoBatcher, the system
            will use the provided autobatcher, but will reset the max_attempts to
            max_steps // steps_between_swaps.
        max_steps (int): Maximum number of total optimization steps
        steps_between_swaps: Number of steps to take before checking convergence
            and swapping out states.

    Returns:
        Optimized system state
    """
    # create a default convergence function if one is not provided
    # TODO: document this behavior
    if convergence_fn is None:

        def convergence_fn(state: SimState, last_energy: torch.Tensor) -> bool:
            return last_energy - state.energy < 1e-6 * unit_system.energy

    # initialize the state
    state: SimState = initialize_state(system, model.device, model.dtype)
    init_fn, update_fn = optimizer(model=model, **optimizer_kwargs)
    state = init_fn(state)

    max_attempts = max_steps // steps_between_swaps
    autobatcher = _configure_hot_swapping_autobatcher(
        model, state, autobatcher, max_attempts
    )

    step: int = 1
    last_energy = None
    all_converged_states, convergence_tensor = [], None
    og_filenames = trajectory_reporter.filenames if trajectory_reporter else None
    while (result := autobatcher.next_batch(state, convergence_tensor))[0] is not None:
        state, converged_states, batch_indices = result
        all_converged_states.extend(converged_states)

        # need to update the trajectory reporter if any states have converged
        if trajectory_reporter and (step == 1 or len(converged_states) > 0):
            trajectory_reporter.load_new_trajectories(
                filenames=[og_filenames[i] for i in batch_indices]
            )

        for _step in range(steps_between_swaps):
            last_energy = state.energy

            state = update_fn(state)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)
            step += 1
            if step > max_steps:
                # TODO: max steps should be tracked for each structure in the batch
                warnings.warn(f"Optimize has reached max steps: {step}", stacklevel=2)
                break

        convergence_tensor = convergence_fn(state, last_energy)

    all_converged_states.extend(result[1])

    if trajectory_reporter:
        trajectory_reporter.finish()

    if autobatcher:
        final_states = autobatcher.restore_original_order(all_converged_states)
        return concatenate_states(final_states)

    return state


def static(
    system: StateLike,
    model: ModelInterface,
    *,
    unit_system: UnitSystem = UnitSystem.metal,  # noqa: ARG001
    trajectory_reporter: TrajectoryReporter | None = None,
    autobatcher: ChunkingAutoBatcher | bool = False,
    variable_atomic_numbers: bool = True,
    save_state: bool = True,
) -> list[dict[str, torch.Tensor]]:
    """Run single point calculations on a batch of systems.

    Args:
        system (StateLike): Input system to calculate properties for
        model (ModelInterface): Neural network model module
        unit_system (UnitSystem): Unit system for energy and forces
        trajectory_reporter (TrajectoryReporter | None): Optional reporter for tracking
            trajectory
        autobatcher (ChunkingAutoBatcher | bool): Optional autobatcher to use for batching
            calculations
        variable_atomic_numbers (bool): Whether atomic numbers vary between frames
        save_state (bool): Whether to save the state to the trajectory file

    Returns:
        list[dict[str, torch.Tensor]]: Maps of property names to tensors for all batches
    """
    # initialize the state
    state: SimState = initialize_state(system, model.device, model.dtype)

    if trajectory_reporter is None:
        props: dict[str, Callable] = {}
        if model.compute_forces:
            props["forces"] = lambda state: state.forces
        if model.compute_stress:
            props["stress"] = lambda state: state.stress
        # TODO: create temporary files for reporting or consider removing
        # TrajectoryReporter entirely
        filenames = [f"static_{idx}.h5md" for idx in range(state.n_batches)]
        trajectory_reporter = TrajectoryReporter(
            filenames=filenames,
            state_frequency=int(save_state),
            prop_calculators={1: props},
            state_kwargs={
                "save_velocities": False,
                "save_forces": model.compute_forces,
                "variable_atomic_numbers": variable_atomic_numbers,
            },
        )
    else:
        if trajectory_reporter.state_frequency != 1:
            raise ValueError(
                f"{trajectory_reporter.state_frequency=} must be 1 for statics"
            )
        prop_calc_keys = set(trajectory_reporter.prop_calculators)
        if prop_calc_keys != {1}:
            raise ValueError(
                "trajectory_reporter.prop_calculators should only have key=1, got "
                f"{prop_calc_keys}"
            )

    batch_iterator = _configure_batches_iterator(model, state, autobatcher)

    final_states: list[SimState] = []
    all_props: list[dict[str, torch.Tensor]] = []
    og_filenames = trajectory_reporter.filenames
    for substate, batch_indices in batch_iterator:
        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[og_filenames[idx] for idx in batch_indices]
            )

        props = trajectory_reporter.report(substate, 0, model=model)
        all_props.extend(props)

        final_states.append(substate)

    trajectory_reporter.finish()

    if isinstance(batch_iterator, ChunkingAutoBatcher):
        # reorder properties to match original order of states
        original_indices = list(chain.from_iterable(batch_iterator.index_bins))
        return [all_props[idx] for idx in original_indices]

    return all_props
