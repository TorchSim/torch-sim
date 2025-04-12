"""High level runners for atomistic simulations.

This module provides functions for running molecular dynamics simulations and geometry
optimizations using various models and integrators. It includes utilities for
converting between different atomistic representations and handling simulation state.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain

import torch

from torch_sim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher
from torch_sim.models.interface import ModelInterface
from torch_sim.quantities import batchwise_max_force, calc_kinetic_energy, calc_kT
from torch_sim.state import SimState, StateLike, concatenate_states, initialize_state
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.units import UnitSystem


def _configure_reporter(
    trajectory_reporter: TrajectoryReporter | dict | None,
    *,
    state_kwargs: dict | None = None,
    properties: list[str] | None = None,
    prop_frequency: int = 10,
    state_frequency: int = 100,
) -> TrajectoryReporter:
    if trajectory_reporter is None:
        return None
    if isinstance(trajectory_reporter, TrajectoryReporter):
        return trajectory_reporter
    possible_properties = {
        "potential_energy": lambda state: state.energy,
        "forces": lambda state: state.forces,
        "stress": lambda state: state.stress,
        "kinetic_energy": lambda state: calc_kinetic_energy(state.momenta, state.masses),
        "temperature": lambda state: calc_kT(state.momenta, state.masses),
    }

    prop_calculators = {
        prop: calculator
        for prop, calculator in possible_properties.items()
        if prop in properties
    }

    # ordering is important to ensure we can override defaults
    return TrajectoryReporter(
        prop_calculators=trajectory_reporter.pop(
            "prop_calculators", {prop_frequency: prop_calculators}
        ),
        state_frequency=trajectory_reporter.pop("state_frequency", state_frequency),
        state_kwargs=state_kwargs or {},
        **trajectory_reporter,
    )


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
    temperature: float | list | torch.Tensor,
    timestep: float,
    trajectory_reporter: TrajectoryReporter | dict | None = None,
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
        integrator_kwargs: Additional keyword arguments for integrator
        trajectory_reporter (TrajectoryReporter | dict | None): Optional reporter for
            tracking trajectory. If a dict, will be passed to the TrajectoryReporter
            constructor.
        autobatcher (ChunkingAutoBatcher | bool): Optional autobatcher to use
        **integrator_kwargs: Additional keyword arguments for integrator init function

    Returns:
        SimState: Final state after integration
    """
    unit_system = UnitSystem.metal
    # create a list of temperatures
    temps = temperature if hasattr(temperature, "__iter__") else [temperature] * n_steps
    if len(temps) != n_steps:
        raise ValueError(
            f"len(temperature) = {len(temps)}. It must equal n_steps = {n_steps}"
        )

    # initialize the state
    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    kTs = torch.tensor(temps, dtype=dtype, device=device) * unit_system.temperature
    init_fn, update_fn = integrator(
        model=model,
        kT=kTs[0],
        dt=torch.tensor(timestep * unit_system.time, dtype=dtype, device=device),
        **integrator_kwargs,
    )
    # state = init_fn(state)

    batch_iterator = _configure_batches_iterator(model, state, autobatcher)
    trajectory_reporter = _configure_reporter(
        trajectory_reporter,
        properties=["kinetic_energy", "potential_energy", "temperature"],
    )

    final_states: list[SimState] = []
    og_filenames = trajectory_reporter.filenames if trajectory_reporter else None
    for state, batch_indices in batch_iterator:
        state = init_fn(state)

        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[og_filenames[i] for i in batch_indices]
            )

        # run the simulation
        for step in range(1, n_steps + 1):
            state = update_fn(state, kT=kTs[step - 1])

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
    return autobatcher


def _chunked_apply(
    fn: Callable,
    states: SimState,
    model: ModelInterface,
    **batcher_kwargs: dict,
) -> SimState:
    """Apply a function to a state in chunks.

    This prevents us from running out of memory when applying a function to a large
    number of states.

    Args:
        fn (Callable): The function to apply
        states (SimState): The state to apply the function to
        model (ModelInterface): The model to use for the autobatcher
        **batcher_kwargs: Additional keyword arguments for the autobatcher

    Returns:
        A state with the function applied
    """
    autobatcher = ChunkingAutoBatcher(
        model=model,
        return_indices=False,
        **batcher_kwargs,
    )
    autobatcher.load_states(states)
    initialized_states = []

    initialized_states = [fn(batch) for batch in autobatcher]

    ordered_states = autobatcher.restore_original_order(initialized_states)
    return concatenate_states(ordered_states)


def generate_force_convergence_fn(force_tol: float = 1e-1) -> Callable:
    """Generate a force-based convergence function for the convergence_fn argument
    of the optimize function.

    Args:
        force_tol (float): Force tolerance for convergence

    Returns:
        Convergence function that takes a state and last energy and
        returns a batchwise boolean function
    """

    def convergence_fn(
        state: SimState,
        last_energy: torch.Tensor | None = None,  # noqa: ARG001
    ) -> bool:
        """Check if the system has converged."""
        return batchwise_max_force(state) < force_tol

    return convergence_fn


def generate_energy_convergence_fn(energy_tol: float = 1e-3) -> Callable:
    """Generate an energy-based convergence function for the convergence_fn argument
    of the optimize function.

    Args:
        energy_tol (float): Energy tolerance for convergence

    Returns:
        Convergence function that takes a state and last energy and
        returns a batchwise boolean function
    """

    def convergence_fn(
        state: SimState,
        last_energy: torch.Tensor | None = None,
    ) -> bool:
        """Check if the system has converged."""
        return torch.abs(state.energy - last_energy) < energy_tol

    return convergence_fn


def optimize(
    system: StateLike,
    model: ModelInterface,
    *,
    optimizer: Callable,
    convergence_fn: Callable | None = None,
    trajectory_reporter: TrajectoryReporter | dict | None = None,
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
        optimizer_kwargs: Additional keyword arguments for optimizer init function
        trajectory_reporter (TrajectoryReporter | dict | None): Optional reporter for
            tracking optimization trajectory. If a dict, will be passed to the
            TrajectoryReporter constructor.
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
        convergence_fn = generate_energy_convergence_fn(energy_tol=1e-3)

    # initialize the state
    state: SimState = initialize_state(system, model.device, model.dtype)
    init_fn, update_fn = optimizer(model=model, **optimizer_kwargs)

    max_attempts = max_steps // steps_between_swaps
    autobatcher = _configure_hot_swapping_autobatcher(
        model, state, autobatcher, max_attempts
    )
    state = _chunked_apply(
        init_fn,
        state,
        model,
        max_memory_scaler=autobatcher.max_memory_scaler,
        memory_scales_with=autobatcher.memory_scales_with,
    )
    autobatcher.load_states(state)
    trajectory_reporter = _configure_reporter(
        trajectory_reporter,
        properties=["potential_energy"],
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
    trajectory_reporter: TrajectoryReporter | dict | None = None,
    autobatcher: ChunkingAutoBatcher | bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Run single point calculations on a batch of systems.

    Unlike the other runners, this function does not return a state. Instead, it
    returns a list of dictionaries, one for each batch in the input state. Each
    dictionary contains the properties calculated for that batch. It will also
    modify the state in place with the "energy", "forces", and "stress" properties
    if they are present in the model output.

    Args:
        system (StateLike): Input system to calculate properties for
        model (ModelInterface): Neural network model module
        unit_system (UnitSystem): Unit system for energy and forces
        trajectory_reporter (TrajectoryReporter | dict | None): Optional reporter for
            tracking trajectory. If a dict, will be passed to the TrajectoryReporter
            constructor and must include at least the "filenames" key. Any prop
            calculators will be executed and the results will be returned in a list.
            Make sure that if multiple unique states are used, that the
            `variable_atomic_numbers` and `variable_masses` are set to `True` in the
            `state_kwargs` argument.
        autobatcher (ChunkingAutoBatcher | bool): Optional autobatcher to use for
            batching calculations

    Returns:
        list[dict[str, torch.Tensor]]: Maps of property names to tensors for all batches
    """
    # initialize the state
    state: SimState = initialize_state(system, model.device, model.dtype)

    batch_iterator = _configure_batches_iterator(model, state, autobatcher)
    properties = ["potential_energy"]
    if model.compute_forces:
        properties.append("forces")
    if model.compute_stress:
        properties.append("stress")
    trajectory_reporter = _configure_reporter(
        trajectory_reporter or dict(filenames=None),
        state_kwargs={
            "variable_atomic_numbers": True,
            "variable_masses": True,
            "save_forces": model.compute_forces,
        },
        properties=properties,
    )

    @dataclass
    class StaticState(type(state)):
        energy: torch.Tensor
        forces: torch.Tensor | None
        stress: torch.Tensor | None

    final_states: list[SimState] = []
    all_props: list[dict[str, torch.Tensor]] = []
    og_filenames = trajectory_reporter.filenames
    for substate, batch_indices in batch_iterator:
        # set up trajectory reporters
        if autobatcher and trajectory_reporter and og_filenames is not None:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[og_filenames[idx] for idx in batch_indices]
            )

        model_outputs = model(substate)

        substate = StaticState(
            **vars(substate),
            energy=model_outputs["energy"],
            forces=model_outputs["forces"] if model.compute_forces else None,
            stress=model_outputs["stress"] if model.compute_stress else None,
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
