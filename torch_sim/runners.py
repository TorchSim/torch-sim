"""Simulation runners for molecular dynamics and optimization.

This module provides functions for running molecular dynamics simulations and geometry
optimizations using various models and integrators. It includes utilities for
converting between different molecular representations and handling simulation state.
"""

import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from numpy.typing import ArrayLike

from torch_sim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher
from torch_sim.models.interface import ModelInterface
from torch_sim.quantities import batchwise_max_force, kinetic_energy, temperature
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
        model: The model to use for the integration
        state: The state to use for the integration
        autobatcher: The autobatcher to use for the integration

    Returns:
        A batches iterator
    """
    # load and properly configure the autobatcher
    if autobatcher and isinstance(autobatcher, bool):
        autobatcher = ChunkingAutoBatcher(
            model=model,
            return_indices=True,
        )
        autobatcher.load_states(state)
        batchs = autobatcher
    elif isinstance(autobatcher, ChunkingAutoBatcher):
        autobatcher.load_states(state)
        autobatcher.return_indices = True
        batchs = autobatcher
    elif not autobatcher:
        batchs = [(state, [])]
    else:
        raise ValueError(
            f"Invalid autobatcher type: {type(autobatcher)}, "
            "must be bool, ChunkingAutoBatcher, or None."
        )
    return batchs


def create_default_reporter(
    filenames: str | Path | list[str | Path],
    property_frequency: int = 10,
    state_frequency: int = 50,
    properties: Iterable[str] = (
        "positions",
        "kinetic_energy",
        "potential_energy",
        "temperature",
        "stress",
    ),
) -> TrajectoryReporter:
    """Create a default trajectory reporter.

    Args:
        filenames: Filenames to save the trajectory to.
        property_frequency: Frequency to save properties at.
        state_frequency: Frequency to save state at.
        properties: Properties to save, possible properties are "positions",
            "kinetic_energy", "potential_energy", "temperature", "stress", "velocities",
            and "forces".
        model: The model to use for the reporter

    Returns:
        A trajectory reporter
    """

    def compute_stress(state: SimState, model: ModelInterface) -> torch.Tensor:
        # Check model type by name rather than instance
        # TODO: this is a bit of a dumb way of tracking stress
        if not model.compute_stress:
            try:
                og_model_stress = model.compute_stress
                model.compute_stress = True
            except AttributeError as err:
                raise ValueError(
                    "Model stress is not set to true and model stress cannot be "
                    "set on the fly. Please set model.compute_stress to True."
                ) from err
        model_outputs = model(state)
        if not model.compute_stress:
            model.compute_stress = og_model_stress

        return model_outputs["stress"]

    possible_properties = {
        "kinetic_energy": lambda state: kinetic_energy(state.momenta, state.masses),
        "potential_energy": lambda state: state.energy,
        "temperature": lambda state: temperature(state.momenta, state.masses),
        "stress": compute_stress,
    }

    prop_calculators = {
        prop: calculator
        for prop, calculator in possible_properties.items()
        if prop in properties
    }

    save_velocities = "velocities" in properties
    save_forces = "forces" in properties
    return TrajectoryReporter(
        filenames=filenames,
        state_frequency=state_frequency,
        prop_calculators={property_frequency: prop_calculators},
        state_kwargs={"save_velocities": save_velocities, "save_forces": save_forces},
    )


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
        system: Input system to simulate
        model: Neural network model module
        integrator: Integration algorithm function
        n_steps: Number of integration steps
        temperature: Temperature or array of temperatures for each step
        timestep: Integration time step
        unit_system: Unit system for temperature and time
        integrator_kwargs: Additional keyword arguments for integrator
        trajectory_reporter: Optional reporter for tracking trajectory.
        autobatcher: Optional autobatcher to use
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

    final_states = []
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
    max_attempts: int, # TODO: change name to max_iterations
) -> HotSwappingAutoBatcher:
    """Configure the hot swapping autobatcher for the optimize function.

    Args:
        model: The model to use for the autobatcher
        state: The state to use for the autobatcher
        autobatcher: The autobatcher to use for the autobatcher
        max_attempts: The maximum number of attempts for the autobatcher

    Returns:
        A hot swapping autobatcher
    """
    # load and properly configure the autobatcher
    if isinstance(autobatcher, HotSwappingAutoBatcher):
        autobatcher.return_indices = True
        autobatcher.max_attempts = max_attempts
        autobatcher.load_states(state)
    else:
        memory_scales_with = getattr(model, "memory_scales_with", "n_atoms")
        max_memory_scaler = None if autobatcher else state.n_atoms + 1
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
        force_tol: Force tolerance for convergence

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
        system: Input system to optimize (ASE Atoms, Pymatgen Structure, or SimState)
        model: Neural network model module
        optimizer: Optimization algorithm function
        convergence_fn: Condition for convergence, should return a boolean tensor
            of length n_batches
        unit_system: Unit system for energy tolerance
        optimizer_kwargs: Additional keyword arguments for optimizer init function
        trajectory_reporter: Optional reporter for tracking optimization trajectory
        autobatcher: Optional autobatcher to use. If False, the system will assume
            infinite memory and will not batch, but will still remove converged
            structures from the batch. If True, the system will estimate the memory
            available and batch accordingly. If a HotSwappingAutoBatcher, the system
            will use the provided autobatcher, but will reset the max_attempts to
            max_steps // steps_between_swaps.
        max_steps: Maximum number of total optimization steps
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
