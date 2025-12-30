"""High level runners for atomistic simulations.

This module provides functions for running molecular dynamics simulations and geometry
optimizations using various models and integrators. It includes utilities for
converting between different atomistic representations and handling simulation state.
"""

import copy
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from tqdm import tqdm

import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher
from torch_sim.integrators import INTEGRATOR_REGISTRY, Integrator
from torch_sim.integrators.md import MDState
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import OPTIM_REGISTRY, FireState, Optimizer, OptimState
from torch_sim.state import SimState
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.typing import StateLike
from torch_sim.units import UnitSystem


def _configure_reporter(
    trajectory_reporter: TrajectoryReporter | dict,
    *,
    properties: list[str] | None = None,
    prop_frequency: int = 10,
    state_frequency: int = 100,
) -> TrajectoryReporter:
    if isinstance(trajectory_reporter, TrajectoryReporter):
        return trajectory_reporter
    possible_properties = {
        "potential_energy": lambda state: state.energy,
        "forces": lambda state: state.forces,
        "stress": lambda state: state.stress,
        "kinetic_energy": lambda state: ts.calc_kinetic_energy(
            velocities=state.velocities, masses=state.masses
        ),
        "temperature": lambda state: state.calc_temperature(),
    }

    prop_calculators = {
        prop: calculator
        for prop, calculator in possible_properties.items()
        if prop in (properties or ())
    }

    # ordering is important to ensure we can override defaults
    trajectory_reporter = copy.deepcopy(trajectory_reporter)
    return TrajectoryReporter(
        prop_calculators=trajectory_reporter.pop(
            "prop_calculators", {prop_frequency: prop_calculators}
        ),
        state_frequency=trajectory_reporter.pop("state_frequency", state_frequency),
        **trajectory_reporter,
    )


def _configure_batches_iterator(
    state: SimState,
    model: ModelInterface,
    *,
    autobatcher: BinningAutoBatcher | bool,
) -> BinningAutoBatcher | list[tuple[SimState, list[int]]]:
    """Create a batches iterator for the integrate function.

    Args:
        model (ModelInterface): The model to use for the integration
        state (SimState): The state to use for the integration
        autobatcher (BinningAutoBatcher | bool): The autobatcher to use for integration

    Returns:
        A batches iterator
    """
    # load and properly configure the autobatcher
    if autobatcher is True:
        autobatcher = BinningAutoBatcher(
            model=model,
            max_memory_padding=0.9,
        )
        autobatcher.load_states(state)
        batches = autobatcher
    elif isinstance(autobatcher, BinningAutoBatcher):
        autobatcher.load_states(state)
        batches = autobatcher
    elif autobatcher is False:
        batches = [(state, [])]
    else:
        autobatcher_type = type(autobatcher).__name__
        raise TypeError(
            f"Invalid {autobatcher_type=}, must be bool or BinningAutoBatcher."
        )
    return batches


def _determine_initial_step_for_integrate(
    trajectory_reporter: TrajectoryReporter | None,
) -> int:
    """Determine the initial step for resuming integration from trajectory files.

    Args:
        trajectory_reporter (TrajectoryReporter | None): The trajectory reporter to
            check for resume information

    Returns:
        int: The initial step to start from (1 if not resuming, otherwise last_step + 1)
    """
    initial_step: int = 1
    if trajectory_reporter is not None and trajectory_reporter.mode == "a":
        last_logged_steps = trajectory_reporter.last_step
        last_logged_step = min(last_logged_steps)
        initial_step = initial_step + last_logged_step
        if len(set(last_logged_steps)) != 1:
            raise ValueError(
                f"Trajectory files have different last steps: {set(last_logged_steps)} "
                "Cannot resume integration from inconsistent states."
                "You can truncate the trajectories to the same step using:\n\n"
                "    trajectory_reporter.truncate_to_step(min(trajectory_reporter.last_step))\n\n"
                "before calling integrate again."
            )
        print(
            f"Detected existing trajectory with last step {last_logged_step}."
            f" Resuming integration from step {initial_step}."
        )
    return initial_step


def _determine_initial_step_for_optimize(
    trajectory_reporter: TrajectoryReporter | None,
    state: SimState,
) -> torch.LongTensor:
    """Determine the initial steps for resuming optimization from trajectory files.

    Args:
        trajectory_reporter (TrajectoryReporter | None): The trajectory reporter to
            check for resume information
        state (SimState): The state being optimized

    Returns:
        torch.LongTensor: Tensor of initial steps for each system (1 if not resuming,
            otherwise last_step + 1 for each system)
    """
    initial_step: torch.LongTensor = torch.full(
        size=(state.n_systems,), fill_value=1, dtype=torch.long, device=state.device
    )
    if trajectory_reporter is not None and trajectory_reporter.mode == "a":
        last_logged_steps = torch.tensor(
            trajectory_reporter.last_step, dtype=torch.long, device=state.device
        )
        initial_step = initial_step + last_logged_steps
    return initial_step


def _normalize_temperature_tensor(
    temperature: float | list | torch.Tensor, n_steps: int, initial_state: SimState
) -> torch.Tensor:
    """Turn the temperature into a tensor of shape (n_steps,) or (n_steps, n_systems).

    Args:
        temperature (float | int | list | torch.Tensor): Temperature input
        n_steps (int): Number of integration steps
        initial_state (SimState): Initial simulation state for dtype and device
    Returns:
        torch.Tensor: Normalized temperature tensor
    """
    # ---- Step 1: Convert to tensor ----
    if isinstance(temperature, (float, int)):
        return torch.full(
            (n_steps,),
            float(temperature),
            dtype=initial_state.dtype,
            device=initial_state.device,
        )

    # Convert list or tensor input to tensor
    if isinstance(temperature, list):
        temps = torch.tensor(
            temperature, dtype=initial_state.dtype, device=initial_state.device
        )
    elif isinstance(temperature, torch.Tensor):
        temps = temperature.to(dtype=initial_state.dtype, device=initial_state.device)
    else:
        raise TypeError(
            f"Invalid temperature type: {type(temperature).__name__}. "
            "Must be float, int, list, or torch.Tensor."
        )

    # ---- Step 2: Determine how to broadcast ----
    temps = torch.atleast_1d(temps)
    if temps.ndim > 2:
        raise ValueError(f"Temperature tensor must be 1D or 2D, got shape {temps.shape}.")

    if temps.shape[0] == 1:
        # A single value in a 1-element list/tensor
        return temps.repeat(n_steps)

    if initial_state.n_systems == n_steps:
        warnings.warn(
            "n_systems is equal to n_steps. Interpreting temperature array of length "
            "n_systems as temperatures for each system, broadcasted over steps.",
            stacklevel=2,
        )

    if temps.shape[0] == initial_state.n_systems:
        if temps.ndim == 2:
            raise ValueError(
                "If temperature tensor is 2D, first dimension must be n_steps."
            )
        # Interpret as single-step multi-system temperatures → broadcast over steps
        return temps.unsqueeze(0).expand(n_steps, -1)  # (n_steps, n_systems)

    if temps.shape[0] == n_steps:
        return temps  # already good: (n_steps,) or (n_steps, n_systems)

    raise ValueError(
        f"Temperature length ({temps.shape[0]}) must be either:\n"
        f" - n_steps ({n_steps}), or\n"
        f" - n_systems ({initial_state.n_systems}), or\n"
        f" - 1 (scalar),\n"
        f"but got {temps.shape[0]}."
    )


def integrate[T: SimState](  # noqa: C901
    system: StateLike,
    model: ModelInterface,
    *,
    integrator: Integrator | tuple[Callable[..., T], Callable[..., T]],
    n_steps: int,
    temperature: float | list | torch.Tensor,
    timestep: float,
    trajectory_reporter: TrajectoryReporter | dict | None = None,
    autobatcher: BinningAutoBatcher | bool = False,
    pbar: bool | dict[str, Any] = False,
    init_kwargs: dict[str, Any] | None = None,
    **integrator_kwargs: Any,
) -> T:
    """Simulate a system using a model and integrator.

    Args:
        system (StateLike): Input system to simulate
        model (ModelInterface): Neural network model module
        integrator (Integrator | tuple): Either a key from Integrator or a tuple of
            (init_func, step_func) functions.
        n_steps (int): Number of integration steps. If resuming from a trajectory, this
            is the  number of additional steps to run.
        temperature (float | ArrayLike): Temperature or array of temperatures for each
            step or system:
            Float: used for all steps and systems
            1D array of length n_steps: used for each step
            1D array of length n_systems: used for each system
            2D array of shape (n_steps, n_systems): used for each step and system.
        timestep (float): Integration time step
        trajectory_reporter (TrajectoryReporter | dict | None): Optional reporter for
            tracking trajectory. If a dict, will be passed to the TrajectoryReporter
            constructor.
        autobatcher (BinningAutoBatcher | bool): Optional autobatcher to use
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.
        init_kwargs (dict[str, Any], optional): Additional keyword arguments for
            integrator init function.
        **integrator_kwargs: Additional keyword arguments for integrator init function

    Returns:
        T: Final state after integration
    """
    unit_system = UnitSystem.metal

    initial_state: SimState = ts.initialize_state(system, model.device, model.dtype)
    dtype, device = initial_state.dtype, initial_state.device
    kTs = _normalize_temperature_tensor(temperature, n_steps, initial_state)
    kTs = kTs * unit_system.temperature
    dt = torch.tensor(timestep * unit_system.time, dtype=dtype, device=device)

    # Handle both string names and direct function tuples
    if isinstance(integrator, Integrator):
        init_func, step_func = INTEGRATOR_REGISTRY[integrator]
    elif (
        isinstance(integrator, tuple)
        and len(integrator) == 2
        and {*map(callable, integrator)} == {True}
    ):
        init_func, step_func = integrator
    else:
        raise ValueError(
            f"integrator must be key from Integrator or a tuple of "
            f"(init_func, step_func), got {type(integrator)}"
        )
    # batch_iterator will be a list if autobatcher is False
    batch_iterator = _configure_batches_iterator(
        initial_state, model, autobatcher=autobatcher
    )
    if trajectory_reporter is not None:
        trajectory_reporter = _configure_reporter(
            trajectory_reporter,
            properties=["kinetic_energy", "potential_energy", "temperature"],
        )
    # Auto-detect initial step from trajectory files for resuming integration
    initial_step = _determine_initial_step_for_integrate(trajectory_reporter)

    final_states: list[T] = []
    og_filenames = trajectory_reporter.filenames if trajectory_reporter else None

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Integrate")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=initial_state.n_systems, **pbar_kwargs)

    # Handle both BinningAutoBatcher and list of tuples
    for state, system_indices in batch_iterator:
        # Pass correct parameters based on integrator type
        batch_kT = (
            kTs[:, system_indices] if (system_indices and len(kTs.shape) == 2) else kTs
        )
        state = init_func(
            state=state, model=model, kT=batch_kT[0], dt=dt, **init_kwargs or {}
        )

        # set up trajectory reporters
        if autobatcher and trajectory_reporter is not None and og_filenames is not None:
            # we must remake the trajectory reporter for each system
            trajectory_reporter.reopen_trajectories(
                filenames=[og_filenames[i] for i in system_indices]
            )

        # run the simulation
        for step in range(initial_step, initial_step + n_steps):
            state = step_func(
                state=state,
                model=model,
                dt=dt,
                kT=batch_kT[step - initial_step],
                **integrator_kwargs,
            )

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)

        # finish the trajectory reporter
        final_states.append(state)
        if tqdm_pbar:
            tqdm_pbar.update(state.n_systems)

    if trajectory_reporter:
        trajectory_reporter.finish()

    if isinstance(batch_iterator, BinningAutoBatcher):
        reordered_states = batch_iterator.restore_original_order(final_states)
        return ts.concatenate_states(reordered_states)

    return state


def _configure_in_flight_autobatcher(
    state: SimState,
    model: ModelInterface,
    *,
    autobatcher: InFlightAutoBatcher | bool,
    max_iterations: int,  # TODO: change name to max_iterations
) -> InFlightAutoBatcher:
    """Configure the hot swapping autobatcher for the optimize function.

    Args:
        model (ModelInterface): The model to use for the autobatcher
        state (SimState): The state to use for the autobatcher
        autobatcher (InFlightAutoBatcher | bool): The autobatcher to use for the
            autobatcher
        max_iterations (int): The maximum number of iterations for each state in
            the autobatcher.

    Returns:
        A hot swapping autobatcher
    """
    # load and properly configure the autobatcher
    if isinstance(autobatcher, InFlightAutoBatcher):
        autobatcher.max_iterations = max_iterations
    elif isinstance(autobatcher, bool):
        if autobatcher:
            memory_scales_with = model.memory_scales_with
            max_memory_scaler = None
        else:
            memory_scales_with = "n_atoms"
            max_memory_scaler = state.n_atoms + 1
        autobatcher = InFlightAutoBatcher(
            model=model,
            max_memory_scaler=max_memory_scaler,
            memory_scales_with=memory_scales_with,
            max_iterations=max_iterations,
            max_memory_padding=0.9,
        )
    else:
        autobatcher_type = type(autobatcher).__name__
        cls_name = InFlightAutoBatcher.__name__
        raise TypeError(f"Invalid {autobatcher_type=}, must be bool or {cls_name}.")
    return autobatcher


def _chunked_apply[T: SimState](
    fn: Callable[..., T],
    states: SimState,
    model: ModelInterface,
    init_kwargs: Any,
    **batcher_kwargs: Any,
) -> T:
    """Apply a function to a state in chunks.

    This prevents us from running out of memory when applying a function to a large
    number of states.

    Args:
        fn (Callable): The state function to apply
        states (SimState): The states to apply the function to
        model (ModelInterface): The model to use for the autobatcher
        init_kwargs (Any): Unpacked into state init function.
        **batcher_kwargs: Additional keyword arguments for the autobatcher

    Returns:
        A state with the function applied
    """
    autobatcher = BinningAutoBatcher(model=model, **batcher_kwargs)
    autobatcher.load_states(states)
    initialized_states = []

    initialized_states = [
        fn(model=model, state=system, **init_kwargs) for system, _indices in autobatcher
    ]

    ordered_states = autobatcher.restore_original_order(initialized_states)
    return ts.concatenate_states(ordered_states)


def generate_force_convergence_fn[T: MDState | FireState](
    force_tol: float = 1e-1, *, include_cell_forces: bool = True
) -> Callable:
    """Generate a force-based convergence function for the convergence_fn argument
    of the optimize function.

    Args:
        force_tol (float): Force tolerance for convergence
        include_cell_forces (bool): Whether to include the `cell_forces` in
            the convergence check. Defaults to True.

    Returns:
        Convergence function that takes a state and last energy and
        returns a systemwise boolean function
    """

    def convergence_fn(
        state: T,
        last_energy: torch.Tensor | None = None,  # noqa: ARG001
    ) -> torch.Tensor:
        """Check if the system has converged.

        Returns:
            torch.Tensor: Boolean tensor of shape (n_systems,) indicating
                convergence status for each system.
        """
        force_conv = ts.system_wise_max_force(state) < force_tol

        if include_cell_forces:
            if (cell_forces := getattr(state, "cell_forces", None)) is None:
                raise ValueError("cell_forces not found in state")
            cell_forces_norm, _ = cell_forces.norm(dim=2).max(dim=1)
            cell_force_conv = cell_forces_norm < force_tol
            return force_conv & cell_force_conv

        return force_conv

    return convergence_fn


def generate_energy_convergence_fn[T: MDState | OptimState](
    energy_tol: float = 1e-3,
) -> Callable[[T, torch.Tensor | None], torch.Tensor]:
    """Generate an energy-based convergence function for the convergence_fn argument
    of the optimize function.

    Args:
        energy_tol (float): Energy tolerance for convergence

    Returns:
        Callable[[T, torch.Tensor | None], torch.Tensor]: Convergence function that takes
            a state and last energy and returns a systemwise boolean function.
    """

    def convergence_fn(state: T, last_energy: torch.Tensor | None = None) -> torch.Tensor:
        """Check if the system has converged.

        Returns:
            torch.Tensor: Boolean tensor of shape (n_systems,) indicating
                convergence status for each system.
        """
        return torch.abs(state.energy - last_energy) < energy_tol

    return convergence_fn


def optimize[T: OptimState](  # noqa: C901, PLR0915
    system: StateLike,
    model: ModelInterface,
    *,
    optimizer: Optimizer | tuple[Callable[..., T], Callable[..., T]],
    convergence_fn: Callable[[T, torch.Tensor | None], torch.Tensor] | None = None,
    max_steps: int = 10_000,
    steps_between_swaps: int = 5,
    trajectory_reporter: TrajectoryReporter | dict | None = None,
    autobatcher: InFlightAutoBatcher | bool = False,
    pbar: bool | dict[str, Any] = False,
    init_kwargs: dict[str, Any] | None = None,
    **optimizer_kwargs: Any,
) -> T:
    """Optimize a system using a model and optimizer.

    Args:
        system (StateLike): Input system to optimize (ASE Atoms, Pymatgen Structure, or
            SimState)
        model (ModelInterface): Neural network model module
        optimizer (Optimizer | tuple): Optimization algorithm function
        convergence_fn (Callable | None): Condition for convergence, should return a
            boolean tensor of length n_systems
        trajectory_reporter (TrajectoryReporter | dict | None): Optional reporter for
            tracking optimization trajectory. If a dict, will be passed to the
            TrajectoryReporter constructor.
        autobatcher (InFlightAutoBatcher | bool): Optional autobatcher to use. If
            False, the system will assume
            infinite memory and will not batch, but will still remove converged
            structures from the batch. If True, the system will estimate the memory
            available and batch accordingly. If a InFlightAutoBatcher, the system
            will use the provided autobatcher, but will reset the max_iterations to
            max_steps // steps_between_swaps.
        max_steps (int): Maximum number of total optimization steps
        steps_between_swaps: Number of steps to take before checking convergence
            and swapping out states.
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.
        init_kwargs (dict[str, Any], optional): Additional keyword arguments for optimizer
            init function.
        **optimizer_kwargs: Additional keyword arguments for optimizer step function

    Returns:
        T: Optimized system state
    """
    # create a default convergence function if one is not provided
    # TODO: document this behavior
    if convergence_fn is None:
        convergence_fn = generate_energy_convergence_fn(energy_tol=1e-3)

    initial_state = ts.initialize_state(system, model.device, model.dtype)
    if isinstance(optimizer, Optimizer):
        init_fn, step_fn = OPTIM_REGISTRY[optimizer]
    elif (
        isinstance(optimizer, tuple)
        and len(optimizer) == 2
        and {*map(callable, optimizer)} == {True}
    ):
        init_fn, step_fn = optimizer
    else:
        optimizer_type = type(optimizer).__name__
        raise TypeError(
            f"Invalid {optimizer_type=}, must be key from Optimizer or a tuple of "
            f"(init_func, step_func), got {optimizer_type}"
        )

    max_iterations = max_steps // steps_between_swaps
    autobatcher = _configure_in_flight_autobatcher(
        initial_state, model, autobatcher=autobatcher, max_iterations=max_iterations
    )

    if isinstance(initial_state, OptimState):
        state = initial_state
    else:
        state = _chunked_apply(
            init_fn,
            initial_state,
            model,
            init_kwargs=dict(**init_kwargs or {}),
            max_memory_scaler=autobatcher.max_memory_scaler,
            memory_scales_with=autobatcher.memory_scales_with,
            max_atoms_to_try=autobatcher.max_atoms_to_try,
            oom_error_message=autobatcher.oom_error_message,
        )
    autobatcher.load_states(state)
    if trajectory_reporter is not None:
        trajectory_reporter = _configure_reporter(
            trajectory_reporter, properties=["potential_energy"]
        )

    # Auto-detect initial step from trajectory files for resuming optimizations
    step = _determine_initial_step_for_optimize(trajectory_reporter, state)
    steps_so_far = 0

    last_energy = None
    all_converged_states: list[T] = []
    convergence_tensor = None
    og_filenames = trajectory_reporter.filenames if trajectory_reporter else None

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Optimize")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=initial_state.n_systems, **pbar_kwargs)

    while True:
        result = autobatcher.next_batch(state, convergence_tensor)
        if result[0] is None:
            # All states have converged, collect the final converged states
            all_converged_states.extend(result[1])
            break
        state, converged_states = result
        all_converged_states.extend(converged_states)

        # need to update the trajectory reporter if any states have converged
        if (
            trajectory_reporter is not None
            and og_filenames is not None
            and (steps_so_far == 0 or len(converged_states) > 0)
        ):
            trajectory_reporter.reopen_trajectories(
                filenames=[og_filenames[i] for i in autobatcher.current_idx]
            )

        for _step in range(steps_between_swaps):
            if hasattr(state, "energy"):
                last_energy = state.energy
            state = step_fn(state=state, model=model, **optimizer_kwargs)

            if trajectory_reporter:
                trajectory_reporter.report(
                    state, step[autobatcher.current_idx].tolist(), model=model
                )
            step[autobatcher.current_idx] += 1
            exceeded_max_steps = step > max_steps
            if exceeded_max_steps.all():
                warnings.warn(
                    f"All systems have reached the maximum number of steps: {max_steps}.",
                    stacklevel=2,
                )
                break

        convergence_tensor = convergence_fn(state, last_energy)
        # Mark states that exceeded max steps as converged to remove them from batch
        convergence_tensor = (
            convergence_tensor | exceeded_max_steps[autobatcher.current_idx]
        )
        if tqdm_pbar:
            # assume convergence_tensor shape is correct
            tqdm_pbar.update(torch.count_nonzero(convergence_tensor).item())

    if trajectory_reporter:
        trajectory_reporter.finish()

    if autobatcher:
        final_states = autobatcher.restore_original_order(all_converged_states)
        return ts.concatenate_states(final_states)

    return state  # type: ignore[return-value]


def static(
    system: StateLike,
    model: ModelInterface,
    *,
    trajectory_reporter: TrajectoryReporter | dict | None = None,
    autobatcher: BinningAutoBatcher | bool = False,
    pbar: bool | dict[str, Any] = False,
) -> list[dict[str, torch.Tensor]]:
    """Run single point calculations on a batch of systems.

    Unlike the other runners, this function does not return a state. Instead, it
    returns a list of dictionaries, one for each system in the input state. Each
    dictionary contains the properties calculated for that system. It will also
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
        autobatcher (BinningAutoBatcher | bool): Optional autobatcher to use for
            batching calculations
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.

    Returns:
        list[dict[str, torch.Tensor]]: Maps of property names to tensors for all batches
    """
    state: SimState = ts.initialize_state(system, model.device, model.dtype)

    batch_iterator = _configure_batches_iterator(state, model, autobatcher=autobatcher)
    properties = ["potential_energy"]
    if model.compute_forces:
        properties.append("forces")
    if model.compute_stress:
        properties.append("stress")
    if isinstance(trajectory_reporter, dict):
        trajectory_reporter = copy.deepcopy(trajectory_reporter)
        trajectory_reporter["state_kwargs"] = {
            "variable_atomic_numbers": True,
            "variable_masses": True,
            "save_forces": model.compute_forces,
        }
    trajectory_reporter = _configure_reporter(
        trajectory_reporter or dict(filenames=None),
        properties=properties,
    )

    @dataclass(kw_only=True)
    class StaticState(SimState):
        energy: torch.Tensor
        forces: torch.Tensor
        stress: torch.Tensor

        _atom_attributes = SimState._atom_attributes | {"forces"}  # noqa: SLF001
        _system_attributes = SimState._system_attributes | {  # noqa: SLF001
            "energy",
            "stress",
        }

    all_props: list[dict[str, torch.Tensor]] = []
    og_filenames = trajectory_reporter.filenames

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Static")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=state.n_systems, **pbar_kwargs)

    # Handle both BinningAutoBatcher and list of tuples
    for sub_state, system_indices in batch_iterator:
        # set up trajectory reporters
        if autobatcher and trajectory_reporter and og_filenames is not None:
            # we must remake the trajectory reporter for each system
            trajectory_reporter.reopen_trajectories(
                filenames=[og_filenames[idx] for idx in system_indices]
            )

        model_outputs = model(sub_state)
        static_state = StaticState(
            positions=sub_state.positions,
            masses=sub_state.masses,
            cell=sub_state.cell,
            pbc=sub_state.pbc,
            atomic_numbers=sub_state.atomic_numbers,
            system_idx=sub_state.system_idx,
            energy=model_outputs["energy"],
            forces=(
                model_outputs["forces"]
                if model.compute_forces
                else torch.full_like(sub_state.positions, fill_value=float("nan"))
            ),
            stress=(
                model_outputs["stress"]
                if model.compute_stress
                else torch.full_like(sub_state.cell, fill_value=float("nan"))
            ),
        )

        props = trajectory_reporter.report(static_state, 0, model=model)
        all_props.extend(props)

        if tqdm_pbar:
            tqdm_pbar.update(static_state.n_systems)

    trajectory_reporter.finish()

    if isinstance(batch_iterator, BinningAutoBatcher):
        # reorder properties to match original order of states
        original_indices = list(chain.from_iterable(batch_iterator.index_bins))
        indexed_props = list(zip(original_indices, all_props, strict=True))
        return [prop for _, prop in sorted(indexed_props, key=lambda x: x[0])]

    return all_props
