"""Simulation runners for molecular dynamics and optimization.

This module provides functions for running molecular dynamics simulations and geometry
optimizations using various calculators and integrators. It includes utilities for
converting between different molecular representations and handling simulation state.
"""

import inspect
import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import ArrayLike

from torchsim.autobatching import ChunkingAutoBatcher, HotSwappingAutoBatcher
from torchsim.state import BaseState, StateLike, concatenate_states, state_to_device
from torchsim.trajectory import TrajectoryReporter
from torchsim.units import UnitSystem
from torchsim.models.interface import ModelInterface


if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Structure

try:
    from pymatgen.core import Structure
except ImportError:

    class Structure:
        """Stub class for pymatgen Structure when not installed."""


try:
    from ase import Atoms
except ImportError:

    class Atoms:
        """Stub class for ASE Atoms when not installed."""


def create_batches_iterator(
    model: ModelInterface,
    state: BaseState,
    autobatcher: ChunkingAutoBatcher | bool,
) -> ChunkingAutoBatcher:
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
) -> BaseState:
    """Simulate a system using a model and integrator.

    Args:
        system: Input system to simulate
        model: Neural network calculator module
        integrator: Integration algorithm function
        n_steps: Number of integration steps
        temperature: Temperature or array of temperatures for each step
        timestep: Integration time step
        unit_system: Unit system for temperature and time
        integrator_kwargs: Additional keyword arguments for integrator
        trajectory_reporter: Optional reporter for tracking trajectory
        autobatcher: Optional autobatcher to use

    Returns:
        BaseState: Final state after integration
    """
    # create a list of temperatures
    temps = temperature if hasattr(temperature, "__iter__") else [temperature] * n_steps
    if len(temps) != n_steps:
        raise ValueError(
            f"len(temperature) = {len(temps)}. It must equal n_steps = {n_steps}"
        )

    # initialize the state
    state: BaseState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    init_fn, update_fn = integrator(
        model=model,
        kT=torch.tensor(temps[0] * unit_system.temperature, dtype=dtype, device=device),
        dt=torch.tensor(timestep * unit_system.time, dtype=dtype, device=device),
        **integrator_kwargs,
    )
    state = init_fn(state)

    batch_iterator = create_batches_iterator(model, state, autobatcher)

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


def configure_hot_swapping_autobatcher(
    model: ModelInterface,
    state: BaseState,
    autobatcher: HotSwappingAutoBatcher | bool,
) -> HotSwappingAutoBatcher:

    # load and properly configure the autobatcher
    if isinstance(autobatcher, HotSwappingAutoBatcher):
        autobatcher.return_indices = True
        autobatcher.load_states(state)
    else:
        memory_scales_with = getattr(model, "memory_scales_with", "n_atoms")
        max_memory_scaler = None if autobatcher else state.n_atoms + 1
        autobatcher = HotSwappingAutoBatcher(
            model=model,
            return_indices=True,
            max_memory_scaler=max_memory_scaler,
            memory_scales_with=memory_scales_with,
        )
        autobatcher.load_states(state)
    return autobatcher


def batchwise_max_force(state: BaseState) -> torch.Tensor:
    batch_wise_max_force = torch.zeros(
        state.n_batches, device=state.device, dtype=state.dtype
    )
    max_forces = state.forces.norm(dim=1)
    return batch_wise_max_force.scatter_reduce(
        dim=0,
        index=state.batch,
        src=max_forces,
        reduce="amax",
    )


def generate_max_force_convergence_fn(force_tol: float = 1e-1) -> Callable:
    def convergence_fn(state: BaseState, last_energy: torch.Tensor) -> bool:
        """Check if the system has converged."""
        return batchwise_max_force(state) < force_tol

    return convergence_fn


def optimize(
    system: StateLike,
    model: ModelInterface,
    optimizer: Callable,
    convergence_fn: Callable | None = None,
    unit_system: UnitSystem = UnitSystem.metal,
    trajectory_reporter: TrajectoryReporter | None = None,
    max_steps: int = 10_000,
    autobatcher: HotSwappingAutoBatcher | bool = False,
    steps_between_swaps: int = 5,
    **optimizer_kwargs: dict,
) -> BaseState:
    """Optimize a system using a model and optimizer.

    Args:
        system: Input system to optimize (ASE Atoms, Pymatgen Structure, or BaseState)
        model: Neural network calculator module
        optimizer: Optimization algorithm function
        convergence_fn: Condition for convergence, should return a boolean tensor
            of length n_batches
        unit_system: Unit system for energy tolerance
        optimizer_kwargs: Additional keyword arguments for optimizer
        trajectory_reporter: Optional reporter for tracking optimization trajectory
        max_steps: Maximum number of optimization steps

    Returns:
        Optimized system state
    """
    # create a default convergence function if one is not provided
    # TODO: document this behavior
    if convergence_fn is None:

        def convergence_fn(state: BaseState, last_energy: torch.Tensor) -> bool:
            return last_energy - state.energy < 1e-6 * unit_system.energy

    # we partially evaluate the function to create a new function with
    # an optional second argument, this can be set to state later on
    # if len(inspect.signature(convergence_fn).parameters) == 1:
    #     convergence_fn = partial(
    #         lambda state, _=None, fn=None: fn(state), fn=convergence_fn
    #     )

    # initialize the state
    state: BaseState = initialize_state(system, model.device, model.dtype)
    init_fn, update_fn = optimizer(
        model=model,
        **optimizer_kwargs,
    )
    state = init_fn(state)

    autobatcher = configure_hot_swapping_autobatcher(model, state, autobatcher)

    step: int = 1
    last_energy = state.energy + 1
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
            state = update_fn(state)
            last_energy = state.energy

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)
            step += 1
            if step > max_steps:
                # TODO: desired behavior?
                warnings.warn(f"Optimize has reached max steps: {step}", stacklevel=2)
                break

        convergence_tensor = convergence_fn(state, last_energy)

    else:
        all_converged_states.extend(result[1])

    if trajectory_reporter:
        trajectory_reporter.finish()

    if autobatcher:
        final_states = autobatcher.restore_original_order(all_converged_states)
        return concatenate_states(final_states)

    return state


def initialize_state(
    system: StateLike,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Initialize state tensors from a system.

    Args:
        system: Input system to convert to state tensors
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: State tensors initialized from input system

    Raises:
        ValueError: If system type is not supported
    """
    # TODO: create a way to pass velocities from pmg and ase

    if isinstance(system, BaseState):
        return state_to_device(system, device, dtype)

    if isinstance(system, list) and all(isinstance(s, BaseState) for s in system):
        if not all(state.n_batches == 1 for state in system):
            raise ValueError(
                "When providing a list of states, to the initialize_state function, "
                "all states must have n_batches == 1. To fix this, you can split the "
                "states into individual states with the split_state function."
            )
        return concatenate_states(system)

    try:
        from pymatgen.core import Structure

        if isinstance(system, Structure) or (
            isinstance(system, list) and all(isinstance(s, Structure) for s in system)
        ):
            return structures_to_state(system, device, dtype)
    except ImportError:
        pass

    try:
        from ase import Atoms

        if isinstance(system, Atoms) or (
            isinstance(system, list) and all(isinstance(s, Atoms) for s in system)
        ):
            return atoms_to_state(system, device, dtype)
    except ImportError:
        pass

    # remaining code just for informative error
    is_list = isinstance(system, list)
    all_same_type = (
        is_list and all(isinstance(s, type(system[0])) for s in system) and system
    )
    if is_list and not all_same_type:
        raise ValueError(
            f"All items in list must be of the same type, "
            f"found {type(system[0])} and {type(system[1])}"
        )

    system_type = f"list[{type(system[0])}]" if is_list else type(system)

    raise ValueError(f"Unsupported system type, {system_type}")


def state_to_atoms(state: BaseState) -> list["Atoms"]:
    """Convert a BaseState to a list of ASE Atoms objects.

    Args:
        state (BaseState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Atoms]: List of ASE Atoms objects, one per batch

    Notes:
        - Output positions and cell will be in Å
        - Output masses will be in amu
    """
    try:
        from ase import Atoms
        from ase.data import chemical_symbols
    except ImportError as err:
        raise ImportError("ASE is required for state_to_atoms conversion") from err

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_batches, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    batch = state.batch.detach().cpu().numpy()

    atoms_list = []
    for batch_idx in np.unique(batch):
        mask = batch == batch_idx
        batch_positions = positions[mask]
        batch_numbers = atomic_numbers[mask]
        batch_cell = cell[batch_idx].T  # Transpose for ASE convention

        # Convert atomic numbers to chemical symbols
        symbols = [chemical_symbols[z] for z in batch_numbers]

        atoms_list.append(
            Atoms(
                symbols=symbols,
                positions=batch_positions,
                cell=batch_cell,
                pbc=state.pbc,
            )
        )

    return atoms_list


def state_to_structures(state: BaseState) -> list["Structure"]:
    """Convert a BaseState to a list of Pymatgen Structure objects.

    Args:
        state (BaseState): Batched state containing positions, cell, and atomic numbers

    Returns:
        list[Structure]: List of Pymatgen Structure objects, one per batch

    Notes:
        - Output positions and cell will be in Å
        - Assumes periodic boundary conditions
    """
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.periodic_table import Element
    except ImportError as err:
        raise ImportError(
            "Pymatgen is required for state_to_structure conversion"
        ) from err

    # Convert tensors to numpy arrays on CPU
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.detach().cpu().numpy()  # Shape: (n_batches, 3, 3)
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    batch = state.batch.detach().cpu().numpy()

    # Get unique batch indices and counts
    unique_batches = np.unique(batch)
    structures = []

    for batch_idx in unique_batches:
        # Get mask for current batch
        mask = batch == batch_idx
        batch_positions = positions[mask]
        batch_numbers = atomic_numbers[mask]
        batch_cell = cell[batch_idx].T  # Transpose for conventional form

        # Create species list from atomic numbers
        species = [Element.from_Z(z) for z in batch_numbers]

        # Create structure for this batch
        structures.append(
            Structure(
                lattice=Lattice(batch_cell),
                species=species,
                coords=batch_positions,
                coords_are_cartesian=True,
            )
        )

    return structures


def atoms_to_state(
    atoms: "Atoms | list[Atoms]",
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Create state tensors from an ASE Atoms object or list of Atoms objects.

    Args:
        atoms: Single ASE Atoms object or list of Atoms objects
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: Batched state tensors in internal units
    """
    try:
        from ase import Atoms
    except ImportError as err:
        raise ImportError("ASE is required for state_to_atoms conversion") from err

    atoms_list = [atoms] if isinstance(atoms, Atoms) else atoms

    # Stack all properties in one go
    positions = torch.tensor(
        np.concatenate([a.positions for a in atoms_list]), dtype=dtype, device=device
    )
    masses = torch.tensor(
        np.concatenate([a.get_masses() for a in atoms_list]), dtype=dtype, device=device
    )
    atomic_numbers = torch.tensor(
        np.concatenate([a.get_atomic_numbers() for a in atoms_list]),
        dtype=torch.int,
        device=device,
    )
    cell = torch.tensor(
        np.stack([a.cell.array.T for a in atoms_list]), dtype=dtype, device=device
    )

    # Create batch indices using repeat_interleave
    atoms_per_batch = torch.tensor([len(a) for a in atoms_list], device=device)
    batch = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_batch
    )

    # Verify consistent pbc
    if not all(all(a.pbc) == all(atoms_list[0].pbc) for a in atoms_list):
        raise ValueError("All systems must have the same periodic boundary conditions")

    return BaseState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=all(atoms_list[0].pbc),
        atomic_numbers=atomic_numbers,
        batch=batch,
    )


def structures_to_state(
    structure: "Structure | list[Structure]",
    device: torch.device,
    dtype: torch.dtype,
) -> BaseState:
    """Create a BaseState from pymatgen Structure(s).

    Args:
        structure: Single Structure or list of Structure objects
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        BaseState: Batched state tensors in internal units

    Notes:
        - Cell matrix follows ASE convention: [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
        - Assumes periodic boundary conditions from Structure
    """
    try:
        from pymatgen.core import Structure
    except ImportError as err:
        raise ImportError(
            "Pymatgen is required for state_to_structure conversion"
        ) from err

    struct_list = [structure] if isinstance(structure, Structure) else structure

    # Stack all properties
    cell = torch.tensor(
        np.stack([s.lattice.matrix.T for s in struct_list]), dtype=dtype, device=device
    )
    positions = torch.tensor(
        np.concatenate([s.cart_coords for s in struct_list]), dtype=dtype, device=device
    )
    masses = torch.tensor(
        np.concatenate([[site.specie.atomic_mass for site in s] for s in struct_list]),
        dtype=dtype,
        device=device,
    )
    atomic_numbers = torch.tensor(
        np.concatenate([[site.specie.number for site in s] for s in struct_list]),
        dtype=torch.int,
        device=device,
    )

    # Create batch indices
    atoms_per_batch = torch.tensor([len(s) for s in struct_list], device=device)
    batch = torch.repeat_interleave(
        torch.arange(len(struct_list), device=device), atoms_per_batch
    )

    return BaseState(
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=True,  # Structures are always periodic
        atomic_numbers=atomic_numbers,
        batch=batch,
    )
