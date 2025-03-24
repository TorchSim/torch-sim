"""TorchSim Trajectory: Trajectory file handling for molecular dynamics simulations.

This module provides classes for reading and writing trajectory data in HDF5 format.
The core classes (TorchSimTrajectory and TrajectoryReporter) allow efficient storage
and retrieval of atomic positions, forces, energies, and other properties from
molecular dynamics simulations.

The TorchSimTrajectory does not aim to be a new trajectory standard, but rather
a simple interface for storing and retrieving trajectory data from HDF5 files.
It aims to support arbitrary arrays from the user in a natural way, allowing
it to be seamlessly extended to whatever attributes are important to the user.

Examples:
    # Creating a trajectory file
    with TorchSimTrajectory('simulation.hdf5', mode='w') as traj:
        traj.write_state(state, step=0)

    # Writing to multiple trajectory files with a reporter
    reporter = TrajectoryReporter(['traj1.hdf5', 'traj2.hdf5'], state_frequency=100)
    reporter.report(state, step=0, model=model)

Notes:
    This module uses PyTables (HDF5) for efficient I/O operations and supports
    compression to reduce file sizes. It can interoperate with ASE and pymatgen
    for visualization and analysis.
"""

import copy
import inspect
import pathlib
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, Self

import numpy as np
import tables
import torch

from torch_sim.state import SimState


DATA_TYPE_MAP = {
    np.dtype("float32"): tables.Float32Atom(),
    np.dtype("float64"): tables.Float64Atom(),
    np.dtype("int32"): tables.Int32Atom(),
    np.dtype("int64"): tables.Int64Atom(),
    np.dtype("bool"): tables.BoolAtom(),
    torch.float32: tables.Float32Atom(),
    torch.float64: tables.Float64Atom(),
    torch.int32: tables.Int32Atom(),
    torch.int64: tables.Int64Atom(),
    torch.bool: tables.BoolAtom(),
    bool: tables.BoolAtom(),
}
# ruff: noqa: SLF001


class TrajectoryReporter:
    """Trajectory reporter for saving simulation data at specified intervals.

    This class manages writing multiple trajectory files simultaneously.
    It handles periodic saving of full system states and custom property calculations.

    Attributes:
        state_frequency (int): How often to save full states (in simulation steps)
        prop_calculators (dict): Dictionary mapping frequencies to property calculators
        state_kwargs (dict): Additional arguments for state writing
        metadata (dict): Metadata to save in trajectory files
        trajectories (list): List of TorchSimTrajectory instances
        filenames (list): List of trajectory file paths
        array_registry (dict): Dictionary of array names to (shape, dtype) tuples
        shape_warned (bool): Whether a shape warning has been issued

    Examples:
        >>> reporter = TrajectoryReporter(
        ...     ["system1.h5", "system2.h5"],
        ...     state_frequency=100,
        ...     prop_calculators={10: {"energy": calculate_energy}},
        ... )
        >>> for step in range(1000):
        ...     # Run simulation step
        ...     state = step_fn(state)
        ...     reporter.report(state, step, model)
        >>> reporter.close()
    """

    def __init__(
        self,
        filenames: str | pathlib.Path | list[str | pathlib.Path],
        state_frequency: int = 100,
        *,
        prop_calculators: dict[int, dict[str, Callable]] | None = None,
        state_kwargs: dict | None = None,
        metadata: dict[str, str] | None = None,
        trajectory_kwargs: dict | None = None,
    ) -> None:
        """Initialize a TrajectoryReporter.

        Args:
            filenames (str | pathlib.Path | list[str | pathlib.Path]): Path(s) to
                save trajectory file(s)
            state_frequency (int): How often to save state (in steps)
            prop_calculators (dict[int, dict[str, Callable]], optional): Dictionary
                mapping frequencies to property calculators where each calculator is a
                function that takes a state and optionally a model and returns a tensor.
                Defaults to None.
            state_kwargs (dict, optional): Additional arguments for state writing.
            metadata (dict[str, str], optional): Metadata to save in trajectory file.
            trajectory_kwargs (dict, optional): Additional arguments for trajectory
                initialization.

        Raises:
            ValueError: If filenames are not unique
        """
        self.state_frequency = state_frequency
        self.trajectory_kwargs = trajectory_kwargs or {}
        self.trajectory_kwargs["mode"] = self.trajectory_kwargs.get(
            "mode", "w"
        )  # default will be to force overwrite if none is set
        self.prop_calculators = prop_calculators or {}
        self.state_kwargs = state_kwargs or {}
        self.shape_warned = False
        self.metadata = metadata

        self.trajectories = []
        self.load_new_trajectories(filenames)

        self._add_model_arg_to_prop_calculators()

    def load_new_trajectories(
        self, filenames: str | pathlib.Path | list[str | pathlib.Path]
    ) -> None:
        """Load new trajectories into the reporter.

        Closes any existing trajectory files and initializes new ones.

        Args:
            filenames (str | pathlib.Path | list[str | pathlib.Path]): Path(s) to save
                trajectory file(s)

        Raises:
            ValueError: If filenames are not unique
        """
        self.finish()

        filenames = [filenames] if not isinstance(filenames, list) else filenames
        self.filenames = [pathlib.Path(filename) for filename in filenames]
        if len(set(self.filenames)) != len(self.filenames):
            raise ValueError("All filenames must be unique.")

        self.trajectories = []
        for filename in self.filenames:
            self.trajectories.append(
                TorchSimTrajectory(
                    filename=filename,
                    metadata=self.metadata,
                    **self.trajectory_kwargs,
                )
            )

    @property
    def array_registry(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        """Get the registry of array shapes and dtypes.

        Returns:
            dict[str, tuple[tuple[int, ...], np.dtype]]: Dictionary mapping array names to
                tuples of (shape, dtype)
        """
        # Return the registry from the first trajectory
        if self.trajectories:
            return self.trajectories[0].array_registry
        return {}

    def _add_model_arg_to_prop_calculators(self) -> None:
        """Add model argument to property calculators that only accept state.

        Transforms single-argument (state) property calculators to accept the
        dual-argument (state, model) interface by creating partial functions with an
        optional second argument.
        """
        for frequency in self.prop_calculators:
            for name, prop_fn in self.prop_calculators[frequency].items():
                # Get function signature
                sig = inspect.signature(prop_fn)
                # If function only takes one parameter, wrap it to accept two
                if len(sig.parameters) == 1:
                    # we partially evaluate the function to create a new function with
                    # an optional second argument, this can be set to state later on
                    new_fn = partial(lambda state, _=None, fn=None: fn(state), fn=prop_fn)
                    self.prop_calculators[frequency][name] = new_fn

    def report(
        self, state: SimState, step: int, model: torch.nn.Module | None = None
    ) -> None:
        """Report a state and step to the trajectory files.

        Writes states and calculated properties to all trajectory files at the
        specified frequencies. Splits multi-batch states across separate trajectory
        files. The number of batches must match the number of trajectory files.

        Args:
            state (SimState): Current system state with n_batches equal to
                len(filenames)
            step (int): Current simulation step
            model (torch.nn.Module, optional): Model used for simulation.
                Defaults to None. Must be provided if any prop_calculators
                are provided.

        Raises:
            ValueError: If number of batches doesn't match number of trajectory files
        """
        # Get unique batch indices
        batch_indices = range(state.n_batches)
        # batch_indices = torch.unique(state.batch).cpu().tolist()

        # Ensure we have the right number of trajectories
        if len(batch_indices) != len(self.trajectories):
            raise ValueError(
                f"Number of batches ({len(batch_indices)}) doesn't match "
                f"number of trajectory files ({len(self.trajectories)})"
            )

        split_states = state.split()
        # Process each batch separately
        for substate, trajectory in zip(split_states, self.trajectories, strict=True):
            # Slice the state once to get only the data for this batch
            self.shape_warned = True

            # Write state to trajectory if it's time
            if self.state_frequency and step % self.state_frequency == 0:
                trajectory.write_state(substate, step, **self.state_kwargs)

            # Process property calculators for this batch
            for report_frequency, calculators in self.prop_calculators.items():
                if step % report_frequency != 0 or report_frequency == 0:
                    continue

                # Calculate properties for this substate
                props = {}
                for prop_name, prop_fn in calculators.items():
                    prop = prop_fn(substate, model)
                    if len(prop.shape) == 0:
                        prop = prop.unsqueeze(0)
                    props[prop_name] = prop

                # Write properties to this trajectory
                if props:
                    trajectory.write_arrays(props, step)

    def finish(self) -> None:
        """Finish writing the trajectory files.

        Closes all open trajectory files.
        """
        for trajectory in self.trajectories:
            trajectory.close()

    def close(self) -> None:
        """Close all trajectory files.

        Ensures all data is written to disk and releases the file handles.
        """
        for trajectory in self.trajectories:
            trajectory.close()

    def __enter__(self) -> "TrajectoryReporter":
        """Support the context manager protocol.

        Returns:
            TrajectoryReporter: The reporter instance
        """
        return self

    def __exit__(self, *exc_info) -> None:
        """Support the context manager protocol.

        Closes all trajectory files when exiting the context.

        Args:
            *exc_info: Exception information
        """
        self.close()


class TorchSimTrajectory:
    """Trajectory storage and retrieval for molecular dynamics simulations.

    This class provides a low-level interface for reading and writing trajectory data
    to/from HDF5 files. It supports storing SimState objects, raw arrays, and
    conversion to common molecular modeling formats (ASE, pymatgen).

    Attributes:
        _file (tables.File): The HDF5 file handle
        array_registry (dict): Registry mapping array names to (shape, dtype) tuples
        type_map (dict): Mapping of numpy/torch dtypes to PyTables atom types

    Examples:
        >>> # Writing a trajectory
        >>> with TorchSimTrajectory('output.hdf5', mode='w') as traj:
        >>>     for step, state in enumerate(simulation):
        >>>         traj.write_state(state, step)
        >>>
        >>> # Reading a trajectory
        >>> with TorchSimTrajectory('output.hdf5', mode='r') as traj:
        >>>     state = traj.get_state(frame=10)
        >>>     structure = traj.get_structure(frame=-1)  # Last frame
    """

    def __init__(
        self,
        filename: str | pathlib.Path,
        *,
        mode: Literal["w", "a", "r"] = "r",
        compress_data: bool = True,
        coerce_to_float32: bool = True,
        coerce_to_int32: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Initialize the trajectory file.

        Args:
            filename (str | pathlib.Path): Path to the HDF5 file
            mode (Literal["w", "a", "r"]): Mode to open the file in. "w" will create
                a new file and overwrite any existing file, "a" will append to the
                existing file and "r" will open the file for reading only. Defaults to
                "r".
            compress_data (bool): Whether to compress the data using zlib compression.
                Defaults to True.
            coerce_to_float32 (bool): Whether to coerce float64 data to float32.
                Defaults to True.
            coerce_to_int32 (bool): Whether to coerce int64 data to int32.
                Defaults to False.
            metadata (dict[str, str], optional): Additional metadata to save in
                trajectory.

        Raises:
            ValueError: If the file cannot be opened or initialized
        """
        filename = pathlib.Path(filename)

        if compress_data:
            compression = tables.Filters(complib="zlib", shuffle=True, complevel=1)
        else:
            compression = None

        # TODO FIX THIS
        if handles := tables.file._open_files.get_handlers_by_name(str(filename)):
            list(handles)[-1].close()

        # create parent directory if it doesn't exist
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = tables.open_file(str(filename), mode=mode, filters=compression)

        self.array_registry: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

        # check if the header has already been written
        if "header" not in [node._v_name for node in self._file.list_nodes("/")]:
            self._initialize_header(metadata)

        self._initialize_registry()

        self.type_map = self._initialize_type_map(
            coerce_to_float32=coerce_to_float32, coerce_to_int32=coerce_to_int32
        )

    def _initialize_header(self, metadata: dict[str, str] | None = None) -> None:
        """Initialize the HDF5 file header with metadata.

        Creates the basic structure of the HDF5 file with header, metadata, data,
        and steps groups.

        Args:
            metadata (dict[str, str], optional): Metadata to store in the header.
        """
        self._file.create_group("/", "header")
        self._file.root.header._v_attrs.program = "TorchSim"
        self._file.root.header._v_attrs.title = "TorchSim Trajectory"

        self._file.create_group("/", "metadata")
        if metadata:
            for key, value in metadata.items():
                setattr(self._file.root.metadata._v_attrs, key, value)

        self._file.create_group("/", "data")
        self._file.create_group("/", "steps")

    def _initialize_registry(self) -> None:
        """Initialize the array registry from an existing file.

        Scans the HDF5 file to build a registry of array names, shapes, and data types
        for validation of subsequent write operations.
        """
        for node in self._file.list_nodes("/data/"):
            name = node.name
            dtype = node.dtype
            shape = tuple(int(ix) for ix in node.shape)[1:]
            self.array_registry[name] = (shape, dtype)

    def _initialize_type_map(
        self, *, coerce_to_float32: bool, coerce_to_int32: bool
    ) -> dict:
        """Initialize the type map for data type coercion.

        Creates a mapping from numpy/torch data types to PyTables atom types,
        with optional type coercion for reduced file size.

        Args:
            coerce_to_float32 (bool): Whether to coerce float64 data to float32
            coerce_to_int32 (bool): Whether to coerce int64 data to int32

        Returns:
            dict: Dictionary mapping numpy/torch dtypes to PyTables atom types
        """
        type_map = copy.copy(DATA_TYPE_MAP)
        if coerce_to_int32:
            type_map[torch.int64] = tables.Int32Atom()
            type_map[np.dtype("int64")] = tables.Int32Atom()
        if coerce_to_float32:
            type_map[torch.float64] = tables.Float32Atom()
            type_map[np.dtype("float64")] = tables.Float32Atom()
        return type_map

    def write_arrays(
        self,
        data: dict[str, np.ndarray | torch.Tensor],
        steps: int | list[int],
    ) -> None:
        """Write arrays to the trajectory file.

        This function is used to write arrays to the trajectory file. If steps is an
        integer, we assume that the arrays in data are for a single frame. If steps is
        a list, we assume that the arrays in data are for multiple frames. This
        determines whether we pad arrays with a first dimension of size 1.

        We also validate that the arrays are compatible with the existing arrays in the
        file and that the steps are monotonically increasing.

        Args:
            data (dict[str, np.ndarray | torch.Tensor]): Dictionary mapping array
                names to numpy arrays or torch tensors with shapes [n_frames, ...]
            steps (int | list[int]): Step number(s) for the frame(s) being written.
                If steps is an integer, arrays will be treated as single frame data.

        Raises:
            ValueError: If array shapes or dtypes don't match existing arrays,
                or if steps are not monotonically increasing
        """
        if isinstance(steps, int):
            pad_first_dim = True
            steps = [steps]
        else:
            pad_first_dim = False

        for name, array in data.items():
            # TODO: coerce dtypes to numpy
            if isinstance(array, torch.Tensor):
                array = array.cpu().detach().numpy()

            if pad_first_dim:
                # pad 1st dim of array with 1
                array = array[np.newaxis, ...]

            if name not in self.array_registry:
                self._initialize_array(name, array)

            self._validate_array(name, array, steps)
            self._serialize_array(name, array, steps)

        self.flush()

    def _initialize_array(self, name: str, array: np.ndarray) -> None:
        """Initialize a single array and add it to the registry.

        Creates a new array in the HDF5 file and registers its shape and dtype.

        Args:
            name (str): Name of the array
            array (np.ndarray): Array data to initialize with shape [n_frames, ...]

        Raises:
            ValueError: If the array dtype is not supported
        """
        if array.dtype not in self.type_map:
            raise ValueError(f"Unsupported dtype: {array.dtype}")

        self._file.create_earray(
            where="/data/",
            name=name,
            atom=self.type_map[array.dtype],
            shape=(0, *array.shape[1:]),
        )

        self._file.create_earray(
            where="/steps/",
            name=name,
            atom=tables.Int32Atom(),
            shape=(0,),
        )

        # in the registry we store the shape of the single-frame array
        # because the multi-frame array shape will change over time
        self.array_registry[name] = (array.shape[1:], array.dtype)

    def _validate_array(self, name: str, data: np.ndarray, steps: list[int]) -> None:
        """Validate that the data is compatible with the existing array.

        Checks that the array shape, dtype, and step numbers are compatible with
        the existing array in the file.

        Args:
            name (str): Name of the array
            data (np.ndarray): Array data to validate with shape [n_frames, ...]
            steps (list[int]): Step numbers to validate

        Raises:
            ValueError: If array shape or dtype doesn't match, or if steps aren't
                monotonically increasing
        """
        # Get the registered shape and dtype
        registered_shape, registered_dtype = self.array_registry[name]

        # Validate shape
        if data.shape[1:] != registered_shape:
            # TODO: update this message
            raise ValueError(
                f"Array {name} shape mismatch. Expected {registered_shape}, "
                f"got {data.shape}"
            )

        # Get the expected dtype from our type map
        expected_atom = self.type_map[data.dtype]
        stored_atom = self.type_map[registered_dtype]

        # Compare the PyTables atoms instead of numpy dtypes
        if type(expected_atom) is not type(stored_atom):
            raise ValueError(
                f"Array {name} dtype mismatch. Cannot convert {data.dtype} "
                f"to match stored dtype {registered_dtype}"
            )

        # Validate step is monotonically increasing by checking HDF5 file directly
        steps_node = self._file.get_node("/steps/", name=name)
        if len(steps_node) > 0:
            last_step = steps_node[-1]  # Get the last recorded step
            if steps[0] <= last_step:
                raise ValueError(
                    f"{steps[0]=} must be greater than the last recorded "
                    f"step {last_step} for array {name}"
                )

    def _serialize_array(self, name: str, data: np.ndarray, steps: list[int]) -> None:
        """Add additional contents to an array already in the registry.

        Appends frames to an existing array and its associated step numbers.

        Args:
            name (str): Name of the array
            data (np.ndarray): Array data to serialize with shape [n_frames, ...]
            steps (list[int]): Step numbers for the frames

        Raises:
            ValueError: If number of steps doesn't match number of frames
        """
        if len(steps) > 1 and data.shape[0] != len(steps):
            raise ValueError(
                f"Number of steps {len(steps)} must match the number of frames "
                f"{data.shape[0]} for array {name}"
            )

        self._file.get_node(where="/data/", name=name).append(data)
        self._file.get_node(where="/steps/", name=name).append(steps)

    def get_array(
        self,
        name: str,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
    ) -> np.ndarray:
        """Get an array from the file.

        Retrieves a subset of frames from the specified array.

        Args:
            name (str): Name of the array to retrieve
            start (int, optional): Starting frame index. Defaults to None.
            stop (int, optional): Ending frame index (exclusive). Defaults to None.
            step (int, optional): Step size between frames. Defaults to 1.

        Returns:
            np.ndarray: Array data as numpy array with shape [n_selected_frames, ...]

        Raises:
            ValueError: If array name not found in registry
        """
        if name not in self.array_registry:
            raise ValueError(f"Array {name} not found in registry")

        return self._file.root.data.__getitem__(name).read(
            start=start,
            stop=stop,
            step=step,
        )

    def get_steps(
        self,
        name: str,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
    ) -> np.ndarray:
        """Get the steps for an array.

        Retrieves the simulation step numbers associated with frames in an array.

        Args:
            name (str): Name of the array
            start (int, optional): Starting frame index. Defaults to None.
            stop (int, optional): Ending frame index (exclusive). Defaults to None.
            step (int, optional): Step size between frames. Defaults to 1.

        Returns:
            np.ndarray: Array of step numbers with shape [n_selected_frames]
        """
        return self._file.root.steps.__getitem__(name).read(
            start=start,
            stop=stop,
            step=step,
        )

    def __str__(self) -> str:
        """Get a string representation of the trajectory.

        Returns:
            str: Summary of arrays in the file including shapes and dtypes
        """
        # summarize arrays and steps in the file
        summary = ["Arrays in file:"]
        for node in self._file.list_nodes("/data/"):
            shape_ints = tuple(int(ix) for ix in node.shape)
            steps = shape_ints[0]
            shape = shape_ints[1:]
            dtype = node.dtype
            summary.append(
                f"  {node.name}: {steps} steps with shape {shape} and dtype {dtype}"
            )
        return "\n".join(summary)

    def write_state(  # noqa: C901
        self,
        state: SimState | list[SimState],
        steps: int | list[int],
        batch_index: int | None = None,
        *,
        save_velocities: bool = False,
        save_forces: bool = False,
        save_energy: bool = False,
        variable_cell: bool = False,
        variable_masses: bool = False,
        variable_atomic_numbers: bool = False,
    ) -> None:
        """Write a SimState or list of SimStates to the file.

        Extracts and stores position, velocity, force, and other data from
        SimState objects. Static data (like cell parameters) is stored only
        once unless flagged as variable.

        If a list, the states are assumed to be different configurations of
        the same system, representing a trajectory.

        Args:
            state (SimState | list[SimState]): SimState or list of SimStates to write
            steps (int | list[int]): Step number(s) for the frame(s)
            batch_index (int, optional): Batch index to save.
            save_velocities (bool, optional): Whether to save velocities.
            save_forces (bool, optional): Whether to save forces.
            save_energy (bool, optional): Whether to save energy.
            variable_cell (bool, optional): Whether the cell varies between frames.
            variable_masses (bool, optional): Whether masses vary between frames.
            variable_atomic_numbers (bool, optional): Whether atomic numbers vary
                between frames.

        Raises:
            ValueError: If number of states doesn't match number of steps or if
                required attributes are missing
        """
        # TODO: consider changing this reporting later

        # we wrap
        if isinstance(state, SimState):
            state = [state]
        if isinstance(steps, int):
            steps = [steps]

        if isinstance(batch_index, int):
            batch_index = [batch_index]
            sub_states = [state[batch_index] for state in state]
        elif batch_index is None and torch.unique(state[0].batch) == 0:
            batch_index = 0
            sub_states = state
        else:
            raise ValueError(
                "Batch index must be specified if there are multiple batches"
            )

        if len(sub_states) != len(steps):
            raise ValueError(
                f"Number of states {len(sub_states)} must match the number of steps "
                f"{len(steps)}"
            )
        # Initialize data dictionary with required arrays
        data = {
            "positions": torch.stack([s.positions for s in state]),
        }

        # Add optional arrays based on flags
        # Define optional arrays to save based on flags
        optional_arrays = {
            "velocities": save_velocities,
            "forces": save_forces,
            "energy": save_energy,
        }
        # Loop through optional arrays and add them if requested
        for array_name, should_save in optional_arrays.items():
            if should_save:
                if not hasattr(state[0], array_name):
                    raise ValueError(
                        f"{array_name.capitalize()} can only be saved "
                        f"if included in the state being reported."
                    )
                data[array_name] = torch.stack([getattr(s, array_name) for s in state])

        # Handle cell and masses based on variable flags
        if variable_cell:
            data["cell"] = torch.cat([s.cell for s in state])
        elif "cell" not in self.array_registry:  # Save cell only for first frame
            # we but cell in list because it doesn't need to be padded
            self.write_arrays({"cell": state[0].cell}, [0])

        if variable_masses:
            data["masses"] = torch.stack([s.masses for s in state])
        elif "masses" not in self.array_registry:  # Save masses only for first frame
            self.write_arrays({"masses": state[0].masses}, 0)

        if variable_atomic_numbers:
            data["atomic_numbers"] = torch.stack([s.atomic_numbers for s in state])
        elif (
            "atomic_numbers" not in self.array_registry
        ):  # Save atomic numbers only for first frame
            self.write_arrays({"atomic_numbers": state[0].atomic_numbers}, 0)

        if "pbc" not in self.array_registry:
            self.write_arrays({"pbc": np.array(state[0].pbc)}, 0)

        # Write all arrays to file
        self.write_arrays(data, steps)

    def _get_state_arrays(self, frame: int) -> dict[str, torch.Tensor]:
        """Get all available state tensors for a given frame.

        Retrieves all state-related arrays (positions, cell, masses, etc.) for a
        specific frame.

        Args:
            frame (int): Frame index to retrieve (-1 for last frame)

        Returns:
            dict[str, torch.Tensor]: Dictionary of tensor names to their values

        Raises:
            ValueError: If required arrays are missing from trajectory or frame is
                out of range
        """
        arrays: dict[str, np.ndarray] = {}

        # Get required data
        if "positions" not in self.array_registry:
            raise ValueError("Positions not found in trajectory so cannot get structure.")

        # check length of positions array
        n_frames = self._file.root.data.positions.shape[0]

        if frame < 0:
            frame = n_frames + frame

        if frame > n_frames:
            raise ValueError(f"{frame=} is out of range. Total frames: {n_frames}")

        arrays["positions"] = self.get_array("positions", start=frame, stop=frame + 1)[0]

        def return_prop(self: Self, prop: str, frame: int) -> np.ndarray:
            if self._file.root.data.cell.shape[0] > 1:  # Variable cell
                start, stop = frame, frame + 1
            else:  # Static cell
                start, stop = 0, 1
            return self.get_array(prop, start=start, stop=stop)[0]

        arrays["cell"] = np.expand_dims(return_prop(self, "cell", frame), axis=0)
        arrays["atomic_numbers"] = return_prop(self, "atomic_numbers", frame)
        arrays["masses"] = return_prop(self, "masses", frame)
        arrays["pbc"] = return_prop(self, "pbc", frame)

        return arrays

    def get_structure(self, frame: int = -1) -> Any:
        """Get a pymatgen Structure object for a given frame.

        Converts the state at the specified frame to a pymatgen Structure object
        for analysis and visualization.

        Args:
            frame (int, optional): Frame index to retrieve (-1 for last frame). Defaults
            to -1.

        Returns:
            Structure: Pymatgen Structure object for the specified frame

        Raises:
            ImportError: If pymatgen is not installed
        """
        from pymatgen.core import Structure

        arrays = self._get_state_arrays(frame)

        # Create pymatgen Structure
        # TODO: check if this is correct
        lattice = arrays["cell"][0].T  # pymatgen expects lattice matrix as rows
        species = [str(num) for num in arrays["atomic_numbers"]]

        return Structure(
            lattice=np.ascontiguousarray(lattice),
            species=species,
            coords=np.ascontiguousarray(arrays["positions"]),
            coords_are_cartesian=True,
            validate_proximity=False,
        )

    def get_atoms(self, frame: int) -> Any:
        """Get an ASE Atoms object for a given frame.

        Converts the state at the specified frame to an ASE Atoms object
        for analysis and visualization.

        Args:
            frame (int): Frame index to retrieve (-1 for last frame)

        Returns:
            Atoms: ASE Atoms object for the specified frame

        Raises:
            ImportError: If ASE is not installed
        """
        from ase import Atoms

        arrays = self._get_state_arrays(frame)

        pbc = arrays.get("pbc", True)

        return Atoms(
            numbers=np.ascontiguousarray(arrays["atomic_numbers"]),
            positions=np.ascontiguousarray(arrays["positions"]),
            cell=np.ascontiguousarray(arrays["cell"])[0],
            pbc=pbc,
        )

    def get_state(
        self,
        frame: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> SimState:
        """Get a SimState object for a given frame.

        Reconstructs a SimState object from the data stored for a specific frame.

        Args:
            frame (int): Frame index to retrieve (-1 for last frame)
            device (torch.device, optional): Device to place tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to None.

        Returns:
            SimState: State object containing all available data for the frame with
                shapes matching the original stored state
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or torch.float64

        arrays = self._get_state_arrays(frame)

        # Create SimState with required attributes
        return SimState(
            positions=torch.tensor(arrays["positions"], device=device, dtype=dtype),
            masses=torch.tensor(arrays.get("masses", None), device=device, dtype=dtype),
            cell=torch.tensor(arrays["cell"], device=device, dtype=dtype),
            pbc=torch.tensor(arrays.get("pbc", True), device=device, dtype=torch.bool),
            atomic_numbers=torch.tensor(
                arrays["atomic_numbers"], device=device, dtype=torch.int
            ),
        )

    def close(self) -> None:
        """Close the HDF5 file handle.

        Ensures all data is written to disk and releases the file handle.
        """
        # TODO: ???
        if self._file.isopen:
            self._file.close()

    def __enter__(self) -> "TorchSimTrajectory":
        """Support the context manager protocol.

        Returns:
            TorchSimTrajectory: The trajectory instance
        """
        return self

    def __exit__(self, *exc_info) -> None:
        """Support the context manager protocol.

        Closes the file when exiting the context.

        Args:
            *exc_info: Exception information
        """
        self.close()

    def flush(self) -> None:
        """Write all buffered data to the disk file.

        Forces any pending data to be written to the physical storage.
        """
        if self._file.isopen:
            self._file.flush()

    def __len__(self) -> int:
        """Get the number of frames in the trajectory.

        Returns:
            int: Number of frames in the trajectory
        """
        return self._file.root.data.positions.shape[0]

    def write_ase_trajectory(self, filename: str | pathlib.Path) -> Any:
        """Convert trajectory to ASE Trajectory format.

        Writes the entire trajectory to a new file in ASE format for compatibility
        with ASE analysis tools.

        Args:
            filename (str | pathlib.Path): Path to the output ASE trajectory file

        Returns:
            ase.io.trajectory.Trajectory: ASE trajectory object

        Raises:
            ImportError: If ASE is not installed
        """
        try:
            from ase.io.trajectory import Trajectory
        except ImportError as err:
            raise ImportError(
                "ASE is required to convert to ASE trajectory. "
                "Please install it with 'pip install ase'"
            ) from err

        # Create ASE trajectory
        traj = Trajectory(filename, mode="w")

        # Write each frame
        for frame in range(len(self)):
            atoms = self.get_atoms(frame)
            traj.write(atoms)

        traj.close()

        # Reopen in read mode
        return Trajectory(filename)
