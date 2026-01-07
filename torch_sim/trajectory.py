"""Trajectory format and reporting.

This module provides classes for reading and writing trajectory data in HDF5 format.
The core classes (TorchSimTrajectory and TrajectoryReporter) allow efficient storage
and retrieval of atomic positions, forces, energies, and other properties from
molecular dynamics simulations.

The TorchSimTrajectory does not aim to be a new trajectory standard, but rather
a simple interface for storing and retrieving trajectory data from HDF5 files.
It aims to support arbitrary arrays from the user in a natural way, allowing
it to be seamlessly extended to whatever attributes are important to the user.

Example:
    Reading and writing a trajectory file::

        # Writing to multiple trajectory files with a reporter
        reporter = TrajectoryReporter(["traj1.hdf5", "traj2.hdf5"], state_frequency=100)
        reporter.report(state, step=0, model=model)

        # Reading the file with a TorchSimTrajectory
        with TorchSimTrajectory("simulation.hdf5", mode="r") as traj:
            state = traj.get_state(frame=0)

Notes:
    This module uses PyTables (HDF5) for efficient I/O operations and supports
    compression to reduce file sizes. It can interoperate with ASE and pymatgen
    for visualization and analysis.
"""

import copy
import inspect
import pathlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import tables
import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

if TYPE_CHECKING:
    from ase import Atoms
    from ase.io.trajectory import TrajectoryReader

_DATA_TYPE_MAP = {
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
        prop_calculators (dict): Map of frequencies to property calculators
        state_kwargs (dict): Additional arguments for state writing
        metadata (dict): Metadata to save in trajectory files
        trajectories (list): TorchSimTrajectory instances
        filenames (list): Trajectory file paths
        array_registry (dict): Map of array names to (shape, dtype) tuples

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

    state_frequency: int
    trajectory_kwargs: dict[str, Any]
    prop_calculators: dict[int, dict[str, Callable]]
    state_kwargs: dict[str, Any]
    metadata: dict[str, str] | None
    trajectories: list["TorchSimTrajectory"]

    def __init__(
        self,
        filenames: str | pathlib.Path | Sequence[str | pathlib.Path] | None,
        state_frequency: int = 100,
        *,
        prop_calculators: dict[int, dict[str, Callable]] | None = None,
        state_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, str] | None = None,
        trajectory_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a TrajectoryReporter.

        Args:
            filenames (str | pathlib.Path | list[str | pathlib.Path]): Path(s) to
                save trajectory file(s). If None, the reporter will not save any
                trajectories but `TrajectoryReporter.report` can still
                be used to compute properties directly.
            state_frequency (int): How often to save state (in steps)
            prop_calculators (dict[int, dict[str, Callable]], optional): Map of
                frequencies to property calculators where each calculator is a
                function that takes a state and optionally a model and returns a tensor.
                Defaults to None.
            state_kwargs (dict, optional): Additional arguments for state writing.
                Passed to the `TorchSimTrajectory.write_state` method. These can be
                set to save the velocities and forces or to allow variable masses,
                and atomic numbers across the trajectory.
            metadata (dict[str, str], optional): Metadata to save in trajectory file.
            trajectory_kwargs (dict, optional): Additional arguments for trajectory
                initialization. Passed to the `TorchSimTrajectory.__init__` method.

        Raises:
            ValueError: If filenames are not unique
        """
        self.state_frequency = state_frequency
        self.trajectory_kwargs = trajectory_kwargs or {}
        # default is to force overwrite
        self.trajectory_kwargs["mode"] = self.trajectory_kwargs.get("mode", "w")

        self.prop_calculators = prop_calculators or {}
        self.state_kwargs = state_kwargs or {}
        self.metadata = metadata

        self.trajectories = []
        if filenames is not None:
            filenames = (
                [filenames]
                if isinstance(filenames, (str, pathlib.Path))
                else list(filenames)
            )
            # Initialize trajectories for the first time. Unlike in reopen_trajectories,
            # if the user specified "w" mode, we respect that here and start fresh.
            self.trajectories = [
                TorchSimTrajectory(
                    filename=filename, metadata=self.metadata, **self.trajectory_kwargs
                )
                for filename in filenames
            ]

        self._add_model_arg_to_prop_calculators()

    @property
    def filenames(self) -> list[str] | None:
        """Get the list of trajectory filenames.

        Returns:
            list[str] | None: List of trajectory file paths,
                or None if no trajectories are loaded.
        """
        if not self.trajectories:
            return None
        return [traj.filename for traj in self.trajectories]

    def reopen_trajectories(
        self, filenames: str | pathlib.Path | Sequence[str | pathlib.Path]
    ) -> None:
        """Closes any existing trajectory files and reopens new ones given by filenames.

        Args:
            filenames (str | pathlib.Path | list[str | pathlib.Path]): Path(s) to save
                trajectory file(s)

        Raises:
            ValueError: If filenames are not unique
        """
        self.finish()

        filenames = (
            [filenames] if isinstance(filenames, (str, pathlib.Path)) else list(filenames)
        )
        filenames = [pathlib.Path(filename) for filename in filenames]
        if len(set(filenames)) != len(filenames):
            raise ValueError("All filenames must be unique.")
        # Avoid wiping existing trajectory files when reopening them, hence
        # we set to "a" mode temporarily (read mode is unaffected).
        _mode = self.trajectory_kwargs.get("mode", "w")
        self.trajectory_kwargs["mode"] = "a" if _mode in ["a", "w"] else "r"
        self.trajectories = [
            TorchSimTrajectory(
                filename=filename,
                metadata=self.metadata,
                **self.trajectory_kwargs,
            )
            for filename in filenames
        ]
        # Restore original mode
        self.trajectory_kwargs["mode"] = _mode

    @property
    def array_registry(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        """Registry of array shapes and dtypes."""
        # Return the registry from the first trajectory
        if self.trajectories:
            return self.trajectories[0].array_registry
        return {}

    def truncate_to_step(self, step: int) -> None:
        """Truncate all trajectory files to the specified step.
        **WARNING**: This operation is irreversible and will remove data from
        the trajectory files.

        Args:
            step (int): The step to truncate to.
        """
        if step <= 0:
            raise ValueError(f"Step must be greater than 0. Got step={step}.")
        last_steps = self.last_steps
        if any(s is None for s in last_steps):
            raise ValueError("Cannot truncate: one or more trajectories are empty.")
        if step > min(last_steps):
            raise ValueError(
                f"Step {step} is greater than the minimum last step "
                f"across trajectories ({min(last_steps)})."
            )
        for trajectory in self.trajectories:
            # trajectory file could be closed
            if trajectory._file.isopen:
                trajectory.truncate_to_step(step)
            else:
                with TorchSimTrajectory(trajectory.filename, mode="a") as traj:
                    traj.truncate_to_step(step)

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
                    new_fn = partial(
                        lambda state, _=None, fn=None: (
                            None if fn is None else fn(state)
                        ),
                        fn=prop_fn,
                    )
                    self.prop_calculators[frequency][name] = new_fn

    def report(
        self, state: SimState, step: int | list[int], model: ModelInterface | None = None
    ) -> list[dict[str, torch.Tensor]]:
        """Report a state and step to the trajectory files.

        Writes states and calculated properties to all trajectory files at the
        specified frequencies. Splits multi-system states across separate trajectory
        files. The number of systems must match the number of trajectory files.

        Args:
            state (SimState): Current system state with n_systems equal to
                len(filenames)
            step (int | list[int]): Current simulation step per system, setting step
                to 0 will write the state and all properties. If a list is provided, it
                must have length equal to n_systems. Otherwise, a single integer step
                is broadcast to all systems.
            model (ModelInterface, optional): Model used for simulation.
                Defaults to None. Must be provided if any prop_calculators
                are provided.
            write_to_file (bool, optional): Whether to write the state to the trajectory
                files. Defaults to True. Should only be set to `False` if the props
                are being collected separately.

        Returns:
            list[dict[str, torch.Tensor]]: Map of property names to tensors for each
                system.

        Raises:
            ValueError: If number of systems doesn't match number of trajectory files
        """
        # Get unique system indices
        system_indices = range(state.n_systems)
        # system_indices = torch.unique(state.system_idx).cpu().tolist()

        # Ensure we have the right number of trajectories
        if self.filenames is not None and len(system_indices) != len(self.trajectories):
            raise ValueError(
                f"Number of systems ({len(system_indices)}) doesn't match "
                f"number of trajectory files ({len(self.trajectories)})"
            )

        split_states = state.split()
        all_props: list[dict[str, torch.Tensor]] = []
        # Process each system separately
        for idx, substate in enumerate(split_states):
            sys_step = step[idx] if isinstance(step, list) else step
            # Write state to trajectory if it's time
            if (
                self.state_frequency
                and sys_step % self.state_frequency == 0
                and self.filenames is not None
            ):
                self.trajectories[idx].write_state(
                    substate, sys_step, **self.state_kwargs
                )

            all_state_props = {}
            # Process property calculators for this system
            for report_frequency, calculators in self.prop_calculators.items():
                if sys_step % report_frequency != 0 or report_frequency == 0:
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
                    all_state_props.update(props)
                    if self.filenames is not None:
                        self.trajectories[idx].write_arrays(props, sys_step)
            all_props.append(all_state_props)

        return all_props

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

    @property
    def mode(self) -> Literal["r", "w", "a"]:
        """Get the mode of the first trajectory file.

        Returns:
            "r" | "w" | "a": Mode from the trajectory_kwargs used during initialization.
        """
        if not self.trajectories:
            raise ValueError("No trajectories loaded.")
        # Key is guaranteed to exist because we set it during initialization.
        return self.trajectory_kwargs["mode"]

    @property
    def last_steps(self) -> list[int | None]:
        """Get the last logged step across all trajectory files.

        This is useful for resuming optimizations from where they left off.

        Returns:
            list[int | None]: The last step number for each trajectory, or None if
                the trajectory is empty. Returns empty list if no trajectories exist.
        """
        if not self.trajectories:
            return []
        last_steps = []
        for trajectory in self.trajectories:
            if trajectory._file.isopen:
                last_steps.append(trajectory.last_step)
            else:
                with TorchSimTrajectory(trajectory._file.filename, mode="r") as traj:
                    last_steps.append(traj.last_step)
        return last_steps

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
            mode ("w" | "a" | "r"): Mode to open the file in. "w" will create
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
        if hasattr(tables, "file") and (
            handles := tables.file._open_files.get_handlers_by_name(str(filename))
        ):
            list(handles)[-1].close()

        # create parent directory if it doesn't exist
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = tables.open_file(str(filename), mode=mode, filters=compression)

        self.array_registry: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

        # check if the header has already been written
        if "header" not in (node._v_name for node in self._file.list_nodes("/")):
            self._initialize_header(metadata)

        self._initialize_registry()

        self.type_map = self._initialize_type_map(
            coerce_to_float32=coerce_to_float32, coerce_to_int32=coerce_to_int32
        )
        if mode == "a" and self.last_step is not None:
            inconsistent_step = any(
                self.get_steps(name)[-1] > self.last_step for name in self.array_registry
            )
            if inconsistent_step:
                warnings.warn(
                    "Inconsistent last steps detected in trajectory arrays. "
                    "Truncating all arrays to the `positions` array's last step.",
                    stacklevel=2,
                )
                self.truncate_to_step(self.last_step)

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
            dict: Map of numpy/torch dtypes to PyTables atom types
        """
        type_map = copy.copy(_DATA_TYPE_MAP)
        if coerce_to_int32:
            type_map[torch.int64] = tables.Int32Atom()
            type_map[np.dtype("int64")] = tables.Int32Atom()
        if coerce_to_float32:
            type_map[torch.float64] = tables.Float32Atom()
            type_map[np.dtype("float64")] = tables.Float32Atom()
        return type_map

    def write_arrays(
        self,
        data: "Mapping[str, np.ndarray | np.generic | torch.Tensor]",
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
            data (Mapping[str, np.ndarray | np.generic | torch.Tensor]): Map of array
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
            # Normalize to numpy arrays
            if isinstance(array, torch.Tensor):
                array = array.cpu().detach().numpy()
            elif not isinstance(array, np.ndarray):
                # Convert numpy scalar (np.generic) or Python scalar to ndarray
                array = np.array(array)

            if pad_first_dim:
                # pad 1st dim of array with 1
                array = array[np.newaxis, ...]

            if name not in self.array_registry:
                self._initialize_array(name, array)

            self._validate_array(name, array, steps)
            self._serialize_array(name, array, steps)

        self.flush()

    def write_global_array(self, name: str, array: np.ndarray | torch.Tensor) -> None:
        """Write a global array to the trajectory file.

        This function is used to write a global array to the trajectory file.
        """
        if isinstance(array, torch.Tensor):
            array = array.cpu().detach().numpy()

        steps = [0]
        if name not in self.array_registry:
            self._initialize_array(name, array)
        self._validate_array(name, array, steps)
        self._serialize_array(name, array, steps)

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
            raise ValueError(f"Unsupported {array.dtype=}")

        self._file.create_earray(
            where="/data/",
            name=name,
            atom=self.type_map[array.dtype],
            shape=(0, *array.shape[1:]),
        )

        self._file.create_earray(
            where="/steps/", name=name, atom=tables.Int32Atom(), shape=(0,)
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
        steps_node = self.get_steps(name)
        if len(steps_node) > 0:
            last_step = steps_node[-1]  # Get the last recorded step
            if steps[0] <= last_step:
                raise ValueError(
                    f"{steps[0]=} must be greater than the last recorded "
                    f"step {last_step} for array {name}"
                )

    @property
    def filename(self) -> str:
        """Get the filename of the trajectory file.

        Returns:
            str: Path to the HDF5 file
        """
        return self._file.filename

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
            start=start, stop=stop, step=step
        )

    def get_steps(
        self,
        name: str,
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
        return self._file.get_node("/steps/", name=name).read()

    @property
    def last_step(self) -> int | None:
        """Get the last step number from the trajectory.

        Retrieves the last time step recorded in the trajectory based
        on the "positions" array.

        Returns:
            int | None: The last recorded step number, or None if no data exists
        """
        if not self.array_registry or "positions" not in self.array_registry:
            return None
        return self.get_steps("positions")[-1].item()

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
            summary.append(f"  {node.name}: {steps=} with {shape=} and {dtype=}")
        return "\n".join(summary)

    def write_state(  # noqa: C901
        self,
        state: SimState | list[SimState],
        steps: int | list[int],
        system_index: int | None = None,
        *,
        save_velocities: bool = False,
        save_forces: bool = False,
        variable_cell: bool = True,
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
            system_index (int, optional): System index to save.
            save_velocities (bool, optional): Whether to save velocities.
            save_forces (bool, optional): Whether to save forces.
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

        if isinstance(system_index, int):
            sub_states = [state[system_index] for state in state]
        elif system_index is None and torch.unique(state[0].system_idx) == 0:
            sub_states = state
        else:
            raise ValueError(
                "System index must be specified if there are multiple systems"
            )

        if len(sub_states) != len(steps):
            raise ValueError(f"{len(sub_states)=} must match the {len(steps)=}")

        # Initialize data dictionary with required arrays
        data = {
            "positions": torch.stack([s.positions for s in state]),
        }

        # Add optional arrays based on flags
        # Define optional arrays to save based on flags
        optional_arrays = {
            "velocities": save_velocities,
            "forces": save_forces,
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
        elif "atomic_numbers" not in self.array_registry:
            # Save atomic numbers only for first frame
            self.write_arrays({"atomic_numbers": state[0].atomic_numbers}, 0)

        if "pbc" not in self.array_registry:
            self.write_global_array("pbc", state[0].pbc)

        # Write all arrays to file
        self.write_arrays(data, steps)

    def _get_state_arrays(self, frame: int) -> dict[str, np.ndarray]:
        """Get all available state tensors for a given frame.

        Retrieves all state-related arrays (positions, cell, masses, etc.) for a
        specific frame.

        Args:
            frame (int): Frame index to retrieve (-1 for last frame)

        Returns:
            dict[str, np.ndarray]: Map of array names to their values

        Raises:
            ValueError: If required arrays are missing from trajectory or frame is
                out of range
        """
        arrays: dict[str, np.ndarray] = {}

        # Get required data
        if "positions" not in self.array_registry:
            keys = list(self.array_registry)
            raise ValueError(
                f"Positions not found in trajectory so cannot get structure. Have {keys=}"
            )

        # check length of positions array
        n_frames = self._file.root.data.positions.shape[0]

        if frame < 0:
            frame = n_frames + frame

        if frame > n_frames:
            raise ValueError(f"{frame=} is out of range. Total frames: {n_frames:,}")

        arrays["positions"] = self.get_array("positions", start=frame, stop=frame + 1)[0]

        def return_prop(self: Self, prop: str, frame: int) -> np.ndarray:
            if prop == "pbc":
                return self.get_array(prop, start=0, stop=3)
            if getattr(self._file.root.data, prop).shape[0] > 1:  # Variable prop
                start, stop = frame, frame + 1
            else:  # Static prop
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
            frame (int, optional): Frame index to retrieve. Defaults to -1 for last frame.

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

    def get_atoms(self, frame: int = -1) -> "Atoms":
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
        try:
            from ase import Atoms
        except ImportError:
            raise ImportError(
                "ASE is required to convert to ASE Atoms. Run `pip install ase`"
            ) from None

        arrays = self._get_state_arrays(frame)

        return Atoms(
            numbers=np.ascontiguousarray(arrays["atomic_numbers"]),
            positions=np.ascontiguousarray(arrays["positions"]),
            cell=np.ascontiguousarray(arrays["cell"])[0],
            pbc=np.ascontiguousarray(arrays["pbc"]),
        )

    def get_state(
        self,
        frame: int = -1,
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
            pbc=torch.tensor(arrays["pbc"], device=device, dtype=torch.bool),
            atomic_numbers=torch.tensor(
                arrays["atomic_numbers"], device=device, dtype=torch.int
            ),
        )

    @property
    def metadata(self) -> dict:
        """Metadata for the trajectory."""
        attrs = self._file.root.metadata._v_attrs
        return {name: getattr(attrs, name) for name in attrs._f_list()}

    def close(self) -> None:
        """Close the HDF5 file handle.

        Ensures all data is written to disk and releases the file handle.
        """
        if self._file.isopen:  # TODO: ???
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

    def write_ase_trajectory(self, filename: str | pathlib.Path) -> "TrajectoryReader":
        """Convert trajectory to ASE Trajectory format.

        Writes the entire trajectory to a new file in ASE format for compatibility
        with ASE analysis tools.

        Args:
            filename (str | pathlib.Path): Path to the output ASE trajectory file

        Returns:
            ase.io.trajectory.TrajectoryReader: ASE trajectory object

        Raises:
            ImportError: If ASE is not installed
        """
        try:
            from ase.io.trajectory import Trajectory
        except ImportError:
            raise ImportError(
                "ASE is required to convert to ASE trajectory. Run `pip install ase`"
            ) from None

        # Create ASE trajectory
        traj = Trajectory(filename, mode="w")

        # Write each frame
        for frame in range(len(self)):
            atoms = self.get_atoms(frame)
            traj.write(atoms)

        traj.close()
        return Trajectory(filename, mode="r")  # Reopen in read mode

    def truncate_to_step(self, step: int) -> None:
        """Truncate the trajectory to a specified step.
        **WARNING**: This operation is irreversible and will permanently
        modify the trajectory file.

        Removes frames from the end of the trajectory to reduce its length such that the
        last logged step is `step`.

        Args:
            step (int): Desired last step of the trajectory after truncation
        """
        if self.last_step is None:
            raise ValueError(
                "Cannot truncate an empty trajectory (no data has been written)."
            )
        if self.last_step < step:
            raise ValueError(
                f"Cannot truncate to a step greater than the last step."
                f" {self.last_step=} < {step=}"
            )
        if self.last_step == step:
            return  # No truncation needed
        if step <= 0:
            raise ValueError(f"Step must be larger than 0. Got {step=}")
        for name in self.array_registry:
            steps_node = self._file.get_node("/steps/", name=name)
            steps_data = steps_node.read()
            if set(steps_data) == {0}:
                continue  # skip global arrays
            # Find the index where the step is less than or equal to the desired step
            # We know that it must be at least one index because of the earlier check.
            indices = np.where(steps_data <= step)[0]
            length = indices[-1] + 1  # +1 because we want to include this index

            data_node = self._file.get_node("/data/", name=name)
            data_node.truncate(length)
            steps_node.truncate(length)

        self.flush()
