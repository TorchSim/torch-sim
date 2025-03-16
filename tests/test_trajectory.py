import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.state import BaseState
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter
from torch_sim.integrators import MDState


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    """Temporary file path for testing."""
    return tmp_path / "test_trajectory_temp.h5"


@pytest.fixture
def random_state() -> MDState:
    """Create a random MDState for testing."""
    return MDState(
        positions=torch.randn(10, 3),
        momenta=torch.randn(10, 3),
        energy=torch.tensor(1.0),
        forces=torch.randn(10, 3),
        masses=torch.ones(
            10,
        ),
        cell=torch.unsqueeze(torch.eye(3) * 10.0, 0),
        atomic_numbers=torch.ones(10, dtype=torch.int32),
        batch=torch.zeros(10, dtype=torch.int32),
        pbc=True,
    )


@pytest.fixture
def trajectory(test_file: Path) -> TorchSimTrajectory:
    """Create a trajectory file for testing."""
    return TorchSimTrajectory(test_file, compress_data=True, mode="w")


def test_initialization(test_file: Path) -> None:
    """Test trajectory file initialization."""
    traj = TorchSimTrajectory(test_file, mode="w")
    assert os.path.exists(test_file)
    assert traj._file.isopen  # noqa: SLF001
    traj.close()


def test_write_arrays_basic(trajectory: TorchSimTrajectory) -> None:
    """Test writing basic arrays."""
    rng = np.random.default_rng(seed=0)
    positions = rng.random((10, 3)).astype(np.float32)
    velocities = rng.random((10, 3)).astype(np.float32)

    data = {"positions": positions, "velocities": velocities}
    trajectory.write_arrays(data, steps=0)

    assert "positions" in trajectory.array_registry
    assert "velocities" in trajectory.array_registry
    assert len(trajectory) == 1


def test_write_arrays_multiple_frames(trajectory: TorchSimTrajectory) -> None:
    """Test writing multiple frames."""
    rng = np.random.default_rng(seed=0)
    positions1 = rng.random((10, 3)).astype(np.float32)
    positions2 = positions1 + 0.1

    trajectory.write_arrays({"positions": positions1}, steps=0)
    trajectory.write_arrays({"positions": positions2}, steps=1)

    assert len(trajectory) == 2
    read_positions = trajectory.get_array("positions")
    assert read_positions.shape == (2, 10, 3)


def test_write_state_single(
    trajectory: TorchSimTrajectory, random_state: MDState
) -> None:
    """Test writing a single MDState."""
    trajectory.write_state(random_state, steps=0, save_energy=True)

    assert "positions" in trajectory.array_registry
    assert "energy" in trajectory.array_registry
    assert len(trajectory) == 1

    assert trajectory.get_array("positions").shape == (1, 10, 3)
    assert trajectory.get_array("energy").shape == (1,)
    assert trajectory.get_array("atomic_numbers").shape == (1, 10)
    assert trajectory.get_array("cell").shape == (1, 3, 3)
    assert trajectory.get_array("pbc").shape == (1,)


def test_write_state_multiple(
    trajectory: TorchSimTrajectory, random_state: MDState
) -> None:
    """Test writing multiple MDStates."""
    trajectory.write_state([random_state, random_state], [0, 1], save_energy=True)

    assert len(trajectory) == 2
    assert trajectory.get_array("positions").shape == (2, 10, 3)
    assert trajectory.get_array("energy").shape == (2,)
    assert trajectory.get_array("atomic_numbers").shape == (1, 10)
    assert trajectory.get_array("cell").shape == (1, 3, 3)
    assert trajectory.get_array("pbc").shape == (1,)


def test_optional_arrays(trajectory: TorchSimTrajectory, random_state: MDState) -> None:
    """Test writing optional arrays."""
    trajectory.write_state(
        random_state,
        steps=0,
        save_velocities=True,
        save_forces=True,
    )

    assert "velocities" in trajectory.array_registry
    assert "forces" in trajectory.array_registry


def test_variable_cell_masses_atomic_numbers(
    trajectory: TorchSimTrajectory, random_state: MDState
) -> None:
    """Test handling of variable cell and masses."""
    trajectory.write_state(
        [random_state, random_state],
        [0, 1],
        variable_cell=True,
        variable_masses=True,
        variable_atomic_numbers=True,
    )

    assert "cell" in trajectory.array_registry
    assert "masses" in trajectory.array_registry
    assert "atomic_numbers" in trajectory.array_registry

    assert trajectory.get_array("cell").shape == (2, 3, 3)
    assert trajectory.get_array("masses").shape == (2, 10)
    assert trajectory.get_array("atomic_numbers").shape == (2, 10)


def test_data_type_coercion(test_file: Path) -> None:
    """Test data type coercion options."""
    traj = TorchSimTrajectory(
        test_file,
        coerce_to_float32=True,
        coerce_to_int32=True,
        mode="w",
    )

    rng = np.random.default_rng(seed=0)
    float64_data = rng.random((10, 3)).astype(np.float64)
    int64_data = rng.integers(0, 10, (10, 3), dtype=np.int64)

    traj.write_arrays(
        {"float_array": float64_data, "int_array": int64_data},
        steps=0,
    )

    assert traj.get_array("float_array").dtype == np.float32
    assert traj.get_array("int_array").dtype == np.int32
    traj.close()


def test_invalid_writes(trajectory: TorchSimTrajectory) -> None:
    """Test error handling for invalid writes."""
    rng = np.random.default_rng(seed=0)
    positions = rng.random((10, 3)).astype(np.float32)
    trajectory.write_arrays({"positions": positions}, steps=0)

    # Test duplicate step
    with pytest.raises(ValueError, match="must be greater than the last recorded step"):
        trajectory.write_arrays({"positions": positions}, steps=0)

    # Test wrong shape
    wrong_shape = rng.random((11, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        trajectory.write_arrays({"positions": wrong_shape}, steps=1)


def test_context_manager(test_file: Path) -> None:
    """Test context manager protocol."""
    with TorchSimTrajectory(test_file, mode="w") as traj:
        assert traj._file.isopen  # noqa: SLF001
        rng = np.random.default_rng(seed=0)
        positions = rng.random((10, 3)).astype(np.float32)
        traj.write_arrays({"positions": positions}, steps=0)

    assert not traj._file.isopen  # noqa: SLF001


def test_get_steps(trajectory: TorchSimTrajectory) -> None:
    """Test step number retrieval."""
    rng = np.random.default_rng(seed=0)
    positions = rng.random((10, 3)).astype(np.float32)
    trajectory.write_arrays({"positions": positions}, steps=5)

    steps = trajectory.get_steps("positions")
    assert steps == [5]


def test_compression(test_file: Path) -> None:
    """Test file compression."""
    # Write same data with and without compression
    rng = np.random.default_rng(seed=0)
    data = rng.random((100, 3)).astype(np.float32)

    traj_compressed = TorchSimTrajectory(test_file, compress_data=True, mode="w")
    traj_compressed.write_arrays({"data": data}, steps=0)
    traj_compressed.close()
    size_compressed = os.path.getsize(test_file)

    traj_uncompressed = TorchSimTrajectory(
        test_file,
        compress_data=False,
        mode="w",
    )
    traj_uncompressed.write_arrays({"data": data}, steps=0)
    traj_uncompressed.close()
    size_uncompressed = os.path.getsize(test_file)

    assert size_compressed < size_uncompressed


def test_file_modes(test_file: Path) -> None:
    """Test different file opening modes."""
    # Write initial data
    traj = TorchSimTrajectory(test_file, mode="w")
    rng = np.random.default_rng(seed=0)
    positions = rng.random((10, 3)).astype(np.float32)
    traj.write_arrays({"positions": positions}, steps=0)
    traj.close()

    # Try to open in read mode
    traj_read = TorchSimTrajectory(test_file, mode="r")
    assert len(traj_read) == 1
    traj_read.close()

    # Try to append
    traj_append = TorchSimTrajectory(test_file, mode="a")
    positions2 = positions + 0.1
    traj_append.write_arrays({"positions": positions2}, steps=1)
    assert len(traj_append) == 2
    traj_append.close()


def test_data_type_conversions(test_file: Path) -> None:
    """Test various data type conversions for both numpy and torch tensors."""
    traj = TorchSimTrajectory(
        test_file,
        coerce_to_float32=True,
        coerce_to_int32=True,
        mode="w",
    )

    rng = np.random.default_rng(seed=0)
    # Test data with different types
    test_data = {
        # NumPy arrays
        "np_float64": rng.random((10, 3)).astype(np.float64),
        "np_float32": rng.random((10, 3)).astype(np.float32),
        "np_int64": rng.integers(0, 10, (10, 3), dtype=np.int64),
        "np_int32": rng.integers(0, 10, (10, 3), dtype=np.int32),
        "np_bool": rng.choice([True, False], (10, 3)),
        # PyTorch tensors
        "torch_float64": torch.randn(10, 3, dtype=torch.float64),
        "torch_float32": torch.randn(10, 3, dtype=torch.float32),
        "torch_int64": torch.randint(0, 10, (10, 3), dtype=torch.int64),
        "torch_int32": torch.randint(0, 10, (10, 3), dtype=torch.int32),
        "torch_bool": torch.randint(0, 2, (10, 3), dtype=torch.bool),
    }

    # Write all arrays
    traj.write_arrays(test_data, steps=0)

    # Expected dtype mappings with coercion
    expected_dtypes = {
        "np_float64": np.float32,  # Coerced to float32
        "np_float32": np.float32,  # Already float32
        "np_int64": np.int32,  # Coerced to int32
        "np_int32": np.int32,  # Already int32
        "np_bool": np.bool_,  # Bools unchanged
        "torch_float64": np.float32,  # Coerced to float32
        "torch_float32": np.float32,  # Already float32
        "torch_int64": np.int32,  # Coerced to int32
        "torch_int32": np.int32,  # Already int32
        "torch_bool": np.bool_,  # Bools unchanged
    }

    # Verify all dtypes
    for name, expected_dtype in expected_dtypes.items():
        stored_array = traj.get_array(name)
        assert stored_array.dtype == expected_dtype, f"Wrong dtype for {name}"

    traj.close()


def test_no_data_type_coercion(test_file: Path) -> None:
    """Test behavior when coercion is disabled."""
    traj = TorchSimTrajectory(
        test_file,
        coerce_to_float32=False,
        coerce_to_int32=False,
        mode="w",
    )

    rng = np.random.default_rng(seed=0)
    test_data = {
        "float64": rng.random((10, 3)).astype(np.float64),
        "int64": rng.integers(0, 10, (10, 3), dtype=np.int64),
    }

    traj.write_arrays(test_data, steps=0)

    # Verify dtypes are preserved
    assert traj.get_array("float64").dtype == np.float64
    assert traj.get_array("int64").dtype == np.int64

    traj.close()


def test_mixed_precision_writes(test_file: Path) -> None:
    """Test writing mixed precision data across multiple frames."""
    traj = TorchSimTrajectory(
        test_file,
        coerce_to_float32=True,
        mode="w",
    )

    # Write float64 first
    frame1 = {"positions": torch.randn(10, 3, dtype=torch.float64)}
    traj.write_arrays(frame1, steps=0)

    # Write float32 second
    frame2 = {"positions": torch.randn(10, 3, dtype=torch.float32)}
    traj.write_arrays(frame2, steps=1)

    # Both should be stored as float32
    stored_data = traj.get_array("positions")
    assert stored_data.dtype == np.float32
    assert stored_data.shape == (2, 10, 3)

    traj.close()


def test_invalid_dtype_handling(test_file: Path) -> None:
    """Test handling of unsupported data types."""
    traj = TorchSimTrajectory(test_file, mode="w")

    # Test complex numbers
    complex_data = {
        "complex": np.random.default_rng(seed=0).random((10, 3)).astype(np.float16)
    }
    with pytest.raises(ValueError, match="Unsupported dtype"):
        traj.write_arrays(complex_data, steps=0)

    # Test string data
    string_data = {"strings": np.array([["a", "b", "c"]] * 10)}
    with pytest.raises(ValueError, match="Unsupported dtype"):
        traj.write_arrays(string_data, steps=0)

    traj.close()


def test_scalar_dtype_handling(test_file: Path) -> None:
    """Test handling of scalar values with different dtypes."""
    traj = TorchSimTrajectory(
        test_file,
        coerce_to_float32=True,
        coerce_to_int32=True,
        mode="w",
    )

    scalar_data = {
        "float64_scalar": np.float64(1.0),
        "float32_scalar": np.float32(1.0),
        "int64_scalar": np.int64(1),
        "int32_scalar": np.int32(1),
        "bool_scalar": np.bool_(True),  # noqa: FBT003
        "torch_float_scalar": torch.tensor(1.0, dtype=torch.float64),
        "torch_int_scalar": torch.tensor(1, dtype=torch.int64),
        "torch_bool_scalar": torch.tensor(data=True),
    }

    traj.write_arrays(scalar_data, steps=0)

    # Verify scalar dtypes
    assert traj.get_array("float64_scalar").dtype == np.float32
    assert traj.get_array("int64_scalar").dtype == np.int32
    assert traj.get_array("bool_scalar").dtype == np.bool_
    assert traj.get_array("torch_float_scalar").dtype == np.float32
    assert traj.get_array("torch_int_scalar").dtype == np.int32
    assert traj.get_array("torch_bool_scalar").dtype == np.bool_

    traj.close()


def test_get_structure(trajectory: TorchSimTrajectory, random_state: MDState) -> None:
    """Test retrieving a pymatgen Structure from trajectory."""
    from pymatgen.core import Structure

    # Write a state to the trajectory
    trajectory.write_state(random_state, steps=0)

    # Get structure
    structure = trajectory.get_structure(frame=0)

    # Test return type
    assert isinstance(structure, Structure)

    # Test basic properties
    assert len(structure) == len(random_state.atomic_numbers)
    np.testing.assert_allclose(random_state.cell.numpy()[0], structure.lattice.matrix.T)
    np.testing.assert_allclose(structure.cart_coords, random_state.positions.numpy())

    # Test species assignment
    expected_species = [int(num) for num in random_state.atomic_numbers]
    assert [int(site.specie.Z) for site in structure] == expected_species


def test_get_atoms(trajectory: TorchSimTrajectory, random_state: MDState) -> None:
    """Test retrieving an ASE Atoms object from trajectory."""
    from ase import Atoms

    # Write a state to the trajectory
    trajectory.write_state(random_state, steps=0)

    # Get atoms
    atoms = trajectory.get_atoms(frame=0)

    # Test return type
    assert isinstance(atoms, Atoms)

    # Test basic properties
    assert len(atoms) == len(random_state.atomic_numbers)
    np.testing.assert_allclose(atoms.get_cell(), random_state.cell.numpy()[0])
    np.testing.assert_allclose(atoms.get_positions(), random_state.positions.numpy())
    np.testing.assert_allclose(
        atoms.get_atomic_numbers(), random_state.atomic_numbers.numpy()
    )
    assert atoms.pbc.all() == random_state.pbc


def test_get_state(trajectory: TorchSimTrajectory, random_state: MDState) -> None:
    """Test retrieving a BaseState object from trajectory."""
    # Write a state to the trajectory
    trajectory.write_state(random_state, steps=0)

    # Get state with different device/dtype combinations
    test_cases = [
        (torch.device("cpu"), torch.float32),  # Explicit CPU, float32
        (torch.device("cpu"), torch.float64),  # Explicit CPU, float64
    ]

    for device, dtype in test_cases:
        state = trajectory.get_state(frame=0, device=device, dtype=dtype)

        # Test basic properties
        assert state.positions.shape == random_state.positions.shape
        assert state.atomic_numbers.shape == random_state.atomic_numbers.shape
        assert state.cell.shape == random_state.cell.shape

        # Test device placement
        expected_device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        assert state.positions.device == expected_device
        assert state.cell.device == expected_device

        # Test dtype
        expected_dtype = dtype or torch.float64
        assert state.positions.dtype == expected_dtype
        assert state.cell.dtype == expected_dtype
        assert state.atomic_numbers.dtype == torch.int  # Should always be int

        # Test values (convert to CPU for comparison)
        np.testing.assert_allclose(state.positions, random_state.positions)
        np.testing.assert_allclose(state.cell, random_state.cell)
        np.testing.assert_allclose(state.atomic_numbers, random_state.atomic_numbers)
        assert state.pbc == random_state.pbc


def test_write_ase_trajectory(
    tmp_path: Path, trajectory: TorchSimTrajectory, random_state: MDState
) -> None:
    """Test converting a trajectory to an ASE Trajectory object."""
    from ase.io.trajectory import TrajectoryReader

    # Write states to the trajectory
    trajectory.write_state(random_state, steps=0)
    trajectory.write_state(random_state, steps=1)

    # Convert to ASE trajectory
    ase_traj = trajectory.write_ase_trajectory(tmp_path / "temp.traj")

    # Test that it's the right type
    assert isinstance(ase_traj, TrajectoryReader)

    # Test number of frames
    assert len(ase_traj) == 2

    # Test frame contents
    for _, atoms in enumerate(ase_traj):
        # Check basic properties match
        assert len(atoms) == len(random_state.atomic_numbers)
        np.testing.assert_allclose(atoms.get_cell(), random_state.cell.numpy()[0])
        np.testing.assert_allclose(
            atoms.get_positions(), random_state.positions.numpy()
        )
        np.testing.assert_allclose(
            atoms.get_atomic_numbers(), random_state.atomic_numbers.numpy()
        )
        assert atoms.pbc.all() == random_state.pbc

    # Clean up
    ase_traj.close()


@pytest.fixture
def prop_calculators() -> dict[int, dict[str, Callable]]:
    """Create property calculators for testing."""
    return {
        1: {  # Report every step
            "ones": lambda _: torch.ones(1),
            "center_of_mass": lambda state: torch.mean(
                state.positions * state.masses.unsqueeze(1)
            ),
        }
    }


def test_single_batch_reporter(
    si_base_state: BaseState, tmp_path: Path, prop_calculators: dict
) -> None:
    """Test TrajectoryReporter with a single batch."""
    # Create a reporter with a single file
    reporter = TrajectoryReporter(
        tmp_path / "single_batch.hdf5",
        state_frequency=1,
        prop_calculators=prop_calculators,
    )

    # Run several steps
    for step in range(5):
        reporter.report(si_base_state, step)

    reporter.close()

    # Verify file was created
    assert os.path.exists(tmp_path / "single_batch.hdf5")

    # Open trajectory and check contents
    trajectory = TorchSimTrajectory(tmp_path / "single_batch.hdf5", mode="r")

    # Check state data
    assert len(trajectory) == 5  # 5 frames
    assert "positions" in trajectory.array_registry
    assert "cell" in trajectory.array_registry
    assert "atomic_numbers" in trajectory.array_registry

    # Check property data
    assert "ones" in trajectory.array_registry
    assert "center_of_mass" in trajectory.array_registry
    assert trajectory.get_array("ones").shape[0] == 5  # 5 frames

    trajectory.close()


def test_multi_batch_reporter(
    si_double_base_state: BaseState, tmp_path: Path, prop_calculators: dict
) -> None:
    """Test TrajectoryReporter with multiple batches."""
    # Create a reporter with multiple files
    reporter = TrajectoryReporter(
        [tmp_path / "batch0.hdf5", tmp_path / "batch1.hdf5"],
        state_frequency=1,
        prop_calculators=prop_calculators,
    )

    # Run several steps
    for step in range(5):
        reporter.report(si_double_base_state, step)

    reporter.close()

    # Verify files were created
    assert os.path.exists(tmp_path / "batch0.hdf5")
    assert os.path.exists(tmp_path / "batch1.hdf5")

    # Open trajectories and check contents
    traj0 = TorchSimTrajectory(tmp_path / "batch0.hdf5", mode="r")
    traj1 = TorchSimTrajectory(tmp_path / "batch1.hdf5", mode="r")

    # Check state data in both files
    assert len(traj0) == 5  # 5 frames
    assert len(traj1) == 5  # 5 frames

    # Check that each trajectory has the correct number of atoms
    # (should be half of the total in the double state)
    atoms_per_batch = si_double_base_state.positions.shape[0] // 2
    assert traj0.get_array("positions").shape[1] == atoms_per_batch
    assert traj1.get_array("positions").shape[1] == atoms_per_batch

    # Check property data
    assert "ones" in traj0.array_registry
    assert "center_of_mass" in traj0.array_registry
    assert "ones" in traj1.array_registry
    assert "center_of_mass" in traj1.array_registry

    traj0.close()
    traj1.close()


def test_property_calculator_consistency(
    si_double_base_state: BaseState, tmp_path: Path, prop_calculators: dict
) -> None:
    """Test  property calculators are consistent for single and multi-batch cases."""
    # Create reporters for single and multi-batch cases
    single_reporters = []
    for batch_idx in range(2):
        # Extract single batch states
        single_state = si_double_base_state[batch_idx]
        reporter = TrajectoryReporter(
            tmp_path / f"single_{batch_idx}.hdf5",
            state_frequency=1,
            prop_calculators=prop_calculators,
        )
        # Run one step
        reporter.report(single_state, 0)
        reporter.close()
        single_reporters.append(
            TorchSimTrajectory(tmp_path / f"single_{batch_idx}.hdf5", mode="r")
        )

    # Create multi-batch reporter
    multi_reporter = TrajectoryReporter(
        [tmp_path / "multi_0.hdf5", tmp_path / "multi_1.hdf5"],
        state_frequency=1,
        prop_calculators=prop_calculators,
    )
    multi_reporter.report(si_double_base_state, 0)
    multi_reporter.close()

    multi_trajectories = [
        TorchSimTrajectory(tmp_path / "multi_0.hdf5", mode="r"),
        TorchSimTrajectory(tmp_path / "multi_1.hdf5", mode="r"),
    ]

    # Compare property values between single and multi-batch approaches
    for batch_idx in range(2):
        single_ke = single_reporters[batch_idx].get_array("ones")[0]
        multi_ke = multi_trajectories[batch_idx].get_array("ones")[0]
        assert torch.allclose(torch.tensor(single_ke), torch.tensor(multi_ke))

        single_com = single_reporters[batch_idx].get_array("center_of_mass")[0]
        multi_com = multi_trajectories[batch_idx].get_array("center_of_mass")[0]
        assert torch.allclose(torch.tensor(single_com), torch.tensor(multi_com))

    # Close all trajectories
    for traj in single_reporters + multi_trajectories:
        traj.close()


def test_reporter_with_model(
    si_double_base_state: BaseState, tmp_path: Path, lj_calculator: LennardJonesModel
) -> None:
    """Test TrajectoryReporter with a model argument in property calculators."""

    # Create a property calculator that uses the model
    def energy_calculator(state: BaseState, model: torch.nn.Module) -> torch.Tensor:
        output = model.forward(state.positions, state.cell)
        # Calculate a property that depends on the model
        return output["energy"]

    prop_calculators = {1: {"energy": energy_calculator}}

    # Create reporter
    reporter = TrajectoryReporter(
        [tmp_path / "model_0.hdf5", tmp_path / "model_1.hdf5"],
        state_frequency=1,
        prop_calculators=prop_calculators,
    )

    # Run with model
    reporter.report(si_double_base_state, 0, lj_calculator)
    reporter.close()

    # Verify property was calculated correctly
    trajectories = [
        TorchSimTrajectory(tmp_path / "model_0.hdf5", mode="r"),
        TorchSimTrajectory(tmp_path / "model_1.hdf5", mode="r"),
    ]

    for batch_idx, trajectory in enumerate(trajectories):
        # Get the property value
        energy = trajectory.get_array("energy")[0]

        # Calculate expected value
        substate = si_double_base_state[batch_idx]
        expected = lj_calculator.forward(substate.positions, substate.cell)["energy"]

        # Compare
        np.testing.assert_allclose(energy, expected.cpu().numpy())
        # assert energy not na
        assert not np.isnan(energy)
        assert len(energy)

        trajectory.close()
