"""Trajectory logging example."""

# %%
import numpy as np
import torch

from torchsim.trajectory import TorchSimTrajectory
from torchsim.unbatched_integrators import MDState


# Create test data
n_atoms = 10
n_frames = 2

positions = np.random.random((n_atoms, 3)).astype(np.float32)
velocities = np.random.random((n_atoms, 3)).astype(np.float32)

# Initialize trajectory file
traj = TorchSimTrajectory("test_trajectory.h5", compress_data=True, force_overwrite=True)

# Write first frame
data_frame1 = {"positions": positions, "velocities": velocities}
traj.write_arrays(data_frame1, steps=0)

# Write second frame with slightly modified data
positions2 = positions + 0.1
velocities2 = velocities + 0.1
data_frame2 = {"positions": positions2, "velocities": velocities2}
traj.write_arrays(data_frame2, steps=1)

print(traj.get_steps("positions", start=0, stop=2))

print(traj._file.list_nodes("/data/"))

print(traj)

print(len(traj))
traj.close()


# traj


# %%
state = MDState(
    positions=torch.randn(10, 3),  # 10 atoms in 3D
    momenta=torch.randn(10, 3),
    energy=torch.tensor(1.0),
    forces=torch.randn(10, 3),
    masses=torch.ones(10),
    cell=torch.eye(3) * 10.0,  # Cubic box of size 10
    pbc=torch.tensor(data=True),
)

traj2 = TorchSimTrajectory("test_trajectory2.h5")

traj2.write_state([state, state], [0, 1])

print(traj2)

read_positions = traj2.get_array("positions", start=0, stop=2)

print(traj2.get_array("potential_energy"))

print(type(traj2.get_array("potential_energy")))

print(traj2.get_array("potential_energy").shape)

print(traj2.get_steps("positions"))

traj2.close()


# %%
traj.close()
traj2.close()


# %%
traj.get_steps("positions")


# %%
[node._v_name for node in traj._file.list_nodes("/")]


# %%
traj._file.list_nodes("/")[0]._v_name


# %%
traj._file._create_earray


# %%
traj.get_steps("cell", start=0, stop=4)


# %%
# traj.close()

print(traj)


# %%
traj._file.isopen


# %%
# access the /steps/velocities array using pytables syntax
# traj._file.root.data.list_nodes()
traj._file.list_nodes("/data")[0].dtype
int(traj._file.list_nodes("/data")[0].shape[0])
tuple(int(ix) for ix in traj._file.list_nodes("/data")[0].shape)[1:]
traj._file.list_nodes("/data")[0].name


# %%
print(traj._array_registry)


# %%
# close the file
traj.close()
traj2 = TorchSimTrajectory(test_file, mode="r")


# %%
# Initialize trajectory file


# %%
traj._file.root.steps.positions.read()


# %%
# # Test writing with invalid step (should raise error)
# with pytest.raises(ValueError):
#     traj.write_arrays(data_frame2, step=0)  # Duplicate step

# # Test writing with wrong shape (should raise error)
# wrong_shape = np.random.random((n_atoms + 1, 3)).astype(np.float32)
# with pytest.raises(ValueError):
#     traj.write_arrays({"positions": wrong_shape}, step=2)

# # Test writing with wrong dtype (should raise error)
# wrong_dtype = np.random.random((n_atoms, 3)).astype(np.float64)
# with pytest.raises(ValueError):
#     traj.write_arrays({"positions": wrong_dtype}, step=2)

# traj.close()

# # Verify file exists and has non-zero size
# assert os.path.exists(test_file)
# assert os.path.getsize(test_file) > 0
