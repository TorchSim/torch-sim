# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[io]"
# ]
# ///


# %% [markdown]
"""
# Reproducible Restarts from Stopped Simulations

This tutorial demonstrates how to save and restore simulation state to enable
reproducible restarts. We run 50 steps of MD, save the state (including RNG state),
resume for another 50 steps, and verify the result is identical to 100 uninterrupted
steps.

For stochastic integrators like Langevin dynamics, you must save the random number
generator (RNG) state alongside positions, momenta, and other state variables.
Without it, the stochastic noise will differ on restart and the trajectory will diverge.
"""

# %% [markdown]
"""
## Setup
"""

# %%
from dataclasses import asdict
from pathlib import Path

import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.integrators import MDState
from torch_sim.models.lennard_jones import LennardJonesModel

# All generated files go in this directory
restart_dir = Path("restart_files")
restart_dir.mkdir(exist_ok=True)

seed = 42
torch.manual_seed(seed)

lj_model = LennardJonesModel(
    sigma=2.0,
    epsilon=0.1,
    device=torch.device("cpu"),
    dtype=torch.float64,
)

si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)

initial_state = ts.initialize_state(
    si_atoms, device=torch.device("cpu"), dtype=torch.float64
)
initial_state.rng = seed  # seed the SimState RNG for reproducibility

print(f"Initial state: {initial_state.n_atoms} atoms")

# %% [markdown]
"""
## Part 1: Run 50 Steps, Save State, Resume for 50 More

We save the complete state with `asdict()` + `torch.save()`. Since `torch.save()`
uses pickle, the `torch.Generator` (RNG) is included automatically — no need to
save it separately.

**PyTorch 2.6+**: You must pass `weights_only=False` to `torch.load()` when loading
checkpoints that contain `torch.Generator` objects.
"""

# %%
# Run first 50 steps
trajectory_file_restart = str(restart_dir / "restart_trajectory.h5")
reporter_restart = ts.TrajectoryReporter(
    filenames=trajectory_file_restart,
    state_frequency=10,
    state_kwargs={"save_velocities": True},
)

state_after_50 = ts.integrate(
    system=initial_state.clone(),
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=50,
    temperature=300,
    timestep=0.001,
    trajectory_reporter=reporter_restart,
)
reporter_restart.close()

# Save the complete state (including RNG) in one file
checkpoint_file = str(restart_dir / "checkpoint.pt")
torch.save(asdict(state_after_50), checkpoint_file)
print(f"Saved checkpoint after 50 steps to {checkpoint_file}")

# %% [markdown]
"""
Now restore the state and continue for another 50 steps:
"""

# %%
# Load checkpoint (weights_only=False needed for torch.Generator in PyTorch 2.6+)
loaded = torch.load(checkpoint_file, weights_only=False)
restored_state = MDState(**loaded)

# Verify RNG was restored
assert torch.equal(restored_state.rng.get_state(), state_after_50.rng.get_state())
print(f"Restored state: {restored_state.n_atoms} atoms, RNG matches ✓")

# Continue for another 50 steps (append to existing trajectory)
reporter_restart_continued = ts.TrajectoryReporter(
    filenames=trajectory_file_restart,
    state_frequency=10,
    state_kwargs={"save_velocities": True},
    trajectory_kwargs={"mode": "a"},
)

state_after_100_restart = ts.integrate(
    system=restored_state,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=50,
    temperature=300,
    timestep=0.001,
    trajectory_reporter=reporter_restart_continued,
)
reporter_restart_continued.close()
print(f"Completed restart simulation: 50 + 50 = 100 steps")

# %% [markdown]
"""
## Part 2: Run 100 Steps Continuously for Comparison
"""

# %%
trajectory_file_continuous = str(restart_dir / "continuous_trajectory.h5")
reporter_continuous = ts.TrajectoryReporter(
    filenames=trajectory_file_continuous,
    state_frequency=10,
    state_kwargs={"save_velocities": True},
)

initial_state_continuous = ts.initialize_state(
    si_atoms, device=torch.device("cpu"), dtype=torch.float64
)
initial_state_continuous.rng = seed

state_after_100_continuous = ts.integrate(
    system=initial_state_continuous,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100,
    temperature=300,
    timestep=0.001,
    trajectory_reporter=reporter_continuous,
)
reporter_continuous.close()
print(f"Completed continuous simulation: 100 steps")

# %% [markdown]
"""
## Part 3: Compare Trajectories

Both runs started from the same initial state and seed. The restarted run saved and
restored the RNG state at step 50. If everything is correct, the trajectories should
match exactly:
"""

# %%
# Compare final RNG states
rng_match = torch.equal(
    state_after_100_restart.rng.get_state(),
    state_after_100_continuous.rng.get_state(),
)
print(f"Final RNG states match: {rng_match}")

# Compare trajectories frame by frame
with ts.TorchSimTrajectory(trajectory_file_restart, mode="r") as traj_restart:
    positions_restart = traj_restart.get_array("positions")
    steps_restart = traj_restart.get_steps("positions")
    velocities_restart = traj_restart.get_array("velocities")

with ts.TorchSimTrajectory(trajectory_file_continuous, mode="r") as traj_continuous:
    positions_continuous = traj_continuous.get_array("positions")
    steps_continuous = traj_continuous.get_steps("positions")
    velocities_continuous = traj_continuous.get_array("velocities")

matching_steps = sorted(set(steps_restart) & set(steps_continuous))
print(f"Comparing {len(matching_steps)} frames at steps: {matching_steps}")

max_pos_diff = 0.0
max_vel_diff = 0.0
all_match = True

for step in matching_steps:
    idx_restart = steps_restart.tolist().index(step)
    idx_continuous = steps_continuous.tolist().index(step)

    pos_restart = torch.tensor(positions_restart[idx_restart])
    pos_continuous = torch.tensor(positions_continuous[idx_continuous])
    vel_restart = torch.tensor(velocities_restart[idx_restart])
    vel_continuous = torch.tensor(velocities_continuous[idx_continuous])

    pos_diff = torch.max(torch.abs(pos_restart - pos_continuous)).item()
    vel_diff = torch.max(torch.abs(vel_restart - vel_continuous)).item()
    max_pos_diff = max(max_pos_diff, pos_diff)
    max_vel_diff = max(max_vel_diff, vel_diff)

    if not torch.allclose(pos_restart, pos_continuous, atol=1e-10, rtol=1e-10):
        print(f"  Step {step}: Position mismatch! Max diff: {pos_diff:.2e}")
        all_match = False
    if not torch.allclose(vel_restart, vel_continuous, atol=1e-10, rtol=1e-10):
        print(f"  Step {step}: Velocity mismatch! Max diff: {vel_diff:.2e}")
        all_match = False

assert all_match, (
    f"Restarted and continuous trajectories differ! "
    f"Max position difference: {max_pos_diff:.2e}, max velocity difference: {max_vel_diff:.2e}"
)
print("\n✓ Restarted and continuous trajectories match exactly.")

# %% [markdown]
"""
## Key Takeaways

1. **Save with `asdict()` + `torch.save()`**: This captures everything — positions,
   momenta, forces, energy, cell, and the `torch.Generator` RNG state — in a single
   checkpoint file.

2. **Restore with `MDState(**torch.load(...))`**: The `torch.Generator` is unpickled
   automatically, so the RNG state is restored without any extra steps.

3. **Use append mode** (`trajectory_kwargs={"mode": "a"}`) in `TrajectoryReporter`
   to continue an existing trajectory file.

4. **Pickle caveats**: The `torch.Generator` object in the checkpoint requires pickle
   (`weights_only=False`) and may not load across PyTorch versions. For portable
   checkpoints, save tensors normally and use `state.rng.get_state()` to extract the
   RNG state as a plain `uint8` tensor that works with `weights_only=True`.

5. **Verify**: Always compare restarted trajectories to continuous runs.
"""

# %%
# Cleanup
import shutil

shutil.rmtree(restart_dir)
print(f"Cleaned up {restart_dir}/")
