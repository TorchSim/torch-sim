# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[mace, io]"
# ]
# ///


# %% [markdown]
"""
# Reproducible Restarts from Stopped Simulations

This tutorial demonstrates how to save and restore simulation state to enable
reproducible restarts from stopped simulations. This is essential for long-running
simulations that may need to be paused and resumed, or for checkpointing workflows.

## Introduction

When running molecular dynamics simulations, you may need to:
- Pause a simulation and resume it later
- Create checkpoints for long-running simulations
- Ensure that a restarted simulation produces identical results to a continuous run

To achieve reproducible restarts, you must save not only the atomic positions, velocities,
and other state variables, but also the random number generator (RNG) state. This is
especially important for stochastic integrators like Langevin dynamics, which use random
numbers for both initial momenta sampling and per-step stochastic noise.

## Key Concepts

1. **State Saving**: Save the complete simulation state including positions, velocities,
   momenta, cell parameters, and RNG state
2. **RNG State**: The `torch.Generator` state must be saved separately using
   `get_state()` and restored using `set_state()`
3. **Trajectory Comparison**: Verify that restarted simulations produce identical
   trajectories to continuous runs
"""

# %% [markdown]
"""
## Setup: Initial System and Model

Let's start by creating a simple system and model for our demonstration:
"""

# %%
import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.models.lennard_jones import LennardJonesModel

# Set up deterministic mode for reproducibility
# Note: This seeds the global RNG, but we'll also seed SimState.rng explicitly
seed = 42
torch.manual_seed(seed)

# Create a Lennard-Jones model
lj_model = LennardJonesModel(
    sigma=2.0,
    epsilon=0.1,
    device=torch.device("cpu"),
    dtype=torch.float64,
)

# Create a silicon crystal structure
si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)

# Initialize state and seed the RNG
initial_state = ts.initialize_state(
    si_atoms, device=torch.device("cpu"), dtype=torch.float64
)
initial_state.rng = seed  # Critical: seed the SimState RNG for reproducibility

print(f"Initial state has {initial_state.n_atoms} atoms")
print(f"RNG device: {initial_state.rng.device}")


# %% [markdown]
"""
## Part 1: Run 50 Steps, Save State, and Resume

First, we'll run 50 steps of MD, save the complete state (including RNG state),
then resume for another 50 steps:
"""

# %%
# Run first 50 steps with trajectory reporting
trajectory_file_restart = "restart_trajectory.h5"
reporter_restart = ts.TrajectoryReporter(
    filenames=trajectory_file_restart,
    state_frequency=10,  # Save state every 10 steps
    state_kwargs={"save_velocities": True},  # Save velocities (momenta) for comparison
)

# Run 50 steps
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

# Check RNG state after 50 steps (before saving)
test_random_after_50_restart = torch.randn(1, generator=state_after_50.rng).item()
print(f"After 50 steps (restart) - Test random: {test_random_after_50_restart:.6f}")

# Save the complete state including RNG state
state_save_file = "saved_state.pt"
rng_state_save_file = "saved_rng_state.pt"

# Save the RNG state separately (as recommended in reproducibility docs)
rng_state = state_after_50.rng.get_state()
torch.save(rng_state, rng_state_save_file)

# Save the state object (this doesn't include the RNG generator itself)
# We'll need to restore the RNG state separately
torch.save(
    {
        "positions": state_after_50.positions,
        "momenta": state_after_50.momenta,
        "cell": state_after_50.cell,
        "atomic_numbers": state_after_50.atomic_numbers,
        "masses": state_after_50.masses,
        "pbc": state_after_50.pbc,
        "system_idx": state_after_50.system_idx,
        "energy": state_after_50.energy,
        "forces": state_after_50.forces,
    },
    state_save_file,
)

print(f"Saved state after 50 steps")
print(f"State file: {state_save_file}")
print(f"RNG state file: {rng_state_save_file}")


# %% [markdown]
"""
Now let's restore the state and continue for another 50 steps:
"""

# %%
# Load the saved state
# Note: PyTorch 2.6+ defaults to weights_only=True. Since we're loading our own
# checkpoints, we use weights_only=False to allow loading Generator objects.
saved_data = torch.load(state_save_file, weights_only=False)
rng_state_loaded = torch.load(rng_state_save_file, weights_only=False)

# Reconstruct the state from saved data
# We need to create an MDState since we have momenta, forces, and energy
from torch_sim.integrators import MDState

restored_state = MDState(
    positions=saved_data["positions"],
    momenta=saved_data["momenta"],
    cell=saved_data["cell"],
    atomic_numbers=saved_data["atomic_numbers"],
    masses=saved_data["masses"],
    pbc=saved_data["pbc"],
    system_idx=saved_data["system_idx"],
    energy=saved_data["energy"],
    forces=saved_data["forces"],
)

# Restore the RNG state - this is critical for reproducibility!
gen = torch.Generator(device=restored_state.device)
gen.set_state(rng_state_loaded)
restored_state.rng = gen

# Verify RNG state was restored correctly
# Draw a random number to verify the RNG state matches
test_random_restored = torch.randn(1, generator=restored_state.rng).item()
print(f"Restored state with {restored_state.n_atoms} atoms")
print(f"Restored RNG device: {restored_state.rng.device}")
print(f"Test random from restored RNG: {test_random_restored:.6f}")

# Verify the RNG state matches what we saved
# (We need to reload the saved state to compare)
saved_rng_check = torch.Generator(device=restored_state.device)
saved_rng_check.set_state(rng_state_loaded)
test_random_saved = torch.randn(1, generator=saved_rng_check).item()
print(f"Test random from saved RNG state: {test_random_saved:.6f}")
assert abs(test_random_restored - test_random_saved) < 1e-10, "RNG state mismatch!"

# Continue simulation for another 50 steps
# Use append mode to continue the trajectory
reporter_restart_continued = ts.TrajectoryReporter(
    filenames=trajectory_file_restart,
    state_frequency=10,
    state_kwargs={"save_velocities": True},  # Save velocities (momenta) for comparison
    trajectory_kwargs={"mode": "a"},  # Append mode to continue existing trajectory
)

# IMPORTANT: When integrate() is called, it internally calls initialize_state() which
# clones the state. The clone() method should preserve the RNG state, but we verify
# that the RNG state is correctly set before calling integrate.
rng_state_before_integrate = restored_state.rng.get_state()

state_after_100_restart = ts.integrate(
    system=restored_state,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=50,  # Additional 50 steps
    temperature=300,
    timestep=0.001,
    trajectory_reporter=reporter_restart_continued,
)

# Check RNG state after 100 steps (restart)
test_random_after_100_restart = torch.randn(
    1, generator=state_after_100_restart.rng
).item()
print(f"After 100 steps (restart) - Test random: {test_random_after_100_restart:.6f}")

reporter_restart_continued.close()

print(f"Completed restart simulation: 50 + 50 = 100 steps total")


# %% [markdown]
"""
## Part 2: Run 100 Steps Continuously for Comparison

Now let's run a continuous simulation that matches the restart scenario: run 50 steps,
then continue for another 50 steps (without saving/restoring in between). This ensures
we're comparing apples to apples:
"""

# %%
# Run continuous simulation: 50 steps, then 50 more steps
trajectory_file_continuous = "continuous_trajectory.h5"
reporter_continuous = ts.TrajectoryReporter(
    filenames=trajectory_file_continuous,
    state_frequency=10,
    state_kwargs={"save_velocities": True},  # Save velocities (momenta) for comparison
)

# Create a fresh initial state with the same seed
initial_state_continuous = ts.initialize_state(
    si_atoms, device=torch.device("cpu"), dtype=torch.float64
)
initial_state_continuous.rng = seed  # Same seed as before

# Run first 50 steps (matching the restart scenario)
print("Running first 50 steps of continuous simulation...")
state_after_50_continuous = ts.integrate(
    system=initial_state_continuous,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=50,
    temperature=300,
    timestep=0.001,
    trajectory_reporter=reporter_continuous,
)

# Check RNG state after 50 steps
rng_state_after_50_continuous = state_after_50_continuous.rng.get_state()
test_random_after_50_continuous = torch.randn(
    1, generator=state_after_50_continuous.rng
).item()
print(f"After 50 steps - Test random: {test_random_after_50_continuous:.6f}")

# Continue for another 50 steps (without saving/restoring)
# Use append mode to continue the same trajectory
reporter_continuous_continued = ts.TrajectoryReporter(
    filenames=trajectory_file_continuous,
    state_frequency=10,
    state_kwargs={"save_velocities": True},
    trajectory_kwargs={"mode": "a"},  # Append mode
)

print("Running second 50 steps of continuous simulation...")
state_after_100_continuous = ts.integrate(
    system=state_after_50_continuous,
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=50,  # Additional 50 steps
    temperature=300,
    timestep=0.001,
    trajectory_reporter=reporter_continuous_continued,
)

reporter_continuous_continued.close()

# Check RNG state after 100 steps
test_random_after_100_continuous = torch.randn(
    1, generator=state_after_100_continuous.rng
).item()
print(f"After 100 steps - Test random: {test_random_after_100_continuous:.6f}")
print(f"Completed continuous simulation: 50 + 50 = 100 steps total")


# %% [markdown]
"""
## Part 3: Compare Trajectories

Let's compare the trajectories from the restarted simulation and the continuous
simulation to verify they are identical:
"""

# %%
# Load both trajectories
with ts.TorchSimTrajectory(trajectory_file_restart, mode="r") as traj_restart:
    positions_restart = traj_restart.get_array("positions")
    steps_restart = traj_restart.get_steps("positions")
    velocities_restart = traj_restart.get_array("velocities")

with ts.TorchSimTrajectory(trajectory_file_continuous, mode="r") as traj_continuous:
    positions_continuous = traj_continuous.get_array("positions")
    steps_continuous = traj_continuous.get_steps("positions")
    velocities_continuous = traj_continuous.get_array("velocities")

print(f"Restart trajectory: {len(steps_restart)} frames")
print(f"Continuous trajectory: {len(steps_continuous)} frames")

# Compare positions at matching steps
# Both should have states at steps 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
matching_steps = sorted(set(steps_restart) & set(steps_continuous))
print(f"\nMatching steps: {matching_steps}")

# Compare positions and momenta at each matching step
max_pos_diff = 0.0
max_mom_diff = 0.0
all_match = True

for step in matching_steps:
    idx_restart = steps_restart.tolist().index(step)
    idx_continuous = steps_continuous.tolist().index(step)

    # Convert numpy arrays to torch tensors for comparison
    pos_restart = torch.tensor(positions_restart[idx_restart])
    pos_continuous = torch.tensor(positions_continuous[idx_continuous])
    vel_restart = torch.tensor(velocities_restart[idx_restart])
    vel_continuous = torch.tensor(velocities_continuous[idx_continuous])

    pos_diff = torch.max(torch.abs(pos_restart - pos_continuous)).item()
    vel_diff = torch.max(torch.abs(vel_restart - vel_continuous)).item()

    max_pos_diff = max(max_pos_diff, pos_diff)
    max_mom_diff = max(max_mom_diff, vel_diff)

    # Check if they match exactly (within floating point precision)
    if not torch.allclose(pos_restart, pos_continuous, atol=1e-10, rtol=1e-10):
        print(f"  Step {step}: Position mismatch! Max diff: {pos_diff:.2e}")
        all_match = False
    if not torch.allclose(vel_restart, vel_continuous, atol=1e-10, rtol=1e-10):
        print(f"  Step {step}: Velocity mismatch! Max diff: {vel_diff:.2e}")
        all_match = False

print(f"\nMaximum position difference: {max_pos_diff:.2e}")
print(f"Maximum velocity difference: {max_mom_diff:.2e}")

if all_match:
    print("\n✓ SUCCESS: Restarted and continuous trajectories match exactly!")
else:
    print(
        "\n✗ WARNING: Trajectories differ - this may indicate an issue with state saving/restoration"
    )


# %% [markdown]
"""
## Part 4: Simplified State Saving with torch.save

For convenience, you can save the entire state object directly using `torch.save()`.
Since `torch.save()` uses pickle, the `torch.Generator` will be saved along with
everything else automatically.

**Note for PyTorch 2.6+**: PyTorch 2.6 changed the default `weights_only` parameter
in `torch.load()` from `False` to `True` for security. When loading checkpoints that
contain `torch.Generator` objects, you need to set `weights_only=False`. This is safe
when loading your own checkpoints, but be cautious when loading files from untrusted
sources as it can result in arbitrary code execution.
"""

# %%
# Simplified approach: save everything together
# Create a fresh state for demonstration
demo_state = ts.integrate(
    system=initial_state.clone(),
    model=lj_model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=25,
    temperature=300,
    timestep=0.001,
)

# Save the entire state dict - Generator is included automatically
from dataclasses import asdict

state_dict = asdict(demo_state)
torch.save(state_dict, "demo_state.pt")

# Restore
# Note: PyTorch 2.6+ defaults to weights_only=True for security, which doesn't allow
# loading Generator objects. Since we're loading our own checkpoint, we set
# weights_only=False. Alternatively, you can use torch.serialization.add_safe_globals()
loaded_dict = torch.load("demo_state.pt", weights_only=False)

# Reconstruct MDState - the Generator is restored automatically
restored_demo = MDState(**loaded_dict)

# Verify restoration
print(f"Original state energy: {demo_state.energy.item():.6f} eV")
print(f"Restored state energy: {restored_demo.energy.item():.6f} eV")
print(f"Positions match: {torch.allclose(demo_state.positions, restored_demo.positions)}")
print(f"Momenta match: {torch.allclose(demo_state.momenta, restored_demo.momenta)}")
print(f"RNG restored: {restored_demo.rng is not None}")


# %% [markdown]
"""
### When to Save RNG State Separately

The approach above works great when using `torch.save()` (which uses pickle). However,
you may need to save RNG state separately if:

1. **Using non-pickle formats**: If you're saving to HDF5, JSON, or other formats that
   don't support pickling, you'll need to extract the RNG state using `get_state()` and
   save it separately.

2. **Device portability**: If you need to restore to a different device, saving the
   state tensor separately gives you more control.

3. **Explicit documentation**: Some workflows prefer explicit RNG state handling for
   clarity and debugging.

For most use cases with `torch.save()`, the simple approach above is sufficient.
"""


# %% [markdown]
"""
## Key Takeaways

1. **RNG State is Critical**: For stochastic integrators (Langevin, NPT with barostat),
   you must save and restore the RNG state. With `torch.save()`, the Generator is
   pickled automatically, but you can also save the state separately using `get_state()`
   and `set_state()` if needed (e.g., for non-pickle formats or device portability).

2. **Complete State Saving**: Save all relevant state variables including positions,
   momenta, cell parameters, energy, and forces.

3. **Trajectory Continuity**: When resuming, use append mode (`trajectory_kwargs={"mode": "a"}`)
   in `TrajectoryReporter` to continue the existing trajectory file.

4. **Verification**: Always compare restarted trajectories to continuous runs to ensure
   reproducibility.

5. **Deterministic Integrators**: For deterministic integrators (NVE, NVT Nosé-Hoover),
   you don't need to save RNG state, but it's still good practice for consistency.

## Best Practices

- Save checkpoints regularly during long simulations
- Include metadata (step number, simulation parameters) with saved states
- Verify reproducibility by comparing trajectories
- Use the same seed and device when restoring states
- Consider saving to a format that's easy to inspect (HDF5 via TorchSimTrajectory)

For more information on reproducibility in TorchSim, see the
[reproducibility documentation](../../../docs/user/reproducibility.md).

## Troubleshooting: Trajectories Don't Match

If your restarted and continuous trajectories don't match exactly, check:

1. **RNG State Preservation**: Verify that the RNG state is correctly restored before
   calling `integrate()`. The RNG state must be set on the state object before it's
   passed to `integrate()`, as `integrate()` will clone the state internally.

2. **Use the Simplified Approach**: If you're having issues with manual RNG state
   management, try using the simplified approach from Part 4 (saving everything with
   `torch.save()`), which automatically preserves the RNG state.

3. **Check Initial Step**: When using append mode, ensure the trajectory reporter
   correctly detects the last step. The `integrate()` function should automatically
   detect this and start from the correct step.

4. **Verify State Restoration**: Print and compare key state variables (positions,
   momenta, RNG state) before continuing the simulation to ensure they match what
   was saved.
"""
