# %%
# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///


# %% [markdown]
"""
# Understanding Autobatching

This tutorial provides a detailed guide to using TorchSim's autobatching features,
which help you efficiently process large collections of simulation states on GPUs
without running out of memory.

This is an intermediate tutorial. Autobatching is automatically handled by the
`integrate`, `optimize`, and `static` functions, you don't need to worry about it
unless:
- you want to manually optimize the batch size for your model
- you want to develop advanced or custom workflows

## Introduction

Simulating many molecular systems on GPUs can be challenging when the total number of
atoms exceeds available GPU memory. The `ts.autobatching` module solves this by:

1. Automatically determining optimal batch sizes based on GPU memory constraints
2. Providing two complementary strategies: binning and in-flight
3. Efficiently managing memory resources during large-scale simulations

Let's explore how to use these powerful features!


This next cell can be ignored, it only exists to allow the tutorial to run
in CI on a CPU. Using the AutoBatcher is generally not supported on CPUs.
"""

# %%
import torch_sim as ts


ts.autobatching.determine_max_batch_size = lambda *args, **kwargs: 3  # type: ignore[invalid-assignment]


# %% [markdown]
"""
## Understanding Memory Requirements

Before diving into autobatching, let's understand how memory usage is estimated:
"""

# %%
import torch
from torch_sim.autobatching import calculate_memory_scaler
from ase.build import bulk


# stack 5 fcc Cu atoms, we choose a small number for fast testing but this
# can be as large as you want
cu_atoms = bulk("Cu", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
many_cu_atoms = [cu_atoms] * 5

# Can be replaced with any SimState object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = ts.initialize_state(many_cu_atoms, device=device, dtype=torch.float64)

# Calculate memory scaling factor based on atom count
atom_metric = calculate_memory_scaler(state, memory_scales_with="n_atoms")

# Calculate memory scaling based on atom count and density
density_metric = calculate_memory_scaler(state, memory_scales_with="n_atoms_x_density")

print(f"Atom-based memory metric: {atom_metric}")
print(f"Density-based memory metric: {density_metric:.2f}")


# %% [markdown]
"""
Different simulation models have different memory scaling characteristics: - For models
with a fixed cutoff radius (like MACE), density matters, so use
`"n_atoms_x_density"` - For models with fixed neighbor counts, or models that
regularly hit their max neighbor count (like most FairChem models), use `"n_atoms"`

The autobatchers will use the memory scaler to determine the maximum batch size for
your model. Generally this max memory metric is roughly fixed for a given model and
hardware, assuming you choose the right scaling metric.
"""

# %%
from torch_sim.autobatching import estimate_max_memory_scaler
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

# Initialize your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)

state_list = state.split()
memory_metric_values = [
    calculate_memory_scaler(s, memory_scales_with="n_atoms") for s in state_list
]

max_memory_metric = estimate_max_memory_scaler(
    state_list, mace_model, metric_values=memory_metric_values
)
print(f"Max memory metric: {max_memory_metric}")


# %% [markdown]
"""
This is a verbose way to determine the max memory metric, we'll see a simpler way
shortly.

## BinningAutoBatcher: Fixed Batching Strategy

Now on to the exciting part, autobatching! The `BinningAutoBatcher` groups states into
batches with a binpacking algorithm, ensuring that we minimize the total number of
batches while maximizing the GPU utilization of each batch. This approach is ideal for
scenarios where all states need to be processed the same number of times, such as
batched integration.

### Basic Usage
"""

# %% Initialize the batcher, the max memory scaler will be computed automatically
batcher = ts.BinningAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
)

# Load a single batched state or a list of states, it returns the max memory scaler
max_memory_scaler = batcher.load_states(state)
print(f"Max memory scaler: {max_memory_scaler}")


# we define a simple function to process the batch, this could be
# any integrator or optimizer
def process_batch(batch):
    # Process the batch (e.g., run dynamics or optimization)
    batch.positions += torch.randn_like(batch.positions) * 0.01
    return batch


# Process each batch
processed_batches = []
for batch, _indices in batcher:
    # Process the batch (e.g., run dynamics or optimization)
    batch = process_batch(batch)
    processed_batches.append(batch)

# Restore original order of states
final_states = batcher.restore_original_order(processed_batches)


# %% [markdown]
"""
If you don't specify `max_memory_scaler`, the batcher will automatically estimate the
maximum safe batch size through test runs on your GPU. However, the max memory scaler
is typically fixed for a given model and simulation setup. To avoid calculating it
every time, which is a bit slow, you can calculate it once and then include it in the
`BinningAutoBatcher` constructor.
"""

# %%
batcher = ts.BinningAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=max_memory_scaler,
)


# %% [markdown]
"""
### Example: NVT Langevin Dynamics

Here's a real example using FIRE optimization from the test suite:
"""

# %% Initialize nvt langevin integrator
nvt_state = ts.nvt_langevin_init(state, mace_model, kT=0.01)

# Initialize the batcher
batcher = ts.BinningAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
)
max_memory_scaler = batcher.load_states(nvt_state)
print(f"Max memory scaler: {max_memory_scaler}")

print(f"There are {len(batcher.index_bins)} bins")
print(f"The indices of the states in each bin are: {batcher.index_bins}")

# Run optimization on each batch
finished_states = []
for batch, _indices in batcher:
    # Run 5 steps of NVT dynamics
    for _ in range(5):
        batch = ts.nvt_langevin_step(mace_model, batch, dt=0.001, kT=0.01)

    finished_states.append(batch)

# Restore original order
restored_states = batcher.restore_original_order(finished_states)


# %% [markdown]
"""
## InFlightAutoBatcher: Dynamic Batching Strategy

The `InFlightAutoBatcher` optimizes GPU utilization by dynamically removing
converged states and adding new ones. This is ideal for processes like geometry
optimization where different states may converge at different rates.

The `InFlightAutoBatcher` is more complex than the `BinningAutoBatcher` because
it requires the batch to be dynamically updated. The swapping logic is handled internally,
but the user must regularly provide a convergence tensor indicating which batches in
the state have converged.

### Usage
"""

# %%
fire_state = ts.fire_init(
    state=state, model=mace_model, cell_filter=ts.CellFilter.frechet
)

# Initialize the batcher
batcher = ts.InFlightAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=1000,
    max_iterations=100,  # Optional: maximum convergence attempts per state
)
# Load states
batcher.load_states(fire_state)

# add some random displacements to each state
fire_state.positions = (
    fire_state.positions + torch.randn_like(fire_state.positions) * 0.05
)
total_states = fire_state.n_systems

# Define a convergence function that checks the force on each atom is less than 5e-1
convergence_fn = ts.generate_force_convergence_fn(5e-1)

# Process states until all are complete
all_converged_states, convergence_tensor = [], None
while (result := batcher.next_batch(fire_state, convergence_tensor))[0] is not None:
    # collect the converged states
    fire_state, converged_states = result
    all_converged_states.extend(converged_states)

    # optimize the batch, we stagger the steps to avoid state processing overhead
    for _ in range(10):
        fire_state = ts.fire_step(state=fire_state, model=mace_model)

    # Check which states have converged
    convergence_tensor = convergence_fn(fire_state, None)
    print(f"Convergence tensor: {batcher.current_idx}")

else:
    all_converged_states.extend(result[1])

# Restore original order
final_states = batcher.restore_original_order(all_converged_states)

# Verify all states were processed
assert len(final_states) == total_states

# Note that the fire_state has been modified in place
assert fire_state.n_systems == 0


# %%
fire_state.n_systems


# %% [markdown]
"""
## Tracking Original Indices

Both batchers can return the original indices of states, which is useful for
tracking the progress of individual states. This is especially critical when
using the `TrajectoryReporter`, because the files must be regularly updated.
"""

# %% Initialize batcher
batcher = ts.BinningAutoBatcher(
    model=mace_model, memory_scales_with="n_atoms", max_memory_scaler=80
)
batcher.load_states(state)

# Iterate over batches
for idx, (batch, indices) in enumerate(batcher):
    print(f"Processing states with original indices: {indices}")
    # Process batch...


# %% [markdown]
"""
## Conclusion

TorchSim's autobatching provides powerful tools for GPU-efficient simulation of
multiple systems:

1. Use `BinningAutoBatcher` for simpler workflows with fixed iteration counts
2. Use `InFlightAutoBatcher` for optimization problems with varying convergence
   rates
3. Let the library handle memory management automatically, or specify limits manually

By leveraging these tools, you can efficiently process thousands of states on a single
GPU without running out of memory!
"""
