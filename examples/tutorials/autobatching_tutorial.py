# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///
# </details>


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
atoms exceeds available GPU memory. The `torch_sim.autobatching` module solves this by:

1. Automatically determining optimal batch sizes based on GPU memory constraints
2. Providing two complementary strategies: chunking and hot-swapping
3. Efficiently managing memory resources during large-scale simulations

Let's explore how to use these powerful features!


This next cell can be ignored, it only exists to allow the tutorial to run
in CI on a CPU. Using the AutoBatcher is generally not supported on CPUs.
"""

# %%
import torch_sim as ts


def mock_determine_max_batch_size(*args, **kwargs):
    return 3


ts.autobatching.determine_max_batch_size = mock_determine_max_batch_size


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
from torch_sim.models import MaceModel

# Initialize your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)

state_list = state.split()
memory_metric_values = [
    calculate_memory_scaler(s, memory_scales_with="n_atoms") for s in state_list
]

max_memory_metric = estimate_max_memory_scaler(
    mace_model, state_list, metric_values=memory_metric_values
)
print(f"Max memory metric: {max_memory_metric}")


# %% [markdown]
"""
This is a verbose way to determine the max memory metric, we'll see a simpler way
shortly.

## ChunkingAutoBatcher: Fixed Batching Strategy

Now on to the exciting part, autobatching! The `ChunkingAutoBatcher` groups states into
batches with a binpacking algorithm, ensuring that we minimize the total number of
batches while maximizing the GPU utilization of each batch. This approach is ideal for
scenarios where all states need to be processed the same number of times, such as
batched integration.

### Basic Usage
"""

# %% Initialize the batcher, the max memory scaler will be computed automatically
batcher = ts.ChunkingAutoBatcher(
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
for batch in batcher:
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
`ChunkingAutoBatcher` constructor.
"""

# %%
batcher = ts.ChunkingAutoBatcher(
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
nvt_init, nvt_update = ts.nvt_langevin(mace_model, dt=0.001, kT=0.01)

# Prepare states for optimization
nvt_state = nvt_init(state)

# Initialize the batcher
batcher = ts.ChunkingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
)
max_memory_scaler = batcher.load_states(nvt_state)
print(f"Max memory scaler: {max_memory_scaler}")

print("There are ", len(batcher.index_bins), " bins")
print("The indices of the states in each bin are: ", batcher.index_bins)

# Run optimization on each batch
finished_states = []
for batch in batcher:
    # Run 5 steps of FIRE optimization
    for _ in range(5):
        batch = nvt_update(batch)

    finished_states.append(batch)

# Restore original order
restored_states = batcher.restore_original_order(finished_states)


# %% [markdown]
"""
## HotSwappingAutoBatcher: Dynamic Batching Strategy

The `HotSwappingAutoBatcher` optimizes GPU utilization by dynamically removing
converged states and adding new ones. This is ideal for processes like geometry
optimization where different states may converge at different rates.

The `HotSwappingAutoBatcher` is more complex than the `ChunkingAutoBatcher` because
it requires the batch to be dynamically updated. The swapping logic is handled internally,
but the user must regularly provide a convergence tensor indicating which batches in
the state have converged.

### Usage
"""

# %%
fire_init, fire_update = ts.frechet_cell_fire(mace_model)
fire_state = fire_init(state)

# Initialize the batcher
batcher = ts.HotSwappingAutoBatcher(
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
total_states = fire_state.n_batches

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
        fire_state = fire_update(fire_state)

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
assert fire_state.n_batches == 0


# %%
fire_state.n_batches


# %% [markdown]
"""
## Tracking Original Indices

Both batchers can return the original indices of states, which is useful for
tracking the progress of individual states. This is especially critical when
using the `TrajectoryReporter`, because the files must be regularly updated.
"""

# %% Initialize with return_indices=True
batcher = ts.ChunkingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=80,
    return_indices=True,
)
batcher.load_states(state)

# Iterate with indices
for batch, indices in batcher:
    print(f"Processing states with original indices: {indices}")
    # Process batch...


# %% [markdown]
"""
## Conclusion

TorchSim's autobatching provides powerful tools for GPU-efficient simulation of
multiple systems:

1. Use `ChunkingAutoBatcher` for simpler workflows with fixed iteration counts
2. Use `HotSwappingAutoBatcher` for optimization problems with varying convergence
   rates
3. Let the library handle memory management automatically, or specify limits manually

By leveraging these tools, you can efficiently process thousands of states on a single
GPU without running out of memory!
"""
