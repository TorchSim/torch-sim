# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///
# # %% [markdown]
"""
# Autobatching
This tutorial will give a detailed look at
how to use the autobatching features of torch-sim.
"""
# %% [markdown]
"""
# Autobatching

This tutorial provides a detailed guide to using torch-sim's autobatching features,
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
"""

# %% [markdown]
"""
This next cell can be ignored, it only exists to allow the tutorial to run
in CI on a CPU. Using the AutoBatcher is generally not supported on CPUs.
"""
# %%
import torch_sim
def mock_calculate_memory_scaler(state, memory_scales_with):
    return 1000 if memory_scales_with == "n_atoms" else 10000

torch_sim.autobatching.calculate_memory_scaler = mock_calculate_memory_scaler

# %% [markdown]
"""
## Understanding Memory Requirements

Before diving into autobatching, let's understand how memory usage is estimated:
"""

# %%
import torch
from torch_sim.autobatching import calculate_memory_scaler
from torch_sim.state import initialize_state
from ase.build import bulk


# stack 5 fcc Cu atoms, we choose a small number for fast testing but this
# can be as large as you want
cu_atoms = bulk("Cu", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
many_cu_atoms = [cu_atoms] * 5

# Can be replaced with any SimState object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = initialize_state(many_cu_atoms, device=device, dtype=torch.float64)

# Calculate memory scaling factor based on atom count
atom_metric = calculate_memory_scaler(state, memory_scales_with="n_atoms")

# Calculate memory scaling based on atom count and density
density_metric = calculate_memory_scaler(state, memory_scales_with="n_atoms_x_density")

print(f"Atom-based memory metric: {atom_metric}")
print(f"Density-based memory metric: {density_metric}")

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

# %%
from torch_sim.autobatching import ChunkingAutoBatcher

# Initialize the batcher, the max memory scaler will be computed automatically
batcher = ChunkingAutoBatcher(
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
batcher = ChunkingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=max_memory_scaler,
)

"""
### Optimization Example

Here's a real example using FIRE optimization from the test suite:
"""

# %%
from torch_sim.integrators import nvt_langevin

# Initialize FIRE optimizer
nvt_init, nvt_update = nvt_langevin(mace_model, dt=0.001, kT=0.01)

# Prepare states for optimization
fire_state = nvt_init(state)

# Add some random displacements to each state
for state in fire_state:
    state.positions += torch.randn_like(state.positions) * 0.05

# Initialize the batcher
batcher = ChunkingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=30,
)
max_memory_scaler = batcher.load_states(fire_state)
print(f"Max memory scaler: {max_memory_scaler}")

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

### Basic Usage
"""

# %%
from torch_sim.autobatching import HotSwappingAutoBatcher
from torch_sim.runners import generate_force_convergence_fn
from torch_sim.optimizers import frechet_cell_fire

fire_init, fire_update = frechet_cell_fire(mace_model)
fire_state = fire_init(state)

# Initialize the batcher
batcher = HotSwappingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=40,
    max_iterations=100,  # Optional: maximum iterations per state
)

# Load states
batcher.load_states(fire_state)

# Define a convergence function that checks the force on each atom is less than 5e-1
convergence_fn = generate_force_convergence_fn(5e-1)

# Process states until all are complete
all_converged_states, convergence_tensor = [], None
while (result := batcher.next_batch(fire_state, convergence_tensor))[0] is not None:
    state, converged_states = result
    all_converged_states.extend(converged_states)

    # Process the batch
    for _ in range(10):  # Run for 10 steps
        batch = fire_update(batch)
    
    # Check which states have converged
    convergence_tensor = convergence_fn(batch)
    
    # Get next batch (converged states are removed, new ones added)
    batch, new_completed = batcher.next_batch(batch, convergence_tensor)

# Restore original order
final_states = batcher.restore_original_order(all_converged_states)

# Verify all states were processed
assert len(final_states) == fire_state.n_batches

# %% [markdown]
"""
## Tracking Original Indices

Both batchers can return the original indices of states, which is useful for 
tracking the progress of individual states. This is especially critical when
using the `TrajectoryReporter`, because the files must be regularly updated.
"""

# %%
# Initialize with return_indices=True
batcher = ChunkingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=260.0,
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

torch-sim's autobatching provides powerful tools for GPU-efficient simulation of
multiple systems:

1. Use `ChunkingAutoBatcher` for simpler workflows with fixed iteration counts
2. Use `HotSwappingAutoBatcher` for optimization problems with varying convergence
   rates
3. Let the library handle memory management automatically, or specify limits manually

By leveraging these tools, you can efficiently process thousands of states on a single
GPU without running out of memory!
"""