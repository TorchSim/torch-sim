# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "pymatgen>=2025.2.18",
# ]
# ///
# </details>


# %% [markdown]
"""
# Implementing New Methods

This tutorial demonstrates how to combine different TorchSim components to implement new
simulation methods. We'll implement a hybrid Monte Carlo method that alternates between:
- Molecular dynamics (MD) for local exploration
- Swap Monte Carlo for composition changes

This is an advanced tutorial that will cover:
- Creating custom state objects
- Combining different TorchSim integrators
- Implementing hybrid simulation methods
"""

# %% [markdown]
"""
## Setting up the Environment

First, let's set up our simulation environment and load a model. We'll use MACE
for this example, but any TorchSim compatible model would work.
"""

# %%
import torch
import torch_sim as ts
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

# Initialize the mace model
device = "cuda" if torch.cuda.is_available() else "cpu"
mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)


# %% [markdown]
"""
## Creating the Initial Structure

For this example, we'll create a binary Cu-Zr alloy structure. We use pymatgen's
Structure class to define our system, but you could also use ASE or other formats.
"""

# %%
from pymatgen.core import Structure

# Create a binary Cu-Zr alloy structure
lattice = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
species = ["Cu", "Cu", "Cu", "Zr", "Cu", "Zr", "Zr", "Zr"]
coords = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.0, 0.5, 0.5],
    [0.25, 0.75, 0.75],
    [0.5, 0.0, 0.5],
    [0.75, 0.25, 0.75],
    [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.25],
]
structure = Structure(lattice, species, coords)

# Convert to TorchSim state
state = ts.initialize_state([structure], device=device, dtype=torch.float64)


# %% [markdown]
"""
## Implementing the Hybrid Method

Our hybrid method requires a custom state object that combines properties from both
MD and Monte Carlo states. In TorchSim, we can create this by inheriting from the
MDState class and adding our Monte Carlo-specific attributes.

The key components we'll combine are:
1. NVT Langevin dynamics for local structure exploration
2. Swap Monte Carlo for jumps in configurational space
"""

# %%
from dataclasses import dataclass


@dataclass
class HybridSwapMCState(ts.integrators.MDState):
    """State for hybrid MD-Monte Carlo simulations.

    This state class extends the standard MDState with:
    - last_permutation: Tracks whether the last MC move was accepted

    All other MD attributes (positions, momenta, forces, etc.) are inherited
    from MDState.
    """

    last_permutation: torch.Tensor


# %% [markdown]
"""
## Setting up the Simulation

Now we'll initialize both the MD and Monte Carlo components. We:
1. Initialize the NVT Langevin integrator
2. Initialize the swap Monte Carlo mover
3. Create our hybrid state that can handle both types of moves
"""

# %%
from torch_sim.units import MetalUnits

kT = 1000 * MetalUnits.temperature

# Initialize NVT Langevin dynamics state
nvt_init, nvt_step = ts.nvt_langevin(model=mace_model, dt=0.002, kT=kT, seed=42)
md_state = nvt_init(state)

# Initialize swap Monte Carlo state
swap_init, swap_step = ts.swap_monte_carlo(model=mace_model, kT=kT, seed=42)
swap_state = swap_init(md_state)

# Create hybrid state combining both
hybrid_state = HybridSwapMCState(
    **vars(md_state),
    last_permutation=torch.zeros(
        md_state.n_batches, device=md_state.device, dtype=torch.bool
    ),
)


# %% [markdown]
"""
## Running the Hybrid Simulation

We'll run our hybrid simulation by alternating between MD and MC moves:
- Every 10 steps, we attempt a swap Monte Carlo move
- All other steps, we perform standard NVT dynamics

This creates a simulation that can both:
- Explore local energy minima through MD
- Make larger compositional changes through swap moves
"""

# %% Run the hybrid simulation
n_steps = 100
for step in range(n_steps):
    if step % 10 == 0:
        # Attempt swap Monte Carlo move
        hybrid_state = swap_step(hybrid_state, kT=torch.tensor(kT))
    else:
        # Perform MD step
        hybrid_state = nvt_step(hybrid_state, dt=torch.tensor(0.002), kT=torch.tensor(kT))

    if step % 20 == 0:
        print(f"Step {step}: Energy = {hybrid_state.energy.item():.3f} eV")


# %% [markdown]
"""
## Concluding Remarks

This tutorial demonstrated how to combine different TorchSim components to create
new simulation methods. Key takeaways:

1. TorchSim's components (integrators, MC movers, etc.) are designed to be modular
2. Custom state objects can combine features from different simulation types
3. Complex simulation workflows can be built by mixing and matching components

"""
