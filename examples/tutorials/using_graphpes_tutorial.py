# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# dependencies = [
#     "graph-pes>=0.0.30",
#     "torch==2.5",
# ]
# ///
# </details>


# %% [markdown]
"""
# Integrating TorchSim with `graph-pes`

This brief tutorial demonstrates how to use models trained with the
[graph-pes](https://github.com/mir-group/graph-pes) package to drive
MD simulations and geometry optimizations in TorchSim.

## Step 1: loading a model

As an output of the `graph-pes-train` command, you receive a path
to a `.pt` file containing your trained model. To use this model
with TorchSim, pass the path to this `.pt` file, or the model itself,
to the `GraphPESWrapper` constructor.

Below, we create a dummy TensorNet model with random weights as a demonstration:
"""

# %%
from graph_pes.models import TensorNet, load_model

# if you had a model saved to disk, you could load it like this:
# model = load_model("path/to/model.pt")

# here, we just create a TensorNet model with random weights
model = TensorNet(cutoff=5.0)

print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
"""
## Step 2: wrapping the model for use with TorchSim

We provide the `GraphPESWrapper` class to wrap a `graph-pes` model for use with TorchSim.
If you intend to drive simulations that require stresses, you will need to specify the
`compute_stress` argument to `True`.
"""

# %%
from torch_sim.models import GraphPESWrapper

# wrap the model for use with TorchSim
ts_model = GraphPESWrapper(model, compute_stress=False)

# or, alternatively, pass a model path directly:
# ts_model = GraphPESWrapper("path/to/model.pt", compute_stress=False)

# %% [markdown]
"""
## Step 3: driving MD with the model

Now that we have a model, we can drive MD simulations with it. For this, we will use the
`integrate` function.
"""
# %%
from ase.build import molecule
import torch_sim as ts
from load_atoms import view

# NVT at 300K
atoms = molecule("H2O")

final_state = ts.integrate(
    system=atoms,
    model=ts_model,
    integrator=ts.nvt_langevin,
    n_steps=50,
    temperature=300,
    timestep=0.001,
)

final_atoms = final_state.to_atoms()[0]
view(final_atoms, show_bonds=True)

# %% [markdown]
"""
Of course, this is a very simple example. However, you are now equipped to
use any `graph-pes` model that you have trained to drive any of the functionality
exposed by TorchSim!
"""
