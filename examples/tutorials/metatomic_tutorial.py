# %%
# /// script
# dependencies = [
#    "torch_sim_atomistic[metatomic]"
# ]
# ///


# %% [markdown]
"""
# Using the PET-MAD model with TorchSim

This tutorial explains how to use the PET-MAD model (https://arxiv.org/abs/2503.14118)
via TorchSim's metatomic interface.

## Loading the model

Loading the model is simple: you simply need to specify the model name (in this case
"pet-mad"), as shown below. All other arguments are optional: for example, you could
specify the device. (If the device is not specified, like in this case, the optimal
device is chosen automatically.)
"""

# %%
from torch_sim.models.metatomic import MetatomicModel

model = MetatomicModel("pet-mad")

# %% [markdown]
"""
## Using the model to run a molecular dynamics simulations

Once the model is loaded, you can use it just like any other TorchSim model to run
simulations. Here, we show how to run a simple MD simulation consisting of an initial
NVT equilibration run followed by an NVE run.
"""
# %%
from ase.build import bulk
import torch_sim as ts

atoms = bulk("Si", "diamond", a=5.43, cubic=True)

equilibrated_state = ts.integrate(
    system=atoms,
    model=model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=100,
    temperature=300,  # K
    timestep=0.001,  # ps
)

final_state = ts.integrate(
    system=equilibrated_state,
    model=model,
    integrator=ts.Integrator.nve,
    n_steps=100,
    temperature=300,  # K
    timestep=0.001,  # ps
)

# %% [markdown]
"""
## Further steps

Of course, in reality, you would want to run the simulation for much longer, probably
save trajectories, and much more. However, this is all you need to get started with
metatomic and PET-MAD. For more details on how to use TorchSim, you can refer to the
other tutorials in this section.
"""
