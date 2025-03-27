# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///

# ruff: noqa: E402


# %% [markdown]
"""# Introduction to torch-sim low-level API."""

# %% [markdown]
"""
This is an introductory tutorial on how to use the low-level API of torch-sim package.
This tutorial will get users familiar with the following:
- Creating a `torch-sim` state
- Calling `torch-sim` models
- Running batched relaxation
- Running batched molecular dynamics
"""

# %% [markdown]
"""
## Create a torch-sim state

`torch-sim`'s state aka `SimState` is a class that contains the information of the system
like positions, cell, etc. of the system(s).
All the models in the `torch-sim` package take in a `SimState` as an input and return
the properties of the system(s).

First we will create two simple structures of 2x2x2 unit cells of
Body Centered Cubic (BCC) Iron and Diamond Cubic Silicon.

Since we want to do batch simulations, we will create a list of Atoms from which we will
create a `SimState`.
"""
import os

from ase.build import bulk


si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
fe_bcc = bulk("Fe", "bcc", a=2.8665, cubic=True).repeat((3, 3, 3))

atoms_list = [si_dc, fe_bcc]


# %% [markdown]
"""
You can use any package to create the initial structure or pass the
attributes directly to the `SimState`. The `io.py` module provides
a utility to convert multiple ASE Atoms, Pymatgen Structure, and
PhonopyAtoms to `SimState` and vice versa.

Note that we need to pass the device and dtype to the `atoms_to_state`
function in order to make sure the correct device and dtype is used.

Let's create a `SimState` from the list of Atoms.
"""
import torch

from torch_sim.io import atoms_to_state


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

state = atoms_to_state(atoms_list, device=device, dtype=dtype)


# %% [markdown]
"""
## Calling torch-sim models

In order to compute the properties of the systems above,
we need to first initialize the models.

In this example, we use the MACE-MPA-0 model for our Si and Fe systems.
First, we need to download the model file and get the raw model from mace-mp.
"""
from mace.calculators.foundations_models import mace_mp


mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)


# %% [markdown]
"""
Now we can initialize the MACE model.
"""
from torch_sim.models import MaceModel


model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)


# %% [markdown]
"""
By default, the model can compute the properties across a batch of systems.
You can also specify model-specific args to the model.

We can now pass the state to the model and compute the energy of the systems.
"""

results = model(state)
print(results["energy"])


# %% [markdown]
"""
This gives us the energies of the two systems.
"""

# %% [markdown]
"""
## Running batched relaxation

We will now run a relaxation of the system (positions + cell)
using the unit cell filter with the FIRE optimizer.

First, we need to initialize the optimizer.
"""
from torch_sim.optimizers import unit_cell_fire


fire_init, fire_update = unit_cell_fire(model=model)


# %% [markdown]
"""
You can set the optimizer-specific arguments in the optimizer function.

The optimizer returns two functions: `fire_init` and `fire_update`.
The `fire_init` function returns the initialized optimizer-specific state,
while the `fire_update` function updates the simulation state.

The optimizer performs optimization across the batch of systems.
We can access the optimizer attributes from the state object like `state.energy` etc.
This gives us the energies of the systems in the batch.
"""
max_steps = 5 if os.environ.get("CI") else 50
state = fire_init(state=state)

for step in range(max_steps):
    state = fire_update(state=state)
    if step % 5 == 0:
        print(f"{step=}: Total energy: {state.energy} eV")


# %% [markdown]
"""
## Running batched molecular dynamics

Similarly, we can do molecular dynamics of the systems.
We need to make sure we are using correct units for the integrator.
`torch-sim` provides a `units.py` module to help with the units system and conversions.
The units system is defined similar to the LAMMPS units system.
Here we use the Metal units as the models return the outputs in similar units.
"""
from torch_sim.integrators import nvt_langevin
from torch_sim.units import MetalUnits


# %% [markdown]
"""
We will run a MD simulation for 500 steps with a timestep of 0.002 ps,
an initial temperature of 300 K, and a Langevin friction coefficient of 10 ps^-1.
"""
max_md_steps = 5 if os.environ.get("CI") else 500
dt = 0.002 * MetalUnits.time  # Timestep (ps)
kT = 300 * MetalUnits.temperature  # Initial temperature (K)
gamma = 10 / MetalUnits.time  # Langevin friction coefficient (ps^-1)


# %% [markdown]
"""
We can also compute quantities like temperature that are not present in the state.
The `quantities.py` module provides a utility to compute quantities
like temperature, kinetic energy, etc.

Now we perform a Constant Volume, Constant Temperature (NVT) Langevin dynamics
for the systems.
Similar to the optimizer, we have two functions:
`nvt_langevin_init` and `nvt_langevin_update`.
"""
from torch_sim.quantities import temperature


nvt_langevin_init, nvt_langevin_update = nvt_langevin(
    model=model, dt=dt, kT=kT, gamma=gamma
)


# %% [markdown]
"""
We can easily pass the final relaxed state to the integrator.
"""
state = nvt_langevin_init(state=state)

for step in range(max_md_steps):
    state = nvt_langevin_update(state=state)
    if step % 20 == 0:
        temp = (
            temperature(masses=state.masses, momenta=state.momenta, batch=state.batch)
            / MetalUnits.temperature
        )
        print(f"{step=}: Temperature: {temp}")


# %% [markdown]
"""
We see that the system temperature is constant around the target temperature of 300 K.
"""

# %% [markdown]
"""
## Concluding remarks

This tutorial shows how to use the low-level API of `torch-sim` to create a state,
call models, and run optimizers and integrators. However, for most cases,
one can use the high-level API to run the simulations as covered in the other
introductory tutorial.

We hope this tutorial gives users a good starting point to use the low-level API in
the `torch-sim` package. More detailed tutorials and example scripts can be
found in the examples folder.

If you have any questions, please refer to the documentation or raise an issue on
the GitHub repository.
"""
