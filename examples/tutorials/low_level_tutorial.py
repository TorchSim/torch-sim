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
# Low-level API: Understanding Optimizers, Integrators, and Models

The `torch-sim` package is designed to be both flexible and easy to use. It achieves this
by providing a high level API for common use cases. For most cases, this is the right choice 
because it bakes in autobatching, reporting, and evaluation. For some use cases, however, 
the high-level API is limiting. This tutorial introduces the design philosophy and usage of the
low-level API.

This is an intermediate tutorial that assumes a basic understanding of SimState and
optimizers.
"""

# %% [markdown]
"""
## Setting up the system

`torch-sim`'s state aka `SimState` is a class that contains the information of the
system like positions, cell, etc. of the system(s). All the models in the `torch-sim`
package take in a `SimState` as an input and return the properties of the system(s).

First we will create two simple structures of 2x2x2 unit cells of Body Centered Cubic
(BCC) Iron and Diamond Cubic Silicon and combine them into a batched state.
"""

# %%
from ase.build import bulk
import torch
from torch_sim.state import initialize_state

si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
fe_bcc = bulk("Fe", "bcc", a=2.8665, cubic=True).repeat((3, 3, 3))
atoms_list = [si_dc, fe_bcc]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

state = initialize_state(atoms_list, device=device, dtype=dtype)


# %% [markdown]
"""
## Calling Models Directly

In order to compute the properties of the systems above, we need to first initialize
the models.

In this example, we use the MACE-MPA-0 model for our Si and Fe systems. First, we need
to download the model file and get the raw model from mace-mp.

Then we can initialize the MaceModel class with the raw model.
"""

# %%
from mace.calculators.foundations_models import mace_mp
from torch_sim.models import MaceModel

# load mace_mp using the mace package
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# wrap the mace_mp model in the MaceModel class
model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
)

# %% [markdown]
"""
`torch-sim`'s MaceModel, and the other MLIP models, are wrappers around the raw models
that allow them to interface with the rest of the `torch-sim` package. They expose
several key properties that are expected by the rest of the package. This contract is
enforced by the `ModelInterface` class that all models must implement.
"""

# %%
print("Model device:", model.device)
print("Model dtype:", model.dtype)
print("Model compute_forces:", model.compute_forces)
print("Model compute_stress:", model.compute_stress)

# see the autobatching tutorial for more details
print("Model memory_scales_with:", model.memory_scales_with)

# %% [markdown]
"""
`SimState` objects can be passed directly to the model and it will compute
the properties of the systems in the batch. The properties will be returned
either batchwise, like the energy, or atomwise, like the forces.

Note that the energy here refers to the potential energy of the system.
"""

# %%
model_outputs = model(state)
print(f"Model outputs: {', '.join(list(model_outputs.keys()))}")

print(f"Energy is a batchwise property with shape: {model_outputs['energy'].shape}")
print(f"Forces are an atomwise property with shape: {model_outputs['forces'].shape}")
print(f"Stress is a batchwise property with shape: {model_outputs['stress'].shape}")
# %% [markdown]
"""
## Optimizers and Integrators

All optimizers and integrators share a similar interface. They accept a model and
return two functions: `init_fn` and `update_fn`. The `init_fn` function returns the
initialized optimizer-specific state, while the `update_fn` function updates the
simulation state.

### Unit Cell Fire

We will walk through the `unit_cell_fire` optimizer as an example.
"""

# %%
from torch_sim.optimizers import unit_cell_fire

fire_init_fn, fire_update_fn = unit_cell_fire(model=model)

# %% [markdown]
"""
We can then initialize the state and evolve the system with the update function.
Of course, we could also enforce some convergence criteria on the energy or forces
and stop the optimization early. Functionality that is automatically handled by the
high-level API.
"""

# %%
state = fire_init_fn(state=state)

# add a little noise so we have something to relax
state.positions = state.positions + torch.randn_like(state.positions) * 0.05

for step in range(20):
    state = fire_update_fn(state=state)
    print(f"{step=}: Total energy: {state.energy} eV")

# %% [markdown]
"""
In general, you can set the optimizer-specific arguments in the `optimize` function
(e.g. `unit_cell_fire`) and they will be baked into the returned functions. Fixed
parameters can usually be passed to the `init_fn` and parameters that vary over
the course of the simulation can be passed to the `update_fn`.
"""

fire_init_fn, fire_update_fn = unit_cell_fire(
    model=model,
    dt_max=0.1,
    dt_start=0.02,
)
state = fire_init_fn(state=state)

for step in range(5):
    state = fire_update_fn(state=state)

# %% [markdown]
"""
## NVT Langevin Dynamics

Similarly, we can do molecular dynamics of the systems. We need to make sure we are
using correct units for the integrator. `torch-sim` provides a `units.py` module to
help with the units system and conversions. All currently supported models implement
[MetalUnits](https://docs.lammps.org/units.html), so we must convert our units into
that system.
"""

# %%
from torch_sim.units import MetalUnits

dt = 0.002 * MetalUnits.time  # Timestep (ps)
kT = 300 * MetalUnits.temperature  # Initial temperature (K)
gamma = 10 / MetalUnits.time  # Langevin friction coefficient (ps^-1)

# %% [markdown]
"""

Like the `unit_cell_fire` optimizer, the `nvt_langevin` integrator accepts
a model and configuration kwargs and returns an `init_fn` and `update_fn`.
"""

# %%
from torch_sim.integrators import nvt_langevin

nvt_langevin_init_fn, nvt_langevin_update_fn = nvt_langevin(
    model=model, dt=dt, kT=kT, gamma=gamma
)

# we'll also reinialize the state to clean up the previous state
state = initialize_state(atoms_list, device=device, dtype=dtype)


# %% [markdown]
"""
Here we can vary the temperature of the system over time and report it as we go. The
`quantities.py` module provides a utility to compute quantities like temperature,
kinetic energy, etc. Note that the temperature will not be stable here because the
simulation is so short.
"""

# %%
from torch_sim.quantities import temperature

state = nvt_langevin_init_fn(state=state)

initial_kT = kT
for step in range(30):
    state = nvt_langevin_update_fn(state=state, kT=initial_kT * (1 + step / 30))
    if step % 5 == 0:
        temp_E_units = temperature(
            masses=state.masses, momenta=state.momenta, batch=state.batch
        )
        temp = temp_E_units / MetalUnits.temperature
        print(f"{step=}: Temperature: {temp}")

# %% [markdown]
"""
If we wanted to report the temperature over time, we could use a `TrajectoryReporter`
to save the array over time. This sort of functionality is automatically handled by the
high-level `integrate` function.

## Concluding remarks

The low-level API is a flexible and powerful way of using `torch-sim`. It provides
maximum flexibility for advanced users. If you have any additional questions, please
refer to the documentation or raise an issue on the GitHub repository.
"""
