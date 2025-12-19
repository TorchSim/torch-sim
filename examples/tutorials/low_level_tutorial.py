# %%
# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///


# %% [markdown]
"""
# Fundamentals of TorchSim

The TorchSim package is designed to be both flexible and easy to use. It achieves this
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

TorchSim's state aka `SimState` is a class that contains the information of the
system like positions, cell, etc. of the system(s). All the models in the TorchSim
package take in a `SimState` as an input and return the properties of the system(s).

First we will create two simple structures of 2x2x2 unit cells of Body Centered Cubic
(BCC) Iron and Diamond Cubic Silicon and combine them into a batched state.
"""

# %%
from ase.build import bulk
import torch
import torch_sim as ts

si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
fe_bcc = bulk("Fe", "bcc", a=2.8665, cubic=True).repeat((3, 3, 3))
atoms_list = [si_dc, fe_bcc]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

state = ts.initialize_state(atoms_list, device=device, dtype=dtype)


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
from torch_sim.models.mace import MaceModel, MaceUrls

# load mace_mp using the mace package
loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

# wrap the mace_mp model in the MaceModel class
model = MaceModel(
    model=loaded_model,  # type: ignore[arg-type]
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
)


# %% [markdown]
"""
TorchSim's MaceModel, and the other MLIP models, are wrappers around the raw models
that allow them to interface with the rest of the TorchSim package. They expose
several key properties that are expected by the rest of the package. This contract is
enforced by the `ModelInterface` class that all models must implement.
"""

# %%
print(f"{model.device=}")
print(f"{model.dtype=}")
print(f"{model.compute_forces=}")
print(f"{model.compute_stress=}")
print(f"{model.memory_scales_with=}")  # see the autobatching tutorial for more details


# %% [markdown]
"""
`SimState` objects can be passed directly to the model and it will compute
the properties of the systems in the batch. The properties will be returned
either systemwise, like the energy, or atomwise, like the forces.

Note that the energy here refers to the potential energy of the system.
"""

# %%
model_outputs = model(state)
print(f"Model outputs: {', '.join(list(model_outputs))}")
print(f"Energy is a systemwise property with shape: {model_outputs['energy'].shape}")
print(f"Forces are an atomwise property with shape: {model_outputs['forces'].shape}")
print(f"Stress is a systemwise property with shape: {model_outputs['stress'].shape}")


# %% [markdown]
"""
## Optimizers and Integrators

All optimizers and integrators have an associated `init_fn` and a `step_fn`.
The `init_fn` function returns the initialized optimizer-specific state,
while the `step_fn` function updates the simulation state. The formal pairings
are stored in the `ts.INTEGRATOR_REGISTRY` and `ts.OPTIM_REGISTRY` dictionaries.

### Unit Cell Fire

We will walk through the fire optimizer with unit cell filter as an example.
"""

# %%
state = ts.fire_init(state=state, model=model, cell_filter=ts.CellFilter.unit)

# add a little noise so we have something to relax
state.positions = state.positions + torch.randn_like(state.positions) * 0.05


# %% [markdown]
"""
We can then initialize the state and evolve the system with the update function.
Of course, we could also enforce some convergence criteria on the energy or forces
and stop the optimization early. Functionality that is automatically handled by the
high-level API.
"""

# %%

for step in range(20):
    state = ts.fire_step(state=state, model=model)
    print(f"{step=}: Total energy: {state.energy} eV")


# %% [markdown]
"""
Fixed parameters can usually be passed to the `init_fn` and parameters that vary over
the course of the simulation can be passed to the `step_fn`. In the `optimize`
function, you set these with the `init_kwargs` and `optimizer_kwargs` arguments.
"""

# %%
state = ts.fire_init(
    state=state, model=model, dt_start=0.02, cell_filter=ts.CellFilter.unit
)

for step in range(5):
    state = ts.fire_step(state=state, model=model, dt_max=0.1)


# %% [markdown]
"""
## NVT Langevin Dynamics

Similarly, we can do molecular dynamics of the systems. We need to make sure we are
using correct units for the integrator. TorchSim provides a `units.py` module to
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
Like the `fire` optimizer with unit cell filter, the `nvt_langevin` integrator accepts
a model, state and config kwargs.
"""

# %% we'll also reinitialize the state to clean up the previous state
state = ts.initialize_state(atoms_list, device=device, dtype=dtype)


# %% [markdown]
"""
Here we can vary the temperature of the system over time and report it as we go. The
`quantities.py` module provides a utility to compute quantities like temperature,
kinetic energy, etc. Note that the temperature will not be stable here because the
simulation is so short.
"""

# %%
state = ts.nvt_langevin_init(state=state, model=model, kT=kT)

initial_kT = kT
for step in range(30):
    current_kT = initial_kT * (1 + step / 30)
    state = ts.nvt_langevin_step(
        state=state, model=model, dt=dt, kT=current_kT, gamma=gamma
    )
    if step % 5 == 0:
        temp_E_units = ts.calc_kT(
            masses=state.masses, momenta=state.momenta, system_idx=state.system_idx
        )
        temp = temp_E_units / MetalUnits.temperature
        print(f"{step=}: Temperature: {temp}")


# %% [markdown]
"""
If we wanted to report the temperature over time, we could use a `TrajectoryReporter`
to save the array over time. This sort of functionality is automatically handled by the
high-level `integrate` function.

## Concluding remarks

The low-level API is a flexible and powerful way of using TorchSim. It provides
maximum flexibility for advanced users. If you have any additional questions, please
refer to the documentation or raise an issue on the GitHub repository.
"""
