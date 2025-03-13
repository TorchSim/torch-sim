"""Examples of using the auto-batching API."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///

"""Run as a interactive script."""
# ruff: noqa: E402


# %%
import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torchsim.autobatching import (
    ChunkingAutoBatcher,
    HotSwappingAutoBatcher,
    calculate_memory_scaler,
    split_state,
)
from torchsim.integrators import nvt_langevin
from torchsim.models.mace import MaceModel
from torchsim.optimizers import unit_cell_fire
from torchsim.runners import atoms_to_state
from torchsim.state import BaseState
from torchsim.units import MetalUnits


if not torch.cuda.is_available():
    raise SystemExit(0)

si_atoms = bulk("Si", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))
fe_atoms = bulk("Fe", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))

device = torch.device("cuda")

mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    periodic=True,
    dtype=torch.float64,
    compute_force=True,
)

si_state = atoms_to_state(si_atoms, device=device, dtype=torch.float64)
fe_state = atoms_to_state(fe_atoms, device=device, dtype=torch.float64)

fire_init, fire_update = unit_cell_fire(mace_model)

si_fire_state = fire_init(si_state)
fe_fire_state = fire_init(fe_state)

fire_states = [si_fire_state, fe_fire_state] * (2 if os.getenv("CI") else 20)
fire_states = [state.clone() for state in fire_states]
for state in fire_states:
    state.positions += torch.randn_like(state.positions) * 0.01

len(fire_states)


# %% run hot swapping autobatcher
def convergence_fn(state: BaseState) -> bool:
    """Check if the system has converged."""
    batch_wise_max_force = torch.zeros(state.n_batches, device=state.device)
    max_forces = state.forces.norm(dim=1)
    batch_wise_max_force = batch_wise_max_force.scatter_reduce(
        dim=0,
        index=state.batch,
        src=max_forces,
        reduce="amax",
    )
    return batch_wise_max_force < 1e-1


single_system_memory = calculate_memory_scaler(fire_states[0])
batcher = HotSwappingAutoBatcher(
    model=mace_model,
    states=fire_states,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=single_system_memory * 2.5 if os.getenv("CI") else None,
)

all_completed_states, convergence_tensor = [], None
while True:
    print(f"Starting new batch of {state.n_batches} states.")

    state, completed_states = batcher.next_batch(state, convergence_tensor)
    print("Number of completed states", len(completed_states))

    all_completed_states.extend(completed_states)
    if state is None:
        break

    # run 10 steps, arbitrary number
    for _step in range(10):
        state = fire_update(state)
    convergence_tensor = convergence_fn(state)


# %% run chunking autobatcher
nvt_init, nvt_update = nvt_langevin(
    model=mace_model, dt=0.001, kT=300 * MetalUnits.temperature
)


si_state = atoms_to_state(si_atoms, device=device, dtype=torch.float64)
fe_state = atoms_to_state(fe_atoms, device=device, dtype=torch.float64)

si_nvt_state = nvt_init(si_state)
fe_nvt_state = nvt_init(fe_state)

nvt_states = [si_nvt_state, fe_nvt_state] * (2 if os.getenv("CI") else 20)
nvt_states = [state.clone() for state in nvt_states]
for state in nvt_states:
    state.positions += torch.randn_like(state.positions) * 0.01


single_system_memory = calculate_memory_scaler(fire_states[0])
batcher = ChunkingAutoBatcher(
    model=mace_model,
    states=nvt_states,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=single_system_memory * 2.5 if os.getenv("CI") else None,
)

finished_states = []
for batch in batcher:
    for _ in range(100):
        batch = nvt_update(batch)

    finished_states.extend(split_state(batch))
