"""Examples of using the auto-batching API."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
# ]
# ///

"""Run as a interactive script."""
# ruff: noqa: E402


# %%
import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.autobatching import (
    ChunkingAutoBatcher,
    HotSwappingAutoBatcher,
    calculate_memory_scaler,
)
from torch_sim.integrators import nvt_langevin
from torch_sim.io import atoms_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import unit_cell_fire
from torch_sim.runners import generate_force_convergence_fn
from torch_sim.units import MetalUnits


if not torch.cuda.is_available():
    raise SystemExit(0)

si_atoms = bulk("Si", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))
fe_atoms = bulk("Fe", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))

device = torch.device("cuda")

mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float64,
    compute_forces=True,
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


# %% TODO: add max steps
converge_max_force = generate_force_convergence_fn(force_tol=1e-1)
single_system_memory = calculate_memory_scaler(fire_states[0])
batcher = HotSwappingAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=single_system_memory * 2.5 if os.getenv("CI") else None,
)
batcher.load_states(fire_states)
all_completed_states, convergence_tensor, state = [], None, None
while (result := batcher.next_batch(state, convergence_tensor))[0] is not None:
    state, completed_states = result
    print(f"Starting new batch of {state.n_batches} states.")

    all_completed_states.extend(completed_states)
    print("Total number of completed states", len(all_completed_states))

    for _step in range(10):
        state = fire_update(state)
    convergence_tensor = converge_max_force(state, last_energy=None)
all_completed_states.extend(result[1])
print("Total number of completed states", len(all_completed_states))


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
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=single_system_memory * 2.5 if os.getenv("CI") else None,
)
batcher.load_states(nvt_states)
finished_states = []
for batch in batcher:
    for _ in range(100):
        batch = nvt_update(batch)

    finished_states.extend(batch.split())
