"""Examples of using the auto-batching API. Meant to be run as an interactive script."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
# %%
import os

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.autobatching import (
    BinningAutoBatcher,
    InFlightAutoBatcher,
    calculate_memory_scaler,
)
from torch_sim.models.mace import MaceModel
from torch_sim.runners import generate_force_convergence_fn
from torch_sim.units import MetalUnits


if not torch.cuda.is_available():
    raise SystemExit(0)

SMOKE_TEST = os.getenv("CI") is not None

si_atoms = bulk("Si", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))
fe_atoms = bulk("Fe", "fcc", a=5.43, cubic=True).repeat((3, 3, 3))
state: ts.FireState | None = None
device = torch.device("cuda")

mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float64,
    compute_forces=True,
)

si_state = ts.io.atoms_to_state(si_atoms, device=device, dtype=torch.float64)
fe_state = ts.io.atoms_to_state(fe_atoms, device=device, dtype=torch.float64)

state = ts.fire_init(model=mace_model, state=si_state, cell_filter=ts.CellFilter.unit)

si_fire_state = ts.fire_init(
    model=mace_model, state=si_state, cell_filter=ts.CellFilter.unit
)
fe_fire_state = ts.fire_init(
    model=mace_model, state=fe_state, cell_filter=ts.CellFilter.unit
)

fire_states = [si_fire_state, fe_fire_state] * (2 if SMOKE_TEST else 20)
fire_states = [state.clone() for state in fire_states]
for state in fire_states:
    state.positions += torch.randn_like(state.positions) * 0.01

len(fire_states)


# %% TODO: add max steps
converge_max_force = generate_force_convergence_fn(force_tol=1e-1)
single_system_memory = calculate_memory_scaler(fire_states[0])
batcher = InFlightAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=single_system_memory * 2.5 if SMOKE_TEST else None,
)
batcher.load_states(fire_states)
all_completed_states, convergence_tensor, state = [], None, None
while (result := batcher.next_batch(state, convergence_tensor))[0] is not None:
    state, completed_states = result[0], result[1]
    print(f"Starting new batch of {state.n_systems} states.")

    all_completed_states.extend(completed_states)
    print(f"Total number of completed states {len(all_completed_states)}")

    for _step in range(10):
        state = ts.fire_step(model=mace_model, state=state)
    convergence_tensor = converge_max_force(state, last_energy=None)
all_completed_states.extend(result[1])
print(f"Total number of completed states {len(all_completed_states)}")


# %% run binning autobatcher
si_nvt_state = ts.nvt_langevin_init(
    model=mace_model,
    state=si_state,
    dt=torch.tensor(0.001),
    kT=torch.tensor(300 * MetalUnits.temperature),
)
fe_nvt_state = ts.nvt_langevin_init(
    model=mace_model,
    state=fe_state,
    dt=torch.tensor(0.001),
    kT=torch.tensor(300 * MetalUnits.temperature),
)

si_state = ts.io.atoms_to_state(si_atoms, device=device, dtype=torch.float64)
fe_state = ts.io.atoms_to_state(fe_atoms, device=device, dtype=torch.float64)

nvt_states = [si_nvt_state, fe_nvt_state] * (2 if SMOKE_TEST else 20)
nvt_states = [state.clone() for state in nvt_states]
for state in nvt_states:
    state.positions += torch.randn_like(state.positions) * 0.01


single_system_memory = calculate_memory_scaler(fire_states[0])
batcher = BinningAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms_x_density",
    max_memory_scaler=single_system_memory * 2.5 if SMOKE_TEST else None,
)
batcher.load_states(nvt_states)
finished_states: list[ts.SimState] = []
for batch, _indices in batcher:
    for _ in range(100):
        batch = ts.nvt_langevin_step(model=mace_model, state=batch)

    finished_states.extend(batch.split())
