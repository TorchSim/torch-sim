import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceModel


si_atoms = bulk("Si", "fcc", a=3.26, cubic=True)
si_atoms.rattle(0.05)

cu_atoms = bulk("Cu", "fcc", a=5.26, cubic=True)
cu_atoms.rattle(0.5)

many_cu_atoms = [si_atoms] * 5 + [cu_atoms] * 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state = ts.initialize_state(many_cu_atoms, device=device, dtype=torch.float64)

mace = mace_mp(model="small", return_raw_model=True)
mace_model = MaceModel(model=mace, device=device)

fire_init, fire_update = ts.optimizers.fire(mace_model)
fire_state = fire_init(state)

batcher = ts.InFlightAutoBatcher(
    model=mace_model,
    memory_scales_with="n_atoms",
    max_memory_scaler=40,
    max_iterations=10000,  # Optional: maximum convergence attempts per state
)

batcher.load_states(fire_state)

convergence_fn = ts.generate_force_convergence_fn(5e-3, include_cell_forces=False)

all_converged_states, convergence_tensor = [], None
while (result := batcher.next_batch(fire_state, convergence_tensor))[0] is not None:
    fire_state, converged_states = result
    all_converged_states.extend(converged_states)

    for _ in range(3):
        fire_state = fire_update(fire_state)

    convergence_tensor = convergence_fn(fire_state, None)
    print(f"Convergence tensor: {convergence_tensor}")
    print(f"Convergence tensor: {batcher.current_idx}")

all_converged_states.extend(result[1])

final_states = batcher.restore_original_order(all_converged_states)
