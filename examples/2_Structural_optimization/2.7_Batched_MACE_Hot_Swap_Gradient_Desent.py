import numpy as np
import torch
from ase.build import bulk

from torchsim.optimizers import unit_cell_gradient_descent
from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.runners import atoms_to_state
from torchsim.units import UnitConversion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
PERIODIC = True

rng = np.random.default_rng()

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43).repeat((4, 4, 4))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((3, 3, 3))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((3, 3, 3))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

batched_model = MaceModel(
    model=torch.load(MODEL_PATH, map_location=device),
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

atoms_list = [si_dc, cu_dc]
state = atoms_to_state(atoms_list, device=device, dtype=dtype)

results = batched_model(
    state.positions, state.cell, atomic_numbers=state.atomic_numbers, batch=state.batch
)
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")
# Use different learning rates for each batch
fmax = 0.1
learning_rate = 0.01
learning_rates = torch.tensor([learning_rate, learning_rate], device=device, dtype=dtype)

# Initialize unit cell gradient descent optimizer
gd_init, gd_update = unit_cell_gradient_descent(
    model=batched_model,
    positions_lr=learning_rate,
    cell_lr=0.1,
)

batch_state = gd_init(state)
results = batched_model(
    batch_state.positions,
    batch_state.cell,
    atomic_numbers=batch_state.atomic_numbers,
    batch=batch_state.batch,
)
total_structures = 3
current_structures = 2
# Run optimization for a few steps
for step in range(200):
    results = batched_model(
        batch_state.positions,
        batch_state.cell,
        atomic_numbers=batch_state.atomic_numbers,
        batch=batch_state.batch,
    )

    # Calculate force norms for each batch
    # Split forces by batch
    forces_by_batch = []
    start_idx = 0
    for n_atoms in batched_model.n_atoms_per_system:
        forces_by_batch.append(results["forces"][start_idx : start_idx + n_atoms])
        start_idx += n_atoms

    # Calculate max force norm for each batch
    force_norms = [forces.norm(dim=-1).max().item() for forces in forces_by_batch]
    force_converged = [norm < fmax for norm in force_norms]

    # Replace converged structures with Fe structure
    for i, is_converged in enumerate(force_converged):
        if is_converged and current_structures < total_structures:
            # Remove converged structure and add new one
            atoms_list.pop(i)
            atoms_list.append(fe_dc)
            state = atoms_to_state(atoms_list, device=device, dtype=dtype)
            batch_state = gd_init(state)
            force_converged[i] = False  # Reset convergence flag for new structure

            batched_model.setup_from_batch(batch_state.atomic_numbers, batch_state.batch)
            # Reinitialize optimizer with updated batch information
            gd_init, gd_update = unit_cell_gradient_descent(
                model=batched_model,
                positions_lr=learning_rate,
                cell_lr=0.1,
            )
            current_structures += 1
    # Get new results with updated atomic numbers
    results = batched_model(
        batch_state.positions,
        batch_state.cell,
        atomic_numbers=batch_state.atomic_numbers,
        batch=batch_state.batch,
    )

    PE_batch = [energy.item() for energy in batch_state.energy]
    P1 = [
        torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
        for stress in results["stress"]
    ]
    P2 = [
        torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
        for stress in results["stress"]
    ]
    print(f"{step=}, E: {PE_batch}, P: B1: {P1}, B2: {P2}")
    print(f"Max force norms: {force_norms}")
    print(f"Force converged: {force_converged}")
    batch_state = gd_update(batch_state)
