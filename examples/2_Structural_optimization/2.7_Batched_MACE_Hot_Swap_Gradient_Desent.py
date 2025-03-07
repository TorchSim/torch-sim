"""Batched MACE hot swap gradient descent example."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///
import numpy as np
import torch
from ase.build import bulk
from mace.tools import atomic_numbers_to_indices, to_one_hot

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import unit_cell_gradient_descent


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

atomic_numbers_1 = si_dc.get_atomic_numbers()
atomic_numbers_2 = cu_dc.get_atomic_numbers()
atomic_numbers_3 = fe_dc.get_atomic_numbers()

atomic_numbers_list = [atomic_numbers_1, atomic_numbers_2]

batched_model = MaceModel(
    model=torch.load(MODEL_PATH, map_location=device),
    atomic_numbers_list=atomic_numbers_list,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

pos_1 = torch.tensor(si_dc.positions, device=device, dtype=dtype)
pos_2 = torch.tensor(cu_dc.positions, device=device, dtype=dtype)
pos_3 = torch.tensor(fe_dc.positions, device=device, dtype=dtype)

cell_1 = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
cell_2 = torch.tensor(cu_dc.cell.array, device=device, dtype=dtype)
cell_3 = torch.tensor(fe_dc.cell.array, device=device, dtype=dtype)

masses_1 = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)
masses_2 = torch.tensor(cu_dc.get_masses(), device=device, dtype=dtype)
masses_3 = torch.tensor(fe_dc.get_masses(), device=device, dtype=dtype)

positions_list = [pos_1, pos_2]
cell_list = [cell_1, cell_2]
masses_list = [masses_1, masses_2]
pbc_list = [True, True]

print(f"Positions: {positions_list}")
print(f"Cell: {cell_list}")

results = batched_model(
    positions_list, cell_list, atomic_numbers_list=atomic_numbers_list
)
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")

print(results["energy"].shape)
print(results["forces"][0].shape, results["forces"][1].shape)
print(results["stress"].shape)

# Create batch indices tensor for scatter operations
n_atoms_per_batch = [pos.shape[0] for pos in positions_list]  # List of atoms per batch
batch_indices = torch.arange(len(positions_list), device=device).repeat_interleave(
    torch.tensor(n_atoms_per_batch, device=device)
)

# Use different learning rates for each batch
fmax = 0.1
learning_rate = 0.01
learning_rates = torch.tensor([learning_rate, learning_rate], device=device, dtype=dtype)

# Initialize unit cell gradient descent optimizer
batch_state, gd_update = unit_cell_gradient_descent(
    model=batched_model,
    positions_list=positions_list,
    masses_list=masses_list,
    cell_list=cell_list,
    batch_indices=batch_indices,
    learning_rates=learning_rates,
    cell_factor=None,  # Will default to n_atoms_per_batch
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)

results = batched_model(positions_list, cell_list)
total_structures = 3
current_structures = 2
# Run optimization for a few steps
for step in range(200):
    # Split positions back into list for model input
    positions_split = torch.split(
        batch_state.positions[: sum(n_atoms_per_batch)], n_atoms_per_batch
    )
    positions_list = [pos.clone() for pos in positions_split]
    cell_list = [batch_state.cell[i].clone() for i in range(len(n_atoms_per_batch))]

    # Calculate force norms for each batch and check if they're converged
    force_norms = [forces.norm(dim=1).max().item() for forces in results["forces"]]
    force_converged = [
        norm < fmax for norm in force_norms
    ]  # Using 0.01 eV/Å as threshold

    # Replace converged structures with Fe structure
    for i, is_converged in enumerate(force_converged):
        if is_converged and current_structures < total_structures:
            positions_list[i] = pos_3.clone()
            cell_list[i] = cell_3.clone()
            atomic_numbers_list[i] = atomic_numbers_3
            masses_list[i] = masses_3.clone()
            force_converged[i] = False  # Reset convergence flag for new structure

            # Update n_atoms_per_batch
            n_atoms_per_batch = [pos.shape[0] for pos in positions_list]

            # Update calculator's batch information
            batched_model.n_atoms_per_system = n_atoms_per_batch
            batched_model.total_atoms = sum(batched_model.n_atoms_per_system)

            # Update ptr tensor
            ptr = [0]
            for n_atoms in batched_model.n_atoms_per_system:
                ptr.append(ptr[-1] + n_atoms)
            batched_model.ptr = torch.tensor(ptr, dtype=torch.long, device=device)

            # Update batch assignments
            batched_model.batch = torch.repeat_interleave(
                torch.arange(len(positions_list), device=device),
                torch.tensor(batched_model.n_atoms_per_system, device=device),
            )

            # Update node attributes
            all_atomic_numbers = [num for nums in atomic_numbers_list for num in nums]
            batched_model.node_attrs = to_one_hot(
                torch.tensor(
                    atomic_numbers_to_indices(
                        all_atomic_numbers, z_table=batched_model.z_table
                    ),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(-1),
                num_classes=len(batched_model.z_table),
            )

            # Update batch indices for optimizer
            batch_indices = torch.arange(
                len(positions_list), device=device
            ).repeat_interleave(
                torch.tensor(batched_model.n_atoms_per_system, device=device)
            )

            # Reinitialize optimizer with updated batch information
            batch_state, gd_update = unit_cell_gradient_descent(
                model=batched_model,
                positions_list=positions_list,
                masses_list=masses_list,
                cell_list=cell_list,
                batch_indices=batch_indices,
                learning_rates=learning_rates,
                cell_factor=None,
                hydrostatic_strain=False,
                constant_volume=False,
                scalar_pressure=0.0,
            )
            current_structures += 1
    # Get new results with updated atomic numbers
    results = batched_model(positions_list, cell_list)

    b1_stress = torch.trace(results["stress"][0]) * 160.21766208 / 3
    b2_stress = torch.trace(results["stress"][1]) * 160.21766208 / 3
    energy = batch_state.energy
    print(f"{step=}, E: {energy}, P: B1: {b1_stress:.4f} GPa, B2: {b2_stress:.4f} GPa")
    print(f"Max force norms: {force_norms}")
    print(f"Force converged: {force_converged}")
    batch_state = gd_update(batch_state)
