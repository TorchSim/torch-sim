# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///
import numpy as np
import torch
from ase.build import bulk

from torchsim.models.mace import MaceModel, UnbatchedMaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import batched_gradient_descent
from torchsim.unbatched_optimizers import gradient_descent

from mace.calculators.foundations_models import mace_mp

# Set device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)
PERIODIC = True

# Set random seed for reproducibility
rng = np.random.default_rng()

# Create diamond cubic Silicon systems
si_dc = bulk("Si", "diamond", a=5.43).repeat((4, 4, 4))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

si_dc_small = bulk("Si", "diamond", a=5.43).repeat((3, 3, 3))
si_dc_small.positions += 0.2 * rng.standard_normal(si_dc_small.positions.shape)

# Create a list of our atomic systems
atoms_list = [si_dc, si_dc, si_dc_small]

# Create batched model
batched_model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Create unbatched model for comparison
unbatched_model = UnbatchedMaceModel(
    model=loaded_model,
    atomic_numbers=si_dc.get_atomic_numbers(),
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Convert data to tensors
positions_list = [
    torch.tensor(atoms.positions, device=device, dtype=dtype) for atoms in atoms_list
]
cell_list = [
    torch.tensor(atoms.cell.array, device=device, dtype=dtype) for atoms in atoms_list
]
masses_list = [
    torch.tensor(atoms.get_masses(), device=device, dtype=dtype) for atoms in atoms_list
]
atomic_numbers_list = [atoms.get_atomic_numbers() for atoms in atoms_list]

# Create batch data format
# First concatenate positions array
positions_numpy = np.concatenate([atoms.positions for atoms in atoms_list])
positions = torch.tensor(positions_numpy, device=device, dtype=dtype)

# Stack cell vectors into a (n_batch, 3, 3) array
cell_numpy = np.stack([atoms.cell.array for atoms in atoms_list])
cell = torch.tensor(cell_numpy, device=device, dtype=dtype)

# Concatenate atomic numbers
atomic_numbers_numpy = np.concatenate(
    [atoms.get_atomic_numbers() for atoms in atoms_list]
)
atomic_numbers = torch.tensor(atomic_numbers_numpy, device=device, dtype=torch.int)

# Concatenate masses
masses_numpy = np.concatenate([atoms.get_masses() for atoms in atoms_list])
masses = torch.tensor(masses_numpy, device=device, dtype=dtype)

# Create batch indices tensor for scatter operations
atoms_per_batch = torch.tensor(
    [len(atoms) for atoms in atoms_list], device=device, dtype=torch.int
)
batch_indices = torch.repeat_interleave(
    torch.arange(len(atoms_per_batch), device=device), atoms_per_batch
)

print(f"Positions shape: {positions.shape}")
print(f"Cell shape: {cell.shape}")
print(f"Batch indices shape: {batch_indices.shape}")

# Run initial inference
results = batched_model(
    positions=positions, cell=cell, atomic_numbers=atomic_numbers, batch=batch_indices
)
# Use different learning rates for each batch
learning_rate = 0.01
learning_rates = torch.tensor(
    [learning_rate] * len(atoms_list), device=device, dtype=dtype
)

# Initialize batched gradient descent optimizer
batch_state, gd_update = batched_gradient_descent(
    model=batched_model,
    positions=positions,
    cell=cell,
    atomic_numbers=atomic_numbers,
    masses=masses,
    batch=batch_indices,
    learning_rates=learning_rates,
)

# Run batched optimization for a few steps
print("\nRunning batched gradient descent:")
for step in range(100):
    if step % 10 == 0:
        print(f"Step {step}, Energy: {batch_state.energy}")
    batch_state = gd_update(batch_state)

print(f"Final batched energy: {batch_state.energy}")

# Compare with unbatched optimization
print("\nRunning unbatched gradient descent for comparison:")
unbatched_pos = torch.tensor(si_dc.positions, device=device, dtype=dtype)
unbatched_cell = torch.tensor(si_dc.cell.array, device=device, dtype=dtype)
unbatched_masses = torch.tensor(si_dc.get_masses(), device=device, dtype=dtype)

state, single_gd_update = gradient_descent(
    model=unbatched_model,
    positions=unbatched_pos,
    masses=unbatched_masses,
    cell=unbatched_cell,
    pbc=PERIODIC,
    learning_rate=learning_rate,
)

for step in range(100):
    if step % 10 == 0:
        print(f"Step {step}, Energy: {state.energy}")
    state = single_gd_update(state)

print(f"Final unbatched energy: {state.energy}")

# Compare final results between batched and unbatched
print("\nComparison between batched and unbatched results:")
print(f"Energy difference: {torch.max(torch.abs(batch_state.energy[0] - state.energy))}")
