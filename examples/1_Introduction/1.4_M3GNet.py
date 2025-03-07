"""M3GNet model example."""

# /// script
# dependencies = [
#     "numpy<2",
#     "mattersim>=1.1.2",
#
# ]
# ///

import torch
from ase.build import bulk

# Import torchsim models and utilities
from torchsim.models.mattersim.m3gnet import M3GnetModel
from torchsim.models.mattersim.utils.build import batch_to_dict, build_dataloader


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Path to the model checkpoint
MODEL_PATH = "../../../checkpoints/MATTERSIM/mattersim-v1.0.0-1M.pth"
PERIODIC = True

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atomic_numbers = si_dc.get_atomic_numbers()
atoms_batch = [si_dc, si_dc]

# Initialize the M3GNet model
model = M3GnetModel(
    model=torch.load(MODEL_PATH, map_location=device),
    device=device,
    compute_force=True,
    compute_stress=True,
)

# Create a dataloader for batch processing
dataloader = build_dataloader(
    atoms_batch,
    model_type="m3gnet",
    cutoff=5.0,
    threebody_cutoff=4.0,
    batch_size=len(atoms_batch),
    only_inference=True,
    device=device,
)

# Process batch through the model
for input_dict in dataloader:
    results = model(batch_to_dict(input_dict))

# Print results
print(f"Energy: {results['energy'].shape}")
print(f"Forces: {results['forces'].shape}")
print(f"Stress: {results['stress'].shape}")

# Print the actual values
print("Energy:")
print(results["energy"])
print("Forces:")
print(results["forces"])
print("Stress:")
print(results["stress"])

# Check if the energy, forces, and stress are the same for the Si system across the batch
if len(atoms_batch) > 1:
    print("Batch consistency check:")
    e_diff = torch.max(torch.abs(results["energy"][0] - results["energy"][1]))
    f_diff = torch.max(torch.abs(results["forces"][0] - results["forces"][1]))
    s_diff = torch.max(torch.abs(results["stress"][0] - results["stress"][1]))
    print(f"Energy diff: {e_diff}")
    print(f"Forces diff: {f_diff}")
    print(f"Stress diff: {s_diff}")
