"""Minimal FairChem example demonstrating batching."""

# /// script
# dependencies = [
#     "fairchem-core>=1.6",
#     "torch",
#     "torch-geometric>=2.6.1",
#     "torch-scatter==2.1.2",
#     "torch-sparse==0.6.18",
#     "torch-cluster==1.6.3",
# ]
# ///
import torch
from ase.build import bulk

from torchsim.models.fairchem import FairChemModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

MODEL_PATH = "../../../checkpoints/FAIRCHEM/EquiformerV2-31M-S2EF-OC20-All+MD.pt"
PERIODIC = True

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43).repeat((4, 4, 4))
atomic_numbers = si_dc.get_atomic_numbers()
batched_model = FairChemModel(
    model=MODEL_PATH,
    atomic_numbers_list=[atomic_numbers, atomic_numbers],
    pbc=PERIODIC,
    cpu=False,
    seed=0,
)

positions = torch.tensor([si_dc.positions, si_dc.positions], device=device, dtype=dtype)
cell = torch.tensor([si_dc.cell.array, si_dc.cell.array], device=device, dtype=dtype)
masses = torch.tensor(
    [si_dc.get_masses(), si_dc.get_masses()], device=device, dtype=dtype
)
pbc = torch.tensor([True, True], device=device, dtype=torch.bool)

# Convert batched tensors to lists for the calculator
cell_list = [cell[idx] for idx in range(len(cell))]
positions_list = [positions[idx] for idx in range(len(positions))]

print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")

results = batched_model(positions_list, cell_list)

print(results["energy"].shape)
print(results["forces"].shape)
print(results["stress"].shape)

print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")

# Check if the energy, forces, and stress are the same for the Si system across the batch
print(torch.max(torch.abs(results["energy"][0] - results["energy"][1])))
print(torch.max(torch.abs(results["forces"][0] - results["forces"][1])))
print(torch.max(torch.abs(results["stress"][0] - results["stress"][1])))
