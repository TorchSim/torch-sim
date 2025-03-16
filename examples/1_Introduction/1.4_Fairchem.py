# ruff: noqa: E501
"""Minimal FairChem example demonstrating batching."""

# /// script
# dependencies = [
#     "fairchem-core>=1.6",
#     "torch>=2.4.0,<2.5.0",
# ]
# extra_install = [
#     "pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html"
# ]
# ///

import torch
from ase.build import bulk

from torch_sim.models.fairchem import FairChemModel


device = "cuda" if torch.cuda.is_available() else "cpu"
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

# TODO: how was this ever passing?
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
