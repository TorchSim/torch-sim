# ruff: noqa: E501
"""Minimal FairChem example demonstrating batching."""

# /// script
# dependencies = [
#     "fairchem-core>=1.6",
#     "torch==2.4.1",
#     "torch_geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.4.0+cpu.html",
#     "torch_scatter==2.1.2",
#     "torch_sparse==0.6.18",
#     "torch_cluster==1.6.3"
# ]
# ///

import torch
from ase.build import bulk
from fairchem.core.models.model_registry import model_name_to_local_file

from torch_sim.models.fairchem import FairChemModel
from torch_sim.state import atoms_to_state


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

MODEL_PATH = model_name_to_local_file(
    "EquiformerV2-31M-S2EF-OC20-All+MD", local_cache="."
)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
atomic_numbers = si_dc.get_atomic_numbers()
model = FairChemModel(
    model=MODEL_PATH,
    cpu=False,
    seed=0,
)
atoms_list = [si_dc, si_dc]
state = atoms_to_state(atoms_list)

results = model(state)

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
