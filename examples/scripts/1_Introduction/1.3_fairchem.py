# ruff: noqa: E501
"""Minimal FairChem example demonstrating batching."""

# /// script
# dependencies = [
#     "fairchem-core>=2.2.0",
# ]
# ///

import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.fairchem import FairChemModel


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# UMA = Unified Machine Learning for Atomistic simulations
MODEL_NAME = "uma-s-1"

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
atomic_numbers = si_dc.get_atomic_numbers()
model = FairChemModel(
    model=None,
    model_name=MODEL_NAME,
    task_name="omat",  # Open Materials task for crystalline systems
    cpu=False,
)
atoms_list = [si_dc, si_dc]
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)

results = model(state)

print(results["energy"].shape)
print(results["forces"].shape)
if stress := results.get("stress"):
    print(stress.shape)

print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
if stress := results.get("stress"):
    print(f"{stress=}")

# Check if the energy, forces, and stress are the same for the Si system across the batch
print(torch.max(torch.abs(results["energy"][0] - results["energy"][1])))
print(torch.max(torch.abs(results["forces"][0] - results["forces"][1])))
if stress := results.get("stress"):
    print(torch.max(torch.abs(stress[0] - stress[1])))
