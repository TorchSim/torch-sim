"""Minimal FairChem example demonstrating batching."""

# /// script
# dependencies = ["fairchem-core>=2.2.0", "huggingface_hub"]
# ///

import os

import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.fairchem import FairChemModel


# Optional Hugging Face login if HF_TOKEN is available (for private model access)
try:
    from huggingface_hub import login as hf_login  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    hf_login = None  # type: ignore[assignment]

hf_token = os.environ.get("HF_TOKEN")
if hf_token and hf_login is not None:
    hf_login(token=hf_token)
else:
    print("Need to login to HuggingFace to access fairchem models")
    raise SystemExit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# UMA = Unified Machine Learning for Atomistic simulations
MODEL_NAME = "uma-s-1"

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
atomic_numbers = si_dc.get_atomic_numbers()
model = FairChemModel(
    model=MODEL_NAME,
    task_name="omat",  # Open Materials task for crystalline systems
    cpu=False,
)
atoms_list = [si_dc, si_dc]
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)

results = model(state)

print(results["energy"].shape)
print(results["forces"].shape)
if "stress" in results:
    print(results["stress"].shape)

print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
if "stress" in results:
    print(f"{results['stress']=}")

# Check if the energy, forces, and stress are the same for the Si system across the batch
print(torch.max(torch.abs(results["energy"][0] - results["energy"][1])))
print(torch.max(torch.abs(results["forces"][0] - results["forces"][1])))
if "stress" in results:
    print(torch.max(torch.abs(results["stress"][0] - results["stress"][1])))
