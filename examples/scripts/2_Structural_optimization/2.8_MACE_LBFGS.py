"""Batched MACE L-BFGS optimizer with ASE comparison."""

# /// script
# dependencies = ["mace-torch>=0.3.12"]
# ///
import os

import numpy as np
import torch
from ase.build import bulk
from ase.optimize import LBFGS as ASE_LBFGS
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    return_raw_model=True,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)

SMOKE_TEST = os.getenv("CI") is not None
N_steps = 10 if SMOKE_TEST else 200

rng = np.random.default_rng(seed=0)

si_dc = bulk("Si", "diamond", a=5.21).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

atoms_list = [si_dc, cu_dc, fe_dc]

model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# torch-sim batched L-BFGS
state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
initial_results = model(state)
state = ts.lbfgs_init(state=state, model=model, alpha=70.0, step_size=1.0)

for _ in range(N_steps):
    state = ts.lbfgs_step(state=state, model=model, max_history=100)

ts_final = [e.item() for e in state.energy]

# ASE L-BFGS comparison
ase_calc = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    default_dtype=str(dtype).removeprefix("torch."),
    device=str(device),
)
ase_final = []
for atoms in atoms_list:
    atoms.calc = ase_calc
    optimizer = ASE_LBFGS(atoms, logfile=None)
    optimizer.run(fmax=0.01, steps=N_steps)
    ase_final.append(atoms.get_potential_energy())

# Results
print(f"Initial energies: {[f'{e.item():.4f}' for e in initial_results['energy']]}")
print(f"torch-sim final: {[f'{e:.4f}' for e in ts_final]}")
print(f"ASE final: {[f'{e:.4f}' for e in ase_final]}")
