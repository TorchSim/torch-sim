"""Batched MACE hot swap gradient descent example."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///

import os

import numpy as np
import torch
from ase.build import bulk
from ase.spacegroup import crystal
from mace.calculators.foundations_models import mace_mp

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import unit_cell_fire
from torchsim.runners import atoms_to_state
from torchsim.units import UnitConversion
from torchsim.workflows.batching_utils import (
    calculate_force_convergence,
    check_max_atoms_in_batch,
    swap_structure,
    write_log_line,
)


# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    return_raw_model=True,
    default_dtype=dtype,
    device=device,
)

# Option 2: Load the compiled model from the local file
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

PERIODIC = True

rng = np.random.default_rng(seed=0)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

# Create FCC Copper
cu_fcc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_fcc.positions += 0.2 * rng.standard_normal(cu_fcc.positions.shape)

# Create BCC Iron
fe_bcc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_bcc.positions += 0.2 * rng.standard_normal(fe_bcc.positions.shape)

# Create FCC Aluminum
al_fcc = bulk("Al", "fcc", a=4.05).repeat((2, 2, 2))
al_fcc.positions += 0.2 * rng.standard_normal(al_fcc.positions.shape)

# Create HCP Titanium
ti_hcp = bulk("Ti", "hcp", a=2.95, c=4.68).repeat((2, 2, 2))
ti_hcp.positions += 0.2 * rng.standard_normal(ti_hcp.positions.shape)

# Create binary NaCl structure
nacl = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
nacl.positions += 0.2 * rng.standard_normal(nacl.positions.shape)

# Create binary CsCl structure
cscl = bulk("CsCl", "cesiumchloride", a=4.12).repeat((2, 2, 2))
cscl.positions += 0.2 * rng.standard_normal(cscl.positions.shape)

# Create ternary perovskite BaTiO3
a = 4.0
batio3 = crystal(
    ["Ba", "Ti", "O"],
    [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
    spacegroup=221,
    cellpar=[a, a, a, 90, 90, 90],
).repeat((2, 2, 2))
batio3.positions += 0.2 * rng.standard_normal(batio3.positions.shape)

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

all_atoms_list = [si_dc, cu_fcc, fe_bcc, al_fcc, ti_hcp, nacl, cscl, batio3]
total_structures = len(all_atoms_list)
batch_size = 4
fmax = 0.05
N_steps = 10 if os.getenv("CI") else 2000
log_file = "relaxation_log.txt"
max_atoms_in_batch = 50 if os.getenv("CI") else 2000

# Run optimization for a few steps
current_idx = batch_size  # Track next structure to add
# Start with first batch_size structures
atoms_list = all_atoms_list[:batch_size].copy()
# Initialize unit cell fire optimizer
fire_init, fire_update = unit_cell_fire(model=batched_model)

# Initialize optimization
batch_state = fire_init(atoms_to_state(atoms_list, device=device, dtype=dtype))

# Main optimization loop
for step in range(N_steps):
    # Calculate force norms and check convergence for each structure in batch
    force_norms, force_mask = calculate_force_convergence(
        state=batch_state, batch_size=batch_size, fmax=fmax
    )

    # Replace converged structures if possible
    for idx, is_converged in enumerate(force_mask):
        if is_converged and current_idx < total_structures:
            next_atoms = all_atoms_list[current_idx]
            if check_max_atoms_in_batch(
                current_struct=atoms_list[idx],
                next_struct=next_atoms,
                struct_list=atoms_list,
                max_atoms=max_atoms_in_batch,
            ):
                batch_state, current_idx = swap_structure(
                    idx=idx,
                    current_idx=current_idx,
                    struct_list=atoms_list,
                    all_struct_list=all_atoms_list,
                    device=device,
                    dtype=dtype,
                    optimizer_init=fire_init,
                )

    # Log current state
    pressures = [(torch.trace(stress) / 3.0).item() for stress in batch_state.stress]
    energies = [energy.item() for energy in batch_state.energy]

    with open(log_file, "a") as f:
        write_log_line(
            f=f,
            step=step,
            properties={
                "energy": energies,
                "pressure": [p * UnitConversion.eV_per_Ang3_to_GPa for p in pressures],
                "force": force_norms,
            },
            converged=force_mask,
            batch_idx=list(range(current_idx - batch_size, current_idx)),
        )

    # Update state
    batch_state = fire_update(batch_state)

    # Break if all structures converged
    if all(force_mask):
        print("All structures converged!")
        break
