"""Batched MACE hot swap gradient descent example."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
#     "matbench-discovery>=1.3.1",
# ]
# ///

import os
import time

import numpy as np
import pandas as pd
import torch
from mace.calculators.foundations_models import mace_mp
from matbench_discovery.data import DataFiles
from pymatgen.core import Structure

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import unit_cell_fire
from torchsim.runners import structures_to_state
from torchsim.units import UnitConversion
from torchsim.workflows.batching_utils import (
    calculate_force_convergence,
    check_max_atoms_in_batch,
    swap_structure,
    write_log_line,
)


# WBM initial structures in pymatgen JSON format
df_init_structs = pd.read_json(DataFiles.wbm_initial_structures.path)

# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model"
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

max_structures_to_relax = 2 if os.getenv("CI") else 200
all_struct_list = []
for struct_dict in df_init_structs["initial_structure"][0:max_structures_to_relax]:
    struct = Structure.from_dict(struct_dict)
    all_struct_list.append(struct)

total_structures = len(all_struct_list)
batch_size = 2 if os.getenv("CI") else 10
fmax = 0.05
N_steps = 10 if os.getenv("CI") else 200_000_000
log_file = "WBM_relaxation_log.txt"
max_atoms_in_batch = 50 if os.getenv("CI") else 2000

# Run optimization for a few steps
current_idx = batch_size  # Track next structure to add
# Start with first batch_size structures
struct_list = all_struct_list[:batch_size].copy()
# Initialize unit cell fire optimizer
fire_init, fire_update = unit_cell_fire(model=batched_model)

# Initialize optimization
batch_state = fire_init(structures_to_state(struct_list, device=device, dtype=dtype))

start_time = time.perf_counter()
# Main optimization loop
for step in range(N_steps):
    # Calculate force norms and check convergence for each structure in batch
    force_norms, force_mask = calculate_force_convergence(
        state=batch_state, batch_size=batch_size, fmax=fmax
    )

    # Replace converged structures if possible
    for idx, is_converged in enumerate(force_mask):
        if is_converged and current_idx < total_structures:
            next_atoms = all_struct_list[current_idx]
            if check_max_atoms_in_batch(
                current_struct=struct_list[idx],
                next_struct=next_atoms,
                struct_list=struct_list,
                max_atoms=max_atoms_in_batch,
            ):
                batch_state, current_idx = swap_structure(
                    idx=idx,
                    current_idx=current_idx,
                    struct_list=struct_list,
                    all_struct_list=all_struct_list,
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

end_time = time.perf_counter()
print(f"Total time taken for WBM relaxation: {end_time - start_time} seconds")
