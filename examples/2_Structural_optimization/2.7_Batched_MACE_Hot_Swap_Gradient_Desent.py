"""Batched MACE hot swap gradient descent example."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.10",
# ]
# ///
import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torchsim.models.mace import MaceModel
from torchsim.neighbors import vesin_nl_ts
from torchsim.optimizers import unit_cell_gradient_descent
from torchsim.runners import atoms_to_state
from torchsim.units import UnitConversion


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

# Option 2: Load the compiled model from the local file
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

PERIODIC = True

rng = np.random.default_rng()

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

cu_dc = bulk("Cu", "fcc", a=3.85).repeat((3, 3, 3))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

fe_dc = bulk("Fe", "bcc", a=2.95).repeat((3, 3, 3))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

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

atoms_list = [si_dc, cu_dc]
base_state = atoms_to_state(atoms_list, device=device, dtype=dtype)
results = batched_model(
    base_state.positions,
    base_state.cell,
    base_state.batch,
    base_state.atomic_numbers,
)
print(f"Energy: {results['energy']}")
print(f"Forces: {results['forces']}")
print(f"Stress: {results['stress']}")

# Use different learning rates for each batch
fmax = 0.1
learning_rate = 0.01

# Initialize unit cell gradient descent optimizer
gd_init, gd_update = unit_cell_gradient_descent(
    model=batched_model,
    positions_lr=learning_rate,
    cell_lr=0.1,
    cell_factor=None,  # Will default to n_atoms_per_batch
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
)

batch_state = gd_init(base_state)

results = batched_model(
    base_state.positions, base_state.cell, base_state.batch, base_state.atomic_numbers
)
total_structures = 3
current_structures = 2
# Run optimization for a few steps
for step in range(200):
    # Calculate force norms for each batch and check if they're converged
    force_norms = torch.stack(
        [batch_state.forces[batch_state.batch == i].norm(dim=-1).max() for i in range(2)]
    ).tolist()
    force_converged = [norm < fmax for norm in force_norms]  # Using 0.1 eV/Å as threshold

    # Replace converged structures with Fe structure
    for i, is_converged in enumerate(force_converged):
        if is_converged and current_structures < total_structures:
            atoms_list[i] = fe_dc
            base_state = atoms_to_state(atoms_list, device=device, dtype=dtype)

            # Reinitialize optimizer with updated batch information
            batch_state = gd_init(base_state)

            torch.cuda.empty_cache()
            current_structures += 1

    b1_stress = torch.trace(batch_state.stress[0]) * UnitConversion.eV_per_Ang3_to_GPa / 3
    b2_stress = torch.trace(batch_state.stress[1]) * UnitConversion.eV_per_Ang3_to_GPa / 3
    PE = [energy.item() for energy in batch_state.energy]
    print(f"{step=}, E: {PE}, P: B1: {b1_stress:.4f} GPa, B2: {b2_stress:.4f} GPa")
    print(f"Max force norms: {force_norms}")
    print(f"Force converged: {force_converged}")
    batch_state = gd_update(batch_state)
