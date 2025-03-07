"""Batched MACE unit cell filter with gradient descent optimizer."""

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

# Option 2: Load from local file (comment out Option 1 to use this)
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

# Set random seed for reproducibility
PERIODIC = True
rng = np.random.default_rng()

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.21, cubic=True).repeat((2, 2, 2))
si_dc.positions += 0.2 * rng.standard_normal(si_dc.positions.shape)

# Create FCC Copper
cu_dc = bulk("Cu", "fcc", a=3.85).repeat((2, 2, 2))
cu_dc.positions += 0.2 * rng.standard_normal(cu_dc.positions.shape)

# Create BCC Iron
fe_dc = bulk("Fe", "bcc", a=2.95).repeat((2, 2, 2))
fe_dc.positions += 0.2 * rng.standard_normal(fe_dc.positions.shape)

# Create a list of our atomic systems
atoms_list = [si_dc, cu_dc, fe_dc]

# Print structure information
print(f"Silicon atoms: {len(si_dc)}")
print(f"Copper atoms: {len(cu_dc)}")
print(f"Iron atoms: {len(fe_dc)}")
print(f"Total number of structures: {len(atoms_list)}")

# Create batched model
model = MaceModel(
    model=loaded_model,
    device=device,
    neighbor_list_fn=vesin_nl_ts,
    periodic=PERIODIC,
    compute_force=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)

# Convert atoms to state
state = atoms_to_state(atoms_list, device=device, dtype=dtype)
# Run initial inference
results = model(
    positions=state.positions,
    cell=state.cell,
    atomic_numbers=state.atomic_numbers,
    batch=state.batch,
)
# Use same learning rate for all batches
positions_lr = 0.01
cell_lr = 0.1

# Initialize unit cell gradient descent optimizer
gd_init, gd_update = unit_cell_gradient_descent(
    model=model,
    cell_factor=None,  # Will default to atoms per batch
    hydrostatic_strain=False,
    constant_volume=False,
    scalar_pressure=0.0,
    positions_lr=positions_lr,
    cell_lr=cell_lr,
)

state = gd_init(state)

# Run optimization for a few steps
print("\nRunning batched unit cell gradient descent:")
for step in range(200):
    P1 = torch.trace(state.stress[0]) * UnitConversion.eV_per_Ang3_to_GPa / 3
    P2 = torch.trace(state.stress[1]) * UnitConversion.eV_per_Ang3_to_GPa / 3
    P3 = torch.trace(state.stress[2]) * UnitConversion.eV_per_Ang3_to_GPa / 3

    if step % 20 == 0:
        print(
            f"Step {step}, Energy: {[energy.item() for energy in state.energy]}, "
            f"P1={P1:.4f} GPa, P2={P2:.4f} GPa, P3={P3:.4f} GPa"
        )

    state = gd_update(state)

print(f"Initial energies: {[energy.item() for energy in results['energy']]} eV")
print(f"Final energies: {[energy.item() for energy in state.energy]} eV")

initial_pressure = [
    torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
    for stress in results["stress"]
]
final_pressure = [
    torch.trace(stress).item() * UnitConversion.eV_per_Ang3_to_GPa / 3
    for stress in state.stress
]
print(f"{initial_pressure=} GPa")
print(f"{final_pressure=} GPa")
