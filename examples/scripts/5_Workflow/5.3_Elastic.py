"""Bulk and Shear modulus with MACE."""

# /// script
# dependencies = ["ase>=3.26", "mace-torch>=0.3.12"]
# ///
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

import torch_sim as ts
from torch_sim.elastic import get_bravais_type
from torch_sim.models.mace import MaceModel, MaceUrls


# Calculator
unit_conv = ts.units.UnitConversion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

loaded_model = mace_mp(
    model=MaceUrls.mace_mpa_medium,
    enable_cueq=False,
    device=str(device),
    default_dtype=str(dtype).removeprefix("torch."),
    return_raw_model=True,
)

# ASE structure
struct = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))

model = MaceModel(
    model=loaded_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)
# Target force tolerance
fmax = 1e-3

# Relax positions and cell
state = ts.io.atoms_to_state(atoms=struct, device=device, dtype=dtype)
state = ts.fire_init(
    model=model, state=state, scalar_pressure=0.0, cell_filter=ts.CellFilter.frechet
)

for step in range(300):
    pressure = -torch.trace(state.stress.squeeze()) / 3 * unit_conv.eV_per_Ang3_to_GPa
    current_fmax = torch.max(torch.abs(state.forces.squeeze()))
    print(
        f"Step {step}, Energy: {state.energy.item():.4f}, "
        f"Pressure: {pressure.item():.4f}, "
        f"Fmax: {current_fmax.item():.4f}"
    )
    if current_fmax < fmax and abs(pressure) < 1e-2:
        break
    state = ts.fire_step(model=model, state=state)

# Get bravais type
bravais_type = get_bravais_type(state)

# Calculate elastic tensor
elastic_tensor = ts.elastic.calculate_elastic_tensor(
    state=state, model=model, bravais_type=bravais_type
)

# Convert to GPa
elastic_tensor = elastic_tensor * unit_conv.eV_per_Ang3_to_GPa

# Calculate elastic moduli
bulk_modulus, shear_modulus, poisson_ratio, pugh_ratio = (
    ts.elastic.calculate_elastic_moduli(elastic_tensor)
)

# Print elastic tensor
print("\nElastic tensor (GPa):")
elastic_tensor_np = elastic_tensor.cpu().numpy()
for row in elastic_tensor_np:
    print("  " + "  ".join(f"{val:10.4f}" for val in row))

# Print mechanical moduli
print(f"Bulk modulus (GPa): {bulk_modulus:.4f}")
print(f"Shear modulus (GPa): {shear_modulus:.4f}")
print(f"Poisson's ratio: {poisson_ratio:.4f}")
print(f"Pugh's ratio (K/G): {pugh_ratio:.4f}")
