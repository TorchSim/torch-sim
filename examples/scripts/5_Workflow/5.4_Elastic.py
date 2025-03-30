"""Bulk and Shear modulus with MACE."""

# /// script
# dependencies = [
#     "ase>=3.24",
#     "mace-torch>=0.3.11",
# ]
# ///

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from torch_sim.elastic import (
    BravaisType,
    calculate_elastic_moduli,
    calculate_elastic_tensor,
)
from torch_sim.io import atoms_to_state
from torch_sim.models.mace import MaceModel
from torch_sim.optimizers import frechet_cell_fire
from torch_sim.state import SimState
from torch_sim.units import UnitConversion


def get_bravais_type(state: SimState, tol: float = 1e-3) -> BravaisType:  # noqa : PLR0911
    """Check and return the crystal system of a structure.

    This function determines the crystal system by analyzing the lattice
    parameters and angles without using spglib.

    Args:
        state: SimState object representing the crystal structure
        tol: Tolerance for floating-point comparisons

    Returns:
        BravaisType: Bravais type
    """
    # Get cell parameters
    cell = state.cell.squeeze()
    a, b, c = torch.linalg.norm(cell, axis=1)

    # Get cell angles in degrees
    alpha = torch.rad2deg(torch.arccos(torch.dot(cell[1], cell[2]) / (b * c)))
    beta = torch.rad2deg(torch.arccos(torch.dot(cell[0], cell[2]) / (a * c)))
    gamma = torch.rad2deg(torch.arccos(torch.dot(cell[0], cell[1]) / (a * b)))

    # Cubic: a = b = c, alpha = beta = gamma = 90°
    if (
        abs(a - b) < tol
        and abs(b - c) < tol
        and abs(alpha - 90) < tol
        and abs(beta - 90) < tol
        and abs(gamma - 90) < tol
    ):
        return BravaisType.CUBIC

    # Hexagonal: a = b ≠ c, alpha = beta = 90°, gamma = 120°
    if (
        abs(a - b) < tol
        and abs(alpha - 90) < tol
        and abs(beta - 90) < tol
        and abs(gamma - 120) < tol
    ):
        return BravaisType.HEXAGONAL

    # Tetragonal: a = b ≠ c, alpha = beta = gamma = 90°
    if (
        abs(a - b) < tol
        and abs(a - c) > tol
        and abs(alpha - 90) < tol
        and abs(beta - 90) < tol
        and abs(gamma - 90) < tol
    ):
        return BravaisType.TETRAGONAL

    # Orthorhombic: a ≠ b ≠ c, alpha = beta = gamma = 90°
    if (
        abs(alpha - 90) < tol
        and abs(beta - 90) < tol
        and abs(gamma - 90) < tol
        and abs(a - b) > tol
        and (abs(b - c) > tol or abs(a - c) > tol)
    ):
        return BravaisType.ORTHORHOMBIC

    # Monoclinic: a ≠ b ≠ c, alpha = gamma = 90°, beta ≠ 90°
    if abs(alpha - 90) < tol and abs(gamma - 90) < tol and abs(beta - 90) > tol:
        return BravaisType.MONOCLINIC

    # Trigonal/Rhombohedral: a = b = c, alpha = beta = gamma ≠ 90°
    if (
        abs(a - b) < tol
        and abs(b - c) < tol
        and abs(alpha - beta) < tol
        and abs(beta - gamma) < tol
        and abs(alpha - 90) > tol
    ):
        return BravaisType.TRIGONAL

    # Triclinic: a ≠ b ≠ c, alpha ≠ beta ≠ gamma ≠ 90°
    return BravaisType.TRICLINIC


# Calculator
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
loaded_model = mace_mp(
    model=mace_checkpoint_url,
    enable_cueq=False,
    device=device,
    default_dtype="float64",
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
fire_init, fire_update = frechet_cell_fire(model=model, scalar_pressure=0.0)

state = atoms_to_state(atoms=struct, device=device, dtype=dtype)
state = fire_init(state=state)

for step in range(300):
    pressure = (
        -torch.trace(state.stress.squeeze()) / 3 * UnitConversion.eV_per_Ang3_to_GPa
    )
    current_fmax = torch.max(torch.abs(state.forces.squeeze()))
    print(
        f"Step {step}, Energy: {state.energy.item():.4f}, "
        f"Pressure: {pressure.item():.4f}, "
        f"Fmax: {current_fmax.item():.4f}"
    )
    if current_fmax < fmax and abs(pressure) < 1e-2:
        break
    state = fire_update(state=state)

# Get bravais type
bravais_type = get_bravais_type(state)

# Calculate elastic tensor
elastic_tensor = calculate_elastic_tensor(model, state=state, bravais_type=bravais_type)

# Convert to GPa
elastic_tensor = elastic_tensor * UnitConversion.eV_per_Ang3_to_GPa

# Calculate elastic moduli
bulk_modulus, shear_modulus, poisson_ratio, pugh_ratio = calculate_elastic_moduli(
    elastic_tensor
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
