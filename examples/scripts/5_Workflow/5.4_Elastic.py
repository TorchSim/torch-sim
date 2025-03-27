"""Bulk and Shear modulus with MACE."""

# /// script
# dependencies = [
#     "ase>=3.24",
#     "mace-torch>=0.3.11",
# ]
# ///

import torch
from ase import units
from ase.atoms import Atoms
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE as FIRE_ASE
from ase.spacegroup import get_spacegroup
from mace.calculators.foundations_models import mace_mp
from torch_sim.elastic import calculate_elastic_tensor, BravaisType, calculate_elastic_moduli
import numpy as np

def get_bravais_type(struct: Atoms, tol: float = 1e-3):
    """Check and return the crystal system of a structure.
    
    This function determines the crystal system by analyzing the lattice
    parameters and angles without using spglib.
    
    Args:
        struct: ASE Atoms object representing the crystal structure
        tol: Tolerance for floating-point comparisons

    Returns:
        BravaisType: Bravais type
    """
    # Get cell parameters
    cell = struct.get_cell()
    a, b, c = np.linalg.norm(cell, axis=1)
    
    # Get cell angles in degrees
    alpha = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))
    
    # Cubic: a = b = c, alpha = beta = gamma = 90°
    if (abs(a - b) < tol and abs(b - c) < tol and 
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return BravaisType.CUBIC
            
    # Hexagonal: a = b ≠ c, alpha = beta = 90°, gamma = 120°
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
          abs(beta - 90) < tol and abs(gamma - 120) < tol):
        return BravaisType.HEXAGONAL
        
    # Tetragonal: a = b ≠ c, alpha = beta = gamma = 90°
    elif (abs(a - b) < tol and abs(a - c) > tol and
          abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return BravaisType.TETRAGONAL
            
    # Orthorhombic: a ≠ b ≠ c, alpha = beta = gamma = 90°
    elif (abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol and
          abs(a - b) > tol and (abs(b - c) > tol or abs(a - c) > tol)):
        return BravaisType.ORTHORHOMBIC
            
    # Monoclinic: a ≠ b ≠ c, alpha = gamma = 90°, beta ≠ 90°
    elif (abs(alpha - 90) < tol and abs(gamma - 90) < tol and abs(beta - 90) > tol):
        return BravaisType.MONOCLINIC
        
    # Trigonal/Rhombohedral: a = b = c, alpha = beta = gamma ≠ 90°
    elif (abs(a - b) < tol and abs(b - c) < tol and
          abs(alpha - beta) < tol and abs(beta - gamma) < tol and abs(alpha - 90) > tol):
        return BravaisType.TRIGONAL
        
    # Triclinic: a ≠ b ≠ c, alpha ≠ beta ≠ gamma ≠ 90°
    else:
        return BravaisType.TRICLINIC

# Calculator
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
calculator = mace_mp(
    model=mace_checkpoint_url,
    enable_cueq=False,
    device=device,
    default_dtype="float64",
)

# ASE structure
N = 2
struct = bulk("Cu", "fcc", a=3.58, cubic=True)
struct = struct.repeat((N, N, N))
struct.calc = calculator

# Relax cell
fcf = FrechetCellFilter(struct)
opt = FIRE_ASE(fcf)
opt.run(fmax=1e-5, steps=300)
struct = fcf.atoms

# Get bravais type
bravais_type = get_bravais_type(struct)

# Calculate elastic tensor
elastic_tensor = calculate_elastic_tensor(struct, device, dtype, bravais_type=bravais_type)

# Calculate elastic moduli
bulk_modulus, shear_modulus, poisson_ratio, pugh_ratio = calculate_elastic_moduli(elastic_tensor)

# Print elastic tensor
print(f"\nElastic tensor (GPa):")
elastic_tensor_np = elastic_tensor.cpu().numpy()
for row in elastic_tensor_np:
    print("  " + "  ".join(f"{val:10.4f}" for val in row))

# Print mechanical moduli
print(f"Bulk modulus (GPa): {bulk_modulus:.4f}")
print(f"Shear modulus (GPa): {shear_modulus:.4f}")
print(f"Poisson's ratio: {poisson_ratio:.4f}")
print(f"Pugh's ratio (K/G): {pugh_ratio:.4f}")