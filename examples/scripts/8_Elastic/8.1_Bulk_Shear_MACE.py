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
from torch_sim.elastic import (
    BravaisType,
    ElasticState,
    get_elastic_tensor,
    get_elementary_deformations,
    get_full_elastic_tensor,
)
import numpy as np

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

# Define elastic state
state = ElasticState(
    position=torch.tensor(struct.get_positions(), device=device, dtype=dtype),
    cell=torch.tensor(struct.get_cell().array, device=device, dtype=dtype),
)
bravais_type = BravaisType.TRICLINIC

# Calculate deformations for the bravais type
deformations = get_elementary_deformations(
    state, n_deform=6, max_strain_normal=0.01, max_strain_shear=0.06, bravais_type=bravais_type
)

# Calculate stresses for deformations
ref_pressure = -torch.mean(torch.tensor(struct.get_stress()[:3], device=device), dim=0)
stresses = torch.zeros((len(deformations), 6), device=device, dtype=torch.float64)
for i, deformation in enumerate(deformations):
    struct.cell = deformation.cell.cpu().numpy()
    struct.positions = deformation.position.cpu().numpy()
    stresses[i] = torch.tensor(struct.get_stress(), device=device)

# Caclulate elastic tensor
C_ij, B_ij = get_elastic_tensor(state, deformations, stresses, ref_pressure, bravais_type)
C = get_full_elastic_tensor(C_ij, bravais_type) / units.GPa
print("\nElastic Tensor (GPa):")
for row in C:
    print("  " + "  ".join(f"{val:10.4f}" for val in row))

# Components of the elastic tensor
C11, C22, C33 = C[0,0], C[1,1], C[2,2]
C12, C23, C31 = C[0,1], C[1,2], C[2,0]
C44, C55, C66 = C[3,3], C[4,4], C[5,5]

# Calculate compliance tensor
S = np.linalg.inv(C.cpu().numpy())
S11, S22, S33 = S[0,0], S[1,1], S[2,2]
S12, S23, S31 = S[0,1], S[1,2], S[2,0]
S44, S55, S66 = S[3,3], S[4,4], S[5,5]

# Voigt averaging (upper bound)
K_V = (1/9) * ((C11 + C22 + C33) + 2*(C12 + C23 + C31))
G_V = (1/15) * ((C11 + C22 + C33) - (C12 + C23 + C31) + 3*(C44 + C55 + C66))

# Reuss averaging (lower bound)
K_R = 1 / ((S11 + S22 + S33) + 2*(S12 + S23 + S31))
G_R = 15 / (4*(S11 + S22 + S33) - 4*(S12 + S23 + S31) + 3*(S44 + S55 + S66))

# Voigt-Reuss-Hill averaging
K_VRH = (K_V + K_R) / 2
G_VRH = (G_V + G_R) / 2

# Young's modulus for each averaging method
E_V = 9 * K_V * G_V / (3 * K_V + G_V)
E_R = 9 * K_R * G_R / (3 * K_R + G_R)
E_VRH = 9 * K_VRH * G_VRH / (3 * K_VRH + G_VRH)

# Poisson's ratio for each averaging method
v_V = (3 * K_V - 2 * G_V) / (6 * K_V + 2 * G_V)
v_R = (3 * K_R - 2 * G_R) / (6 * K_R + 2 * G_R)
v_VRH = (3 * K_VRH - 2 * G_VRH) / (6 * K_VRH + 2 * G_VRH)

# Pugh's ratio for each averaging method
pugh_ratio_V = K_V / G_V
pugh_ratio_R = K_R / G_R
pugh_ratio_VRH = K_VRH / G_VRH

# Cauchy pressure (C12-C44)
cauchy_pressure = C12 - C44

# Print mechanical moduli
print(f"\nBulk modulus (GPa):")
print(f"- Voigt: {K_V:.4f}")
print(f"- Reuss: {K_R:.4f}")
print(f"- VRH: {K_VRH:.4f}")

print(f"\nShear modulus (GPa):")
print(f"- Voigt: {G_V:.4f}")
print(f"- Reuss: {G_R:.4f}")
print(f"- VRH: {G_VRH:.4f}")

print(f"\nYoung's modulus:")
print(f"- Voigt: {E_V:.4f}")
print(f"- Reuss: {E_R:.4f}")
print(f"- VRH: {E_VRH:.4f}")

print(f"\nPoisson's ratio:")
print(f"- Voigt: {v_V:.4f}")
print(f"- Reuss: {v_R:.4f}")
print(f"- VRH: {v_VRH:.4f}")

print(f"\nPugh's ratio (K/G):")
print(f"- Voigt: {pugh_ratio_V:.4f}")
print(f"- Reuss: {pugh_ratio_R:.4f}")
print(f"- VRH: {pugh_ratio_VRH:.4f}")

print(f"\nCauchy pressure (C12-C44): {cauchy_pressure:.4f}")