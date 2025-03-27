
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
import copy
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from typing import Any
import spglib
from ase.spacegroup import crystal

# Set global seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Calculator
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
mace_checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
calculator = mace_mp(
    model="medium",
    enable_cueq=False,
    device=device,
    default_dtype="float64",
)

def print_structure_info(struct: Atoms):
    """Print the information of a structure."""
    pmg_struct = AseAtomsAdaptor.get_structure(struct)
    print("\nRelaxed structure:")
    print(f"- Lattice parameters: a={pmg_struct.lattice.a:.4f}, b={pmg_struct.lattice.b:.4f}, c={pmg_struct.lattice.c:.4f}")
    print(f"- Angles: alpha={pmg_struct.lattice.alpha:.2f}, beta={pmg_struct.lattice.beta:.2f}, gamma={pmg_struct.lattice.gamma:.2f}")

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
    
# --------------
# TEST FUNCTIONS
# --------------

def test_cubic(calculator: mace_mp, verbose: bool = False):
    """Test the elastic tensor of a cubic structure of Cu"""

    if verbose:
        print("\n### Testing cubic symmetry... ###")

    # cubic Cu
    N = 2
    struct = bulk("Cu", "fcc", a=3.58, cubic=True)
    struct = struct.repeat((N, N, N))
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)
    
    # Verify the space group is cubic for the relaxed structure
    assert get_bravais_type(struct) == BravaisType.CUBIC, f"Structure is not cubic"

    # Calculate elastic tensor
    C_cubic = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.CUBIC)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    if verbose:
        print("\nCubic")
        for row in C_cubic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
        print("\nTriclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

    # Check if the elastic tensors are equal
    assert torch.allclose(C_cubic, C_triclinic, atol=1e-1)


def test_hexagonal(calculator: mace_mp, verbose: bool = False):
    """Test the elastic tensor of a hexagonal structure of Mg"""

    if verbose:
        print("\n### Testing hexagonal symmetry... ###")

    # hexagonal Mg
    N = 2
    struct = bulk("Mg", crystalstructure="hcp", a=3.17, c=5.14)
    struct = struct.repeat((N, N, N))
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is hexagonal for the relaxed structure
    assert get_bravais_type(struct) == BravaisType.HEXAGONAL, f"Structure is not hexagonal"

    # Calculate elastic tensor
    C_hexagonal = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.HEXAGONAL)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    if verbose:
        print("\nHexagonal")
        for row in C_hexagonal:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        print("Triclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

    # Check if the elastic tensors are equal
    assert torch.allclose(C_hexagonal, C_triclinic, atol=1e-1)


def test_trigonal(calculator: mace_mp, verbose: bool = False):
    """Test the elastic tensor of a trigonal structure of Si"""

    if verbose:
        print("\n### Testing trigonal symmetry... ###")

    # Rhombohedral Si
    a = 5.431
    alpha = 60.0
    cos_alpha = np.cos(np.radians(alpha))
    tx = np.sqrt((1 - cos_alpha)/2)
    ty = np.sqrt((1 - cos_alpha)/6)
    tz = np.sqrt((1 + 2*cos_alpha)/3)
    cell = [
        [a*tx, -a*ty, a*tz],
        [0, 2*a*ty, a*tz],
        [-a*tx, -a*ty, a*tz]
    ]
    positions = [
        (0.0000, 0.0000, 0.0000),  # Si1
        (0.2500, 0.2500, 0.2500),  # Si2
    ]
    symbols = ["Si", "Si"]
    struct = Atoms(symbols, scaled_positions=positions, cell=cell, pbc=True)
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-4, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)
    
    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is trigonal for the relaxed structure
    assert get_bravais_type(struct) == BravaisType.TRIGONAL, f"Structure is not trigonal"

    # Calculate elastic tensor
    C_trigonal = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.TRIGONAL)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    if verbose:
        print("\nTrigonal")
        for row in C_trigonal:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        print("Triclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
        
    # Check if the elastic tensors are equal
    assert torch.allclose(C_trigonal, C_triclinic, atol=2e-1)

def test_tetragonal(calculator: mace_mp, verbose: bool = False):
    """Test the elastic tensor of a tetragonal structure of BaTiO3"""

    if verbose:
        print("\n### Testing tetragonal symmetry... ###")

    # tetragonal BaTiO3
    a, c = 3.99, 4.03
    symbols = ["Ba", "Ti", "O", "O", "O"]
    basis = [
        (0, 0, 0),        # Ba
        (0.5, 0.5, 0.48), # Ti
        (0.5, 0.5, 0),    # O1
        (0.5, 0, 0.52),   # O2
        (0, 0.5, 0.52),   # O3
    ]
    
    struct = crystal(
        symbols,
        basis=basis,
        spacegroup=99,  # P4mm
        cellpar=[a, a, c, 90, 90, 90],
    )
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is tetragonal for the relaxed structure
    assert get_bravais_type(struct) == BravaisType.TETRAGONAL, f"Structure is not tetragonal"

    # Calculate elastic tensor
    C_tetragonal = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.TETRAGONAL)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    if verbose:
        print("\nTetragonal")
        for row in C_tetragonal:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        print("\nTriclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
    
    # Check if the elastic tensors are equal
    assert torch.allclose(C_tetragonal, C_triclinic, atol=1e-1)


def test_orthorhombic(calculator: mace_mp, verbose: bool = False):
    """Test the elastic tensor of a orthorhombic structure of BaTiO3"""

    if verbose:
        print("\n### Testing orthorhombic symmetry... ###")

    # orthorhombic BaTiO3
    a, b, c = 3.8323, 2.8172, 5.8771
    scaled_positions = [
        (0.0000, 0.0000, 0.0000),  # Ba
        (0.5000, 0.0000, 0.5000),  # Ti
        (0.5000, 0.5000, 0.2450),  # O1
        (0.5000, 0.5000, 0.7550),  # O2
        (0.0000, 0.5000, 0.5000),  # O3
    ]
    symbols = ["Ba", "Ti", "O", "O", "O"]
    struct = Atoms(symbols, scaled_positions=scaled_positions, 
                  cell=[(a, 0, 0), (0, b, 0), (0, 0, c)], pbc=True)
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-4, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is orthorhombic for the relaxed structure
    assert get_bravais_type(struct) == BravaisType.ORTHORHOMBIC, f"Structure is not orthorhombic"

    # Calculate elastic tensor
    C_orthorhombic = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.ORTHORHOMBIC)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    if verbose:
        print("\nOrthorhombic")
        for row in C_orthorhombic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        print("\nTriclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
    
    # Check if the elastic tensors are equal
    assert torch.allclose(C_orthorhombic, C_triclinic, atol=1e-1)

def test_monoclinic(calculator: mace_mp, verbose: bool = False):
    """Test the elastic tensor of a monoclinic structure of β-Ga2O3"""

    if verbose:
        print("\n### Testing monoclinic symmetry... ###")
   
    # monoclinic β-Ga2O3
    a, b, c, beta = 12.214, 3.037, 5.798, 103.7
    beta_rad = np.radians(beta)
    cell = [
        [a, 0, 0],
        [0, b, 0],
        [c * np.cos(beta_rad), 0, c * np.sin(beta_rad)],
    ]
    positions = [
        (0.0903, 0.0000, 0.7947),  # Ga1
        (0.9097, 0.0000, 0.2053),  # Ga2
        (0.1660, 0.5000, 0.4742),  # Ga3
        (0.8340, 0.5000, 0.5258),  # Ga4
        (0.3062, 0.0000, 0.3235),  # O1
        (0.6938, 0.0000, 0.6765),  # O2
        (0.4824, 0.5000, 0.8177),  # O3
        (0.5176, 0.5000, 0.1823),  # O4
        (0.3138, 0.5000, 0.0346),  # O5
        (0.6862, 0.5000, 0.9654),  # O6
    ]
    symbols = ["Ga", "Ga", "Ga", "Ga", "O", "O", "O", "O", "O", "O"]
    struct = Atoms(symbols, scaled_positions=positions, cell=cell, pbc=True)
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=5e-3, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is monoclinic for the relaxed structure
    assert get_bravais_type(struct) == BravaisType.MONOCLINIC, f"Structure is not monoclinic"

    # Calculate elastic tensor
    C_monoclinic = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.MONOCLINIC)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    if verbose:
        print("\nMonoclinic")
        for row in C_monoclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        print("\nTriclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
    
    # Check if the elastic tensors are equal
    assert torch.allclose(C_monoclinic, C_triclinic, atol=1e-1)


def test_copper_elastic_properties(calculator: mace_mp, verbose: bool = False):
    """Test calculation of elastic properties for copper."""
    
    # Create copper structure
    N = 2
    struct = bulk("Cu", "fcc", a=3.58, cubic=True)
    struct = struct.repeat((N, N, N))
    struct.calc = calculator
    
    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    
    # Calculate elastic tensor
    bravais_type = get_bravais_type(struct)
    elastic_tensor = calculate_elastic_tensor(struct, device, dtype, bravais_type=bravais_type)
    
    # Calculate elastic moduli
    bulk_modulus, shear_modulus, _, _ = calculate_elastic_moduli(elastic_tensor)
    
    # Expected values
    expected_elastic_tensor = torch.tensor([
        [171.3434, 130.5782, 130.5782, 0.0000, 0.0000, 0.0000],
        [130.5782, 171.3434, 130.5782, 0.0000, 0.0000, 0.0000],
        [130.5782, 130.5782, 171.3434, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 70.8565, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 70.8565, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 70.8565]
    ], device=device, dtype=dtype)
    
    expected_bulk_modulus = 144.1666
    expected_shear_modulus = 43.1319 
    
    # Print results for verification
    if verbose:
        print(f"\nElastic tensor (GPa):")
        elastic_tensor_np = elastic_tensor.cpu().numpy()
        for row in elastic_tensor_np:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
        
        print(f"Bulk modulus (GPa): {bulk_modulus:.4f}")
        print(f"Shear modulus (GPa): {shear_modulus:.4f}")

    # Assert with tolerance
    assert torch.allclose(elastic_tensor, expected_elastic_tensor, rtol=1e-4)
    assert abs(bulk_modulus - expected_bulk_modulus) < 1e-4 * expected_bulk_modulus
    assert abs(shear_modulus - expected_shear_modulus) < 1e-4 * expected_shear_modulus
    

if __name__ == "__main__":
    
    # test symmetries
    test_cubic(calculator)
    test_hexagonal(calculator)
    test_trigonal(calculator)
    test_tetragonal(calculator)
    test_orthorhombic(calculator)
    test_monoclinic(calculator)

    # test elastic properties of Cu
    test_copper_elastic_properties(calculator)
