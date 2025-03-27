
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
import copy
from pymatgen.io.ase import AseAtomsAdaptor
from matcalc.elasticity import ElasticityCalc
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

def calculate_elastic_tensor(
    struct: Atoms, 
    device: torch.device, 
    dtype: torch.dtype,
    bravais_type: BravaisType = BravaisType.TRICLINIC,
    max_strain_normal: float = 0.01,
    max_strain_shear: float = 0.06,
    n_deform: int = 7,
) -> torch.Tensor:

    """Calculate the elastic tensor of a structure."""

    # Define elastic state
    state = ElasticState(
        position=torch.tensor(struct.get_positions(), device=device, dtype=dtype),
        cell=torch.tensor(struct.get_cell().array, device=device, dtype=dtype),
    )

    # Calculate deformations for the bravais type
    deformations = get_elementary_deformations(
        state, 
        n_deform=n_deform, 
        max_strain_normal=max_strain_normal, 
        max_strain_shear=max_strain_shear, 
        bravais_type=bravais_type
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
    
    return C

def calculate_elastic_tensor_matcalc(
        struct: Atoms, 
        calculator: Any,
        norm_strains: tuple = (-0.01, -0.005, 0.005, 0.01),
        shear_strains: tuple = (-0.06, -0.03, 0.03, 0.06)
    ):
    """Calculate the elastic tensor of a structure using matcalc."""
    
    pmg_struct = AseAtomsAdaptor.get_structure(struct)

    # Use elasticity calcualtor
    elastic_calc = ElasticityCalc(
        calculator,
        norm_strains=norm_strains,
        shear_strains=shear_strains,
        fmax=1e-5,
        relax_structure=False,
        relax_deformed_structures=False
    )
    
    # Elastic tensor
    elastic_tensor = ElasticTensor(elastic_calc.calc(pmg_struct)["elastic_tensor"]).voigt/units.GPa
  
    return elastic_tensor


def print_structure_info(struct: Atoms):
    """Print the information of a structure."""
    pmg_struct = AseAtomsAdaptor.get_structure(struct)
    print("Relaxed structure:")
    print(f"- Lattice parameters: a={pmg_struct.lattice.a:.4f}, b={pmg_struct.lattice.b:.4f}, c={pmg_struct.lattice.c:.4f}")
    print(f"- Angles: alpha={pmg_struct.lattice.alpha:.2f}, beta={pmg_struct.lattice.beta:.2f}, gamma={pmg_struct.lattice.gamma:.2f}")

def get_spacegroup_number(struct: Atoms):
    """Get the spacegroup number of a structure."""
    cell = (struct.get_cell(), struct.get_scaled_positions(), struct.get_atomic_numbers())
    spg_data = spglib.get_symmetry_dataset(cell, symprec=1e-5)
    return spg_data.number

# --------------
# TEST FUNCTIONS
# --------------

def test_cubic(verbose: bool = False):

    """Test the elastic tensor of a cubic structure of Cu (mp-30)"""

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
    struct_copy = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)
    
    # Verify the space group is cubic for the relaxed structure
    spg_number = get_spacegroup_number(struct)
    assert 221 <= spg_number <= 230, f"Structure is not cubic (space group {spg_number})"

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


def test_hexagonal(verbose: bool = False):

    """Test the elastic tensor of a hexagonal structure of Mg (mp-153)"""

    # ASE structure
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
    spg_number = get_spacegroup_number(struct)
    assert 194 <= spg_number <= 199, f"Structure is not hexagonal (space group {spg_number})"

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


def test_trigonal(verbose: bool = False):
    """Test the elastic tensor of a trigonal structure of As (mp-11)"""

    # ASE structure
    N = 2
    struct = bulk("As", crystalstructure="rhombohedral", a=3.75, c=11.11)
    struct = struct.repeat((N, N, N))
    struct.calc = calculator

    # Relax cell
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)
    struct_matcalc = copy.deepcopy(struct)
    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is trigonal for the relaxed structure
    spg_number = get_spacegroup_number(struct)
    assert 143 <= spg_number <= 167, f"Structure is not trigonal (space group {spg_number})"

    # Calculate elastic tensor
    C_trigonal = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.TRIGONAL, n_deform=9, max_strain_normal=0.01, max_strain_shear=0.03)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC, n_deform=9, max_strain_normal=0.01, max_strain_shear=0.03)

    if verbose:
        print("\nTrigonal")
        for row in C_trigonal:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        print("Triclinic")
        for row in C_triclinic:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))

        # Calculate elastic tensor using matcalc
        print("Matcalc")
        C_matcalc = calculate_elastic_tensor_matcalc(struct_matcalc, calculator)
        for row in C_matcalc:
            print("  " + "  ".join(f"{val:10.4f}" for val in row))
        
    # Check if the elastic tensors are equal
    assert torch.allclose(C_trigonal, C_triclinic, atol=1e-1)

def test_tetragonal(verbose: bool = False):
    """Test the elastic tensor of a tetragonal structure of BaTiO3"""

    # Create tetragonal BaTiO3 structure
    a, c = 3.99, 4.03
    symbols = ["Ba", "Ti", "O", "O", "O"]
    basis = [
        (0, 0, 0),        # Ba at (0,0,0)
        (0.5, 0.5, 0.48), # Ti displaced slightly along c-axis
        (0.5, 0.5, 0),    # O1 in basal plane
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
    struct_matcalc = copy.deepcopy(struct)

    # Relaxed structure
    if verbose:
        print_structure_info(struct)

    # Verify the space group is tetragonal for the relaxed structure
    spg_number = get_spacegroup_number(struct)
    assert 75 <= spg_number <= 142, f"Structure is not tetragonal (space group {spg_number})"

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


def test_orthorhombic(verbose: bool = False):
    """Test the elastic tensor of a orthorhombic structure of BaTiO3"""

    # Create orthorhombic BaTiO3 structure
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
    spg_number = get_spacegroup_number(struct)
    assert 16 <= spg_number <= 74, f"Structure is not orthorhombic (space group {spg_number})"

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

if __name__ == "__main__":
    
    #test_cubic()  # this works
    #test_hexagonal()  # This works
    #test_trigonal(verbose=True) # This fails

    #test_tetragonal()
    test_orthorhombic()
    #test_monoclinic() # TODO


    # Take the structure from MP
    #from pymatgen.ext.matproj import MPRester
    #from pymatgen.io.ase import AseAtomsAdaptor
    #API_KEY = "z7DfJ90MRrH8XMPegrsN7vR1ts1aXNN1"
    #with MPRester(API_KEY) as mpr:
    #    struct_pmg = mpr.get_structure_by_material_id("mp-11")
    #struct = AseAtomsAdaptor.get_atoms(struct_pmg)
    #struct.calc = calculator