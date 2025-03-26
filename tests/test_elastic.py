
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
    n_deform: int = 5,
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


def test_cubic():

    """Test the elastic tensor of a cubic structure of Cu (mp-30)"""

    # ASE structure
    N = 2
    struct = bulk("Cu", "fcc", a=3.58, cubic=True)
    struct = struct.repeat((N, N, N))

    # Relax cell
    struct.calc = calculator
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Calculate elastic tensor
    C_cubic = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.CUBIC)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    print("\nCubic")
    for row in C_cubic:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    print("Triclinic")
    for row in C_triclinic:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    # Check if the elastic tensors are equal
    assert torch.allclose(C_cubic, C_triclinic, atol=1e-1)


def test_hexagonal():

    """Test the elastic tensor of a hexagonal structure of Mg (mp-153)"""

    # ASE structure
    N = 2
    struct = bulk("Mg", crystalstructure="hcp", a=3.17, c=5.14)
    struct = struct.repeat((N, N, N))

    # Relax cell
    struct.calc = calculator
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Calculate elastic tensor
    C_hexagonal = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.HEXAGONAL)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    print("\nHexagonal")
    for row in C_hexagonal:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    print("Triclinic")
    for row in C_triclinic:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    # Check if the elastic tensors are equal
    assert torch.allclose(C_hexagonal, C_triclinic, atol=1e-1)


def test_trigonal():
    """Test the elastic tensor of a trigonal structure of As (mp-11)"""

    # ASE structure
    N = 2
    struct = bulk("As", crystalstructure="rhombohedral", a=3.75, c=11.11)
    struct = struct.repeat((N, N, N))

    # Relax cell
    struct.calc = calculator
    fcf = FrechetCellFilter(struct)
    opt = FIRE_ASE(fcf)
    opt.run(fmax=1e-5, steps=300)
    struct = fcf.atoms
    struct_copy = copy.deepcopy(struct)

    # Calculate elastic tensor
    C_trigonal = calculate_elastic_tensor(struct, device, dtype, bravais_type=BravaisType.TRIGONAL)
    C_triclinic = calculate_elastic_tensor(struct_copy, device, dtype, bravais_type=BravaisType.TRICLINIC)

    print("\nTrigonal")
    for row in C_trigonal:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    print("Triclinic")
    for row in C_triclinic:
        print("  " + "  ".join(f"{val:10.4f}" for val in row))

    # Check if the elastic tensors are equal
    assert torch.allclose(C_trigonal, C_triclinic, atol=1e-1)


if __name__ == "__main__":
    
    #test_cubic() 
    #test_hexagonal() 
    test_trigonal() # This fails

    #test_tetragonal() # TODO
    #test_orthorhombic() # TODO
    #test_monoclinic() # TODO
