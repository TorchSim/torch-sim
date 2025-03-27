# ruff: noqa: RUF002, RUF003, PLC2401
"""Primary Sources and References for Crystal Elasticity.

- Nye, J.F. (1985) "Physical Properties of Crystals: Their Representation by Tensors and
  Matrices"
  - Definitive text on crystal elasticity
  - Contains detailed derivations of elastic constant matrices for all crystal systems
  - Pages 131-149 cover elastic compliance and stiffness matrices

- Landau, L.D. & Lifshitz, E.M. "Theory of Elasticity" (Volume 7 of Course of
  Theoretical Physics)
  - Fundamental treatment of elasticity theory
  - Section 10 covers crystal elasticity
  - Pages 134-142 specifically discuss elastic symmetry classes

- Teodosiu, C. (1982) "Elastic Models of Crystal Defects"
  - Chapter 2: "Elastic Properties of Crystals"
  - Detailed treatment of symmetry constraints on elastic constants

Review Articles:
- Fast, L., Wills, J. M., Johansson, B., & Eriksson, O. (1995).
  "Elastic constants of hexagonal transition metals: Theory"
  Physical Review B, 51(24), 17431
  - Specific treatment of hexagonal systems
  - Verification of C66 = (C11-C12)/2 relationship

- Mouhat, F., & Coudert, F. X. (2014).
  "Necessary and sufficient elastic stability conditions in various crystal systems"
  Physical Review B, 90(22), 224104
  - Modern treatment of elastic stability conditions
  - Tables of independent elastic constants for each system

Online Resources:
- Materials Project Documentation
  https://docs.materialsproject.org/methodology/elasticity/
  - Modern computational implementation details
  - Verification of symmetry relationships

- USPEX Wiki on Elastic Constants
  http://uspex-team.org/online_utilities/elastic_constants
  - Practical guide to elastic constant calculations
  - Symmetry relationships and constraints
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch


@dataclass
class ElasticState:
    """Represents the elastic state of a crystal structure.

    This class encapsulates the atomic positions and unit cell parameters
    that define the elastic state of a crystalline material.

    Attributes:
        position: Tensor containing atomic positions in Cartesian coordinates.
                 Shape: (n_atoms, 3) where each row represents [x, y, z] coordinates.
        cell: Tensor containing the unit cell matrix.
              Shape: (3, 3) where rows are the lattice vectors a, b, c.
    """

    position: torch.Tensor  # Shape: (n_atoms, 3)
    cell: torch.Tensor  # Shape: (3, 3)


@dataclass
class DeformationRule:
    """Defines rules for applying deformations based on crystal symmetry.

    This class specifies which axes to deform and how to handle symmetry
    constraints when calculating elastic properties.

    Attributes:
        axes: List of indices indicating which strain components to consider
              for the specific crystal symmetry, following Voigt notation:
              [0=xx, 1=yy, 2=zz, 3=yz, 4=xz, 5=xy]
        symmetry_handler: Callable function that constructs the stress-strain
                         relationship matrix according to the crystal symmetry.
    """

    axes: list[int]
    symmetry_handler: Callable


class BravaisType(Enum):
    """Enumeration of the seven Bravais lattice types in 3D crystals.

    These lattice types represent the distinct crystal systems classified
    by their symmetry properties, from highest symmetry (cubic) to lowest
    symmetry (triclinic).

    Each type has specific constraints on lattice parameters and angles,
    which determine the number of independent elastic constants.
    """

    CUBIC = "cubic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"


def regular_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for cubic (regular) crystal symmetry.

    Constructs the stress-strain relationship matrix for cubic symmetry,
    which has three independent elastic constants: C11, C12, and C44.

    The matrix relates strains to stresses according to the equation:
    σᵢ = Σⱼ Cᵢⱼ εⱼ

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]
                where:
                - εxx, εyy, εzz are normal strains
                - εyz, εxz, εxy are shear strains

    Returns:
        torch.Tensor: Matrix of shape (6, 3) where columns correspond to
                     coefficients for C11, C12, and C44 respectively

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    (εyy + εzz)    0      ⎤
        ⎢ εyy    (εxx + εzz)    0      ⎥
        ⎢ εzz    (εxx + εyy)    0      ⎥
        ⎢ 0      0              2εyz   ⎥
        ⎢ 0      0              2εxz   ⎥
        ⎣ 0      0              2εxy   ⎦

        This represents the relationship:
        σxx = C11*εxx + C12*(εyy + εzz)
        σyy = C11*εyy + C12*(εxx + εzz)
        σzz = C11*εzz + C12*(εxx + εyy)
        σyz = 2*C44*εyz
        σxz = 2*C44*εxz
        σxy = 2*C44*εxy
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 3), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements
    # First column (C11 coefficients)
    matrix[0, 0] = εxx
    matrix[1, 0] = εyy
    matrix[2, 0] = εzz

    # Second column (C12 coefficients)
    matrix[0, 1] = εyy + εzz
    matrix[1, 1] = εxx + εzz
    matrix[2, 1] = εxx + εyy

    # Third column (C44 coefficients)
    matrix[3, 2] = 2 * εyz
    matrix[4, 2] = 2 * εxz
    matrix[5, 2] = 2 * εxy

    return matrix


def tetragonal_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for tetragonal crystal symmetry.

    Constructs the stress-strain relationship matrix for tetragonal symmetry,
    which has 7 independent elastic constants: C11, C12, C13, C16, C33, C44, C66.

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]
                where:
                - εxx, εyy, εzz are normal strains
                - εyz, εxz, εxy are shear strains

    Returns:
        torch.Tensor: Matrix of shape (6, 7) where columns correspond to
                     coefficients for C11, C12, C13, C16, C33, C44, C66

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz     2εxy    0      0      0    ⎤
        ⎢ εyy    εxx    εzz    -2εxy    0      0      0    ⎥
        ⎢ 0      0      εxx+εyy 0       εzz    0      0    ⎥
        ⎢ 0      0      0       0       0      2εyz   0    ⎥
        ⎢ 0      0      0       0       0      2εxz   0    ⎥
        ⎣ 0      0      0       εxx-εyy 0      0      2εxy ⎦
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 7), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements according to the image
    # First row - C11, C12, C13, C16, C33, C44
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εxy

    # Second row
    matrix[1, 0] = εyy
    matrix[1, 1] = εxx
    matrix[1, 2] = εzz
    matrix[1, 3] = -2 * εxy

    # Third row
    matrix[2, 2] = εxx + εyy
    matrix[2, 4] = εzz

    # Fourth and fifth rows (shear terms)
    matrix[3, 5] = 2 * εyz
    matrix[4, 5] = 2 * εxz

    # Sixth row
    matrix[5, 3] = εxx - εyy
    matrix[5, 6] = 2 * εxy

    return matrix

def orthorhombic_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for orthorhombic crystal symmetry.

    Constructs the stress-strain relationship matrix for orthorhombic symmetry,
    which has nine independent elastic constants: C11, C12, C13, C22, C23, C33,
    C44, C55, and C66.

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        torch.Tensor: Matrix of shape (6, 9) where columns correspond to
                     coefficients for C11, C12, C13, C22, C23, C33, C44, C55, C66

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz    0      0      0      0      0      0  ⎤
        ⎢ 0      εxx    0      εyy    εzz    0      0      0      0  ⎥
        ⎢ 0      0      εxx    0      εyy    εzz    0      0      0  ⎥
        ⎢ 0      0      0      0      0      0      2εyz   0      0  ⎥
        ⎢ 0      0      0      0      0      0      0      2εxz   0  ⎥
        ⎣ 0      0      0      0      0      0      0      0      2εxy⎦
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 9), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements according to the image
    # First row - C11, C12, C13, C22, C23, C33, C44, C55, C66
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz

    # Second row
    matrix[1, 1] = εxx
    matrix[1, 3] = εyy
    matrix[1, 4] = εzz

    # Third row
    matrix[2, 2] = εxx
    matrix[2, 4] = εyy
    matrix[2, 5] = εzz

    # Fourth row (shear terms)
    matrix[3, 6] = 2 * εyz

    # Fifth row
    matrix[4, 7] = 2 * εxz

    # Sixth row
    matrix[5, 8] = 2 * εxy

    return matrix

def trigonal_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for trigonal crystal symmetry.

    Constructs the stress-strain relationship matrix for trigonal symmetry,
    which has 7 independent elastic constants: C11, C12, C13, C14, C15, C33, C44.
    Matrix construction follows the standard form for trigonal symmetry.

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        torch.Tensor: Matrix of shape (6, 7) where columns correspond to
                     coefficients for C11, C12, C13, C14, C15, C33, C44

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz       2εyz        2εxz      0      0    ⎤
        ⎢ εyy    εxx    εzz      -2εyz       -2εxz      0      0    ⎥
        ⎢ 0      0      εxx+εyy   0           0         εzz    0    ⎥
        ⎢ 0      0      0         εxx-εyy    -2εxy      0      2εyz ⎥
        ⎢ 0      0      0         2εxy        εxx-εyy   0      2εxz ⎥
        ⎣ εxy   -εxy    0         2εxz       -2εyz      0      0    ⎦
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 7), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements according to the correct matrix
    # First row - C11, C12, C13, C14, C15, C33, C44
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εyz
    matrix[0, 4] = 2 * εxz

    # Second row
    matrix[1, 0] = εyy
    matrix[1, 1] = εxx
    matrix[1, 2] = εzz
    matrix[1, 3] = -2 * εyz
    matrix[1, 4] = -2 * εxz

    # Third row
    matrix[2, 2] = εxx + εyy
    matrix[2, 5] = εzz

    # Fourth row
    matrix[3, 3] = εxx - εyy
    matrix[3, 4] = -2 * εxy
    matrix[3, 6] = 2 * εyz

    # Fifth row
    matrix[4, 3] = 2 * εxy
    matrix[4, 4] = εxx - εyy
    matrix[4, 6] = 2 * εxz

    # Sixth row
    matrix[5, 0] = εxy
    matrix[5, 1] = -εxy
    matrix[5, 3] = 2 * εxz
    matrix[5, 4] = -2 * εyz

    return matrix

def hexagonal_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for hexagonal crystal symmetry.

    Constructs the stress-strain relationship matrix for hexagonal symmetry,
    which has 5 independent elastic constants: C11, C33, C12, C13, C44.
    Note: C66 = (C11-C12)/2 is dependent.

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        torch.Tensor: Matrix of shape (6, 5) where columns correspond to
                     coefficients for C11, C33, C12, C13, C44

    Notes:
        The resulting matrix M has the form:
        ⎡ εxx    εyy    εzz      0     0   ⎤
        ⎢ εyy    εxx    εzz      0     0   ⎥
        ⎢ 0      0      εxx+εyy  εzz   0   ⎥
        ⎢ 0      0      0        0     2εyz⎥
        ⎢ 0      0      0        0     2εxz⎥
        ⎣ εxy   -εxy    0        0     0   ⎦
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 5), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements according to the correct structure
    # First row
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz

    # Second row
    matrix[1, 0] = εyy
    matrix[1, 1] = εxx
    matrix[1, 2] = εzz

    # Third row
    matrix[2, 2] = εxx + εyy
    matrix[2, 3] = εzz

    # Fourth and fifth rows (shear terms)
    matrix[3, 4] = 2 * εyz
    matrix[4, 4] = 2 * εxz

    # Sixth row
    matrix[5, 0] = εxy
    matrix[5, 1] = -εxy

    return matrix

def monoclinic_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for monoclinic crystal symmetry.

    Constructs the stress-strain relationship matrix for monoclinic symmetry,
    which has 13 independent elastic constants: C11, C12, C13, C15, C22, C23, C25, 
    C33, C35, C44, C46, C55, C66.

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        torch.Tensor: Matrix of shape (6, 13) where columns correspond to
                     coefficients for the 13 independent constants in order:
                     [C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66]

    Notes:
        For monoclinic symmetry with unique axis b (y), the matrix has the form:
        ⎡ εxx    εyy    εzz    2εxz    0      0      0      0      0      0      0      0      0  ⎤
        ⎢ 0      εxx    0      0       εyy    εzz    2εxz   0      0      0      0      0      0  ⎥
        ⎢ 0      0      εxx    0       0      εyy    0      εzz    2εxz   0      0      0      0  ⎥
        ⎢ 0      0      0      0       0      0      0      0      0      2εyz   2εxy   0      0  ⎥
        ⎢ 0      0      0      εxx     0      0      εyy    0      εzz    0      0      2εxz   0  ⎥
        ⎣ 0      0      0      0       0      0      0      0      0      0      2εyz   0      2εxy⎦
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 13), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements according to the image
    # Order: C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66

    # First row - σxx
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εxz

    # Second row - σyy
    matrix[1, 1] = εxx
    matrix[1, 4] = εyy
    matrix[1, 5] = εzz
    matrix[1, 6] = 2 * εxz

    # Third row - σzz
    matrix[2, 2] = εxx
    matrix[2, 5] = εyy
    matrix[2, 7] = εzz
    matrix[2, 8] = 2 * εxz

    # Fourth row - σyz
    matrix[3, 9] = 2 * εyz
    matrix[3, 10] = 2 * εxy

    # Fifth row - σxz
    matrix[4, 3] = εxx
    matrix[4, 6] = εyy
    matrix[4, 8] = εzz
    matrix[4, 11] = 2 * εxz

    # Sixth row - σxy
    matrix[5, 10] = 2 * εyz
    matrix[5, 12] = 2 * εxy

    return matrix

def triclinic_symmetry(strains: torch.Tensor) -> torch.Tensor:
    """Generate equation matrix for triclinic crystal symmetry.

    Constructs the stress-strain relationship matrix for triclinic symmetry,
    which has 21 independent elastic constants (the most general case).

    Args:
        strains: Tensor of shape (6,) containing strain components
                [εxx, εyy, εzz, εyz, εxz, εxy]

    Returns:
        torch.Tensor: Matrix of shape (6, 21) where columns correspond to
                     all possible elastic constants in order:
                     [C11, C12, C13, C14, C15, C16,
                          C22, C23, C24, C25, C26,
                              C33, C34, C35, C36,
                                  C44, C45, C46,
                                      C55, C56,
                                          C66]
    """
    if not isinstance(strains, torch.Tensor):
        strains = torch.tensor(strains)

    if strains.shape != (6,):
        raise ValueError("Strains tensor must have shape (6,)")

    # Unpack strain components
    εxx, εyy, εzz, εyz, εxz, εxy = strains.unbind()

    # Create the matrix using torch.zeros for proper device/dtype handling
    matrix = torch.zeros((6, 21), dtype=strains.dtype, device=strains.device)

    # Fill in the matrix elements according to the image
    # First row - σxx
    matrix[0, 0] = εxx
    matrix[0, 1] = εyy
    matrix[0, 2] = εzz
    matrix[0, 3] = 2 * εyz
    matrix[0, 4] = 2 * εxz
    matrix[0, 5] = 2 * εxy

    # Second row - σyy
    matrix[1, 1] = εxx
    matrix[1, 6] = εyy
    matrix[1, 7] = εzz
    matrix[1, 8] = 2 * εyz
    matrix[1, 9] = 2 * εxz
    matrix[1, 10] = 2 * εxy

    # Third row - σzz
    matrix[2, 2] = εxx
    matrix[2, 7] = εyy
    matrix[2, 11] = εzz
    matrix[2, 12] = 2 * εyz
    matrix[2, 13] = 2 * εxz
    matrix[2, 14] = 2 * εxy

    # Fourth row - σyz
    matrix[3, 3] = εxx
    matrix[3, 8] = εyy
    matrix[3, 12] = εzz
    matrix[3, 15] = 2 * εyz
    matrix[3, 16] = 2 * εxz
    matrix[3, 17] = 2 * εxy

    # Fifth row - σxz
    matrix[4, 4] = εxx
    matrix[4, 9] = εyy
    matrix[4, 13] = εzz
    matrix[4, 16] = 2 * εyz
    matrix[4, 18] = 2 * εxz
    matrix[4, 19] = 2 * εxy

    # Sixth row - σxy
    matrix[5, 5] = εxx
    matrix[5, 10] = εyy
    matrix[5, 14] = εzz
    matrix[5, 17] = 2 * εyz
    matrix[5, 19] = 2 * εxz
    matrix[5, 20] = 2 * εxy

    return matrix


def get_cart_deformed_cell(
    base_state: ElasticState, axis: int = 0, size: float = 1.0
) -> ElasticState:
    """Deform a unit cell and scale atomic positions accordingly.

    Args:
        base_state: ElasticState containing positions, mass, and cell
        axis: Direction of deformation:
            - 0,1,2 for x,y,z cartesian deformations
            - 3,4,5 for yz,xz,xy shear deformations
        size: Deformation magnitude

    Returns:
        ElasticState: New state with deformed cell and scaled positions

    Raises:
        ValueError: If axis is not in range [0-5]
        ValueError: If cell is not a 3x3 tensor
        ValueError: If positions is not a (n_atoms, 3) tensor
    """
    if not (0 <= axis <= 5):
        raise ValueError("Axis must be between 0 and 5")
    if base_state.cell.shape != (3, 3):
        raise ValueError("Cell must be a 3x3 tensor")
    if base_state.position.shape[-1] != 3:
        raise ValueError("Positions must have shape (n_atoms, 3)")

    # Create identity matrix for transformation
    L = torch.eye(3, dtype=base_state.cell.dtype, device=base_state.cell.device)

    # Apply deformation based on axis
    if axis < 3:
        L[axis, axis] += size
    elif axis == 3:
        L[1, 2] += size  # yz shear
    elif axis == 4:
        L[0, 2] += size  # xz shear
    else:  # axis == 5
        L[0, 1] += size  # xy shear

    # Convert positions to fractional coordinates
    old_inv = torch.linalg.inv(base_state.cell)
    frac_coords = torch.matmul(base_state.position, old_inv)

    # Apply transformation to cell and convert positions back to cartesian
    new_cell = torch.matmul(base_state.cell, L)
    new_positions = torch.matmul(frac_coords, new_cell)

    return ElasticState(position=new_positions, cell=new_cell)


def get_elementary_deformations(
    base_state: ElasticState,
    n_deform: int = 5,
    max_strain_normal: float = 0.01,
    max_strain_shear: float = 0.06,
    bravais_type: BravaisType = None,
) -> list[ElasticState]:
    """Generate elementary deformations for elastic tensor calculation.

    Creates a series of deformed structures based on the crystal symmetry. The
    deformations are limited to non-equivalent axes of the crystal as determined by its
    Bravais lattice type.

    Args:
        base_state: ElasticState containing the base structure to be deformed
        n_deform: Number of deformations per non-equivalent axis
        max_strain_normal: Maximum deformation magnitude
        max_strain_shear: Maximum deformation magnitude
        bravais_type: BravaisType enum specifying the crystal system. If None,
                     defaults to lowest symmetry (triclinic)

    Returns:
        List[ElasticState]: List of deformed structures

    Notes:
        - For normal strains (axes 0,1,2), deformations range from -max_strain_normal to
          +max_strain_normal
        - For shear strains (axes 3,4,5), deformations range from -max_strain_shear to
          +max_strain_shear
        - Deformation axes are:
            0,1,2: x,y,z cartesian deformations
            3,4,5: yz,xz,xy shear deformations
    """
    # Deformation rules for different Bravais lattices
    # Each tuple contains (allowed_axes, symmetry_handler_function)
    deformation_rules: dict[BravaisType, DeformationRule] = {
        BravaisType.CUBIC: DeformationRule([0, 3], regular_symmetry),
        BravaisType.HEXAGONAL: DeformationRule([0, 2, 3, 5], hexagonal_symmetry),
        BravaisType.TRIGONAL: DeformationRule([0, 2, 3, 4, 5], trigonal_symmetry),
        BravaisType.TETRAGONAL: DeformationRule([0, 2, 3, 5], tetragonal_symmetry),
        BravaisType.ORTHORHOMBIC: DeformationRule(
            [0, 1, 2, 3, 4, 5], orthorhombic_symmetry
        ),
        BravaisType.MONOCLINIC: DeformationRule([0, 1, 2, 3, 4, 5], monoclinic_symmetry),
        BravaisType.TRICLINIC: DeformationRule([0, 1, 2, 3, 4, 5], triclinic_symmetry),
    }

    # Default to triclinic (lowest symmetry) if bravais_type not specified
    if bravais_type is None:
        bravais_type = BravaisType.TRICLINIC

    # Get deformation rules for this Bravais lattice
    rule = deformation_rules[bravais_type]
    allowed_axes = rule.axes

    # Generate deformed structures
    deformed_states = []
    device = base_state.cell.device
    dtype = base_state.cell.dtype

    for axis in allowed_axes:
        if axis < 3:  # Normal strain
            # Generate symmetric strains around zero
            strains = torch.linspace(
                -max_strain_normal, max_strain_normal, n_deform, device=device, dtype=dtype
            )
        else:  # Shear strain
            # Generate symmetric strains around zero
            strains = torch.linspace(
                -max_strain_shear, max_strain_shear, n_deform, device=device, dtype=dtype
            )
    
        # Skip zero strain
        strains = strains[strains != 0]

        for strain in strains:
            deformed = get_cart_deformed_cell(
                base_state=base_state, axis=axis, size=strain.item()
            )
            deformed_states.append(deformed)

    return deformed_states


def get_strain(
    deformed_state: ElasticState, reference_state: ElasticState | None = None
) -> torch.Tensor:
    """Calculate strain tensor in Voigt notation.

    Computes the strain tensor as a 6-component vector following Voigt notation.
    The calculation is performed relative to a reference (undeformed) state.

    Args:
        deformed_state: ElasticState containing the deformed configuration
        reference_state: Optional reference (undeformed) state. If None,
                        uses deformed_state as reference

    Returns:
        torch.Tensor: 6-component strain vector [εxx, εyy, εzz, εyz, εxz, εxy]
                     following Voigt notation

    Notes:
        The strain is computed as ε = (u + u^T)/2 where u = M^(-1)ΔM,
        with M being the cell matrix and ΔM the cell difference.

        Voigt notation mapping:
        - ε[0] = εxx = u[0,0]
        - ε[1] = εyy = u[1,1]
        - ε[2] = εzz = u[2,2]
        - ε[3] = εyz = u[2,1]
        - ε[4] = εxz = u[2,0]
        - ε[5] = εxy = u[1,0]
    """
    if not isinstance(deformed_state, ElasticState):
        raise TypeError("deformed_state must be an ElasticState")

    # Use deformed state as reference if none provided
    if reference_state is None:
        reference_state = deformed_state

    # Get cell matrices
    deformed_cell = deformed_state.cell
    reference_cell = reference_state.cell

    # Calculate displacement gradient tensor: u = M^(-1)ΔM
    cell_difference = deformed_cell - reference_cell
    reference_inverse = torch.linalg.inv(reference_cell)
    u = torch.matmul(reference_inverse, cell_difference)

    # Compute symmetric strain tensor: ε = (u + u^T)/2
    strain = (u + u.transpose(-2, -1)) / 2

    # Convert to Voigt notation
    return torch.tensor(
        [
            strain[0, 0],  # εxx
            strain[1, 1],  # εyy
            strain[2, 2],  # εzz
            strain[2, 1],  # εyz
            strain[2, 0],  # εxz
            strain[1, 0],  # εxy
        ],
        device=deformed_cell.device,
        dtype=deformed_cell.dtype,
    )


def get_elastic_tensor(
    base_state: ElasticState,
    deformed_states: list[ElasticState],
    stresses: torch.Tensor,
    base_pressure: torch.Tensor,
    bravais_type: BravaisType,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]:
    """Calculate elastic tensor from stress-strain relationships.

    Computes the elastic tensor by fitting stress-strain relations to a set of
    linear equations built from crystal symmetry and deformation data.

    Args:
        base_state: ElasticState containing reference structure
        deformed_states: List of deformed ElasticStates with calculated stresses
        stresses: Tensor of shape (n_states, 6) containing stress components for each
                 state
        base_pressure: Reference pressure of the base state
        bravais_type: Crystal system (BravaisType enum)

    Returns:
        Tuple containing:
        - torch.Tensor: Cij elastic constants
        - Tuple containing:
            - torch.Tensor: Bij Birch coefficients
            - torch.Tensor: Residuals from least squares fit
            - int: Rank of solution
            - torch.Tensor: Singular values

    Notes:
        The elastic tensor is calculated as Cij = Bij - P, where:
        - Bij are the Birch coefficients from least squares fitting
        - P is a pressure-dependent correction specific to each symmetry

        Stress and strain are related by: σᵢ = Σⱼ Cᵢⱼ εⱼ
    """
    # Deformation rules for different Bravais lattices
    deformation_rules: dict[BravaisType, DeformationRule] = {
        BravaisType.CUBIC: DeformationRule([0, 3], regular_symmetry),
        BravaisType.HEXAGONAL: DeformationRule([0, 2, 3, 5], hexagonal_symmetry),
        BravaisType.TRIGONAL: DeformationRule([0, 2, 3, 4, 5], trigonal_symmetry),
        BravaisType.TETRAGONAL: DeformationRule([0, 2, 3, 4, 5], tetragonal_symmetry),
        BravaisType.ORTHORHOMBIC: DeformationRule(
            [0, 1, 2, 3, 4, 5], orthorhombic_symmetry
        ),
        BravaisType.MONOCLINIC: DeformationRule([0, 1, 2, 3, 4, 5], monoclinic_symmetry),
        BravaisType.TRICLINIC: DeformationRule([0, 1, 2, 3, 4, 5], triclinic_symmetry),
    }

    # Get symmetry handler for this Bravais lattice
    rule = deformation_rules[bravais_type]
    symmetry_handler = rule.symmetry_handler

    # Calculate strains for all deformed states
    strains = []
    for deformed in deformed_states:
        strain = get_strain(deformed, reference_state=base_state)
        strains.append(strain)

    # Remove ambient pressure from stresses
    p_correction = torch.tensor(
        [base_pressure] * 3 + [0] * 3, device=stresses.device, dtype=stresses.dtype
    )
    corrected_stresses = stresses - p_correction

    # Build equation matrix using symmetry
    eq_matrices = [symmetry_handler(strain) for strain in strains]
    eq_matrix = torch.stack(eq_matrices)

    # Reshape for least squares solving
    eq_matrix = eq_matrix.reshape(-1, eq_matrix.shape[-1])
    stress_vector = corrected_stresses.reshape(-1)

    # Solve least squares problem
    Bij, residuals, rank, singular_values = torch.linalg.lstsq(eq_matrix, stress_vector)

    # Calculate elastic constants with pressure correction
    p = base_pressure
    pressure_corrections = {
        BravaisType.CUBIC: torch.tensor([-p, p, -p]),
        BravaisType.HEXAGONAL: torch.tensor([-p, -p, p, p, -p]),
        BravaisType.TRIGONAL: torch.tensor([-p, -p, p, p, p, p, -p]),
        BravaisType.TETRAGONAL: torch.tensor([-p, -p, p, p, -p, -p, -p]),
        BravaisType.ORTHORHOMBIC: torch.tensor([-p, -p, -p, p, p, p, -p, -p, -p]),
        BravaisType.MONOCLINIC: torch.tensor(
            [-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p]
        ),
        BravaisType.TRICLINIC: torch.tensor(
            [
                -p,
                p,
                p,
                p,
                p,
                p,  # C11-C16
                -p,
                p,
                p,
                p,
                p,  # C22-C26
                -p,
                p,
                p,
                p,  # C33-C36
                -p,
                p,
                p,  # C44-C46
                -p,
                p,  # C55-C56
                -p,  # C66
            ]
        ),
    }

    # Apply pressure correction for the specific symmetry
    Cij = Bij - pressure_corrections[bravais_type].to(Bij.device)

    return Cij, (Bij, residuals, rank, singular_values)


def get_full_elastic_tensor(  # noqa: C901
    Cij: torch.Tensor,
    bravais_type: BravaisType,
) -> torch.Tensor:
    """Convert the symmetry-reduced elastic constants to full 6x6 elastic tensor.

    Args:
        Cij: Tensor containing independent elastic constants for the given symmetry
        bravais_type: Crystal system determining the symmetry rules

    Returns:
        torch.Tensor: Full 6x6 elastic tensor with all components

    Notes:
        The mapping follows Voigt notation where:
        1 = xx, 2 = yy, 3 = zz, 4 = yz, 5 = xz, 6 = xy

        The number of independent constants varies by symmetry:
        - Cubic: 3 (C11, C12, C44)
        - Hexagonal: 5 (C11, C12, C13, C33, C44)
        - Trigonal: 6 (C11, C12, C13, C14, C33, C44)
        - Tetragonal: 7 (C11, C12, C13, C16, C33, C44, C66)
        - Orthorhombic: 9 (C11, C22, C33, C12, C13, C23, C44, C55, C66)
        - Monoclinic: 13 constants (C11, C22, C33, C12, C13, C23, C44, C55, C66, C15, C25, C35, C46)
        - Triclinic: 21 constants
    """
    # Initialize full tensor
    C = torch.zeros((6, 6), dtype=Cij.dtype, device=Cij.device)

    if bravais_type == BravaisType.TRICLINIC:
        # For triclinic, we expect 21 independent constants
        if len(Cij) != 21:
            raise ValueError(
                f"Triclinic symmetry requires 21 independent constants, "
                f"but got {len(Cij)}"
            )

        # Fill the symmetric matrix
        C = torch.zeros((6, 6), dtype=Cij.dtype, device=Cij.device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                C[i, j] = C[j, i] = Cij[idx]
                idx += 1

    elif bravais_type == BravaisType.CUBIC:
        # C11, C12, C44
        C11, C12, C44 = Cij
        diag = torch.tensor([C11, C11, C11, C44, C44, C44])
        C.diagonal().copy_(diag)
        C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C12

    elif bravais_type == BravaisType.HEXAGONAL:
        # C11, C12, C13, C33, C44
        C11, C12, C13, C33, C44 = Cij
        C.diagonal().copy_(torch.tensor([C11, C11, C33, C44, C44, (C11 - C12) / 2]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13

    elif bravais_type == BravaisType.TRIGONAL:
        # C11, C12, C13, C14, C15, C33, C44
        C11, C12, C13, C14, C15, C33, C44 = Cij 
        C.diagonal().copy_(torch.tensor([C11, C11, C33, C44, C44, (C11 - C12) / 2]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
        C[0, 3] = C[3, 0] = C14
        C[0, 4] = C[4, 0] = C15
        C[1, 3] = C[3, 1] = -C14 
        C[1, 4] = C[4, 1] = -C15
        C[3, 5] = C[5, 3] = -C15
        C[4, 5] = C[5, 4] = C14

    elif bravais_type == BravaisType.TETRAGONAL:
        # C11, C12, C13, C16, C33, C44, C66
        C11, C12, C13, C16, C33, C44, C66 = Cij
        C.diagonal().copy_(torch.tensor([C11, C11, C33, C44, C44, C66]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
        C[0, 5] = C[5, 0] = C16
        C[1, 5] = C[5, 1] = -C16

    elif bravais_type == BravaisType.ORTHORHOMBIC:
        # C11, C22, C33, C12, C13, C23, C44, C55, C66
        C11, C12, C13, C22, C23, C33, C44, C55, C66 = Cij
        C.diagonal().copy_(torch.tensor([C11, C22, C33, C44, C55, C66]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C13
        C[1, 2] = C[2, 1] = C23

    elif bravais_type == BravaisType.MONOCLINIC:
        # 13 independent constants
        C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66 = Cij
        C.diagonal().copy_(torch.tensor([C11, C22, C33, C44, C55, C66]))
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C13
        C[0, 4] = C[4, 0] = C15
        C[1, 2] = C[2, 1] = C23
        C[1, 4] = C[4, 1] = C25
        C[2, 4] = C[4, 2] = C35
        C[3, 5] = C[5, 3] = C46

    return C
