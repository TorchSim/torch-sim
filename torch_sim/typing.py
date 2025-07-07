"""Types used across torch-sim."""

from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeVar, Union

import torch


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

    from torch_sim.state import SimState


MemoryScaling = Literal["n_atoms_x_density", "n_atoms"]
StateKey = Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "graph_idx"]
StateDict = dict[StateKey, torch.Tensor]
SimStateVar = TypeVar("SimStateVar", bound="SimState")


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


StateLike = Union[
    "Atoms",
    "Structure",
    "PhonopyAtoms",
    list["Atoms"],
    list["Structure"],
    list["PhonopyAtoms"],
    SimStateVar,
    list[SimStateVar],
]
