"""Types used across TorchSim."""

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Union

import torch


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

    from torch_sim.state import SimState


MemoryScaling = Literal["n_atoms_x_density", "n_atoms"]
StateKey = Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "system_idx"]
StateDict = dict[StateKey, torch.Tensor]


class BravaisType(StrEnum):
    """Enumeration of the seven Bravais lattice types in 3D crystals.

    These lattice types represent the distinct crystal systems classified
    by their symmetry properties, from highest symmetry (cubic) to lowest
    symmetry (triclinic).

    Each type has specific constraints on lattice parameters and angles,
    which determine the number of independent elastic constants.
    """

    cubic = "cubic"
    hexagonal = "hexagonal"
    trigonal = "trigonal"
    tetragonal = "tetragonal"
    orthorhombic = "orthorhombic"
    monoclinic = "monoclinic"
    triclinic = "triclinic"


StateLike = Union[
    "Atoms",
    "Structure",
    "PhonopyAtoms",
    list["Atoms"],
    list["Structure"],
    list["PhonopyAtoms"],
    "SimState",
]

# Type alias accepted by coerce_prng
PRNGLike = int | torch.Generator | None
