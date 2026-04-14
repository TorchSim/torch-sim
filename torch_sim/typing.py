"""Types used across TorchSim."""

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Union

import torch


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

    from torch_sim.state import SimState


class AtomExtras(StrEnum):
    """Preferred names for per-atom :class:`~torch_sim.state.SimState` extras.

    Stored in ``SimState._atom_extras``; leading dimension is ``n_atoms``.
    """

    PARTIAL_CHARGES = "partial_charges"
    BORN_EFFECTIVE_CHARGES = "born_effective_charges"
    MAGNETIC_MOMENTS = "magnetic_moments"


class SystemExtras(StrEnum):
    """Preferred names for per-system :class:`~torch_sim.state.SimState` extras.

    Stored in ``SimState._system_extras``; leading dimension is ``n_systems``.
    """

    CHARGE = "charge"  # TOTAL_CHARGE preferred for less ambiguity with partial charges
    SPIN = "spin"  # TOTAL_SPIN preferred
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    EXTERNAL_E_FIELD = "external_E_field"
    POLARIZABILITY = "polarizability"
    TOTAL_POLARIZATION = "total_polarization"
    EXTERNAL_H_FIELD = "external_H_field"
    MAGNETIC_SUSCEPTIBILITY = "magnetic_susceptibility"
    TOTAL_MAGNETIZATION = "total_magnetization"


class BravaisType(StrEnum):
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


ExtrasMap = dict[str, str]

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

MemoryScaling = Literal["n_atoms_x_density", "n_atoms", "n_edges"]
