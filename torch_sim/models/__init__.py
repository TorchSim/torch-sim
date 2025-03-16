"""Models for Torch-Sim."""

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.morse import MorseModel
from torch_sim.models.soft_sphere import SoftSphereModel


try:
    from torch_sim.models.fairchem import FairChemModel

    _has_fairchem = True
except ImportError:
    _has_fairchem = False

try:
    from torch_sim.models.mace import MaceModel

    _has_mace = True
except ImportError:
    _has_mace = False


__all__ = [
    "LennardJonesModel",
    "MorseModel",
    "SoftSphereModel",
]

if _has_fairchem:
    __all__ += ["FairChemModel"]

if _has_mace:
    __all__.append("MaceModel")
