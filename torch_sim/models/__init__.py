"""Models for Torch-Sim."""

# ruff: noqa: F401

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.soft_sphere import SoftSphereModel


try:
    from torch_sim.models.fairchem import FairChemModel
except ImportError:
    pass

try:
    from torch_sim.models.mace import MaceModel
except ImportError:
    pass
