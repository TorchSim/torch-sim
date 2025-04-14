"""Models for TorchSim."""

# ruff: noqa: F401

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.morse import MorseModel
from torch_sim.models.soft_sphere import SoftSphereModel


try:
    from torch_sim.models.orb import OrbModel
except ImportError:
    pass

try:
    from torch_sim.models.fairchem import FairChemModel
except ImportError:
    pass

try:
    from torch_sim.models.mace import MaceModel
except ImportError:
    pass

try:
    from torch_sim.models.sevennet import SevenNetModel
except ImportError:
    pass

try:
    from torch_sim.models.mattersim import MatterSimModel
except ImportError:
    pass

try:
    from torch_sim.models.graphpes import GraphPESWrapper
except ImportError:
    pass

try:
    from torch_sim.models.metatensor import MetatensorModel
except ImportError:
    pass
