"""Models for TorchSim."""

# ruff: noqa: F401
import importlib
import sys
from typing import Any

from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.models.morse import MorseModel
from torch_sim.models.soft_sphere import SoftSphereModel


# Dictionary mapping attribute names to the modules where they are defined
_LAZY_IMPORTS = {
    "OrbModel": ".orb",
    "FairChemModel": ".fairchem",
    "MaceModel": ".mace",
    "SevenNetModel": ".sevennet",
    "MatterSimModel": ".mattersim",
    "GraphPESWrapper": ".graphpes",
    "MetatensorModel": ".metatensor",
}


def __getattr__(name: str) -> Any:
    """Lazily import models when they are first accessed."""
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        try:
            # Perform the import relative to the current package ('torch_sim.models')
            module = importlib.import_module(module_name, __name__)
            obj = getattr(module, name)
            # Cache the imported object in the module's dictionary
            sys.modules[__name__].__dict__[name] = obj
        except ImportError as e:
            # If the import fails, raise an AttributeError
            raise AttributeError(
                f"Module '{__name__}' has no attribute '{name}', "
                f"as the optional dependency '{module_name}' could not be imported."
            ) from e
        except AttributeError as e:
            # Handle case where module exists but the class doesn't
            raise AttributeError(
                f"Module '{__name__}' has no attribute '{name}', "
                f"as it was not found in the imported module '{module_name}'."
            ) from e

        return obj

    # If the name is not in our lazy map, raise the standard AttributeError.
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Include lazily imported names in directory listing."""
    # Get all names eagerly defined in the module's global scope
    names = list(globals().keys())
    # Add the names that can be lazily imported
    names.extend(_LAZY_IMPORTS.keys())
    # Return a sorted list of unique names
    return sorted(set(names))
