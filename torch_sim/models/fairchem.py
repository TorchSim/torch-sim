"""Wrapper for FairChem models in TorchSim.

This module re-exports the FairChem package's torch-sim integration for convenient
importing. The actual implementation is maintained in the `fairchem-core` package.

References:
    - FairChem Models Package: https://github.com/facebookresearch/fairchem
"""

import traceback
import warnings
from typing import Any


try:
    from fairchem.core.calculate.torchsim_interface import FairChemModel

except ImportError as exc:
    warnings.warn(f"FairChem import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class FairChemModel(ModelInterface):
        """Dummy FairChem model wrapper for torch-sim to enable safe imports.

        NOTE: This class is a placeholder when `fairchem-core` is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["FairChemModel"]
