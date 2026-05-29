"""Wrapper for MatterSim models in TorchSim.

This module re-exports the MatterSim package's torch-sim integration for
convenient importing. The actual implementation is maintained in the
``mattersim`` package (``mattersim.torchsim``).

References:
    - MatterSim: https://github.com/microsoft/mattersim
"""

import traceback
import warnings
from typing import Any


try:
    from mattersim.torchsim import TorchSimWrapper as MatterSimModel

except ImportError as exc:
    warnings.warn(f"MatterSim import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class MatterSimModel(ModelInterface):
        """Dummy MatterSim model wrapper for torch-sim to enable safe imports.

        NOTE: This class is a placeholder when ``mattersim`` is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["MatterSimModel"]
