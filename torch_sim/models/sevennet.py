"""Wrapper for SevenNet models in TorchSim.

This module re-exports the SevenNet package's torch-sim integration for convenient
importing. The actual implementation is maintained in the `sevenn` package.

References:
    - SevenNet Models Package: https://github.com/MDIL-SNU/SevenNet
"""

import traceback
import warnings
from typing import Any


try:
    from sevenn.torchsim import SevenNetModel

except ImportError as exc:
    warnings.warn(f"SevenNet import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class SevenNetModel(ModelInterface):
        """Dummy SevenNet model wrapper for torch-sim to enable safe imports.

        NOTE: This class is a placeholder when `sevenn` is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


__all__ = ["SevenNetModel"]
