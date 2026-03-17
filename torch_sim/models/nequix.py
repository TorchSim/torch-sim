"""Wrapper for Nequix models in TorchSim.

This module re-exports the nequix model's torch-sim integration for convenient
importing. The actual implementation is maintained in the nequix package.

References:
    - nequix Package: https://github.com/atomicarchitects/nequix

"""

import traceback
import warnings
from typing import Any, Self


try:
    from nequix.torch_sim import NequixTorchSimModel

    # Re-export with backward-compatible name
    class NequixModel(NequixTorchSimModel):
        """Nequix model wrapper for torch-sim."""

except ImportError as exc:
    _nequix_import_error = exc  # capture before except block ends (exc is deleted)
    warnings.warn(f"Nequix import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class NequixModel(ModelInterface):
        """Nequix model wrapper for torch-sim.

        NOTE: This class is a placeholder when nequix is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err

        @classmethod
        def from_compiled_model(cls, _path: Any, *_args: Any, **_kwargs: Any) -> Self:
            """Dummy classmethod for type checking when nequix is not installed."""
            raise _nequix_import_error
