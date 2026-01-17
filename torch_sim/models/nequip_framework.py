"""Wrapper for NequIP-Allegro models in TorchSim.

This module re-exports the NequIP framework's torch-sim integration for convenient
importing. The actual implementation is maintained in the NequIP package.

References:
    - NequIP Package: https://github.com/mir-group/nequip
"""

import traceback
import warnings
from typing import Any


try:
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    # Re-export with backward-compatible name
    class NequIPFrameworkModel(NequIPTorchSimCalc):
        """NequIP model framework wrapper for torch-sim.

        NOTE: NequIPFrameworkModel.dtype is always set to torch.float64.
        The AOTInductor may actually contain a different dtype but the
        model will cast to the correct dtype internally.
        """

except ImportError as exc:
    warnings.warn(f"NequIP import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class NequIPFrameworkModel(ModelInterface):  # type: ignore[no-redef]
        """NequIP model framework wrapper for torch-sim.

        NOTE:This class is a placeholder when NequIP is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


__all__ = ["NequIPFrameworkModel"]
