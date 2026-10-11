"""Wrapper for NequIP-Allegro models in TorchSim.

This module re-exports the NequIP framework's torch-sim integration for convenient
importing. The actual implementation is maintained in the NequIP package.

References:
    - NequIP Package: https://github.com/mir-group/nequip
"""

import traceback
import warnings
from typing import Any, Self


try:
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    from torch_sim.state import SimState, _to_legacy_pbc_state

    # Re-export with backward-compatible name
    class NequIPFrameworkModel(NequIPTorchSimCalc):
        """NequIP model framework wrapper for torch-sim.

        NOTE: NequIPFrameworkModel.dtype is always set to torch.float64.
        The AOTInductor may actually contain a different dtype but the
        model will cast to the correct dtype internally.
        """

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Run forward pass with the legacy global pbc row.

            The NequIP integration reads state.pbc as a single global setting,
            so present the shared (3,) row until it supports per-system pbc.
            """
            if args and isinstance(args[0], SimState):
                args = (_to_legacy_pbc_state(args[0]), *args[1:])
            elif isinstance(kwargs.get("state"), SimState):
                kwargs["state"] = _to_legacy_pbc_state(kwargs["state"])
            return super().forward(*args, **kwargs)

except ImportError as exc:
    _nequip_import_error = exc  # capture before except block ends (exc is deleted)
    warnings.warn(f"NequIP import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class NequIPFrameworkModel(ModelInterface):
        """NequIP model framework wrapper for torch-sim.

        NOTE:This class is a placeholder when NequIP is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(
            self, err: ImportError = _nequip_import_error, *_args: Any, **_kwargs: Any
        ) -> None:
            """Dummy init for type checking."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError

        @classmethod
        def from_compiled_model(cls, _path: Any, *_args: Any, **_kwargs: Any) -> Self:
            """Dummy classmethod for type checking when NequIP is not installed."""
            raise _nequip_import_error
