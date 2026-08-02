"""Wrapper for metatomic-based models in TorchSim.

Re-exports the metatomic-torchsim package's TorchSim integration.
"""

import traceback
import warnings
from typing import Any


try:
    from metatomic_torchsim import (  # pyright: ignore[reportMissingImports]
        MetatomicModel as _MetatomicTorchSimModel,
    )

    from torch_sim.state import SimState, _to_legacy_pbc_state

    class MetatomicModel(_MetatomicTorchSimModel):
        """Metatomic model wrapper for torch-sim."""

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Run forward pass with the legacy global pbc row.

            The metatomic integration reads state.pbc as a single global
            setting, so present the shared (3,) row until it supports
            per-system pbc.
            """
            if args and isinstance(args[0], SimState):
                args = (_to_legacy_pbc_state(args[0]), *args[1:])
            elif isinstance(kwargs.get("state"), SimState):
                kwargs["state"] = _to_legacy_pbc_state(kwargs["state"])
            return super().forward(*args, **kwargs)

except ImportError as exc:
    warnings.warn(
        f"metatomic-torchsim import failed: {traceback.format_exc()}", stacklevel=2
    )

    from torch_sim.models.interface import ModelInterface

    class MetatomicModel(ModelInterface):
        """Placeholder when metatomic-torchsim is not installed."""

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Raise the original ImportError."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["MetatomicModel"]
