"""Wrapper for metatomic-based models in TorchSim.

Re-exports the metatomic-torchsim package's TorchSim integration.
"""

import traceback
import warnings
from typing import Any


try:
    from metatomic_torchsim import MetatomicModel  # pyright: ignore[reportMissingImports]
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
