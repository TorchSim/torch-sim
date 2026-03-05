"""Wrapper for ORB models in TorchSim.

This module re-exports the ORB package's torch-sim integration for convenient
importing. The actual implementation is maintained in the orb-models package.

References:
    - ORB Models Package: https://github.com/orbital-materials/orb-models
"""

import traceback
import warnings
from typing import Any


try:
    from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

    # Re-export with backward-compatible name
    class OrbModel(OrbTorchSimModel):
        """ORB model wrapper for torch-sim."""

        def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Run forward pass, detaching outputs unless retain_graph is True."""
            output = super().forward(*args, **kwargs)
            return {  # detach tensors as energy is not detached by default
                k: v.detach() if hasattr(v, "detach") else v for k, v in output.items()
            }

except ImportError as exc:
    warnings.warn(f"Orb import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class OrbModel(ModelInterface):
        """ORB model wrapper for torch-sim.

        NOTE: This class is a placeholder when orb-models is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


__all__ = ["OrbModel"]
