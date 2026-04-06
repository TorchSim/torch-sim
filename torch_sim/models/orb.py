"""Wrapper for ORB models in TorchSim.

This module re-exports the ORB package's torch-sim integration for convenient
importing. The actual implementation is maintained in the orb-models package.

References:
    - ORB Models Package: https://github.com/orbital-materials/orb-models
"""

import traceback
import warnings
from typing import Any

import torch


try:
    from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

    import torch_sim as ts

    # Re-export with backward-compatible name
    class OrbModel(OrbTorchSimModel):
        """ORB model wrapper for torch-sim."""

        @staticmethod
        def _normalize_charge_spin(state: "ts.SimState") -> "ts.SimState":
            """Provide ORB's optional charge/spin inputs when they are missing."""
            charge = getattr(state, "charge", None)
            spin = getattr(state, "spin", None)
            if charge is not None and spin is not None:
                return state
            zeros = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
            return ts.SimState.from_state(
                state,
                charge=charge if charge is not None else zeros,
                spin=spin if spin is not None else zeros,
            )

        def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Run forward pass, detaching outputs unless retain_graph is True."""
            if args and isinstance(args[0], ts.SimState):
                args = (self._normalize_charge_spin(args[0]), *args[1:])
            elif isinstance(kwargs.get("state"), ts.SimState):
                kwargs["state"] = self._normalize_charge_spin(kwargs["state"])
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

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["OrbModel"]
