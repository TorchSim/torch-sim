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
    from orb_models.forcefield.inference.d3_model import D3SumModel
    from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

    import torch_sim as ts
    from torch_sim.elastic import voigt_6_to_full_3x3_stress

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

        def _get_results(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Parses the results into a final output dictionary."""
            results = {}
            model = (
                self.model.xc_model if isinstance(self.model, D3SumModel) else self.model
            )
            heads = getattr(model, "heads", {})
            no_direct_energy_head = "energy" not in heads
            no_direct_force_head = "forces" not in heads
            no_direct_stress_head = "stress" not in heads
            for prop in self.implemented_properties:
                if prop == "free_energy" and no_direct_energy_head:
                    continue
                if prop == "forces" and no_direct_force_head:
                    continue
                if prop == "stress" and no_direct_stress_head:
                    continue
                _prop = "energy" if prop == "free_energy" else prop

                # Do not squeeze the output tensors in the case of single atom cells
                # TODO: remove after https://github.com/orbital-materials/orb-models/pull/158
                results[prop] = torch.atleast_1d(out[_prop])

            # Rename certain keys for the conservative model
            if self.conservative:
                if model.forces_name in results:
                    results["direct_forces"] = results[model.forces_name]
                results["forces"] = results[model.grad_forces_name]

                if model.has_stress:
                    if model.stress_name in results:
                        results["direct_stress"] = results[model.stress_name]
                    results["stress"] = results[model.grad_stress_name]

            # Ensure stress has shape [-1, 3, 3]
            if "stress" in results and results["stress"].shape[-1] == 6:
                results["stress"] = voigt_6_to_full_3x3_stress(
                    torch.atleast_2d(results["stress"])
                )

            return results

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

        # Capture the original ImportError in a closure-safe default so the
        # fallback always re-raises the real import failure, even when callers
        # pass positional/keyword args (e.g. ``OrbModel(orb_ff, adapter, ...)``)
        # that would otherwise shadow an ``err`` parameter.
        def __init__(self, *_args: Any, _err: ImportError = exc, **_kwargs: Any) -> None:
            """Dummy init that re-raises the original import failure."""
            raise _err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["OrbModel"]
