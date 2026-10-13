"""Wrapper for FairChem models in TorchSim.

This module re-exports the FairChem package's torch-sim integration for convenient
importing. The actual implementation is maintained in the `fairchem-core` package.

References:
    - FairChem Models Package: https://github.com/facebookresearch/fairchem
"""

import traceback
import warnings
from typing import Any


try:
    from fairchem.core.calculate.torchsim_interface import (
        FairChemModel as _FairChemTorchSimModel,
    )

    from torch_sim.state import SimState, _to_legacy_pbc_state

    class FairChemModel(_FairChemTorchSimModel):
        """FairChem model wrapper for torch-sim."""

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Run forward pass with the legacy global pbc row.

            The FairChem integration reads state.pbc as a single global setting,
            so present the shared (3,) row until it supports per-system pbc.
            """
            if args and isinstance(args[0], SimState):
                args = (_to_legacy_pbc_state(args[0]), *args[1:])
            elif isinstance(kwargs.get("state"), SimState):
                kwargs["state"] = _to_legacy_pbc_state(kwargs["state"])
            return super().forward(*args, **kwargs)

except ImportError as exc:
    warnings.warn(f"FairChem import failed: {traceback.format_exc()}", stacklevel=2)

    from torch_sim.models.interface import ModelInterface

    class FairChemModel(ModelInterface):
        """Dummy FairChem model wrapper for torch-sim to enable safe imports.

        NOTE: This class is a placeholder when `fairchem-core` is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["FairChemModel"]
