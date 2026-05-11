"""Wrapper for MACE model in TorchSim.

This module re-exports the MACE package's torch-sim integration for convenient
importing. The actual implementation is maintained in the `mace` package.

References:
    - MACE Package: https://github.com/ACEsuit/mace
"""

import traceback
import warnings
from typing import Any

from torch_sim.models.interface import ModelInterface


try:
    from mace.calculators.mace_torchsim import MaceTorchSimModel
except ImportError as exc:
    warnings.warn(f"MACE import failed: {traceback.format_exc()}", stacklevel=2)

    class MaceModel(ModelInterface):
        """Dummy MACE model wrapper for torch-sim to enable safe imports.

        NOTE: This class is a placeholder when `mace` is not installed.
        It raises an ImportError if accessed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err
else:
    # Create a backwards-compatible wrapper around MaceTorchSimModel
    class MaceModel(MaceTorchSimModel):
        """Computes energies for multiple systems using a MACE model.

        This class wraps the MACE first-party TorchSim interface, providing
        backwards compatibility with the previous torch-sim implementation.

        This class wraps a MACE model to compute energies, forces, and stresses for
        atomic systems within the TorchSim framework. It supports batched calculations
        for multiple systems and handles the necessary transformations between
        TorchSim's data structures and MACE's expected inputs.
        """


__all__ = ["MaceModel"]
