"""Wrapper for DeePMD-kit models in TorchSim.

This module provides :class:`DeepmdModel`, the TorchSim
`ModelInterface` implementation for the PyTorch backend of DeePMD-kit.
The underlying implementation is maintained in the standalone
`deepmd_torchsim` package, available from
`GitHub <https://github.com/rahulumrao/deepmd_torchsim>`_ and
`PyPI <https://pypi.org/project/deepmd-torchsim/>`_.

`DeepmdModel` evaluates DeePMD-kit interatomic potential models and
provides energies, atomic forces, and stress tensors derived from the
virial. It supports custom-trained `se_e2_a` models as well as
multitask and multidomain foundation-model checkpoints, including DPA-3
models through the `head=` argument. See the `deepmd_torchsim`
documentation for usage examples, installation instructions, and
requirements for a compatible `deepmd-kit` backend.

If `deepmd_torchsim` is not installed, this module will throw a warning and
provides a placeholder :class:`DeepmdModel` that raises the underlying
`ImportError` when instantiated.

References:
    - DeePMD-kit: https://github.com/deepmodeling/deepmd-kit
    - deepmd_torchsim: https://github.com/rahulumrao/deepmd_torchsim
"""

import traceback
import warnings
from typing import Any


try:
    from deepmd_torchsim import DeepmdModel
except ImportError as exc:
    warnings.warn(
        f"deepmd_torchsim import failed: {traceback.format_exc()}", stacklevel=2
    )

    from torch_sim.models.interface import ModelInterface

    class DeepmdModel(ModelInterface):
        """Placeholder when deepmd_torchsim is not installed."""

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Raise the original ImportError."""
            raise err

        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            """Unreachable — __init__ always raises."""
            raise NotImplementedError


__all__ = ["DeepmdModel"]
