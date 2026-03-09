"""Wrapper for MACE model in TorchSim.

This module re-exports the MACE package's torch-sim integration for convenient
importing. The actual implementation is maintained in the `mace` package.

References:
    - MACE Package: https://github.com/ACEsuit/mace
"""

import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

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

        Attributes:
            r_max (float): Cutoff radius for neighbor interactions.
            model (torch.nn.Module): The underlying MACE neural network model.
            neighbor_list_fn (Callable): Function used to compute neighbor lists.
            atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms].
            system_idx (torch.Tensor): System indices with shape [n_atoms].
            n_systems (int): Number of systems in the batch.
            n_atoms_per_system (list[int]): Number of atoms in each system.
            ptr (torch.Tensor): Pointers to the start of each system in the batch with
                shape [n_systems + 1].
            total_atoms (int): Total number of atoms across all systems.
            node_attrs (torch.Tensor): One-hot encoded atomic types with shape
                [n_atoms, n_elements].
        """

        def __init__(
            self,
            model: str | Path | torch.nn.Module,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype = torch.float64,
            neighbor_list_fn: Callable | None = None,
            compute_forces: bool = True,
            compute_stress: bool = True,
            enable_cueq: bool = False,
            atomic_numbers: torch.Tensor | None = None,
            system_idx: torch.Tensor | None = None,
            enable_oeq: bool = False,
            compile_mode: str | None = None,
        ) -> None:
            """Initialize the MACE model for energy and force calculations.

            Sets up the MACE model for energy, force, and stress calculations within
            the TorchSim framework. The model can be initialized with atomic numbers
            and system indices, or these can be provided during the forward pass.

            Args:
                model: The MACE neural network model, either as a path to a saved
                    model or as a loaded torch.nn.Module instance.
                device: The device to run computations on. Defaults to CUDA if
                    available, otherwise CPU.
                dtype: The data type for tensor operations. Defaults to
                    torch.float64.
                atomic_numbers: Atomic numbers with shape [n_atoms]. If provided
                    at initialization, cannot be provided again during forward.
                system_idx: System indices with shape [n_atoms] indicating which
                    system each atom belongs to. If not provided with
                    atomic_numbers, all atoms are assumed to be in the same
                    system.
                neighbor_list_fn: Function to compute neighbor lists. Defaults to
                    torchsim_nl from torch-sim.
                compute_forces: Whether to compute forces. Defaults to True.
                compute_stress: Whether to compute stress. Defaults to True.
                enable_cueq: Whether to enable CuEq acceleration. Defaults to
                    False.
                enable_oeq: Whether to enable OEq acceleration. Defaults to
                    False.
                compile_mode: PyTorch compilation mode (e.g., "reduce-overhead").
                    Defaults to None (no compilation).

            Raises:
                TypeError: If model is neither a path nor a torch.nn.Module.
            """
            super().__init__(
                model=model,
                device=device,
                dtype=dtype,
                neighbor_list_fn=neighbor_list_fn,
                compute_forces=compute_forces,
                compute_stress=compute_stress,
                enable_cueq=enable_cueq,
                enable_oeq=enable_oeq,
                compile_mode=compile_mode,
                atomic_numbers=atomic_numbers,
                system_idx=system_idx,
            )


__all__ = ["MaceModel"]
