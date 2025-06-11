"""Wrapper for FairChem ecosystem models in TorchSim.

This module provides a TorchSim wrapper of the FairChem models for computing
energies, forces, and stresses of atomistic systems. It serves as a wrapper around
the FairChem library, integrating it with the torch_sim framework to enable seamless
simulation of atomistic systems with machine learning potentials.

The FairChemModel class adapts FairChem models to the ModelInterface protocol,
allowing them to be used within the broader torch_sim simulation framework.

Notes:
    This implementation requires FairChem to be installed and accessible.
    It supports various model configurations through configuration files or
    pretrained model checkpoints.
"""

# ruff: noqa: T201

from __future__ import annotations

import traceback
import typing
import warnings
from typing import Any

import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface


try:
    from fairchem.core.calculate.ase_calculator import (
        FAIRChemCalculator,
        InferenceSettings,
        UMATask,
    )
    from fairchem.core.common.utils import setup_imports, setup_logging

except ImportError as exc:
    warnings.warn(f"FairChem import failed: {traceback.format_exc()}", stacklevel=2)

    class FairChemModel(ModelInterface):
        """FairChem model wrapper for torch_sim.

        This class is a placeholder for the FairChemModel class.
        It raises an ImportError if FairChem is not installed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch_sim.typing import StateDict


class FairChemModel(ModelInterface):
    """Computes atomistic energies, forces and stresses using a FairChem model.

    This class wraps a FairChem model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, checkpoint loading, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    The model can be initialized either with a configuration file or a pretrained
    checkpoint. It supports various model architectures and configurations supported by
    FairChem.

    This version uses the modern fairchem-core-2.2.0+ API with FAIRChemCalculator.

    Attributes:
        calculator (FAIRChemCalculator): The underlying FairChem calculator
        _device (torch.device): Device where computation is performed
        _dtype (torch.dtype): Data type used for computation
        _compute_stress (bool): Whether to compute stress tensor
        implemented_properties (list): Model outputs the model can compute

    Examples:
        >>> model = FairChemModel(model="path/to/checkpoint.pt", compute_stress=True)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: str | Path | None,
        neighbor_list_fn: Callable | None = None,
        *,  # force remaining arguments to be keyword-only
        model_name: str | None = None,
        cpu: bool = False,
        seed: int = 41,
        dtype: torch.dtype | None = None,
        compute_stress: bool = False,
        task_name: UMATask | str | None = None,
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
    ) -> None:
        """Initialize the FairChemModel with specified configuration.

        Uses the modern FAIRChemCalculator.from_model_checkpoint API for simplified
        model loading and configuration.

        Args:
            model (str | Path | None): Path to model checkpoint file
            neighbor_list_fn (Callable | None): Function to compute neighbor lists
                (not currently supported)
            model_name (str | None): Name of pretrained model to load
            cpu (bool): Whether to use CPU instead of GPU for computation
            seed (int): Random seed for reproducibility
            dtype (torch.dtype | None): Data type to use for computation
            compute_stress (bool): Whether to compute stress tensor
            task_name (UMATask | str | None): Task type for the model
            inference_settings (InferenceSettings | str): Inference configuration
            overrides (dict | None): Configuration overrides

        Raises:
            RuntimeError: If both model_name and model are specified
            NotImplementedError: If custom neighbor list function is provided
            ValueError: If neither model nor model_name is provided

        Notes:
            This uses the new fairchem-core-2.2.0+ API which is much simpler than
            the previous versions.
        """
        setup_imports()
        setup_logging()
        super().__init__()

        self._dtype = dtype or torch.float32
        self._compute_stress = compute_stress
        self._compute_forces = True
        self._memory_scales_with = "n_atoms"

        if neighbor_list_fn is not None:
            raise NotImplementedError(
                "Custom neighbor list is not supported for FairChemModel."
            )

        if model_name is not None:
            if model is not None:
                raise RuntimeError(
                    "model_name and checkpoint_path were both specified, "
                    "please use only one at a time"
                )
            # For fairchem-core-2.2.0+, model_name can be used directly
            # as it supports pretrained model names from available_models
            model = model_name

        if model is None:
            raise ValueError("Either model or model_name must be provided")

        # Convert task_name to UMATask if it's a string
        if isinstance(task_name, str):
            task_name = UMATask(task_name)

        # Use the new simplified API
        device_str = "cpu" if cpu else "cuda" if torch.cuda.is_available() else "cpu"

        self.calculator = FAIRChemCalculator.from_model_checkpoint(
            name_or_path=str(model),
            task_name=task_name,
            inference_settings=inference_settings,
            overrides=overrides,
            device=device_str,
            seed=seed,
        )

        self._device = torch.device(device_str)

        # Determine implemented properties from the calculator
        # This is a simplified approach - in practice you might want to
        # inspect the model configuration more carefully
        self.implemented_properties = ["energy", "forces"]
        if compute_stress:
            self.implemented_properties.append("stress")

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type used by the model."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Return the device where the model is located."""
        return self._device

    def forward(self, state: ts.SimState | StateDict) -> dict:
        """Perform forward pass to compute energies, forces, and other properties.

        Takes a simulation state and computes the properties implemented by the model,
        such as energy, forces, and stresses.

        Args:
            state (SimState | StateDict): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3]

        Notes:
            This implementation uses the FAIRChemCalculator which expects ASE Atoms
            objects. The conversion is handled internally.
        """
        if isinstance(state, dict):
            state = ts.SimState(**state, masses=torch.ones_like(state["positions"]))

        if state.device != self._device:
            state = state.to(self._device)

        # Convert torch_sim SimState to ASE Atoms objects for FAIRChemCalculator
        from ase import Atoms

        if state.batch is None:
            state.batch = torch.zeros(state.positions.shape[0], dtype=torch.int)

        natoms = torch.bincount(state.batch)
        atoms_list = []

        for i, (n, c) in enumerate(
            zip(natoms, torch.cumsum(natoms, dim=0), strict=False)
        ):
            positions = state.positions[c - n : c].cpu().numpy()
            atomic_numbers = state.atomic_numbers[c - n : c].cpu().numpy()
            cell = (
                state.row_vector_cell[i].cpu().numpy()
                if state.row_vector_cell is not None
                else None
            )

            atoms = Atoms(
                numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=state.pbc if cell is not None else False,
            )
            atoms_list.append(atoms)

        # Use FAIRChemCalculator to compute properties
        results = {}
        energies = []
        forces_list = []
        stress_list = []

        for atoms in atoms_list:
            atoms.calc = self.calculator

            # Get energy
            energy = atoms.get_potential_energy()
            energies.append(energy)

            # Get forces
            forces = atoms.get_forces()
            forces_list.append(
                torch.from_numpy(forces).to(self._device, dtype=self._dtype)
            )

            # Get stress if requested
            if self._compute_stress:
                try:
                    stress = atoms.get_stress(voigt=False)  # 3x3 tensor
                    stress_list.append(
                        torch.from_numpy(stress).to(self._device, dtype=self._dtype)
                    )
                except (RuntimeError, AttributeError, NotImplementedError):
                    # If stress computation fails, fill with zeros
                    stress_list.append(
                        torch.zeros(3, 3, device=self._device, dtype=self._dtype)
                    )

        # Combine results
        results["energy"] = torch.tensor(energies, device=self._device, dtype=self._dtype)
        results["forces"] = torch.cat(forces_list, dim=0)

        if self._compute_stress and stress_list:
            results["stress"] = torch.stack(stress_list, dim=0)

        return results
