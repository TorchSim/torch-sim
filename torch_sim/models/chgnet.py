"""Wrapper for CHGNet model in TorchSim.

This module provides a TorchSim wrapper of the CHGNet model for computing
energies, forces, and stresses for atomistic systems. It integrates the CHGNet model
with TorchSim's simulation framework, handling batched computations for multiple
systems simultaneously.

The implementation supports various features including:

* Computing energies, forces, and stresses
* Handling periodic boundary conditions (PBC)
* Native batching support for multiple systems
* Magnetic moment calculations

Notes:
    This module depends on the CHGNet package and implements the ModelInterface
    for compatibility with the broader TorchSim framework.
"""

import traceback
import warnings
from typing import Any

import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.typing import StateDict


try:
    from chgnet.model.model import CHGNet
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(f"CHGNet import failed: {traceback.format_exc()}", stacklevel=2)

    class CHGNetModel(ModelInterface):
        """CHGNet model wrapper for torch-sim.

        This class is a placeholder for the CHGNetModel class.
        It raises an ImportError if CHGNet is not installed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


class CHGNetModel(ModelInterface):
    """Computes energies for multiple systems using a CHGNet model.

    This class wraps a CHGNet model to compute energies, forces, and stresses for
    atomic systems within the TorchSim framework. It supports batched calculations
    for multiple systems and handles the necessary transformations between
    TorchSim's data structures and CHGNet's expected inputs.

    Attributes:
        model (CHGNet): The underlying CHGNet neural network model.
        _memory_scales_with (str): Memory scaling metric, set to "n_atoms_x_density".
    """

    def __init__(
        self,
        model: CHGNet | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
    ) -> None:
        """Initialize the CHGNet model for energy and force calculations.

        Sets up the CHGNet model for energy, force, and stress calculations within
        the TorchSim framework. The model can be initialized with a pre-loaded CHGNet
        instance or will load the default pre-trained model.

        Args:
            model (CHGNet | None): The CHGNet neural network model instance.
                If None, loads the default pre-trained model.
            device (torch.device | None): The device to run computations on.
                Defaults to CUDA if available, otherwise CPU.
            dtype (torch.dtype): The data type for tensor operations.
                Defaults to torch.float64.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress. Defaults to True.

        Raises:
            ImportError: If CHGNet is not installed.
        """
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"

        # Load model
        if model is None:
            self.model = CHGNet.load()
        else:
            self.model = model

        # Move model to device
        self.model = self.model.to(self._device)
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying CHGNet model. Handles batched calculations for
        multiple systems and constructs the necessary data structures.

        Args:
            state (SimState | StateDict): State object containing positions, cell,
                and other system information. Can be either a SimState object or a
                dictionary with the relevant fields.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True
                - 'magnetic_moments': Magnetic moments with shape [n_atoms, 3] if
                    available in CHGNet output

        Raises:
            ValueError: If atomic numbers are not provided in the state.
        """
        # Handle state dict
        if isinstance(state, dict) and state.get("atomic_numbers") is None:
            raise ValueError("Atomic numbers must be provided in the state for CHGNet.")

        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # Validate that atomic numbers
        if sim_state.atomic_numbers is None:
            raise ValueError("Atomic numbers must be provided in the state for CHGNet.")

        # Convert SimState to list of pymatgen Structures
        structures = sim_state.to_structures()

        # Use CHGNet's batching support
        chgnet_results = self.model.predict_structure(structures)

        # Handle both single and multiple structures
        if len(structures) == 1:
            # Single structure returns a single dict
            chgnet_results = [chgnet_results]

        # Convert results to TorchSim format
        results: dict[str, torch.Tensor] = {}

        # Process energy (CHGNet returns energy per atom, multiply by number of atoms)
        energies = []
        for i, result in enumerate(chgnet_results):
            chgnet_energy_per_atom = (
                result["e"].item() if hasattr(result["e"], "item") else result["e"]
            )

            # Get number of atoms in this structure
            structure = structures[i]
            total_atoms = len(structure)

            # Multiply by number of atoms to get total energy
            total_energy = chgnet_energy_per_atom * total_atoms

            energies.append(
                torch.tensor(total_energy, device=self.device, dtype=self.dtype)
            )

        results["energy"] = torch.stack(energies)

        # Process forces
        if self.compute_forces:
            forces_list = [
                torch.tensor(result["f"], device=self.device, dtype=self.dtype)
                for result in chgnet_results
            ]
            forces = torch.cat(forces_list, dim=0)
            results["forces"] = forces

        # Process stress
        if self.compute_stress:
            stresses = torch.stack(
                [
                    torch.tensor(result["s"], device=self.device, dtype=self.dtype)
                    for result in chgnet_results
                ]
            )
            results["stress"] = stresses

        # Process magnetic moments (if available)
        if "m" in chgnet_results[0]:
            magnetic_moments_list = [
                torch.tensor(result["m"], device=self.device, dtype=self.dtype)
                for result in chgnet_results
            ]
            # Concatenate along atom dimension, similar to forces
            magnetic_moments = torch.cat(magnetic_moments_list, dim=0)
            results["magnetic_moments"] = magnetic_moments

        return results
