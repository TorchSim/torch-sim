"""Wrapper for MACE model in TorchSim.

This module provides a TorchSim wrapper of the MACE model for computing
energies, forces, and stresses for atomistic systems. It integrates the MACE model
with TorchSim's simulation framework, handling batched computations for multiple
systems simultaneously.

The implementation supports various features including:

* Computing energies, forces, and stresses
* Handling periodic boundary conditions (PBC)
* Optional CuEq acceleration for improved performance
* Batched calculations for multiple systems

Notes:
    This module depends on the MACE package and implements the ModelInterface
    for compatibility with the broader TorchSim framework.
"""

import traceback
import warnings
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl


try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.tools import atomic_numbers_to_indices, utils
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(f"MACE import failed: {traceback.format_exc()}", stacklevel=2)

    class MaceModel(ModelInterface):
        """MACE model wrapper for torch-sim.

        This class is a placeholder for the MaceModel class.
        It raises an ImportError if MACE is not installed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


def to_one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generates one-hot encoding from indices.

    NOTE: this is a modified version of the to_one_hot function in mace.tools,
    consider using upstream version if possible after https://github.com/ACEsuit/mace/pull/903/
    is merged.

    Args:
        indices: A tensor of shape (N x 1) containing class indices.
        num_classes: An integer specifying the total number of classes.
        dtype: The desired data type of the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (N x num_classes) containing the
            one-hot encodings.
    """
    shape = (*indices.shape[:-1], num_classes)
    oh = torch.zeros(shape, device=indices.device, dtype=dtype).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


class MaceModel(ModelInterface):
    """Computes energies for multiple systems using a MACE model.

    This class wraps a MACE model to compute energies, forces, and stresses for
    atomic systems within the TorchSim framework. It supports batched calculations
    for multiple systems and handles the necessary transformations between
    TorchSim's data structures and MACE's expected inputs.

    Attributes:
        r_max (float): Cutoff radius for neighbor interactions.
        z_table (utils.AtomicNumberTable): Table mapping atomic numbers to indices.
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
        model: str | Path | torch.nn.Module | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_fn: Callable = torchsim_nl,
        compute_forces: bool = True,
        compute_stress: bool = True,
        enable_cueq: bool = False,
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
    ) -> None:
        """Initialize the MACE model for energy and force calculations.

        Sets up the MACE model for energy, force, and stress calculations within
        the TorchSim framework. The model can be initialized with atomic numbers
        and system indices, or these can be provided during the forward pass.

        Args:
            model (str | Path | torch.nn.Module | None): The MACE neural network model,
                either as a path to a saved model or as a loaded torch.nn.Module instance.
            device (torch.device | None): The device to run computations on.
                Defaults to CUDA if available, otherwise CPU.
            dtype (torch.dtype): The data type for tensor operations.
                Defaults to torch.float64.
            atomic_numbers (torch.Tensor | None): Atomic numbers with shape [n_atoms].
                If provided at initialization, cannot be provided again during forward.
            system_idx (torch.Tensor | None): System indices with shape [n_atoms]
                indicating which system each atom belongs to. If not provided with
                atomic_numbers, all atoms are assumed to be in the same system.
            neighbor_list_fn (Callable): Function to compute neighbor lists.
                Defaults to torch_nl_linked_cell.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress. Defaults to True.
            enable_cueq (bool): Whether to enable CuEq acceleration. Defaults to False.

        Raises:
            NotImplementedError: If model is provided as a file path (not
                implemented yet).
            TypeError: If model is neither a path nor a torch.nn.Module.
        """
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.neighbor_list_fn = neighbor_list_fn
        self._memory_scales_with = "n_atoms_x_density"

        # Load model if provided as path
        if isinstance(model, str | Path):
            self.model = torch.load(model, map_location=self.device, weights_only=False)
        elif isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
        else:
            raise TypeError("Model must be a path or torch.nn.Module")

        self.model = self.model.eval()

        # Move all model components to device
        self.model = self.model.to(device=self._device)
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        if enable_cueq:
            print("Converting models to CuEq for acceleration")  # noqa: T201
            self.model = run_e3nn_to_cueq(self.model, device=self.device.type)

        # Set model properties
        self.r_max = self.model.r_max
        atomic_nums = self.model.atomic_numbers
        if not isinstance(atomic_nums, torch.Tensor):
            raise TypeError("MACE model atomic_numbers must be a tensor")
        self.z_table = utils.AtomicNumberTable([int(z) for z in atomic_nums])
        self.model.atomic_numbers = atomic_nums.detach().clone().to(device=self.device)

        self.atomic_numbers_in_init = atomic_numbers is not None
        self.system_idx_in_init = system_idx is not None

        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
            self._setup_node_attrs(atomic_numbers)

        if system_idx is not None:
            self.system_idx = system_idx
            self._setup_ptr(system_idx)

        if (
            atomic_numbers is not None
            and system_idx is not None
            and system_idx.shape[0] != atomic_numbers.shape[0]
        ):
            raise ValueError(
                f"system_idx length {system_idx.shape[0]} must match "
                f"atomic_numbers length {atomic_numbers.shape[0]}."
            )

    def _setup_ptr(self, system_idx: torch.Tensor) -> None:
        """Compute system boundary pointers from system indices.

        Args:
            system_idx (torch.Tensor): System indices tensor with shape [n_atoms].
        """
        counts = torch.bincount(system_idx)
        self.n_systems = len(counts)
        self.n_atoms_per_system = counts.tolist()
        self.ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    def _setup_node_attrs(self, atomic_numbers: torch.Tensor) -> None:
        """Compute one-hot encoded node attributes from atomic numbers.

        Args:
            atomic_numbers (torch.Tensor): Atomic numbers tensor with shape [n_atoms].
        """
        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(
                    atomic_numbers.detach().cpu().numpy(), z_table=self.z_table
                ),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(-1),
            num_classes=len(self.z_table),
            dtype=self.dtype,
        )

    def forward(  # noqa: C901
        self, state: ts.SimState, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying MACE model. Handles batched calculations for
        multiple systems and constructs the necessary neighbor lists.

        Args:
            state (SimState): State object containing positions, cell, and other
                system information.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True

        Raises:
            ValueError: If atomic numbers are not provided either in the constructor
                or in the forward pass, or if provided in both places.
            ValueError: If system indices are not provided when needed.
        """
        if self.atomic_numbers_in_init:
            if state.positions.shape[0] != self.atomic_numbers.shape[0]:
                raise ValueError(
                    f"Expected {self.atomic_numbers.shape[0]} atoms, "
                    f"got {state.positions.shape[0]}."
                )
        elif not hasattr(self, "atomic_numbers") or not torch.equal(
            state.atomic_numbers, self.atomic_numbers
        ):
            self._setup_node_attrs(state.atomic_numbers)
            self.atomic_numbers = state.atomic_numbers

        if self.system_idx_in_init:
            if state.system_idx.shape[0] != self.system_idx.shape[0]:
                raise ValueError(
                    f"Expected system_idx of length {self.system_idx.shape[0]}, "
                    f"got {state.system_idx.shape[0]}."
                )
        elif not hasattr(self, "system_idx") or not torch.equal(
            state.system_idx, self.system_idx
        ):
            self._setup_ptr(state.system_idx)
            self.system_idx = state.system_idx

        # Wrap positions into the unit cell
        wrapped_positions = (
            ts.transforms.pbc_wrap_batched(
                state.positions,
                state.cell,
                state.system_idx,
                state.pbc,
            )
            if state.pbc.any()
            else state.positions
        )

        # Batched neighbor list using linked-cell algorithm
        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            wrapped_positions,
            state.row_vector_cell,
            state.pbc,
            self.r_max,
            state.system_idx,
        )
        # Convert unit cell shift indices to Cartesian shifts
        shifts = ts.transforms.compute_cell_shifts(
            state.row_vector_cell, unit_shifts, mapping_system
        )

        # Build data dict for MACE model
        data_dict = dict(
            ptr=self.ptr,
            node_attrs=self.node_attrs,
            batch=state.system_idx,
            pbc=state.pbc,
            cell=state.row_vector_cell,
            positions=wrapped_positions,
            edge_index=edge_index,
            unit_shifts=unit_shifts,
            shifts=shifts,
            total_charge=state.charge,
            total_spin=state.spin,
        )

        # Get model output
        out = self.model(
            data_dict,
            compute_force=self.compute_forces,
            compute_stress=self.compute_stress,
        )

        results: dict[str, torch.Tensor] = {}

        # Process energy
        energy = out["energy"]
        if energy is not None:
            results["energy"] = energy.detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self.device)

        # Process forces
        if self.compute_forces:
            forces = out["forces"]
            if forces is not None:
                results["forces"] = forces.detach()

        # Process stress
        if self.compute_stress:
            stress = out["stress"]
            if stress is not None:
                results["stress"] = stress.detach()

        return results


class MaceUrls(StrEnum):
    """Checkpoint download URLs for MACE models."""

    mace_mp_small = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model"
    mace_mpa_medium = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
    mace_off_small = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true"
