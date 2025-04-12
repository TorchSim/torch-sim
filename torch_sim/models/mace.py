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

import typing
from collections.abc import Callable
from pathlib import Path

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict


try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.tools import atomic_numbers_to_indices, utils
except ImportError:

    class MaceModel(torch.nn.Module, ModelInterface):
        """MACE model wrapper for torch_sim.

        This class is a placeholder for the MaceModel class.
        It raises an ImportError if MACE is not installed.
        """

        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            """Dummy init for type checking."""
            raise ImportError("MACE must be installed to use this model.")


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
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device, dtype=dtype).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


class MaceModel(torch.nn.Module, ModelInterface):
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
        batch (torch.Tensor): Batch indices with shape [n_atoms].
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
        neighbor_list_fn: Callable = vesin_nl_ts,
        compute_forces: bool = True,
        compute_stress: bool = True,
        enable_cueq: bool = False,
        atomic_numbers: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> None:
        """Initialize the MACE model for energy and force calculations.

        Sets up the MACE model for energy, force, and stress calculations within
        the TorchSim framework. The model can be initialized with atomic numbers
        and batch indices, or these can be provided during the forward pass.

        Args:
            model (str | Path | torch.nn.Module | None): The MACE neural network model,
                either as a path to a saved model or as a loaded torch.nn.Module instance.
            device (torch.device | None): The device to run computations on.
                Defaults to CUDA if available, otherwise CPU.
            dtype (torch.dtype): The data type for tensor operations.
                Defaults to torch.float64.
            atomic_numbers (torch.Tensor | None): Atomic numbers with shape [n_atoms].
                If provided at initialization, cannot be provided again during forward.
            batch (torch.Tensor | None): Batch indices with shape [n_atoms] indicating
                which system each atom belongs to. If not provided with atomic_numbers,
                all atoms are assumed to be in the same system.
            neighbor_list_fn (Callable): Function to compute neighbor lists.
                Defaults to vesin_nl_ts.
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
            # Implement model loading from file
            raise NotImplementedError("Loading model from file not implemented yet")
        if isinstance(model, torch.nn.Module):
            self.model = model
        else:
            raise TypeError("Model must be a path or torch.nn.Module")

        self.model = model.to(self._device)
        self.model = self.model.eval()

        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        if enable_cueq:
            print("Converting models to CuEq for acceleration")
            self.model = run_e3nn_to_cueq(self.model)

        # Set model properties
        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = torch.tensor(
            self.model.atomic_numbers.detach().clone(), device=self.device
        )

        # Store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None

        # Set up batch information if atomic numbers are provided
        if atomic_numbers is not None:
            if batch is None:
                # If batch is not provided, assume all atoms belong to same system
                batch = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )

            self.setup_from_batch(atomic_numbers, batch)

    def setup_from_batch(self, atomic_numbers: torch.Tensor, batch: torch.Tensor) -> None:
        """Set up internal state from atomic numbers and batch indices.

        Processes the atomic numbers and batch indices to prepare the model for
        forward pass calculations. Creates the necessary data structures for
        batched processing of multiple systems.

        Args:
            atomic_numbers (torch.Tensor): Atomic numbers tensor with shape [n_atoms].
            batch (torch.Tensor): Batch indices tensor with shape [n_atoms] indicating
                which system each atom belongs to.
        """
        self.atomic_numbers = atomic_numbers
        self.batch = batch

        # Determine number of systems and atoms per system
        self.n_systems = batch.max().item() + 1

        # Create ptr tensor for batch boundaries
        self.n_atoms_per_system = []
        ptr = [0]
        for b in range(self.n_systems):
            batch_mask = batch == b
            n_atoms = batch_mask.sum().item()
            self.n_atoms_per_system.append(n_atoms)
            ptr.append(ptr[-1] + n_atoms)

        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self.device)
        self.total_atoms = atomic_numbers.shape[0]

        # Create one-hot encodings for all atoms
        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(atomic_numbers.cpu(), z_table=self.z_table),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(-1),
            num_classes=len(self.z_table),
            dtype=self.dtype,
        )

    def forward(  # noqa: C901
        self,
        state: SimState | StateDict,
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying MACE model. Handles batched calculations for
        multiple systems and constructs the necessary neighbor lists.

        Args:
            state (SimState | StateDict): State object containing positions, cell,
                and other system information. Can be either a SimState object or a
                dictionary with the relevant fields.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True

        Raises:
            ValueError: If atomic numbers are not provided either in the constructor
                or in the forward pass, or if provided in both places.
            ValueError: If batch indices are not provided when needed.
        """
        # Extract required data from input
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        # Handle input validation for atomic numbers
        if state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # Use batch from init if not provided
        if state.batch is None:
            if not hasattr(self, "batch"):
                raise ValueError(
                    "Batch indices must be provided if not set during initialization"
                )
            state.batch = self.batch

        # Update batch information if new atomic numbers are provided
        if (
            state.atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and not torch.equal(
                state.atomic_numbers,
                getattr(self, "atomic_numbers", torch.zeros(0, device=self.device)),
            )
        ):
            self.setup_from_batch(state.atomic_numbers, state.batch)

        # Process each system's neighbor list separately
        edge_indices = []
        shifts_list = []
        unit_shifts_list = []
        offset = 0

        # TODO (AG): Currently doesn't work for batched neighbor lists
        for b in range(self.n_systems):
            batch_mask = state.batch == b
            # Calculate neighbor list for this system
            edge_idx, shifts_idx = self.neighbor_list_fn(
                positions=state.positions[batch_mask],
                cell=state.row_vector_cell[b],
                pbc=state.pbc,
                cutoff=self.r_max,
            )

            # Adjust indices for the batch
            edge_idx = edge_idx + offset
            shifts = torch.mm(shifts_idx, state.row_vector_cell[b])

            edge_indices.append(edge_idx)
            unit_shifts_list.append(shifts_idx)
            shifts_list.append(shifts)

            offset += len(state.positions[batch_mask])

        # Combine all neighbor lists
        edge_index = torch.cat(edge_indices, dim=1)
        unit_shifts = torch.cat(unit_shifts_list, dim=0)

        # Get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                node_attrs=self.node_attrs,
                batch=state.batch,
                pbc=state.pbc,
                cell=state.row_vector_cell,
                positions=state.positions,
                edge_index=edge_index,
                unit_shifts=unit_shifts,
                shifts=shifts_list,
            ),
            compute_force=self.compute_forces,
            compute_stress=self.compute_stress,
        )

        results = {}

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
