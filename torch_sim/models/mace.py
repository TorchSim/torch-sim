"""MACE model for computing energies for multiple systems."""

from collections.abc import Callable
from pathlib import Path

import torch
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import atomic_numbers_to_indices, to_one_hot, utils

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import BaseState, StateDict


class MaceModel(torch.nn.Module, ModelInterface):
    """Computes energies for multiple systems using a MACE model.

    Implements the ModelInterface for compatibility with TorchSim.
    """

    def __init__(
        self,
        model: str | Path | torch.nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        atomic_numbers: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        *,
        neighbor_list_fn: Callable = vesin_nl_ts,
        periodic: bool = True,
        compute_force: bool = True,
        compute_stress: bool = True,
        enable_cueq: bool = False,
    ) -> None:
        """Initialize the BatchedMaceModel.

        Args:
            device: The device to run computations on.
            dtype: The data type for tensor operations.
            model: The MACE neural network model.
            atomic_numbers: Atomic numbers with shape [n_atoms].
            batch: Batch indices with shape [n_atoms].
            neighbor_list_fn: The neighbor list function to use.
            periodic: Whether to use periodic boundary conditions.
            compute_force: Whether to compute forces.
            compute_stress: Whether to compute stress.
            enable_cueq: Whether to enable CuEq acceleration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self.neighbor_list_fn = neighbor_list_fn

        # TODO: can we get rid of this shit?
        torch.set_default_dtype(self._dtype)

        print(
            f"Running BatchedMACEForce on device: {self._device} "
            f"with dtype: {self._dtype}"
        )

        # Load model if provided as path
        if isinstance(model, str | Path):
            # Implement model loading from file
            raise NotImplementedError("Loading model from file not implemented yet")
        if isinstance(model, torch.nn.Module):
            self.model = model
        else:
            raise TypeError("Model must be a path or torch.nn.Module")

        if enable_cueq:
            print("Converting models to CuEq for acceleration")
            self.model = run_e3nn_to_cueq(self.model, device=self._device).to(
                self._device
            )

        self.model = self.model.to(dtype=self._dtype, device=self._device)
        self.model.eval()

        # Set model properties
        self.periodic = periodic
        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = torch.tensor(
            self.model.atomic_numbers.clone(), device=self._device
        )

        # Store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None

        # Set PBC
        pbc = [periodic] * 3
        self.pbc_template = torch.tensor([pbc], device=self._device)
        self.pbc = None  # Will be set in forward

        # Set up batch information if atomic numbers are provided
        if atomic_numbers is not None:
            if batch is None:
                # If batch is not provided, assume all atoms belong to same system
                batch = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self._device
                )

            self.setup_from_batch(atomic_numbers, batch)

    def setup_from_batch(self, atomic_numbers: torch.Tensor, batch: torch.Tensor) -> None:
        """Set up internal state from atomic numbers and batch indices.

        Args:
            atomic_numbers: Atomic numbers tensor [n_atoms]
            batch: Batch indices tensor [n_atoms]
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

        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self._device)
        self.total_atoms = atomic_numbers.shape[0]

        # Create one-hot encodings for all atoms
        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(atomic_numbers.cpu(), z_table=self.z_table),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(-1),
            num_classes=len(self.z_table),
        )

        # Set up PBC
        self.pbc = self.pbc_template.expand(self.n_systems, -1)

    def forward(  # noqa: C901
        self,
        state: BaseState | StateDict,
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the system(s).

        Args:
            state: State object

        Returns:
            dict: Dictionary with 'energy', 'forces', and optionally 'stress'
        """
        # Extract required data from input
        if isinstance(state, dict):
            state = BaseState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.periodic:
            raise ValueError("PBC mismatch between model and state")

        # Handle input validation for atomic numbers
        if state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )
        # if atomic_numbers is None and self.atomic_numbers_in_init is False:
        #     raise ValueError(
        #         "Atomic numbers must be provided in either the constructor or forward."
        #     )
        # if atomic_numbers is not None and self.atomic_numbers_in_init is True:
        #     raise ValueError(
        #         "Atomic numbers cannot be provided in both the constructor and forward."
        #     )
        # if atomic_numbers is not None and self.atomic_numbers_in_init is False:
        #     new_atomic_number_tensor = torch.tensor(atomic_numbers, device=self.device)
        #     if new_atomic_number_tensor != self.atomic_number_tensor:
        #         self.ptr, self.batch, self.node_attrs = self.compute_atomic_numbers(
        #             new_atomic_number_tensor, self.z_table, self.device
        #         )
        #         self.atomic_number_tensor = new_atomic_number_tensor

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
                getattr(self, "atomic_numbers", torch.zeros(0, device=self._device)),
            )
        ):
            self.setup_from_batch(state.atomic_numbers, state.batch)

        cell = state.cell
        positions = state.positions

        # Ensure cell has correct shape
        # if cell is None:
        #     cell = torch.zeros(
        #         (self.n_systems, 3, 3), device=self._device, dtype=self._dtype
        #     )

        # Process each system's neighbor list separately
        edge_indices = []
        shifts_list = []
        unit_shifts_list = []
        offset = 0

        # TODO (AG): Currently doesn't work for batched neighbor lists
        for b in range(self.n_systems):
            batch_mask = state.batch == b
            # Calculate neighbor list for this system
            mapping, shifts_idx = self.neighbor_list_fn(
                positions=positions[batch_mask],
                cell=cell[b],
                pbc=self.periodic,
                cutoff=self.r_max,
            )

            # Adjust indices for the batch
            mapping = mapping + offset

            edge_indices.append(mapping)
            unit_shifts_list.append(shifts_idx)
            shifts = torch.mm(shifts_idx, cell[b])
            shifts_list.append(shifts)

            offset += len(state.positions[batch_mask])

        # Combine all neighbor lists
        edge_index = torch.cat(edge_indices, dim=1)
        shifts = torch.cat(shifts_list, dim=0)
        unit_shifts = torch.cat(unit_shifts_list, dim=0)

        # Get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                node_attrs=self.node_attrs,
                batch=state.batch,
                pbc=self.pbc,
                cell=cell,
                positions=positions,
                edge_index=edge_index,
                unit_shifts=unit_shifts,
                shifts=shifts,
            ),
            compute_force=self._compute_force,
            compute_stress=self._compute_stress,
        )

        results = {}

        # Process energy
        energy = out["energy"]
        if energy is not None:
            results["energy"] = energy.detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self._device)

        # Process forces
        if self._compute_force:
            forces = out["forces"]
            if forces is not None:
                results["forces"] = forces.detach()

        # Process stress
        if self._compute_stress:
            stress = out["stress"]
            if stress is not None:
                results["stress"] = stress.detach()

        return results
