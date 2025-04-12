"""MACE model calculator for computing energies, forces and stresses.

This module provides a PyTorch implementation of the MACE model calculator.
"""

import typing
from collections.abc import Callable

import torch
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import atomic_numbers_to_indices, to_one_hot, utils

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict


try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.tools import atomic_numbers_to_indices, utils

    from torch_sim.models.mace import to_one_hot
except ImportError:

    class UnbatchedMaceModel(torch.nn.Module, ModelInterface):
        """Unbatched MACE model wrapper for torch_sim.

        This class is a placeholder for the UnbatchedMaceModel class.
        It raises an ImportError if MACE is not installed.
        """

        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            """Dummy init for type checking."""
            raise ImportError("MACE must be installed to use this model.")


class UnbatchedMaceModel(torch.nn.Module, ModelInterface):
    """Computes the energy of a system using a MACE model.

    Attributes:
        model (torch.nn.Module): The MACE model.
        device (str): The device (CPU or GPU) on which computations are performed.
        neighbor_list_fn (Callable): neighbor list function used for atom interactions.
        default_dtype (torch.dtype): The default data type for tensor operations.
        r_max (float): The maximum cutoff radius for atomic interactions.
        z_table (utils.AtomicNumberTable): Table for converting between
            atomic numbers and indices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        neighbor_list_fn: Callable = vesin_nl_ts,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        compute_forces: bool = False,
        compute_stress: bool = False,
        enable_cueq: bool = False,
        atomic_numbers: list[int] | torch.Tensor | None = None,
    ) -> None:
        """Initialize the MaceForce object.

        Args:
            model (torch.nn.Module): The MACE neural network model.
            atomic_numbers (list[int]): Atomic numbers for the system.
            device (str | None): The device to run computations on ('cuda', 'cpu',
                or None for auto-detection).
            neighbor_list_fn (Callable): The neighbor list function to use.
            compute_forces (bool, optional): Whether to compute forces.
                Defaults to False.
            compute_stress (bool, optional): Whether to compute stress.
                Defaults to False.
            dtype (torch.dtype, optional): The data type for tensor operations.
                Defaults to torch.float32.
            enable_cueq (bool, optional): Whether to enable CuEq acceleration.
                Defaults to False.
        """
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.neighbor_list_fn = neighbor_list_fn

        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        if enable_cueq:
            print("Converting models to CuEq for acceleration")
            self.model = run_e3nn_to_cueq(self.model)

        # set model properties
        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = torch.tensor(
            self.model.atomic_numbers.detach().clone(), device=self.device
        )

        if atomic_numbers is not None:
            self.ptr, self.batch, self.node_attrs = self.compute_atomic_numbers(
                atomic_numbers, self.z_table, self.device
            )
            self.atomic_numbers_in_init = True
            self.atomic_number_tensor = torch.tensor(atomic_numbers, device=self.device)
        else:
            self.atomic_numbers_in_init = False
            self.atomic_number_tensor = None

        # compile model
        # TODO: fix jit compile error
        # self.model = jit.compile(self.model)

        # TODO: test the speed with torch.compile
        # self.model = torch.compile(self.model)

    @staticmethod
    def compute_atomic_numbers(
        atomic_numbers: list[int] | torch.Tensor,
        z_table: utils.AtomicNumberTable,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the atomic numbers for the system.

        Args:
            atomic_numbers (list[int] | torch.Tensor): The atomic numbers of the system.
            z_table (utils.AtomicNumberTable): The atomic number table.
            device (torch.device): The device to run the computation on.
            dtype (torch.dtype): The data type for tensor operations.
        """
        if isinstance(atomic_numbers, torch.Tensor):
            atomic_numbers = atomic_numbers.tolist()

        n_atoms = len(atomic_numbers)
        ptr = torch.tensor([0, n_atoms], dtype=torch.long, device=device)
        batch = torch.zeros(n_atoms, dtype=torch.long, device=device)

        node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(atomic_numbers, z_table=z_table),
                dtype=torch.long,
                device=device,
            ).unsqueeze(-1),
            num_classes=len(z_table),
            dtype=dtype,
        )
        return ptr, batch, node_attrs

    def forward(  # noqa: C901
        self,
        state: SimState | StateDict,
    ) -> dict[str, torch.Tensor]:
        """Compute the energy of the system given atomic positions and box vectors.

        This method calculates the neighbor list, prepares the input for the MACE
        model, and returns the computed energy.

        Args:
            state (SimState | StateDict): The state of the system.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed energy,
                forces, and stress of the system.
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        if state.batch is not None and state.batch.max() > 0:
            raise ValueError("UnbatchedMaceModel does not support batched systems.")

        if state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )

        if state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        if state.atomic_numbers is not None and not self.atomic_numbers_in_init:
            new_atomic_number_tensor = torch.tensor(
                state.atomic_numbers, device=self.device
            )
            if self.atomic_number_tensor is None or not torch.equal(
                new_atomic_number_tensor, self.atomic_number_tensor
            ):
                self.ptr, self.batch, self.node_attrs = self.compute_atomic_numbers(
                    new_atomic_number_tensor, self.z_table, self.device, self.dtype
                )
                self.atomic_number_tensor = new_atomic_number_tensor

        row_vector_cell = (
            state.row_vector_cell
        )  # MACE uses row vector cell convention for nbr list
        positions = state.positions
        pbc = state.pbc

        if row_vector_cell.dim() == 3:  # Check if there is an extra batch dimension
            row_vector_cell = row_vector_cell.squeeze(0)  # Squeeze the first dimension

        # calculate neighbor list
        edge_index, shifts_idx = self.neighbor_list_fn(
            positions=positions,
            cell=row_vector_cell,
            pbc=pbc,
            cutoff=self.r_max,
        )
        shifts = torch.mm(shifts_idx, row_vector_cell)

        # get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                node_attrs=self.node_attrs,
                batch=self.batch,
                pbc=pbc,
                cell=row_vector_cell,
                positions=positions,
                edge_index=edge_index,
                unit_shifts=shifts_idx,
                shifts=shifts,
            ),
            compute_force=self.compute_forces,
            compute_stress=self.compute_stress,
        )

        energy = out["energy"]

        results = {}

        if energy is not None:
            results["energy"] = energy
        else:
            results["energy"] = torch.tensor(0.0, device=self.device)

        if self.compute_forces:
            forces = out["forces"]
            results["forces"] = forces

        if self.compute_stress:
            stress = out["stress"].squeeze()
            results["stress"] = stress

        return results
