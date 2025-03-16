"""MACE model calculator for computing energies, forces and stresses.

This module provides a PyTorch implementation of the MACE model calculator.
"""

from collections.abc import Callable

import torch
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import atomic_numbers_to_indices, to_one_hot, utils

from torch_sim.models.interface import ModelInterface
from torch_sim.state import BaseState, StateDict


class UnbatchedMaceModel(torch.nn.Module, ModelInterface):
    """Computes the energy of a system using a MACE model.

    Attributes:
        model (torch.nn.Module): The MACE model.
        device (str): The device (CPU or GPU) on which computations are performed.
        neighbor_list_fn (Callable): neighbor list function used for atom interactions.
        periodic (bool): Whether to use periodic boundary conditions.
        default_dtype (torch.dtype): The default data type for tensor operations.
        r_max (float): The maximum cutoff radius for atomic interactions.
        z_table (utils.AtomicNumberTable): Table for converting between
            atomic numbers and indices.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        neighbor_list_fn: Callable,
        *,
        device: torch.device | None = None,
        periodic: bool = True,
        compute_force: bool = False,
        compute_stress: bool = False,
        dtype: torch.dtype = torch.float32,
        enable_cueq: bool = False,
        atomic_numbers: list[int] | torch.Tensor | None = None,
    ) -> None:
        """Initialize the MaceForce object.

        Args:
            model (torch.nn.Module): The MACE neural network model.
            atomic_numbers (list[int]): List of atomic numbers for the system.
            device (str | None): The device to run computations on ('cuda', 'cpu',
                or None for auto-detection).
            neighbor_list_fn (Callable): The neighbor list function to use.
            periodic (bool, optional): Whether to use periodic boundary conditions.
                Defaults to True.
            compute_force (bool, optional): Whether to compute forces.
                Defaults to False.
            compute_stress (bool, optional): Whether to compute stress.
                Defaults to False.
            dtype (torch.dtype, optional): The data type for tensor operations.
                Defaults to torch.float32.
            enable_cueq (bool, optional): Whether to enable CuEq acceleration.
                Defaults to False.
        """
        super().__init__()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self.neighbor_list_fn = neighbor_list_fn
        self.periodic = periodic

        torch.set_default_dtype(self._dtype)

        print(f"Running MACEForce on device: {self._device} with dtype: {self._dtype} ")

        if enable_cueq:
            print("Converting models to CuEq for acceleration")
            self.model = run_e3nn_to_cueq(model, device=device).to(device)
        else:
            self.model = model

        self.model = self.model.to(dtype=self._dtype, device=self._device)
        self.model.eval()

        # set model properties
        self.r_max = self.model.r_max

        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = torch.tensor(
            self.model.atomic_numbers.clone(), device=self.device
        )

        # setup system boundary conditions
        pbc = [periodic] * 3
        self.pbc = torch.tensor([pbc], device=self.device)

        if atomic_numbers is not None:
            self.ptr, self.batch, self.node_attrs = self.compute_atomic_numbers(
                atomic_numbers, self.z_table, self.device
            )
            self.atomic_numbers_in_init = True
            self.atomic_number_tensor = torch.tensor(atomic_numbers, device=self.device)
        else:
            self.atomic_numbers_in_init = False
            self.atomic_number_tensor = None
        # TODO: reimplement this to avoid warning
        # self.model.atomic_numbers = self.model.atomic_numbers.clone().
        # detach().to(device=self.device)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the atomic numbers for the system.

        Args:
            atomic_numbers (list[int] | torch.Tensor): The atomic numbers of the system.
            z_table (utils.AtomicNumberTable): The atomic number table.
            device (torch.device): The device to run the computation on.
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
        )
        return ptr, batch, node_attrs

    def forward(  # noqa: C901
        self,
        state: BaseState | StateDict,
    ) -> dict[str, torch.Tensor]:
        """Compute the energy of the system given atomic positions and box vectors.

        This method calculates the neighbor list, prepares the input for the MACE
        model, and returns the computed energy.

        Args:
            state (BaseState | StateDict): The state of the system.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed energy,
                forces, and stress of the system.
        """
        if not isinstance(state, BaseState):
            state = BaseState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.periodic:
            raise ValueError("PBC mismatch between model and state")

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
                    new_atomic_number_tensor, self.z_table, self.device
                )
                self.atomic_number_tensor = new_atomic_number_tensor

        if state.cell.dim() == 3:  # Check if there is an extra batch dimension
            state.cell = state.cell.squeeze(0)  # Squeeze the first dimension

        # calculate neighbor list
        mapping, shifts_idx = self.neighbor_list_fn(
            positions=state.positions,
            cell=state.cell,
            pbc=self.periodic,
            cutoff=self.r_max,
        )
        edge_index = torch.stack((mapping[0], mapping[1]))
        shifts = torch.mm(shifts_idx, state.cell)

        # get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                node_attrs=self.node_attrs,
                batch=self.batch,
                pbc=self.pbc,
                cell=state.cell,
                positions=state.positions,
                edge_index=edge_index,
                unit_shifts=shifts_idx,
                shifts=shifts,
            ),
            compute_force=self._compute_force,
            compute_stress=self._compute_stress,
        )

        # num_atoms_arange = torch.arange(len(positions), device=self.device)
        # node_e0 = self.model.atomic_energies_fn(self.node_attrs)[num_atoms_arange]
        # energy = out["interaction_energy"] + node_e0.sum()

        # Don't use interaction energy
        # energy = out["interaction_energy"]

        energy = out["energy"]

        results = {}

        if energy is not None:
            results["energy"] = energy
        else:
            results["energy"] = torch.tensor(0.0, device=self.device)

        if self._compute_force:
            forces = out["forces"]
            results["forces"] = forces

        if self._compute_stress:
            stress = out["stress"].squeeze()
            results["stress"] = stress

        return results
