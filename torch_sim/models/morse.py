"""Morse potential model."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict
from torch_sim.transforms import get_pair_displacements
from torch_sim.unbatched.models.morse import morse_pair, morse_pair_force


class MorseModel(torch.nn.Module, ModelInterface):
    """Calculator for Morse potential."""

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 5.0,
        alpha: float = 5.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,  # Force keyword-only arguments
        periodic: bool = True,
        compute_force: bool = False,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        use_neighbor_list: bool = True,
        cutoff: float | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            sigma: Distance at which potential reaches its minimum
            epsilon: Depth of the potential well (energy scale)
            alpha: Controls the width of the potential well
            device: Torch device to use for calculations
            dtype: Data type for torch tensors
            periodic: Whether to use periodic boundary conditions
            compute_force: Whether to compute forces
            compute_stress: Whether to compute stress tensor
            per_atom_energies: Whether to return per-atom energies
            per_atom_stresses: Whether to return per-atom stress tensors
            use_neighbor_list: Whether to use neighbor lists for efficiency
            cutoff: Cutoff distance for interactions (default: 2.5*sigma)
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self._per_atom_energies = per_atom_energies
        self._per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list
        self.periodic = periodic
        # Convert parameters to tensors
        self.sigma = torch.tensor(sigma, dtype=self._dtype, device=self._device)
        self.cutoff = torch.tensor(
            cutoff or 2.5 * sigma, dtype=self._dtype, device=self._device
        )
        self.epsilon = torch.tensor(epsilon, dtype=self._dtype, device=self._device)
        self.alpha = torch.tensor(alpha, dtype=self._dtype, device=self._device)

    def unbatched_forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces.

        Args:
            state: State object containing positions, cell, and other properties

        Returns:
            Dictionary containing computed properties (energy, forces, stress, etc.)
        """
        if isinstance(state, dict):
            state = SimState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )

        positions = state.positions
        cell = state.cell
        cell = cell.squeeze()

        if self.use_neighbor_list:
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                cutoff=self.cutoff,
                sort_id=False,
            )
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                pairs=mapping,
                shifts=shifts,
            )
        else:
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
            )
            mask = torch.eye(positions.shape[0], dtype=torch.bool, device=self._device)
            distances = distances.masked_fill(mask, float("inf"))
            mask = distances < self.cutoff
            i, j = torch.where(mask)
            mapping = torch.stack([j, i])
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Calculate pair energies and apply cutoff
        pair_energies = morse_pair(
            distances, sigma=self.sigma, epsilon=self.epsilon, alpha=self.alpha
        )
        mask = distances < self.cutoff
        pair_energies = torch.where(mask, pair_energies, torch.zeros_like(pair_energies))

        # Initialize results with total energy (sum/2 to avoid double counting)
        results = {"energy": 0.5 * pair_energies.sum()}

        if self._per_atom_energies:
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self._dtype, device=self._device
            )
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_force or self.compute_stress:
            pair_forces = morse_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon, alpha=self.alpha
            )
            pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self._compute_force:
                forces = torch.zeros_like(state.positions)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces

            if self._compute_stress and state.cell is not None:
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(state.cell))

                results["stress"] = -stress_per_pair.sum(dim=0) / volume

                if self._per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (state.positions.shape[0], 3, 3),
                        dtype=self._dtype,
                        device=self._device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces."""
        if isinstance(state, dict):
            state = SimState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.periodic:
            raise ValueError("PBC mismatch between model and state")

        if state.batch is None and state.cell.shape[0] > 1:
            raise ValueError("Batch can only be inferred for batch size 1.")

        outputs = [self.unbatched_forward(state[i]) for i in range(state.n_batches)]
        properties = outputs[0]

        # we always return tensors
        # per atom properties are returned as (atoms, ...) tensors
        # global properties are returned as shape (..., n) tensors
        results = {}
        for key in ("stress", "energy"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])
        for key in ("forces",):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results
