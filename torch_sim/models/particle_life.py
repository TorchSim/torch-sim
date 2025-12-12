"""Particle life model for computing forces between particles."""

import torch

import torch_sim as ts
from torch_sim import transforms
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.typing import StateDict


DEFAULT_BETA = torch.tensor(0.3)
DEFAULT_SIGMA = torch.tensor(1.0)


def asymmetric_particle_pair_force(
    dr: torch.Tensor,
    A: torch.Tensor,
    beta: torch.Tensor = DEFAULT_BETA,
    sigma: torch.Tensor = DEFAULT_SIGMA,
) -> torch.Tensor:
    """Asymmetric interaction between particles.

    Args:
        dr: A tensor of shape [n, m] of pairwise distances between particles.
        A: Interaction scale. Either a float scalar or a tensor of shape [n, m].
        beta: Inner radius of the interaction. Either a float scalar or tensor of
            shape [n, m].
        sigma: Outer radius of the interaction. Either a float scalar or tensor of
            shape [n, m].

    Returns:
        torch.Tensor: Energies with shape [n, m].
    """
    inner_mask = dr < beta
    outer_mask = (dr < sigma) & (dr > beta)

    def inner_force_fn(dr: torch.Tensor) -> torch.Tensor:
        return dr / beta - 1

    def intermediate_force_fn(dr: torch.Tensor) -> torch.Tensor:
        return A * (1 - torch.abs(2 * dr - 1 - beta) / (1 - beta))

    return torch.where(inner_mask, inner_force_fn(dr), 0) + torch.where(
        outer_mask,
        intermediate_force_fn(dr),
        0,
    )


def asymmetric_particle_pair_force_jit(
    dr: torch.Tensor,
    A: torch.Tensor,
    beta: torch.Tensor = DEFAULT_BETA,
    sigma: torch.Tensor = DEFAULT_SIGMA,
) -> torch.Tensor:
    """Asymmetric interaction between particles.

    Args:
        dr: A tensor of shape [n, m] of pairwise distances between particles.
        A: Interaction scale. Either a float scalar or a tensor of shape [n, m].
        beta: Inner radius of the interaction. Either a float scalar or tensor of
            shape [n, m].
        sigma: Outer radius of the interaction. Either a float scalar or tensor of
            shape [n, m].

    Returns:
        torch.Tensor: Energies with shape [n, m].
    """
    inner_mask = dr < beta
    outer_mask = (dr < sigma) & (dr > beta)

    # Calculate inner forces directly
    inner_forces = torch.where(inner_mask, dr / beta - 1, torch.zeros_like(dr))

    # Calculate outer forces directly
    outer_forces = torch.where(
        outer_mask,
        A * (1 - torch.abs(2 * dr - 1 - beta) / (1 - beta)),
        torch.zeros_like(dr),
    )

    return inner_forces + outer_forces


class ParticleLifeModel(ModelInterface):
    """Calculator for asymmetric particle interaction.

    This model implements an asymmetric interaction between particles based on
    distance-dependent forces. The interaction is defined by three parameters:
    sigma, epsilon, and beta.

    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,  # Force keyword-only arguments
        compute_forces: bool = False,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        use_neighbor_list: bool = True,
        cutoff: float | None = None,
    ) -> None:
        """Initialize the calculator."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._per_atom_energies = per_atom_energies
        self._per_atom_stresses = per_atom_stresses

        self.use_neighbor_list = use_neighbor_list

        # Convert parameters to tensors
        self.sigma = torch.tensor(sigma, dtype=self.dtype, device=self.device)
        self.cutoff = torch.tensor(
            cutoff or 2.5 * sigma, dtype=self.dtype, device=self.device
        )
        self.epsilon = torch.tensor(epsilon, dtype=self.dtype, device=self.device)

    def unbatched_forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        """Compute energies and forces for a single unbatched system.

        Internal implementation that processes a single, non-batched simulation state.
        This method handles the core computations of pair interactions, neighbor lists,
        and property calculations.

        Args:
            state: Single, non-batched simulation state containing atomic positions,
                cell vectors, and other system information.

        Returns:
            A dictionary containing the energy, forces, and stresses
        """
        if isinstance(state, dict):
            state = ts.SimState(**state, masses=torch.ones_like(state["positions"]))

        positions = state.positions
        cell = state.row_vector_cell
        pbc = state.pbc

        if cell.dim() == 3:  # Check if there is an extra batch dimension
            cell = cell.squeeze(0)  # Squeeze the first dimension

        if self.use_neighbor_list:
            # Get neighbor list using torchsim_nl
            # Ensure system_idx exists (create if None for single system)
            system_idx = (
                state.system_idx
                if state.system_idx is not None
                else torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
            )
            mapping, system_mapping, shifts_idx = torchsim_nl(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                system_idx=system_idx,
            )
            # Pass shifts_idx directly - get_pair_displacements will convert them
            dr_vec, distances = transforms.get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
                pairs=(mapping[0], mapping[1]),
                shifts=shifts_idx,
            )
        else:
            # Get all pairwise displacements
            dr_vec, distances = transforms.get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            # Mask out self-interactions
            mask = torch.eye(positions.shape[0], dtype=torch.bool, device=self.device)
            distances = distances.masked_fill(mask, float("inf"))
            # Apply cutoff
            mask = distances < self.cutoff
            # Get valid pairs - match neighbor list convention for pair order
            i, j = torch.where(mask)
            mapping = torch.stack([j, i])
            # Get valid displacements and distances
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Zero out energies beyond cutoff
        mask = distances < self.cutoff

        # Initialize results with total energy (sum/2 to avoid double counting)
        results = {"energy": 0.0}

        # Calculate forces and apply cutoff
        pair_forces = asymmetric_particle_pair_force_jit(
            dr=distances, A=self.epsilon, sigma=self.sigma, beta=self.beta
        )
        pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

        # Project forces along displacement vectors
        force_vectors = (pair_forces / distances)[:, None] * dr_vec

        # Initialize forces tensor
        forces = torch.zeros_like(state.positions)
        # Add force contributions (f_ij on i, -f_ij on j)
        forces.index_add_(0, mapping[0], -force_vectors)
        forces.index_add_(0, mapping[1], force_vectors)
        results["forces"] = forces

        return results

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute particle life energies and forces for a system.

        Main entry point for particle life calculations that handles batched states by
        dispatching each batch to the unbatched implementation and combining results.

        Args:
            state: Input state containing atomic positions, cell vectors, and other
                system information. Can be a SimState object or a dictionary with the
                same keys.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Potential energy with shape [n_systems]
                - "forces": Atomic forces with shape [n_atoms, 3] (if
                    compute_forces=True)
                - "stress": Stress tensor with shape [n_systems, 3, 3] (if
                    compute_stress=True)
                - "energies": Per-atom energies with shape [n_atoms] (if
                    per_atom_energies=True)
                - "stresses": Per-atom stresses with shape [n_atoms, 3, 3] (if
                    per_atom_stresses=True)

        Raises:
            ValueError: If batch cannot be inferred for multi-cell systems.
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        if sim_state.system_idx is None and sim_state.cell.shape[0] > 1:
            raise ValueError(
                "system_idx can only be inferred if there is only one system."
            )

        outputs = [
            self.unbatched_forward(sim_state[idx]) for idx in range(sim_state.n_systems)
        ]
        properties = outputs[0]

        # we always return tensors
        # per atom properties are returned as (atoms, ...) tensors
        # global properties are returned as shape (..., n) tensors
        results: dict[str, torch.Tensor] = {}
        for key in ("stress", "energy"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])
        for key in ("forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results
