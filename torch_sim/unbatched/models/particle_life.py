"""Particle life model for computing forces between particles."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState
from torch_sim.transforms import get_pair_displacements


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
        Tensor of energies with shape [n, m].
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
        Tensor of energies with shape [n, m].
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


class UnbatchedParticleLifeModel(torch.nn.Module, ModelInterface):
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

    def forward(self, state: SimState) -> dict[str, torch.Tensor]:
        """Compute energies and forces.

        Args:
            state: Either a SimState object or a dictionary containing positions,
                cell, pbc

        Returns:
            A dictionary containing the energy, forces, and stresses
        """
        # Extract required data from input
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        positions = state.positions
        cell = state.row_vector_cell
        pbc = state.pbc

        if cell.dim() == 3:  # Check if there is an extra batch dimension
            cell = cell.squeeze(0)  # Squeeze the first dimension

        if self.use_neighbor_list:
            # Get neighbor list using wrapping_nl
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=float(self.cutoff),
                sort_id=False,
            )
            # Get displacements using neighbor list
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
                pairs=mapping,
                shifts=shifts,
            )
        else:
            # Get all pairwise displacements
            dr_vec, distances = get_pair_displacements(
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
            mapping = torch.stack([j, i])  # Changed from [j, i] to [i, j]
            # Get valid displacements and distances
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Zero out energies beyond cutoff
        mask = distances < self.cutoff

        # Initialize results with total energy (sum/2 to avoid double counting)
        results = {"energy": 0.0}

        # Calculate forces and apply cutoff
        pair_forces = asymmetric_particle_pair_force_jit(
            distances, sigma=self.sigma, epsilon=self.epsilon
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
