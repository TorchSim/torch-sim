"""Morse model for computing energies, forces and stresses."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict
from torch_sim.transforms import get_pair_displacements


# Default parameter values defined at module level
DEFAULT_SIGMA = torch.tensor(1.0)
DEFAULT_EPSILON = torch.tensor(5.0)
DEFAULT_ALPHA = torch.tensor(5.0)


def morse_pair(
    dr: torch.Tensor,
    sigma: torch.Tensor = DEFAULT_SIGMA,
    epsilon: torch.Tensor = DEFAULT_EPSILON,
    alpha: torch.Tensor = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Calculate pairwise Morse potential energies between particles.

    Implements the Morse potential that combines short-range repulsion with
    longer-range attraction. The potential has a minimum at r=sigma and approaches
    -epsilon as r→∞.

    The functional form is:
    V(r) = epsilon * (1 - exp(-alpha*(r-sigma)))^2 - epsilon

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Distance at which potential reaches its minimum. Either a scalar float
            or tensor of shape [n, m] for particle-specific equilibrium distances.
        epsilon: Depth of the potential well (energy scale). Either a scalar float
            or tensor of shape [n, m] for pair-specific interaction strengths.
        alpha: Controls the width of the potential well. Larger values give a narrower
            well. Either a scalar float or tensor of shape [n, m].

    Returns:
        Pairwise Morse interaction energies between particles. Shape: [n, m].
        Each element [i,j] represents the interaction energy between particles i and j.
    """
    # Calculate potential energy
    energy = epsilon * (1.0 - torch.exp(-alpha * (dr - sigma))).pow(2) - epsilon

    # Handle potential numerical instabilities
    return torch.where(dr > 0, energy, torch.zeros_like(energy))
    # return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


def morse_pair_force(
    dr: torch.Tensor,
    sigma: torch.Tensor = DEFAULT_SIGMA,
    epsilon: torch.Tensor = DEFAULT_EPSILON,
    alpha: torch.Tensor = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Calculate pairwise Morse forces between particles.

    Implements the force derived from the Morse potential. The force changes
    from repulsive to attractive at r=sigma.

    The functional form is:
    F(r) = 2*alpha*epsilon * exp(-alpha*(r-sigma)) * (1 - exp(-alpha*(r-sigma)))

    This is the negative gradient of the Morse potential energy.

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Distance at which force changes from repulsive to attractive.
            Either a scalar float or tensor of shape [n, m].
        epsilon: Energy scale of the interaction. Either a scalar float or tensor
            of shape [n, m].
        alpha: Controls the force range and stiffness. Either a scalar float or
            tensor of shape [n, m].

    Returns:
        Pairwise Morse forces between particles. Shape: [n, m].
        Positive values indicate repulsion, negative values indicate attraction.
    """
    exp_term = torch.exp(-alpha * (dr - sigma))
    force = -2.0 * alpha * epsilon * exp_term * (1.0 - exp_term)

    # Handle potential numerical instabilities
    return torch.where(dr > 0, force, torch.zeros_like(force))
    # return torch.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)


class UnbatchedMorseModel(torch.nn.Module, ModelInterface):
    """Calculator for Morse potential.

    This model implements the Morse potential energy and force calculator.
    It supports customizable interaction parameters for different particle pairs.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 5.0,
        alpha: float = 5.0,
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
        """Initialize the calculator.

        Args:
            sigma: Distance at which potential reaches its minimum
            epsilon: Depth of the potential well (energy scale)
            alpha: Controls the width of the potential well
            device: Torch device to use for calculations
            dtype: Data type for torch tensors
            compute_forces: Whether to compute forces
            compute_stress: Whether to compute stress tensor
            per_atom_energies: Whether to return per-atom energies
            per_atom_stresses: Whether to return per-atom stress tensors
            use_neighbor_list: Whether to use neighbor lists for efficiency
            cutoff: Cutoff distance for interactions (default: 2.5*sigma)
        """
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
        self.alpha = torch.tensor(alpha, dtype=self.dtype, device=self.device)

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces.

        Args:
            state: State object containing positions, cell, and other properties

        Returns:
            Dictionary containing computed properties (energy, forces, stress, etc.)
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        positions = state.positions
        cell = state.row_vector_cell
        pbc = state.pbc

        if cell.dim() == 3:  # Check if there is an extra batch dimension
            cell = cell.squeeze(0)  # Squeeze the first dimension

        if self.use_neighbor_list:
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                sort_id=False,
            )
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
                pairs=mapping,
                shifts=shifts,
            )
        else:
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            mask = torch.eye(positions.shape[0], dtype=torch.bool, device=self.device)
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
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_forces or self.compute_stress:
            pair_forces = morse_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon, alpha=self.alpha
            )
            pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_forces:
                forces = torch.zeros_like(positions)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces

            if self.compute_stress and cell is not None:
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(cell))

                results["stress"] = -stress_per_pair.sum(dim=0) / volume

                if self._per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results
