"""Lennard-Jones model for computing energies, forces and stresses."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.transforms import get_pair_displacements
from torch_sim.unbatched.unbatched_integrators import SimState, StateDict


# Default parameter values defined at module level
DEFAULT_SIGMA = torch.tensor(1.0)
DEFAULT_EPSILON = torch.tensor(1.0)


def lennard_jones_pair(
    dr: torch.Tensor,
    sigma: torch.Tensor = DEFAULT_SIGMA,
    epsilon: torch.Tensor = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Calculate pairwise Lennard-Jones interaction energies between particles.

    Implements the standard 12-6 Lennard-Jones potential that combines short-range
    repulsion with longer-range attraction. The potential has a minimum at r=sigma.

    The functional form is:
    V(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Distance at which potential reaches its minimum. Either a scalar float
            or tensor of shape [n, m] for particle-specific interaction distances.
        epsilon: Depth of the potential well (energy scale). Either a scalar float
            or tensor of shape [n, m] for pair-specific interaction strengths.

    Returns:
        Pairwise Lennard-Jones interaction energies between particles. Shape: [n, m].
        Each element [i,j] represents the interaction energy between particles i and j.
    """
    # Calculate inverse dr and its powers
    idr = sigma / dr
    idr2 = idr * idr
    idr6 = idr2 * idr2 * idr2
    idr12 = idr6 * idr6

    # Calculate potential energy
    energy = 4.0 * epsilon * (idr12 - idr6)

    # Handle potential numerical instabilities and infinities
    return torch.where(dr > 0, energy, torch.zeros_like(energy))
    # return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


def lennard_jones_pair_force(
    dr: torch.Tensor,
    sigma: torch.Tensor = DEFAULT_SIGMA,
    epsilon: torch.Tensor = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Calculate pairwise Lennard-Jones forces between particles.

    Implements the force derived from the 12-6 Lennard-Jones potential. The force
    is repulsive at short range and attractive at long range, with a zero-crossing
    at r=sigma.

    The functional form is:
    F(r) = 24*epsilon/r * [(2*sigma^12/r^12) - (sigma^6/r^6)]

    This is the negative gradient of the Lennard-Jones potential energy.

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Distance at which force changes from repulsive to attractive.
            Either a scalar float or tensor of shape [n, m] for particle-specific
            interaction distances.
        epsilon: Energy scale of the interaction. Either a scalar float or tensor
            of shape [n, m] for pair-specific interaction strengths.

    Returns:
        Pairwise Lennard-Jones forces between particles. Shape: [n, m].
        Each element [i,j] represents the force magnitude between particles i and j.
        Positive values indicate repulsion, negative values indicate attraction.
    """
    # Calculate inverse dr and its powers
    idr = sigma / dr
    idr2 = idr * idr
    idr6 = idr2 * idr2 * idr2
    idr12 = idr6 * idr6

    # Calculate force (negative gradient of potential)
    # F = -24*epsilon/r * ((sigma/r)^6 - 2*(sigma/r)^12)
    force = 24.0 * epsilon / dr * (2.0 * idr12 - idr6)

    # Handle potential numerical instabilities and infinities
    return torch.where(dr > 0, force, torch.zeros_like(force))
    # return torch.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)


class UnbatchedLennardJonesModel(torch.nn.Module, ModelInterface):
    """Calculator for Lennard-Jones potential.

    This model implements the Lennard-Jones potential energy and force calculator.
    It supports customizable interaction parameters for different particle pairs.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,  # Force keyword-only arguments
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        use_neighbor_list: bool = True,
        cutoff: float | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            sigma: The sigma parameter of the Lennard-Jones potential
            epsilon: The epsilon parameter of the Lennard-Jones potential
            device: The device to use for the calculation
            dtype: The data type to use for the calculation
            compute_forces: Whether to compute forces
            compute_stress: Whether to compute stresses
            per_atom_energies: Whether to compute per-atom energies
            per_atom_stresses: Whether to compute per-atom stresses
            use_neighbor_list: Whether to use a neighbor list
            cutoff: The cutoff radius for the Lennard-Jones potential
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
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self.device)
        self.cutoff = torch.tensor(cutoff or 2.5 * sigma, dtype=dtype, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self.device)

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces.

        Args:
            state: Either a SimState object or a dictionary containing positions,
                cell, pbc

        Returns:
            A dictionary containing the energy, forces, and stresses
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        cell = state.row_vector_cell
        positions = state.positions
        pbc = state.pbc

        if cell.dim() == 3:  # Check if there is an extra batch dimension
            cell = cell.squeeze(0)  # Squeeze the first dimension

        if self.use_neighbor_list:
            # Get neighbor list using vesin_nl_ts
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
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
            mapping = torch.stack([j, i])
            # Get valid displacements and distances
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Calculate pair energies and apply cutoff
        pair_energies = lennard_jones_pair(
            distances, sigma=self.sigma, epsilon=self.epsilon
        )
        # Zero out energies beyond cutoff
        mask = distances < self.cutoff
        pair_energies = torch.where(mask, pair_energies, torch.zeros_like(pair_energies))

        # Initialize results with total energy (sum/2 to avoid double counting)
        results = {"energy": 0.5 * pair_energies.sum()}

        if self._per_atom_energies:
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_forces or self.compute_stress:
            # Calculate forces and apply cutoff
            pair_forces = lennard_jones_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon
            )
            pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

            # Project forces along displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_forces:
                # Initialize forces tensor
                forces = torch.zeros_like(positions)
                # Add force contributions (f_ij on i, -f_ij on j)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces

            if self.compute_stress and cell is not None:
                # Compute stress tensor
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
