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
    """Calculator for Lennard-Jones potential."""

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,  # Force keyword-only arguments
        periodic: bool = True,
        compute_force: bool = True,
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

        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self._per_atom_energies = per_atom_energies
        self._per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list
        self.periodic = periodic

        # Convert parameters to tensors
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self._device)
        self.cutoff = torch.tensor(
            cutoff or 2.5 * sigma, dtype=dtype, device=self._device
        )
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self._device)

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces."""
        if isinstance(state, dict):
            state = SimState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.periodic:
            raise ValueError("PBC mismatch between model and state")

        cell = state.cell
        positions = state.positions

        if cell.dim() == 3:  # Check if there is an extra batch dimension
            cell = cell.squeeze(0)  # Squeeze the first dimension

        if self.use_neighbor_list:
            # Get neighbor list using vesin_nl_ts
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                cutoff=self.cutoff,
                sort_id=False,
            )
            # Get displacements using neighbor list
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                pairs=mapping,
                shifts=shifts,
            )
        else:
            # Get all pairwise displacements
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
            )
            # Mask out self-interactions
            mask = torch.eye(positions.shape[0], dtype=torch.bool, device=self._device)
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
                positions.shape[0], dtype=self._dtype, device=self._device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self._compute_force or self._compute_stress:
            # Calculate forces and apply cutoff
            pair_forces = lennard_jones_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon
            )
            pair_forces = torch.where(mask, pair_forces, torch.zeros_like(pair_forces))

            # Project forces along displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self._compute_force:
                # Initialize forces tensor
                forces = torch.zeros_like(positions)
                # Add force contributions (f_ij on i, -f_ij on j)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces

            if self._compute_stress and cell is not None:
                # Compute stress tensor
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(cell))

                results["stress"] = -stress_per_pair.sum(dim=0) / volume

                if self._per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3),
                        dtype=self._dtype,
                        device=self._device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results
