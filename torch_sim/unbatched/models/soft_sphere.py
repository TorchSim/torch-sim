"""Soft sphere model for computing energies, forces and stresses."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import SimState, StateDict
from torch_sim.transforms import get_pair_displacements, safe_mask


# Default parameter values defined at module level
DEFAULT_SIGMA = torch.tensor(1.0)
DEFAULT_EPSILON = torch.tensor(1.0)
DEFAULT_ALPHA = torch.tensor(2.0)


def soft_sphere_pair(
    dr: torch.Tensor,
    sigma: torch.Tensor = DEFAULT_SIGMA,
    epsilon: torch.Tensor = DEFAULT_EPSILON,
    alpha: torch.Tensor = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Calculate pairwise repulsive energies between soft spheres with finite-range
    interactions.

    Computes a soft-core repulsive potential between particle pairs based on
    their separation distance, size, and interaction parameters. The potential
    goes to zero at finite range.

    Args:
        dr: Pairwise distances between particles. Shape: [n, m].
        sigma: Particle diameters. Either a scalar float or tensor of shape [n, m]
            for particle-specific sizes.
        epsilon: Energy scale of the interaction. Either a scalar float or tensor
            of shape [n, m] for pair-specific interaction strengths.
        alpha: Stiffness exponent controlling the interaction decay. Either a scalar
            float or tensor of shape [n, m].

    Returns:
        Pairwise interaction energies between particles. Shape: [n, m]. Each element
        [i,j] represents the repulsive energy between particles i and j.
    """

    def fn(dr: torch.Tensor) -> torch.Tensor:
        return epsilon / alpha * (1.0 - (dr / sigma)).pow(alpha)

    # Create mask for distances within cutoff i.e sigma
    mask = dr < sigma

    # Use safe_mask to compute energies only where mask is True
    return safe_mask(mask, fn, dr)


def soft_sphere_pair_force(
    dr: torch.Tensor,
    sigma: torch.Tensor = DEFAULT_SIGMA,
    epsilon: torch.Tensor = DEFAULT_EPSILON,
    alpha: torch.Tensor = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Computes the pairwise repulsive forces between soft spheres with finite range.

    This function implements a soft-core repulsive interaction that smoothly goes to zero
    at the cutoff distance sigma. The force magnitude is controlled by epsilon and its
    stiffness by alpha.

    Args:
        dr: A tensor of shape [n, m] containing pairwise distances between particles,
            where n and m represent different particle indices.
        sigma: Particle diameter defining the interaction cutoff distance. Can be either
            a float scalar or a tensor of shape [n, m] for particle-specific diameters.
        epsilon: Energy scale of the interaction. Can be either a float scalar or a
            tensor of shape [n, m] for particle-specific interaction strengths.
        alpha: Exponent controlling the stiffness of the repulsion. Higher values create
            a harder repulsion. Can be either a float scalar or a tensor of shape [n, m].

    Returns:
        torch.Tensor: Forces between particle pairs with shape [n, m]. Forces are zero
        for distances greater than sigma.
    """

    def fn(dr: torch.Tensor) -> torch.Tensor:
        return (-epsilon / sigma) * (1.0 - (dr / sigma)).pow(alpha - 1)

    # Create mask for distances within cutoff i.e sigma
    mask = dr < sigma

    # Use safe_mask to compute energies only where mask is True
    return safe_mask(mask, fn, dr)


class UnbatchedSoftSphereModel(torch.nn.Module, ModelInterface):
    """Calculator for soft sphere potential.

    This model implements a soft sphere potential where the interaction
    parameters (sigma, epsilon, alpha) can be specified independently for each pair
    of atomic species. The potential creates repulsive forces between overlapping atoms
    with species-specific interaction strengths and ranges.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        alpha: float = 2.0,
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
        """Initialize the soft sphere model."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype

        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._per_atom_energies = per_atom_energies
        self._per_atom_stresses = per_atom_stresses

        self.use_neighbor_list = use_neighbor_list

        # Convert interaction parameters to tensors with proper dtype/device
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self.device)
        self.cutoff = torch.tensor(cutoff or sigma, dtype=dtype, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=self.device)

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces for a single system.

        Args:
            state: Either a SimState object or a dictionary containing positions,
                cell, pbc

        Returns:
            A dictionary containing the energy, forces, and stresses
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        positions = state.positions
        cell = state.row_vector_cell
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
            # Get displacements between neighbor pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
                pairs=mapping,
                shifts=shifts,
            )

        else:
            # Direct N^2 computation of all pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            # Remove self-interactions and apply cutoff
            mask = torch.eye(positions.shape[0], dtype=torch.bool, device=self.device)
            distances = distances.masked_fill(mask, float("inf"))
            mask = distances < self.cutoff

            # Get valid pairs and their displacements
            i, j = torch.where(mask)
            mapping = torch.stack([j, i])
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Calculate pair energies using soft sphere potential
        pair_energies = soft_sphere_pair(
            distances, sigma=self.sigma, epsilon=self.epsilon, alpha=self.alpha
        )

        # Initialize results with total energy (divide by 2 to avoid double counting)
        results = {"energy": 0.5 * pair_energies.sum()}

        if self._per_atom_energies:
            # Compute per-atom energy contributions
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_forces or self.compute_stress:
            # Calculate pair forces
            pair_forces = soft_sphere_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon, alpha=self.alpha
            )

            # Project scalar forces onto displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_forces:
                # Compute atomic forces by accumulating pair contributions
                forces = torch.zeros_like(positions)
                # Add force contributions (f_ij on j, -f_ij on i)
                forces.index_add_(0, mapping[0], force_vectors)
                forces.index_add_(0, mapping[1], -force_vectors)
                results["forces"] = forces

            if self.compute_stress and cell is not None:
                # Compute stress tensor using virial formula
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(cell))

                results["stress"] = -stress_per_pair.sum(dim=0) / volume

                if self._per_atom_stresses:
                    # Compute per-atom stress contributions
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results


# TODO: Standardize the interface for multi-species models


class UnbatchedSoftSphereMultiModel(torch.nn.Module, ModelInterface):
    """Calculator for soft sphere potential with multiple atomic species.

    This model implements a multi-species soft sphere potential where the interaction
    parameters (sigma, epsilon, alpha) can be specified independently for each pair
    of atomic species. The potential creates repulsive forces between overlapping atoms
    with species-specific interaction strengths and ranges.

    The total energy is computed as a sum over all atom pairs:
        E = sum_{i,j} E_{ij}(r_{ij})
    where E_{ij} is the pair potential between species i and j.

    The pair potential has the form:
        E_{ij}(r) = epsilon_{ij}/alpha_{ij} * (1 - r/sigma_{ij})^alpha_{ij}
        for r < sigma_{ij}, and 0 otherwise

    Forces are computed as the negative gradient of the energy with respect to atomic
    positions. The stress tensor is computed using the virial formula.
    """

    def __init__(
        self,
        species: torch.Tensor | None = None,
        sigma_matrix: torch.Tensor | None = None,
        epsilon_matrix: torch.Tensor | None = None,
        alpha_matrix: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,  # Force keyword-only arguments
        pbc: bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        use_neighbor_list: bool = True,
        cutoff: float | None = None,
    ) -> None:
        """Initialize the multi-species soft sphere calculator.

        Args:
            species: List of species labels/indices for each atom in the system.
                Used to look up interaction parameters.
            sigma_matrix: Matrix of interaction diameters for each species pair.
                Shape [n_species, n_species]. Must be symmetric.
            epsilon_matrix: Matrix of interaction strengths for each species pair.
                Shape [n_species, n_species]. Must be symmetric.
            alpha_matrix: Matrix of stiffness exponents for each species pair.
                Shape [n_species, n_species]. Must be symmetric.
            device: PyTorch device to use for calculations (CPU/GPU).
            dtype: PyTorch data type for numerical precision.
            pbc: Whether to use periodic boundary conditions.
            compute_forces: Whether to compute atomic forces.
            compute_stress: Whether to compute the stress tensor.
            per_atom_energies: Whether to compute per-atom energy contributions.
            per_atom_stresses: Whether to compute per-atom stress contributions.
            use_neighbor_list: Whether to use neighbor lists for efficiency.
            cutoff: Global cutoff distance for interactions. If None, uses
                maximum sigma value from sigma_matrix.
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.pbc = pbc
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._per_atom_energies = per_atom_energies
        self._per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list

        # Store species list and determine number of unique species
        self.species = species
        n_species = len(torch.unique(species))

        # Initialize parameter matrices with defaults if not provided
        default_sigma = DEFAULT_SIGMA.to(device=self.device, dtype=self.dtype)
        default_epsilon = DEFAULT_EPSILON.to(device=self.device, dtype=self.dtype)
        default_alpha = DEFAULT_ALPHA.to(device=self.device, dtype=self.dtype)

        # Validate matrix shapes match number of species
        if sigma_matrix is not None and sigma_matrix.shape != (n_species, n_species):
            raise ValueError(f"sigma_matrix must have shape ({n_species}, {n_species})")
        if epsilon_matrix is not None and epsilon_matrix.shape != (
            n_species,
            n_species,
        ):
            raise ValueError(f"epsilon_matrix must have shape ({n_species}, {n_species})")
        if alpha_matrix is not None and alpha_matrix.shape != (n_species, n_species):
            raise ValueError(f"alpha_matrix must have shape ({n_species}, {n_species})")

        # Create parameter matrices, using defaults if not provided
        self.sigma_matrix = (
            sigma_matrix
            if sigma_matrix is not None
            else default_sigma
            * torch.ones((n_species, n_species), dtype=self.dtype, device=self.device)
        )
        self.epsilon_matrix = (
            epsilon_matrix
            if epsilon_matrix is not None
            else default_epsilon
            * torch.ones((n_species, n_species), dtype=self.dtype, device=self.device)
        )
        self.alpha_matrix = (
            alpha_matrix
            if alpha_matrix is not None
            else default_alpha
            * torch.ones((n_species, n_species), dtype=self.dtype, device=self.device)
        )

        # Ensure parameter matrices are symmetric (required for energy conservation)
        assert torch.allclose(self.sigma_matrix, self.sigma_matrix.T)
        assert torch.allclose(self.epsilon_matrix, self.epsilon_matrix.T)
        assert torch.allclose(self.alpha_matrix, self.alpha_matrix.T)

        # Set interaction cutoff distance
        self.cutoff = torch.tensor(
            cutoff or float(self.sigma_matrix.max()),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces and stresses for the multi-species system.

        Args:
            positions: Atomic positions tensor of shape [n_atoms, 3].
            cell: Unit cell tensor of shape [3, 3] for periodic systems.
            species: Species labels/indices for each atom. If None, uses
                the species provided at initialization.

        Returns:
            Dictionary containing computed quantities:
            - 'energy': Total potential energy of the system
            - 'energies': Per-atom energies (if per_atom_energies=True)
            - 'forces': Atomic forces (if compute_forces=True)
            - 'stress': Stress tensor (if compute_stress=True)
            - 'stresses': Per-atom stress tensors (if per_atom_stresses=True)
        """
        # Convert inputs to proper device/dtype and handle species
        if cell is not None:
            cell = cell.to(device=self.device, dtype=self.dtype)

        if species is not None:
            species = species.to(device=self.device, dtype=torch.long)
        else:
            species = self.species

        species_idx = species

        # Compute neighbor list or full distance matrix
        if self.use_neighbor_list:
            # Get neighbor list for efficient computation
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=self.pbc,
                cutoff=self.cutoff,
                sorti=False,
            )
            # Get displacements between neighbor pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.pbc,
                pairs=mapping,
                shifts=shifts,
            )

        else:
            # Direct N^2 computation of all pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.pbc,
            )
            # Remove self-interactions and apply cutoff
            mask = torch.eye(positions.shape[0], dtype=torch.bool, device=self.device)
            distances = distances.masked_fill(mask, float("inf"))
            mask = distances < self.cutoff

            # Get valid pairs and their displacements
            i, j = torch.where(mask)
            mapping = torch.stack([j, i])
            dr_vec = dr_vec[mask]
            distances = distances[mask]

        # Look up species-specific parameters for each interacting pair
        pair_species_1 = species_idx[mapping[0]]  # Species of first atom in pair
        pair_species_2 = species_idx[mapping[1]]  # Species of second atom in pair

        # Get interaction parameters from parameter matrices
        pair_sigmas = self.sigma_matrix[pair_species_1, pair_species_2]
        pair_epsilons = self.epsilon_matrix[pair_species_1, pair_species_2]
        pair_alphas = self.alpha_matrix[pair_species_1, pair_species_2]

        # Calculate pair energies using species-specific parameters
        pair_energies = soft_sphere_pair(
            distances, sigma=pair_sigmas, epsilon=pair_epsilons, alpha=pair_alphas
        )

        # Initialize results with total energy (divide by 2 to avoid double counting)
        results = {"energy": 0.5 * pair_energies.sum()}

        if self._per_atom_energies:
            # Compute per-atom energy contributions
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_forces or self.compute_stress:
            # Calculate pair forces
            pair_forces = soft_sphere_pair_force(
                distances, sigma=pair_sigmas, epsilon=pair_epsilons, alpha=pair_alphas
            )

            # Project scalar forces onto displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_forces:
                # Compute atomic forces by accumulating pair contributions
                forces = torch.zeros_like(positions)
                # Add force contributions (f_ij on j, -f_ij on i)
                forces.index_add_(0, mapping[0], force_vectors)
                forces.index_add_(0, mapping[1], -force_vectors)
                results["forces"] = forces

            if self.compute_stress and cell is not None:
                # Compute stress tensor using virial formula
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volume = torch.abs(torch.linalg.det(cell))

                results["stress"] = -stress_per_pair.sum(dim=0) / volume

                if self._per_atom_stresses:
                    # Compute per-atom stress contributions
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results
