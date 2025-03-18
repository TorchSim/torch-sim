"""Soft sphere model for computing energies, forces and stresses."""

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.state import BaseState, StateDict
from torch_sim.transforms import get_pair_displacements
from torch_sim.unbatched.models.soft_sphere import (
    soft_sphere_pair,
    soft_sphere_pair_force,
)


# Default parameter values defined at module level
DEFAULT_SIGMA = torch.tensor(1.0)
DEFAULT_EPSILON = torch.tensor(1.0)
DEFAULT_ALPHA = torch.tensor(2.0)


class SoftSphereModel(torch.nn.Module, ModelInterface):
    """Calculator for soft sphere potential."""

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        alpha: float = 2.0,
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
        """Initialize the soft sphere model."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self.periodic = periodic
        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list

        # Convert interaction parameters to tensors with proper dtype/device
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self.device)
        self.cutoff = torch.tensor(cutoff or sigma, dtype=dtype, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=self.device)

    def unbatched_forward(
        self,
        state: BaseState,
    ) -> dict[str, torch.Tensor]:
        """Compute energies and forces for a single system."""
        if isinstance(state, dict):
            state = BaseState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )

        positions = state.positions
        cell = state.cell
        cell = cell.squeeze()

        if self.use_neighbor_list:
            # Get neighbor list using vesin_nl_ts
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                cutoff=self.cutoff,
                sort_id=False,
            )
            # Get displacements between neighbor pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                pairs=mapping,
                shifts=shifts,
            )

        else:
            # Direct N^2 computation of all pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
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

        if self.per_atom_energies:
            # Compute per-atom energy contributions
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_force or self.compute_stress:
            # Calculate pair forces
            pair_forces = soft_sphere_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon, alpha=self.alpha
            )

            # Project scalar forces onto displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_force:
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

                if self.per_atom_stresses:
                    # Compute per-atom stress contributions
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3), dtype=self.dtype, device=self.device
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results

    def forward(
        self, state: BaseState | StateDict
    ) -> dict[str, torch.Tensor]:  # TODO: what are the shapes?
        """Compute energies and forces for batched systems.

        Args:
            state: State object

        Returns:
            Dictionary with computed properties:
            - energy: Energy for each system. Shape: [n_systems]
            - forces: Forces for all atoms. Shape: [total_atoms, 3]
            - stress: Stress tensor for each system. Shape: [n_systems, 3, 3]
        """
        if isinstance(state, dict):
            state = BaseState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.periodic:
            raise ValueError("PBC mismatch between model and state")

        # Handle batch indices if not provided
        if state.batch is None and state.cell.shape[0] > 1:
            raise ValueError("Batch can only be inferred for batch size 1.")

        outputs = [self.unbatched_forward(state[i]) for i in range(state.n_batches)]
        properties = outputs[0]

        # Combine results
        results = {}
        for key in ("stress", "energy"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])
        for key in ("forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results


class SoftSphereMultiModel(torch.nn.Module):
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
        periodic: bool = True,
        compute_force: bool = True,
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
            periodic: Whether to use periodic boundary conditions.
            compute_force: Whether to compute atomic forces.
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
        self.periodic = periodic
        self.compute_force = compute_force
        self.compute_stress = compute_stress
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
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
            * torch.ones((n_species, n_species), dtype=dtype, device=device)
        )
        self.epsilon_matrix = (
            epsilon_matrix
            if epsilon_matrix is not None
            else default_epsilon
            * torch.ones((n_species, n_species), dtype=dtype, device=device)
        )
        self.alpha_matrix = (
            alpha_matrix
            if alpha_matrix is not None
            else default_alpha
            * torch.ones((n_species, n_species), dtype=dtype, device=device)
        )

        # Ensure parameter matrices are symmetric (required for energy conservation)
        assert torch.allclose(self.sigma_matrix, self.sigma_matrix.T)
        assert torch.allclose(self.epsilon_matrix, self.epsilon_matrix.T)
        assert torch.allclose(self.alpha_matrix, self.alpha_matrix.T)

        # Set interaction cutoff distance
        self.cutoff = torch.tensor(
            cutoff or float(self.sigma_matrix.max()), dtype=dtype, device=device
        )

    def unbatched_forward(  # noqa: PLR0915
        self,
        state: BaseState,
        species: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces and stresses for the multi-species system.

        Args:
            state: State object
            species: Species labels/indices for each atom. If None, uses
                the species provided at initialization.

        Returns:
            Dictionary containing computed quantities:
            - 'energy': Total potential energy of the system
            - 'energies': Per-atom energies (if per_atom_energies=True)
            - 'forces': Atomic forces (if compute_force=True)
            - 'stress': Stress tensor (if compute_stress=True)
            - 'stresses': Per-atom stress tensors (if per_atom_stresses=True)
        """
        # Convert inputs to proper device/dtype and handle species
        if not isinstance(state, BaseState):
            state = BaseState(**state)

        if species is not None:
            species = species.to(device=self.device, dtype=torch.long)
        else:
            species = self.species

        positions = state.positions
        cell = state.cell
        cell = cell.squeeze()
        species_idx = species

        # Compute neighbor list or full distance matrix
        if self.use_neighbor_list:
            # Get neighbor list for efficient computation
            mapping, shifts = vesin_nl_ts(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                cutoff=self.cutoff,
                sorti=False,
            )
            # Get displacements between neighbor pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
                pairs=mapping,
                shifts=shifts,
            )

        else:
            # Direct N^2 computation of all pairs
            dr_vec, distances = get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.periodic,
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

        if self.per_atom_energies:
            # Compute per-atom energy contributions
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            # Each atom gets half of the pair energy
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_force or self.compute_stress:
            # Calculate pair forces
            pair_forces = soft_sphere_pair_force(
                distances, sigma=pair_sigmas, epsilon=pair_epsilons, alpha=pair_alphas
            )

            # Project scalar forces onto displacement vectors
            force_vectors = (pair_forces / distances)[:, None] * dr_vec

            if self.compute_force:
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

                if self.per_atom_stresses:
                    # Compute per-atom stress contributions
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3), dtype=self.dtype, device=self.device
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results

    def forward(self, state: BaseState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies and forces for batched systems.

        Args:
            state: State object

        Returns:
            Dictionary with computed properties:
            - energy: Energy for each system. Shape: [n_systems]
            - forces: Forces for all atoms. Shape: [total_atoms, 3]
            - stress: Stress tensor for each system. Shape: [n_systems, 3, 3]
        """
        if not isinstance(state, BaseState):
            state = BaseState(
                **state, pbc=self.periodic, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.periodic:
            raise ValueError("PBC mismatch between model and state")

        # Handle batch indices if not provided
        if state.batch is None and state.cell.shape[0] > 1:
            raise ValueError("Batch can only be inferred for batch size 1.")

        outputs = [self.unbatched_forward(state[i]) for i in range(state.n_batches)]
        properties = outputs[0]

        # Combine results
        results = {}
        for key in ("stress", "energy", "forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])

        for key in ("forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results
