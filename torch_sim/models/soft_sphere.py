"""Soft sphere model for computing energies, forces and stresses.

This module provides implementations of soft sphere potentials for molecular dynamics
simulations. Soft sphere potentials are repulsive interatomic potentials that model
the core repulsion between atoms, avoiding the infinite repulsion of hard sphere models
while maintaining computational efficiency.

The soft sphere potential has the form:
    V(r) = epsilon * (sigma/r)^alpha

Where:

* r is the distance between particles
* sigma is the effective diameter of the particles
* epsilon controls the energy scale
* alpha determines the steepness of the repulsion (typically alpha >= 2)

Soft sphere models are particularly useful for:

* Granular matter simulations
* Modeling excluded volume effects
* Initial equilibration of dense systems
* Coarse-grained molecular dynamics

Example::

    # Create a soft sphere model with default parameters
    model = SoftSphereModel()

    # Calculate properties for a simulation state
    results = model(sim_state)
    energy = results["energy"]
    forces = results["forces"]

    # For multiple species with different interaction parameters
    multi_model = SoftSphereMultiModel(
        species=particle_types,
        sigma_matrix=size_matrix,
        epsilon_matrix=strength_matrix,
    )
    results = multi_model(sim_state)
"""

import torch

import torch_sim as ts
from torch_sim import transforms
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.typing import StateDict


DEFAULT_SIGMA = torch.tensor(1.0)
DEFAULT_EPSILON = torch.tensor(1.0)
DEFAULT_ALPHA = torch.tensor(2.0)


def soft_sphere_pair(
    dr: torch.Tensor,
    sigma: float | torch.Tensor = DEFAULT_SIGMA,
    epsilon: float | torch.Tensor = DEFAULT_EPSILON,
    alpha: float | torch.Tensor = DEFAULT_ALPHA,
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
        torch.Tensor: Pairwise interaction energies between particles. Shape: [n, m].
            Each element [i,j] represents the repulsive energy between particles i and j.
    """

    def fn(dr: torch.Tensor) -> torch.Tensor:
        return epsilon / alpha * (1.0 - (dr / sigma)).pow(alpha)

    # Create mask for distances within cutoff i.e sigma
    mask = dr < sigma

    # Use transforms.safe_mask to compute energies only where mask is True
    return transforms.safe_mask(mask, fn, dr)


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

    # Use transforms.safe_mask to compute energies only where mask is True
    return transforms.safe_mask(mask, fn, dr)


class SoftSphereModel(ModelInterface):
    """Calculator for soft sphere potential energies and forces.

    Implements a model for computing properties based on the soft sphere potential,
    which describes purely repulsive interactions between particles. This potential
    is useful for modeling systems where particles should not overlap but don't have
    attractive interactions, such as granular materials and some colloidal systems.

    The potential energy between particles i and j is:
        V_ij(r) = epsilon * (sigma/r)^alpha

    Attributes:
        sigma (torch.Tensor): Effective particle diameter in distance units.
        epsilon (torch.Tensor): Energy scale parameter in energy units.
        alpha (torch.Tensor): Exponent controlling repulsion steepness (typically ≥ 2).
        cutoff (torch.Tensor): Cutoff distance for interactions.
        use_neighbor_list (bool): Whether to use neighbor list optimization.
        _device (torch.device): Computation device (CPU/GPU).
        _dtype (torch.dtype): Data type for tensor calculations.
        _compute_forces (bool): Whether to compute forces.
        _compute_stress (bool): Whether to compute stress tensor.
        per_atom_energies (bool): Whether to compute per-atom energy decomposition.
        per_atom_stresses (bool): Whether to compute per-atom stress decomposition.

    Examples:
        ```py
        # Basic usage with default parameters
        model = SoftSphereModel()
        results = model(sim_state)

        # Custom parameters for colloidal system
        colloid_model = SoftSphereModel(
            sigma=2.0,  # particle diameter in nm
            epsilon=10.0,  # energy scale in kJ/mol
            alpha=12.0,  # steep repulsion for hard colloids
            compute_stress=True,
        )

        # Get forces for a system with periodic boundary conditions
        results = colloid_model(
            ts.SimState(
                positions=positions,
                cell=box_vectors,
                pbc=torch.tensor([True, True, True]),
            )
        )
        forces = results["forces"]  # shape: [n_particles, 3]
        ```
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
        """Initialize the soft sphere model.

        Creates a soft sphere model with specified parameters for particle interactions
        and computation options.

        Args:
            sigma (float): Effective particle diameter. Determines the distance
                scale of the interaction. Defaults to 1.0.
            epsilon (float): Energy scale parameter. Controls the strength of
                the repulsion. Defaults to 1.0.
            alpha (float): Exponent controlling repulsion steepness. Higher values
                create steeper, more hard-sphere-like repulsion. Defaults to 2.0.
            device (torch.device | None): Device for computations. If None, uses CPU.
                Defaults to None.
            dtype (torch.dtype): Data type for calculations. Defaults to torch.float32.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress tensor. Defaults to False.
            per_atom_energies (bool): Whether to compute per-atom energy decomposition.
                Defaults to False.
            per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
                Defaults to False.
            use_neighbor_list (bool): Whether to use a neighbor list for optimization.
                Significantly faster for large systems. Defaults to True.
            cutoff (float | None): Cutoff distance for interactions. If None, uses
                the value of sigma. Defaults to None.

        Examples:
            ```py
            # Default model
            model = SoftSphereModel()

            # WCA-like repulsive potential (derived from Lennard-Jones)
            wca_model = SoftSphereModel(
                sigma=1.0,
                epsilon=1.0,
                alpha=12.0,  # Steep repulsion similar to r^-12 term in LJ
                cutoff=2 ** (1 / 6),  # WCA cutoff at minimum of LJ potential
            )
            ```
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
        self.use_neighbor_list = use_neighbor_list

        # Convert interaction parameters to tensors with proper dtype/device
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self.device)
        self.cutoff = torch.tensor(cutoff or sigma, dtype=dtype, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self.device)
        self.alpha = torch.tensor(alpha, dtype=dtype, device=self.device)

    def unbatched_forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        """Compute energies and forces for a single unbatched system.

        Internal implementation that processes a single, non-batched simulation state.
        This method handles the core computations for pair interactions, including
        neighbor list construction, distance calculations, and property computation.

        Args:
            state (SimState): Single, non-batched simulation state containing atomic
                positions, cell vectors, and other system information.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Total potential energy (scalar)
                - "forces": Atomic forces with shape [n_atoms, 3] (if
                    compute_forces=True)
                - "stress": Stress tensor with shape [3, 3] (if compute_stress=True)
                - "energies": Per-atom energies with shape [n_atoms] (if
                    per_atom_energies=True)
                - "stresses": Per-atom stresses with shape [n_atoms, 3, 3] (if
                    per_atom_stresses=True)

        Notes:
            This method can work with both neighbor list and full pairwise calculations.
            The soft sphere potential is purely repulsive, and forces are truncated at
            the cutoff distance.
        """
        if isinstance(state, dict):
            state = ts.SimState(**state, masses=torch.ones_like(state["positions"]))

        positions = state.positions
        cell = state.row_vector_cell
        cell = cell.squeeze()
        pbc = state.pbc

        if self.use_neighbor_list:
            # Get neighbor list using vesin_nl_ts
            mapping, shifts = torchsim_nl(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                sort_id=False,
            )
            # Get displacements between neighbor pairs
            dr_vec, distances = transforms.get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=pbc,
                pairs=mapping,
                shifts=shifts,
            )

        else:
            # Direct N^2 computation of all pairs
            dr_vec, distances = transforms.get_pair_displacements(
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

        if self.per_atom_energies:
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

                if self.per_atom_stresses:
                    # Compute per-atom stress contributions
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3), dtype=self.dtype, device=self.device
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute soft sphere potential energies, forces, and stresses for a system.

        Main entry point for soft sphere potential calculations that handles batched
        states by dispatching each system to the unbatched implementation and combining
        results.

        Args:
            state (SimState | StateDict): Input state containing atomic positions,
                cell vectors, and other system information. Can be a SimState object
                or a dictionary with the same keys.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Potential energy with shape [n_systems]
                - "forces": Atomic forces with shape [n_atoms, 3]
                    (if compute_forces=True)
                - "stress": Stress tensor with shape [n_systems, 3, 3]
                    (if compute_stress=True)
                - May include additional outputs based on configuration

        Raises:
            ValueError: If system indices cannot be inferred for multi-cell systems.

        Examples:
            ```py
            # Compute properties for a simulation state
            model = SoftSphereModel(compute_forces=True)
            results = model(sim_state)

            energy = results["energy"]  # Shape: [n_systems]
            forces = results["forces"]  # Shape: [n_atoms, 3]
            ```
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # Handle System indices if not provided
        if sim_state.system_idx is None and sim_state.cell.shape[0] > 1:
            raise ValueError(
                "system_idx can only be inferred if there is only one system"
            )

        outputs = [
            self.unbatched_forward(sim_state[i]) for i in range(sim_state.n_systems)
        ]
        properties = outputs[0]

        # Combine results
        results: dict[str, torch.Tensor] = {}
        for key in ("stress", "energy"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])
        for key in ("forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results


class SoftSphereMultiModel(ModelInterface):
    """Calculator for systems with multiple particle types.

    Extends the basic soft sphere model to support multiple particle types with
    different interaction parameters for each pair of particle types. This enables
    simulation of heterogeneous systems like mixtures, composites, or biomolecular
    systems with different interaction strengths between different components.

    This model maintains matrices of interaction parameters (sigma, epsilon, alpha)
    where each element [i,j] represents the parameter for interactions between
    particle types i and j.

    Attributes:
        species (torch.Tensor): Particle type indices for each particle in the system.
        sigma_matrix (torch.Tensor): Matrix of distance parameters for each pair of types.
            Shape: [n_types, n_types].
        epsilon_matrix (torch.Tensor): Matrix of energy scale parameters for each pair.
            Shape: [n_types, n_types].
        alpha_matrix (torch.Tensor): Matrix of exponents for each pair of types.
            Shape: [n_types, n_types].
        cutoff (torch.Tensor): Maximum interaction distance.
        compute_forces (bool): Whether to compute forces.
        compute_stress (bool): Whether to compute stress tensor.
        per_atom_energies (bool): Whether to compute per-atom energy decomposition.
        per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
        use_neighbor_list (bool): Whether to use neighbor list optimization.
        periodic (bool): Whether to use periodic boundary conditions.
        _device (torch.device): Computation device (CPU/GPU).
        _dtype (torch.dtype): Data type for tensor calculations.

    Examples:
        ```py
        # Create a binary mixture with different interaction parameters
        # Define interaction matrices (size 2x2 for binary system)
        sigma_matrix = torch.tensor(
            [
                [1.0, 0.8],  # Type 0-0 and 0-1 interactions
                [0.8, 0.6],  # Type 1-0 and 1-1 interactions
            ]
        )

        epsilon_matrix = torch.tensor(
            [
                [1.0, 0.5],  # Type 0-0 and 0-1 interactions
                [0.5, 2.0],  # Type 1-0 and 1-1 interactions
            ]
        )

        # Particle type assignments (0 or 1 for each particle)
        species = torch.tensor([0, 0, 1, 1, 0, 1])

        # Create the model
        model = SoftSphereMultiModel(
            species=species,
            sigma_matrix=sigma_matrix,
            epsilon_matrix=epsilon_matrix,
            compute_forces=True,
        )

        # Compute properties
        results = model(simulation_state)
        ```
    """

    def __init__(
        self,
        species: torch.Tensor | None = None,
        sigma_matrix: torch.Tensor | None = None,
        epsilon_matrix: torch.Tensor | None = None,
        alpha_matrix: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,  # Force keyword-only arguments
        pbc: torch.Tensor | bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        use_neighbor_list: bool = True,
        cutoff: float | None = None,
    ) -> None:
        """Initialize a soft sphere model for multi-component systems.

        Creates a model for systems with multiple particle types, each with potentially
        different interaction parameters.

        Args:
            species (torch.Tensor | None): Particle type indices, shape [n_particles].
                Each value should be an integer in range [0, n_types-1]. If None,
                assumes all particles are the same type (0). Defaults to None.
            sigma_matrix (torch.Tensor | None): Matrix of distance parameters for
                each pair of types. Shape [n_types, n_types]. If None, uses default
                value 1.0 for all pairs. Defaults to None.
            epsilon_matrix (torch.Tensor | None): Matrix of energy scale parameters
                for each pair of types. Shape [n_types, n_types]. If None, uses
                default value 1.0 for all pairs. Defaults to None.
            alpha_matrix (torch.Tensor | None): Matrix of exponents for each pair.
                Shape [n_types, n_types]. If None, uses default value 2.0 for all
                pairs. Defaults to None.
            device (torch.device | None): Device for computations. If None, uses CPU.
                Defaults to None.
            dtype (torch.dtype): Data type for calculations. Defaults to torch.float32.
            pbc (torch.Tensor | bool): Boolean tensor of shape (3,) indicating periodic
                boundary conditions in each axis. If None, all axes are assumed to be
                periodic. Defaults to True.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress tensor. Defaults to False.
            per_atom_energies (bool): Whether to compute per-atom energy decomposition.
                Defaults to False.
            per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
                Defaults to False.
            use_neighbor_list (bool): Whether to use a neighbor list for optimization.
                Defaults to True.
            cutoff (float | None): Cutoff distance for interactions. If None, uses
                the maximum value from sigma_matrix. Defaults to None.

        Examples:
            ```py
            # Binary polymer mixture with different interactions
            # Polymer A (type 0): larger, softer particles
            # Polymer B (type 1): smaller, harder particles

            # Create species assignment (100 particles total)
            species = torch.cat(
                [
                    torch.zeros(50, dtype=torch.long),  # 50 particles of type 0
                    torch.ones(50, dtype=torch.long),  # 50 particles of type 1
                ]
            )

            # Interaction matrices
            sigma = torch.tensor(
                [
                    [1.2, 1.0],  # A-A and A-B interactions
                    [1.0, 0.8],  # B-A and B-B interactions
                ]
            )

            epsilon = torch.tensor(
                [
                    [1.0, 1.5],  # A-A and A-B interactions
                    [1.5, 2.0],  # B-A and B-B interactions
                ]
            )

            # Create model with mixing rules
            model = SoftSphereMultiModel(
                species=species,
                sigma_matrix=sigma,
                epsilon_matrix=epsilon,
                compute_forces=True,
            )
            ```

        Notes:
            The interaction matrices must be symmetric for physical consistency
            (e.g., interaction of type 0 with type 1 should be the same as type 1
            with type 0).
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self.pbc = torch.tensor([pbc] * 3) if isinstance(pbc, bool) else pbc
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
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
        for matrix_name in ("sigma_matrix", "epsilon_matrix", "alpha_matrix"):
            matrix = getattr(self, matrix_name)
            if not torch.allclose(matrix, matrix.T):
                raise ValueError(f"{matrix_name} is not symmetric")

        # Set interaction cutoff distance
        self.cutoff = torch.tensor(
            cutoff or float(self.sigma_matrix.max()), dtype=dtype, device=device
        )

    def unbatched_forward(  # noqa: PLR0915
        self,
        state: ts.SimState,
        species: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute energies and forces for a single unbatched system with multiple
        species.

        Internal implementation that processes a single, non-batched simulation state.
        This method handles all pair interactions between particles of different types
        using the appropriate interaction parameters from the parameter matrices.

        Args:
            state (SimState): Single, non-batched simulation state containing atomic
                positions, cell vectors, and other system information.
            species (torch.Tensor | None): Optional species indices to override the
                ones provided during initialization. Shape: [n_particles]. If None,
                uses the species defined at initialization. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Total potential energy (scalar)
                - "forces": Atomic forces with shape [n_atoms, 3]
                    (if compute_forces=True)
                - "stress": Stress tensor with shape [3, 3]
                    (if compute_stress=True)
                - "energies": Per-atom energies with shape [n_atoms]
                    (if per_atom_energies=True)
                - "stresses": Per-atom stresses with shape [n_atoms, 3, 3]
                    (if per_atom_stresses=True)

        Notes:
            This method supports both neighbor list optimization and full pairwise
            calculations based on the use_neighbor_list parameter. For each pair of
            particles, it looks up the appropriate parameters based on the species
            of the two particles.
        """
        # Convert inputs to proper device/dtype and handle species
        if not isinstance(state, ts.SimState):
            state = ts.SimState(**state)

        if species is not None:
            species = species.to(device=self.device, dtype=torch.long)
        else:
            species = self.species

        positions = state.positions
        cell = state.row_vector_cell
        cell = cell.squeeze()
        species_idx = species

        # Compute neighbor list or full distance matrix
        if self.use_neighbor_list:
            # Get neighbor list for efficient computation
            mapping, shifts = torchsim_nl(
                positions=positions,
                cell=cell,
                pbc=self.pbc,
                cutoff=self.cutoff,
                sort_id=False,
            )
            # Get displacements between neighbor pairs
            dr_vec, distances = transforms.get_pair_displacements(
                positions=positions,
                cell=cell,
                pbc=self.pbc,
                pairs=mapping,
                shifts=shifts,
            )

        else:
            # Direct N^2 computation of all pairs
            dr_vec, distances = transforms.get_pair_displacements(
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

        if self.per_atom_energies:
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

                if self.per_atom_stresses:
                    # Compute per-atom stress contributions
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3), dtype=self.dtype, device=self.device
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute soft sphere potential properties for multi-component systems.

        Main entry point for multi-species soft sphere calculations that handles
        batched states by dispatching each batch to the unbatched implementation
        and combining results.

        Args:
            state (SimState | StateDict): Input state containing atomic positions,
                cell vectors, and other system information. Can be a SimState object
                or a dictionary with the same keys.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - "energy": Potential energy with shape [n_systems]
                - "forces": Atomic forces with shape [n_atoms, 3]
                    (if compute_forces=True)
                - "stress": Stress tensor with shape [n_systems, 3, 3]
                    (if compute_stress=True)
                - May include additional outputs based on configuration

        Raises:
            ValueError: If batch cannot be inferred for multi-cell systems or if
                species information is missing.

        Examples:
            ```py
            # Create model for binary mixture
            model = SoftSphereMultiModel(
                species=particle_types,
                sigma_matrix=distance_matrix,
                epsilon_matrix=strength_matrix,
                compute_forces=True,
            )

            # Calculate properties
            results = model(simulation_state)
            energy = results["energy"]
            forces = results["forces"]
            ```

        Notes:
            This method requires species information either provided during initialization
            or included in the state object's metadata.
        """
        if not isinstance(state, ts.SimState):
            state = ts.SimState(
                **state, pbc=self.pbc, masses=torch.ones_like(state["positions"])
            )
        elif state.pbc != self.pbc:
            raise ValueError("PBC mismatch between model and state")

        # Handle system indices if not provided
        if state.system_idx is None and state.cell.shape[0] > 1:
            raise ValueError(
                "system_idx can only be inferred if there is only one system"
            )

        outputs = [
            self.unbatched_forward(state[sys_idx]) for sys_idx in range(state.n_systems)
        ]
        properties = outputs[0]

        # Combine results
        results: dict[str, torch.Tensor] = {}
        for key in ("stress", "energy", "forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.stack([out[key] for out in outputs])

        for key in ("forces", "energies", "stresses"):
            if key in properties:
                results[key] = torch.cat([out[key] for out in outputs], dim=0)

        return results
