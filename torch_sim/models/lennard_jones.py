"""Classical pairwise interatomic potential model.

This module implements the Lennard-Jones potential for molecular dynamics simulations.
It provides efficient calculation of energies, forces, and stresses based on the
classic 12-6 potential function. The implementation supports both full pairwise
calculations and neighbor list-based optimizations.

Example::

    # Create a Lennard-Jones model with default parameters
    model = LennardJonesModel(device=torch.device("cuda"))

    # Create a model with custom parameters
    model = LennardJonesModel(
        sigma=3.405,  # Angstroms
        epsilon=0.01032,  # eV
        cutoff=10.0,  # Angstroms
        compute_stress=True,
    )

    # Calculate properties for a simulation state
    output = model(sim_state)
    energy = output["energy"]
    forces = output["forces"]
"""

from collections.abc import Callable

import torch

import torch_sim as ts
from torch_sim import transforms
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl


DEFAULT_SIGMA = 1.0
DEFAULT_EPSILON = 1.0


def lennard_jones_pair(
    dr: torch.Tensor,
    sigma: float | torch.Tensor = DEFAULT_SIGMA,
    epsilon: float | torch.Tensor = DEFAULT_EPSILON,
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
        torch.Tensor: Pairwise Lennard-Jones interaction energies between particles.
            Shape: [n, m]. Each element [i,j] represents the interaction energy between
            particles i and j.
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
    sigma: float | torch.Tensor = DEFAULT_SIGMA,
    epsilon: float | torch.Tensor = DEFAULT_EPSILON,
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
        torch.Tensor: Pairwise Lennard-Jones forces between particles. Shape: [n, m].
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


class UnbatchedLennardJonesModel(ModelInterface):
    """Unbatched Lennard-Jones model.

    Implements the Lennard-Jones 12-6 potential for molecular dynamics simulations.
    This implementation loops over systems in batched inputs and is intended for
    testing or baseline comparisons with the default batched model.

    Attributes:
        sigma (torch.Tensor): Length parameter controlling particle size/repulsion
            distance.
        epsilon (torch.Tensor): Energy parameter controlling interaction strength.
        cutoff (torch.Tensor): Distance cutoff for truncating potential calculation.
        device (torch.device): Device where calculations are performed.
        dtype (torch.dtype): Data type used for calculations.
        compute_forces (bool): Whether to compute atomic forces.
        compute_stress (bool): Whether to compute stress tensor.
        per_atom_energies (bool): Whether to compute per-atom energy decomposition.
        per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
        neighbor_list_fn (Callable): Function used to construct neighbor lists.

    Example::

        # Basic usage with default parameters
        lj_model = UnbatchedLennardJonesModel(device=torch.device("cuda"))
        results = lj_model(sim_state)

        # Custom parameterization for Argon
        ar_model = UnbatchedLennardJonesModel(
            sigma=3.405,  # Å
            epsilon=0.0104,  # eV
            cutoff=8.5,  # Å
            compute_stress=True,
        )
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
        neighbor_list_fn: Callable = torchsim_nl,
        cutoff: float | None = None,
    ) -> None:
        """Initialize the Lennard-Jones potential calculator.

        Creates a model with specified interaction parameters and computational flags.
        The model can be configured to compute different properties (forces, stresses)
        and use different optimization strategies.

        Args:
            sigma (float): Length parameter of the Lennard-Jones potential in distance
                units. Controls the size of particles. Defaults to 1.0.
            epsilon (float): Energy parameter of the Lennard-Jones potential in energy
                units. Controls the strength of the interaction. Defaults to 1.0.
            device (torch.device | None): Device to run computations on. If None, uses
                CPU. Defaults to None.
            dtype (torch.dtype): Data type for calculations. Defaults to torch.float32.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress tensor. Defaults to False.
            per_atom_energies (bool): Whether to compute per-atom energy decomposition.
                Defaults to False.
            per_atom_stresses (bool): Whether to compute per-atom stress decomposition.
                Defaults to False.
            neighbor_list_fn (Callable): Batched neighbor-list function to use when
                constructing interactions. Defaults to torchsim_nl.
            cutoff (float | None): Cutoff distance for interactions in distance units.
                If None, uses 2.5*sigma. Defaults to None.

        Example::

            # Model with custom parameters
            model = UnbatchedLennardJonesModel(
                sigma=3.405,
                epsilon=0.01032,
                device=torch.device("cuda"),
                dtype=torch.float64,
                compute_stress=True,
                per_atom_energies=True,
                cutoff=10.0,
            )
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
        self.neighbor_list_fn = neighbor_list_fn

        # Convert parameters to tensors
        self.sigma = torch.tensor(sigma, dtype=dtype, device=self.device)
        self.cutoff = torch.tensor(cutoff or 2.5 * sigma, dtype=dtype, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=dtype, device=self.device)

    def unbatched_forward(
        self,
        state: ts.SimState,
    ) -> dict[str, torch.Tensor]:
        """Compute Lennard-Jones properties for a single unbatched system.

        Internal implementation that processes a single, non-batched simulation state.
        This method handles the core computations of pair interactions, neighbor lists,
        and property calculations.

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
            Neighbor lists are always used to construct interacting pairs.
        """
        positions = state.positions
        cell = state.row_vector_cell
        cell = cell.squeeze()

        # Ensure system_idx exists (create if None for single system)
        system_idx = (
            state.system_idx
            if state.system_idx is not None
            else torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        )

        # Wrap positions into the unit cell
        wrapped_positions = (
            ts.transforms.pbc_wrap_batched(positions, state.cell, system_idx, state.pbc)
            if state.pbc.any()
            else positions
        )

        mapping, _, shifts_idx = self.neighbor_list_fn(
            positions=wrapped_positions,
            cell=cell,
            pbc=state.pbc,
            cutoff=self.cutoff,
            system_idx=system_idx,
        )
        # Pass shifts_idx directly - get_pair_displacements will convert them
        dr_vec, distances = transforms.get_pair_displacements(
            positions=wrapped_positions,
            cell=cell,
            pbc=state.pbc,
            pairs=(mapping[0], mapping[1]),
            shifts=shifts_idx,
        )

        # Calculate pair energies and apply cutoff
        pair_energies = lennard_jones_pair(
            distances, sigma=self.sigma, epsilon=self.epsilon
        )
        # Zero out energies beyond cutoff
        mask = distances < self.cutoff
        pair_energies = torch.where(mask, pair_energies, torch.zeros_like(pair_energies))

        # Initialize results with total energy (sum/2 to avoid double counting)
        results = {"energy": 0.5 * pair_energies.sum()}

        if self.per_atom_energies:
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

                if self.per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (state.positions.shape[0], 3, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    results["stresses"] = atom_stresses / volume

        return results

    def forward(self, state: ts.SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute Lennard-Jones energies, forces, and stresses for a system.

        Main entry point for Lennard-Jones calculations that handles batched states by
        dispatching each system to the unbatched implementation and combining results.

        Args:
            state (SimState): Input state containing atomic positions, cell vectors,
                and other system information.
            **_kwargs: Unused; accepted for interface compatibility.

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
            ValueError: If system cannot be inferred for multi-cell systems.

        Example::

            # Compute properties for a simulation state
            model = UnbatchedLennardJonesModel(compute_stress=True)
            results = model(sim_state)

            energy = results["energy"]  # Shape: [n_systems]
            forces = results["forces"]  # Shape: [n_atoms, 3]
            stress = results["stress"]  # Shape: [n_systems, 3, 3]
            energies = results["energies"]  # Shape: [n_atoms]
            stresses = results["stresses"]  # Shape: [n_atoms, 3, 3]
        """
        sim_state = state

        if sim_state.system_idx is None and sim_state.cell.shape[0] > 1:
            raise ValueError("System can only be inferred for batch size 1.")

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


class LennardJonesModel(UnbatchedLennardJonesModel):
    """Default vectorized Lennard-Jones model for batched systems.

    This class computes Lennard-Jones energies, forces, and stresses for all systems in
    a batch in one pass, avoiding Python loops over systems in the model forward path.
    Use this class for production runs.
    """

    def forward(  # noqa: PLR0915
        self, state: ts.SimState, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Compute Lennard-Jones properties with batched tensor operations."""
        sim_state = state

        if sim_state.system_idx is None and sim_state.cell.shape[0] > 1:
            raise ValueError("System can only be inferred for batch size 1.")

        positions = sim_state.positions
        row_cell = sim_state.row_vector_cell
        pbc = sim_state.pbc

        system_idx = (
            sim_state.system_idx
            if sim_state.system_idx is not None
            else torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
        )

        wrapped_positions = (
            ts.transforms.pbc_wrap_batched(positions, sim_state.cell, system_idx, pbc)
            if pbc.any()
            else positions
        )

        if pbc.ndim == 1:
            pbc_batched = pbc.unsqueeze(0).expand(sim_state.n_systems, -1)
        else:
            pbc_batched = pbc

        mapping, system_mapping, shifts_idx = self.neighbor_list_fn(
            positions=wrapped_positions,
            cell=row_cell,
            pbc=pbc_batched,
            cutoff=self.cutoff,
            system_idx=system_idx,
        )

        cell_shifts = transforms.compute_cell_shifts(row_cell, shifts_idx, system_mapping)
        dr_vec = (
            wrapped_positions[mapping[1]] - wrapped_positions[mapping[0]] + cell_shifts
        )
        distances = dr_vec.norm(dim=1)

        cutoff_mask = distances < self.cutoff
        pair_energies = lennard_jones_pair(
            distances, sigma=self.sigma, epsilon=self.epsilon
        )
        pair_energies = torch.where(
            cutoff_mask, pair_energies, torch.zeros_like(pair_energies)
        )

        n_systems = sim_state.n_systems
        results: dict[str, torch.Tensor] = {}
        energy = torch.zeros(n_systems, dtype=self.dtype, device=self.device)
        energy.index_add_(0, system_mapping, 0.5 * pair_energies)
        results["energy"] = energy

        if self.per_atom_energies:
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self.dtype, device=self.device
            )
            atom_energies.index_add_(0, mapping[0], 0.5 * pair_energies)
            atom_energies.index_add_(0, mapping[1], 0.5 * pair_energies)
            results["energies"] = atom_energies

        if self.compute_forces or self.compute_stress:
            pair_forces = lennard_jones_pair_force(
                distances, sigma=self.sigma, epsilon=self.epsilon
            )
            pair_forces = torch.where(
                cutoff_mask, pair_forces, torch.zeros_like(pair_forces)
            )
            safe_distances = torch.where(
                distances > 0, distances, torch.ones_like(distances)
            )
            force_vectors = (pair_forces / safe_distances)[:, None] * dr_vec

            if self.compute_forces:
                forces = torch.zeros_like(positions)
                forces.index_add_(0, mapping[0], -force_vectors)
                forces.index_add_(0, mapping[1], force_vectors)
                results["forces"] = forces

            if self.compute_stress:
                stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
                volumes = torch.abs(torch.linalg.det(row_cell))
                stress = torch.zeros(
                    (n_systems, 3, 3),
                    dtype=self.dtype,
                    device=self.device,
                )
                stress.index_add_(0, system_mapping, -stress_per_pair)
                results["stress"] = stress / volumes[:, None, None]

                if self.per_atom_stresses:
                    atom_stresses = torch.zeros(
                        (positions.shape[0], 3, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    atom_stresses.index_add_(0, mapping[0], -0.5 * stress_per_pair)
                    atom_stresses.index_add_(0, mapping[1], -0.5 * stress_per_pair)
                    atom_volumes = volumes[system_idx]
                    results["stresses"] = atom_stresses / atom_volumes[:, None, None]

        return results
