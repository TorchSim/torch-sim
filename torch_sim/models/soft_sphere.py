"""Soft sphere potential model.

Thin wrapper around :class:`~torch_sim.models.pair_potential.PairPotentialModel` with
the :func:`soft_sphere_pair` energy function baked in.

The soft sphere potential has the form:

    V(r) = ε/α * (1 - r/σ)^α  for r < σ,  else 0

Example::

    model = SoftSphereModel(sigma=1.0, epsilon=1.0, alpha=2.0)
    results = model(sim_state)

    # For multiple species with different interaction parameters
    multi_model = SoftSphereMultiModel(
        atomic_numbers=torch.tensor([18, 36]),
        sigma_matrix=torch.tensor([[1.0, 0.8], [0.8, 0.6]]),
        epsilon_matrix=torch.tensor([[1.0, 0.5], [0.5, 2.0]]),
    )
    results = multi_model(sim_state)
"""

from __future__ import annotations

import functools
from collections.abc import Callable  # noqa: TC003

import torch

from torch_sim.models.pair_potential import PairPotentialModel
from torch_sim.neighbors import torchsim_nl


def soft_sphere_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: torch.Tensor | float = 1.0,
    epsilon: torch.Tensor | float = 1.0,
    alpha: torch.Tensor | float = 2.0,
) -> torch.Tensor:
    """Soft-sphere repulsive pair energy (zero beyond sigma).

    V(r) = ε/α * (1 - r/σ)^α  for r < σ,  else 0

    Args:
        dr: Pairwise distances, shape [n_pairs].
        zi: Atomic numbers of first atoms (unused).
        zj: Atomic numbers of second atoms (unused).
        sigma: Interaction diameter / cutoff. Defaults to 1.0.
        epsilon: Energy scale. Defaults to 1.0.
        alpha: Repulsion exponent. Defaults to 2.0.

    Returns:
        Pair energies, shape [n_pairs].
    """
    energy = epsilon / alpha * (1.0 - dr / sigma).pow(alpha)
    return torch.where(dr < sigma, energy, torch.zeros_like(energy))


def soft_sphere_pair_force(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: torch.Tensor | float = 1.0,
    epsilon: torch.Tensor | float = 1.0,
    alpha: torch.Tensor | float = 2.0,
) -> torch.Tensor:
    """Soft-sphere pair force (negative gradient of energy).

    F(r) = (ε/σ) (1 - r/σ)^(α-1)  for r < σ,  else 0

    Args:
        dr: Pairwise distances.
        zi: Atomic numbers of first atoms (unused).
        zj: Atomic numbers of second atoms (unused).
        sigma: Interaction diameter. Defaults to 1.0.
        epsilon: Energy scale. Defaults to 1.0.
        alpha: Repulsion exponent. Defaults to 2.0.

    Returns:
        Pair force magnitudes.
    """
    force = (epsilon / sigma) * (1.0 - (dr / sigma)).pow(alpha - 1)
    mask = dr < sigma
    return torch.where(mask, force, torch.zeros_like(force))


class MultiSoftSpherePairFn(torch.nn.Module):
    """Species-dependent soft-sphere pair energy function.

    Holds per-species-pair parameter matrices and looks up sigma, epsilon, and alpha
    for each interacting pair via their atomic numbers.  Pass an instance to
    :class:`PairPotentialModel`.

    Example::

        fn = MultiSoftSpherePairFn(
            atomic_numbers=torch.tensor([18, 36]),  # Ar and Kr
            sigma_matrix=torch.tensor([[3.4, 3.6], [3.6, 3.7]]),
            epsilon_matrix=torch.tensor([[0.01, 0.012], [0.012, 0.014]]),
        )
        model = PairPotentialModel(pair_fn=fn, cutoff=float(fn.sigma_matrix.max()))
    """

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        sigma_matrix: torch.Tensor,
        epsilon_matrix: torch.Tensor,
        alpha_matrix: torch.Tensor | None = None,
    ) -> None:
        """Initialize species-dependent soft-sphere parameters.

        Args:
            atomic_numbers: 1-D tensor of the unique atomic numbers present, used to
                map ``zi``/``zj`` to row/column indices. Shape: [n_species].
            sigma_matrix: Symmetric matrix of interaction diameters. Shape:
                [n_species, n_species].
            epsilon_matrix: Symmetric matrix of energy scales. Shape:
                [n_species, n_species].
            alpha_matrix: Symmetric matrix of repulsion exponents. If None, defaults
                to 2.0 for all pairs. Shape: [n_species, n_species].
        """
        super().__init__()
        n = len(atomic_numbers)
        if sigma_matrix.shape != (n, n):
            raise ValueError(f"sigma_matrix must have shape ({n}, {n})")
        if epsilon_matrix.shape != (n, n):
            raise ValueError(f"epsilon_matrix must have shape ({n}, {n})")
        if alpha_matrix is not None and alpha_matrix.shape != (n, n):
            raise ValueError(f"alpha_matrix must have shape ({n}, {n})")

        self.register_buffer("atomic_numbers", atomic_numbers)
        self.sigma_matrix = sigma_matrix
        self.epsilon_matrix = epsilon_matrix
        self.alpha_matrix = (
            alpha_matrix if alpha_matrix is not None else torch.full((n, n), 2.0)
        )
        max_z = int(atomic_numbers.max().item()) + 1
        z_to_idx = torch.full((max_z,), -1, dtype=torch.long)
        for idx, z in enumerate(atomic_numbers.tolist()):
            z_to_idx[int(z)] = idx
        self.z_to_idx: torch.Tensor
        self.register_buffer("z_to_idx", z_to_idx)

    def forward(
        self, dr: torch.Tensor, zi: torch.Tensor, zj: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-pair soft-sphere energies using species lookup.

        Args:
            dr: Pairwise distances, shape [n_pairs].
            zi: Atomic numbers of first atoms, shape [n_pairs].
            zj: Atomic numbers of second atoms, shape [n_pairs].

        Returns:
            Pair energies, shape [n_pairs].
        """
        idx_i = self.z_to_idx[zi]
        idx_j = self.z_to_idx[zj]
        sigma = self.sigma_matrix[idx_i, idx_j]
        epsilon = self.epsilon_matrix[idx_i, idx_j]
        alpha = self.alpha_matrix[idx_i, idx_j]
        energy = epsilon / alpha * (1.0 - dr / sigma).pow(alpha)
        return torch.where(dr < sigma, energy, torch.zeros_like(energy))


DEFAULT_SIGMA = torch.tensor(1.0)
DEFAULT_EPSILON = torch.tensor(1.0)
DEFAULT_ALPHA = torch.tensor(2.0)


class SoftSphereModel(PairPotentialModel):
    """Soft-sphere repulsive pair potential model.

    Convenience subclass that fixes the pair function to :func:`soft_sphere_pair`
    so the caller only needs to supply ``sigma``, ``epsilon``, and ``alpha``.

    Example::

        model = SoftSphereModel(
            sigma=3.405,
            epsilon=0.0104,
            alpha=2.0,
            compute_forces=True,
        )
        results = model(sim_state)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        alpha: float = 2.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        neighbor_list_fn: Callable = torchsim_nl,
        use_neighbor_list: bool = True,  # noqa: ARG002
        cutoff: float | None = None,
        retain_graph: bool = False,
    ) -> None:
        """Initialize the soft sphere model.

        Args:
            sigma: Effective particle diameter. Defaults to 1.0.
            epsilon: Energy scale parameter. Defaults to 1.0.
            alpha: Repulsion exponent. Defaults to 2.0.
            device: Device for computations. Defaults to CPU.
            dtype: Floating-point dtype. Defaults to torch.float32.
            compute_forces: Whether to compute atomic forces. Defaults to True.
            compute_stress: Whether to compute the stress tensor. Defaults to False.
            per_atom_energies: Whether to return per-atom energies. Defaults to False.
            per_atom_stresses: Whether to return per-atom stresses. Defaults to False.
            neighbor_list_fn: Neighbor-list constructor. Defaults to torchsim_nl.
            use_neighbor_list: Accepted for backward compatibility (ignored).
            cutoff: Interaction cutoff. Defaults to sigma.
            retain_graph: Keep computation graph for differentiable simulation.
        """
        self.sigma = sigma
        self.epsilon = epsilon
        self.alpha = alpha

        pair_fn = functools.partial(
            soft_sphere_pair, sigma=sigma, epsilon=epsilon, alpha=alpha
        )
        super().__init__(
            pair_fn=pair_fn,
            cutoff=cutoff if cutoff is not None else sigma,
            device=device,
            dtype=dtype,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            per_atom_energies=per_atom_energies,
            per_atom_stresses=per_atom_stresses,
            neighbor_list_fn=neighbor_list_fn,
            reduce_to_half_list=True,
            retain_graph=retain_graph,
        )


class SoftSphereMultiModel(PairPotentialModel):
    """Multi-species soft-sphere potential model.

    Uses :class:`MultiSoftSpherePairFn` internally
    to look up per-species-pair parameters from matrices.

    Example::

        model = SoftSphereMultiModel(
            atomic_numbers=torch.tensor([18, 36]),
            sigma_matrix=torch.tensor([[1.0, 0.8], [0.8, 0.6]]),
            epsilon_matrix=torch.tensor([[1.0, 0.5], [0.5, 2.0]]),
            compute_forces=True,
        )
        results = model(sim_state)
    """

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        sigma_matrix: torch.Tensor | None = None,
        epsilon_matrix: torch.Tensor | None = None,
        alpha_matrix: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        pbc: torch.Tensor | bool = True,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        use_neighbor_list: bool = True,
        neighbor_list_fn: Callable = torchsim_nl,
        cutoff: float | None = None,
        retain_graph: bool = False,
    ) -> None:
        """Initialize the multi-species soft sphere model.

        Args:
            atomic_numbers: Atomic numbers of atoms in the system. May contain
                duplicates; only the sorted unique values are used to define
                species and determine matrix dimensions.
            sigma_matrix: Symmetric matrix of interaction diameters.
                Shape [n_species, n_species]. Defaults to 1.0 for all pairs.
            epsilon_matrix: Symmetric matrix of energy scales.
                Shape [n_species, n_species]. Defaults to 1.0 for all pairs.
            alpha_matrix: Symmetric matrix of repulsion exponents.
                Shape [n_species, n_species]. Defaults to 2.0 for all pairs.
            device: Device for computations. Defaults to CPU.
            dtype: Floating-point dtype. Defaults to torch.float64.
            pbc: Periodic boundary conditions (kept for backward compat). Defaults
                to True.
            compute_forces: Whether to compute atomic forces. Defaults to True.
            compute_stress: Whether to compute the stress tensor. Defaults to False.
            per_atom_energies: Whether to return per-atom energies. Defaults to False.
            per_atom_stresses: Whether to return per-atom stresses. Defaults to False.
            use_neighbor_list: Accepted for backward compatibility (a neighbor list
                is always used internally). Defaults to True.
            neighbor_list_fn: Neighbor-list constructor. Defaults to torchsim_nl.
            cutoff: Interaction cutoff. Defaults to max of sigma_matrix.
            retain_graph: Keep computation graph for differentiable simulation.
        """
        self.pbc = torch.tensor([pbc] * 3) if isinstance(pbc, bool) else pbc
        self.use_neighbor_list = use_neighbor_list

        unique_z = torch.unique(atomic_numbers).sort().values.long()
        n_species = len(unique_z)
        self.n_species = n_species

        _device = device or torch.device("cpu")
        default_sigma = DEFAULT_SIGMA.to(device=_device, dtype=dtype)
        default_epsilon = DEFAULT_EPSILON.to(device=_device, dtype=dtype)
        default_alpha = DEFAULT_ALPHA.to(device=_device, dtype=dtype)

        if sigma_matrix is not None and sigma_matrix.shape != (n_species, n_species):
            raise ValueError(f"sigma_matrix must have shape ({n_species}, {n_species})")
        if epsilon_matrix is not None and epsilon_matrix.shape != (
            n_species,
            n_species,
        ):
            raise ValueError(f"epsilon_matrix must have shape ({n_species}, {n_species})")
        if alpha_matrix is not None and alpha_matrix.shape != (n_species, n_species):
            raise ValueError(f"alpha_matrix must have shape ({n_species}, {n_species})")

        self.sigma_matrix = (
            sigma_matrix
            if sigma_matrix is not None
            else default_sigma
            * torch.ones((n_species, n_species), dtype=dtype, device=_device)
        )
        self.epsilon_matrix = (
            epsilon_matrix
            if epsilon_matrix is not None
            else default_epsilon
            * torch.ones((n_species, n_species), dtype=dtype, device=_device)
        )
        self.alpha_matrix = (
            alpha_matrix
            if alpha_matrix is not None
            else default_alpha
            * torch.ones((n_species, n_species), dtype=dtype, device=_device)
        )

        for matrix_name in ("sigma_matrix", "epsilon_matrix", "alpha_matrix"):
            matrix = getattr(self, matrix_name)
            if not torch.allclose(matrix, matrix.T):
                raise ValueError(f"{matrix_name} is not symmetric")

        _cutoff = cutoff or float(self.sigma_matrix.detach().max())

        pair_fn = MultiSoftSpherePairFn(
            atomic_numbers=unique_z.to(device=_device),
            sigma_matrix=self.sigma_matrix,
            epsilon_matrix=self.epsilon_matrix,
            alpha_matrix=self.alpha_matrix,
        )

        super().__init__(
            pair_fn=pair_fn,
            cutoff=_cutoff,
            device=device,
            dtype=dtype,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
            per_atom_energies=per_atom_energies,
            per_atom_stresses=per_atom_stresses,
            neighbor_list_fn=neighbor_list_fn,
            reduce_to_half_list=True,
            retain_graph=retain_graph,
        )
