"""General batched pair potential model and standard pair interaction functions.

This module provides :class:`PairPotentialModel`, a flexible wrapper that turns any
pairwise energy function into a full TorchSim model with forces (via autograd) and
optional stress / per-atom output.  It generalises Lennard-Jones, Morse, soft-sphere,
and similar potentials that depend only on pairwise distances and atomic numbers.

It also provides :class:`PairForcesModel` for potentials defined directly as forces
(e.g. the asymmetric particle-life interaction) that cannot be expressed as the
gradient of a scalar energy.

Standard pair energy functions (all JIT-compatible):

* :func:`lj_pair` — Lennard-Jones 12-6
* :func:`morse_pair` — Morse potential
* :func:`soft_sphere_pair` — soft-sphere repulsion
* :func:`particle_life_pair_force` — asymmetric particle-life force (use with
  :class:`PairForcesModel`)

Example::

    from torch_sim.models.pair_potential import PairPotentialModel, lj_pair
    import functools

    fn = functools.partial(lj_pair, sigma=1.0, epsilon=1.0)
    model = PairPotentialModel(pair_fn=fn, cutoff=2.5)
    results = model(sim_state)
"""

# ruff: noqa: RUF002

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.state import SimState, ensure_sim_state, pbc_to_tensor
from torch_sim.transforms import compute_cell_shifts, pbc_wrap_batched


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.typing import StateDict


@torch.jit.script
def lj_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: float = 1.0,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Lennard-Jones 12-6 pair energy.

    V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]

    Args:
        dr: Pairwise distances, shape [n_pairs].
        zi: Atomic numbers of first atoms (unused, for interface compatibility).
        zj: Atomic numbers of second atoms (unused, for interface compatibility).
        sigma: Length scale. Defaults to 1.0.
        epsilon: Energy scale. Defaults to 1.0.

    Returns:
        Pair energies, shape [n_pairs].
    """
    idr6 = (sigma / dr).pow(6)
    return 4.0 * epsilon * (idr6 * idr6 - idr6)


@torch.jit.script
def morse_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: float = 1.0,
    epsilon: float = 5.0,
    alpha: float = 5.0,
) -> torch.Tensor:
    """Morse pair energy.

    V(r) = ε(1 - exp(-α(r - σ)))² - ε

    Args:
        dr: Pairwise distances, shape [n_pairs].
        zi: Atomic numbers of first atoms (unused).
        zj: Atomic numbers of second atoms (unused).
        sigma: Equilibrium bond distance. Defaults to 1.0.
        epsilon: Well depth / dissociation energy. Defaults to 5.0.
        alpha: Width parameter. Defaults to 5.0.

    Returns:
        Pair energies, shape [n_pairs].
    """
    return epsilon * (1.0 - torch.exp(-alpha * (dr - sigma))).pow(2) - epsilon


@torch.jit.script
def soft_sphere_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: float = 1.0,
    epsilon: float = 1.0,
    alpha: float = 2.0,
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


@torch.jit.script
def particle_life_pair_force(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    A: float = 1.0,
    beta: float = 0.3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Asymmetric particle-life scalar force magnitude.

    This is a *force* function (not an energy), intended for use with
    :class:`PairForcesModel`.

    Args:
        dr: Pairwise distances, shape [n_pairs].
        zi: Atomic numbers of first atoms (unused).
        zj: Atomic numbers of second atoms (unused).
        A: Interaction amplitude. Defaults to 1.0.
        beta: Inner radius. Defaults to 0.3.
        sigma: Outer radius / cutoff. Defaults to 1.0.

    Returns:
        Scalar force magnitudes, shape [n_pairs].
    """
    inner_mask = dr < beta
    outer_mask = (dr >= beta) & (dr < sigma)
    inner_force = dr / beta - 1.0
    outer_force = A * (1.0 - torch.abs(2.0 * dr - 1.0 - beta) / (1.0 - beta))
    return torch.where(inner_mask, inner_force, torch.zeros_like(dr)) + torch.where(
        outer_mask, outer_force, torch.zeros_like(dr)
    )


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
        self.z_to_idx: torch.Tensor
        self.atomic_numbers: torch.Tensor
        self.sigma_matrix: torch.Tensor
        self.epsilon_matrix: torch.Tensor
        self.alpha_matrix: torch.Tensor

        n = len(atomic_numbers)
        if sigma_matrix.shape != (n, n):
            raise ValueError(f"sigma_matrix must have shape ({n}, {n})")
        if epsilon_matrix.shape != (n, n):
            raise ValueError(f"epsilon_matrix must have shape ({n}, {n})")
        if alpha_matrix is not None and alpha_matrix.shape != (n, n):
            raise ValueError(f"alpha_matrix must have shape ({n}, {n})")

        self.register_buffer("atomic_numbers", atomic_numbers)
        self.register_buffer("sigma_matrix", sigma_matrix)
        self.register_buffer("epsilon_matrix", epsilon_matrix)
        self.register_buffer(
            "alpha_matrix",
            alpha_matrix if alpha_matrix is not None else torch.full((n, n), 2.0),
        )
        # Build a lookup table: atomic_number -> species index
        max_z = int(atomic_numbers.max().item()) + 1
        z_to_idx = torch.full((max_z,), -1, dtype=torch.long)
        for idx, z in enumerate(atomic_numbers.tolist()):
            z_to_idx[int(z)] = idx
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


def full_to_half_list(
    mapping: torch.Tensor,
    system_mapping: torch.Tensor,
    shifts_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce a full neighbor list to a half list.

    Keeps each unordered pair exactly once.  For ``i != j`` pairs, the copy with
    ``i < j`` is kept.  For self-image pairs (``i == j``, non-zero periodic shift),
    the copy whose first non-zero shift component is positive is kept.

    Args:
        mapping: Pair indices, shape [2, n_pairs].
        system_mapping: System index per pair, shape [n_pairs].
        shifts_idx: Periodic shift vectors per pair, shape [n_pairs, 3].

    Returns:
        (mapping, system_mapping, shifts_idx) with duplicates removed.
    """
    i, j = mapping[0], mapping[1]
    # For i != j: keep i < j
    diff_mask = i < j
    # For i == j (self-image through PBC): keep the copy whose shift vector
    # is lexicographically positive (first non-zero component > 0).
    same = i == j
    if same.any():
        # Compute sign of first non-zero shift component per pair.
        # shifts_idx columns are checked in order (x, y, z).
        s = shifts_idx[same]  # [n_self, 3]
        # Find first non-zero component: mark columns that are non-zero,
        # then take the value at the first such column.
        first_nz_sign = torch.zeros(s.shape[0], dtype=s.dtype, device=s.device)
        resolved = torch.zeros(s.shape[0], dtype=torch.bool, device=s.device)
        for dim in range(3):
            col = s[:, dim]
            is_nz = (col != 0) & ~resolved
            first_nz_sign = torch.where(is_nz, col, first_nz_sign)
            resolved = resolved | is_nz
        self_mask = first_nz_sign > 0
        diff_mask = diff_mask.clone()
        diff_mask[same] = self_mask
    return mapping[:, diff_mask], system_mapping[diff_mask], shifts_idx[diff_mask]


def _prepare_pairs(
    state: SimState | StateDict,
    *,
    cutoff: torch.Tensor,
    neighbor_list_fn: Callable,
    reduce_to_half_list: bool,
    device: torch.device,
) -> tuple[
    torch.Tensor,  # positions
    torch.Tensor,  # mapping [2, n_pairs]
    torch.Tensor,  # system_mapping [n_pairs]
    torch.Tensor,  # system_idx [n_atoms]
    torch.Tensor,  # dr_vec [n_pairs, 3]
    torch.Tensor,  # distances [n_pairs]
    torch.Tensor,  # zi [n_pairs]
    torch.Tensor,  # zj [n_pairs]
    torch.Tensor,  # cutoff_mask [n_pairs]
    torch.Tensor,  # row_cell [n_systems, 3, 3]
    int,  # n_systems
]:
    """Unpack state, build neighbor list, compute pair vectors and distances."""
    sim_state = ensure_sim_state(state)

    positions = sim_state.positions
    row_cell = sim_state.row_vector_cell
    pbc = pbc_to_tensor(sim_state.pbc, sim_state.device)
    atomic_numbers = sim_state.atomic_numbers

    system_idx = (
        sim_state.system_idx
        if sim_state.system_idx is not None
        else torch.zeros(positions.shape[0], dtype=torch.long, device=device)
    )

    wrapped_positions = (
        pbc_wrap_batched(positions, sim_state.cell, system_idx, pbc)
        if pbc.any()
        else positions
    )

    pbc_batched = (
        pbc.unsqueeze(0).expand(sim_state.n_systems, -1) if pbc.ndim == 1 else pbc
    )

    mapping, system_mapping, shifts_idx = neighbor_list_fn(
        positions=wrapped_positions,
        cell=row_cell,
        pbc=pbc_batched,
        cutoff=cutoff,
        system_idx=system_idx,
    )

    if reduce_to_half_list:
        mapping, system_mapping, shifts_idx = full_to_half_list(
            mapping, system_mapping, shifts_idx
        )

    cell_shifts = compute_cell_shifts(row_cell, shifts_idx, system_mapping)
    dr_vec = wrapped_positions[mapping[1]] - wrapped_positions[mapping[0]] + cell_shifts
    distances = dr_vec.norm(dim=1)

    return (
        positions,
        mapping,
        system_mapping,
        system_idx,
        dr_vec,
        distances,
        atomic_numbers[mapping[0]],
        atomic_numbers[mapping[1]],
        distances < cutoff,
        row_cell,
        sim_state.n_systems,
    )


def _virial_stress(
    dr_vec: torch.Tensor,
    force_vectors: torch.Tensor,
    system_mapping: torch.Tensor,
    row_cell: torch.Tensor,
    n_systems: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute virial stress tensor from pair displacements and force vectors.

    Uses the pair virial formula: σ = -1/V Σ_{ij} r_ij ⊗ f_ij

    Args:
        dr_vec: Pair displacement vectors r_j - r_i (+shifts), shape [n_pairs, 3].
        force_vectors: Force vectors on atom j due to i, shape [n_pairs, 3].
        system_mapping: System index per pair, shape [n_pairs].
        row_cell: Row-vector cell tensors, shape [n_systems, 3, 3].
        n_systems: Number of systems.
        dtype: Output dtype.
        device: Output device.

    Returns:
        ``(stress, stress_per_pair, volumes)`` where stress has shape
        ``[n_systems, 3, 3]``, stress_per_pair ``[n_pairs, 3, 3]``, and
        volumes ``[n_systems]``.
    """
    volumes = torch.abs(torch.linalg.det(row_cell))
    stress_per_pair = torch.einsum("...i,...j->...ij", dr_vec, force_vectors)
    stress = torch.zeros((n_systems, 3, 3), dtype=dtype, device=device)
    stress.index_add_(0, system_mapping, -stress_per_pair)
    stress = stress / volumes[:, None, None]
    return stress, stress_per_pair, volumes


def _accumulate_stress(
    positions: torch.Tensor,
    mapping: torch.Tensor,
    system_mapping: torch.Tensor,
    system_idx: torch.Tensor,
    dr_vec: torch.Tensor,
    force_vectors: torch.Tensor,
    row_cell: torch.Tensor,
    n_systems: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    half: bool,
    per_atom: bool,
) -> dict[str, torch.Tensor]:
    """Compute system and (optionally) per-atom virial stresses."""
    stress, stress_per_pair, volumes = _virial_stress(
        dr_vec,
        force_vectors,
        system_mapping,
        row_cell,
        n_systems,
        dtype,
        device,
    )
    stress_scale = 2.0 if half else 1.0
    out: dict[str, torch.Tensor] = {"stress": stress * stress_scale}

    if per_atom:
        # Half list: each pair once → weight 1.0 per endpoint.
        # Full list: each pair twice (i→j and j→i) → weight 0.5 per endpoint.
        w = 1.0 if half else 0.5
        n_atoms = positions.shape[0]
        atom_stresses = torch.zeros((n_atoms, 3, 3), dtype=dtype, device=device)
        atom_stresses.index_add_(0, mapping[0], -w * stress_per_pair)
        atom_stresses.index_add_(0, mapping[1], -w * stress_per_pair)
        out["stresses"] = atom_stresses / volumes[system_idx, None, None]

    return out


class PairPotentialModel(ModelInterface):
    """General batched pair potential model.

    Computes energies, forces, and stresses for any pairwise potential defined by a
    callable of the form ``pair_fn(distances, atomic_numbers_i, atomic_numbers_j) ->
    pair_energies``, where all arguments are 1-D tensors of length n_pairs and the
    return value is a 1-D tensor of pair energies.  Forces are obtained analytically
    via autograd.

    Example::

        def lj_fn(dr, zi, zj):
            idr6 = (1.0 / dr) ** 6
            return 4.0 * (idr6**2 - idr6)


        model = PairPotentialModel(pair_fn=lj_fn, cutoff=2.5)
        results = model(sim_state)
    """

    def __init__(
        self,
        pair_fn: Callable,
        cutoff: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        neighbor_list_fn: Callable = torchsim_nl,
        reduce_to_half_list: bool = False,
    ) -> None:
        """Initialize the pair potential model.

        Args:
            pair_fn: Callable with signature
                ``(distances, atomic_numbers_i, atomic_numbers_j) -> pair_energies``.
                All tensors are 1-D with length n_pairs.
            cutoff: Interaction cutoff distance in the same units as positions.
            device: Device for computations. Defaults to CPU.
            dtype: Floating-point dtype. Defaults to torch.float32.
            compute_forces: Whether to compute atomic forces. Defaults to True.
            compute_stress: Whether to compute the stress tensor. Defaults to False.
            per_atom_energies: Whether to return per-atom energies. Defaults to False.
            per_atom_stresses: Whether to return per-atom stresses.
            neighbor_list_fn: Neighbor-list constructor. Defaults to torchsim_nl.
            reduce_to_half_list: If True, reduce the full neighbor list to i < j pairs
                before computing interactions. Halves pair operations and makes
                accumulation patterns unambiguous. Only valid for symmetric pair
                functions; do not use for asymmetric interactions. Defaults to False.
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.per_atom_energies = per_atom_energies
        self.per_atom_stresses = per_atom_stresses
        self.pair_fn = pair_fn
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = torch.tensor(cutoff, dtype=dtype, device=self._device)
        self.reduce_to_half_list = reduce_to_half_list

    def forward(
        self, state: SimState | StateDict, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Compute pair-potential properties with batched tensor operations.

        Args:
            state: Simulation state or equivalent state dict.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with keys ``"energy"`` (shape ``[n_systems]``), optionally
            ``"forces"`` (``[n_atoms, 3]``), ``"stress"`` (``[n_systems, 3, 3]``),
            ``"energies"`` (``[n_atoms]``), ``"stresses"`` (``[n_atoms, 3, 3]``).
        """
        half = self.reduce_to_half_list
        (
            positions,
            mapping,
            system_mapping,
            system_idx,
            dr_vec,
            distances,
            zi,
            zj,
            cutoff_mask,
            row_cell,
            n_systems,
        ) = _prepare_pairs(
            state,
            cutoff=self.cutoff,
            neighbor_list_fn=self.neighbor_list_fn,
            reduce_to_half_list=half,
            device=self._device,
        )

        need_grad = self._compute_forces or self._compute_stress
        dist_for_grad = distances.requires_grad_() if need_grad else distances

        pair_energies = self.pair_fn(dist_for_grad, zi, zj)
        pair_energies = torch.where(
            cutoff_mask, pair_energies, torch.zeros_like(pair_energies)
        )

        # Half list: each pair appears once → weight 1.0.
        # Full list: each pair appears as (i,j) and (j,i) → weight 0.5.
        ew = 1.0 if half else 0.5

        results: dict[str, torch.Tensor] = {}
        energy = torch.zeros(n_systems, dtype=self._dtype, device=self._device)
        energy.index_add_(0, system_mapping, ew * pair_energies)
        results["energy"] = energy

        if self.per_atom_energies:
            atom_energies = torch.zeros(
                positions.shape[0], dtype=self._dtype, device=self._device
            )
            atom_energies.index_add_(0, mapping[0], ew * pair_energies)
            atom_energies.index_add_(0, mapping[1], ew * pair_energies)
            results["energies"] = atom_energies

        if need_grad:
            (dv_dr,) = torch.autograd.grad(
                pair_energies.sum(),
                dist_for_grad,
                create_graph=False,
            )
            safe_dist = torch.where(distances > 0, distances, torch.ones_like(distances))
            # force_vectors = -dV/dr * r̂_ij: positive (repulsive) pushes j away from i.
            force_vectors = (-dv_dr / safe_dist)[:, None] * dr_vec

            if self._compute_forces:
                forces = torch.zeros_like(positions)
                if half:
                    # Half list: each pair once → apply Newton's third law explicitly.
                    forces.index_add_(0, mapping[0], -force_vectors)
                    forces.index_add_(0, mapping[1], force_vectors)
                else:
                    # Full list: atom i appears as mapping[0] for every i→j pair,
                    # covering all its neighbors.  mapping[1] accumulation would
                    # double-count, so we only accumulate on the source atom.
                    forces.index_add_(0, mapping[0], -force_vectors)
                results["forces"] = forces

        if self._compute_stress:
            results.update(
                _accumulate_stress(
                    positions,
                    mapping,
                    system_mapping,
                    system_idx,
                    dr_vec,
                    force_vectors,
                    row_cell,
                    n_systems,
                    self._dtype,
                    self._device,
                    half=half,
                    per_atom=self.per_atom_stresses,
                )
            )

        return results


class PairForcesModel(ModelInterface):
    """Batched pair model for potentials defined directly as forces.

    Use this when the interaction is specified as a scalar force magnitude
    ``force_fn(distances, zi, zj) -> force_magnitudes`` rather than as an energy.
    This covers asymmetric or non-conservative interactions such as the particle-life
    potential where no scalar energy exists.

    Forces are accumulated as:
        F_i += -f_ij * r̂_ij,  F_j += +f_ij * r̂_ij

    Example::

        from torch_sim.models.pair_potential import (
            PairForcesModel,
            particle_life_pair_force,
        )
        import functools

        fn = functools.partial(particle_life_pair_force, A=1.0, beta=0.3, sigma=1.0)
        model = PairForcesModel(force_fn=fn, cutoff=1.0)
        results = model(sim_state)
    """

    def __init__(
        self,
        force_fn: Callable,
        cutoff: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,
        compute_stress: bool = False,
        per_atom_stresses: bool = False,
        neighbor_list_fn: Callable = torchsim_nl,
        reduce_to_half_list: bool = False,
    ) -> None:
        """Initialize the pair forces model.

        Args:
            force_fn: Callable with signature
                ``(distances, zi, zj) -> force_magnitudes``.
                All tensors are 1-D with length n_pairs.
            cutoff: Interaction cutoff distance.
            device: Device for computations. Defaults to CPU.
            dtype: Floating-point dtype. Defaults to torch.float32.
            compute_stress: Whether to compute the virial stress tensor.
            per_atom_stresses: Whether to return per-atom stresses.
            neighbor_list_fn: Neighbor-list constructor. Defaults to torchsim_nl.
            reduce_to_half_list: If True, reduce the full neighbor list to i < j pairs
                before computing interactions. Only valid for symmetric force functions;
                do not use for asymmetric interactions where f(i→j) ≠ f(j→i).
        """
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = compute_stress
        self.per_atom_stresses = per_atom_stresses
        self.force_fn = force_fn
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = torch.tensor(cutoff, dtype=dtype, device=self._device)
        self.reduce_to_half_list = reduce_to_half_list

    def forward(
        self, state: SimState | StateDict, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Compute forces from a direct pair force function.

        Args:
            state: Simulation state or equivalent state dict.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with keys ``"energy"`` (zeros, shape ``[n_systems]``),
            ``"forces"`` (shape ``[n_atoms, 3]``), and optionally ``"stress"``
            (shape ``[n_systems, 3, 3]``) and ``"stresses"``
            (shape ``[n_atoms, 3, 3]``).
        """
        half = self.reduce_to_half_list
        (
            positions,
            mapping,
            system_mapping,
            system_idx,
            dr_vec,
            distances,
            zi,
            zj,
            cutoff_mask,
            row_cell,
            n_systems,
        ) = _prepare_pairs(
            state,
            cutoff=self.cutoff,
            neighbor_list_fn=self.neighbor_list_fn,
            reduce_to_half_list=half,
            device=self._device,
        )

        pair_forces = self.force_fn(distances, zi, zj)
        pair_forces = torch.where(cutoff_mask, pair_forces, torch.zeros_like(pair_forces))

        safe_dist = torch.where(distances > 0, distances, torch.ones_like(distances))
        force_vectors = (pair_forces / safe_dist)[:, None] * dr_vec

        forces = torch.zeros_like(positions)
        forces.index_add_(0, mapping[0], -force_vectors)
        forces.index_add_(0, mapping[1], force_vectors)

        results: dict[str, torch.Tensor] = {
            "energy": torch.zeros(n_systems, dtype=self._dtype, device=self._device),
            "forces": forces,
        }

        if self._compute_stress:
            results.update(
                _accumulate_stress(
                    positions,
                    mapping,
                    system_mapping,
                    system_idx,
                    dr_vec,
                    force_vectors,
                    row_cell,
                    n_systems,
                    self._dtype,
                    self._device,
                    half=half,
                    per_atom=self.per_atom_stresses,
                )
            )

        return results
