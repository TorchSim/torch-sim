"""General batched pair potential and pair forces models.

This module provides :class:`PairPotentialModel`, a flexible wrapper that turns any
pairwise energy function into a full TorchSim model with forces (via autograd) and
optional stress / per-atom output.  It generalises Lennard-Jones, Morse, soft-sphere,
and similar potentials that depend only on pairwise distances and atomic numbers.

It also provides :class:`PairForcesModel` for potentials defined directly as forces
(e.g. the asymmetric particle-life interaction) that cannot be expressed as the
gradient of a scalar energy.

The pair function signature required by :class:`PairPotentialModel` is:
``pair_fn(distances, atomic_numbers_i, atomic_numbers_j) -> pair_energies``,
where all arguments are 1-D tensors of length n_pairs and the return value is a
1-D tensor of pair energies. Additional parameters (e.g., ``sigma``, ``epsilon``)
can be bound using :func:`functools.partial`.

Notes:
    - The ``cutoff`` parameter determines the neighbor list construction range.
      Pairs beyond the cutoff are excluded from energy/force calculations. If your
      potential has its own natural cutoff (e.g., WCA potential), ensure the model's
      ``cutoff`` is at least as large.
    - The ``atomic_numbers_i`` and ``atomic_numbers_j`` arguments are provided for
      type-dependent potentials, but can be ignored (e.g., with ``# noqa: ARG001``)
      for type-independent potentials like Lennard-Jones.
    - The ``dtype`` of the SimState must match the model's ``dtype``. The model will
      raise a ``TypeError`` if they don't match.
    - Use ``reduce_to_half_list=True`` for symmetric potentials to halve computation
      time. Only use ``False`` for asymmetric interactions or when you need the
      full neighbor list for other purposes.


Example::

    from torch_sim.models.pair_potential import PairPotentialModel
    from torch_sim import io
    from ase.build import bulk
    import functools
    import torch


    def bmhtf_pair(dr, zi, zj, A, B, C, D, sigma):
        # Born-Meyer-Huggins-Tosi-Fumi (BMHTF) potential for ionic crystals
        # V(r) = A * exp(B * (sigma - r)) - C/r^6 - D/r^8
        exp_term = A * torch.exp(B * (sigma - dr))
        r6_term = C / dr.pow(6)
        r8_term = D / dr.pow(8)
        energy = exp_term - r6_term - r8_term
        return torch.where(dr > 0, energy, torch.zeros_like(energy))


    # Na-Cl interaction parameters
    fn = functools.partial(
        bmhtf_pair,
        A=20.3548,
        B=3.1546,
        C=674.4793,
        D=837.0770,
        sigma=2.755,
    )
    model = PairPotentialModel(pair_fn=fn, cutoff=10.0)

    # Create NaCl structure using ASE
    nacl_atoms = bulk("NaCl", "rocksalt", a=5.64)
    sim_state = io.atoms_to_state(nacl_atoms, device=torch.device("cpu"))
    results = model(sim_state)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.transforms import compute_cell_shifts, pbc_wrap_batched


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.state import SimState


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
    state: SimState,
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
    sim_state = state

    positions = sim_state.positions
    row_cell = sim_state.row_vector_cell
    pbc = sim_state.pbc
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
    stress = stress.index_add(0, system_mapping, -stress_per_pair)
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
    stress_scale = 1.0 if half else 0.5
    out: dict[str, torch.Tensor] = {"stress": stress * stress_scale}

    if per_atom:
        # Each endpoint (i and j) gets half the pair's contribution.
        # Half list: each unique pair appears once → w = 0.5.
        # Full list: each unique pair appears twice → w = 0.25.
        w = 0.5 if half else 0.25
        n_atoms = positions.shape[0]
        atom_stresses = torch.zeros((n_atoms, 3, 3), dtype=dtype, device=device)
        atom_stresses = atom_stresses.index_add(0, mapping[0], -w * stress_per_pair)
        atom_stresses = atom_stresses.index_add(0, mapping[1], -w * stress_per_pair)
        out["stresses"] = atom_stresses / volumes[system_idx, None, None]

    return out


class PairPotentialModel(ModelInterface):
    """General batched pair potential model.

    Computes energies, forces, and stresses for any pairwise potential defined by a
    callable of the form ``pair_fn(distances, atomic_numbers_i, atomic_numbers_j) ->
    pair_energies``, where all arguments are 1-D tensors of length n_pairs and the
    return value is a 1-D tensor of pair energies.  Forces are obtained analytically
    via autograd by differentiating the energy with respect to positions.

    When stress is computed, it uses the virial formula: σ = -1/V Σ_{ij} r_ij ⊗ f_ij,
    where r_ij is the pair displacement vector and f_ij is the force vector.

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
        *,
        cutoff: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        neighbor_list_fn: Callable = torchsim_nl,
        reduce_to_half_list: bool = False,
        retain_graph: bool = False,
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
            retain_graph: If True, keep the computation graph after computing forces
                so that the energy can still be differentiated w.r.t. model parameters
                (e.g. for differentiable simulation / meta-optimization).
                Defaults to False.
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
        self.retain_graph = retain_graph

    def forward(self, state: SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute pair-potential properties with batched tensor operations.

        Args:
            state: Simulation state.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with keys ``"energy"`` (shape ``[n_systems]``), optionally
            ``"forces"`` (``[n_atoms, 3]``), ``"stress"`` (``[n_systems, 3, 3]``),
            ``"energies"`` (``[n_atoms]``), ``"stresses"`` (``[n_atoms, 3, 3]``).

        Raises:
            TypeError: If the SimState's dtype does not match the model's dtype.
        """
        if state.dtype != self._dtype:
            raise TypeError(
                f"SimState dtype {state.dtype} does not match model dtype {self._dtype}. "
                f"Either set the model dtype to {state.dtype} or convert the SimState "
                f"to {self._dtype} using sim_state.to(dtype={self._dtype})."
            )
        dtype = self._dtype
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
        energy = torch.zeros(n_systems, dtype=dtype, device=self._device)
        energy = energy.index_add(0, system_mapping, ew * pair_energies)
        results["energy"] = energy

        if self.per_atom_energies:
            atom_energies = torch.zeros(
                positions.shape[0], dtype=dtype, device=self._device
            )
            atom_energies = atom_energies.index_add(0, mapping[0], ew * pair_energies)
            atom_energies = atom_energies.index_add(0, mapping[1], ew * pair_energies)
            results["energies"] = atom_energies

        if need_grad:
            (dv_dr,) = torch.autograd.grad(
                pair_energies.sum(),
                dist_for_grad,
                create_graph=False,
                retain_graph=self.retain_graph,
            )
            safe_dist = torch.where(distances > 0, distances, torch.ones_like(distances))
            # force_vectors = -dV/dr * r̂_ij: positive (repulsive) pushes j away from i.
            force_vectors = (-dv_dr / safe_dist)[:, None] * dr_vec

            if self._compute_forces:
                forces = torch.zeros_like(positions)
                if half:
                    # Half list: each pair once → apply Newton's third law explicitly.
                    forces = forces.index_add(0, mapping[0], -force_vectors)
                    forces = forces.index_add(0, mapping[1], force_vectors)
                else:
                    # Full list: atom i appears as mapping[0] for every i→j pair,
                    # covering all its neighbors.  mapping[1] accumulation would
                    # double-count, so we only accumulate on the source atom.
                    forces = forces.index_add(0, mapping[0], -force_vectors)
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
                    dtype,
                    self._device,
                    half=half,
                    per_atom=self.per_atom_stresses,
                )
            )

        if not self.retain_graph:
            results = {k: v.detach() for k, v in results.items()}

        return results


class PairForcesModel(ModelInterface):
    """Batched pair model for potentials defined directly as forces.

    Use this when the interaction is specified as a scalar force magnitude
    ``force_fn(distances, zi, zj) -> force_magnitudes`` rather than as an energy.
    This covers asymmetric or non-conservative interactions such as the particle-life
    potential where no scalar energy exists.

    Forces are accumulated as:
        F_i += -f_ij * r̂_ij,  F_j += +f_ij * r̂_ij

    Note:
        Unlike :class:`PairPotentialModel`, this class does not compute energies
        (returns zeros) since there is no underlying energy function. Use
        :class:`PairPotentialModel` when your interaction can be expressed as an
        energy function, as it provides automatic force computation via autograd
        and is generally more efficient.

    Example::

        from torch_sim.models.particle_life import particle_life_pair_force
        from torch_sim.models.pair_potential import PairForcesModel
        import functools

        fn = functools.partial(particle_life_pair_force, A=1.0, beta=0.3, sigma=1.0)
        model = PairForcesModel(force_fn=fn, cutoff=1.0)
        results = model(sim_state)
    """

    def __init__(
        self,
        force_fn: Callable,
        *,
        cutoff: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
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

    def forward(self, state: SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute forces from a direct pair force function.

        Args:
            state: Simulation state.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with keys ``"energy"`` (zeros, shape ``[n_systems]``),
            ``"forces"`` (shape ``[n_atoms, 3]``), and optionally ``"stress"``
            (shape ``[n_systems, 3, 3]``) and ``"stresses"``
            (shape ``[n_atoms, 3, 3]``).

        Raises:
            TypeError: If the SimState's dtype does not match the model's dtype.
        """
        if state.dtype != self._dtype:
            raise TypeError(
                f"SimState dtype {state.dtype} does not match model dtype {self._dtype}. "
                f"Either set the model dtype to {state.dtype} or convert the SimState "
                f"to {self._dtype} using sim_state.to(dtype={self._dtype})."
            )
        dtype = self._dtype
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
        forces = forces.index_add(0, mapping[0], -force_vectors)
        forces = forces.index_add(0, mapping[1], force_vectors)

        results: dict[str, torch.Tensor] = {
            "energy": torch.zeros(n_systems, dtype=dtype, device=self._device),
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
                    dtype,
                    self._device,
                    half=half,
                    per_atom=self.per_atom_stresses,
                )
            )

        return results
