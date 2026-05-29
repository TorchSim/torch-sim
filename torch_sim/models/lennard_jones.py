"""Lennard-Jones 12-6 potential model.

Thin wrapper around :class:`~torch_sim.models.pair_potential.PairPotentialModel` with
the :func:`lennard_jones_pair` energy function baked in.

Example::

    model = LennardJonesModel(sigma=3.405, epsilon=0.0104, cutoff=8.5)
    results = model(sim_state)
"""

from __future__ import annotations

import functools
from collections.abc import Callable  # noqa: TC003

import torch

from torch_sim.models.pair_potential import PairPotentialModel
from torch_sim.neighbors import torchsim_nl


def lennard_jones_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: torch.Tensor | float = 1.0,
    epsilon: torch.Tensor | float = 1.0,
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
    idr12 = idr6 * idr6
    energy = 4.0 * epsilon * (idr12 - idr6)
    return torch.where(dr > 0, energy, torch.zeros_like(energy))


def lennard_jones_pair_force(
    dr: torch.Tensor,
    sigma: torch.Tensor | float = 1.0,
    epsilon: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Lennard-Jones 12-6 pair force (negative gradient of energy).

    F(r) = 24ε/r [2(σ/r)¹² - (σ/r)⁶]

    Args:
        dr: Pairwise distances, shape [n_pairs].
        sigma: Length scale. Defaults to 1.0.
        epsilon: Energy scale. Defaults to 1.0.

    Returns:
        Pair force magnitudes (positive = repulsive), shape [n_pairs].
    """
    idr = sigma / dr
    idr6 = idr.pow(6)
    idr12 = idr6 * idr6
    force = 24.0 * epsilon / dr * (2.0 * idr12 - idr6)
    return torch.where(dr > 0, force, torch.zeros_like(force))


class LennardJonesModel(PairPotentialModel):
    """Lennard-Jones 12-6 pair potential model.

    Convenience subclass that fixes the pair function to :func:`lj_pair` so the
    caller only needs to supply ``sigma`` and ``epsilon``.

    Example::

        model = LennardJonesModel(
            sigma=3.405,
            epsilon=0.0104,
            cutoff=2.5 * 3.405,
            compute_forces=True,
            compute_stress=True,
        )
        results = model(sim_state)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_forces: bool = True,
        compute_stress: bool = False,
        per_atom_energies: bool = False,
        per_atom_stresses: bool = False,
        neighbor_list_fn: Callable = torchsim_nl,
        cutoff: float | None = None,
        retain_graph: bool = False,
    ) -> None:
        """Initialize the Lennard-Jones model.

        Args:
            sigma: Length scale parameter. Defaults to 1.0.
            epsilon: Energy scale parameter. Defaults to 1.0.
            device: Device for computations. Defaults to CPU.
            dtype: Floating-point dtype. Defaults to torch.float32.
            compute_forces: Whether to compute atomic forces. Defaults to True.
            compute_stress: Whether to compute the stress tensor. Defaults to False.
            per_atom_energies: Whether to return per-atom energies. Defaults to False.
            per_atom_stresses: Whether to return per-atom stresses. Defaults to False.
            neighbor_list_fn: Neighbor-list constructor. Defaults to torchsim_nl.
            cutoff: Interaction cutoff. Defaults to 2.5 * sigma.
            retain_graph: Keep computation graph for differentiable simulation.
        """
        self.sigma = sigma
        self.epsilon = epsilon
        pair_fn = functools.partial(lennard_jones_pair, sigma=sigma, epsilon=epsilon)
        super().__init__(
            pair_fn=pair_fn,
            cutoff=cutoff if cutoff is not None else 2.5 * sigma,
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


# Keep old name as alias for backward compatibility
UnbatchedLennardJonesModel = LennardJonesModel
