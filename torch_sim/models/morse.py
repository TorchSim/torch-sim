"""Morse potential model.

Thin wrapper around :class:`~torch_sim.models.pair_potential.PairPotentialModel` with
the :func:`morse_pair` energy function baked in.

Example::

    model = MorseModel(sigma=2.55, epsilon=0.436, alpha=1.359, cutoff=6.0)
    results = model(sim_state)
"""

from __future__ import annotations

import functools
from collections.abc import Callable  # noqa: TC003

import torch

from torch_sim.models.pair_potential import PairPotentialModel
from torch_sim.neighbors import torchsim_nl


def morse_pair(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: torch.Tensor | float = 1.0,
    epsilon: torch.Tensor | float = 5.0,
    alpha: torch.Tensor | float = 5.0,
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
    energy = epsilon * (1.0 - torch.exp(-alpha * (dr - sigma))).pow(2) - epsilon
    return torch.where(dr > 0, energy, torch.zeros_like(energy))


def morse_pair_force(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    sigma: torch.Tensor | float = 1.0,
    epsilon: torch.Tensor | float = 5.0,
    alpha: torch.Tensor | float = 5.0,
) -> torch.Tensor:
    """Morse pair force (negative gradient of energy).

    F(r) = -2αε exp(-α(r-σ)) (1 - exp(-α(r-σ)))

    Args:
        dr: Pairwise distances.
        zi: Atomic numbers of first atoms (unused).
        zj: Atomic numbers of second atoms (unused).
        sigma: Equilibrium distance. Defaults to 1.0.
        epsilon: Well depth. Defaults to 5.0.
        alpha: Width parameter. Defaults to 5.0.

    Returns:
        Pair force magnitudes.
    """
    exp_term = torch.exp(-alpha * (dr - sigma))
    force = -2.0 * alpha * epsilon * exp_term * (1.0 - exp_term)
    return torch.where(dr > 0, force, torch.zeros_like(force))


class MorseModel(PairPotentialModel):
    """Morse pair potential model.

    Convenience subclass that fixes the pair function to :func:`morse_pair` so the
    caller only needs to supply ``sigma``, ``epsilon``, and ``alpha``.

    Example::

        model = MorseModel(
            sigma=2.55,
            epsilon=0.436,
            alpha=1.359,
            cutoff=6.0,
            compute_forces=True,
        )
        results = model(sim_state)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 5.0,
        alpha: float = 5.0,
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
        """Initialize the Morse potential model.

        Args:
            sigma: Equilibrium bond distance. Defaults to 1.0.
            epsilon: Well depth / dissociation energy. Defaults to 5.0.
            alpha: Width parameter. Defaults to 5.0.
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
        self.alpha = alpha
        pair_fn = functools.partial(morse_pair, sigma=sigma, epsilon=epsilon, alpha=alpha)
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
