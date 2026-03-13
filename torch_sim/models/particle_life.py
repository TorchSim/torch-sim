"""Particle life model.

Thin wrapper around :class:`~torch_sim.models.pair_potential.PairForcesModel` with
the :func:`particle_life_pair_force` force function
baked in.

Example::

    model = ParticleLifeModel(sigma=1.0, epsilon=1.0, beta=0.3, cutoff=1.0)
    results = model(sim_state)
"""

from __future__ import annotations

import functools
from collections.abc import Callable  # noqa: TC003

import torch

from torch_sim.models.pair_potential import PairForcesModel
from torch_sim.neighbors import torchsim_nl


def particle_life_pair_force(
    dr: torch.Tensor,
    zi: torch.Tensor,  # noqa: ARG001
    zj: torch.Tensor,  # noqa: ARG001
    A: torch.Tensor | float = 1.0,
    beta: torch.Tensor | float = 0.3,
    sigma: torch.Tensor | float = 1.0,
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


class ParticleLifeModel(PairForcesModel):
    """Asymmetric particle-life force model.

    Convenience subclass that fixes the force function to
    :func:`particle_life_pair_force` so the caller only needs to supply
    ``sigma``, ``epsilon`` (amplitude), and ``beta``.

    Example::

        model = ParticleLifeModel(
            sigma=1.0,
            epsilon=1.0,
            beta=0.3,
            cutoff=1.0,
        )
        results = model(sim_state)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        beta: float = 0.3,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        *,
        compute_stress: bool = False,
        per_atom_stresses: bool = False,
        neighbor_list_fn: Callable = torchsim_nl,
        cutoff: float | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the particle life model.

        Args:
            sigma: Outer radius of the interaction. Defaults to 1.0.
            epsilon: Interaction amplitude (``A`` parameter). Defaults to 1.0.
            beta: Inner radius of the interaction. Defaults to 0.3.
            device: Device for computations. Defaults to CPU.
            dtype: Floating-point dtype. Defaults to torch.float32.
            compute_forces: Accepted for backward compatibility (always True).
            compute_stress: Whether to compute stress tensor. Defaults to False.
            per_atom_energies: Accepted for backward compatibility (ignored — this
                is a force-only model, energy is always zero).
            per_atom_stresses: Whether to return per-atom stresses. Defaults to False.
            use_neighbor_list: Accepted for backward compatibility (ignored).
            neighbor_list_fn: Neighbor-list constructor. Defaults to torchsim_nl.
            cutoff: Interaction cutoff. Defaults to 2.5 * sigma.
            **kwargs: Additional keyword arguments.
        """
        self.sigma_param = sigma
        self.epsilon = epsilon
        self.beta = beta
        force_fn = functools.partial(
            particle_life_pair_force, A=epsilon, beta=beta, sigma=sigma
        )
        super().__init__(
            force_fn=force_fn,
            cutoff=cutoff if cutoff is not None else 2.5 * sigma,
            device=device,
            dtype=dtype,
            compute_stress=compute_stress,
            per_atom_stresses=per_atom_stresses,
            neighbor_list_fn=neighbor_list_fn,
            reduce_to_half_list=False,
        )
