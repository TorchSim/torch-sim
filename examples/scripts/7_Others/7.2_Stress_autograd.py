"""Measure performance of different methods for calculating stress.

The stress is calculated using three different methods:

1. The stress_fn function, explicit force method.
2. The stress_autograd_fn function, automatic differentiation method.
3. The stress_autograd_fn_functorch function, functorch automatic differentiation method.
"""

# /// script
# dependencies = [
#     "scipy>=1.15",
# ]
# ///

import timeit

import torch

from torch_sim.unbatched.models.lennard_jones import (
    lennard_jones_pair,
    lennard_jones_pair_force,
)


torch.set_default_tensor_type(torch.DoubleTensor)
# Set simulation parameters
n_steps = 10000
kT = 0.722  # Temperature in energy units
sigma = 1.0  # Length parameter
epsilon = 1.0  # Energy parameter

# Grid initialization
Nx = 10
length_scale = 1.1  # Increased to reduce initial overlap
positions = torch.zeros((Nx * Nx, 2))
for i in range(Nx):
    for j in range(Nx):
        positions[i * Nx + j] = torch.tensor([i * length_scale, j * length_scale])

num_particles = Nx * Nx
box_size = Nx * length_scale
mass = torch.ones(num_particles)


def energy_fn(
    R: torch.Tensor, box: torch.Tensor, perturbation: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate energy using a brute force method."""
    # Create displacement vectors for all pairs
    ri = R.unsqueeze(0)
    rj = R.unsqueeze(1)
    dr = rj - ri

    # Apply periodic boundary conditions
    dr = dr - box.diagonal() * torch.round(dr / box.diagonal())
    if perturbation is not None:
        # Apply transformation directly (R + dR)
        dr = dr + torch.einsum("ij,nmj->nmi", perturbation, dr)

    # Calculate distances
    distances = torch.norm(dr, dim=2)

    # Mask out self-interactions
    mask = torch.eye(R.shape[0], dtype=torch.bool, device=R.device)
    distances = distances.masked_fill(mask, torch.inf)

    # Calculate potential energy
    energy = lennard_jones_pair(distances, sigma, epsilon)

    return energy.sum() / 2.0  # Divide by 2 to avoid double counting


def force_fn(R: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Calculate forces using a brute force method."""
    # Create displacement vectors for all pairs
    ri = R.unsqueeze(0)
    rj = R.unsqueeze(1)
    dr = rj - ri

    # Apply periodic boundary conditions
    dr = dr - box.diagonal() * torch.round(dr / box.diagonal())

    # Calculate distances
    distances = torch.norm(dr, dim=2)

    # Mask out self-interactions
    mask = torch.eye(R.shape[0], dtype=torch.bool, device=R.device)

    distances = distances.masked_fill(mask, torch.inf)
    forces = lennard_jones_pair_force(distances, sigma, epsilon)

    # Project forces along displacement vectors
    unit_vectors = dr / distances.unsqueeze(-1)
    force_components = forces.unsqueeze(-1) * unit_vectors
    return force_components.sum(dim=0)


def stress_fn(R: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Calculate stress using a brute force method."""
    # Create displacement vectors for all pairs
    ri = R.unsqueeze(0)
    rj = R.unsqueeze(1)
    dr = rj - ri

    # Apply periodic boundary conditions
    dr = dr - box.diagonal() * torch.round(dr / box.diagonal())

    # Calculate distances
    distances = torch.norm(dr, dim=2)

    # Mask out self-interactions
    mask = torch.eye(R.shape[0], dtype=torch.bool, device=R.device)
    distances = distances.masked_fill(mask, torch.inf)

    # Calculate forces between pairs
    forces = lennard_jones_pair_force(distances, sigma, epsilon)

    # Calculate stress components using outer product of force and distance
    # Project forces along displacement vectors
    unit_vectors = dr / distances.unsqueeze(-1)
    force_components = forces.unsqueeze(-1) * unit_vectors

    # Compute outer product for stress tensor
    stress_per_pair = torch.einsum("...i,...j->...ij", dr, force_components)

    # Sum over all pairs and divide by volume
    volume = torch.prod(box.diagonal())
    stress = -stress_per_pair.sum(dim=(0, 1)) / volume

    return force_components.sum(dim=0), stress


def stress_autograd_fn(R: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Calculate stress using autograd."""
    # Get volume and dimension
    volume = torch.prod(box.diagonal())
    dim = R.shape[1]

    # Create identity and zero matrices
    eye = torch.eye(dim, device=R.device)
    zero = torch.zeros((dim, dim), device=R.device).requires_grad_(True)  # noqa: FBT003

    def internal_energy(eps: torch.Tensor) -> torch.Tensor:
        return energy_fn(R, box, perturbation=(eye + eps))

    # Calculate energy at zero strain
    energy = internal_energy(zero)

    stress = -torch.autograd.grad(energy, zero)[0]

    return 2 * stress / volume


def stress_autograd_fn_functorch(R: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """Calculate stress using functorch.grad."""
    # Get volume and dimension
    volume = torch.prod(box.diagonal())
    dim = R.shape[1]

    # Create identity and zero matrices
    eye = torch.eye(dim, device=R.device, dtype=torch.float64)
    zero = torch.zeros((dim, dim), device=R.device, dtype=torch.float64)

    def internal_energy(eps: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # Apply strain perturbation to positions
        return energy_fn(R, box, perturbation=(eye + eps))

    # Use functorch.grad
    from torch.func import grad

    # Get gradient of U with respect to eps
    grad_U = grad(internal_energy, argnums=0)  # take gradient w.r.t. first argument (eps)

    # Calculate stress using grad
    stress = -grad_U(zero, R)

    return 2 * stress / volume


lat_vec = torch.eye(2) * box_size

stress_fn_time = timeit.timeit(
    "stress_fn(positions, lat_vec)", globals=globals(), number=100
)
print(f"{stress_fn_time=:.4f} sec")

stress_autograd_fn_time = timeit.timeit(
    "stress_autograd_fn(positions, lat_vec)", globals=globals(), number=100
)
print(f"{stress_autograd_fn_time=:.4f} sec")

stress_autograd_fn_functorch_time = timeit.timeit(
    "stress_autograd_fn_functorch(positions, lat_vec)", globals=globals(), number=100
)
print(f"{stress_autograd_fn_functorch_time=:.4f} sec")

print(energy_fn(positions, lat_vec, None))
print(force_fn(positions, lat_vec))
print(f"stress_fn: {stress_fn(positions, lat_vec)[1]}")
print(f"stress_autograd_fn: {stress_autograd_fn(positions, lat_vec)}")
print(f"stress_autograd_fn_functorch: {stress_autograd_fn_functorch(positions, lat_vec)}")
