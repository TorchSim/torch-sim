import timeit

import torch

from torchsim.models.lennard_jones import lennard_jones_pair, lennard_jones_pair_force
from torchsim.transforms import raw_transform


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
    # Create displacement vectors for all pairs
    ri = R.unsqueeze(0)
    rj = R.unsqueeze(1)
    dr = rj - ri

    # Apply periodic boundary conditions
    dr = dr - box.diagonal() * torch.round(dr / box.diagonal())
    if perturbation is not None:
        dr = raw_transform(perturbation, dr)

    # Calculate distances
    distances = torch.norm(dr, dim=2)

    # Mask out self-interactions
    mask = torch.eye(R.shape[0], dtype=torch.bool, device=R.device)
    distances = distances.masked_fill(mask, torch.inf)

    # Calculate potential energy
    energy = lennard_jones_pair(distances, sigma, epsilon)

    return energy.sum() / 2.0  # Divide by 2 to avoid double counting


def force_fn(R: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
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
    # Ensure we track gradients

    # Get volume and dimension
    volume = torch.prod(box.diagonal())
    dim = R.shape[1]

    # Create identity and zero matrices
    eye = torch.eye(dim, device=R.device)
    zero = torch.zeros((dim, dim), device=R.device).requires_grad_(mode=True)

    def U(eps: torch.Tensor) -> torch.Tensor:
        return energy_fn(R, box, perturbation=(eye + eps))

    # Calculate energy at zero strain
    energy = U(zero)

    stress = -torch.autograd.grad(energy, zero)[0]

    return 2 * stress / volume


def stress_autograd_fn_functorch(R: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    # Get volume and dimension
    volume = torch.prod(box.diagonal())
    dim = R.shape[1]

    # Create identity and zero matrices
    I = torch.eye(dim, device=R.device, dtype=torch.float64)
    zero = torch.zeros((dim, dim), device=R.device, dtype=torch.float64)

    def U(eps, R):
        # Apply strain perturbation to positions
        return energy_fn(R, box, perturbation=(I + eps))

    # Use functorch.grad
    from torch.func import grad

    # Get gradient of U with respect to eps
    grad_U = grad(U, argnums=0)  # take gradient with respect to first argument (eps)

    # Calculate stress using grad
    stress = -grad_U(zero, R)

    return 2 * stress / volume


latvec = torch.eye(2) * box_size

print(
    f"stress_fn time: {timeit.timeit('stress_fn(positions, latvec)', globals=globals(), number=100)} seconds"
)
print(
    f"stress_autograd_fn time: {timeit.timeit('stress_autograd_fn(positions, latvec)', globals=globals(), number=100)} seconds"
)
print(
    f"stress_autograd_fn_functorch time: {timeit.timeit('stress_autograd_fn_functorch(positions, latvec)', globals=globals(), number=100)} seconds"
)

print(energy_fn(positions, latvec, None))
print(force_fn(positions, latvec))
print(f"stress_fn: {stress_fn(positions, latvec)[1]}")
print(f"stress_autograd_fn: {stress_autograd_fn(positions, latvec)}")
print(f"stress_autograd_fn_functorch: {stress_autograd_fn_functorch(positions, latvec)}")
