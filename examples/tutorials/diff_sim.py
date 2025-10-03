# %%
# /// script
# dependencies = ["matplotlib"]
# ///

# %%
import torch_sim as ts
from typing import cast
from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt
from torch_sim.models.interface import ModelInterface
from torch_sim.models.soft_sphere import (
    soft_sphere_pair,
    DEFAULT_SIGMA,
    DEFAULT_EPSILON,
    DEFAULT_ALPHA,
)
from torch_sim import transforms
from collections.abc import Callable
from dataclasses import dataclass
from torch._functorch import config

config.donated_buffer = False

# %% [markdown]
"""
# Differentiable Simulation

In this tutorial, we will explore how to use TorchSim to perform differentiable simulations.
This tutorial will reproduce the bubble raft example from [JAX-MD](https://github.com/jax-md/jax-md/blob/main/notebooks/meta_optimization.ipynb)
and perform meta-optimization to find the optimal diameter.
"""


# %%
def draw_system(
    R: torch.Tensor, box_size: float, marker_size: float, color: list[float] | None = None
):
    """Draw a system of particles on the plot."""
    if color == None:
        color = [64 / 256] * 3
    ms = marker_size / box_size

    R = torch.tensor(R)

    marker_style = dict(
        linestyle="none",
        markeredgewidth=3,
        marker="o",
        markersize=ms,
        color=color,
        fillstyle="none",
    )

    plt.plot(R[:, 0], R[:, 1], **marker_style)
    plt.plot(R[:, 0] + box_size, R[:, 1], **marker_style)
    plt.plot(R[:, 0], R[:, 1] + box_size, **marker_style)
    plt.plot(R[:, 0] + box_size, R[:, 1] + box_size, **marker_style)
    plt.plot(R[:, 0] - box_size, R[:, 1], **marker_style)
    plt.plot(R[:, 0], R[:, 1] - box_size, **marker_style)
    plt.plot(R[:, 0] - box_size, R[:, 1] - box_size, **marker_style)

    plt.xlim([0, box_size])
    plt.ylim([0, box_size])
    plt.axis("off")
    plt.gca().set_facecolor([1, 1, 1])


# %% [markdown]
"""
## Soft Sphere potential

We will use the soft sphere potential as our model.

$$
U(r_{ij}) = \begin{cases}
    \left(1 - \frac{r_{ij}}{\sigma_{ij}}\right)^2 & \text{if } r_{ij} < \sigma_{ij} \\
    0 & \text{if } r_{ij} \geq \sigma_{ij}
\end{cases}
$$
"""
# %%
plt.gca().axhline(y=0, color="k")
plt.xlim([0, 1.5])
plt.ylim([-0.2, 0.8])

# model = SoftSphereMultiModel(sigma_matrix=torch.tensor([1.0]))
dr = torch.linspace(0, 3.0, 80)
plt.plot(dr, soft_sphere_pair(dr, sigma=1), "b-", linewidth=3)
plt.fill_between(dr, soft_sphere_pair(dr), alpha=0.4)

plt.xlabel(r"$r$", fontsize=20)
plt.ylabel(r"$U(r)$", fontsize=20)

plt.show()

# %% [markdown]
"""## Define the simple TorchSim model for the soft sphere potential."""


# %%
class SoftSphereMultiModel(ModelInterface):
    """Soft sphere potential"""

    def __init__(
        self,
        species: torch.Tensor | None = None,
        sigma_matrix: torch.Tensor | None = None,
        epsilon_matrix: torch.Tensor | None = None,
        alpha_matrix: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        *,  # Force keyword-only arguments
        pbc: bool = True,
        cutoff: float | None = None,
    ) -> None:
        """Initialize a soft sphere model for multi-component systems."""
        super().__init__()
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self.pbc = pbc

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

    def forward(
        self, custom_state: ts.SimState, species: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Compute energies and forces for a single unbatched system with multiple
        species."""
        # Convert inputs to proper device/dtype and handle species
        positions = custom_state.positions.requires_grad_(True)
        cell = custom_state.cell
        species = custom_state.atomic_numbers

        if species is not None:
            species = species.to(device=self.device, dtype=torch.long)
        else:
            species = self.species

        species_idx = species

        # Direct N^2 computation of all pairs (minimum image convention)
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
        potential_energy = pair_energies.sum() / 2

        grad_outputs: list[torch.Tensor | None] = [torch.ones_like(potential_energy)]
        grad = torch.autograd.grad(
            outputs=[potential_energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
        )

        force_grad = grad[0]
        if force_grad is not None:
            forces = torch.neg(force_grad)

        return {"energy": potential_energy, "forces": forces}


# %% [markdown]
"""
## Gradient Descent

We will use a simple gradient descent to optimize the positions of the particles.
"""


# %% [markdown]
"""## Setup the simulation environment."""


# %%
def box_size_at_number_density(
    particle_count: int, number_density: torch.Tensor
) -> torch.Tensor:
    return (particle_count / number_density) ** 0.5


def box_size_at_packing_fraction(
    diameter: torch.Tensor, packing_fraction: float
) -> torch.Tensor:
    bubble_volume = N_2 * torch.pi * (torch.square(diameter) + 1) / 4
    return torch.sqrt(bubble_volume / packing_fraction)


def species_sigma(diameter: torch.Tensor) -> torch.Tensor:
    d_AA = diameter
    d_BB = 1
    d_AB = 0.5 * (diameter + 1)
    return torch.tensor([[d_AA, d_AB], [d_AB, d_BB]])


N = 128
N_2 = N // 2
species = torch.tensor([0] * (N_2) + [1] * (N_2), dtype=torch.int32)
simulation_steps = 1000
packing_fraction = 0.98
marker_size = 260


# %%
def simulation(
    diameter: torch.Tensor, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create the simulation environment.
    box_size = box_size_at_packing_fraction(diameter, packing_fraction)
    cell = torch.eye(2) * box_size
    # Create the energy function.
    sigma = species_sigma(diameter)
    model = SoftSphereMultiModel(sigma_matrix=sigma, species=species)
    model = cast(SoftSphereMultiModel, torch.compile(model))
    # Randomly initialize the system.
    # Fix seed for reproducible random positions
    torch.manual_seed(seed)
    R = torch.rand(N, 2) * box_size

    # Minimize to the nearest minimum.
    custom_state = ts.SimState(
        atomic_numbers=species,
        masses=torch.ones(N),
        system_idx=torch.arange(N),
        positions=R,
        cell=cell,
        pbc=True,
    )
    state = ts.gradient_descent_init(model, state=custom_state)

    for _ in range(simulation_steps):
        state = ts.gradient_descent_step(model, state, pos_lr=0.1)
    return box_size, model(state)["energy"], state.positions


# %% [markdown]
"""## Packing at different diameters."""

# %%
plt.subplot(1, 2, 1)

box_size, raft_energy, bubble_positions = simulation(torch.tensor(1.0))
draw_system(bubble_positions, box_size.numpy(), marker_size)

plt.subplot(1, 2, 2)

box_size, raft_energy, bubble_positions = simulation(torch.tensor(0.8))
draw_system(bubble_positions[:N_2], box_size.numpy(), 0.8 * marker_size)
draw_system(bubble_positions[N_2:], box_size.numpy(), marker_size)
# %% [markdown]
"""## Forward simulation for different diameters and seeds."""

# %%
diameters = torch.linspace(0.4, 1.0, 10)
seeds = torch.arange(1, 6)
box_size_tensor = torch.zeros(len(diameters), len(seeds))
raft_energy_tensor = torch.zeros(len(diameters), len(seeds))
bubble_positions_tensor = torch.zeros(len(diameters), len(seeds), N, 2)
for ii, diam in enumerate(diameters):
    for jj, seed in enumerate(seeds):
        box_size, raft_energy, bubble_positions = simulation(diam, seed)
        box_size_tensor[ii, jj] = box_size
        raft_energy_tensor[ii, jj] = raft_energy.detach()
        bubble_positions_tensor[ii, jj] = bubble_positions
    print(
        f"Finished simulation for diameter {diam}, final energy: {raft_energy.detach()}"
    )
# %%
U_mean = torch.mean(raft_energy_tensor, dim=1)
U_std = torch.std(raft_energy_tensor, dim=1)
plt.plot(diameters.detach().numpy(), U_mean, linewidth=3)
plt.fill_between(diameters.detach().numpy(), U_mean + U_std, U_mean - U_std, alpha=0.4)

plt.xlim([0.4, 1.0])
plt.xlabel(r"$D$", fontsize=20)
plt.ylabel(r"$U$", fontsize=20)
plt.show()
# %%
marker_size = 185
for ii, diam in enumerate(diameters):
    plt.subplot(2, 5, ii + 1)
    c = min(1, max(0, (U_mean[ii].detach().numpy() - 0.4) * 4))
    color = [c, 0, 1 - c]
    draw_system(
        bubble_positions_tensor[ii, 0, :N_2].detach().numpy(),
        box_size_tensor[ii, 0].detach().numpy(),
        diam * marker_size,
        color=color,
    )
    draw_system(
        bubble_positions_tensor[ii, 0, N_2:].detach().numpy(),
        box_size_tensor[ii, 0].detach().numpy(),
        marker_size,
        color=color,
    )


# %% [markdown]
"""## Meta-optimization with differentiable simulation."""

# %%
short_simulation_steps = 10


def short_simulation(
    diameter: torch.Tensor, R: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    diameter = diameter.requires_grad_(True)
    box_size = box_size_at_packing_fraction(diameter, packing_fraction)
    cell = torch.eye(2) * box_size
    # Create the energy function.
    sigma = species_sigma(diameter)
    model = SoftSphereMultiModel(sigma_matrix=sigma, species=species)

    # Minimize to the nearest minimum.
    custom_state = ts.SimState(
        atomic_numbers=species,
        masses=torch.ones(N),
        system_idx=torch.arange(N),
        positions=R,
        cell=cell,
        pbc=True,
    )
    state = ts.gradient_descent_init(model, state=custom_state)

    for i in range(short_simulation_steps):
        state = ts.gradient_descent_step(model, state, pos_lr=0.1)

    grad_outputs: list[torch.Tensor | None] = [
        torch.ones_like(
            diameter,
        )
    ]
    grad = torch.autograd.grad(
        outputs=[model(state)["energy"]],
        inputs=[diameter],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=False,
    )

    dU_dd = grad[0]
    return model(state)["energy"], dU_dd


# %%
dU_dD = torch.zeros(len(diameters), len(seeds))
for ii, diam in enumerate(diameters):
    for jj, seed in enumerate(seeds):
        _, dU_dD[ii, jj] = short_simulation(diam, bubble_positions_tensor[ii, jj])

# %%
plt.subplot(2, 1, 1)
dU_dD = dU_dD.detach()
dU_mean = torch.mean(dU_dD, dim=1)
dU_std = torch.std(dU_dD, dim=1)
plt.plot(diameters.detach().numpy(), dU_mean, linewidth=3)
plt.fill_between(
    diameters.detach().numpy(), dU_mean + dU_std, dU_mean - dU_std, alpha=0.4
)


plt.xlim([0.4, 1.0])
plt.xlabel(r"$D$", fontsize=20)
plt.ylabel(r"$\langle{dU}/{dD}\rangle$", fontsize=20)

plt.subplot(2, 1, 2)
plt.plot(diameters.detach().numpy(), U_mean, linewidth=3)
plt.fill_between(diameters.detach().numpy(), U_mean + U_std, U_mean - U_std, alpha=0.4)

plt.xlim([0.4, 1.0])
plt.xlabel(r"$D$", fontsize=20)
plt.ylabel(r"$U$", fontsize=20)
