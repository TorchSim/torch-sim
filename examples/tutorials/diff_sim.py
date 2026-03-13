# %% [markdown]
# <details>
#   <summary>Dependencies</summary>
# /// script
# dependencies = [
#     "matplotlib",
# ]
# ///
# </details>

# %%
import typing
import torch
import matplotlib.pyplot as plt
from torch_sim.state import SimState
from torch_sim.models.soft_sphere import (
    SoftSphereMultiModel,
    soft_sphere_pair,
)
from torch_sim.neighbors import torch_nl_n2
from torch_sim.optimizers.gradient_descent import (
    gradient_descent_init,
    gradient_descent_step,
)
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
def finalize_plot(shape: tuple[int, int] = (1, 1)) -> None:
    """Finalize the plot by setting the size and layout."""
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1],
    )
    plt.tight_layout()


def draw_system(
    R: torch.Tensor, box_size: float, marker_size: float, color: list[float] | None = None
) -> None:
    """Draw a system of particles on the plot."""
    if color is None:
        color = [64 / 256] * 3
    ms = marker_size / box_size

    positions = torch.as_tensor(R).detach().cpu()
    x_coords = positions[:, 0].numpy()
    y_coords = positions[:, 1].numpy()

    for x_offset, y_offset in (
        (0.0, 0.0),
        (box_size, 0.0),
        (0.0, box_size),
        (box_size, box_size),
        (-box_size, 0.0),
        (0.0, -box_size),
        (-box_size, -box_size),
    ):
        plt.plot(
            x_coords + x_offset,
            y_coords + y_offset,
            linestyle="none",
            markeredgewidth=3,
            marker="o",
            markersize=float(ms),
            color=color,
            fillstyle="none",
        )

    plt.xlim([0, box_size])
    plt.ylim([0, box_size])
    plt.axis("off")
    plt.gca().set_facecolor((1, 1, 1))


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

dr = torch.linspace(0, 3.0, 80)
_z = torch.zeros_like(dr, dtype=torch.long)
plt.plot(dr, soft_sphere_pair(dr, _z, _z, sigma=1), "b-", linewidth=3)
plt.fill_between(dr, soft_sphere_pair(dr, _z, _z), alpha=0.4)

plt.xlabel(r"$r$", fontsize=20)
plt.ylabel(r"$U(r)$", fontsize=20)

plt.show()

# %% [markdown]
"""
## Setup the simulation environment.
"""


# %%
def box_size_at_number_density(
    particle_count: int, number_density: torch.Tensor
) -> torch.Tensor:
    return (particle_count / number_density) ** 0.5


def box_size_at_packing_fraction(
    diameter: torch.Tensor, packing_fraction: float
) -> torch.Tensor:
    bubble_volume = N_2 * torch.pi * (diameter**2 + 1) / 4
    return torch.sqrt(bubble_volume / packing_fraction)


def species_sigma(diameter: torch.Tensor) -> torch.Tensor:
    d_BB = torch.ones_like(diameter)
    d_AB = 0.5 * (diameter + 1)
    return torch.stack([diameter, d_AB, d_AB, d_BB]).reshape(2, 2)


N = 128
N_2 = N // 2
species = torch.tensor([0] * (N_2) + [1] * (N_2), dtype=torch.long)
simulation_steps = 1000
packing_fraction = 0.98
markersize = 260


# %%
def simulation(
    diameter: torch.Tensor, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_size = box_size_at_packing_fraction(diameter, packing_fraction)
    cell = (torch.eye(3) * box_size).unsqueeze(0)
    sigma = species_sigma(diameter)
    model = SoftSphereMultiModel(
        atomic_numbers=species,
        sigma_matrix=sigma,
        dtype=torch.float32,
        neighbor_list_fn=torch_nl_n2,
        retain_graph=True,
    )
    # Use aot_eager backend as Inductor has issues with scatter operations (index_add/scatter_add)
    model = typing.cast(SoftSphereMultiModel, torch.compile(model, backend="aot_eager"))
    torch.manual_seed(seed)
    R = torch.rand(N, 3) * box_size
    state = SimState(
        positions=R,
        masses=torch.ones(N),
        cell=cell,
        pbc=True,
        atomic_numbers=species,
    )
    state = gradient_descent_init(state, model)
    for _ in range(simulation_steps):
        state = gradient_descent_step(state, model, pos_lr=0.1)
    return box_size, model(state)["energy"], state.positions


# %% [markdown]
"""
## Packing at different diameters.
"""
# %%
plt.subplot(1, 2, 1)

box_size, raft_energy, bubble_positions = simulation(torch.tensor(1.0))
draw_system(bubble_positions, float(box_size), float(markersize))
finalize_plot((1, 1))

plt.subplot(1, 2, 2)

box_size, raft_energy, bubble_positions = simulation(torch.tensor(0.8))
draw_system(bubble_positions[:N_2], float(box_size), 0.8 * markersize)
draw_system(bubble_positions[N_2:], float(box_size), float(markersize))
finalize_plot((2, 1))
# %% [markdown]
"""
## Forward simulation for different diameters and seeds.
"""
# %%
diameters = torch.linspace(0.4, 1.0, 10)
seeds = torch.arange(1, 6)
box_size_tensor = torch.zeros(len(diameters), len(seeds))
raft_energy_tensor = torch.zeros(len(diameters), len(seeds))
bubble_positions_tensor = torch.zeros(len(diameters), len(seeds), N, 3)
for i, d in enumerate(diameters):
    for j, s in enumerate(seeds):
        box_size, raft_energy, bubble_positions = simulation(d, s)
        box_size_tensor[i, j] = box_size
        raft_energy_tensor[i, j] = raft_energy.detach()
        bubble_positions_tensor[i, j] = bubble_positions
    print(f"Finished simulation for diameter {d}, final energy: {raft_energy.detach()}")
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
ms = 185
for i, d in enumerate(diameters):
    plt.subplot(2, 5, i + 1)
    c = min(1, max(0, (U_mean[i].detach().numpy() - 0.4) * 4))
    color = [c, 0, 1 - c]
    draw_system(
        bubble_positions_tensor[i, 0, :N_2].detach().numpy(),
        float(box_size_tensor[i, 0]),
        float(d * ms),
        color=color,
    )
    draw_system(
        bubble_positions_tensor[i, 0, N_2:].detach().numpy(),
        float(box_size_tensor[i, 0]),
        float(ms),
        color=color,
    )

finalize_plot((2, 1))

# %% [markdown]
"""
## Meta-optimization with differentiable simulation.
"""
# %%

short_simulation_steps = 10


def short_simulation(
    diameter: torch.Tensor, R: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    diameter = diameter.requires_grad_(True)
    box_size = box_size_at_packing_fraction(diameter, packing_fraction)
    cell = (torch.eye(3) * box_size).unsqueeze(0)
    sigma = species_sigma(diameter)
    model = SoftSphereMultiModel(
        atomic_numbers=species,
        sigma_matrix=sigma,
        dtype=torch.float32,
        neighbor_list_fn=torch_nl_n2,
        retain_graph=True,
    )
    state = SimState(
        positions=R,
        masses=torch.ones(N),
        cell=cell,
        pbc=True,
        atomic_numbers=species,
    )
    state = gradient_descent_init(state, model)
    for _ in range(short_simulation_steps):
        state = gradient_descent_step(state, model, pos_lr=0.1)
    energy = model(state)["energy"]
    (dU_dd,) = torch.autograd.grad(energy, diameter, create_graph=True)
    return energy, dU_dd


# %%
dU_dD = torch.zeros(len(diameters), len(seeds))
for i, d in enumerate(diameters):
    for j, s in enumerate(seeds):
        _, dU_dD[i, j] = short_simulation(d, bubble_positions_tensor[i, j])

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

finalize_plot((1, 1))
