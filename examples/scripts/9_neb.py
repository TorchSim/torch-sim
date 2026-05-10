"""Compare torch-sim and ASE Nudged Elastic Band trajectories."""
# ruff: noqa: D101, D102, D103, D107

# %%
# /// script
# dependencies = [
#     "ase",
#     "matplotlib",
# ]
# ///

from dataclasses import dataclass
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.mep import NEB as ASENEB
from ase.optimize import FIRE

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import fire_init, fire_step
from torch_sim.workflows.neb import (
    as_sim_state,
    assemble_path,
    interpolate_path,
    neb_convergence_fn,
    neb_init,
    neb_step,
)


@dataclass(frozen=True)
class CurvedDoubleWellParams:
    valley_scale: float = 5.0
    valley_curve: float = 0.5


def curved_double_well(
    positions: torch.Tensor, params: CurvedDoubleWellParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-atom energies and forces for a curved double-well surface."""
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    u = x**2 - 1.0
    v = y - params.valley_curve * u
    energy = u**2 + params.valley_scale * v**2 + z**2
    dE_dx = 4.0 * x * u - 4.0 * params.valley_scale * params.valley_curve * x * v
    dE_dy = 2.0 * params.valley_scale * v
    dE_dz = 2.0 * z
    forces = -torch.stack([dE_dx, dE_dy, dE_dz], dim=1)
    return energy, forces


class TorchCurvedDoubleWellModel(ModelInterface):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        params: CurvedDoubleWellParams,
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = True
        self.params = params

    def forward(self, state: ts.SimState, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        per_atom_energy, forces = curved_double_well(state.positions, self.params)
        energy = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
        energy.scatter_add_(0, state.system_idx, per_atom_energy)
        return {
            "energy": energy,
            "forces": forces,
            "stress": torch.zeros(
                state.n_systems, 3, 3, device=state.device, dtype=state.dtype
            ),
        }


class ASECurvedDoubleWellCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, params: CurvedDoubleWellParams) -> None:
        super().__init__()
        self.params = params

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = torch.tensor(self.atoms.positions, dtype=torch.float64)
        per_atom_energy, forces = curved_double_well(positions, self.params)
        self.results["energy"] = float(per_atom_energy.sum().item())
        self.results["forces"] = forces.detach().cpu().numpy()


def make_state(
    position: tuple[float, float, float], *, device: torch.device
) -> ts.SimState:
    return ts.SimState(
        positions=torch.tensor([position], device=device, dtype=torch.float64),
        masses=torch.ones(1, device=device, dtype=torch.float64),
        cell=torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0) * 10.0,
        pbc=False,
        atomic_numbers=torch.tensor([18], device=device),
        system_idx=torch.zeros(1, device=device, dtype=torch.long),
    )


def relative_energies_torch(state: ts.SimState, model: ModelInterface) -> np.ndarray:
    energies = model(state)["energy"].detach().cpu().numpy()
    return energies - energies[0]


def run_torch_sim_neb(
    initial_state: ts.SimState,
    final_state: ts.SimState,
    model: ModelInterface,
    *,
    n_images: int,
    spring_constant: float,
    max_steps: int,
    fmax: float,
) -> tuple[ts.SimState, list[np.ndarray], list[float]]:
    movable_images = interpolate_path(initial_state, final_state, n_images)
    endpoint_output = model(
        ts.concatenate_states([as_sim_state(initial_state), as_sim_state(final_state)])
    )
    endpoint_kwargs = {
        "initial_state": as_sim_state(initial_state),
        "final_state": as_sim_state(final_state),
        "initial_energy": endpoint_output["energy"][0],
        "final_energy": endpoint_output["energy"][1],
        "spring_constant": spring_constant,
        "use_climbing_image": True,
    }
    energy_history: list[np.ndarray] = []
    max_force_history: list[float] = []

    def record(state: ts.SimState) -> None:
        full_path = assemble_path(initial_state, state, final_state)
        energy_history.append(relative_energies_torch(full_path, model))
        max_force_history.append(float(torch.linalg.norm(state.forces, dim=1).max()))

    def convergence(state: ts.OptimState, last_energy: torch.Tensor) -> torch.Tensor:
        record(state)
        return neb_convergence_fn(state, last_energy, fmax=fmax)

    initial_opt_state = neb_init(
        movable_images,
        model,
        **endpoint_kwargs,
        base_init_fn=fire_init,
        base_init_kwargs={"fire_flavor": "ase_fire"},
    )
    record(initial_opt_state)

    final_movable = ts.optimize(
        movable_images,
        model,
        optimizer=(neb_init, neb_step),
        convergence_fn=convergence,
        max_steps=max_steps,
        steps_between_swaps=1,
        autobatcher=False,
        init_kwargs={
            **endpoint_kwargs,
            "base_init_fn": fire_init,
            "base_init_kwargs": {"fire_flavor": "ase_fire"},
        },
        **endpoint_kwargs,
        base_step_fn=fire_step,
        base_step_kwargs={"fire_flavor": "ase_fire"},
    )
    final_path = assemble_path(initial_state, final_movable, final_state)
    return final_path, energy_history, max_force_history


def run_ase_neb(
    initial_atoms: Atoms,
    final_atoms: Atoms,
    *,
    params: CurvedDoubleWellParams,
    n_images: int,
    spring_constant: float,
    max_steps: int,
    fmax: float,
) -> tuple[list[Atoms], list[np.ndarray], list[float]]:
    images = [initial_atoms.copy()]
    images.extend(initial_atoms.copy() for _ in range(n_images))
    images.append(final_atoms.copy())
    for image in images:
        image.calc = ASECurvedDoubleWellCalculator(params)

    neb = ASENEB(images, k=spring_constant, climb=True, method="improvedtangent")
    neb.interpolate(mic=True)
    optimizer = FIRE(neb, logfile=None)
    energy_history: list[np.ndarray] = []
    max_force_history: list[float] = []

    def record() -> None:
        energies = np.array([image.get_potential_energy() for image in images])
        energy_history.append(energies - energies[0])
        forces = neb.get_forces().reshape(-1, 3)
        max_force_history.append(float(np.linalg.norm(forces, axis=1).max()))

    optimizer.attach(record, interval=1)
    optimizer.run(fmax=fmax, steps=max_steps)
    return images, energy_history, max_force_history


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = CurvedDoubleWellParams()
n_images = 7
spring_constant = 0.1
max_steps = 200
fmax = 0.03

initial_state = make_state((-1.0, 0.0, 0.0), device=device)
final_state = make_state((1.0, 0.0, 0.0), device=device)
model = TorchCurvedDoubleWellModel(device=device, dtype=torch.float64, params=params)

initial_atoms = Atoms("Ar", positions=[[-1.0, 0.0, 0.0]], cell=np.eye(3) * 10.0)
final_atoms = Atoms("Ar", positions=[[1.0, 0.0, 0.0]], cell=np.eye(3) * 10.0)

torch_path, torch_energy_history, torch_fmax = run_torch_sim_neb(
    initial_state,
    final_state,
    model,
    n_images=n_images,
    spring_constant=spring_constant,
    max_steps=max_steps,
    fmax=fmax,
)
ase_images, ase_energy_history, ase_fmax = run_ase_neb(
    initial_atoms,
    final_atoms,
    params=params,
    n_images=n_images,
    spring_constant=spring_constant,
    max_steps=max_steps,
    fmax=fmax,
)

torch_final = relative_energies_torch(torch_path, model)
ase_final = ase_energy_history[-1]
reaction_coordinate = np.linspace(0.0, 1.0, n_images + 2)

print("Final relative energies (eV)")
print("image  torch-sim      ASE          abs diff")
for idx, (torch_energy, ase_energy) in enumerate(
    zip(torch_final, ase_final, strict=True)
):
    print(
        f"{idx:5d}  {torch_energy: .8f}  {ase_energy: .8f}  "
        f"{abs(torch_energy - ase_energy):.3e}"
    )
print(f"Barrier difference: {abs(torch_final.max() - ase_final.max()):.3e} eV")

common_steps = min(len(torch_fmax), len(ase_fmax))
step_axis = np.arange(common_steps)
final_energy_residual = torch_final - ase_final
force_residual = np.array(torch_fmax[:common_steps]) - np.array(ase_fmax[:common_steps])

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0, 0].plot(reaction_coordinate, torch_final, "o-", label="torch-sim")
axes[0, 0].plot(reaction_coordinate, ase_final, "s--", label="ASE")
axes[0, 0].set_ylabel("Relative energy")
axes[0, 0].set_title("Final NEB profile")
axes[0, 0].legend()

axes[0, 1].plot(torch_fmax, label="torch-sim")
axes[0, 1].plot(ase_fmax, label="ASE")
axes[0, 1].axhline(fmax, color="k", linestyle=":", label="fmax")
axes[0, 1].set_ylabel("Max NEB force")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Convergence")
axes[0, 1].legend()

axes[1, 0].axhline(0.0, color="k", linewidth=0.8)
axes[1, 0].plot(reaction_coordinate, final_energy_residual, "o-")
axes[1, 0].set_xlabel("Reaction coordinate")
axes[1, 0].set_ylabel("TS - ASE")
axes[1, 0].set_title("Final energy residual")

axes[1, 1].axhline(0.0, color="k", linewidth=0.8)
axes[1, 1].plot(step_axis, force_residual)
axes[1, 1].set_xlabel("Optimization step")
axes[1, 1].set_ylabel("TS - ASE")
axes[1, 1].set_title("Max-force residual")

fig.tight_layout()
fig.savefig("neb_ase_torchsim_comparison.png", dpi=200)
print("Saved comparison plot to neb_ase_torchsim_comparison.png")
