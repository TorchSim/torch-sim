"""Compare plain ASE FIRE and torch-sim ase_fire on one analytic system."""
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
from ase.optimize import FIRE

import torch_sim as ts
from torch_sim.models.interface import ModelInterface


@dataclass(frozen=True)
class PotentialParams:
    valley_scale: float = 5.0
    valley_curve: float = 0.5


def energy_forces(
    positions: torch.Tensor, params: PotentialParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-atom energies and forces for a curved double well."""
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


class TorchModel(ModelInterface):
    def __init__(self, params: PotentialParams) -> None:
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float64
        self._compute_forces = True
        self._compute_stress = True
        self.params = params

    def forward(self, state: ts.SimState, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        per_atom_energy, forces = energy_forces(state.positions, self.params)
        energy = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
        energy.scatter_add_(0, state.system_idx, per_atom_energy)
        return {
            "energy": energy,
            "forces": forces,
            "stress": torch.zeros(
                state.n_systems, 3, 3, device=state.device, dtype=state.dtype
            ),
        }


class ASECalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, params: PotentialParams) -> None:
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
        per_atom_energy, forces = energy_forces(positions, self.params)
        self.results["energy"] = float(per_atom_energy.sum())
        self.results["forces"] = forces.detach().cpu().numpy()


def make_state(position: tuple[float, float, float]) -> ts.SimState:
    return ts.SimState(
        positions=torch.tensor([position], dtype=torch.float64),
        masses=torch.ones(1, dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
        pbc=False,
        atomic_numbers=torch.tensor([18]),
        system_idx=torch.zeros(1, dtype=torch.long),
    )


def run_torch_fire(
    state: ts.SimState, model: ModelInterface, *, steps: int, fmax: float
) -> tuple[ts.SimState, list[float], list[float]]:
    energy_history: list[float] = []
    fmax_history: list[float] = []

    def record(state: ts.OptimState) -> None:
        energy_history.append(float(state.energy[0]))
        fmax_history.append(float(torch.linalg.norm(state.forces, dim=1).max()))

    initial_opt_state = ts.fire_init(state, model, fire_flavor="ase_fire")
    record(initial_opt_state)

    def convergence_fn(state: ts.OptimState, last_energy: torch.Tensor) -> torch.Tensor:
        del last_energy
        record(state)
        return ts.generate_force_convergence_fn(force_tol=fmax)(state, state.energy)

    result = ts.optimize(
        state,
        model,
        optimizer=ts.Optimizer.fire,
        convergence_fn=convergence_fn,
        max_steps=steps,
        steps_between_swaps=1,
        autobatcher=False,
        fire_flavor="ase_fire",
    )
    return result, energy_history, fmax_history


def run_ase_fire(
    atoms: Atoms, *, params: PotentialParams, steps: int, fmax: float
) -> tuple[Atoms, list[float], list[float]]:
    atoms = atoms.copy()
    atoms.calc = ASECalculator(params)
    optimizer = FIRE(atoms, logfile=None)
    energy_history: list[float] = []
    fmax_history: list[float] = []

    def record() -> None:
        energy_history.append(float(atoms.get_potential_energy()))
        fmax_history.append(float(np.linalg.norm(atoms.get_forces(), axis=1).max()))

    optimizer.attach(record, interval=1)
    optimizer.run(fmax=fmax, steps=steps)
    return atoms, energy_history, fmax_history


params = PotentialParams()
steps = 80
fmax = 0.03
initial_position = (-0.2, 0.9, 0.0)
model = TorchModel(params)
state = make_state(initial_position)
atoms = Atoms("Ar", positions=[initial_position], cell=np.eye(3) * 10.0, pbc=False)

ts_final, ts_energy, ts_force = run_torch_fire(state, model, steps=steps, fmax=fmax)
ase_final, ase_energy, ase_force = run_ase_fire(
    atoms, params=params, steps=steps, fmax=fmax
)

ts_position = ts_final.positions.detach().cpu().numpy()[0]
ase_position = ase_final.positions[0]
print(f"torch-sim steps: {len(ts_force)}")
print(f"ASE steps:       {len(ase_force)}")
print(f"final position abs diff: {np.max(np.abs(ts_position - ase_position)):.3e}")
print(f"final energy abs diff:   {abs(ts_energy[-1] - ase_energy[-1]):.3e}")
print(f"final fmax ts/ase:       {ts_force[-1]:.3e} / {ase_force[-1]:.3e}")

common_steps = min(len(ts_energy), len(ase_energy))
step_axis = np.arange(common_steps)
energy_residual = np.array(ts_energy[:common_steps]) - np.array(ase_energy[:common_steps])
force_residual = np.array(ts_force[:common_steps]) - np.array(ase_force[:common_steps])

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
axes[0, 0].plot(ts_energy, label="torch-sim")
axes[0, 0].plot(ase_energy, "--", label="ASE")
axes[0, 0].set_ylabel("Energy")
axes[0, 0].set_title("Plain FIRE energy")
axes[0, 0].legend()

axes[0, 1].plot(ts_force, label="torch-sim")
axes[0, 1].plot(ase_force, "--", label="ASE")
axes[0, 1].axhline(fmax, color="k", linestyle=":", label="fmax")
axes[0, 1].set_ylabel("Max force")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Plain FIRE convergence")
axes[0, 1].legend()

axes[1, 0].axhline(0.0, color="k", linewidth=0.8)
axes[1, 0].plot(step_axis, energy_residual)
axes[1, 0].set_xlabel("Optimization step")
axes[1, 0].set_ylabel("TS - ASE")
axes[1, 0].set_title("Energy residual")

axes[1, 1].axhline(0.0, color="k", linewidth=0.8)
axes[1, 1].plot(step_axis, force_residual)
axes[1, 1].set_xlabel("Optimization step")
axes[1, 1].set_ylabel("TS - ASE")
axes[1, 1].set_title("Max-force residual")

fig.tight_layout()
fig.savefig("fire_ase_torchsim_comparison.png", dpi=200)
print("Saved comparison plot to fire_ase_torchsim_comparison.png")
