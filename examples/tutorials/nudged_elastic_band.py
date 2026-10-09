"""Tutorial: run batched torch-sim Nudged Elastic Band calculations."""
# ruff: noqa: D101, D102, D103, D107

# %%
# /// script
# dependencies = [
#     "ase",
#     "matplotlib",
# ]
# ///

# %%
from dataclasses import dataclass
from time import perf_counter
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
    calculate_neb_forces,
    interpolate_path,
)

# %% [markdown]
"""
# Nudged Elastic Band

The nudged elastic band (NEB) method finds a minimum-energy pathway between two
endpoints. The useful TorchSim pattern is not just "run one NEB", but "run many
related NEBs together": each path is an optimizer group inside one batched
`SimState`, so model evaluations and optimizer bookkeeping happen together.

ASE is kept in this tutorial as a reference implementation. It makes the result
easier to trust, and it gives a clear baseline for reporting the speedup from
batching several independent paths in TorchSim.
"""


# %%
@dataclass(frozen=True)
class NEBCase:
    name: str
    barrier_height: float
    valley_scale: float
    valley_curve: float


def curved_double_well(
    positions: torch.Tensor,
    barrier_height: torch.Tensor | float,
    valley_scale: torch.Tensor | float,
    valley_curve: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    u = x**2 - 1.0
    v = y - valley_curve * u
    energy = barrier_height * u**2 + valley_scale * v**2 + z**2
    dE_dx = 4.0 * barrier_height * x * u - 4.0 * valley_scale * valley_curve * x * v
    dE_dy = 2.0 * valley_scale * v
    dE_dz = 2.0 * z
    forces = -torch.stack([dE_dx, dE_dy, dE_dz], dim=1)
    return energy, forces


# %% [markdown]
"""
The analytic surface below keeps the tutorial fast and reproducible. Every case
has minima at `x = -1` and `x = 1`, but each one has a different valley shape.
That gives us several independent NEB calculations with different barriers and
curvatures.
"""


# %%
class TorchBatchedDoubleWellModel(ModelInterface):
    def __init__(
        self,
        cases: list[NEBCase],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = True
        self.cases = cases
        self.valley_scale = torch.tensor(
            [case.valley_scale for case in cases], device=device, dtype=dtype
        )
        self.valley_curve = torch.tensor(
            [case.valley_curve for case in cases], device=device, dtype=dtype
        )
        self.barrier_height = torch.tensor(
            [case.barrier_height for case in cases], device=device, dtype=dtype
        )

    def forward(self, state: ts.SimState, **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        case_idx = state.group_idx[state.system_idx]
        per_atom_energy, forces = curved_double_well(
            state.positions,
            self.barrier_height[case_idx],
            self.valley_scale[case_idx],
            self.valley_curve[case_idx],
        )
        energy = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
        energy.scatter_add_(0, state.system_idx, per_atom_energy)
        return {
            "energy": energy,
            "forces": forces,
            "stress": torch.zeros(
                state.n_systems, 3, 3, device=state.device, dtype=state.dtype
            ),
        }


class ASEDoubleWellCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, case: NEBCase) -> None:
        super().__init__()
        self.case = case

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = torch.tensor(self.atoms.positions, dtype=torch.float64)
        per_atom_energy, forces = curved_double_well(
            positions,
            self.case.barrier_height,
            self.case.valley_scale,
            self.case.valley_curve,
        )
        self.results["energy"] = float(per_atom_energy.sum().item())
        self.results["forces"] = forces.detach().cpu().numpy()


# %% [markdown]
"""
The same potential is exposed through two interfaces. TorchSim gets a vectorized
model that chooses the right parameters from `group_idx`; ASE gets one calculator
per case. In a real workflow, the TorchSim model would usually be an ML
interatomic potential and the groups would be different reactions, defects, or
starting guesses.
"""


# %%
def make_state(position: tuple[float, float, float], device: torch.device) -> ts.SimState:
    return ts.SimState(
        positions=torch.tensor([position], device=device, dtype=torch.float64),
        masses=torch.ones(1, device=device, dtype=torch.float64),
        cell=torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0) * 10.0,
        pbc=False,
        atomic_numbers=torch.tensor([18], device=device),
        system_idx=torch.zeros(1, device=device, dtype=torch.long),
    )


def make_endpoint_batch(
    cases: list[NEBCase],
    position: tuple[float, float, float],
    device: torch.device,
) -> ts.SimState:
    states = [make_state(position, device) for _ in cases]
    return ts.concatenate_states(states)


def state_for_group(state: ts.SimState, group_idx: int) -> ts.SimState:
    system_indices = torch.where(state.group_idx == group_idx)[0]
    return state[system_indices]


def interpolate_batched_paths(
    initial_state: ts.SimState,
    final_state: ts.SimState,
    n_images: int,
) -> ts.SimState:
    paths = [
        interpolate_path(
            state_for_group(initial_state, group_idx),
            state_for_group(final_state, group_idx),
            n_images,
        )
        for group_idx in range(initial_state.n_groups)
    ]
    return ts.concatenate_states(paths)


def assemble_batched_paths(
    initial_state: ts.SimState,
    movable_state: ts.SimState,
    final_state: ts.SimState,
) -> ts.SimState:
    paths = []
    for group_idx in range(initial_state.n_groups):
        path = assemble_path(
            state_for_group(initial_state, group_idx),
            state_for_group(movable_state, group_idx),
            state_for_group(final_state, group_idx),
        )
        path.group_idx = torch.zeros(path.n_systems, device=path.device, dtype=torch.long)
        paths.append(path)
    return ts.concatenate_states(paths)


def relative_energies_by_group(
    state: ts.SimState,
    model: ModelInterface,
    n_groups: int,
) -> np.ndarray:
    energies = model(state)["energy"].detach().cpu().numpy()
    profiles = energies.reshape(n_groups, -1)
    return profiles - profiles[:, :1]


def store_batched_neb_forces(
    state: ts.OptimState,
    neb_forces: torch.Tensor,
) -> None:
    max_force_by_group = torch.zeros(
        state.n_groups, device=state.device, dtype=state.dtype
    )
    atom_group_idx = state.group_idx[state.system_idx]
    for group_idx in range(state.n_groups):
        group_mask = atom_group_idx == group_idx
        max_force_by_group[group_idx] = torch.linalg.norm(
            neb_forces[group_mask], dim=1
        ).max()
    state.forces = neb_forces
    state.neb_forces = neb_forces
    state.neb_max_force_by_group = max_force_by_group
    state.neb_max_force = max_force_by_group.max()


def calculate_batched_neb_forces(
    state: ts.SimState,
    true_forces: torch.Tensor,
    true_energies: torch.Tensor,
    initial_state: ts.SimState,
    final_state: ts.SimState,
    initial_energies: torch.Tensor,
    final_energies: torch.Tensor,
    *,
    spring_constant: float,
    use_climbing_image: bool,
) -> torch.Tensor:
    neb_forces = torch.zeros_like(true_forces)
    atom_group_idx = state.group_idx[state.system_idx]
    for group_idx in range(initial_state.n_groups):
        system_indices = torch.where(state.group_idx == group_idx)[0]
        atom_mask = atom_group_idx == group_idx
        path_state = assemble_path(
            state_for_group(initial_state, group_idx),
            state[system_indices],
            state_for_group(final_state, group_idx),
        )
        neb_forces[atom_mask] = calculate_neb_forces(
            path_state,
            true_forces[atom_mask],
            true_energies[system_indices],
            initial_energies[group_idx],
            final_energies[group_idx],
            spring_constant=spring_constant,
            use_climbing_image=use_climbing_image,
        )
    return neb_forces


def batched_neb_init(
    state: ts.SimState,
    model: ModelInterface,
    *,
    initial_state: ts.SimState,
    final_state: ts.SimState,
    initial_energies: torch.Tensor,
    final_energies: torch.Tensor,
    spring_constant: float,
    use_climbing_image: bool,
) -> ts.OptimState:
    opt_state = fire_init(state, model, fire_flavor="ase_fire")
    neb_forces = calculate_batched_neb_forces(
        opt_state,
        opt_state.forces,
        opt_state.energy,
        initial_state,
        final_state,
        initial_energies,
        final_energies,
        spring_constant=spring_constant,
        use_climbing_image=use_climbing_image,
    )
    store_batched_neb_forces(opt_state, neb_forces)
    return opt_state


def batched_neb_step(
    state: ts.OptimState,
    model: ModelInterface,
    *,
    initial_state: ts.SimState,
    final_state: ts.SimState,
    initial_energies: torch.Tensor,
    final_energies: torch.Tensor,
    spring_constant: float,
    use_climbing_image: bool,
) -> ts.OptimState:
    state = fire_step(state, model, fire_flavor="ase_fire")
    true_forces = state.forces.clone()
    neb_forces = calculate_batched_neb_forces(
        state,
        true_forces,
        state.energy,
        initial_state,
        final_state,
        initial_energies,
        final_energies,
        spring_constant=spring_constant,
        use_climbing_image=use_climbing_image,
    )
    state.true_forces = true_forces
    store_batched_neb_forces(state, neb_forces)
    return state


def run_torch_sim_batched_neb(
    initial_state: ts.SimState,
    final_state: ts.SimState,
    model: ModelInterface,
    *,
    n_images: int,
    spring_constant: float,
    max_steps: int,
    fmax: float,
) -> tuple[ts.SimState, list[np.ndarray], list[np.ndarray], float]:
    start_time = perf_counter()
    movable_images = interpolate_batched_paths(initial_state, final_state, n_images)
    initial_energies = model(initial_state)["energy"]
    final_energies = model(final_state)["energy"]
    energy_history: list[np.ndarray] = []
    max_force_history: list[np.ndarray] = []

    state = batched_neb_init(
        movable_images,
        model,
        initial_state=as_sim_state(initial_state),
        final_state=as_sim_state(final_state),
        initial_energies=initial_energies,
        final_energies=final_energies,
        spring_constant=spring_constant,
        use_climbing_image=True,
    )

    def record(current_state: ts.OptimState) -> None:
        full_path = assemble_batched_paths(initial_state, current_state, final_state)
        energy_history.append(
            relative_energies_by_group(full_path, model, initial_state.n_groups)
        )
        max_force_history.append(
            current_state.neb_max_force_by_group.detach().cpu().numpy()
        )

    record(state)
    for _ in range(max_steps):
        state = batched_neb_step(
            state,
            model,
            initial_state=as_sim_state(initial_state),
            final_state=as_sim_state(final_state),
            initial_energies=initial_energies,
            final_energies=final_energies,
            spring_constant=spring_constant,
            use_climbing_image=True,
        )
        record(state)
        if bool((state.neb_max_force_by_group < fmax).all()):
            break

    elapsed = perf_counter() - start_time
    final_path = assemble_batched_paths(initial_state, state, final_state)
    return final_path, energy_history, max_force_history, elapsed


def run_ase_neb(
    initial_atoms: Atoms,
    final_atoms: Atoms,
    *,
    case: NEBCase,
    n_images: int,
    spring_constant: float,
    max_steps: int,
    fmax: float,
) -> tuple[list[Atoms], list[np.ndarray], list[float], float]:
    start_time = perf_counter()
    images = [initial_atoms.copy()]
    images.extend(initial_atoms.copy() for _ in range(n_images))
    images.append(final_atoms.copy())
    for image in images:
        image.calc = ASEDoubleWellCalculator(case)

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
    elapsed = perf_counter() - start_time
    return images, energy_history, max_force_history, elapsed


# %% [markdown]
"""
Now set up a small batch. Each case is one independent NEB calculation with the
same endpoints and a different curved valley. TorchSim will optimize all of
these paths together; ASE will run the same cases sequentially.
"""

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cases = [
    NEBCase("0.8 eV", barrier_height=0.8, valley_scale=5.0, valley_curve=0.50),
    NEBCase("0.9 eV", barrier_height=0.9, valley_scale=3.0, valley_curve=0.25),
    NEBCase("1.0 eV", barrier_height=1.0, valley_scale=8.0, valley_curve=0.35),
    NEBCase("1.1 eV", barrier_height=1.1, valley_scale=5.5, valley_curve=-0.45),
    NEBCase("1.2 eV", barrier_height=1.2, valley_scale=7.0, valley_curve=0.70),
    NEBCase("1.3 eV", barrier_height=1.3, valley_scale=4.0, valley_curve=-0.75),
    NEBCase("1.4 eV", barrier_height=1.4, valley_scale=2.5, valley_curve=0.60),
    NEBCase("1.5 eV", barrier_height=1.5, valley_scale=9.0, valley_curve=-0.25),
]
n_images = 7
spring_constant = 0.1
max_steps = 200
fmax = 0.03

initial_state = make_endpoint_batch(cases, (-1.0, 0.0, 0.0), device)
final_state = make_endpoint_batch(cases, (1.0, 0.0, 0.0), device)
model = TorchBatchedDoubleWellModel(cases, device=device, dtype=torch.float64)

initial_atoms = Atoms("Ar", positions=[[-1.0, 0.0, 0.0]], cell=np.eye(3) * 10.0)
final_atoms = Atoms("Ar", positions=[[1.0, 0.0, 0.0]], cell=np.eye(3) * 10.0)


# %% [markdown]
"""
The TorchSim call below is the main workflow. The movable images for every case
are stored in one state, and FIRE uses `group_idx` so each NEB keeps its own
optimizer state while still sharing one batched model call per step.
"""

# %%
torch_path, torch_energy_history, torch_fmax, torch_elapsed = run_torch_sim_batched_neb(
    initial_state,
    final_state,
    model,
    n_images=n_images,
    spring_constant=spring_constant,
    max_steps=max_steps,
    fmax=fmax,
)


# %% [markdown]
"""
ASE is the validation baseline. It does not batch these independent NEBs here,
so we run the same cases one after another and compare final profiles.
"""

# %%
ase_energy_history: list[list[np.ndarray]] = []
ase_fmax: list[list[float]] = []
ase_elapsed_by_case: list[float] = []
ase_start = perf_counter()
for case in cases:
    _, case_energy_history, case_fmax, case_elapsed = run_ase_neb(
        initial_atoms,
        final_atoms,
        case=case,
        n_images=n_images,
        spring_constant=spring_constant,
        max_steps=max_steps,
        fmax=fmax,
    )
    ase_energy_history.append(case_energy_history)
    ase_fmax.append(case_fmax)
    ase_elapsed_by_case.append(case_elapsed)
ase_elapsed = perf_counter() - ase_start


# %% [markdown]
"""
Finally, compare accuracy and runtime. The per-case barrier differences should
be tiny; once that is true, the runtime comparison shows why batching many NEBs
together is the more relevant TorchSim workflow.
"""

# %%
n_cases = len(cases)
reaction_coordinate = np.linspace(0.0, 1.0, n_images + 2)
torch_final = relative_energies_by_group(torch_path, model, n_cases)
ase_final = np.stack([history[-1] for history in ase_energy_history])
barrier_difference = np.abs(torch_final.max(axis=1) - ase_final.max(axis=1))
max_barrier_difference = barrier_difference.max()
speedup = ase_elapsed / torch_elapsed if torch_elapsed > 0 else float("inf")

print("Final barrier comparison")
print("case         torch-sim      ASE          abs diff")
for case, torch_profile, ase_profile, diff in zip(
    cases, torch_final, ase_final, barrier_difference, strict=True
):
    print(
        f"{case.name:10s}  {torch_profile.max(): .8f}  {ase_profile.max(): .8f}  "
        f"{diff:.3e}"
    )
print(f"Max barrier difference: {max_barrier_difference:.3e} eV")
print(f"torch-sim batched runtime: {torch_elapsed:.3f} s")
print(f"ASE sequential runtime: {ase_elapsed:.3f} s")
print(f"torch-sim speedup vs sequential ASE: {speedup:.2f}x")


# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
colors = plt.cm.viridis(np.linspace(0.05, 0.95, n_cases))

for idx, (case, color) in enumerate(zip(cases, colors, strict=True)):
    axes[0, 0].plot(
        reaction_coordinate,
        torch_final[idx],
        color=color,
        linewidth=2,
        label=case.name,
    )
    axes[0, 0].plot(reaction_coordinate, ase_final[idx], "o", color=color, markersize=3)
axes[0, 0].set_ylabel("Relative energy")
axes[0, 0].set_title("Final profiles: lines are torch-sim, dots are ASE")
axes[0, 0].legend(fontsize=8)

axes[0, 1].bar([case.name for case in cases], barrier_difference)
axes[0, 1].set_ylabel("Barrier |torch-sim - ASE|")
axes[0, 1].set_yscale("log")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].set_title("Validation error by NEB case")

torch_force_history = np.stack(torch_fmax)
axes[1, 0].plot(
    torch_force_history.max(axis=1),
    color="tab:blue",
    linestyle="--",
    linewidth=2,
    label="torch-sim batch max",
)
for case, case_fmax in zip(cases, ase_fmax, strict=True):
    axes[1, 0].plot(case_fmax, alpha=0.35, linewidth=1, label=f"ASE {case.name}")
axes[1, 0].axhline(fmax, color="k", linestyle=":", label="fmax")
axes[1, 0].set_xlabel("Optimization step")
axes[1, 0].set_ylabel("Max NEB force")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Convergence")
axes[1, 0].legend(fontsize=7)

axes[1, 1].bar(["torch-sim batched", "ASE sequential"], [torch_elapsed, ase_elapsed])
axes[1, 1].set_ylabel("Runtime (s)")
axes[1, 1].set_title(f"Batching speedup: {speedup:.2f}x")

fig.tight_layout()
fig.savefig("neb_ase_torchsim_comparison.png", dpi=200)
print("Saved comparison plot to neb_ase_torchsim_comparison.png")
