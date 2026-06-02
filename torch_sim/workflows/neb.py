"""Nudged Elastic Band (NEB) workflow."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import (
    OptimState,
    fire_init,
    fire_step,
    gradient_descent_init,
    gradient_descent_step,
)
from torch_sim.runners import optimize
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.transforms import minimum_image_displacement
from torch_sim.typing import StateLike


logger = logging.getLogger(__name__)

_EPS = torch.finfo(torch.float64).eps

OptimizerType = Literal["fire", "gd", "ase_fire"]


def _extract_kwargs_from_params(
    params: dict[str, Any], func: Callable[..., Any], exclude: set[str] | None = None
) -> dict[str, Any]:
    """Return the entries in ``params`` accepted by ``func``."""
    exclude = exclude or {"state", "model"}
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters and k not in exclude}


@dataclass(frozen=True)
class _OptimizerConfig:
    """Functional optimizer pair and argument modifiers."""

    init_fn: Callable[..., OptimState]
    step_fn: Callable[..., OptimState]
    init_kwargs_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    step_kwargs_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None


_OPTIMIZER_REGISTRY: dict[OptimizerType, _OptimizerConfig] = {
    "fire": _OptimizerConfig(init_fn=fire_init, step_fn=fire_step),
    "gd": _OptimizerConfig(
        init_fn=gradient_descent_init,
        step_fn=gradient_descent_step,
        step_kwargs_modifier=lambda kwargs: (
            kwargs if "pos_lr" in kwargs else {**kwargs, "pos_lr": kwargs.get("lr", 0.01)}
        ),
    ),
    "ase_fire": _OptimizerConfig(
        init_fn=fire_init,
        step_fn=fire_step,
        init_kwargs_modifier=lambda kwargs: (
            kwargs if "fire_flavor" in kwargs else {**kwargs, "fire_flavor": "ase_fire"}
        ),
        step_kwargs_modifier=lambda kwargs: (
            kwargs if "fire_flavor" in kwargs else {**kwargs, "fire_flavor": "ase_fire"}
        ),
    ),
}


def validate_endpoints(initial_state: SimState, final_state: SimState) -> None:
    """Validate that endpoints define a fixed-cell single-chain NEB path."""
    if initial_state.n_systems != 1 or final_state.n_systems != 1:
        raise ValueError("Initial and final states must each contain one system.")
    if initial_state.n_atoms != final_state.n_atoms:
        raise ValueError(
            f"Initial ({initial_state.n_atoms}) and final ({final_state.n_atoms}) "
            "states must have the same number of atoms."
        )
    if not torch.equal(initial_state.atomic_numbers, final_state.atomic_numbers):
        raise ValueError("Initial and final states must have the same atom types.")
    if not torch.equal(initial_state.pbc, final_state.pbc):
        raise ValueError("Initial and final states must have the same PBC setting.")
    if not torch.allclose(initial_state.cell, final_state.cell):
        raise ValueError("Fixed-cell NEB requires matching endpoint cells.")


def interpolate_path(
    initial_state: SimState, final_state: SimState, n_images: int
) -> SimState:
    """Linearly interpolate movable NEB images using the minimum image convention."""
    validate_endpoints(initial_state, final_state)
    if n_images < 1:
        raise ValueError("n_images must be at least 1.")

    n_atoms = initial_state.n_atoms
    displacement = minimum_image_displacement(
        dr=final_state.positions - initial_state.positions,
        cell=initial_state.cell[0],
        pbc=initial_state.pbc,
    ).reshape(n_atoms, 3)
    factors = torch.linspace(
        0.0,
        1.0,
        steps=n_images + 2,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )[1:-1]
    positions = (
        initial_state.positions.unsqueeze(0)
        + factors.view(-1, 1, 1) * displacement.unsqueeze(0)
    ).reshape(-1, 3)
    system_idx = torch.repeat_interleave(
        torch.arange(n_images, device=initial_state.device, dtype=torch.int64),
        repeats=n_atoms,
    )
    return SimState(
        positions=positions,
        masses=initial_state.masses.repeat(n_images),
        cell=initial_state.cell.repeat(n_images, 1, 1),
        pbc=initial_state.pbc,
        atomic_numbers=initial_state.atomic_numbers.repeat(n_images),
        system_idx=system_idx,
        group_idx=torch.zeros(n_images, device=initial_state.device, dtype=torch.int64),
    )


def as_sim_state(state: SimState) -> SimState:
    """Drop optimizer-only fields while preserving the atomistic state."""
    return SimState.from_state(state)


def assemble_path(
    initial_state: SimState, movable_state: SimState, final_state: SimState
) -> SimState:
    """Return the full NEB path as endpoints plus movable images."""
    path = concatenate_states(
        [
            as_sim_state(initial_state),
            as_sim_state(movable_state),
            as_sim_state(final_state),
        ]
    )
    path.group_idx = torch.zeros(path.n_systems, device=path.device, dtype=torch.long)
    return path


def compute_tangents(
    all_positions: torch.Tensor,
    all_energies: torch.Tensor,
    cell: torch.Tensor,
    *,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Compute improved normalized tangents for the intermediate NEB images."""
    n_total_images, n_atoms, _ = all_positions.shape
    n_intermediate = n_total_images - 2
    tangents = torch.zeros(
        (n_intermediate, n_atoms, 3),
        device=all_positions.device,
        dtype=all_positions.dtype,
    )
    displacements = minimum_image_displacement(
        dr=all_positions[1:] - all_positions[:-1],
        cell=cell,
        pbc=pbc,
    ).reshape(n_total_images - 1, n_atoms, 3)
    dE_forward = all_energies[1:] - all_energies[:-1]

    for i in range(n_intermediate):
        image_idx = i + 1
        dR_plus = displacements[image_idx]
        dR_minus = displacements[image_idx - 1]
        dE_plus = dE_forward[image_idx]
        dE_minus = dE_forward[image_idx - 1]

        if dE_plus > 0 and dE_minus > 0:
            tangent = dR_plus
        elif dE_plus < 0 and dE_minus < 0:
            tangent = dR_minus
        else:
            abs_dE_plus = torch.abs(dE_plus)
            abs_dE_minus = torch.abs(dE_minus)
            delta_max = torch.maximum(abs_dE_plus, abs_dE_minus)
            delta_min = torch.minimum(abs_dE_plus, abs_dE_minus)
            if (dE_plus + dE_minus) > 0:
                tangent = dR_plus * delta_max + dR_minus * delta_min
            else:
                tangent = dR_plus * delta_min + dR_minus * delta_max

        norm = torch.linalg.norm(tangent)
        if norm > _EPS:
            tangents[i] = tangent / norm

    return tangents


def calculate_neb_forces(
    path_state: SimState,
    true_forces: torch.Tensor,
    true_energies: torch.Tensor,
    initial_energy: torch.Tensor,
    final_energy: torch.Tensor,
    *,
    spring_constant: float,
    use_climbing_image: bool,
) -> torch.Tensor:
    """Calculate NEB forces for the movable images in a single path."""
    n_total_images = path_state.n_systems
    n_intermediate = n_total_images - 2
    if n_intermediate <= 0:
        raise ValueError("A NEB path must include at least one movable image.")
    if path_state.n_atoms % n_total_images != 0:
        raise ValueError("NEB path images must contain the same number of atoms.")
    if true_energies.shape[0] != n_intermediate:
        raise ValueError(f"{true_energies.shape[0]=} does not match {n_intermediate=}.")

    n_atoms = path_state.n_atoms // n_total_images
    all_positions = path_state.positions.reshape(n_total_images, n_atoms, 3)
    all_energies = torch.cat(
        [initial_energy.reshape(1), true_energies, final_energy.reshape(1)]
    )
    true_forces_by_image = true_forces.reshape(n_intermediate, n_atoms, 3)
    cell = path_state.cell[0]

    tangents = compute_tangents(
        all_positions,
        all_energies,
        cell,
        pbc=path_state.pbc,
    )
    displacements = minimum_image_displacement(
        dr=all_positions[1:] - all_positions[:-1],
        cell=cell,
        pbc=path_state.pbc,
    ).reshape(n_total_images - 1, n_atoms, 3)
    segment_lengths = torch.linalg.norm(displacements, dim=(-1, -2))

    true_dot_tangent = (true_forces_by_image * tangents).sum(dim=(-1, -2), keepdim=True)
    perpendicular_forces = true_forces_by_image - true_dot_tangent * tangents
    spring_magnitude = spring_constant * (segment_lengths[1:] - segment_lengths[:-1])
    spring_forces = spring_magnitude.view(-1, 1, 1) * tangents
    neb_forces = perpendicular_forces + spring_forces

    if use_climbing_image:
        climbing_idx = int(torch.argmax(true_energies).item())
        neb_forces[climbing_idx] = true_forces_by_image[climbing_idx] - (
            2 * true_dot_tangent[climbing_idx] * tangents[climbing_idx]
        )

    return neb_forces.reshape(-1, 3)


def _endpoint_energies(
    initial_state: SimState, final_state: SimState, model: ModelInterface
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        model(as_sim_state(initial_state))["energy"][0],
        model(as_sim_state(final_state))["energy"][0],
    )


def _store_neb_force_metadata(state: OptimState, neb_forces: torch.Tensor) -> None:
    state.forces = neb_forces
    state.neb_forces = neb_forces
    state.neb_max_force = torch.linalg.norm(neb_forces, dim=-1).max()


def neb_init(
    state: SimState,
    model: ModelInterface,
    *,
    initial_state: SimState,
    final_state: SimState,
    initial_energy: torch.Tensor,
    final_energy: torch.Tensor,
    base_init_fn: Callable[..., OptimState],
    base_init_kwargs: dict[str, Any] | None = None,
    spring_constant: float = 0.1,
    use_climbing_image: bool = False,
) -> OptimState:
    """Initialize the base optimizer state and replace true forces with NEB forces."""
    opt_state = base_init_fn(state, model, **(base_init_kwargs or {}))
    full_path = assemble_path(initial_state, opt_state, final_state)
    neb_forces = calculate_neb_forces(
        full_path,
        opt_state.forces,
        opt_state.energy,
        initial_energy,
        final_energy,
        spring_constant=spring_constant,
        use_climbing_image=use_climbing_image,
    )
    _store_neb_force_metadata(opt_state, neb_forces)
    return opt_state


def neb_step(
    state: OptimState,
    model: ModelInterface,
    *,
    initial_state: SimState,
    final_state: SimState,
    initial_energy: torch.Tensor,
    final_energy: torch.Tensor,
    base_step_fn: Callable[..., OptimState],
    base_step_kwargs: dict[str, Any] | None = None,
    spring_constant: float = 0.1,
    use_climbing_image: bool = False,
) -> OptimState:
    """Advance one NEB step by delegating position updates to a base optimizer."""
    state = base_step_fn(state, model, **(base_step_kwargs or {}))
    true_forces = state.forces.clone()
    full_path = assemble_path(initial_state, state, final_state)
    neb_forces = calculate_neb_forces(
        full_path,
        true_forces,
        state.energy,
        initial_energy,
        final_energy,
        spring_constant=spring_constant,
        use_climbing_image=use_climbing_image,
    )
    state.true_forces = true_forces
    _store_neb_force_metadata(state, neb_forces)
    return state


def neb_convergence_fn(
    state: OptimState, last_energy: torch.Tensor, *, fmax: float
) -> torch.Tensor:
    """Return all-or-nothing NEB convergence for the movable images."""
    del last_energy
    converged = torch.linalg.norm(state.forces, dim=-1).max() < fmax
    return converged.expand(state.n_systems)


@dataclass
class NEB:
    """Single-chain Nudged Elastic Band workflow."""

    model: ModelInterface
    n_images: int
    spring_constant: float = 0.1
    use_climbing_image: bool = False
    optimizer_type: OptimizerType = "ase_fire"
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    trajectory_filename: str | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Initialize device, dtype, and optimizer configuration."""
        if self.device is None:
            self.device = self.model.device
        if self.dtype is None:
            self.dtype = self.model.dtype
        if self.optimizer_type not in _OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unsupported optimizer_type={self.optimizer_type!r}; expected one of "
                f"{list(_OPTIMIZER_REGISTRY)}."
            )

        config = _OPTIMIZER_REGISTRY[self.optimizer_type]
        init_kwargs = _extract_kwargs_from_params(
            self.optimizer_params, config.init_fn, exclude={"state", "model"}
        )
        step_kwargs = _extract_kwargs_from_params(
            self.optimizer_params, config.step_fn, exclude={"state", "model"}
        )
        if config.init_kwargs_modifier is not None:
            init_kwargs = config.init_kwargs_modifier(init_kwargs)
        if config.step_kwargs_modifier is not None:
            step_kwargs = config.step_kwargs_modifier(step_kwargs)

        self._init_fn = config.init_fn
        self._step_fn = config.step_fn
        self._init_kwargs = init_kwargs
        self._step_kwargs = step_kwargs

    def run(
        self,
        initial_system: StateLike,
        final_system: StateLike,
        max_steps: int = 100,
        fmax: float = 0.05,
    ) -> SimState:
        """Run a single-chain NEB optimization through ``ts.optimize``."""
        logger.info("Starting NEB optimization")
        initial_state = initialize_state(initial_system, self.device, self.dtype)
        final_state = initialize_state(final_system, self.device, self.dtype)
        validate_endpoints(initial_state, final_state)
        initial_energy, final_energy = _endpoint_energies(
            initial_state, final_state, self.model
        )
        movable_images = interpolate_path(initial_state, final_state, self.n_images)
        logger.info(
            "Running NEB for max %d steps or fmax < %.4f eV/Ang.",
            max_steps,
            fmax,
        )

        endpoint_kwargs: dict[str, Any] = {
            "initial_state": as_sim_state(initial_state),
            "final_state": as_sim_state(final_state),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "spring_constant": self.spring_constant,
            "use_climbing_image": self.use_climbing_image,
        }
        trajectory_reporter = (
            {"filenames": self.trajectory_filename}
            if self.trajectory_filename is not None
            else None
        )
        opt_state = optimize(
            movable_images,
            self.model,
            optimizer=(neb_init, neb_step),
            convergence_fn=partial(neb_convergence_fn, fmax=fmax),
            max_steps=max_steps,
            steps_between_swaps=1,
            trajectory_reporter=trajectory_reporter,
            autobatcher=False,
            init_kwargs={
                **endpoint_kwargs,
                "base_init_fn": self._init_fn,
                "base_init_kwargs": self._init_kwargs,
            },
            **endpoint_kwargs,
            base_step_fn=self._step_fn,
            base_step_kwargs=self._step_kwargs,
        )
        final_neb_max_force = torch.linalg.norm(opt_state.forces, dim=-1).max()
        if final_neb_max_force >= fmax:
            logger.warning("NEB optimization did not converge within max_steps.")
        else:
            logger.info("NEB optimization converged.")
        return assemble_path(initial_state, opt_state, final_state)
