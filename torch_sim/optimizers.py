"""Optimizers for geometry relaxations.

This module provides optimization algorithms for atomic structures in a batched format,
enabling efficient relaxation of multiple atomic structures simultaneously. It includes
several gradient-based methods with support for both atomic position and unit cell
optimization.

The module offers:

* Standard gradient descent for atomic positions
* Gradient descent with unit cell optimization
* FIRE (Fast Inertial Relaxation Engine) optimization with unit cell parameters
* FIRE optimization with Frechet cell parameterization for improved cell relaxation

ASE-style FIRE: https://gitlab.com/ase/ase/-/blob/master/ase/optimize/fire.py?ref_type=heads
Velocity Verlet-style FIRE: https://doi.org/10.1103/PhysRevLett.97.170201

"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch

import torch_sim.math as tsm
from torch_sim.models.interface import ModelInterface
from torch_sim.state import DeformGradMixin, SimState
from torch_sim.typing import StateDict


MdFlavor = Literal["vv_fire", "ase_fire"]
vv_fire_key, ase_fire_key = get_args(MdFlavor)

_md_atom_attributes = SimState._atom_attributes | {"forces", "velocities"}  # noqa: SLF001
_fire_system_attributes = (
    SimState._system_attributes  # noqa: SLF001
    | DeformGradMixin._system_attributes  # noqa: SLF001
    | {
        "energy",
        "stress",
        "cell_positions",
        "cell_velocities",
        "cell_forces",
        "cell_masses",
        "cell_factor",
        "pressure",
        "dt",
        "alpha",
        "n_pos",
    }
)
_fire_global_attributes = SimState._global_attributes | {  # noqa: SLF001
    "hydrostatic_strain",
    "constant_volume",
}


@dataclass
class GDState(SimState):
    """State class for batched gradient descent optimization.

    This class extends SimState to store and track the evolution of system state
    during gradient descent optimization. It maintains the energies and forces
    needed to perform gradient-based structure relaxation in a batched manner.

    Attributes:
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_systems, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        system_idx (torch.Tensor): System indices with shape [n_atoms]
        forces (torch.Tensor): Forces acting on atoms with shape [n_atoms, 3]
        energy (torch.Tensor): Potential energy with shape [n_systems]
    """

    forces: torch.Tensor
    energy: torch.Tensor

    _atom_attributes = SimState._atom_attributes | {"forces"}  # noqa: SLF001
    _system_attributes = SimState._system_attributes | {"energy"}  # noqa: SLF001


def gradient_descent(
    model: ModelInterface, *, lr: torch.Tensor | float = 0.01
) -> tuple[Callable[[StateDict | SimState], GDState], Callable[[GDState], GDState]]:
    """Initialize a batched gradient descent optimization.

    Creates an optimizer that performs standard gradient descent on atomic positions
    for multiple systems in parallel. The optimizer updates atomic positions based on
    forces computed by the provided model. The cell is not optimized with this optimizer.

    Args:
        model (torch.nn.Module): Model that computes energies and forces
        lr (torch.Tensor | float): Learning rate(s) for optimization. Can be a single
            float applied to all systems or a tensor with shape [n_systems] for
            system-specific rates

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates the initial BatchedGDState
            - Update function that performs one gradient descent step

    Notes:
        The learning rate controls the step size during optimization. Larger values can
        speed up convergence but may cause instability in the optimization process.
    """
    device, dtype = model.device, model.dtype

    def gd_init(
        state: SimState | StateDict,
        **kwargs: Any,
    ) -> GDState:
        """Initialize the batched gradient descent optimization state.

        Args:
            state: SimState containing positions, masses, cell, etc.
            kwargs: Additional keyword arguments to override state attributes

        Returns:
            Initialized BatchedGDState with forces and energy
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        atomic_numbers = kwargs.get("atomic_numbers", state.atomic_numbers)

        # Get initial forces and energy from model
        model_output = model(state)
        energy = model_output["energy"]
        forces = model_output["forces"]

        return GDState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            atomic_numbers=atomic_numbers,
            system_idx=state.system_idx,
        )

    def gd_step(state: GDState, lr: torch.Tensor = lr) -> GDState:
        """Perform one gradient descent optimization step to update the
        atomic positions. The cell is not optimized.

        Args:
            state: Current optimization state
            lr: Learning rate(s) to use for this step, overriding the default

        Returns:
            Updated GDState after one optimization step
        """
        # Get per-atom learning rates by mapping batch learning rates to atoms
        if isinstance(lr, float):
            lr = torch.full((state.n_systems,), lr, device=device, dtype=dtype)

        atom_lr = lr[state.system_idx].unsqueeze(-1)  # shape: (total_atoms, 1)

        # Update positions using forces and per-atom learning rates
        state.positions = state.positions + atom_lr * state.forces

        # Get updated forces and energy from model
        model_output = model(state)

        # Update state with new forces and energy
        state.forces = model_output["forces"]
        state.energy = model_output["energy"]

        return state

    return gd_init, gd_step


@dataclass(kw_only=True)
class UnitCellGDState(GDState, DeformGradMixin):
    """State class for batched gradient descent optimization with unit cell.

    Extends GDState to include unit cell optimization parameters and stress
    information. This class maintains the state variables needed for simultaneously
    optimizing atomic positions and unit cell parameters.

    Attributes:
        # Inherited from GDState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_systems, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        system_idx (torch.Tensor): System indices with shape [n_atoms]
        forces (torch.Tensor): Forces acting on atoms with shape [n_atoms, 3]
        energy (torch.Tensor): Potential energy with shape [n_systems]

        # Additional attributes for cell optimization
        stress (torch.Tensor): Stress tensor with shape [n_systems, 3, 3]
        reference_cell (torch.Tensor): Reference unit cells with shape
            [n_systems, 3, 3]
        cell_factor (torch.Tensor): Scaling factor for cell optimization with shape
            [n_systems, 1, 1]
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
        constant_volume (bool): Whether to maintain constant volume
        pressure (torch.Tensor): Applied pressure tensor with shape [n_systems, 3, 3]
        cell_positions (torch.Tensor): Cell positions with shape [n_systems, 3, 3]
        cell_forces (torch.Tensor): Cell forces with shape [n_systems, 3, 3]
        cell_masses (torch.Tensor): Cell masses with shape [n_systems, 3]
    """

    # Required attributes not in BatchedGDState
    reference_cell: torch.Tensor
    cell_factor: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool
    pressure: torch.Tensor
    stress: torch.Tensor

    # Cell attributes
    cell_positions: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    _system_attributes = (
        GDState._system_attributes  # noqa: SLF001
        | DeformGradMixin._system_attributes  # noqa: SLF001
        | {
            "cell_forces",
            "pressure",
            "stress",
            "cell_positions",
            "cell_factor",
            "cell_masses",
        }
    )
    _global_attributes = (
        GDState._global_attributes | {"hydrostatic_strain", "constant_volume"}  # noqa: SLF001
    )


def unit_cell_gradient_descent(  # noqa: PLR0915, C901
    model: ModelInterface,
    *,
    positions_lr: float = 0.01,
    cell_lr: float = 0.1,
    cell_factor: float | torch.Tensor | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
) -> tuple[
    Callable[[SimState | StateDict], UnitCellGDState],
    Callable[[UnitCellGDState], UnitCellGDState],
]:
    """Initialize a batched gradient descent optimization with unit cell parameters.

    Creates an optimizer that performs gradient descent on both atomic positions and
    unit cell parameters for multiple systems in parallel. Supports constraints on cell
    deformation and applied external pressure.

    This optimizer extends standard gradient descent to simultaneously optimize
    both atomic coordinates and unit cell parameters based on forces and stress
    computed by the provided model.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress
        positions_lr (float): Learning rate for atomic positions optimization. Default
            is 0.01.
        cell_lr (float): Learning rate for unit cell optimization. Default is 0.1.
        cell_factor (float | torch.Tensor | None): Scaling factor for cell
            optimization. If None, defaults to number of atoms per system
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
            (isotropic scaling). Default is False.
        constant_volume (bool): Whether to maintain constant volume during optimization
            Default is False.
        scalar_pressure (float): Applied external pressure in GPa. Default is 0.0.

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a BatchedUnitCellGDState
            - Update function that performs one gradient descent step with cell
                optimization

    Notes:
        - To fix the cell and only optimize atomic positions, set both
          constant_volume=True and hydrostatic_strain=True
        - The cell_factor parameter controls the relative scale of atomic vs cell
          optimization
        - Larger values for positions_lr and cell_lr can speed up convergence but
          may cause instability in the optimization process
    """
    device, dtype = model.device, model.dtype

    def gd_init(
        state: SimState,
        cell_factor: float | torch.Tensor | None = cell_factor,
        hydrostatic_strain: bool = hydrostatic_strain,  # noqa: FBT001
        constant_volume: bool = constant_volume,  # noqa: FBT001
        scalar_pressure: float = scalar_pressure,
    ) -> UnitCellGDState:
        """Initialize the batched gradient descent optimization state with unit cell.

        Args:
            state: Initial system state containing positions, masses, cell, etc.
            cell_factor: Scaling factor for cell optimization (default: number of atoms)
            hydrostatic_strain: Whether to only allow hydrostatic deformation
            constant_volume: Whether to maintain constant volume
            scalar_pressure: Applied pressure in GPa
            **kwargs: Additional keyword arguments for state initialization

        Returns:
            Initial UnitCellGDState with system configuration and forces
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        n_systems = state.n_systems

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per system
            _, counts = torch.unique(state.system_idx, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all systems
            cell_factor = torch.full(
                (state.n_systems,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_systems, 1, 1) for broadcasting
        cell_factor = cell_factor.view(n_systems, 1, 1)

        scalar_pressure = torch.full(
            (state.n_systems, 1, 1), scalar_pressure, device=device, dtype=dtype
        )
        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device)

        # Get initial forces and energy from model
        model_output = model(state)
        energy = model_output["energy"]
        forces = model_output["forces"]
        stress = model_output["stress"]  # Already shape: (n_systems, 3, 3)

        # Create cell masses
        cell_masses = torch.ones(
            (state.n_systems, 3), device=device, dtype=dtype
        )  # One mass per cell DOF

        # Get current deformation gradient
        cur_deform_grad = DeformGradMixin._deform_grad(  # noqa: SLF001
            state.row_vector_cell, state.row_vector_cell
        )

        # Calculate cell positions
        cell_factor_expanded = cell_factor.expand(
            state.n_systems, 3, 1
        )  # shape: (n_systems, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(state.n_systems, 3, 3) * cell_factor_expanded
        )  # shape: (n_systems, 3, 3)

        # Calculate virial
        volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
        virial = -volumes * (stress + pressure)

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(state.n_systems, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(state.n_systems, -1, -1)

        return UnitCellGDState(
            positions=state.positions,
            forces=forces,
            energy=energy,
            stress=stress,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
            pressure=pressure,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            cell_positions=cell_positions,
            cell_forces=virial / cell_factor,
            cell_masses=cell_masses,
        )

    def gd_step(
        state: UnitCellGDState,
        positions_lr: torch.Tensor = positions_lr,
        cell_lr: torch.Tensor = cell_lr,
    ) -> UnitCellGDState:
        """Perform one gradient descent optimization step with unit cell.

        Updates both atomic positions and cell parameters based on forces and stress.

        Args:
            state: Current optimization state
            positions_lr: Learning rate for atomic positions optimization
            cell_lr: Learning rate for unit cell optimization

        Returns:
            Updated UnitCellGDState after one optimization step
        """
        # Get dimensions
        n_systems = state.n_systems

        # Get per-atom learning rates by mapping system learning rates to atoms
        if isinstance(positions_lr, float):
            positions_lr = torch.full(
                (state.n_systems,), positions_lr, device=device, dtype=dtype
            )

        if isinstance(cell_lr, float):
            cell_lr = torch.full((state.n_systems,), cell_lr, device=device, dtype=dtype)

        # Get current deformation gradient
        cur_deform_grad = state.deform_grad()

        # Calculate cell positions from deformation gradient
        cell_factor_expanded = state.cell_factor.expand(n_systems, 3, 1)
        cell_positions = (
            cur_deform_grad.reshape(n_systems, 3, 3) * cell_factor_expanded
        )  # shape: (n_systems, 3, 3)

        # Get per-atom and per-cell learning rates
        atom_wise_lr = positions_lr[state.system_idx].unsqueeze(-1)
        cell_wise_lr = cell_lr.view(n_systems, 1, 1)  # shape: (n_systems, 1, 1)

        # Update atomic and cell positions
        atomic_positions_new = state.positions + atom_wise_lr * state.forces
        cell_positions_new = cell_positions + cell_wise_lr * state.cell_forces

        # Update cell with deformation gradient
        cell_update = cell_positions_new / cell_factor_expanded
        new_row_vector_cell = torch.bmm(state.reference_row_vector_cell, cell_update.mT)

        # Update state
        state.positions = atomic_positions_new
        state.row_vector_cell = new_row_vector_cell

        # Get new forces and energy
        model_output = model(state)

        state.energy = model_output["energy"]
        state.forces = model_output["forces"]
        state.stress = model_output["stress"]

        # Calculate virial for cell forces
        volumes = torch.linalg.det(new_row_vector_cell).view(n_systems, 1, 1)
        virial = -volumes * (state.stress + state.pressure)
        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_systems, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_systems, -1, -1)

        # Update cell forces
        state.cell_positions = cell_positions_new
        state.cell_forces = virial / state.cell_factor

        return state

    return gd_init, gd_step


@dataclass(kw_only=True)
class FireState(SimState):
    """State information for batched FIRE optimization.

    This class extends SimState to store and track the system state during FIRE
    (Fast Inertial Relaxation Engine) optimization. It maintains the atomic
    parameters along with their velocities and forces for structure relaxation using
    the FIRE algorithm.

    Attributes:
        # Inherited from SimState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_systems, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        system_idx (torch.Tensor): System indices with shape [n_atoms]

        # Atomic quantities
        forces (torch.Tensor): Forces on atoms with shape [n_atoms, 3]
        velocities (torch.Tensor): Atomic velocities with shape [n_atoms, 3]
        energy (torch.Tensor): Energy per system with shape [n_systems]

        # FIRE optimization parameters
        dt (torch.Tensor): Current timestep per system with shape [n_systems]
        alpha (torch.Tensor): Current mixing parameter per system with shape [n_systems]
        n_pos (torch.Tensor): Number of positive power steps per system with shape
            [n_systems]

    Properties:
        momenta (torch.Tensor): Atomwise momenta of the system with shape [n_atoms, 3],
            calculated as velocities * masses
    """

    # Required attributes not in SimState
    forces: torch.Tensor
    energy: torch.Tensor
    velocities: torch.Tensor

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor

    _atom_attributes = _md_atom_attributes
    _system_attributes = (
        SimState._system_attributes  # noqa: SLF001
        | {
            "energy",
            "dt",
            "alpha",
            "n_pos",
        }
    )


def fire(
    model: ModelInterface,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    max_step: float = 0.2,
    md_flavor: MdFlavor = ase_fire_key,
) -> tuple[
    Callable[[SimState | StateDict], FireState],
    Callable[[FireState], FireState],
]:
    """Initialize a batched FIRE optimization.

    Creates an optimizer that performs FIRE (Fast Inertial Relaxation Engine)
    optimization on atomic positions.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress
        dt_max (float): Maximum allowed timestep
        dt_start (float): Initial timestep
        n_min (int): Minimum steps before timestep increase
        f_inc (float): Factor for timestep increase when power is positive
        f_dec (float): Factor for timestep decrease when power is negative
        alpha_start (float): Initial velocity mixing parameter
        f_alpha (float): Factor for mixing parameter decrease
        max_step (float): Maximum distance an atom can move per iteration (default
            value is 0.2). Only used when md_flavor='ase_fire'.
        md_flavor ("vv_fire" | "ase_fire"): Optimization flavor. Default is "ase_fire".

    Returns:
        tuple[Callable, Callable]:
            - Initialization function that creates a FireState
            - Update function (either vv_fire_step or ase_fire_step) that performs
              one FIRE optimization step.

    Notes:
        - md_flavor="vv_fire" follows the original paper closely, including
          integration with Velocity Verlet steps. See https://doi.org/10.1103/PhysRevLett.97.170201
          and https://github.com/Radical-AI/torch-sim/issues/90#issuecomment-2826179997
          for details.
        - md_flavor="ase_fire" mimics the implementation in ASE, which differs slightly
          in the update steps and does not explicitly use atomic masses in the
          velocity update step. See https://gitlab.com/ase/ase/-/blob/66963e6e38/ase/optimize/fire.py#L164-214
          for details.
        - FIRE is generally more efficient than standard gradient descent for atomic
          structure optimization.
        - The algorithm adaptively adjusts step sizes and mixing parameters based
          on the dot product of forces and velocities (power).
    """
    if md_flavor not in get_args(MdFlavor):
        raise ValueError(f"Unknown {md_flavor=}, must be one of {get_args(MdFlavor)}")

    device, dtype = model.device, model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters, added max_step for ASE style
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step = (
        torch.as_tensor(p, device=device, dtype=dtype)
        for p in (dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step)
    )

    def fire_init(
        state: SimState | StateDict,
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
    ) -> FireState:
        """Initialize a batched FIRE optimization state.

        Args:
            state: Input state as SimState object or state parameter dict
            dt_start: Initial timestep per system
            alpha_start: Initial mixing parameter per system

        Returns:
            FireState with initialized optimization tensors
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Get dimensions
        n_systems = state.n_systems

        # Get initial forces and energy from model
        model_output = model(state)

        energy = model_output["energy"]  # [n_systems]
        forces = model_output["forces"]  # [n_total_atoms, 3]

        # Setup parameters
        dt_start = torch.full((n_systems,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_systems,), alpha_start, device=device, dtype=dtype)
        n_pos = torch.zeros((n_systems,), device=device, dtype=torch.int32)

        return FireState(  # Create initial state
            # Copy SimState attributes
            positions=state.positions.clone(),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            system_idx=state.system_idx.clone(),
            pbc=state.pbc,
            velocities=torch.full(
                state.positions.shape, torch.nan, device=device, dtype=dtype
            ),
            forces=forces,
            energy=energy,
            # Optimization attributes
            dt=dt_start,
            alpha=alpha_start,
            n_pos=n_pos,
        )

    step_func_kwargs = dict(
        model=model,
        dt_max=dt_max,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        alpha_start=alpha_start,
        f_alpha=f_alpha,
        eps=eps,
        is_cell_optimization=False,
        is_frechet=False,
    )
    if md_flavor == ase_fire_key:
        step_func_kwargs["max_step"] = max_step
    step_func = {vv_fire_key: _vv_fire_step, ase_fire_key: _ase_fire_step}[md_flavor]
    return fire_init, functools.partial(step_func, **step_func_kwargs)


@dataclass(kw_only=True)
class UnitCellFireState(SimState, DeformGradMixin):
    """State information for batched FIRE optimization with unit cell degrees of
    freedom.

    This class extends SimState to store and track the system state during FIRE
    (Fast Inertial Relaxation Engine) optimization. It maintains both atomic and cell
    parameters along with their velocities and forces for structure relaxation using
    the FIRE algorithm.

    Attributes:
        # Inherited from SimState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_systems, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        system_idx (torch.Tensor): System indices with shape [n_atoms]

        # Atomic quantities
        forces (torch.Tensor): Forces on atoms with shape [n_atoms, 3]
        velocities (torch.Tensor): Atomic velocities with shape [n_atoms, 3]
        energy (torch.Tensor): Energy per system with shape [n_systems]
        stress (torch.Tensor): Stress tensor with shape [n_systems, 3, 3]

        # Cell quantities
        cell_positions (torch.Tensor): Cell positions with shape [n_systems, 3, 3]
        cell_velocities (torch.Tensor): Cell velocities with shape [n_systems, 3, 3]
        cell_forces (torch.Tensor): Cell forces with shape [n_systems, 3, 3]
        cell_masses (torch.Tensor): Cell masses with shape [n_systems, 3]

        # Cell optimization parameters
        reference_cell (torch.Tensor): Original unit cells with shape [n_systems, 3, 3]
        cell_factor (torch.Tensor): Cell optimization scaling factor with shape
            [n_systems, 1, 1]
        pressure (torch.Tensor): Applied pressure tensor with shape [n_systems, 3, 3]
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
        constant_volume (bool): Whether to maintain constant volume

        # FIRE optimization parameters
        dt (torch.Tensor): Current timestep per system with shape [n_systems]
        alpha (torch.Tensor): Current mixing parameter per system with shape [n_systems]
        n_pos (torch.Tensor): Number of positive power steps per system with shape
            [n_systems]

    Properties:
        momenta (torch.Tensor): Atomwise momenta of the system with shape [n_atoms, 3],
            calculated as velocities * masses
    """

    # Required attributes not in SimState
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    velocities: torch.Tensor

    # Cell attributes
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    # Optimization-specific attributes
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor

    _atom_attributes = _md_atom_attributes
    _system_attributes = _fire_system_attributes
    _global_attributes = _fire_global_attributes


def unit_cell_fire(
    model: ModelInterface,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
    max_step: float = 0.2,
    md_flavor: MdFlavor = ase_fire_key,
) -> tuple[
    Callable[[SimState | StateDict], UnitCellFireState],
    Callable[[UnitCellFireState], UnitCellFireState],
]:
    """Initialize a batched FIRE optimization with unit cell degrees of freedom.

    Creates an optimizer that performs FIRE (Fast Inertial Relaxation Engine)
    optimization on both atomic positions and unit cell parameters for multiple systems
    in parallel. FIRE combines molecular dynamics with velocity damping and adjustment
    of time steps to efficiently find local minima.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress
        dt_max (float): Maximum allowed timestep
        dt_start (float): Initial timestep
        n_min (int): Minimum steps before timestep increase
        f_inc (float): Factor for timestep increase when power is positive
        f_dec (float): Factor for timestep decrease when power is negative
        alpha_start (float): Initial velocity mixing parameter
        f_alpha (float): Factor for mixing parameter decrease
        cell_factor (float | None): Scaling factor for cell optimization.
            If None, defaults to number of atoms per system
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
            (isotropic scaling)
        constant_volume (bool): Whether to maintain constant volume during optimization
        scalar_pressure (float): Applied external pressure in GPa
        max_step (float): Maximum allowed step size for ase_fire
        md_flavor ("vv_fire" | "ase_fire"): Optimization flavor. Default is "ase_fire".

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a BatchedUnitCellFireState
            - Update function that performs one FIRE optimization step

    Notes:
        - md_flavor="vv_fire" follows the original paper closely, including
          integration with Velocity Verlet steps. See https://doi.org/10.1103/PhysRevLett.97.170201
          and https://github.com/Radical-AI/torch-sim/issues/90#issuecomment-2826179997
          for details.
        - md_flavor="ase_fire" mimics the implementation in ASE, which differs slightly
          in the update steps and does not explicitly use atomic masses in the
          velocity update step. See https://gitlab.com/ase/ase/-/blob/66963e6e38/ase/optimize/fire.py#L164-214
          for details.
        - FIRE is generally more efficient than standard gradient descent for atomic
          structure optimization
        - The algorithm adaptively adjusts step sizes and mixing parameters based
          on the dot product of forces and velocities
        - To fix the cell and only optimize atomic positions, set both
          constant_volume=True and hydrostatic_strain=True
        - The cell_factor parameter controls the relative scale of atomic vs cell
          optimization
    """
    if md_flavor not in get_args(MdFlavor):
        raise ValueError(f"Unknown {md_flavor=}, must be one of {get_args(MdFlavor)}")
    device, dtype = model.device, model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step = (
        torch.as_tensor(p, device=device, dtype=dtype)
        for p in (dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step)
    )

    def fire_init(
        state: SimState | StateDict,
        cell_factor: torch.Tensor | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
    ) -> UnitCellFireState:
        """Initialize a batched FIRE optimization state with unit cell.

        Args:
            state: Input state as SimState object or state parameter dict
            cell_factor: Cell optimization scaling factor. If None, uses atoms per system.
                Single value or tensor of shape [n_systems].
            scalar_pressure: Applied pressure in energy units
            dt_start: Initial timestep per system
            alpha_start: Initial mixing parameter per system

        Returns:
            UnitCellFireState with initialized optimization tensors
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Get dimensions
        n_systems = state.n_systems

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per system
            _, counts = torch.unique(state.system_idx, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all systems
            cell_factor = torch.full(
                (state.n_systems,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_systems, 1, 1) for broadcasting
        cell_factor = cell_factor.view(n_systems, 1, 1)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
        pressure = pressure.unsqueeze(0).expand(n_systems, -1, -1)

        # Get initial forces and energy from model
        model_output = model(state)

        energy = model_output["energy"]  # [n_systems]
        forces = model_output["forces"]  # [n_total_atoms, 3]
        stress = model_output["stress"]  # [n_systems, 3, 3]

        volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
        virial = -volumes * (stress + pressure)  # P is P_ext * I

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_systems, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_systems, -1, -1)

        cell_forces = virial / cell_factor

        # Sum masses per system using segment_reduce
        # TODO (AG): check this
        system_counts = torch.bincount(state.system_idx)

        cell_masses = torch.segment_reduce(
            state.masses, reduce="sum", lengths=system_counts
        )  # shape: (n_systems,)
        cell_masses = cell_masses.unsqueeze(-1).expand(-1, 3)  # shape: (n_systems, 3)

        # Setup parameters
        dt_start = torch.full((n_systems,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_systems,), alpha_start, device=device, dtype=dtype)
        n_pos = torch.zeros((n_systems,), device=device, dtype=torch.int32)

        return UnitCellFireState(  # Create initial state
            # Copy SimState attributes
            positions=state.positions.clone(),
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            system_idx=state.system_idx.clone(),
            pbc=state.pbc,
            velocities=torch.full(
                state.positions.shape, torch.nan, device=device, dtype=dtype
            ),
            forces=forces,
            energy=energy,
            stress=stress,
            # Cell attributes
            cell_positions=torch.zeros(n_systems, 3, 3, device=device, dtype=dtype),
            cell_velocities=torch.full(
                cell_forces.shape, torch.nan, device=device, dtype=dtype
            ),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
            # Optimization attributes
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            pressure=pressure,
            dt=dt_start,
            alpha=alpha_start,
            n_pos=n_pos,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )

    step_func_kwargs = dict(
        model=model,
        dt_max=dt_max,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        alpha_start=alpha_start,
        f_alpha=f_alpha,
        eps=eps,
        is_cell_optimization=True,
        is_frechet=False,
    )
    if md_flavor == ase_fire_key:
        step_func_kwargs["max_step"] = max_step
    step_func = {vv_fire_key: _vv_fire_step, ase_fire_key: _ase_fire_step}[md_flavor]
    return fire_init, functools.partial(step_func, **step_func_kwargs)


@dataclass(kw_only=True)
class FrechetCellFIREState(SimState, DeformGradMixin):
    """State class for batched FIRE optimization with Frechet cell derivatives.

    This class extends SimState to store and track the system state during FIRE
    optimization with matrix logarithm parameterization for cell degrees of freedom.
    This parameterization provides improved handling of cell deformations during
    optimization.

    Attributes:
        # Inherited from SimState
        positions (torch.Tensor): Atomic positions with shape [n_atoms, 3]
        masses (torch.Tensor): Atomic masses with shape [n_atoms]
        cell (torch.Tensor): Unit cell vectors with shape [n_systems, 3, 3]
        pbc (bool): Whether to use periodic boundary conditions
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms]
        system_idx (torch.Tensor): System indices with shape [n_atoms]

        # Additional atomic quantities
        forces (torch.Tensor): Forces on atoms with shape [n_atoms, 3]
        energy (torch.Tensor): Energy per system with shape [n_systems]
        velocities (torch.Tensor): Atomic velocities with shape [n_atoms, 3]
        stress (torch.Tensor): Stress tensor with shape [n_systems, 3, 3]

        # Optimization-specific attributes
        reference_cell (torch.Tensor): Original unit cell with shape [n_systems, 3, 3]
        cell_factor (torch.Tensor): Scaling factor for cell optimization with shape
            [n_systems, 1, 1]
        pressure (torch.Tensor): Applied pressure tensor with shape [n_systems, 3, 3]
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
        constant_volume (bool): Whether to maintain constant volume

        # Cell attributes using log parameterization
        cell_positions (torch.Tensor): Cell positions using log parameterization with
            shape [n_systems, 3, 3]
        cell_velocities (torch.Tensor): Cell velocities with shape [n_systems, 3, 3]
        cell_forces (torch.Tensor): Cell forces with shape [n_systems, 3, 3]
        cell_masses (torch.Tensor): Cell masses with shape [n_systems, 3]

        # FIRE algorithm parameters
        dt (torch.Tensor): Current timestep per system with shape [n_systems]
        alpha (torch.Tensor): Current mixing parameter per system with shape [n_systems]
        n_pos (torch.Tensor): Number of positive power steps per system with shape
            [n_systems]

    Properties:
        momenta (torch.Tensor): Atomwise momenta of the system with shape [n_atoms, 3],
            calculated as velocities * masses
    """

    # Required attributes not in SimState
    forces: torch.Tensor
    energy: torch.Tensor
    velocities: torch.Tensor
    stress: torch.Tensor

    # Optimization-specific attributes
    reference_cell: torch.Tensor
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    hydrostatic_strain: bool
    constant_volume: bool

    # Cell attributes
    cell_positions: torch.Tensor
    cell_velocities: torch.Tensor
    cell_forces: torch.Tensor
    cell_masses: torch.Tensor

    # FIRE algorithm parameters
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor

    _atom_attributes = _md_atom_attributes
    _system_attributes = _fire_system_attributes
    _global_attributes = _fire_global_attributes


def frechet_cell_fire(
    model: ModelInterface,
    *,
    dt_max: float = 1.0,
    dt_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    cell_factor: float | None = None,
    hydrostatic_strain: bool = False,
    constant_volume: bool = False,
    scalar_pressure: float = 0.0,
    max_step: float = 0.2,
    md_flavor: MdFlavor = ase_fire_key,
) -> tuple[
    Callable[[SimState | StateDict], FrechetCellFIREState],
    Callable[[FrechetCellFIREState], FrechetCellFIREState],
]:
    """Initialize a batched FIRE optimization with Frechet cell parameterization.

    Creates an optimizer that performs FIRE optimization on both atomic positions and
    unit cell parameters using matrix logarithm parameterization for cell degrees of
    freedom. This parameterization provides forces consistent with numerical
    derivatives of the potential energy with respect to cell variables, resulting in
    more robust cell optimization.

    Args:
        model (torch.nn.Module): Model that computes energies, forces, and stress.
        dt_max (float): Maximum allowed timestep
        dt_start (float): Initial timestep
        n_min (int): Minimum steps before timestep increase
        f_inc (float): Factor for timestep increase when power is positive
        f_dec (float): Factor for timestep decrease when power is negative
        alpha_start (float): Initial velocity mixing parameter
        f_alpha (float): Factor for mixing parameter decrease
        cell_factor (float | None): Scaling factor for cell optimization.
            If None, defaults to number of atoms per system
        hydrostatic_strain (bool): Whether to only allow hydrostatic deformation
            (isotropic scaling)
        constant_volume (bool): Whether to maintain constant volume during optimization
        scalar_pressure (float): Applied external pressure in GPa
        max_step (float): Maximum allowed step size for ase_fire
        md_flavor ("vv_fire" | "ase_fire"): Optimization flavor. Default is "ase_fire".

    Returns:
        tuple: A pair of functions:
            - Initialization function that creates a FrechetCellFIREState
            - Update function that performs one FIRE step with Frechet derivatives

    Notes:
        - md_flavor="vv_fire" follows the original paper closely, including
          integration with Velocity Verlet steps. See https://doi.org/10.1103/PhysRevLett.97.170201
          and https://github.com/Radical-AI/torch-sim/issues/90#issuecomment-2826179997
          for details.
        - md_flavor="ase_fire" mimics the implementation in ASE, which differs slightly
          in the update steps and does not explicitly use atomic masses in the
          velocity update step. See https://gitlab.com/ase/ase/-/blob/66963e6e38/ase/optimize/fire.py#L164-214
          for details.
        - Frechet cell parameterization uses matrix logarithm to represent cell
          deformations, which provides improved numerical properties for cell
          optimization
        - This method generally performs better than standard unit cell optimization
          for cases with large cell deformations
        - To fix the cell and only optimize atomic positions, set both
          constant_volume=True and hydrostatic_strain=True
    """
    if md_flavor not in get_args(MdFlavor):
        raise ValueError(f"Unknown {md_flavor=}, must be one of {get_args(MdFlavor)}")
    device, dtype = model.device, model.dtype

    eps = 1e-8 if dtype == torch.float32 else 1e-16

    # Setup parameters
    dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step = (
        torch.as_tensor(p, device=device, dtype=dtype)
        for p in (dt_max, dt_start, alpha_start, f_inc, f_dec, f_alpha, n_min, max_step)
    )

    def fire_init(
        state: SimState | StateDict,
        cell_factor: torch.Tensor | None = cell_factor,
        scalar_pressure: float = scalar_pressure,
        dt_start: float = dt_start,
        alpha_start: float = alpha_start,
    ) -> FrechetCellFIREState:
        """Initialize a batched FIRE optimization state with Frechet cell
        parameterization.

        Args:
            state: Input state as SimState object or state parameter dict
            cell_factor: Cell optimization scaling factor. If None, uses atoms per system.
                         Single value or tensor of shape [n_systems].
            scalar_pressure: Applied pressure in energy units
            dt_start: Initial timestep per system
            alpha_start: Initial mixing parameter per system

        Returns:
            FrechetCellFIREState with initialized optimization tensors
        """
        if not isinstance(state, SimState):
            state = SimState(**state)

        # Get dimensions
        n_systems = state.n_systems

        # Setup cell_factor
        if cell_factor is None:
            # Count atoms per system
            _, counts = torch.unique(state.system_idx, return_counts=True)
            cell_factor = counts.to(dtype=dtype)

        if isinstance(cell_factor, int | float):
            # Use same factor for all systems
            cell_factor = torch.full(
                (state.n_systems,), cell_factor, device=device, dtype=dtype
            )

        # Reshape to (n_systems, 1, 1) for broadcasting
        cell_factor = cell_factor.view(n_systems, 1, 1)

        # Setup pressure tensor
        pressure = scalar_pressure * torch.eye(3, device=device, dtype=dtype)
        pressure = pressure.unsqueeze(0).expand(n_systems, -1, -1)

        # Get initial forces and energy from model
        model_output = model(state)

        energy = model_output["energy"]  # [n_systems]
        forces = model_output["forces"]  # [n_total_atoms, 3]
        stress = model_output["stress"]  # [n_systems, 3, 3]

        # Calculate initial cell positions using matrix logarithm
        # Calculate current deformation gradient (identity matrix at start)
        cur_deform_grad = DeformGradMixin._deform_grad(  # noqa: SLF001
            state.row_vector_cell, state.row_vector_cell
        )  # shape: (n_systems, 3, 3)

        # For identity matrix, logm gives zero matrix
        # Initialize cell positions to zeros
        cell_positions = torch.zeros((n_systems, 3, 3), device=device, dtype=dtype)

        # Calculate virial for cell forces
        volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
        virial = -volumes * (stress + pressure)  # P is P_ext * I

        if hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(3, device=device).unsqueeze(
                0
            ).expand(n_systems, -1, -1)

        if constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device
            ).unsqueeze(0).expand(n_systems, -1, -1)

        # Calculate UCF-style cell gradient
        ucf_cell_grad = torch.zeros_like(virial)
        for b in range(n_systems):
            ucf_cell_grad[b] = virial[b] @ torch.linalg.inv(cur_deform_grad[b].T)
        # Calculate cell forces using Frechet derivative approach (all zeros for identity)
        cell_forces = ucf_cell_grad / cell_factor

        # Sum masses per system
        system_counts = torch.bincount(state.system_idx)
        cell_masses = torch.segment_reduce(
            state.masses, reduce="sum", lengths=system_counts
        )  # shape: (n_systems,)
        cell_masses = cell_masses.unsqueeze(-1).expand(-1, 3)  # shape: (n_systems, 3)

        # Setup parameters
        dt_start = torch.full((n_systems,), dt_start, device=device, dtype=dtype)
        alpha_start = torch.full((n_systems,), alpha_start, device=device, dtype=dtype)
        n_pos = torch.zeros((n_systems,), device=device, dtype=torch.int32)

        return FrechetCellFIREState(  # Create initial state
            # Copy SimState attributes
            positions=state.positions,
            masses=state.masses,
            cell=state.cell,
            atomic_numbers=state.atomic_numbers,
            system_idx=state.system_idx,
            pbc=state.pbc,
            velocities=torch.full(
                state.positions.shape, torch.nan, device=device, dtype=dtype
            ),
            forces=forces,
            energy=energy,
            stress=stress,
            # Cell attributes
            cell_positions=cell_positions,
            cell_velocities=torch.full(
                cell_forces.shape, torch.nan, device=device, dtype=dtype
            ),
            cell_forces=cell_forces,
            cell_masses=cell_masses,
            # Optimization attributes
            reference_cell=state.cell.clone(),
            cell_factor=cell_factor,
            pressure=pressure,
            dt=dt_start,
            alpha=alpha_start,
            n_pos=n_pos,
            hydrostatic_strain=hydrostatic_strain,
            constant_volume=constant_volume,
        )

    step_func_kwargs = dict(
        model=model,
        dt_max=dt_max,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        alpha_start=alpha_start,
        f_alpha=f_alpha,
        eps=eps,
        is_cell_optimization=True,
        is_frechet=True,
    )
    if md_flavor == ase_fire_key:
        step_func_kwargs["max_step"] = max_step
    step_func = {vv_fire_key: _vv_fire_step, ase_fire_key: _ase_fire_step}[md_flavor]
    return fire_init, functools.partial(step_func, **step_func_kwargs)


AnyFireCellState = UnitCellFireState | FrechetCellFIREState


def _vv_fire_step(  # noqa: C901, PLR0915
    state: FireState | AnyFireCellState,
    model: ModelInterface,
    *,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_inc: torch.Tensor,
    f_dec: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    eps: float,
    is_cell_optimization: bool = False,
    is_frechet: bool = False,
) -> FireState | AnyFireCellState:
    """Perform one Velocity-Verlet based FIRE optimization step.

    Implements one step of the Fast Inertial Relaxation Engine (FIRE) algorithm for
    optimizing atomic positions and optionally unit cell parameters in a batched setting.
    Uses velocity Verlet integration with adaptive velocity mixing.

    Args:
        state: Current optimization state (FireState, UnitCellFireState, or
            FrechetCellFIREState).
        model: Model that computes energies, forces, and potentially stress.
        dt_max: Maximum allowed timestep.
        n_min: Minimum steps before timestep increase.
        f_inc: Factor for timestep increase when power is positive.
        f_dec: Factor for timestep decrease when power is negative.
        alpha_start: Initial mixing parameter for velocity update.
        f_alpha: Factor for mixing parameter decrease.
        eps: Small epsilon value for numerical stability.
        is_cell_optimization: Flag indicating if cell optimization is active.
        is_frechet: Flag indicating if Frechet cell parameterization is used.

    Returns:
        Updated state after performing one VV-FIRE step.
    """
    n_systems = state.n_systems
    device = state.positions.device
    dtype = state.positions.dtype
    deform_grad_new: torch.Tensor | None = None

    nan_velocities = state.velocities.isnan().any(dim=1)
    if nan_velocities.any():
        state.velocities[nan_velocities] = torch.zeros_like(
            state.positions[nan_velocities]
        )
        if is_cell_optimization:
            if not isinstance(state, AnyFireCellState):
                raise ValueError(
                    f"Cell optimization requires one of {get_args(AnyFireCellState)}."
                )
            nan_cell_velocities = state.cell_velocities.isnan().any(dim=(1, 2))
            state.cell_velocities[nan_cell_velocities] = torch.zeros_like(
                state.cell_positions[nan_cell_velocities]
            )

    alpha_start_system = torch.full(
        (n_systems,), alpha_start.item(), device=device, dtype=dtype
    )

    atom_wise_dt = state.dt[state.system_idx].unsqueeze(-1)
    state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)

    if is_cell_optimization:
        cell_wise_dt = state.dt.unsqueeze(-1).unsqueeze(-1)
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

    state.positions = state.positions + atom_wise_dt * state.velocities

    if is_cell_optimization:
        cell_factor_reshaped = state.cell_factor.view(n_systems, 1, 1)
        if is_frechet:
            if not isinstance(state, expected_cls := FrechetCellFIREState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            cur_deform_grad = state.deform_grad()
            deform_grad_log = torch.zeros_like(cur_deform_grad)
            for b in range(n_systems):
                deform_grad_log[b] = tsm.matrix_log_33(cur_deform_grad[b])

            cell_positions_log_scaled = deform_grad_log * cell_factor_reshaped
            cell_positions_log_scaled_new = (
                cell_positions_log_scaled + cell_wise_dt * state.cell_velocities
            )
            deform_grad_log_new = cell_positions_log_scaled_new / cell_factor_reshaped
            deform_grad_new = torch.matrix_exp(deform_grad_log_new)
            new_row_vector_cell = torch.bmm(
                state.reference_row_vector_cell, deform_grad_new.transpose(1, 2)
            )
            state.row_vector_cell = new_row_vector_cell
            state.cell_positions = cell_positions_log_scaled_new
        else:
            if not isinstance(state, expected_cls := UnitCellFireState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            cur_deform_grad = state.deform_grad()
            cell_factor_expanded = state.cell_factor.expand(n_systems, 3, 1)
            current_cell_positions_scaled = (
                cur_deform_grad.view(n_systems, 3, 3) * cell_factor_expanded
            )

            cell_positions_scaled_new = (
                current_cell_positions_scaled + cell_wise_dt * state.cell_velocities
            )
            cell_update = cell_positions_scaled_new / cell_factor_expanded
            new_cell = torch.bmm(
                state.reference_row_vector_cell, cell_update.transpose(1, 2)
            )
            state.row_vector_cell = new_cell
            state.cell_positions = cell_positions_scaled_new

    results = model(state)
    state.forces = results["forces"]
    state.energy = results["energy"]

    if is_cell_optimization:
        state.stress = results["stress"]
        volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
        virial = -volumes * (state.stress + state.pressure)

        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device, dtype=dtype
            ).unsqueeze(0).expand(n_systems, -1, -1)
        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device, dtype=dtype
            ).unsqueeze(0).expand(n_systems, -1, -1)

        if is_frechet:
            if not isinstance(state, expected_cls := FrechetCellFIREState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            ucf_cell_grad = torch.bmm(
                virial, torch.linalg.inv(torch.transpose(deform_grad_new, 1, 2))
            )
            directions = torch.zeros((9, 3, 3), device=device, dtype=dtype)
            for idx, (mu, nu) in enumerate([(i, j) for i in range(3) for j in range(3)]):
                directions[idx, mu, nu] = 1.0

            new_cell_forces = torch.zeros_like(ucf_cell_grad)
            for b in range(n_systems):
                expm_derivs = torch.stack(
                    [
                        tsm.expm_frechet(
                            deform_grad_log_new[b], direction, compute_expm=False
                        )
                        for direction in directions
                    ]
                )
                forces_flat = torch.sum(
                    expm_derivs * ucf_cell_grad[b].unsqueeze(0), dim=(1, 2)
                )
                new_cell_forces[b] = forces_flat.reshape(3, 3)
            state.cell_forces = new_cell_forces / cell_factor_reshaped
        else:
            if not isinstance(state, expected_cls := UnitCellFireState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            state.cell_forces = virial / cell_factor_reshaped

    state.velocities += 0.5 * atom_wise_dt * state.forces / state.masses.unsqueeze(-1)
    if is_cell_optimization:
        state.cell_velocities += (
            0.5 * cell_wise_dt * state.cell_forces / state.cell_masses.unsqueeze(-1)
        )

    system_power = tsm.batched_vdot(state.forces, state.velocities, state.system_idx)

    if is_cell_optimization:
        system_power += (state.cell_forces * state.cell_velocities).sum(dim=(1, 2))

    # 2. Update dt, alpha, n_pos
    pos_mask_system = system_power > 0.0
    neg_mask_system = ~pos_mask_system

    state.n_pos[pos_mask_system] += 1
    inc_mask = (state.n_pos > n_min) & pos_mask_system
    state.dt[inc_mask] = torch.minimum(state.dt[inc_mask] * f_inc, dt_max)
    state.alpha[inc_mask] *= f_alpha

    state.dt[neg_mask_system] *= f_dec
    state.alpha[neg_mask_system] = alpha_start_system[neg_mask_system]
    state.n_pos[neg_mask_system] = 0

    v_scaling_system = tsm.batched_vdot(
        state.velocities, state.velocities, state.system_idx
    )
    f_scaling_system = tsm.batched_vdot(state.forces, state.forces, state.system_idx)

    if is_cell_optimization:
        v_scaling_system += state.cell_velocities.pow(2).sum(dim=(1, 2))
        f_scaling_system += state.cell_forces.pow(2).sum(dim=(1, 2))

        v_scaling_cell = torch.sqrt(v_scaling_system.view(n_systems, 1, 1))
        f_scaling_cell = torch.sqrt(f_scaling_system.view(n_systems, 1, 1))
        v_mixing_cell = state.cell_forces / (f_scaling_cell + eps) * v_scaling_cell

        alpha_cell_bc = state.alpha.view(n_systems, 1, 1)
        state.cell_velocities = torch.where(
            pos_mask_system.view(n_systems, 1, 1),
            (1.0 - alpha_cell_bc) * state.cell_velocities + alpha_cell_bc * v_mixing_cell,
            torch.zeros_like(state.cell_velocities),
        )

    v_scaling_atom = torch.sqrt(v_scaling_system[state.system_idx].unsqueeze(-1))
    f_scaling_atom = torch.sqrt(f_scaling_system[state.system_idx].unsqueeze(-1))
    v_mixing_atom = state.forces * (v_scaling_atom / (f_scaling_atom + eps))

    alpha_atom = state.alpha[state.system_idx].unsqueeze(-1)  # per-atom alpha
    state.velocities = torch.where(
        pos_mask_system[state.system_idx].unsqueeze(-1),
        (1.0 - alpha_atom) * state.velocities + alpha_atom * v_mixing_atom,
        torch.zeros_like(state.velocities),
    )

    return state


def _ase_fire_step(  # noqa: C901, PLR0915
    state: FireState | AnyFireCellState,
    model: ModelInterface,
    *,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_inc: torch.Tensor,
    f_dec: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    max_step: torch.Tensor,
    eps: float,
    is_cell_optimization: bool = False,
    is_frechet: bool = False,
) -> FireState | AnyFireCellState:
    """Perform one ASE-style FIRE optimization step.

    Implements one step of the Fast Inertial Relaxation Engine (FIRE) algorithm
    mimicking the ASE implementation. It can handle atomic position optimization
    only, or combined position and cell optimization (standard or Frechet).

    Args:
        state: Current optimization state.
        model: Model that computes energies, forces, and potentially stress.
        dt_max: Maximum allowed timestep.
        n_min: Minimum steps before timestep increase.
        f_inc: Factor for timestep increase when power is positive.
        f_dec: Factor for timestep decrease when power is negative.
        alpha_start: Initial mixing parameter for velocity update.
        f_alpha: Factor for mixing parameter decrease.
        max_step: Maximum allowed step size.
        eps: Small epsilon value for numerical stability.
        is_cell_optimization: Flag indicating if cell optimization is active.
        is_frechet: Flag indicating if Frechet cell parameterization is used.

    Returns:
        Updated state after performing one ASE-FIRE step.
    """
    device, dtype = state.positions.device, state.positions.dtype
    n_systems = state.n_systems

    cur_deform_grad = None  # Initialize cur_deform_grad to prevent UnboundLocalError

    nan_velocities = state.velocities.isnan().any(dim=1)
    if nan_velocities.any():
        state.velocities[nan_velocities] = torch.zeros_like(
            state.positions[nan_velocities]
        )
        forces = state.forces
        if is_cell_optimization:
            if not isinstance(state, AnyFireCellState):
                raise ValueError(
                    f"Cell optimization requires one of {get_args(AnyFireCellState)}."
                )
            nan_cell_velocities = state.cell_velocities.isnan().any(dim=(1, 2))
            state.cell_velocities[nan_cell_velocities] = torch.zeros_like(
                state.cell_positions[nan_cell_velocities]
            )
            cur_deform_grad = state.deform_grad()
    else:
        alpha_start_system = torch.full(
            (n_systems,), alpha_start.item(), device=device, dtype=dtype
        )

        if is_cell_optimization:
            cur_deform_grad = state.deform_grad()
            forces = torch.bmm(
                state.forces.unsqueeze(1), cur_deform_grad[state.system_idx]
            ).squeeze(1)
        else:
            forces = state.forces

        # 1. Current power (F·v) per system (atoms + cell)
        system_power = tsm.batched_vdot(forces, state.velocities, state.system_idx)

        if is_cell_optimization:
            system_power += (state.cell_forces * state.cell_velocities).sum(dim=(1, 2))

        # 2. Update dt, alpha, n_pos
        pos_mask_system = system_power > 0.0
        neg_mask_system = ~pos_mask_system

        inc_mask = (state.n_pos > n_min) & pos_mask_system
        state.dt[inc_mask] = torch.minimum(state.dt[inc_mask] * f_inc, dt_max)
        state.alpha[inc_mask] *= f_alpha
        state.n_pos[pos_mask_system] += 1

        state.dt[neg_mask_system] *= f_dec
        state.alpha[neg_mask_system] = alpha_start_system[neg_mask_system]
        state.n_pos[neg_mask_system] = 0

        # 3. Velocity mixing BEFORE acceleration (ASE ordering)
        v_scaling_system = tsm.batched_vdot(
            state.velocities, state.velocities, state.system_idx
        )
        f_scaling_system = tsm.batched_vdot(forces, forces, state.system_idx)

        if is_cell_optimization:
            v_scaling_system += state.cell_velocities.pow(2).sum(dim=(1, 2))
            f_scaling_system += state.cell_forces.pow(2).sum(dim=(1, 2))

            v_scaling_cell = torch.sqrt(v_scaling_system.view(n_systems, 1, 1))
            f_scaling_cell = torch.sqrt(f_scaling_system.view(n_systems, 1, 1))
            v_mixing_cell = state.cell_forces / (f_scaling_cell + eps) * v_scaling_cell

            alpha_cell_bc = state.alpha.view(n_systems, 1, 1)
            state.cell_velocities = torch.where(
                pos_mask_system.view(n_systems, 1, 1),
                (1.0 - alpha_cell_bc) * state.cell_velocities
                + alpha_cell_bc * v_mixing_cell,
                torch.zeros_like(state.cell_velocities),
            )

        v_scaling_atom = torch.sqrt(v_scaling_system[state.system_idx].unsqueeze(-1))
        f_scaling_atom = torch.sqrt(f_scaling_system[state.system_idx].unsqueeze(-1))
        v_mixing_atom = forces * (v_scaling_atom / (f_scaling_atom + eps))

        alpha_atom = state.alpha[state.system_idx].unsqueeze(-1)  # per-atom alpha
        state.velocities = torch.where(
            pos_mask_system[state.system_idx].unsqueeze(-1),
            (1.0 - alpha_atom) * state.velocities + alpha_atom * v_mixing_atom,
            torch.zeros_like(state.velocities),
        )

    # 4. Acceleration (single forward-Euler, no mass for ASE FIRE)
    state.velocities += forces * state.dt[state.system_idx].unsqueeze(-1)
    dr_atom = state.velocities * state.dt[state.system_idx].unsqueeze(-1)
    dr_scaling_system = tsm.batched_vdot(dr_atom, dr_atom, state.system_idx)

    if is_cell_optimization:
        state.cell_velocities += state.cell_forces * state.dt.view(n_systems, 1, 1)
        dr_cell = state.cell_velocities * state.dt.view(n_systems, 1, 1)

        dr_scaling_system += dr_cell.pow(2).sum(dim=(1, 2))
        dr_scaling_cell = torch.sqrt(dr_scaling_system).view(n_systems, 1, 1)
        dr_cell = torch.where(
            dr_scaling_cell > max_step,
            max_step * dr_cell / (dr_scaling_cell + eps),
            dr_cell,
        )

    dr_scaling_atom = torch.sqrt(dr_scaling_system)[state.system_idx].unsqueeze(-1)

    dr_atom = torch.where(
        dr_scaling_atom > max_step, max_step * dr_atom / (dr_scaling_atom + eps), dr_atom
    )

    if is_cell_optimization:
        state.positions = (
            torch.linalg.solve(
                cur_deform_grad[state.system_idx], state.positions.unsqueeze(-1)
            ).squeeze(-1)
            + dr_atom
        )

        if is_frechet:
            if not isinstance(state, expected_cls := FrechetCellFIREState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            new_logm_F_scaled = state.cell_positions + dr_cell
            state.cell_positions = new_logm_F_scaled
            logm_F_new = new_logm_F_scaled / (state.cell_factor + eps)
            F_new = torch.matrix_exp(logm_F_new)
            new_row_vector_cell = torch.bmm(state.reference_row_vector_cell, F_new.mT)
            state.row_vector_cell = new_row_vector_cell
        else:
            if not isinstance(state, expected_cls := UnitCellFireState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            F_current = state.deform_grad()
            cell_factor_exp_mult = state.cell_factor.expand(n_systems, 3, 1)
            current_F_scaled = F_current * cell_factor_exp_mult

            F_new_scaled = current_F_scaled + dr_cell
            state.cell_positions = F_new_scaled
            F_new = F_new_scaled / (cell_factor_exp_mult + eps)
            new_row_vector_cell = torch.bmm(state.reference_row_vector_cell, F_new.mT)
            state.row_vector_cell = new_row_vector_cell

        state.positions = torch.bmm(
            state.positions.unsqueeze(1), F_new[state.system_idx].mT
        ).squeeze(1)
    else:
        state.positions = state.positions + dr_atom

    # 7. Force / stress refresh & new cell forces
    results = model(state)
    state.forces = results["forces"]
    state.energy = results["energy"]

    if is_cell_optimization:
        state.stress = results["stress"]
        volumes = torch.linalg.det(state.cell).view(n_systems, 1, 1)
        if torch.any(volumes <= 0):
            bad_indices = torch.where(volumes <= 0)[0].tolist()
            print(  # noqa: T201
                f"WARNING: Non-positive volume(s) detected during _ase_fire_step: "
                f"{volumes[bad_indices].tolist()} at {bad_indices=} ({is_frechet=})"
            )

        virial = -volumes * (state.stress + state.pressure)

        if state.hydrostatic_strain:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device, dtype=dtype
            ).unsqueeze(0).expand(n_systems, -1, -1)

        if state.constant_volume:
            diag_mean = torch.diagonal(virial, dim1=1, dim2=2).mean(dim=1, keepdim=True)
            virial = virial - diag_mean.unsqueeze(-1) * torch.eye(
                3, device=device, dtype=dtype
            ).unsqueeze(0).expand(n_systems, -1, -1)

        if is_frechet:
            if not isinstance(state, expected_cls := FrechetCellFIREState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            if F_new is None:
                raise ValueError(
                    "F_new should be defined for Frechet cell force calculation"
                )
            if logm_F_new is None:
                raise ValueError(
                    "logm_F_new should be defined for Frechet cell force calculation"
                )
            ucf_cell_grad = torch.bmm(
                virial, torch.linalg.inv(torch.transpose(F_new, 1, 2))
            )
            directions = torch.zeros((9, 3, 3), device=device, dtype=dtype)
            for idx, (mu, nu) in enumerate(
                [(i_idx, j_idx) for i_idx in range(3) for j_idx in range(3)]
            ):
                directions[idx, mu, nu] = 1.0

            new_cell_forces_log_space = torch.zeros_like(state.cell_forces)
            for b_idx in range(n_systems):
                expm_derivs = torch.stack(
                    [
                        tsm.expm_frechet(logm_F_new[b_idx], direction, compute_expm=False)
                        for direction in directions
                    ]
                )
                forces_flat = torch.sum(
                    expm_derivs * ucf_cell_grad[b_idx].unsqueeze(0), dim=(1, 2)
                )
                new_cell_forces_log_space[b_idx] = forces_flat.reshape(3, 3)
            state.cell_forces = new_cell_forces_log_space / (state.cell_factor + eps)
        else:
            if not isinstance(state, expected_cls := UnitCellFireState):
                raise ValueError(f"{type(state)=} must be a {expected_cls.__name__}")
            state.cell_forces = virial / state.cell_factor

    return state
