"""Boxed molecular dynamics in energy space (BXDE).

BXDE accelerates the discovery of rare events by progressively raising a lower
bound on the accessible potential energy, forcing the system to climb into
high-energy regions it would otherwise visit only rarely. The method follows
Shannon et al., *J. Chem. Theory Comput.* **2018**, 14, 4541
(10.1021/acs.jctc.8b00515).

The scheme alternates free-sampling windows with adaptive boundary placement:

1. Run MD freely, tracking the running maximum potential energy ``PE_max``.
2. Once at least ``i_samp`` steps have elapsed *and* a fresh maximum is reached,
   freeze a reflective lower boundary at ``PE_max`` and begin a new window.
3. Thereafter, whenever a step would take the potential energy below the active
   boundary, revert to the previous step and invert the velocity component
   along the energy gradient, so the boundary bounds the energy from below only.

Because the physical force always points toward lower potential energy -- into
the forbidden region -- the boundary is run inside a Langevin (NVT) integrator:
the stochastic force lets the trajectory random-walk along the boundary rather
than becoming trapped against it.

This module is a trajectory-level controller, not a bias potential: it adds no
energy or forces and so is not a
:class:`~torch_sim.models.interface.ModelInterface`. Drive a whole run with
:func:`run_boxed_md`, or step box-by-box with :class:`BoxedMD` for finer control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torch_sim.enhanced_sampling.history import History
from torch_sim.integrators.md import MDState
from torch_sim.integrators.nvt import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import UnitSystem


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState
    from torch_sim.trajectory import TrajectoryReporter


EPS = 1e-12


def velocity_inversion(
    momenta: torch.Tensor, forces: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    r"""Reflect the velocity component along the potential-energy gradient.

    Implements the BXD velocity inversion (Eqs 2-3 of 10.1021/acs.jctc.8b00515)
    for a boundary in potential-energy space, where the constraint gradient
    :math:`\nabla\phi = \nabla E = -F` is given directly by the forces. In terms
    of momenta :math:`p = M v` the update is

    .. math::

        p' = p - 2\,\frac{F \cdot v}{F \cdot M^{-1} F}\,F,
        \qquad v = M^{-1} p

    which reverses the sign of :math:`\nabla\phi \cdot v` (so the system is
    pushed back toward higher energy) while conserving kinetic energy.

    Args:
        momenta: Particle momenta with shape [n_atoms, 3].
        forces: Forces on the particles with shape [n_atoms, 3].
        masses: Particle masses with shape [n_atoms].

    Returns:
        The reflected momenta with shape [n_atoms, 3].
    """
    inv_m = (1.0 / masses).unsqueeze(-1)  # [n_atoms, 1]
    f_dot_v = (forces * momenta * inv_m).sum()  # F . v   (single system)
    f_m_f = (forces.pow(2) * inv_m).sum()  # F . M^-1 . F
    coeff = 2.0 * f_dot_v / (f_m_f + EPS)
    return momenta - coeff * forces


class BoxedMD:
    """Stateful BXDE controller that drives one energy box at a time.

    The controller owns everything that must persist across boxes: the active
    lower boundary, the running ``PE_max`` and step counter of the current
    sampling window, the record of placed boundaries, and the previous accepted
    step used for roll-backs. Call :meth:`run_epoch` repeatedly; it advances one
    box and returns control either when a new boundary is placed or when a step
    budget is exhausted mid-box (in which case the next call resumes the same
    box).

    Args:
        model: Energy/force model passed to the inner integrator step.
        i_samp: Minimum number of accepted steps in a window before a new
            boundary may be placed.
        dt: Integration timestep in the model's internal time units.
        kT: Target temperature in energy units for the Langevin step.
        gamma: Langevin friction coefficient forwarded to the step function.
            Defaults to the integrator's own default (``1/(100*dt)``).
        floor_capacity: Maximum number of placed boundaries to retain in the
            history buffer. Defaults to 10000.
        step_fn: Inner integrator step. Defaults to ``nvt_langevin_step``.
        trajectory_reporter: Optional :class:`~torch_sim.trajectory.TrajectoryReporter`
            whose ``report`` is called once per attempted step (indexed by the
            cumulative attempted-step count), recording rolled-back steps too.
        device: Device for the history/counters. Defaults to ``model.device``.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
    """

    NEW_BOUNDARY = "new_boundary"
    STEP_LIMIT = "step_limit"

    def __init__(
        self,
        model: ModelInterface,
        *,
        i_samp: int,
        dt: torch.Tensor | float,
        kT: torch.Tensor | float,
        gamma: torch.Tensor | float | None = None,
        floor_capacity: int = 10000,
        step_fn: Callable[..., MDState] = nvt_langevin_step,
        trajectory_reporter: TrajectoryReporter | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the BXDE controller."""
        if i_samp < 1:
            raise ValueError(f"{i_samp=} must be >= 1")
        self.model = model
        self.step_fn = step_fn
        self.trajectory_reporter = trajectory_reporter
        self.i_samp = int(i_samp)
        self.dt = dt
        self.kT = kT
        self.gamma = gamma
        self._device = device or model.device
        self._dtype = dtype

        self.floors = History(
            capacity=floor_capacity, stride=1, device=self._device, dtype=self._dtype
        )
        self.v_bxde: torch.Tensor | None = None  # active lower boundary (scalar)
        self.pe_max = self._neg_inf()
        self.i = 0  # accepted steps in the current window
        self.total_steps = 0  # attempted integrator steps over the whole run
        self.n_inversions = 0

    def _neg_inf(self) -> torch.Tensor:
        return torch.tensor(float("-inf"), device=self._device, dtype=self._dtype)

    def _begin_new_window(self) -> None:
        """Reset the per-window running max and counter, keeping the boundary."""
        self.pe_max = self._neg_inf()
        self.i = 0

    def reset(self) -> None:
        """Clear all boundaries and counters, returning to a pristine state."""
        self.floors.reset()
        self.v_bxde = None
        self._begin_new_window()
        self.total_steps = 0
        self.n_inversions = 0

    def _report(self, state: MDState) -> None:
        """Record the current state to the trajectory reporter, if configured."""
        if self.trajectory_reporter is not None:
            self.trajectory_reporter.report(state, self.total_steps, self.model)

    def run_epoch(self, state: MDState, max_steps: int) -> tuple[MDState, int, str]:
        """Advance the current energy box until a boundary is placed or budget runs out.

        Args:
            state: Current MD state (single system).
            max_steps: Maximum number of attempted integrator steps to take.

        Returns:
            A tuple ``(state, n_used, status)`` where ``n_used`` is the number of
            attempted steps taken (each costs one model call, including rolled-back
            steps) and ``status`` is :attr:`NEW_BOUNDARY` if a boundary was placed
            or :attr:`STEP_LIMIT` if the budget was reached first.
        """
        used = 0
        while used < max_steps:
            floor_active = self.v_bxde is not None
            if floor_active:
                # snapshot the accepted step so we can revert to it (time t)
                prev_positions = state.positions.clone()
                prev_momenta = state.momenta.clone()
                prev_energy = state.energy.clone()
                prev_forces = state.forces.clone()

            state = self.step_fn(
                state, self.model, dt=self.dt, kT=self.kT, gamma=self.gamma
            )
            used += 1
            self.total_steps += 1
            pe = state.energy.reshape(())  # scalar potential energy

            if floor_active and bool(pe < self.v_bxde):
                # crossed below the boundary: revert to t and invert the velocity
                state.positions = prev_positions
                state.momenta = velocity_inversion(
                    prev_momenta, prev_forces, state.masses
                )
                state.energy = prev_energy
                state.forces = prev_forces
                self.n_inversions += 1
                self._report(state)
                continue

            # accepted step: track the running maximum and place a boundary once
            # the window is long enough and a fresh maximum is reached
            self.i += 1
            if bool(pe > self.pe_max):
                if self.i > self.i_samp:
                    self.v_bxde = self.pe_max.clone()  # freeze at the running max
                    self.floors.push(self.v_bxde)
                    self._begin_new_window()
                    self._report(state)
                    return state, used, self.NEW_BOUNDARY
                self.pe_max = pe.clone()

            self._report(state)

        return state, used, self.STEP_LIMIT


def run_boxed_md(
    state: SimState,
    model: ModelInterface,
    *,
    n_steps: int,
    i_samp: int,
    timestep: float,
    temperature: float,
    gamma: torch.Tensor | float | None = None,
    seed: int | None = None,
    floor_capacity: int = 10000,
    trajectory_reporter: TrajectoryReporter | None = None,
    unit_system: UnitSystem = UnitSystem.metal,
    **init_kwargs: Any,
) -> tuple[MDState, torch.Tensor]:
    """Run boxed molecular dynamics in energy space for a single system.

    Wraps a :class:`BoxedMD` controller in a loop that respects a global step
    budget: it advances box by box, and stops as soon as ``n_steps`` attempted
    integrator steps have been taken -- interrupting mid-box if necessary.

    Args:
        state: Initial system (single system). If not already an
            :class:`~torch_sim.integrators.md.MDState`, momenta are sampled from
            a Maxwell-Boltzmann distribution at ``temperature``.
        model: Energy/force model.
        n_steps: Total number of attempted integrator steps (the hard budget).
        i_samp: Minimum accepted steps per window before a boundary may be placed.
        timestep: Integration timestep in ``unit_system`` time units.
        temperature: Target temperature in Kelvin (converted to energy units).
        gamma: Optional Langevin friction coefficient.
        seed: Optional RNG seed for momentum initialization and the thermostat.
        floor_capacity: Maximum number of placed boundaries to retain.
        trajectory_reporter: Optional reporter recording every attempted step
            (including the initial frame at step 0). The caller owns its
            lifecycle and is responsible for closing it.
        unit_system: Unit system for temperature/timestep conversion. Defaults
            to metal units (eV, Angstrom, ps).
        **init_kwargs: Extra keyword arguments forwarded to ``nvt_langevin_init``.

    Returns:
        A tuple ``(final_state, floors)`` where ``floors`` is a 1-D tensor of the
        placed lower boundaries (in energy units), ordered and monotonically
        non-decreasing.

    Raises:
        ValueError: If ``state`` contains more than one system.
    """
    if state.n_systems != 1:
        raise ValueError(f"run_boxed_md expects a single system, got {state.n_systems}")

    device, dtype = state.device, state.dtype
    kT = float(temperature) * unit_system.temperature
    dt = torch.as_tensor(timestep * unit_system.time, dtype=dtype, device=device)

    if seed is not None:
        state.rng = seed
    if not isinstance(state, MDState):
        state = nvt_langevin_init(state, model, kT=kT, **init_kwargs)

    controller = BoxedMD(
        model,
        i_samp=i_samp,
        dt=dt,
        kT=kT,
        gamma=gamma,
        floor_capacity=floor_capacity,
        trajectory_reporter=trajectory_reporter,
        device=device,
        dtype=dtype,
    )

    if trajectory_reporter is not None:
        trajectory_reporter.report(state, 0, model)  # initial frame

    remaining = n_steps
    while remaining > 0:
        state, used, status = controller.run_epoch(state, remaining)
        remaining -= used
        if status == BoxedMD.STEP_LIMIT:
            break

    floors = controller.floors.stack()
    if floors is None:
        floors = torch.empty(0, device=device, dtype=dtype)
    return state, floors
