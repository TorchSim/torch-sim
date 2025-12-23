"""Thermodynamic integration module for molecular dynamics simulations."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

import torch_sim as ts
from torch_sim.autobatching import BinningAutoBatcher
from torch_sim.integrators.md import (
    MDState,
    calculate_momenta,
    momentum_step,
    position_step,
)
from torch_sim.models.interface import ModelInterface
from torch_sim.runners import _configure_batches_iterator, _configure_reporter
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.typing import StateDict
from torch_sim.units import UnitSystem


def linear_lambda_schedule(step: int, n_steps: int) -> float:
    """Linear lambda schedule: λ(t) = t/T."""
    return step / n_steps


def quadratic_lambda_schedule(step: int, n_steps: int) -> float:
    """Quadratic lambda schedule: λ(t) = -1*(1-t/T)² + 1."""
    t_normalized = step / n_steps
    return -1 * (1 - t_normalized) ** 2 + 1


def cubic_lambda_schedule(step: int, n_steps: int) -> float:
    """Cubic lambda schedule: λ(t) = -1*(1-t/T)³ + 1."""
    t_normalized = step / n_steps
    return -1 * (1 - t_normalized) ** 3 + 1


def lammps_lambda_schedule(step: int, n_steps: int) -> float:
    """Lambda schedule used in LAMMPS paper: λ(t) = 0.5*(1 - cos(π*t/T))."""
    t = step / n_steps
    return t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)


LAMBDA_SCHEDULES = {
    "linear": linear_lambda_schedule,
    "quadratic": quadratic_lambda_schedule,
    "lammps": lammps_lambda_schedule,
    "cubic": cubic_lambda_schedule,
}


@dataclass
class ThermodynamicIntegrationMDState(MDState):
    """Custom state for thermodynamic integration in MD simulations.

    This state can hold additional properties like lambda_ for TI.
    """

    lambda_: torch.Tensor
    energy_difference: torch.Tensor
    energy1: torch.Tensor
    energy2: torch.Tensor

    _system_attributes = MDState._system_attributes | {  # noqa: SLF001
        "lambda_",
        "energy_difference",
        "energy1",
        "energy2",
    }


class MixedModel(ModelInterface):
    """A model that mixes two models for thermodynamic integration.

    This class implements a linear combination of two models based on a lambda
    parameter, which is used for thermodynamic integration calculations to
    compute free energy differences.
    """

    def __init__(
        self,
        model1: ModelInterface,
        model2: ModelInterface,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        *,
        compute_stress: bool = False,
        compute_forces: bool = True,
    ) -> None:
        """Initialize the mixed model.

        Args:
            model1: First model in the mixture
            model2: Second model in the mixture
            device: Device to run computations on
            dtype: Data type for computations
            compute_stress: Whether to compute stress
            compute_forces: Whether to compute forces
        """
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._dtype = dtype
        self._compute_stress = compute_stress
        self._compute_forces = compute_forces

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        """Forward pass through the mixed model.

        Args:
            state: Simulation state containing positions, masses, etc.

        Returns:
            Dictionary with mixed energies and forces
        """
        if "lambda_" not in state.__dict__:
            lambda_ = 1
            lambda_per_atom = torch.tensor(1.0, device=self._device, dtype=self._dtype)
        else:
            lambda_ = state.lambda_
            lambda_per_atom = lambda_[state.system_idx]
        out1 = self.model1(state)
        out2 = self.model2(state)

        # Combine matching keys
        output = {}
        output["energy"] = (1 - lambda_) * out1["energy"] + lambda_ * out2["energy"]
        output["forces"] = (1 - lambda_per_atom).view(-1, 1) * out1["forces"] + (
            lambda_per_atom
        ).view(-1, 1) * out2["forces"]
        output["energy_difference"] = out2["energy"] - out1["energy"]
        output["energy1"] = out1["energy"]
        output["energy2"] = out2["energy"]
        return output


def nvt_langevin_thermodynamic_integration(  # noqa: C901
    model: torch.nn.Module,
    *,
    dt: torch.Tensor,
    kT: torch.Tensor,
    gamma: torch.Tensor | None = None,
    seed: int | None = None,
) -> tuple[
    Callable[[SimState | StateDict, torch.Tensor], MDState],
    Callable[[MDState, torch.Tensor], MDState],
]:
    """Initialize and return an NVT (canonical) integrator using Langevin dynamics.

    This function sets up integration in the NVT ensemble, where particle number (N),
    volume (V), and temperature (T) are conserved. It returns both an initial state
    and an update function for time evolution.

    It uses Langevin dynamics with stochastic noise and friction to maintain constant
    temperature. The integration scheme combines deterministic velocity Verlet steps with
    stochastic Ornstein-Uhlenbeck processes following the BAOAB splitting scheme.

    Args:
        model (torch.nn.Module): Neural network model that computes energies and forces.
            Must return a dict with 'energy' and 'forces' keys.
        dt (torch.Tensor): Integration timestep, either scalar or with shape [n_batches]
        kT (torch.Tensor): Target temperature in energy units, either scalar or
            with shape [n_batches]
        gamma (torch.Tensor, optional): Friction coefficient for Langevin thermostat,
            either scalar or with shape [n_batches]. Defaults to 1/(100*dt).
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple:
            - callable: Function to initialize the MDState from input data
              with signature: init_fn(state, kT=kT, seed=seed) -> MDState
            - callable: Update function that evolves system by one timestep
              with signature: update_fn(state, dt=dt, kT=kT, gamma=gamma) -> MDState

    Notes:
        - Uses BAOAB splitting scheme for Langevin dynamics
        - Preserves detailed balance for correct NVT sampling
        - Handles periodic boundary conditions if enabled in state
        - Friction coefficient gamma controls the thermostat coupling strength
        - Weak coupling (small gamma) preserves dynamics but with slower thermalization
        - Strong coupling (large gamma) faster thermalization but may distort dynamics
    """
    device, dtype = model.device, model.dtype

    if gamma is None:
        gamma = 1 / (100 * dt)

    if isinstance(gamma, float):
        gamma = torch.tensor(gamma, device=device, dtype=dtype)

    if isinstance(dt, float):
        dt = torch.tensor(dt, device=device, dtype=dtype)

    def ou_step(
        state: ThermodynamicIntegrationMDState,
        dt: torch.Tensor,
        kT: torch.Tensor,
        gamma: torch.Tensor,
    ) -> ThermodynamicIntegrationMDState:
        """Apply stochastic noise and friction for Langevin dynamics.

        This function implements the Ornstein-Uhlenbeck process for Langevin dynamics,
        applying random noise and friction forces to particle momenta. The noise amplitude
        is chosen to satisfy the fluctuation-dissipation theorem, ensuring proper
        sampling of the canonical ensemble at temperature kT.

        Args:
            state (ThermodynamicIntegrationMDState): Current system state containing
                positions, momenta, etc.
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_batches]
            gamma (torch.Tensor): Friction coefficient controlling noise strength,
                either scalar or with shape [n_batches]

        Returns:
            ThermodynamicIntegrationMDState: Updated state with new momenta
                after stochastic step

        Notes:
            - Implements the "O" step in the BAOAB Langevin integration scheme
            - Uses Ornstein-Uhlenbeck process for correct thermal sampling
            - Noise amplitude scales with sqrt(mass) for equipartition
            - Preserves detailed balance through fluctuation-dissipation relation
            - The equation implemented is:
              p(t+dt) = c1*p(t) + c2*sqrt(m)*N(0,1)
              where c1 = exp(-gamma*dt) and c2 = sqrt(kT*(1-c1²))
        """
        c1 = torch.exp(-gamma * dt)

        if isinstance(kT, torch.Tensor) and len(kT.shape) > 0:
            # kT is a tensor with shape (n_batches,)
            kT = kT[state.system_idx]

        # Index c1 and c2 with state.system_idx to align shapes with state.momenta
        if isinstance(c1, torch.Tensor) and len(c1.shape) > 0:
            c1 = c1[state.system_idx]

        c2 = torch.sqrt(kT * (1 - c1**2)).unsqueeze(-1)

        # Generate random noise from normal distribution
        noise = torch.randn_like(state.momenta, device=state.device, dtype=state.dtype)
        new_momenta = (
            c1.unsqueeze(-1) * state.momenta
            + c2 * torch.sqrt(state.masses).unsqueeze(-1) * noise
        )
        state.momenta = new_momenta
        return state

    def langevin_init(
        state: SimState | StateDict,
        lambda_: torch.Tensor,
        kT: torch.Tensor = kT,
        seed: int | None = seed,
    ) -> ThermodynamicIntegrationMDState:
        """Initialize an NVT state from input data for Langevin dynamics.

        Creates an initial state for NVT molecular dynamics by computing initial
        energies and forces, and sampling momenta from a Maxwell-Boltzmann distribution
        at the specified temperature.

        Args:
            state (SimState | StateDict): Either a SimState object or a dictionary
                containing positions, masses, cell, pbc, and other required state vars
            lambda_ (torch.Tensor): Initial lambda values for each system in the batch
            kT (torch.Tensor): Temperature in energy units for initializing momenta,
                either scalar or with shape [n_batches]
            seed (int, optional): Random seed for reproducibility

        Returns:
            MDState: Initialized state for NVT integration containing positions,
                momenta, forces, energy, and other required attributes

        Notes:
            The initial momenta are sampled from a Maxwell-Boltzmann distribution
            at the specified temperature. This provides a proper thermal initial
            state for the subsequent Langevin dynamics.
        """
        if not isinstance(state, SimState):
            state = SimState(**state)
        model_output = model(state)
        momenta = getattr(
            state,
            "momenta",
            calculate_momenta(state.positions, state.masses, state.system_idx, kT, seed),
        )

        initial_state = ThermodynamicIntegrationMDState(
            positions=state.positions,
            momenta=momenta,
            energy=model_output["energy"],
            forces=model_output["forces"],
            energy_difference=model_output["energy_difference"],
            energy1=model_output["energy1"],
            energy2=model_output["energy2"],
            lambda_=lambda_,
            masses=state.masses,
            cell=state.cell,
            pbc=state.pbc,
            system_idx=state.system_idx,
            atomic_numbers=state.atomic_numbers,
        )

        return initial_state  # noqa: RET504

    def langevin_update(
        state: ThermodynamicIntegrationMDState,
        dt: torch.Tensor = dt,
        kT: torch.Tensor = kT,
        gamma: torch.Tensor = gamma,
    ) -> ThermodynamicIntegrationMDState:
        """Perform one complete Langevin dynamics integration step.

        This function implements the BAOAB splitting scheme for Langevin dynamics,
        which provides accurate sampling of the canonical ensemble. The integration
        sequence is:
        1. Half momentum update using forces (B step)
        2. Half position update using updated momenta (A step)
        3. Full stochastic update with noise and friction (O step)
        4. Half position update using updated momenta (A step)
        5. Half momentum update using new forces (B step)

        Args:
            state (ThermodynamicIntegrationMDState): Current system state
                containing positions, momenta, forces
            dt (torch.Tensor): Integration timestep, either scalar or shape [n_batches]
            kT (torch.Tensor): Target temperature in energy units, either scalar or
                with shape [n_batches]
            gamma (torch.Tensor): Friction coefficient for Langevin thermostat,
                either scalar or with shape [n_batches]

        Returns:
            ThermodynamicIntegrationMDState: Updated state after one complete Langevin
                step with new positions, momenta, forces, and energy
        """
        # if isinstance(gamma, float):
        #     gamma = torch.tensor(gamma, device=device, dtype=dtype)

        if isinstance(dt, float):
            dt = torch.tensor(dt, device=device, dtype=dtype)
        state = momentum_step(state, dt / 2)
        state = position_step(state, dt / 2)
        state = ou_step(state, dt, kT, gamma)
        state = position_step(state, dt / 2)
        model_output = model(state)
        state.energy = model_output["energy"]
        state.forces = model_output["forces"]
        state.energy_difference = model_output["energy_difference"]
        state.energy1 = model_output["energy1"]
        state.energy2 = model_output["energy2"]

        return momentum_step(state, dt / 2)

    return langevin_init, langevin_update


def run_non_equilibrium_md(  # noqa: C901 PLR0915
    system: Any,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    save_dir: str,
    integrator: Callable,
    *,
    n_steps: int = 1000,
    lambda_schedule: str | Callable = "linear",
    reverse: bool = False,
    temperature: float = 300.0,
    timestep: float = 0.002,
    pbar: bool | dict[str, Any] = False,
    trajectory_reporter: ts.TrajectoryReporter | None = None,
    step_frequency: int = 1,
    autobatcher: bool = False,
    state_frequency: int = 50,
    **integrator_kwargs,
) -> ts.SimState:
    """Run non-equilibrium molecular dynamics simulation.

    Args:
        system: Initial system state, possibly batched
        model_a: First model for thermodynamic integration
        model_b: Second model for thermodynamic integration
        save_dir: Directory to save trajectory files
        integrator: Integration function
        n_steps: Number of simulation steps
        lambda_schedule: Lambda schedule type ("linear", "quadratic", "paper")
        reverse: Reverse the Lambda schedule for backward TI
            for non symmetric lambda paths
        temperature: Temperature for simulation
        timestep: Integration timestep
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.
        trajectory_reporter: Reporter for trajectory data
        step_frequency: Frequency for reporting steps
        autobatcher: Whether to use automatic batching
        state_frequency: Frequency for state reporting
        **integrator_kwargs: Additional integrator arguments

    Returns:
        Final simulation state
    """
    unit_system = UnitSystem.metal

    # Validate lambda schedule
    if isinstance(lambda_schedule, str):
        if lambda_schedule not in LAMBDA_SCHEDULES:
            raise ValueError(
                f"Unknown lambda schedule: {lambda_schedule}. "
                f"Available: {list(LAMBDA_SCHEDULES.keys())}"
            )
        schedule_fn = LAMBDA_SCHEDULES[lambda_schedule]

    if isinstance(lambda_schedule, Callable):
        schedule_fn = lambda_schedule

    def lambda_schedule(step: int) -> float:
        if reverse:
            return schedule_fn(n_steps - 1 - step, n_steps - 1)
        return schedule_fn(step, n_steps - 1)

    # Ensure system is a single system (not batched)
    if isinstance(system, list):
        raise TypeError("system should be a single system, not a list. ")

    model = MixedModel(
        model1=model_a,
        model2=model_b,
        device=model_b.device,
        dtype=model_b.dtype,
    )
    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    kT = (
        torch.as_tensor(temperature, dtype=dtype, device=device) * unit_system.temperature
    )

    # Create filenames for trajectory files
    filenames = [
        os.path.join(save_dir, f"trajectory_steered_{replica_idx}.h5")
        for replica_idx in range(state.n_systems)
    ]

    trajectory_reporter = ts.TrajectoryReporter(
        filenames=filenames,
        state_frequency=state_frequency,
        prop_calculators={
            step_frequency: {
                "energy_diff": lambda state: state.energy_difference,
                "energy": lambda state: state.energy,
                "energy1": lambda state: state.energy1,
                "energy2": lambda state: state.energy2,
                "lambda_": lambda state: state.lambda_,
            },
            10: {
                "temperature": lambda state: ts.quantities.calc_temperature(
                    masses=state.masses,
                    momenta=state.momenta,
                    system_idx=state.system_idx,
                )
            },
        },
    )

    if not kT.ndim == 0:
        raise TypeError("temperature must be a single float value.")

    init_fn, update_fn = integrator(
        model=model,
        kT=kT,
        dt=torch.tensor(timestep * unit_system.time, dtype=dtype, device=device),
        **integrator_kwargs,
    )

    # batch_iterator will be a list if autobatcher is False
    batch_iterator = _configure_batches_iterator(model, state, autobatcher)
    trajectory_reporter = _configure_reporter(
        trajectory_reporter,
        properties=["kinetic_energy", "potential_energy", "temperature"],
    )

    final_states: list[SimState] = []
    log_filenames = trajectory_reporter.filenames if trajectory_reporter else None

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Integrate")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=state.n_systems, **pbar_kwargs)

    for state, batch_indices in batch_iterator:
        # Initialize lambda values based on batch indices
        lambda_values = torch.ones(
            state.n_systems, dtype=dtype, device=device
        ) * lambda_schedule(0)
        state = init_fn(state, lambda_=lambda_values, kT=kT)

        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[log_filenames[i] for i in batch_indices]
            )

        # Thermodynamic integration phase
        ti_bar = tqdm(
            range(1, n_steps + 1),
            desc="TI Integration",
            disable=not pbar,
            mininterval=0.5,
        )

        for step in ti_bar:
            # Calculate lambda using the selected schedule
            lambda_value = lambda_schedule(step - 1)

            # Update lambda values
            if len(batch_indices) > 0:
                new_lambdas = torch.full_like(
                    batch_indices, lambda_value, dtype=dtype, device=device
                )
            else:
                new_lambdas = torch.full(
                    (state.n_systems,), lambda_value, dtype=dtype, device=device
                )

            state.lambda_ = new_lambdas

            # Update state
            state = update_fn(state, kT=kT)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)

        # finish the trajectory reporter
        final_states.append(state)
        if tqdm_pbar:
            tqdm_pbar.update(state.n_batches)

    if trajectory_reporter:
        trajectory_reporter.finish()

    if isinstance(batch_iterator, BinningAutoBatcher):
        reordered_states = batch_iterator.restore_original_order(final_states)
        return concatenate_states(reordered_states)

    return state


def run_equilibrium_md(  # noqa: C901
    system: Any,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    lambdas: torch.Tensor,
    save_dir: str,
    integrator: Callable,
    *,
    n_steps: int = 1000,
    temperature: float = 300.0,
    timestep: float = 0.002,
    pbar: bool | dict[str, Any] = False,
    trajectory_reporter: ts.TrajectoryReporter | None = None,
    step_frequency: int = 1,
    filenames: str | None = None,
    autobatcher: bool = False,
    state_frequency: int = 50,
    **integrator_kwargs,
) -> ts.SimState:
    """Run equilibrium molecular dynamics simulation.

    Args:
        system: Initial system state, possibly batched
        model_a: First model for thermodynamic integration
        model_b: Second model for thermodynamic integration
        lambdas: Tensor of lambda values for each system in the batch
        save_dir: Directory to save trajectory files
        integrator: Integration function
        n_steps: Number of simulation steps
        reverse: Reverse the Lambda schedule for backward TI for
            non symmetric lambda paths
        temperature: Temperature for simulation
        timestep: Integration timestep
        pbar (bool | dict[str, Any], optional): Show a progress bar.
            Only works with an autobatcher in interactive shell. If a dict is given,
            it's passed to `tqdm` as kwargs.
        trajectory_reporter: Reporter for trajectory data
        step_frequency: Frequency for reporting steps
        filenames: List of filenames for trajectory files. If None, defaults will be used.
            Useful when running sequential thermodynamic integration
        autobatcher: Whether to use automatic batching
        state_frequency: Frequency for state reporting
        **integrator_kwargs: Additional integrator arguments

    Returns:
        Final simulation state
    """
    unit_system = UnitSystem.metal

    if lambdas.ndim == 0:
        lambdas = lambdas.unsqueeze(0)

    # Ensure system is a single system (not batched)
    if isinstance(system, list):
        raise TypeError("system should be a single system, not a list. ")
    if len(lambdas) != len(lambdas.unique()):
        raise ValueError(
            "Lambda list must be unique.Batch of different systems is not supported yet."
        )

    model = MixedModel(
        model1=model_a,
        model2=model_b,
        device=model_b.device,
        dtype=model_b.dtype,
    )
    state: SimState = initialize_state(system, model.device, model.dtype)
    dtype, device = state.dtype, state.device
    kT = (
        torch.as_tensor(temperature, dtype=dtype, device=device) * unit_system.temperature
    )
    if state.n_systems != len(lambdas):
        raise ValueError(
            f"Number of systems in state ({state.n_systems}) must match "
            f"number of lambda values ({len(lambdas)})."
        )

    # Create filenames for trajectory files
    if filenames is None:
        filenames = [
            os.path.join(save_dir, f"trajectory_lambda_{replica_idx}.h5")
            for replica_idx in range(len(lambdas))
        ]
    else:
        filenames = [os.path.join(save_dir, filename) for filename in filenames]

    trajectory_reporter = ts.TrajectoryReporter(
        filenames=filenames,
        state_frequency=state_frequency,
        prop_calculators={
            step_frequency: {
                "energy_diff": lambda state: state.energy_difference,
                "energy": lambda state: state.energy,
                # "energy1": lambda state: state.energy1,
                # "energy2": lambda state: state.energy2,
                "lambda_": lambda state: state.lambda_,
            },
            10: {
                "temperature": lambda state: ts.quantities.calc_temperature(
                    masses=state.masses,
                    momenta=state.momenta,
                    system_idx=state.system_idx,
                )
            },
        },
    )

    if not kT.ndim == 0:
        raise TypeError("temperature must be a single float value.")

    init_fn, update_fn = integrator(
        model=model,
        kT=kT,
        dt=torch.tensor(timestep * unit_system.time, dtype=dtype, device=device),
        **integrator_kwargs,
    )

    # batch_iterator will be a list if autobatcher is False
    batch_iterator = _configure_batches_iterator(model, state, autobatcher)
    trajectory_reporter = _configure_reporter(
        trajectory_reporter,
        properties=["kinetic_energy", "potential_energy", "temperature"],
    )

    final_states: list[SimState] = []
    log_filenames = trajectory_reporter.filenames if trajectory_reporter else None

    tqdm_pbar = None
    if pbar and autobatcher:
        pbar_kwargs = pbar if isinstance(pbar, dict) else {}
        pbar_kwargs.setdefault("desc", "Integrate")
        pbar_kwargs.setdefault("disable", None)
        tqdm_pbar = tqdm(total=state.n_systems, **pbar_kwargs)

    for state, batch_indices in batch_iterator:
        state = init_fn(state, lambda_=lambdas, kT=kT)

        # set up trajectory reporters
        if autobatcher and trajectory_reporter:
            # we must remake the trajectory reporter for each batch
            trajectory_reporter.load_new_trajectories(
                filenames=[log_filenames[i] for i in batch_indices]
            )

        # Thermodynamic integration phase
        ti_bar = tqdm(
            range(1, n_steps + 1),
            desc="TI Integration",
            disable=not pbar,
            mininterval=0.5,
        )

        for step in ti_bar:
            # Update state
            state = update_fn(state, kT=kT)

            if trajectory_reporter:
                trajectory_reporter.report(state, step, model=model)

        # finish the trajectory reporter
        final_states.append(state)
        if tqdm_pbar:
            tqdm_pbar.update(state.n_batches)

    if trajectory_reporter:
        trajectory_reporter.finish()

    if isinstance(batch_iterator, BinningAutoBatcher):
        reordered_states = batch_iterator.restore_original_order(final_states)
        return concatenate_states(reordered_states)

    return state
