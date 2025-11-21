"""Free energy prediction workflows using thermodynamic integration.

This module provides complete workflows for free energy prediction using
thermodynamic integration methods with Einstein model references.
"""

import logging
import os
from collections.abc import Callable

import torch

import torch_sim as ts
from torch_sim.integrators.nvt import nvt_langevin
from torch_sim.models.einstein import EinsteinModel
from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState
from torch_sim.steered_md import (
    nvt_langevin_thermodynamic_integration,
    run_equilibrium_md,
    run_non_equilibrium_md,
)
from torch_sim.units import BaseConstant, UnitConversion


logger = logging.getLogger(__name__)


def compute_free_energy_jarzynski(work: torch.Tensor, temperature: float) -> torch.Tensor:
    r"""Compute free energy difference using Jarzynski equality.

    \Delta_F = -kT ln < exp(-W/kT) > = -kT ln (1/N sum_i exp(-W_i/kT))

    Uses logsumexp for numerical stability.

    Args:
        work: Tensor of shape [n_trajectories] with work values.
        temperature: Temperature in K.

    Returns:
        free_energy: Tensor with free energy difference at each step.
    """
    kB = BaseConstant.k_B / UnitConversion.eV_to_J  # Boltzmann constant in eV/K
    beta = 1 / (kB * temperature)
    n_traj = torch.tensor(work.shape[0], device=work.device)
    return -torch.logsumexp(-beta * work, dim=0) / beta + torch.log(n_traj) / beta


def compute_work_steered_md(
    energy_difference: torch.Tensor, lamdbas: torch.Tensor
) -> torch.Tensor:
    """Compute work done during steered MD.

    Args:
        energy_difference: Tensor of shape [n_steps] with energy differences.
        lamdbas: Tensor of shape [n_steps] with lambda values.

    Returns:
        work: Tensor of shape [n_trajectories] with work values.
    """
    delta_lambda = lamdbas[1:] - lamdbas[:-1]
    return torch.sum(energy_difference[:-1] * delta_lambda, dim=-1)


# TODO: modify to output one frequency per atom type instead of per atom
def compute_einstein_frequencies_from_nvt(
    system: SimState,
    model: ModelInterface,
    temperature: float,
    save_dir: str,
    n_steps: int = 1000,
    timestep: float = 0.001,
    filename: str = "nvt_frequency_calc.h5",
) -> torch.Tensor:
    """Compute Einstein model frequencies from NVT simulation square deviations.

    Uses equipartition theorem: omega = sqrt(3 k_B T / (m * <x^2>))

    Args:
        system: Simulation state.
        model: Model to use for forces and energy.
        temperature: Temperature in K.
        save_dir: Directory to save results.
        n_steps: Number of NVT steps for frequency calculation.
        timestep: Timestep for NVT simulation.
        filename: Name of the file to save results.

    Returns:
        frequencies: Tensor with Einstein frequencies for each atom.
    """
    filename = os.path.join(save_dir, filename)
    reporter = ts.TrajectoryReporter(
        filenames=[filename],
        state_frequency=10,
    )

    # Run NVT simulation
    _ = ts.integrate(
        system,
        model,
        integrator=nvt_langevin,
        n_steps=n_steps,
        timestep=timestep,
        temperature=temperature,
        trajectory_reporter=reporter,
    )

    # Load trajectory and compute square deviations
    with ts.TorchSimTrajectory(filename, mode="r") as traj:
        positions = traj.get_array("positions")  # [n_frames, n_atoms, 3]
        positions = torch.from_numpy(positions).to(
            device=system.device, dtype=system.dtype
        )

    # Compute average square deviation from mean position
    unwrapped_positions = ts.transforms.unwrap_positions(positions, system.cell[0])
    square_deviations = unwrapped_positions.var(dim=0).sum(dim=-1)  # (n_particles)

    # Compute frequencies using equipartition theorem
    # omega = sqrt(3/2 k_B T / (1/2 * m * <x^2>))
    kB = BaseConstant.k_B / UnitConversion.eV_to_J  # Boltzmann constant in eV/K

    return torch.sqrt(3 * kB * temperature / (system.masses * square_deviations))


# Complete workflow implementations for free energy prediction using
# thermodynamic integration
#
# Two main approaches are implemented:
#
# 1. Forward TI with Einstein reference:
#    - Compute Einstein model frequencies from NVT simulation of reference model
#    - Run multiple forward TI trajectories from Einstein model to target model
#    - Compute free energy difference using Jarzynski equality
#   Reference:
#   - Jarzynski equality: <https://doi.org/10.1103/PhysRevLett.78.2690>
# 2. Forward-backward TI:
#    - Run forward TI from model_a to model_b
#    - Equilibrate at model_b
#    - Run backward TI from model_b to model_a (with reverse=True)
#    - Compute free energy using Jarzynski equality and adiabatic switching method
#  Reference:
#   - de Koning, Maurice, and A. Antonelli.
#   "Adiabatic switching applied to realistic crystalline solids: Vacancy-formation
#   free energy in copper." Physical Review B 55.2 (1997): 735.
#
# Inspiration and sources:
# - https://calorine.materialsmodeling.org/get_started/free_energy_tutorial.html
# - Freitas, Rodrigo, Mark Asta, and Maurice De Koning.
#   "Nonequilibrium free-energy calculation of solids using LAMMPS."
#   Computational Materials Science 112 (2016): 333-341.
#
# Example usage:
#
# # Forward TI with Einstein reference
# from torch_sim.workflows.free_energy_workflows import (
#     run_forward_ti_with_einstein_workflow
# )
# results = run_forward_ti_with_einstein_workflow(
#     system=my_system,
#     model_a=reference_model,  # Used to compute Einstein frequencies
#     model_b=target_model,
#     temperature=300.0,
#     save_dir="./ti_results",
#     n_trajectories=10,
#     n_ti_steps=1000
# )
#
# # Forward-backward TI
# from torch_sim.workflows.free_energy_workflows import run_forward_backward_ti_workflow
# results = run_forward_backward_ti_workflow(
#     system=my_system,
#     model_a=model_a,
#     model_b=model_b,
#     temperature=300.0,
#     save_dir="./ti_results",
#     n_trajectories=10,
#     n_ti_steps=1000,
#     n_equil_steps=500
# )
#
# Both workflows return dictionaries with:
# - free energy differences
# - work distributions
# - trajectory data
# - statistical analysis (mean, std dev)
#


def run_forward_ti_workflow_from_einstein(
    system: SimState,
    model: ModelInterface,
    temperature: float,
    save_dir: str,
    *,
    frequencies: torch.Tensor | None = None,
    n_trajectories: int = 10,
    n_steps_frequency: int = 1000,
    n_ti_steps: int = 1000,
    timestep: float = 0.002,
    lambda_schedule: str | Callable = "linear",
) -> dict[str, torch.Tensor]:
    """Run standard forward thermodynamic integration workflow.

    Args:
        system: Initial system state.
        model: Target model.
        temperature: Temperature in K.
        save_dir: Directory to save trajectory files.
        frequencies: Precomputed Einstein model frequencies.
        n_trajectories: Number of TI trajectories.
        n_steps_frequency: Number of NVT steps to compute frequencies for Einstein model.
        n_ti_steps: Number of TI steps per trajectory.
        timestep: Integration timestep for TI.
        lambda_schedule: Lambda schedule ("linear" or "quadratic") or a custom function.

    Returns:
        Dictionary with free energy results and trajectory data.
    """
    os.makedirs(save_dir, exist_ok=True)

    if system.n_systems != 1:
        raise NotImplementedError("Only single system input is supported.")

    if frequencies is None:
        # Find frequencies for Einstein reference model
        frequencies = compute_einstein_frequencies_from_nvt(
            system=system,
            model=model,
            temperature=temperature,
            save_dir=save_dir,
            n_steps=n_steps_frequency,
            timestep=timestep,
        )
        # Use same frequencies for all atoms of same type
        frequencies = torch.full(
            (system.n_atoms,),
            frequencies.mean().item(),
            device=system.device,
            dtype=system.dtype,
        )

    logger.info(
        "Einstein frequencies (eV^(0.5)/A/amu^(0.5)): %s", frequencies.cpu().numpy()
    )

    # Prepare batched systems for multiple trajectories
    systems = [system.clone() for _ in range(n_trajectories)]
    batched_system = ts.state.concatenate_states(systems)

    # Define reference Einstein model
    einstein_model = EinsteinModel.from_atom_and_frequencies(
        atom=batched_system,
        frequencies=frequencies.repeat(n_trajectories),
        device=batched_system.device,
        dtype=batched_system.dtype,
    )
    einstein_model.compile()

    batched_system = einstein_model.sample(batched_system, temperature)

    # Run forward TI
    _ = run_non_equilibrium_md(
        system=batched_system,
        model_a=einstein_model,
        model_b=model,
        save_dir=save_dir,
        integrator=nvt_langevin_thermodynamic_integration,
        n_steps=n_ti_steps,
        lambda_schedule=lambda_schedule,
        reverse=False,
        temperature=temperature,
        timestep=timestep,
        pbar=True,
    )

    # Load trajectory data and compute work values
    work_values = []
    energy_differences = []
    lambdas = []

    for i in range(n_trajectories):
        filename = os.path.join(save_dir, f"trajectory_steered_{i}.h5")
        with ts.TorchSimTrajectory(filename, mode="r") as traj:
            energy_diff = (
                torch.from_numpy(traj.get_array("energy_diff"))
                .to(batched_system.device)
                .squeeze()
            )
            lambda_vals = (
                torch.from_numpy(traj.get_array("lambda_"))
                .to(batched_system.device)
                .squeeze()
            )

            energy_diff_per_atom = energy_diff / batched_system.n_atoms_per_system[i]

            # Compute work for this trajectory
            work = compute_work_steered_md(energy_diff_per_atom, lambda_vals).item()

            work_values.append(work)
            energy_differences.append(energy_diff_per_atom)
            lambdas.append(lambda_vals)

    work_tensor = torch.tensor(work_values).to(batched_system.device)

    # Compute free energy using Jarzynski equality
    free_energy = compute_free_energy_jarzynski(work_tensor, temperature)
    free_energy_einstein = (
        einstein_model.get_free_energy(temperature)["free_energy"][0]
        / batched_system.n_atoms_per_system[i]
    )

    return {
        "free_energy": free_energy + free_energy_einstein,
        "free_energy_einstein": free_energy_einstein,
        "free_energy_difference": free_energy,
        "work_values": work_tensor,
        "mean_work": work_tensor.mean(),
        "std_work": work_tensor.std(),
        "energy_differences": torch.stack(energy_differences),
        "lambda_values": torch.stack(lambdas),
    }


def run_forward_backward_ti_workflow_from_einstein(
    system: SimState,
    model: ModelInterface,
    temperature: float,
    save_dir: str,
    *,
    n_trajectories: int = 10,
    n_steps_frequency: int = 1000,
    frequencies: torch.Tensor | None = None,
    n_ti_steps: int = 1000,
    n_equil_steps: int = 500,
    timestep: float = 0.002,
    lambda_schedule: str = "linear",
) -> dict[str, torch.Tensor]:
    """Run forward-backward thermodynamic integration workflow.

    Workflow:
    1. Run forward TI from Einstein to model
    2. Equilibrate at model for n_equil_steps
    3. Run backward TI from model back to Einstein
    4. Compute free energy using linear response theory

    Args:
        system: Initial system state.
        model: Target model.
        temperature: Temperature in K.
        save_dir: Directory to save trajectory files.
        n_trajectories: Number of TI trajectories.
        n_steps_frequency: Number of NVT steps to compute frequencies for Einstein model.
        frequencies: Precomputed Einstein model frequencies.
        n_ti_steps: Number of TI steps per trajectory.
        n_equil_steps: Number of equilibration steps at model_b.
        timestep: Integration timestep.
        lambda_schedule: Lambda schedule:
            "linear", "quadratic", "cubic", "lammps" or a custom function.

    Returns:
        Dictionary with free energy results and trajectory data.
    """
    os.makedirs(save_dir, exist_ok=True)

    if system.n_systems != 1:
        raise NotImplementedError("Only single system input is supported.")

    if frequencies is None:
        # Find frequencies for Einstein reference model
        frequencies = compute_einstein_frequencies_from_nvt(
            system=system,
            model=model,
            temperature=temperature,
            save_dir=save_dir,
            n_steps=n_steps_frequency,
            timestep=timestep,
        )
        # Use same frequencies for all atoms of same type
        frequencies = torch.full(
            (system.n_atoms,),
            frequencies.mean().item(),
            device=system.device,
            dtype=system.dtype,
        )

    # Prepare batched systems for multiple trajectories
    systems = [system.clone() for _ in range(n_trajectories)]
    batched_system = ts.state.concatenate_states(systems)

    # Define reference Einstein model
    einstein_model = EinsteinModel.from_atom_and_frequencies(
        atom=batched_system,
        frequencies=frequencies.repeat(n_trajectories),
        device=batched_system.device,
        dtype=batched_system.dtype,
    )

    batched_system = einstein_model.sample(batched_system, temperature)

    logger.info("Running %d forward TI trajectories...", n_trajectories)

    # Create separate save directories for forward and backward
    forward_dir = os.path.join(save_dir, "forward")
    backward_dir = os.path.join(save_dir, "backward")
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(backward_dir, exist_ok=True)

    # Run forward TI (A -> B)
    forward_final_state = run_non_equilibrium_md(
        system=batched_system,
        model_a=einstein_model,
        model_b=model,
        save_dir=forward_dir,
        integrator=nvt_langevin_thermodynamic_integration,
        n_steps=n_ti_steps,
        lambda_schedule=lambda_schedule,
        reverse=False,
        temperature=temperature,
        timestep=timestep,
        pbar=True,
    )

    # Equilibrate at model_b
    logger.info("Equilibrating at target model for %d steps...", n_equil_steps)
    equilibrated_state = ts.integrate(
        system=forward_final_state,
        model=model,
        integrator=nvt_langevin,
        n_steps=n_equil_steps,
        temperature=temperature,
        timestep=timestep,
    )

    logger.info("Running %d backward TI trajectories...", n_trajectories)

    # Run backward TI (B -> A)
    _ = run_non_equilibrium_md(
        system=equilibrated_state,
        model_a=einstein_model,
        model_b=model,
        save_dir=backward_dir,
        integrator=nvt_langevin_thermodynamic_integration,
        n_steps=n_ti_steps,
        lambda_schedule=lambda_schedule,
        reverse=True,  # This is the key difference
        temperature=temperature,
        timestep=timestep,
        pbar=True,
    )

    # Compute work values for both directions
    forward_work_values = []
    backward_work_values = []

    for i in range(n_trajectories):
        # Forward work
        forward_filename = os.path.join(forward_dir, f"trajectory_steered_{i}.h5")
        with ts.TorchSimTrajectory(forward_filename, mode="r") as traj:
            energy_diff = (
                torch.from_numpy(traj.get_array("energy_diff"))
                .to(batched_system.device)
                .squeeze()
            )
            lambda_vals = (
                torch.from_numpy(traj.get_array("lambda_"))
                .to(batched_system.device)
                .squeeze()
            )

            energy_diff_per_atom = energy_diff / batched_system.n_atoms_per_system[i]

            forward_work = compute_work_steered_md(
                energy_diff_per_atom, lambda_vals
            ).item()
            forward_work_values.append(forward_work)

        # Backward work
        backward_filename = os.path.join(backward_dir, f"trajectory_steered_{i}.h5")
        with ts.TorchSimTrajectory(backward_filename, mode="r") as traj:
            energy_diff = (
                torch.from_numpy(traj.get_array("energy_diff"))
                .to(batched_system.device)
                .squeeze()
            )
            lambda_vals = (
                torch.from_numpy(traj.get_array("lambda_"))
                .to(batched_system.device)
                .squeeze()
            )

            energy_diff_per_atom = energy_diff / batched_system.n_atoms_per_system[i]

            backward_work = compute_work_steered_md(
                energy_diff_per_atom, lambda_vals
            ).item()
            backward_work_values.append(backward_work)

    forward_work = torch.tensor(forward_work_values).to(batched_system.device)
    backward_work = torch.tensor(backward_work_values).to(batched_system.device)

    forward_free_energy = compute_free_energy_jarzynski(forward_work, temperature)
    backward_free_energy = -compute_free_energy_jarzynski(backward_work, temperature)

    free_energy_difference = (forward_work.mean() - backward_work.mean()) / 2
    free_energy_einstein = (
        einstein_model.get_free_energy(temperature)["free_energy"][0]
        / batched_system.n_atoms_per_system[i]
    )

    return {
        "forward_free_energy": forward_free_energy + free_energy_einstein,
        "backward_free_energy": backward_free_energy + free_energy_einstein,
        "free_energy": free_energy_difference + free_energy_einstein,
        "free_energy_difference": free_energy_difference,
        "free_energy_einstein": free_energy_einstein,
        "forward_work": forward_work,
        "backward_work": backward_work,
        "forward_mean_work": forward_work.mean(),
        "forward_std_work": forward_work.std(),
        "backward_mean_work": backward_work.mean(),
        "backward_std_work": backward_work.std(),
    }


def run_thermodynamic_integration_from_einstein(
    system: SimState,
    model: ModelInterface,
    temperature: float,
    save_dir: str,
    *,
    run_parallel: bool = False,
    lambdas: torch.Tensor,
    n_steps_frequency: int = 1000,
    frequencies: torch.Tensor | None = None,
    n_ti_steps: int = 1000,
    timestep: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Run standard forward thermodynamic integration workflow.

    Args:
        system: Initial system state.
        model: Target model.
        temperature: Temperature in K.
        save_dir: Directory to save trajectory files.
        run_parallel: Whether to run all trajectories in parallel from same initial state
            or sequentially, using final state of previous trajectory as initial state
            for next trajectory.
        lambdas: tensor with lambda values to use for TI.
        n_steps_frequency: Number of NVT steps to compute frequencies for Einstein model.
        frequencies: Precomputed Einstein model frequencies.
        n_ti_steps: Number of TI steps per trajectory.
        timestep: Integration timestep for TI.

    Returns:
        Dictionary with free energy results and trajectory data.
    """
    os.makedirs(save_dir, exist_ok=True)

    if system.n_systems != 1:
        raise NotImplementedError("Only single system input is supported.")

    if frequencies is None:
        # Find frequencies for Einstein reference model
        frequencies = compute_einstein_frequencies_from_nvt(
            system=system,
            model=model,
            temperature=temperature,
            save_dir=save_dir,
            n_steps=n_steps_frequency,
            timestep=timestep,
        )
        # Use same frequencies for all atoms of same type
        frequencies = torch.full(
            (system.n_atoms,),
            frequencies.mean().item(),
            device=system.device,
            dtype=system.dtype,
        )

    if run_parallel:
        # Prepare batched systems for multiple trajectories
        systems = [system.clone() for _ in range(len(lambdas))]
        batched_system = ts.state.concatenate_states(systems)

        # Define reference Einstein model
        einstein_model = EinsteinModel.from_atom_and_frequencies(
            atom=batched_system,
            frequencies=frequencies.repeat(len(lambdas)),
            device=batched_system.device,
            dtype=batched_system.dtype,
        )

        # Run forward TI
        _ = run_equilibrium_md(
            system=batched_system,
            model_a=einstein_model,
            model_b=model,
            lambdas=lambdas,
            save_dir=save_dir,
            integrator=nvt_langevin_thermodynamic_integration,
            n_steps=n_ti_steps,
            temperature=temperature,
            timestep=timestep,
            pbar=True,
        )
    else:
        # Define reference Einstein model
        einstein_model = EinsteinModel.from_atom_and_frequencies(
            atom=system,
            frequencies=frequencies,
            device=system.device,
            dtype=system.dtype,
        )
        batched_system = system.clone()
        for i, lambda_ in enumerate(lambdas):
            # Run forward TI
            batched_system = run_equilibrium_md(
                system=batched_system,
                model_a=einstein_model,
                model_b=model,
                lambdas=lambda_,
                save_dir=save_dir,
                integrator=nvt_langevin_thermodynamic_integration,
                n_steps=n_ti_steps,
                filenames=[f"trajectory_lambda_{i}.h5"],
                temperature=temperature,
                timestep=timestep,
                pbar=True,
            )

    # Load trajectory data and compute work values
    work_values = []

    for i in range(len(lambdas)):
        filename = os.path.join(save_dir, f"trajectory_lambda_{i}.h5")
        with ts.TorchSimTrajectory(filename, mode="r") as traj:
            energy_diff = (
                torch.from_numpy(traj.get_array("energy_diff"))
                .to(batched_system.device)
                .squeeze()
            )

            energy_diff_per_atom = energy_diff / batched_system.n_atoms_per_system[0]

            # Compute work for this trajectory
            work = energy_diff_per_atom.mean().item()

            work_values.append(work)

    work_tensor = torch.tensor(work_values).to(batched_system.device)

    # integrate work values over lambda, e.g. using trapezoidal rule
    free_energy_difference = torch.trapezoid(work_tensor, lambdas).item()

    # Compute free energy using Jarzynski equality
    free_energy_einstein = (
        einstein_model.get_free_energy(temperature)["free_energy"][0]
        / batched_system.n_atoms_per_system[0]
    )

    return {
        "free_energy": free_energy_difference + free_energy_einstein,
        "free_energy_einstein": free_energy_einstein,
        "free_energy_difference": free_energy_difference,
        "lambda_values": lambdas,
        "work_values": work_tensor,
    }
