"""Velocity autocorrelation example."""

# /// script
# dependencies = [
#     "ase>=3.24",
#     "matplotlib",
#     "numpy",
# ]
# ///

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.properties.correlations import VelocityAutoCorrelation
from torch_sim.units import MetalUnits as Units


def prepare_system() -> tuple[
    Any, Any, torch.Tensor, torch.Tensor, torch.device, torch.dtype, float
]:
    """Create and prepare Ar system with LJ potential."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Using solid Ar w/ LJ for ease
    atoms = bulk("Ar", crystalstructure="fcc", a=5.256, cubic=True)
    atoms = atoms.repeat((3, 3, 3))
    temperature = 50.0  # Kelvin
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

    epsilon = 0.0104  # eV
    sigma = 3.4  # Å
    cutoff = 2.5 * sigma

    lj_model = LennardJonesModel(
        sigma=sigma,
        epsilon=epsilon,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        compute_forces=True,
    )

    timestep = 0.001  # ps (1 fs)
    dt = torch.tensor(timestep * Units.time, device=device, dtype=dtype)
    temp_kT = temperature * Units.temperature  # Convert K to internal units
    kT = torch.tensor(temp_kT, device=device, dtype=dtype)

    return state, lj_model, dt, kT, device, dtype, timestep


def plot_results(*, time: np.ndarray, vacf: np.ndarray, window_count: int) -> None:
    """Plot VACF results."""
    plt.figure(figsize=(10, 8))
    plt.plot(time, vacf, "b-", linewidth=2)
    plt.xlabel("Time (fs)", fontsize=12)
    plt.ylabel("VACF", fontsize=12)
    plt.title(f"VACF (Average of {window_count} windows)", fontsize=14)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.ylim(-0.6, 1.1)
    plt.tight_layout()
    plt.savefig("vacf_example.png")


def main() -> None:
    """Run velocity autocorrelation simulation using Lennard-Jones model."""
    state, lj_model, dt, kT, device, dtype, timestep = prepare_system()
    nve_init, nve_update = ts.integrators.nve(model=lj_model, dt=dt, kT=kT)
    state = nve_init(state)  # type: ignore[call-arg]

    window_size = 150  # Length of correlation: dt * correlation_dt * window_size
    vacf_calc = VelocityAutoCorrelation(
        window_size=window_size,
        device=device,
        use_running_average=True,
        normalize=True,
    )

    # Sampling freq is controlled by prop_calculators
    trajectory = "vacf_example.h5"
    correlation_dt = 10  # Step delta between correlations
    reporter = ts.TrajectoryReporter(
        trajectory,
        state_frequency=100,
        prop_calculators={correlation_dt: {"vacf": vacf_calc}},
    )

    num_steps = 15000  # NOTE: short run
    for step in range(num_steps):
        state = nve_update(state)  # type: ignore[call-arg]
        reporter.report(state, step)

    reporter.close()

    # VACF results and plot
    # Timesteps -> Time in fs
    time_steps = np.arange(window_size)
    time = time_steps * correlation_dt * timestep * 1000

    if vacf_calc.vacf is not None:
        plot_results(
            time=time,
            vacf=vacf_calc.vacf.cpu().numpy(),
            # Just for demo purposes
            window_count=vacf_calc._window_count,  # noqa: SLF001
        )


if __name__ == "__main__":
    main()
