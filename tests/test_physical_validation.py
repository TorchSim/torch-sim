"""Physical validation tests for torch-sim MD integrators.

Uses the physical_validation library (https://github.com/shirtsgroup/physical_validation)
to verify that integrators produce physically correct results. These tests are
long-running (~5 min total) and excluded by default. Run with:

    pytest -m physical_validation -v
"""

import numpy as np
import pytest
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import MetalUnits

physical_validation = pytest.importorskip("physical_validation")

DEVICE = torch.device("cpu")
DTYPE = torch.float64

# LJ Argon parameters
SIGMA = 3.405
EPSILON = 0.0104
CUTOFF = 2.5 * SIGMA


def _make_unit_data():
    """Create UnitData for torch-sim's MetalUnits system."""
    return physical_validation.data.UnitData(
        kb=float(MetalUnits.temperature),  # k_B in eV/K = 8.617e-5
        energy_str="eV",
        energy_conversion=1.0,
        length_str="Ang",
        length_conversion=1.0,
        volume_str="Ang^3",
        volume_conversion=1.0,
        temperature_str="K",
        temperature_conversion=1.0,
        pressure_str="eV/Ang^3",
        pressure_conversion=1.0,
        time_str="internal",
        time_conversion=1.0,
    )


def _make_lj_model():
    """Create a Lennard-Jones model for Argon."""
    return LennardJonesModel(
        use_neighbor_list=False,
        sigma=SIGMA,
        epsilon=EPSILON,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
        cutoff=CUTOFF,
    )


def _make_ar_supercell(repeat=(2, 2, 2)):
    """Create an FCC Argon supercell SimState."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(repeat)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


def _run_nvt_langevin(
    sim_state,
    model,
    temperature,
    timestep_ps,
    n_steps,
    n_equilibration,
    seed=42,
):
    """Run NVT Langevin simulation and collect per-step observables."""
    kT = temperature * float(MetalUnits.temperature)
    dt_internal = timestep_ps * float(MetalUnits.time)
    natoms = int(sim_state.positions.shape[0])

    state = ts.nvt_langevin_init(sim_state, model, kT=kT, seed=seed)

    # Equilibration
    for _ in range(n_equilibration):
        state = ts.nvt_langevin_step(state, model, dt=dt_internal, kT=kT)

    # Production - collect observables
    ke_list = []
    pe_list = []
    total_e_list = []
    position_list = []
    velocity_list = []

    for _ in range(n_steps):
        state = ts.nvt_langevin_step(state, model, dt=dt_internal, kT=kT)

        ke = ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta)
        pe = state.energy.sum()
        ke_list.append(float(ke))
        pe_list.append(float(pe))
        total_e_list.append(float(ke + pe))
        position_list.append(state.positions.detach().cpu().numpy().copy())
        velocity_list.append(state.velocities.detach().cpu().numpy().copy())

    # Compute volume from cell
    cell = sim_state.cell[0].detach().cpu().numpy()
    volume = float(np.abs(np.linalg.det(cell)))

    return {
        "kinetic_energy": np.array(ke_list),
        "potential_energy": np.array(pe_list),
        "total_energy": np.array(total_e_list),
        "positions": np.array(position_list),
        "velocities": np.array(velocity_list),
        "volume": volume,
        "masses": sim_state.masses.detach().cpu().numpy(),
        "dt_internal": dt_internal,
        "natoms": natoms,
    }


def _run_nve(sim_state, model, kT_init, timestep_ps, n_steps, seed=42):
    """Run NVE simulation and collect constant of motion."""
    dt_internal = timestep_ps * float(MetalUnits.time)

    state = ts.nve_init(sim_state, model, kT=kT_init, seed=seed)

    com_list = []
    for _ in range(n_steps):
        state = ts.nve_step(state, model, dt=dt_internal)
        ke = ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta)
        pe = state.energy.sum()
        com_list.append(float(ke + pe))

    return {
        "constant_of_motion": np.array(com_list),
        "dt_internal": dt_internal,
    }


def _build_nvt_simulation_data(run_data, temperature):
    """Build a physical_validation SimulationData from NVT run results."""
    units = _make_unit_data()

    system = physical_validation.data.SystemData(
        natoms=run_data["natoms"],
        nconstraints=0,
        ndof_reduction_tra=3,
        ndof_reduction_rot=0,
        mass=run_data["masses"],
    )

    ensemble_data = physical_validation.data.EnsembleData(
        ensemble="NVT",
        natoms=run_data["natoms"],
        volume=run_data["volume"],
        temperature=temperature,
    )

    observables = physical_validation.data.ObservableData(
        kinetic_energy=run_data["kinetic_energy"],
        potential_energy=run_data["potential_energy"],
        total_energy=run_data["total_energy"],
    )

    trajectory = physical_validation.data.TrajectoryData(
        position=run_data["positions"],
        velocity=run_data["velocities"],
    )

    return physical_validation.data.SimulationData(
        units=units,
        dt=run_data["dt_internal"],
        system=system,
        ensemble=ensemble_data,
        observables=observables,
        trajectory=trajectory,
    )


@pytest.mark.physical_validation
def test_ke_distribution():
    """Test that kinetic energy follows the Maxwell-Boltzmann distribution.

    Runs NVT Langevin at 100K on a 2x2x2 Ar supercell (32 atoms) and checks
    that the KE distribution matches the analytical Maxwell-Boltzmann prediction.
    """
    sim_state = _make_ar_supercell(repeat=(2, 2, 2))
    model = _make_lj_model()
    temperature = 100.0  # K

    run_data = _run_nvt_langevin(
        sim_state,
        model,
        temperature=temperature,
        timestep_ps=0.004,
        n_steps=10_000,
        n_equilibration=2_000,
        seed=42,
    )

    data = _build_nvt_simulation_data(run_data, temperature)

    result = physical_validation.kinetic_energy.distribution(
        data,
        strict=False,
        verbosity=0,
    )
    # strict=False returns (d_mean, d_width) in sigma units
    d_mean, d_width = result

    assert abs(d_mean) < 3, (
        f"KE mean deviation {d_mean:.2f} sigma exceeds threshold"
    )
    assert abs(d_width) < 3, (
        f"KE width deviation {d_width:.2f} sigma exceeds threshold"
    )


@pytest.mark.physical_validation
def test_integrator_convergence():
    """Test that NVE energy error scales as dt^2 (velocity Verlet).

    Runs NVE at 3 different timesteps from identical initial conditions on a
    4-atom Ar unit cell at low temperature (5K). Low temperature minimizes
    thermal fluctuations so the integration error dominates the RMSD of the
    conserved quantity, allowing the dt^2 convergence to be observed.
    """
    sim_state = _make_ar_supercell(repeat=(1, 1, 1))  # 4 atoms
    model = _make_lj_model()
    temperature = 5.0  # K, low T so integration error dominates
    kT_init = temperature * float(MetalUnits.temperature)

    # Timesteps chosen so integration error >> thermal fluctuations at all dt.
    # Factor of ~sqrt(2) spacing gives dt^2 ratio of ~2.0 per step.
    timesteps = [0.008, 0.00566, 0.004]  # ps
    n_steps = 5_000
    seed = 42

    natoms = int(sim_state.positions.shape[0])
    masses = sim_state.masses.detach().cpu().numpy()
    volume = float(
        np.abs(np.linalg.det(sim_state.cell[0].detach().cpu().numpy()))
    )
    units = _make_unit_data()

    simulations = []
    for dt_ps in timesteps:
        run_data = _run_nve(
            sim_state,
            model,
            kT_init=kT_init,
            timestep_ps=dt_ps,
            n_steps=n_steps,
            seed=seed,
        )

        system = physical_validation.data.SystemData(
            natoms=natoms,
            nconstraints=0,
            ndof_reduction_tra=3,
            ndof_reduction_rot=0,
            mass=masses,
        )

        ensemble_data = physical_validation.data.EnsembleData(
            ensemble="NVE",
            natoms=natoms,
            volume=volume,
        )

        observables = physical_validation.data.ObservableData(
            constant_of_motion=run_data["constant_of_motion"],
        )

        sim_data = physical_validation.data.SimulationData(
            units=units,
            dt=run_data["dt_internal"],
            system=system,
            ensemble=ensemble_data,
            observables=observables,
        )
        simulations.append(sim_data)

    result = physical_validation.integrator.convergence(
        simulations,
        verbose=False,
    )

    assert result < 0.5, (
        f"Integrator convergence deviation {result:.3f} exceeds threshold 0.5"
    )


@pytest.mark.physical_validation
def test_ensemble_check():
    """Test NVT ensemble validity via Boltzmann weight ratio at two temperatures.

    Runs NVT Langevin at 80K and 100K on a 2x2x2 Ar supercell (32 atoms),
    then checks that the total energy distributions satisfy the expected
    Boltzmann weight relationship. Uses total_energy=True for the ensemble
    check which includes both kinetic and potential energy contributions.
    """
    sim_state = _make_ar_supercell(repeat=(2, 2, 2))
    model = _make_lj_model()

    temp_low = 80.0
    temp_high = 100.0

    run_low = _run_nvt_langevin(
        sim_state,
        model,
        temperature=temp_low,
        timestep_ps=0.004,
        n_steps=10_000,
        n_equilibration=2_000,
        seed=42,
    )
    run_high = _run_nvt_langevin(
        sim_state,
        model,
        temperature=temp_high,
        timestep_ps=0.004,
        n_steps=10_000,
        n_equilibration=2_000,
        seed=123,
    )

    data_low = _build_nvt_simulation_data(run_low, temp_low)
    data_high = _build_nvt_simulation_data(run_high, temp_high)

    quantiles = physical_validation.ensemble.check(
        data_low,
        data_high,
        total_energy=True,
        data_is_uncorrelated=True,
        verbosity=0,
    )

    for i, q in enumerate(quantiles):
        assert abs(q) < 3, (
            f"Ensemble quantile {i} = {q:.2f} sigma exceeds threshold"
        )
