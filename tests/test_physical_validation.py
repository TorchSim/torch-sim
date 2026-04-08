"""Physical validation tests for torch-sim MD integrators.

Uses the physical_validation library (https://github.com/shirtsgroup/physical_validation)
to verify that integrators produce physically correct results. These tests require CUDA
and are long-running. Run with:

    pytest -m physical_validation -v

Options:
    --validation-plots          Save plots to tests/physical_validation_data/plots/
    --clean-validation-data     Delete saved validation data before running

Run a specific integrator:

    pytest -m physical_validation -v -k "nvt_langevin"

Tested integrators:

    NVT:
        - nvt_langevin
        - nvt_nose_hoover
        - nvt_vrescale

    NPT:
        - npt_langevin_anisotropic (independent per-dimension strain,
          like LAMMPS couple=none)
        - npt_langevin_isotropic (isotropic logarithmic strain)
        - npt_nose_hoover_isotropic
        - npt_crescale_isotropic
        - npt_crescale_triclinic

Clean up saved data programmatically:

    from tests.test_physical_validation import clean_validation_data
    clean_validation_data()
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk
from numpy.typing import NDArray

import torch_sim as ts
from torch_sim.integrators.npt import npt_crescale_triclinic_average_step
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import MetalUnits


physical_validation = pytest.importorskip("physical_validation")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

# ---------------------------------------------------------------------------
# Device & dtype — CUDA required
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda")
DTYPE = torch.float64

# ---------------------------------------------------------------------------
# LJ Argon parameters
# ---------------------------------------------------------------------------
SIGMA = 3.405
EPSILON = 0.0104
CUTOFF = 2.5 * SIGMA

# ---------------------------------------------------------------------------
# Simulation parameters (matched to fast_integrator_tests_batch)
# ---------------------------------------------------------------------------
TIMESTEP_PS = 0.005
N_STEPS_NVT = 10_000
N_STEPS_NPT = 10_000
N_EQUILIBRATION_NVT = 4_000
N_EQUILIBRATION_NPT = 5_000
LOG_EVERY = 5

# Ensemble check temperatures and pressures (matched to fast_integrator_tests_batch)
TEMPERATURES = [58.3, 60.0]
EXTERNAL_PRESSURE = 0.0
PRESSURE_SWEEP_TEMP = 60.0
PRESSURE_SWEEP_BAR = 90.0
PRESSURE_SWEEP_EVA3 = PRESSURE_SWEEP_BAR * float(MetalUnits.pressure)

# Physical validation thresholds (in sigma units)
KE_SIGMA_WARNING = 2.0
KE_SIGMA_THRESHOLD = 3.0
ENSEMBLE_SIGMA_WARNING = 2.0
ENSEMBLE_SIGMA_THRESHOLD = 3.0

# Data & plot directories
DATA_DIR = Path(__file__).parent / "physical_validation_data"
PLOTS_DIR = DATA_DIR / "plots"

RunData = dict[str, NDArray[np.floating] | float | int | str]

torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# Cleanup utility
# ---------------------------------------------------------------------------
def clean_validation_data() -> None:
    """Delete all saved physical validation data and plots."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print(f"Removed {DATA_DIR}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_kt(temperature_K: float) -> float:
    return temperature_K * float(MetalUnits.temperature)


def _to_dt(timestep_ps: float) -> float:
    return timestep_ps * float(MetalUnits.time)


def _save_run_data(data: RunData, label: str) -> Path:
    """Save run data to a .npz file and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{label}.npz"
    np.savez(path, **data)
    return path


def _get_plot_path(request: pytest.FixtureRequest, name: str) -> str | None:
    """Return plot file path if --validation-plots is enabled, else None."""
    if not request.config.getoption("--validation-plots", default=False):
        return None
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(PLOTS_DIR / f"{name}.png")


def _pressure_to_bar(p_eva3: float) -> float:
    """Convert eV/Ang^3 to bar."""
    return p_eva3 / float(MetalUnits.pressure)


# ---------------------------------------------------------------------------
# Helpers: unit data, model, structure
# ---------------------------------------------------------------------------
def _make_unit_data() -> physical_validation.data.UnitData:
    """Create UnitData for torch-sim's MetalUnits system."""
    return physical_validation.data.UnitData(
        kb=float(MetalUnits.temperature),  # k_B in eV/K = 8.617e-5
        energy_str="eV",
        energy_conversion=96.485,  # Convert to kJ/mol
        length_str="Ang",
        length_conversion=1e-1,  # Convert to nm
        volume_str="Ang^3",
        volume_conversion=1e-3,  # Convert to nm^3
        temperature_str="K",
        temperature_conversion=1.0,
        pressure_str="bar",
        pressure_conversion=1.0,
    )


def _make_lj_model(*, compute_stress: bool = False) -> LennardJonesModel:
    """Create a Lennard-Jones model for Argon."""
    return LennardJonesModel(
        sigma=SIGMA,
        epsilon=EPSILON,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=compute_stress,
        cutoff=CUTOFF,
    )


def _make_ar_supercell(
    repeat: tuple[int, int, int] = (8, 8, 8),
) -> ts.SimState:
    """Create an FCC Argon supercell SimState."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(repeat)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


# ---------------------------------------------------------------------------
# Generic NVT runner
# ---------------------------------------------------------------------------
def _run_nvt(
    integrator_name: str,
    sim_state: ts.SimState,
    model: LennardJonesModel,
    temperature: float,
    timestep_ps: float = TIMESTEP_PS,
    n_steps: int = N_STEPS_NVT,
    n_equilibration: int = N_EQUILIBRATION_NVT,
    log_every: int = LOG_EVERY,
    seed: int = 42,
) -> RunData:
    """Run an NVT simulation with the specified integrator."""
    kT = _to_kt(temperature)
    dt = _to_dt(timestep_ps)
    natoms = int(sim_state.positions.shape[0])

    sim_state = sim_state.clone()
    sim_state.rng = seed

    # Initialize (params matched to fast_integrator_tests_batch)
    if integrator_name == "nvt_langevin":
        state = ts.nvt_langevin_init(sim_state, model, kT=kT)
    elif integrator_name == "nvt_nose_hoover":
        state = ts.nvt_nose_hoover_init(sim_state, model, kT=kT, dt=dt, tau=10 * dt)
    elif integrator_name == "nvt_vrescale":
        state = ts.nvt_vrescale_init(sim_state, model, kT=kT)
    else:
        msg = f"Unknown NVT integrator: {integrator_name}"
        raise ValueError(msg)

    def _step(s: ts.SimState) -> ts.SimState:
        if integrator_name == "nvt_langevin":
            return ts.nvt_langevin_step(s, model, dt=dt, kT=kT, gamma=1 / (50 * dt))
        if integrator_name == "nvt_nose_hoover":
            return ts.nvt_nose_hoover_step(s, model, dt=dt, kT=kT)
        return ts.nvt_vrescale_step(model, s, dt=dt, kT=kT, tau=10 * dt)

    # Equilibration
    for _ in range(n_equilibration):
        state = _step(state)

    # Production (subsampled every log_every steps)
    ke_list, pe_list, total_e_list = [], [], []

    for i in range(n_steps):
        state = _step(state)
        if (i + 1) % log_every == 0:
            ke = float(ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta))
            pe = float(state.energy.sum())
            ke_list.append(ke)
            pe_list.append(pe)
            total_e_list.append(ke + pe)

    cell = sim_state.cell[0].detach().cpu().numpy()
    volume = float(np.abs(np.linalg.det(cell)))

    return {
        "kinetic_energy": np.array(ke_list),
        "potential_energy": np.array(pe_list),
        "total_energy": np.array(total_e_list),
        "volume": volume,
        "masses": sim_state.masses.detach().cpu().numpy(),
        "dt_internal": dt,
        "natoms": natoms,
        "target_temperature": temperature,
        "timestep_ps": timestep_ps,
        "integrator": integrator_name,
    }


# ---------------------------------------------------------------------------
# Generic NPT runner
# ---------------------------------------------------------------------------
def _run_npt(  # noqa: C901
    integrator_name: str,
    sim_state: ts.SimState,
    model: LennardJonesModel,
    temperature: float,
    external_pressure: float = 0.0,
    timestep_ps: float = TIMESTEP_PS,
    n_steps: int = N_STEPS_NPT,
    n_equilibration: int = N_EQUILIBRATION_NPT,
    log_every: int = LOG_EVERY,
    seed: int = 42,
) -> RunData:
    """Run an NPT simulation with the specified integrator."""
    kT = _to_kt(temperature)
    dt = torch.tensor(_to_dt(timestep_ps), device=DEVICE, dtype=DTYPE)
    ext_p = torch.tensor(external_pressure, device=DEVICE, dtype=DTYPE)
    natoms = int(sim_state.positions.shape[0])

    sim_state = sim_state.clone()
    sim_state.rng = seed

    # Initialize (params matched to fast_integrator_tests_batch)
    if integrator_name == "npt_langevin_anisotropic":
        state = ts.npt_langevin_anisotropic_init(
            sim_state,
            model,
            kT=kT,
            dt=dt,
            alpha=1 / (5 * dt),
            cell_alpha=1 / (30 * dt),
            b_tau=300 * dt,
        )
    elif integrator_name == "npt_langevin_isotropic":
        state = ts.npt_langevin_isotropic_init(
            sim_state,
            model,
            kT=kT,
            dt=dt,
            alpha=1 / (5 * dt),
            cell_alpha=1 / (30 * dt),
            b_tau=300 * dt,
        )
    elif integrator_name == "npt_nose_hoover_isotropic":
        state = ts.npt_nose_hoover_isotropic_init(
            sim_state,
            model,
            kT=kT,
            dt=dt,
            t_tau=10 * dt,
            b_tau=100 * dt,
        )
    elif integrator_name in (
        "npt_crescale_isotropic",
        "npt_crescale_triclinic",
    ):
        state = ts.npt_crescale_init(
            sim_state,
            model,
            kT=kT,
            dt=dt,
            tau_p=3 * dt,
            isothermal_compressibility=1e-6 / MetalUnits.pressure,
        )
    else:
        msg = f"Unknown NPT integrator: {integrator_name}"
        raise ValueError(msg)

    def _step(s: ts.SimState) -> ts.SimState:
        if integrator_name == "npt_langevin_anisotropic":
            return ts.npt_langevin_anisotropic_step(
                s,
                model,
                dt=dt,
                kT=kT,
                external_pressure=ext_p,
            )
        if integrator_name == "npt_langevin_isotropic":
            return ts.npt_langevin_isotropic_step(
                s,
                model,
                dt=dt,
                kT=kT,
                external_pressure=ext_p,
            )
        if integrator_name == "npt_nose_hoover_isotropic":
            return ts.npt_nose_hoover_isotropic_step(
                s,
                model,
                dt=dt,
                kT=kT,
                external_pressure=ext_p,
            )
        if integrator_name == "npt_crescale_triclinic":
            return npt_crescale_triclinic_average_step(
                s,
                model,
                dt=dt,
                kT=kT,
                external_pressure=ext_p,
                tau=1 * dt,
            )
        return ts.npt_crescale_isotropic_step(
            s,
            model,
            dt=dt,
            kT=kT,
            external_pressure=ext_p,
            tau=1 * dt,
        )

    # Equilibration
    for _ in range(n_equilibration):
        state = _step(state)

    # Production (subsampled every log_every steps)
    ke_list, pe_list, total_e_list = [], [], []
    volume_list = []

    for i in range(n_steps):
        state = _step(state)
        if (i + 1) % log_every == 0:
            ke = float(ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta))
            pe = float(state.energy.sum())
            cell = state.cell[0].detach().cpu().numpy()
            vol = float(np.abs(np.linalg.det(cell)))
            ke_list.append(ke)
            pe_list.append(pe)
            total_e_list.append(ke + pe)
            volume_list.append(vol)

    return {
        "kinetic_energy": np.array(ke_list),
        "potential_energy": np.array(pe_list),
        "total_energy": np.array(total_e_list),
        "volumes": np.array(volume_list),
        "masses": sim_state.masses.detach().cpu().numpy(),
        "dt_internal": float(dt),
        "natoms": natoms,
        "target_temperature": temperature,
        "external_pressure": external_pressure,
        "timestep_ps": timestep_ps,
        "integrator": integrator_name,
    }


# ---------------------------------------------------------------------------
# SimulationData builders
# ---------------------------------------------------------------------------
def _build_nvt_simulation_data(
    run_data: RunData,
    temperature: float,
) -> physical_validation.data.SimulationData:
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

    return physical_validation.data.SimulationData(
        units=units,
        dt=run_data["timestep_ps"],
        system=system,
        ensemble=ensemble_data,
        observables=observables,
    )


def _build_npt_simulation_data(
    run_data: RunData,
    temperature: float,
    pressure: float,
) -> physical_validation.data.SimulationData:
    """Build a physical_validation SimulationData from NPT run results."""
    units = _make_unit_data()

    system = physical_validation.data.SystemData(
        natoms=run_data["natoms"],
        nconstraints=0,
        ndof_reduction_tra=3,
        ndof_reduction_rot=0,
        mass=run_data["masses"],
    )

    ensemble_data = physical_validation.data.EnsembleData(
        ensemble="NPT",
        natoms=run_data["natoms"],
        pressure=pressure,
        temperature=temperature,
    )

    observables = physical_validation.data.ObservableData(
        kinetic_energy=run_data["kinetic_energy"],
        potential_energy=run_data["potential_energy"],
        total_energy=run_data["total_energy"],
        volume=run_data["volumes"],
    )

    return physical_validation.data.SimulationData(
        units=units,
        dt=run_data["timestep_ps"],
        system=system,
        ensemble=ensemble_data,
        observables=observables,
    )


# ===========================================================================
# Session fixture: cleanup saved data
# ===========================================================================
@pytest.fixture(autouse=True, scope="session")
def _manage_validation_data(request: pytest.FixtureRequest) -> None:
    """Clean data directory if --clean-validation-data is set."""
    if request.config.getoption("--clean-validation-data", default=False):
        clean_validation_data()


# ===========================================================================
# Tests: KE distribution (Maxwell-Boltzmann)
# ===========================================================================
@pytest.mark.physical_validation
@pytest.mark.parametrize(
    "integrator_name",
    ["nvt_langevin", "nvt_nose_hoover", "nvt_vrescale"],
)
def test_nvt_ke_distribution(
    integrator_name: str, request: pytest.FixtureRequest
) -> None:
    """Test that KE follows the Maxwell-Boltzmann distribution for NVT."""
    sim_state = _make_ar_supercell(repeat=(8, 8, 8))
    model = _make_lj_model()
    temperature = TEMPERATURES[1]

    run_data = _run_nvt(
        integrator_name,
        sim_state,
        model,
        temperature=temperature,
        seed=42,
    )
    _save_run_data(run_data, f"{integrator_name}_T{temperature:.1f}K_ke")

    data = _build_nvt_simulation_data(run_data, temperature)
    plot_path = _get_plot_path(request, f"{integrator_name}_nvt_ke")

    kwargs = {}
    if plot_path:
        kwargs["filename"] = plot_path
    d_mean, d_width = physical_validation.kinetic_energy.distribution(
        data,
        strict=False,
        verbosity=0,
        **kwargs,
    )

    if abs(d_mean) > KE_SIGMA_WARNING:
        warnings.warn(
            f"[{integrator_name}] KE mean deviation {d_mean:.2f} sigma exceeds "
            f"{KE_SIGMA_WARNING} sigma warning threshold",
            stacklevel=1,
        )
    if abs(d_width) > KE_SIGMA_WARNING:
        warnings.warn(
            f"[{integrator_name}] KE width deviation {d_width:.2f} sigma exceeds "
            f"{KE_SIGMA_WARNING} sigma warning threshold",
            stacklevel=1,
        )
    assert abs(d_mean) < KE_SIGMA_THRESHOLD, (
        f"[{integrator_name}] KE mean deviation {d_mean:.2f} sigma"
    )
    assert abs(d_width) < KE_SIGMA_THRESHOLD, (
        f"[{integrator_name}] KE width deviation {d_width:.2f} sigma"
    )


@pytest.mark.physical_validation
@pytest.mark.parametrize(
    "integrator_name",
    [
        "npt_langevin_anisotropic",
        "npt_langevin_isotropic",
        "npt_nose_hoover_isotropic",
        "npt_crescale_isotropic",
        "npt_crescale_triclinic",
    ],
)
def test_npt_ke_distribution(
    integrator_name: str, request: pytest.FixtureRequest
) -> None:
    """Test that KE follows the Maxwell-Boltzmann distribution for NPT."""
    sim_state = _make_ar_supercell(repeat=(8, 8, 8))
    model = _make_lj_model(compute_stress=True)
    temperature = TEMPERATURES[1]

    run_data = _run_npt(
        integrator_name,
        sim_state,
        model,
        temperature=temperature,
        external_pressure=EXTERNAL_PRESSURE,
        seed=42,
    )
    _save_run_data(run_data, f"{integrator_name}_T{temperature:.1f}K_ke")

    # Use NVT builder with mean volume for KE distribution check
    run_data_nvt = {**run_data, "volume": float(np.mean(run_data["volumes"]))}
    data = _build_nvt_simulation_data(run_data_nvt, temperature)
    plot_path = _get_plot_path(request, f"{integrator_name}_npt_ke")

    kwargs = {}
    if plot_path:
        kwargs["filename"] = plot_path
    d_mean, d_width = physical_validation.kinetic_energy.distribution(
        data,
        strict=False,
        verbosity=0,
        **kwargs,
    )

    if abs(d_mean) > KE_SIGMA_WARNING:
        warnings.warn(
            f"[{integrator_name}] KE mean deviation {d_mean:.2f} sigma exceeds "
            f"{KE_SIGMA_WARNING} sigma warning threshold",
            stacklevel=1,
        )
    if abs(d_width) > KE_SIGMA_WARNING:
        warnings.warn(
            f"[{integrator_name}] KE width deviation {d_width:.2f} sigma exceeds "
            f"{KE_SIGMA_WARNING} sigma warning threshold",
            stacklevel=1,
        )
    assert abs(d_mean) < KE_SIGMA_THRESHOLD, (
        f"[{integrator_name}] KE mean deviation {d_mean:.2f} sigma"
    )
    assert abs(d_width) < KE_SIGMA_THRESHOLD, (
        f"[{integrator_name}] KE width deviation {d_width:.2f} sigma"
    )


# ===========================================================================
# Tests: ensemble validity (Boltzmann weight ratio at two temperatures)
# ===========================================================================
@pytest.mark.physical_validation
@pytest.mark.parametrize(
    "integrator_name",
    ["nvt_langevin", "nvt_nose_hoover", "nvt_vrescale"],
)
def test_nvt_ensemble_check(integrator_name: str, request: pytest.FixtureRequest) -> None:
    """Test NVT ensemble validity at two temperatures."""
    sim_state = _make_ar_supercell(repeat=(8, 8, 8))
    model = _make_lj_model()

    temp_low, temp_high = TEMPERATURES

    run_low = _run_nvt(
        integrator_name,
        sim_state,
        model,
        temperature=temp_low,
        seed=42,
    )
    run_high = _run_nvt(
        integrator_name,
        sim_state,
        model,
        temperature=temp_high,
        seed=123,
    )
    _save_run_data(run_low, f"{integrator_name}_T{temp_low:.1f}K_ens")
    _save_run_data(run_high, f"{integrator_name}_T{temp_high:.1f}K_ens")

    data_low = _build_nvt_simulation_data(run_low, temp_low)
    data_high = _build_nvt_simulation_data(run_high, temp_high)
    plot_path = _get_plot_path(request, f"{integrator_name}_nvt_ens")

    kwargs = {}
    if plot_path:
        kwargs["filename"] = plot_path
    quantiles = physical_validation.ensemble.check(
        data_low,
        data_high,
        total_energy=True,
        data_is_uncorrelated=True,
        verbosity=0,
        **kwargs,
    )

    for i, q in enumerate(quantiles):
        if abs(q) > ENSEMBLE_SIGMA_WARNING:
            warnings.warn(
                f"[{integrator_name}] Ensemble quantile {i} = {q:.2f} sigma exceeds "
                f"{ENSEMBLE_SIGMA_WARNING} sigma warning threshold",
                stacklevel=1,
            )
        assert abs(q) < ENSEMBLE_SIGMA_THRESHOLD, (
            f"[{integrator_name}] Ensemble quantile {i} = {q:.2f} sigma"
        )


@pytest.mark.physical_validation
@pytest.mark.parametrize(
    "integrator_name",
    [
        "npt_langevin_anisotropic",
        "npt_langevin_isotropic",
        "npt_nose_hoover_isotropic",
        "npt_crescale_isotropic",
        "npt_crescale_triclinic",
    ],
)
def test_npt_ensemble_check(integrator_name: str, request: pytest.FixtureRequest) -> None:
    """Test NPT ensemble validity at two temperatures.

    Uses temperatures both in the solid phase (below LJ Ar melting point ~84K)
    to avoid the solid-liquid phase transition which causes non-overlapping
    energy distributions in the NPT ensemble.
    """
    sim_state = _make_ar_supercell(repeat=(8, 8, 8))
    model = _make_lj_model(compute_stress=True)

    temp_low, temp_high = TEMPERATURES

    run_low = _run_npt(
        integrator_name,
        sim_state,
        model,
        temperature=temp_low,
        external_pressure=EXTERNAL_PRESSURE,
        seed=42,
    )
    run_high = _run_npt(
        integrator_name,
        sim_state,
        model,
        temperature=temp_high,
        external_pressure=EXTERNAL_PRESSURE,
        seed=123,
    )
    _save_run_data(run_low, f"{integrator_name}_T{temp_low:.1f}K_ens")
    _save_run_data(run_high, f"{integrator_name}_T{temp_high:.1f}K_ens")

    data_low = _build_npt_simulation_data(run_low, temp_low, EXTERNAL_PRESSURE)
    data_high = _build_npt_simulation_data(run_high, temp_high, EXTERNAL_PRESSURE)
    plot_path = _get_plot_path(request, f"{integrator_name}_npt_ens_temp")

    kwargs = {}
    if plot_path:
        kwargs["filename"] = plot_path
    quantiles = physical_validation.ensemble.check(
        data_low,
        data_high,
        total_energy=True,
        data_is_uncorrelated=True,
        verbosity=0,
        **kwargs,
    )

    for i, q in enumerate(quantiles):
        if abs(q) > ENSEMBLE_SIGMA_WARNING:
            warnings.warn(
                f"[{integrator_name}] Ensemble quantile {i} = {q:.2f} sigma exceeds "
                f"{ENSEMBLE_SIGMA_WARNING} sigma warning threshold",
                stacklevel=1,
            )
        assert abs(q) < ENSEMBLE_SIGMA_THRESHOLD, (
            f"[{integrator_name}] Ensemble quantile {i} = {q:.2f} sigma"
        )


# ===========================================================================
# Tests: ensemble validity (Boltzmann weight ratio at two pressures)
# ===========================================================================
@pytest.mark.physical_validation
@pytest.mark.parametrize(
    "integrator_name",
    [
        "npt_langevin_anisotropic",
        "npt_langevin_isotropic",
        "npt_nose_hoover_isotropic",
        "npt_crescale_isotropic",
        "npt_crescale_triclinic",
    ],
)
def test_npt_pressure_ensemble_check(
    integrator_name: str, request: pytest.FixtureRequest
) -> None:
    """Test NPT ensemble validity at two pressures (fixed temperature)."""
    sim_state = _make_ar_supercell(repeat=(8, 8, 8))
    model = _make_lj_model(compute_stress=True)

    p_low = EXTERNAL_PRESSURE
    p_high = PRESSURE_SWEEP_EVA3

    run_low = _run_npt(
        integrator_name,
        sim_state,
        model,
        temperature=PRESSURE_SWEEP_TEMP,
        external_pressure=p_low,
        seed=42,
    )
    run_high = _run_npt(
        integrator_name,
        sim_state,
        model,
        temperature=PRESSURE_SWEEP_TEMP,
        external_pressure=p_high,
        seed=123,
    )
    _save_run_data(
        run_low,
        f"{integrator_name}_T{PRESSURE_SWEEP_TEMP:.1f}K_P0bar",
    )
    _save_run_data(
        run_high,
        f"{integrator_name}_T{PRESSURE_SWEEP_TEMP:.1f}K_P{PRESSURE_SWEEP_BAR:.0f}bar",
    )

    data_low = _build_npt_simulation_data(run_low, PRESSURE_SWEEP_TEMP, p_low)
    data_high = _build_npt_simulation_data(run_high, PRESSURE_SWEEP_TEMP, p_high)
    plot_path = _get_plot_path(request, f"{integrator_name}_npt_ens_press")

    kwargs = {}
    if plot_path:
        kwargs["filename"] = plot_path
    quantiles = physical_validation.ensemble.check(
        data_low,
        data_high,
        total_energy=True,
        data_is_uncorrelated=True,
        verbosity=0,
        **kwargs,
    )

    for i, q in enumerate(quantiles):
        if abs(q) > ENSEMBLE_SIGMA_WARNING:
            warnings.warn(
                f"[{integrator_name}] Pressure ensemble quantile {i} = {q:.2f} sigma "
                f"exceeds {ENSEMBLE_SIGMA_WARNING} sigma warning threshold",
                stacklevel=1,
            )
        assert abs(q) < ENSEMBLE_SIGMA_THRESHOLD, (
            f"[{integrator_name}] Pressure ensemble quantile {i} = {q:.2f} sigma"
        )
