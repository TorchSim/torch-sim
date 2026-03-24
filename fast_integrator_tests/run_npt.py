"""Run NPT simulations for all NPT integrators and save observables.

Usage:
    python run_npt.py                       # temperature sweep (default)
    python run_npt.py --mode pressure       # pressure sweep at fixed T
    python run_npt.py --mode all            # both sweeps
    python run_npt.py --integrator npt_langevin
"""

import argparse
import time

import numpy as np
import torch

import torch_sim as ts
from common import DATA_DIR, DEVICE, DTYPE, make_ar_supercell, make_lj_model, to_dt, to_kT
from torch_sim.units import MetalUnits

NPT_INTEGRATORS = [
    "npt_langevin",
    "npt_nose_hoover",
    "npt_isotropic_crescale",
    "npt_anisotropic_crescale",
]

TEMPERATURES = [88.0, 100.0]
TIMESTEP_PS = 0.004
EXTERNAL_PRESSURE = 0.0
N_STEPS = 20_000
N_EQUILIBRATION = 3_000

# Pressure sweep: two pressures at a fixed temperature.
# physical_validation compares volume distributions at same T, different P.
PRESSURE_SWEEP_TEMP = 100.0  # K
PRESSURES_EVA3 = [0.0, 0.0001]  # eV/Å³  (0 bar and ~160 bar)


def run_npt(integrator_name, sim_state, model, temperature, external_pressure=0.0,
            seed=42):
    kT = to_kT(temperature)
    dt = torch.tensor(to_dt(TIMESTEP_PS), device=DEVICE, dtype=DTYPE)
    ext_p = torch.tensor(external_pressure, device=DEVICE, dtype=DTYPE)
    natoms = int(sim_state.positions.shape[0])

    sim_state = sim_state.clone()
    sim_state.rng = seed

    # Initialize
    if integrator_name == "npt_langevin":
        # state = ts.npt_langevin_init(sim_state, model, kT=kT, dt=dt)
        state = ts.npt_langevin_init(sim_state, model, kT=kT, dt=dt, b_tau = 1 * dt, alpha= 5 * dt) # Better parameters
    elif integrator_name == "npt_nose_hoover":
        state = ts.npt_nose_hoover_init(sim_state, model, kT=kT, dt=dt, t_tau=10 * dt, b_tau=100 * dt)
    elif integrator_name == "npt_isotropic_crescale":
        state = ts.npt_crescale_init(sim_state, model, kT=kT, dt=dt, tau_p = 10 * dt, isothermal_compressibility = 1e-6 / MetalUnits.pressure)
    elif integrator_name == "npt_anisotropic_crescale":
        state = ts.npt_crescale_init(sim_state, model, kT=kT, dt=dt/2, tau_p = 100 * dt, isothermal_compressibility = 1e-6 / MetalUnits.pressure)
    else:
        raise ValueError(f"Unknown: {integrator_name}")

    def step(s):
        if integrator_name == "npt_langevin":
            return ts.npt_langevin_step(s, model, dt=dt, kT=kT, external_pressure=ext_p)
        if integrator_name == "npt_nose_hoover":
            return ts.npt_nose_hoover_step(s, model, dt=dt, kT=kT, external_pressure=ext_p)
        if integrator_name == "npt_isotropic_crescale":
            return ts.npt_crescale_isotropic_step(
                s, model, dt=dt, kT=kT, external_pressure=ext_p, tau = 5 * dt
            )
        return ts.npt_crescale_anisotropic_step(
            s, model, dt=dt/2, kT=kT, external_pressure=ext_p, tau = 5 * dt
        )

    # Equilibration
    print(f"    Equilibrating {N_EQUILIBRATION} steps...")
    for _ in range(N_EQUILIBRATION):
        state = step(state)

    # Production
    print(f"    Producing {N_STEPS} steps...")
    ke_list, pe_list, total_e_list = [], [], []
    temp_list, volume_list = [], []

    for i in range(N_STEPS):
        state = step(state)
        ke = float(ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta))
        pe = float(state.energy.sum())
        temp = float(
            ts.calc_temperature(masses=state.masses, momenta=state.momenta)
        )
        cell = state.cell[0].detach().cpu().numpy()
        vol = float(np.abs(np.linalg.det(cell)))

        ke_list.append(ke)
        pe_list.append(pe)
        total_e_list.append(ke + pe)
        temp_list.append(temp)
        volume_list.append(vol)

    return {
        "kinetic_energy": np.array(ke_list),
        "potential_energy": np.array(pe_list),
        "total_energy": np.array(total_e_list),
        "temperature": np.array(temp_list),
        "volumes": np.array(volume_list),
        "masses": sim_state.masses.detach().cpu().numpy(),
        "dt_internal": to_dt(TIMESTEP_PS),
        "natoms": natoms,
        "target_temperature": temperature,
        "external_pressure": external_pressure,
        "timestep_ps": TIMESTEP_PS,
        "integrator": integrator_name,
    }


def pressure_to_bar(p_eva3):
    """Convert eV/Å³ to bar for display."""
    return p_eva3 / float(MetalUnits.pressure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrator", choices=NPT_INTEGRATORS, default=None)
    parser.add_argument(
        "--mode", choices=["temperature", "pressure", "all"], default="all",
        help="'temperature' = vary T at P=0 (default), "
             "'pressure' = vary P at fixed T, 'all' = both",
    )
    args = parser.parse_args()

    integrators = [args.integrator] if args.integrator else NPT_INTEGRATORS
    DATA_DIR.mkdir(exist_ok=True)

    sim_state = make_ar_supercell(repeat=(3, 3, 3))
    model = make_lj_model(compute_stress=True)

    # --- Temperature sweep (same P=0, vary T) ---
    if args.mode in ("temperature", "all"):
        print("=== Temperature sweep ===")
        for name in integrators:
            for temp in TEMPERATURES:
                seed = 42 if temp == TEMPERATURES[0] else 123
                label = f"{name}_T{temp:.0f}K"
                print(f"  Running {label} ...")
                t0 = time.time()
                data = run_npt(name, sim_state, model, temp, seed=seed)
                elapsed = time.time() - t0
                outpath = DATA_DIR / f"{label}.npz"
                np.savez(outpath, **data)
                print(f"    Saved {outpath.name}  ({elapsed:.1f}s)")

    # --- Pressure sweep (same T, vary P) ---
    if args.mode in ("pressure", "all"):
        print(f"\n=== Pressure sweep at T={PRESSURE_SWEEP_TEMP:.0f}K ===")
        for name in integrators:
            for i, p_eva3 in enumerate(PRESSURES_EVA3):
                p_bar = pressure_to_bar(p_eva3)
                seed = 42 if i == 0 else 123
                label = f"{name}_T{PRESSURE_SWEEP_TEMP:.0f}K_P{p_bar:.0f}bar"
                print(f"  Running {label} ...")
                t0 = time.time()
                data = run_npt(
                    name, sim_state, model, PRESSURE_SWEEP_TEMP,
                    external_pressure=p_eva3, seed=seed,
                )
                elapsed = time.time() - t0
                outpath = DATA_DIR / f"{label}.npz"
                np.savez(outpath, **data)
                print(f"    Saved {outpath.name}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
