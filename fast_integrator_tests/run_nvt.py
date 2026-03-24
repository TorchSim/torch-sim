"""Run NVT simulations for all NVT integrators and save observables.

Usage:
    python run_nvt.py
    python run_nvt.py --integrator nvt_langevin
"""

import argparse
import time

import numpy as np
import torch

import torch_sim as ts
from common import DATA_DIR, make_ar_supercell, make_lj_model, to_dt, to_kT

NVT_INTEGRATORS = ["nvt_langevin", "nvt_nose_hoover", "nvt_vrescale"]

# Two temperatures for ensemble check
TEMPERATURES = [95.0, 100.0]
TIMESTEP_PS = 0.004
N_STEPS = 10_000
N_EQUILIBRATION = 2_000


def run_nvt(integrator_name, sim_state, model, temperature, seed=42):
    kT = to_kT(temperature)
    dt = to_dt(TIMESTEP_PS)
    natoms = int(sim_state.positions.shape[0])

    sim_state = sim_state.clone()
    sim_state.rng = seed

    # Initialize
    if integrator_name == "nvt_langevin":
        state = ts.nvt_langevin_init(sim_state, model, kT=kT)
    elif integrator_name == "nvt_nose_hoover":
        state = ts.nvt_nose_hoover_init(sim_state, model, kT=kT, dt=dt)
    elif integrator_name == "nvt_vrescale":
        state = ts.nvt_vrescale_init(sim_state, model, kT=kT)
    else:
        raise ValueError(f"Unknown: {integrator_name}")

    def step(s):
        if integrator_name == "nvt_langevin":
            return ts.nvt_langevin_step(s, model, dt=dt, kT=kT)
        if integrator_name == "nvt_nose_hoover":
            return ts.nvt_nose_hoover_step(s, model, dt=dt, kT=kT)
        return ts.nvt_vrescale_step(model, s, dt=dt, kT=kT)

    # Equilibration
    print(f"    Equilibrating {N_EQUILIBRATION} steps...")
    for _ in range(N_EQUILIBRATION):
        state = step(state)

    # Production
    print(f"    Producing {N_STEPS} steps...")
    ke_list, pe_list, total_e_list = [], [], []
    temp_list = []

    for i in range(N_STEPS):
        state = step(state)
        ke = float(ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta))
        pe = float(state.energy.sum())
        temp = float(
            ts.calc_temperature(masses=state.masses, momenta=state.momenta)
        )
        ke_list.append(ke)
        pe_list.append(pe)
        total_e_list.append(ke + pe)
        temp_list.append(temp)

    cell = sim_state.cell[0].detach().cpu().numpy()
    volume = float(np.abs(np.linalg.det(cell)))

    return {
        "kinetic_energy": np.array(ke_list),
        "potential_energy": np.array(pe_list),
        "total_energy": np.array(total_e_list),
        "temperature": np.array(temp_list),
        "volume": volume,
        "masses": sim_state.masses.detach().cpu().numpy(),
        "dt_internal": to_dt(TIMESTEP_PS),
        "natoms": natoms,
        "target_temperature": temperature,
        "timestep_ps": TIMESTEP_PS,
        "integrator": integrator_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrator", choices=NVT_INTEGRATORS, default=None)
    args = parser.parse_args()

    integrators = [args.integrator] if args.integrator else NVT_INTEGRATORS
    DATA_DIR.mkdir(exist_ok=True)

    sim_state = make_ar_supercell(repeat=(6, 6, 6))
    model = make_lj_model()

    for name in integrators:
        for temp in TEMPERATURES:
            seed = 42 if temp == TEMPERATURES[0] else 123
            label = f"{name}_T{temp:.0f}K"
            print(f"  Running {label} ...")
            t0 = time.time()
            data = run_nvt(name, sim_state, model, temp, seed=seed)
            elapsed = time.time() - t0
            outpath = DATA_DIR / f"{label}.npz"
            np.savez(outpath, **data)
            print(f"    Saved {outpath.name}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
