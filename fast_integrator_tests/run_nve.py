"""Run NVE simulations at multiple timesteps for convergence analysis.

Usage:
    python run_nve.py
"""

import time

import numpy as np
import torch

import torch_sim as ts
from common import DATA_DIR, make_ar_supercell, make_lj_model, to_dt, to_kT

# Timesteps: sweep a range so the notebook can pick the best subset
TIMESTEPS_PS = [0.010, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002]
TEMPERATURE = 5.0  # K - low so integration error dominates
N_STEPS = 10_000
SEED = 42


def run_nve(sim_state, model, kT_init, timestep_ps):
    dt = to_dt(timestep_ps)

    sim_state = sim_state.clone()
    sim_state.rng = SEED
    state = ts.nve_init(sim_state, model, kT=kT_init)

    ke_list, pe_list, com_list = [], [], []

    for _ in range(N_STEPS):
        state = ts.nve_step(state, model, dt=dt)
        ke = float(ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta))
        pe = float(state.energy.sum())
        ke_list.append(ke)
        pe_list.append(pe)
        com_list.append(ke + pe)

    return {
        "kinetic_energy": np.array(ke_list),
        "potential_energy": np.array(pe_list),
        "constant_of_motion": np.array(com_list),
        "dt_internal": dt,
        "timestep_ps": timestep_ps,
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # sim_state = make_ar_supercell(repeat=(1, 1, 1))
    sim_state = make_ar_supercell(repeat=(6, 6, 6))
    model = make_lj_model()
    kT_init = to_kT(TEMPERATURE)
    natoms = int(sim_state.positions.shape[0])
    masses = sim_state.masses.detach().cpu().numpy()
    cell = sim_state.cell[0].detach().cpu().numpy()
    volume = float(np.abs(np.linalg.det(cell)))

    all_results = {}
    for dt_ps in TIMESTEPS_PS:
        label = f"nve_dt{dt_ps:.4f}"
        print(f"  Running {label} ...")
        t0 = time.time()
        data = run_nve(sim_state, model, kT_init, dt_ps)
        elapsed = time.time() - t0
        all_results[f"dt_{dt_ps}"] = data
        print(f"    std(E_tot) = {data['constant_of_motion'].std():.3e}  ({elapsed:.1f}s)")

    # Save everything into one npz
    save_dict = {
        "timesteps_ps": np.array(TIMESTEPS_PS),
        "temperature": TEMPERATURE,
        "natoms": natoms,
        "masses": masses,
        "volume": volume,
        "n_steps": N_STEPS,
    }
    for dt_ps in TIMESTEPS_PS:
        key = f"dt_{dt_ps}"
        for field in ("constant_of_motion", "kinetic_energy", "potential_energy"):
            save_dict[f"{key}_{field}"] = all_results[key][field]
        save_dict[f"{key}_dt_internal"] = all_results[key]["dt_internal"]

    outpath = DATA_DIR / "nve_convergence.npz"
    np.savez(outpath, **save_dict)
    print(f"\n  Saved {outpath.name}")


if __name__ == "__main__":
    main()
