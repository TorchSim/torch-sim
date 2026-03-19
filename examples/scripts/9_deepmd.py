"""DeepMD DPA-3 inference on Si/Cu, batching, NVE dynamics, and ASE comparison."""

# /// script
# dependencies = ["deepmd-kit[torch]>=3"]
# ///

import os

import numpy as np
import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.deepmd import DeepMDModel
from torch_sim.units import MetalUnits as Units


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

SMOKE_TEST = os.getenv("CI") is not None
N_steps = 50 if SMOKE_TEST else 500

torch.manual_seed(42)

MODEL_PATH = "dpa-3.1-3m-ft.pth"

deepmd_model = DeepMDModel(
    model=MODEL_PATH,
    device=device,
    dtype=dtype,
    compute_forces=True,
    compute_stress=True,
)

n_types = len(deepmd_model.type_map)
print(f"type_map: {deepmd_model.type_map[:10]}... ({n_types} elements)")
print(f"cutoff: {deepmd_model.rcut:.1f} A")

si_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
state_si = ts.io.atoms_to_state(si_atoms, device=device, dtype=dtype)
results_si = deepmd_model(state_si)

print(f"Si energy: {results_si['energy'].item():.6f} eV  ({state_si.n_atoms} atoms)")

cu_atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
state_cu = ts.io.atoms_to_state(cu_atoms, device=device, dtype=dtype)
results_cu = deepmd_model(state_cu)

print(f"Cu energy: {results_cu['energy'].item():.6f} eV  ({state_cu.n_atoms} atoms)")

batched_state = ts.io.atoms_to_state([si_atoms, cu_atoms], device=device, dtype=dtype)
batched_results = deepmd_model(batched_state)

energy_diff_si = torch.abs(batched_results["energy"][0] - results_si["energy"][0])
energy_diff_cu = torch.abs(batched_results["energy"][1] - results_cu["energy"][0])

print(f"Batch energy: {batched_results['energy'].tolist()}")
print(f"Si energy difference (batch vs single): {energy_diff_si:.2e}")
print(f"Cu energy difference (batch vs single): {energy_diff_cu:.2e}")

assert energy_diff_si < 1e-5  # noqa: S101
assert energy_diff_cu < 1e-5  # noqa: S101

same_state = ts.io.atoms_to_state([si_atoms, si_atoms], device=device, dtype=dtype)
same_results = deepmd_model(same_state)

energy_match = torch.abs(same_results["energy"][0] - same_results["energy"][1])
print(f"Same-size batch energies: {same_results['energy'].tolist()}")
print(f"Energy difference between identical systems: {energy_match:.2e}")

assert energy_match < 1e-5  # noqa: S101

md_state = ts.io.atoms_to_state(si_atoms, device=device, dtype=dtype)
kT = torch.tensor(300 * Units.temperature, device=device, dtype=dtype)
dt = torch.tensor(0.001 * Units.time, device=device, dtype=dtype)

md_state.rng = 42
md_state = ts.nve_init(state=md_state, model=deepmd_model, kT=kT)

initial_total_energy = md_state.energy + ts.calc_kinetic_energy(
    masses=md_state.masses, momenta=md_state.momenta, system_idx=md_state.system_idx
)

for step in range(N_steps):
    md_state = ts.nve_step(state=md_state, model=deepmd_model, dt=dt)

    if step % (N_steps // 5) == 0 or step == N_steps - 1:
        ke = ts.calc_kinetic_energy(
            masses=md_state.masses,
            momenta=md_state.momenta,
            system_idx=md_state.system_idx,
        )
        total_e = md_state.energy + ke
        print(
            f"Step {step:4d}: PE={md_state.energy.item():10.4f}  "
            f"KE={ke.item():8.4f}  Total={total_e.item():10.4f} eV"
        )

final_total_energy = md_state.energy + ts.calc_kinetic_energy(
    masses=md_state.masses, momenta=md_state.momenta, system_idx=md_state.system_idx
)
energy_drift = torch.abs(final_total_energy - initial_total_energy)
print(f"Energy drift: {energy_drift.item():.2e} eV")

from deepmd.calculator import DP  # noqa: E402


ase_calc = DP(model=MODEL_PATH)

ts_precomputed = {"Si": results_si, "Cu": results_cu}

for name, atoms in [("Si", si_atoms.copy()), ("Cu", cu_atoms.copy())]:
    atoms.calc = ase_calc
    ase_energy = atoms.get_potential_energy()
    ase_forces = atoms.get_forces()
    ase_stress = atoms.get_stress()

    ts_res = ts_precomputed[name]
    ts_energy = ts_res["energy"].item()
    ts_forces = ts_res["forces"].cpu().numpy()
    ts_stress_voigt = ts.elastic.full_3x3_to_voigt_6_stress(
        ts_res["stress"]
    ).squeeze(0).cpu().numpy()

    e_diff = abs(ase_energy - ts_energy)
    f_diff = np.max(np.abs(ase_forces - ts_forces))
    s_diff = np.max(np.abs(ase_stress - ts_stress_voigt))

    print(
        f"{name}: energy difference={e_diff:.2e}  "
        f"max force difference={f_diff:.2e}  max stress difference={s_diff:.2e}"
    )
    assert e_diff < 1e-5, f"{name} energy mismatch"  # noqa: S101
    assert f_diff < 1e-5, f"{name} force mismatch"  # noqa: S101
    assert s_diff < 1e-5, f"{name} stress mismatch"  # noqa: S101
