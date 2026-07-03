# %% [markdown]
# # TorchSim + SevenNet-D3 MD Check
#
# Run short NVT molecular dynamics with TorchSim + SevenNet or SevenNet-D3, then check NaN values, temperature, energy, and minimum pair distances.

# %%
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import csv
import gc
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.neighborlist import neighbor_list
from torch_sim.integrators import Integrator
from torch_sim.runners import integrate, static
from sevenn.torchsim import SevenNetModel, SevenNetD3Model

print("Imports OK")
print(f"Running TorchSim + SevenNet on {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# ============================================================================
# Parameters
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# Structures
structure_dir = Path("/data/curtis/dev/torch-sim/stru")
structure_pattern = "seed*/NaTaCl6_128_rho2p904686_seed*.extxyz"
# Optional explicit file list. If non-empty, it overrides structure_dir and structure_pattern.
structure_files = []
# structure_files = [
#     "/root/shared-nvme/yz/.../seed11001/NaTaCl6_128_rho2p904686_seed11001.extxyz",
# ]

n_structures = 16

# Model
model_path = "/data/curtis/dev/torch-sim/.venv/lib/python3.12/site-packages/sevenn/pretrained_potentials/SevenNet_omni/checkpoint_sevennet_omni.pth"
modal = "omat24"
dtype = torch.float32

# SevenNet tensor-product accelerator. Enable only one at a time.
accelerator = "flash"  # "flash" | "cueq" | "oeq" | "none"

# D3 correction
use_d3 = False
d3_mode = "auto"          # "auto" | "serial" | "batch"
d3_batch_threshold = 4
d3_damping_type = "damp_bj"
d3_functional_name = "pbe"
d3_vdw_cutoff = 9000.0    # Bohr^2
d3_cn_cutoff = 1600.0     # Bohr^2

# MD settings
temperature = 300.0       # K
timestep_fs = 2.0         # fs
tdamp_fs = 200.0          # fs
precondition_steps = 200
md_steps = 1000
velocity_seed = 12345
use_autobatcher = False

# Output
output_dir = Path("/data/curtis/dev/torch-sim/results/torchsim_sevennet_d3_md_check")
output_dir.mkdir(parents=True, exist_ok=True)
output_json = output_dir / "md_check.json"
output_csv = output_dir / "md_check.csv"

np.random.seed(velocity_seed)
torch.manual_seed(velocity_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(velocity_seed)

print(f"structure_dir = {structure_dir}")
print(f"structure_pattern = {structure_pattern}")
print(f"model_path = {model_path}")
print(f"use_d3 = {use_d3}, d3_mode = {d3_mode}, accelerator = {accelerator}")
print(f"temperature = {temperature} K, timestep = {timestep_fs} fs, md_steps = {md_steps}")

# %%
# ============================================================================
# Helper functions
# ============================================================================

def get_structure_paths():
    if structure_files:
        paths = [Path(x) for x in structure_files]
    else:
        paths = sorted(structure_dir.glob(structure_pattern))
    paths = paths[:n_structures]
    if not paths:
        raise FileNotFoundError(f"No structure files found: {structure_dir}/{structure_pattern}")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing structure files:\n" + "\n".join(missing))
    return paths


def shortest_pair_distance(atoms, cutoff_max=8.0):
    if len(atoms) < 2:
        return None
    cutoff = min(float(np.min(atoms.cell.lengths())) / 2.0, cutoff_max)
    i, d = neighbor_list("id", atoms, cutoff)
    d = d[i < len(atoms)]
    if len(d) == 0:
        return None
    return float(np.min(d))


def accelerator_flags(name):
    if name == "none":
        return dict(enable_cueq=False, enable_flash=False, enable_oeq=False)
    if name == "flash":
        return dict(enable_cueq=False, enable_flash=True, enable_oeq=False)
    if name == "cueq":
        return dict(enable_cueq=True, enable_flash=False, enable_oeq=False)
    if name == "oeq":
        return dict(enable_cueq=False, enable_flash=False, enable_oeq=True)
    raise ValueError(f"Unknown accelerator: {name}")


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def build_model():
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    common = dict(
        model=model_path,
        modal=modal,
        device=device,
        dtype=dtype,
        **accelerator_flags(accelerator),
    )
    if not use_d3:
        return SevenNetModel(**common)
    if device != "cuda":
        raise RuntimeError("use_d3=True requires CUDA")
    return SevenNetD3Model(
        **common,
        d3_mode=d3_mode,
        d3_batch_threshold=d3_batch_threshold,
        damping_type=d3_damping_type,
        functional_name=d3_functional_name,
        vdw_cutoff=d3_vdw_cutoff,
        cn_cutoff=d3_cn_cutoff,
    )


def tensor_to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def all_finite(x):
    arr = tensor_to_numpy(x)
    return arr is not None and bool(np.all(np.isfinite(arr)))


def bad_indices_from_arrays(n_systems, **arrays):
    bad = set()
    reasons = []
    for name, value in arrays.items():
        arr = tensor_to_numpy(value)
        if arr is None:
            continue
        if arr.ndim == 0:
            if not np.isfinite(arr):
                bad.update(range(n_systems))
                reasons.append(f"{name}: scalar is non-finite")
            continue
        if arr.shape[0] != n_systems:
            if not np.all(np.isfinite(arr)):
                bad.update(range(n_systems))
                reasons.append(f"{name}: shape {arr.shape} is not system-indexed and contains non-finite values")
            continue
        arr_by_system = arr.reshape((n_systems, -1))
        for idx, row in enumerate(arr_by_system):
            if not np.all(np.isfinite(row)):
                bad.add(idx)
                reasons.append(f"{name}: system {idx} contains non-finite values")
    return sorted(bad), reasons

# %%
# ============================================================================
# Load structures and initialize model
# ============================================================================

structure_paths = get_structure_paths()
atoms_list = []

print(f"Found {len(structure_paths)} structure files:")
for idx, path in enumerate(structure_paths, start=1):
    atoms = read(path)
    atoms.pbc = True
    atoms_list.append(atoms)
    rho = atoms.get_masses().sum() / units.mol / atoms.get_volume() * 1.0e24
    print(
        f"{idx:02d}. {path.name} | {atoms.get_chemical_formula()} | "
        f"N={len(atoms)} | V={atoms.get_volume():.3f} A^3 | "
        f"rho={rho:.4f} g/cm^3 | min_dist={shortest_pair_distance(atoms):.4f} A"
    )

clear_cuda_cache()
model = build_model()
print(f"\nModel initialized: {model.__class__.__name__}")
if torch.cuda.is_available():
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")

# %%
# ============================================================================
# Single-point forward check
# ============================================================================

try:
    static_outputs = static(
        atoms_list,
        model=model,
        trajectory_reporter=None,
        autobatcher=use_autobatcher,
        pbar=False,
    )
    energy0 = [out.get("energy", out.get("potential_energy")) for out in static_outputs]
    forces0 = [out.get("forces") for out in static_outputs]
    stress0 = [out.get("stress") for out in static_outputs]
    single_point_error = None
except Exception as exc:
    static_outputs = []
    energy0 = []
    forces0 = []
    stress0 = []
    single_point_error = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }

print("Single-point check:")
if single_point_error is not None:
    print(json.dumps(single_point_error, indent=2))
else:
    print(f"  n outputs: {len(static_outputs)}")
    print(f"  energy finite: {[all_finite(x) for x in energy0]}")
    print(f"  forces finite: {[all_finite(x) for x in forces0]}")
    print(f"  stress finite: {[all_finite(x) if x is not None else None for x in stress0]}")
    print(f"  energy: {[tensor_to_numpy(x).tolist() if hasattr(tensor_to_numpy(x), 'tolist') else tensor_to_numpy(x) for x in energy0]}")

# %%
# ============================================================================
# Run short NVT MD and check anomalies
# ============================================================================

integrate_kwargs = {}
if use_autobatcher:
    integrate_kwargs["autobatcher"] = True

init_kwargs = {
    "tau": tdamp_fs / 1000.0,
    "chain_length": 3,
    "chain_steps": 1,
    "sy_steps": 3,
}

timestep_ps = timestep_fs / 1000.0
run_error = None
final_state = None

try:
    clear_cuda_cache()
    system = atoms_list

    if precondition_steps > 0:
        t0 = time.perf_counter()
        system = integrate(
            system,
            model=model,
            integrator=Integrator.nvt_nose_hoover,
            n_steps=precondition_steps,
            temperature=temperature,
            timestep=timestep_ps,
            init_kwargs=init_kwargs,
            trajectory_reporter=None,
            pbar=False,
            **integrate_kwargs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        precondition_wall = time.perf_counter() - t0
    else:
        precondition_wall = 0.0

    t1 = time.perf_counter()
    final_state = integrate(
        system,
        model=model,
        integrator=Integrator.nvt_nose_hoover,
        n_steps=md_steps,
        temperature=temperature,
        timestep=timestep_ps,
        init_kwargs=init_kwargs,
        trajectory_reporter=None,
        pbar=False,
        **integrate_kwargs,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    md_wall = time.perf_counter() - t1

except Exception as exc:
    run_error = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }

if run_error is not None:
    print("MD failed:")
    print(json.dumps(run_error, indent=2))
else:
    final_atoms = final_state.to_atoms()
    final_temperature = final_state.calc_temperature().detach().cpu().numpy()
    final_energy = final_state.energy.detach().cpu().numpy()
    final_positions = final_state.positions.detach().cpu().numpy()
    final_cell = final_state.cell.detach().cpu().numpy()
    final_min_dist = [shortest_pair_distance(atoms) for atoms in final_atoms]

    n_systems = int(final_state.n_systems)
    bad_indices, bad_reasons = bad_indices_from_arrays(
        n_systems,
        temperature=final_temperature,
        energy=final_energy,
        positions=final_positions,
        cell=final_cell,
    )
    for idx, dist in enumerate(final_min_dist):
        if dist is None or (not math.isfinite(dist)) or dist < 0.8:
            bad_indices.append(idx)
            bad_reasons.append(f"min_pair_distance: system {idx} has distance {dist}")
    bad_indices = sorted(i for i in set(bad_indices) if 0 <= i < len(structure_paths))

    system_steps_per_s = n_systems * md_steps / md_wall
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None

    print("MD completed")
    print(f"  systems: {int(final_state.n_systems)}")
    print(f"  precondition wall: {precondition_wall:.3f} s")
    print(f"  md wall: {md_wall:.3f} s")
    print(f"  system-steps/s: {system_steps_per_s:.3f}")
    print(f"  peak CUDA memory: {peak_mem:.3f} GB" if peak_mem is not None else "  peak CUDA memory: None")
    print(f"  bad indices: {bad_indices}")
    print(f"  bad reasons: {bad_reasons}")
    print(f"  final temperature K: {final_temperature}")
    print(f"  final energy eV: {final_energy}")
    print(f"  final min distance A: {final_min_dist}")

# %%
# ============================================================================
# Save compact results
# ============================================================================

if run_error is not None:
    result = {
        "status": "error",
        "error": run_error,
    }
else:
    result = {
        "status": "valid" if not bad_indices else "invalid",
        "bad_indices": bad_indices,
        "bad_structure_files": [str(structure_paths[i]) for i in bad_indices if 0 <= i < len(structure_paths)],
        "bad_reasons": bad_reasons,
        "temperature_K": final_temperature.tolist(),
        "energy_eV": final_energy.tolist(),
        "min_pair_distance_A": final_min_dist,
        "precondition_wall_s": precondition_wall,
        "md_wall_s": md_wall,
        "system_steps_per_s": system_steps_per_s,
        "peak_cuda_memory_GB": peak_mem,
    }

record = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "structure_dir": str(structure_dir),
    "structure_pattern": structure_pattern,
    "structure_files": [str(p) for p in structure_paths],
    "model_path": model_path,
    "modal": modal,
    "device": device,
    "accelerator": accelerator,
    "use_d3": use_d3,
    "d3_mode": d3_mode if use_d3 else None,
    "d3_batch_threshold": d3_batch_threshold if use_d3 else None,
    "use_autobatcher": use_autobatcher,
    "temperature_K": temperature,
    "timestep_fs": timestep_fs,
    "precondition_steps": precondition_steps,
    "md_steps": md_steps,
    "result": result,
}

with open(output_json, "w") as f:
    json.dump(record, f, indent=2)

row = {
    "created_at": record["created_at"],
    "n_structures": len(structure_paths),
    "accelerator": accelerator,
    "use_d3": use_d3,
    "d3_mode": d3_mode if use_d3 else "",
    "use_autobatcher": use_autobatcher,
    "temperature_K": temperature,
    "timestep_fs": timestep_fs,
    "precondition_steps": precondition_steps,
    "md_steps": md_steps,
    "status": result["status"],
    "bad_indices": json.dumps(result.get("bad_indices", [])),
    "system_steps_per_s": result.get("system_steps_per_s"),
    "peak_cuda_memory_GB": result.get("peak_cuda_memory_GB"),
    "error_type": result.get("error", {}).get("type"),
    "error_message": result.get("error", {}).get("message"),
}
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

print(f"Saved JSON: {output_json}")
print(f"Saved CSV: {output_csv}")

# %%