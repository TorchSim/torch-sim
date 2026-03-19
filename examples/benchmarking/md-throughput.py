# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
#   "numpy",
#   "pandas",
#   "torch",
# ]
# ///
"""Throughput benchmark: ASE Langevin vs torch-sim NVT-Langevin MD.

Benchmarks three approaches on FCC copper systems of varying size:
  1. ASE Langevin dynamics (single system, sequential)
  2. torch-sim integrate (batched, GPU-accelerated)
  3. Direct model forward passes (raw throughput)

Results are saved to benchmark_results/<model>_<timestamp>.csv.

Example:
    uv run --with ".[mace]" examples/benchmarking/md-throughput.py --model mace
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ase import units
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin

import torch_sim as ts
from torch_sim.integrators import Integrator
from torch_sim.io import atoms_to_state


SIZES = range(2, 8)
DTYPES = [torch.float32, torch.float64]
N_STEPS = 100
TEMPERATURE = 300.0
TIMESTEP = 0.001
N_FORWARD_CALLS = 100
MAX_ATOMS = {"mace": 7000, "fairchem": 3000}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark ASE vs torch-sim MD")
    parser.add_argument(
        "--model",
        choices=["mace", "fairchem"],
        default="mace",
        help="Model to benchmark.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model checkpoint. Uses bundled MACE-MP-0 small if omitted.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(SIZES),
        help="FCC supercell sizes to benchmark (repeats along each axis).",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
        help="Precisions to benchmark.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=N_STEPS,
        help="MD steps per benchmark.",
    )
    parser.add_argument(
        "--n-forward-calls",
        type=int,
        default=N_FORWARD_CALLS,
        help="Forward-pass repetitions for raw throughput benchmark.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Max atoms per batch (overrides per-model default).",
    )
    parser.add_argument(
        "--skip-ase",
        action="store_true",
        help="Skip the ASE baseline (useful when no ASE calculator is available).",
    )
    return parser.parse_args()


def clear_gpu_memory() -> None:
    """Empty the CUDA cache and run the Python GC."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def report_gpu_memory(label: str = "") -> None:
    """Print current CUDA memory allocation."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  GPU ({label}): {allocated:.1f}MB alloc, {reserved:.1f}MB reserved")


def create_fcc_copper(size: int) -> Any:
    """Create a periodic FCC copper supercell."""
    return FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol="Cu",
        size=(size, size, size),
        pbc=True,
    )


def load_model(
    model_type: str,
    model_path: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    """Return (torchsim_model, ase_calculator)."""
    if model_type == "mace":
        from mace.calculators import MACECalculator

        from torch_sim.models.mace import MaceModel, MaceUrls

        path = model_path or MaceUrls.mace_mp_small
        dtype_str = str(dtype).split(".")[-1]
        model = MaceModel(model=path, device=device, dtype=dtype, enable_cueq=False)
        calculator = MACECalculator(
            model_paths=path, device=str(device), default_dtype=dtype_str
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, calculator


def run_ase_md(atoms: Any, calculator: Any, n_steps: int) -> float:
    """Run ASE Langevin dynamics. Returns wall time in seconds."""
    atoms = atoms.copy()
    atoms.calc = calculator
    dyn = Langevin(
        atoms,
        TIMESTEP * 1000 * units.fs,
        TEMPERATURE * units.kB,
        friction=0.002,
    )
    t0 = time.perf_counter()
    dyn.run(n_steps)
    return time.perf_counter() - t0


def run_torchsim_md(
    atoms: Any,
    model: Any,
    n_steps: int,
    max_atoms: int,
) -> float:
    """Run torch-sim batched NVT-Langevin. Returns wall time per system in seconds."""
    n_atoms = len(atoms)
    batch_size = max(1, max_atoms // n_atoms)
    print(f"    batch_size={batch_size} ({batch_size * n_atoms} total atoms)")

    t0 = time.perf_counter()
    ts.integrate(
        system=[atoms] * batch_size,
        model=model,
        integrator=Integrator.nvt_langevin,
        n_steps=n_steps,
        temperature=TEMPERATURE,
        timestep=TIMESTEP,
    )
    elapsed = time.perf_counter() - t0
    return elapsed / batch_size


def run_forward_passes(
    atoms: Any,
    model: Any,
    n_calls: int,
    max_atoms: int,
) -> float:
    """Time raw model forward passes. Returns wall time per (system * call)."""
    n_atoms = len(atoms)
    batch_size = max(1, max_atoms // n_atoms)
    print(f"    batch_size={batch_size} ({batch_size * n_atoms} total atoms)")

    state = atoms_to_state([atoms] * batch_size, device=model.device, dtype=model.dtype)

    is_cuda = str(model.device).startswith("cuda")
    # warmup
    model(state)
    if is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        model(state)
        if is_cuda:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    return float(np.median(timings)) / batch_size


def _benchmark_size(
    args: argparse.Namespace,
    model: Any,
    calculator: Any,
    dtype_str: str,
    size: int,
    max_atoms: int,
) -> dict[str, Any]:
    """Run all benchmarks for one (dtype, size) combination."""
    atoms = create_fcc_copper(size)
    n_atoms = len(atoms)
    print(f"\n  {size}x{size}x{size} FCC Cu — {n_atoms} atoms")

    row: dict[str, Any] = {
        "model": args.model,
        "dtype": dtype_str,
        "size": size,
        "n_atoms": n_atoms,
        "n_steps": args.n_steps,
    }

    if not args.skip_ase:
        print("  ASE Langevin...")
        ase_time = run_ase_md(atoms, calculator, args.n_steps)
        row["ase_total_s"] = round(ase_time, 4)
        row["ase_s_per_step"] = round(ase_time / args.n_steps, 6)
        print(f"    {ase_time:.3f}s total, {ase_time / args.n_steps:.5f}s/step")
    else:
        row["ase_total_s"] = None
        row["ase_s_per_step"] = None

    clear_gpu_memory()
    report_gpu_memory("pre-torchsim")

    print("  torch-sim NVT-Langevin...")
    ts_time = run_torchsim_md(atoms, model, args.n_steps, max_atoms)
    row["ts_s_per_system"] = round(ts_time, 6)
    row["ts_s_per_step"] = round(ts_time / args.n_steps, 8)
    print(f"    {ts_time:.4f}s/system, {ts_time / args.n_steps:.6f}s/step")

    clear_gpu_memory()
    report_gpu_memory("pre-forward")

    print(f"  Direct forward ({args.n_forward_calls} calls)...")
    fwd_time = run_forward_passes(atoms, model, args.n_forward_calls, max_atoms)
    row["fwd_median_s_per_system"] = round(fwd_time, 8)
    print(f"    {fwd_time * 1000:.3f}ms/system (median)")

    if not args.skip_ase and row["ase_total_s"] is not None:
        row["ts_speedup_vs_ase"] = round(row["ase_total_s"] / ts_time, 2)

    return row


def _save_and_print(all_results: list[dict[str, Any]], model_name: str) -> None:
    """Persist results to CSV and print a summary table."""
    results_df = pd.DataFrame(all_results)
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = Path(f"benchmark_results/{model_name}_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    _summary_cols = [
        "model",
        "dtype",
        "size",
        "n_atoms",
        "ase_s_per_step",
        "ts_s_per_step",
        "fwd_median_s_per_system",
        "ts_speedup_vs_ase",
    ]
    summary_cols = [c for c in _summary_cols if c in results_df.columns]
    print("\nSummary:")
    print(results_df[summary_cols].to_string(index=False))
    print(json.dumps(all_results, indent=2, default=str))


def main() -> None:
    """Entry point."""
    args = parse_args()
    device = torch.device(args.device)
    dtypes = [torch.float32 if d == "float32" else torch.float64 for d in args.dtypes]
    max_atoms = args.max_atoms or MAX_ATOMS.get(args.model, 3000)

    print(f"Benchmarking {args.model} on {device}")
    print(f"Sizes: {args.sizes}, dtypes: {args.dtypes}, steps: {args.n_steps}")

    clear_gpu_memory()
    report_gpu_memory("start")

    all_results: list[dict[str, Any]] = []

    for dtype in dtypes:
        dtype_str = str(dtype).split(".")[-1]
        print(f"\n=== dtype={dtype_str} ===")
        clear_gpu_memory()

        print(f"Loading {args.model}...")
        model, calculator = load_model(args.model, args.model_path, device, dtype)
        report_gpu_memory("model loaded")

        for size in args.sizes:
            row = _benchmark_size(args, model, calculator, dtype_str, size, max_atoms)
            all_results.append(row)
            clear_gpu_memory()

        del model, calculator
        clear_gpu_memory()

    _save_and_print(all_results, args.model)


if __name__ == "__main__":
    main()
