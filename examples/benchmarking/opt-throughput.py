# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
#   "matbench-discovery",
#   "numpy",
#   "pandas",
#   "pymatgen",
#   "torch",
# ]
# ///
"""Optimization throughput benchmark on WBM initial structures.

Relaxes a random sample of WBM structures using torch-sim's LBFGS or FIRE
optimizer with a batched MACE (or FairChem) model and reports throughput
in structures per minute.

Results are saved to benchmark_results/opt-<model>-<optimizer>-<timestamp>.csv.

Example:
    uv run --with ".[mace]" examples/benchmarking/opt-throughput.py \
        --model mace --optimizer lbfgs --n-structures 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch_sim.typing import MemoryScaling

import numpy as np
import pandas as pd
import torch


TEMPERATURE = 300.0
MAX_ATOMS = {"mace": 5000, "fairchem": 2000}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Optimization throughput benchmark on WBM structures."
    )
    parser.add_argument(
        "--model",
        choices=["mace", "fairchem"],
        default="mace",
        help="Model to use.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model checkpoint. Uses MACE-MP-0 small if omitted.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["lbfgs", "fire"],
        default="lbfgs",
        help="Optimizer to benchmark.",
    )
    parser.add_argument(
        "--n-structures",
        type=int,
        default=100,
        help="Number of WBM structures to relax.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for WBM sampling.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum optimizer steps per structure.",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=0.05,
        help="Force convergence threshold (eV/Å).",
    )
    parser.add_argument(
        "--cell-filter",
        choices=["frechet", "exp", "none"],
        default="frechet",
        help="Cell filter for variable-cell relaxation. Use 'none' for fixed cell.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Torch dtype.",
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
        help="Skip the sequential ASE baseline.",
    )
    parser.add_argument(
        "--ase-n-structures",
        type=int,
        default=10,
        help="Number of structures to relax with ASE (subset, since it is slow).",
    )
    return parser.parse_args()


def clear_gpu_memory() -> None:
    """Empty CUDA cache and run Python GC."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def _sample_wbm_structures(n_structures: int, seed: int) -> list[dict[str, Any]]:
    """Fetch random WBM structures as pymatgen dicts."""
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip
    from pymatgen.io.ase import AseAtomsAdaptor

    wbm_zip_path = DataFiles.wbm_initial_atoms.path
    all_atoms = ase_atoms_from_zip(wbm_zip_path)
    if n_structures > len(all_atoms):
        raise RuntimeError(
            f"Requested {n_structures} structures but WBM only has {len(all_atoms)}."
        )
    np_rng = np.random.default_rng(seed=seed)
    indices = np_rng.choice(len(all_atoms), size=n_structures, replace=False)
    sampled = [all_atoms[int(i)] for i in indices.tolist()]
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_structure(a).as_dict() for a in sampled]


def _structures_to_sim_state(
    structures: list[dict[str, Any]],
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    """Convert pymatgen structure dicts to a batched SimState."""
    from pymatgen.core import Structure

    from torch_sim.io import structures_to_state

    return structures_to_state(
        [Structure.from_dict(s) for s in structures],
        device=device,
        dtype=dtype,
    )


def run_ase_optimization(
    structures: list[dict[str, Any]],
    calculator: Any,
    optimizer_name: str,
    cell_filter_name: str,
    max_steps: int,
    f_max: float,
) -> dict[str, Any]:
    """Run sequential ASE relaxation. Returns timing + convergence metrics."""
    from ase.filters import ExpCellFilter, FrechetCellFilter
    from ase.optimize import FIRE, LBFGS
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    ase_optimizer_cls = LBFGS if optimizer_name == "lbfgs" else FIRE
    cell_filter_cls = {
        "frechet": FrechetCellFilter,
        "exp": ExpCellFilter,
        "none": None,
    }[cell_filter_name]

    converged = 0
    t0 = time.perf_counter()
    for struct_dict in structures:
        atoms = adaptor.get_atoms(Structure.from_dict(struct_dict))
        atoms.calc = calculator
        system: Any = cell_filter_cls(atoms) if cell_filter_cls is not None else atoms
        opt = ase_optimizer_cls(system, logfile=os.devnull)  # type: ignore[arg-type]
        opt.run(fmax=f_max, steps=max_steps)
        if opt.get_number_of_steps() < max_steps:
            converged += 1
    elapsed = time.perf_counter() - t0

    n = len(structures)
    return {
        "n_relaxed": n,
        "n_converged": converged,
        "converged_pct": round(100 * converged / n, 1) if n else 0,
        "total_s": round(elapsed, 3),
        "s_per_structure": round(elapsed / n, 4) if n else 0,
        "structures_per_min": round(n / elapsed * 60, 2) if elapsed > 0 else 0,
    }


def load_model(
    model_type: str,
    model_path: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, MemoryScaling]:
    """Return (torchsim_model, ase_calculator, memory_scales_with).

    memory_scales_with is model-dependent:
    - MACE uses a radial cutoff, so n_atoms_x_density is the right proxy.
    - FairChem builds its own graph, so n_atoms suffices.
    """
    dtype_str = str(dtype).split(".")[-1]
    if model_type == "mace":
        from mace.calculators import MACECalculator

        from torch_sim.models.mace import MaceModel, MaceUrls

        path = model_path or MaceUrls.mace_mp_small
        model = MaceModel(model=path, device=device, dtype=dtype, enable_cueq=False)
        calculator = MACECalculator(
            model_paths=path, device=str(device), default_dtype=dtype_str
        )
        return model, calculator, "n_atoms_x_density"

    raise ValueError(f"Unknown model type: {model_type}")


def run_torchsim_optimization(
    sim_state: Any,
    model: Any,
    memory_scales_with: MemoryScaling,
    optimizer_name: str,
    cell_filter_name: str,
    max_steps: int,
    f_max: float,
    max_atoms: int,
) -> dict[str, Any]:
    """Run batched optimization and return timing + convergence metrics."""
    import torch_sim as ts
    from torch_sim.autobatching import InFlightAutoBatcher
    from torch_sim.optimizers import Optimizer

    optimizer = Optimizer[optimizer_name]
    convergence_fn = ts.generate_force_convergence_fn(force_tol=f_max)

    autobatcher = InFlightAutoBatcher(
        model=model,
        memory_scales_with=memory_scales_with,
        max_memory_scaler=max_atoms,
    )

    init_kwargs: dict[str, Any] = {}
    if cell_filter_name != "none":
        init_kwargs["cell_filter"] = ts.CellFilter[cell_filter_name]

    t0 = time.perf_counter()
    final_state = ts.optimize(
        system=sim_state,
        model=model,
        optimizer=optimizer,
        max_steps=max_steps,
        convergence_fn=convergence_fn,
        steps_between_swaps=5,
        autobatcher=autobatcher,
        init_kwargs=init_kwargs or None,
    )
    elapsed = time.perf_counter() - t0

    final_states = (
        final_state.split() if isinstance(final_state, ts.SimState) else final_state
    )
    n_relaxed = len(final_states)

    converged = 0
    for state in final_states:
        forces = model(state)["forces"]
        if float(torch.linalg.norm(forces, dim=1).max()) <= f_max:
            converged += 1

    return {
        "n_relaxed": n_relaxed,
        "n_converged": converged,
        "converged_pct": round(100 * converged / n_relaxed, 1) if n_relaxed else 0,
        "total_s": round(elapsed, 3),
        "s_per_structure": round(elapsed / n_relaxed, 4) if n_relaxed else 0,
        "structures_per_min": round(n_relaxed / elapsed * 60, 2) if elapsed > 0 else 0,
    }


def _save_and_print(results: list[dict[str, Any]], tag: str) -> None:
    """Save results CSV and print summary."""
    df = pd.DataFrame(results)
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = Path(f"benchmark_results/{tag}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    print(json.dumps(results, indent=2, default=str))


def main() -> None:
    """Entry point."""
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    max_atoms = args.max_atoms or MAX_ATOMS.get(args.model, 3000)

    print(
        f"Optimization throughput: {args.model} / {args.optimizer} "
        f"on {args.n_structures} WBM structures ({device}, {args.dtype})"
    )

    print("Loading WBM structures...")
    t0 = time.perf_counter()
    structures = _sample_wbm_structures(args.n_structures, args.seed)
    load_s = time.perf_counter() - t0
    print(f"  Loaded {len(structures)} structures in {load_s:.2f}s")

    n_atoms_list = []
    for s in structures:
        from pymatgen.core import Structure

        n_atoms_list.append(len(Structure.from_dict(s)))

    print(
        f"  Atom count: min={min(n_atoms_list)}, "
        f"max={max(n_atoms_list)}, mean={np.mean(n_atoms_list):.1f}"
    )

    sim_state = _structures_to_sim_state(structures, dtype=dtype, device=device)

    ase_metrics: dict[str, Any] = {}
    clear_gpu_memory()
    print(f"Loading {args.model} model...")
    model, calculator, memory_scales_with = load_model(
        args.model, args.model_path, device, dtype
    )

    if not args.skip_ase:
        n_ase = min(args.ase_n_structures, len(structures))
        print(
            f"Running ASE {args.optimizer.upper()} on {n_ase} structures "
            f"(cell_filter={args.cell_filter})..."
        )
        ase_metrics = run_ase_optimization(
            structures=structures[:n_ase],
            calculator=calculator,
            optimizer_name=args.optimizer,
            cell_filter_name=args.cell_filter,
            max_steps=args.max_steps,
            f_max=args.f_max,
        )
        print(
            f"  ASE: {ase_metrics['n_converged']}/{ase_metrics['n_relaxed']} converged "
            f"— {ase_metrics['structures_per_min']} structures/min"
        )
        clear_gpu_memory()

    print(
        f"Running torch-sim {args.optimizer.upper()} (max_steps={args.max_steps}, "
        f"f_max={args.f_max}, cell_filter={args.cell_filter})..."
    )
    ts_metrics = run_torchsim_optimization(
        sim_state=sim_state,
        model=model,
        memory_scales_with=memory_scales_with,
        optimizer_name=args.optimizer,
        cell_filter_name=args.cell_filter,
        max_steps=args.max_steps,
        f_max=args.f_max,
        max_atoms=max_atoms,
    )
    print(
        f"  torch-sim: {ts_metrics['n_converged']}/{ts_metrics['n_relaxed']} converged "
        f"— {ts_metrics['structures_per_min']} structures/min"
    )

    speedup = None
    if ase_metrics and ase_metrics["s_per_structure"] > 0:
        speedup = round(ase_metrics["s_per_structure"] / ts_metrics["s_per_structure"], 2)
        print(f"  Speedup vs ASE: {speedup}x")

    row = {
        "model": args.model,
        "optimizer": args.optimizer,
        "cell_filter": args.cell_filter,
        "dtype": args.dtype,
        "device": str(device),
        "n_structures_requested": args.n_structures,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "f_max": args.f_max,
        "load_s": round(load_s, 3),
        "n_atoms_mean": round(float(np.mean(n_atoms_list)), 1),
        "n_atoms_max": max(n_atoms_list),
        **{f"ts_{k}": v for k, v in ts_metrics.items()},
        **{f"ase_{k}": v for k, v in ase_metrics.items()},
        "speedup_vs_ase": speedup,
    }

    del model
    clear_gpu_memory()

    _save_and_print([row], f"opt-{args.model}-{args.optimizer}")


if __name__ == "__main__":
    main()
