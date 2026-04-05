# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matbench-discovery",
#   "mp-api",
#   "numpy",
#   "pymatgen",
#   "torch",
#   "vesin[torch]",
# ]
# ///
"""Neighbor-list backend benchmark using random MP or WBM structures.

Directly times each torch-sim NL backend without any model evaluation.

Example:
    uv run --with-editable . examples/benchmarking/neighborlists.py \
        --source wbm --n-structures 100 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


import numpy as np
import torch


VALID_NL_BACKENDS = (
    "torch_linked_cell",
    "torch_n2",
    "vesin",
    "alchemi_n2",
    "alchemi_cell",
)
DEFAULT_CUTOFF = 5.0


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark torch-sim neighbor-list backends on random public structures."
        )
    )
    parser.add_argument(
        "--source",
        choices=("mp", "wbm"),
        required=True,
        help="Public structure source: mp (Materials Project) or wbm (Matbench).",
    )
    parser.add_argument(
        "--n-structures",
        type=int,
        default=100,
        help="How many random structures to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--nl-backend",
        choices=VALID_NL_BACKENDS,
        nargs="+",
        default=list(VALID_NL_BACKENDS),
        help="Neighbor-list backend(s) to benchmark. Defaults to all.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CUTOFF,
        help="Neighbor-list cutoff radius in Angstrom.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Torch device, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Torch dtype for position/cell tensors.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of timed repetitions (median is reported).",
    )
    parsed_args = parser.parse_args()
    if parsed_args.n_structures <= 0:
        parser.error("--n-structures must be > 0")
    if parsed_args.cutoff <= 0:
        parser.error("--cutoff must be > 0")
    return parsed_args


def _sample_mp_structures(n_structures: int, seed: int) -> list[dict[str, Any]]:
    """Fetch random MP structures as pymatgen dicts."""
    from mp_api.client import MPRester

    if not os.environ.get("MP_API_KEY"):
        raise RuntimeError("MP_API_KEY is required for --source mp")

    with MPRester() as mpr:
        sampled_material_ids: list[str] = []
        py_rng = random.Random(seed)
        id_docs = mpr.summary.search(
            fields=["material_id"],
            all_fields=False,
            chunk_size=2_000,
        )
        for stream_idx, doc in enumerate(id_docs):
            material_id = str(doc.material_id)
            if stream_idx < n_structures:
                sampled_material_ids.append(material_id)
                continue
            replacement_idx = py_rng.randint(0, stream_idx)
            if replacement_idx < n_structures:
                sampled_material_ids[replacement_idx] = material_id

        if len(sampled_material_ids) < n_structures:
            raise RuntimeError(
                f"Requested {n_structures} structures but only found "
                f"{len(sampled_material_ids)} in MP."
            )

        structure_docs = mpr.summary.search(
            material_ids=sampled_material_ids,
            fields=["material_id", "structure"],
            all_fields=False,
        )

    structure_by_id: dict[str, dict[str, Any]] = {}
    for doc in structure_docs:
        structure_by_id[str(doc.material_id)] = doc.structure.as_dict()

    missing_ids = [mid for mid in sampled_material_ids if mid not in structure_by_id]
    if missing_ids:
        raise RuntimeError(
            f"Failed to fetch structures for {len(missing_ids)} sampled MP IDs."
        )

    return [structure_by_id[mid] for mid in sampled_material_ids]


def _sample_wbm_structures(n_structures: int, seed: int) -> list[dict[str, Any]]:
    """Fetch random WBM structures as pymatgen dicts."""
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip
    from pymatgen.io.ase import AseAtomsAdaptor

    wbm_zip_path = DataFiles.wbm_initial_atoms.path
    all_atoms = ase_atoms_from_zip(wbm_zip_path)
    if n_structures > len(all_atoms):
        raise RuntimeError(
            f"Requested {n_structures} structures, but WBM has {len(all_atoms)}"
        )

    np_rng = np.random.default_rng(seed=seed)
    sampled_indices = np_rng.choice(len(all_atoms), size=n_structures, replace=False)
    sampled_atoms = [all_atoms[int(idx)] for idx in sampled_indices.tolist()]
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_structure(atoms).as_dict() for atoms in sampled_atoms]


def load_public_structures(
    source: str,
    n_structures: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load random structures from the requested public source."""
    if source == "mp":
        return _sample_mp_structures(n_structures=n_structures, seed=seed)
    if source == "wbm":
        return _sample_wbm_structures(n_structures=n_structures, seed=seed)
    raise ValueError(f"Unsupported source: {source}")


def _build_tensors(
    structures: list[dict[str, Any]],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert pymatgen structure dicts to (positions, cell, pbc, system_idx) tensors."""
    from pymatgen.core import Structure

    from torch_sim.io import structures_to_state

    state = structures_to_state(
        [Structure.from_dict(s) for s in structures],
        device=torch.device(device),
        dtype=dtype,
    )
    return state.positions, state.cell, state.pbc, state.system_idx


def _get_nl_fn(backend: str) -> Callable:
    """Return the neighbor-list function for the given backend name."""
    if backend == "torch_linked_cell":
        from torch_sim.neighbors.torch_nl import torch_nl_linked_cell

        return torch_nl_linked_cell
    if backend == "torch_n2":
        from torch_sim.neighbors.torch_nl import torch_nl_n2

        return torch_nl_n2
    if backend == "vesin":
        from torch_sim.neighbors.vesin import vesin_nl_ts

        return vesin_nl_ts
    if backend == "alchemi_n2":
        from torch_sim.neighbors.alchemiops import alchemiops_nl_n2

        return alchemiops_nl_n2
    if backend == "alchemi_cell":
        from torch_sim.neighbors.alchemiops import alchemiops_nl_cell_list

        return alchemiops_nl_cell_list
    raise ValueError(f"Unknown backend: {backend}")


def _benchmark_backend(
    backend: str,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    system_idx: torch.Tensor,
    n_repeats: int,
    device: str,
) -> dict[str, Any] | None:
    """Time one backend, returning None if it is unavailable."""
    try:
        nl_fn = _get_nl_fn(backend)
    except ImportError as exc:
        return {"nl_backend": backend, "skipped": str(exc)}

    n_atoms = positions.shape[0]
    is_cuda = device.startswith("cuda")

    nl_fn(positions, cell, pbc, cutoff, system_idx)
    if is_cuda:
        torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        mapping, _, _ = nl_fn(positions, cell, pbc, cutoff, system_idx)
        if is_cuda:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    median_s = float(np.median(timings))
    return {
        "nl_backend": backend,
        "n_pairs": int(mapping.shape[1]),
        "median_nl_s": round(median_s, 6),
        "timings_s": [round(t, 6) for t in timings],
        "atoms_per_s": round(n_atoms / median_s, 1) if median_s > 0 else 0,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run benchmarks for all requested backends and return compact metrics."""
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32

    wall_start = time.perf_counter()
    structures = load_public_structures(
        source=args.source,
        n_structures=args.n_structures,
        seed=args.seed,
    )
    load_s = time.perf_counter() - wall_start

    positions, cell, pbc, system_idx = _build_tensors(
        structures, dtype=torch_dtype, device=args.device
    )
    cutoff = torch.tensor(args.cutoff, dtype=torch_dtype, device=args.device)
    n_atoms = positions.shape[0]

    backends = args.nl_backend if isinstance(args.nl_backend, list) else [args.nl_backend]
    results = [
        _benchmark_backend(
            backend=b,
            positions=positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            system_idx=system_idx,
            n_repeats=args.n_repeats,
            device=args.device,
        )
        for b in backends
    ]

    return {
        "source": args.source,
        "n_structures": len(structures),
        "n_atoms": n_atoms,
        "cutoff_angstrom": args.cutoff,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "n_repeats": args.n_repeats,
        "load_s": round(load_s, 3),
        "backends": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(parse_args()), indent=2))
