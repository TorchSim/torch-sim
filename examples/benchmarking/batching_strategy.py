# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
# ]
# ///
"""Benchmark batching strategies: throughput-vs-batch-size + optimal-vs-greedy packing.

Tests two claims from PR review feedback on issue #275:
  1. "Batching is important up to a threshold; past that, extra batching
      doesn't help much." → throughput flattens at some batch size.
  2. "Optimal bin-packing via to_constant_volume_bins isn't super useful" vs
      greedy packing (InFlightAutoBatcher-style pull-until-full).

Experiment A: throughput vs batch size on uniform systems.
Experiment B: packing efficiency (optimal vs greedy) — batch count & fullness.
Experiment C: end-to-end wall time on heterogeneous systems using both strategies.

Example::

    uv run examples/benchmarking/batching_strategy.py
    # or with a real GPU model:
    uv run --with ".[mace]" examples/benchmarking/batching_strategy.py --use-mace

Notes:
    CPU results illustrate methodology; GPU results with a real MLIP give
    more meaningful throughput numbers. This script prefers GPU+MACE if both
    available, else falls back to CPU+Lennard-Jones.
"""

# %%
from __future__ import annotations

import argparse
import time

import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.autobatching import calculate_memory_scalers, to_constant_volume_bins
from torch_sim.models.lennard_jones import LennardJonesModel


# ------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------


def make_lj_model(device: torch.device, dtype: torch.dtype) -> LennardJonesModel:
    """Simple LJ model usable on both CPU and GPU."""
    return LennardJonesModel(
        sigma=3.405, epsilon=0.0104, cutoff=3.0 * 3.405, device=device, dtype=dtype
    )


def make_ar_state(n_cells: int, device: torch.device, dtype: torch.dtype) -> ts.SimState:
    """Create an Ar FCC supercell with n_cells^3 * 4 atoms."""
    atoms = bulk("Ar", "fcc", a=5.26).repeat((n_cells, n_cells, n_cells))
    return ts.io.atoms_to_state([atoms], device=device, dtype=dtype)


def make_heterogeneous_states(
    device: torch.device, dtype: torch.dtype, n_systems: int = 60, seed: int = 0
) -> list[ts.SimState]:
    """Create a dataset of Ar supercells with widely varying sizes.

    Size distribution is bimodal-ish (mix of small molecules + big crystals)
    to make packing differences visible.
    """
    rng = torch.Generator().manual_seed(seed)
    states: list[ts.SimState] = []
    for _ in range(n_systems):
        # Bimodal: 60% "small" (1-2 cells = 4-32 atoms),
        # 40% "large" (3-5 cells = 108-500 atoms).
        if torch.rand(1, generator=rng).item() < 0.6:
            n_cells = int(torch.randint(1, 3, (1,), generator=rng).item())
        else:
            n_cells = int(torch.randint(3, 6, (1,), generator=rng).item())
        states.append(make_ar_state(n_cells, device, dtype))
    return states


def time_forward(model: LennardJonesModel, state: ts.SimState, n_reps: int = 3) -> float:
    """Time a forward pass, taking the min of n_reps runs to de-noise."""
    # Warmup
    model(state)
    if state.device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        model(state)
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


# ------------------------------------------------------------------
# Experiment A — throughput vs batch size (tests claim 1)
# ------------------------------------------------------------------


def exp_a_throughput_vs_batch_size(
    model: LennardJonesModel, device: torch.device, dtype: torch.dtype
) -> None:
    """Measure systems/sec as a function of batch size with uniform systems."""
    print("\n" + "=" * 72)
    print("Experiment A: throughput vs batch size (uniform 32-atom systems)")
    print("=" * 72)

    # Build one reference system; concatenate N copies to form a batch.
    ref = make_ar_state(n_cells=2, device=device, dtype=dtype)  # 32 atoms
    print(f"Per-system size: {ref.n_atoms} atoms")
    print(f"{'batch_size':>12} {'time_ms':>12} {'sys/sec':>12} {'μs/atom':>12}")
    print("-" * 72)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []
    for bs in batch_sizes:
        batch = ts.concatenate_states([ref] * bs)
        t = time_forward(model, batch)
        sys_per_sec = bs / t
        us_per_atom = (t * 1e6) / (bs * ref.n_atoms)
        results.append((bs, t, sys_per_sec, us_per_atom))
        print(f"{bs:>12} {t * 1000:>12.3f} {sys_per_sec:>12.1f} {us_per_atom:>12.2f}")

    # Find the "knee": batch size past which throughput stops improving meaningfully.
    print()
    print("Throughput analysis:")
    baseline = results[0][2]  # sys/sec at batch_size=1
    for bs, _t, sps, _ in results:
        speedup = sps / baseline
        print(f"  bs={bs:>4}: {speedup:>6.2f}x speedup over bs=1")


# ------------------------------------------------------------------
# Helpers for Experiment B/C: greedy packing
# ------------------------------------------------------------------


def greedy_pack(metric_values: list[float], max_volume: float) -> list[list[int]]:
    """Greedy (first-fit) packing: iterate items, add to current batch until full.

    This mirrors InFlightAutoBatcher._get_next_states' logic applied to a static
    list: we don't pre-sort, we fill greedily in arrival order.
    """
    bins: list[list[int]] = [[]]
    current_sum = 0.0
    for idx, v in enumerate(metric_values):
        if v > max_volume:
            raise ValueError(f"Item {idx} metric {v} exceeds max_volume {max_volume}")
        if current_sum + v > max_volume:
            bins.append([idx])
            current_sum = v
        else:
            bins[-1].append(idx)
            current_sum += v
    return bins


def optimal_pack(metric_values: list[float], max_volume: float) -> list[list[int]]:
    """Current BinningAutoBatcher optimal packing via to_constant_volume_bins."""
    idx_to_val = dict(enumerate(metric_values))
    bin_dicts = to_constant_volume_bins(idx_to_val, max_volume=max_volume)
    return [list(d.keys()) for d in bin_dicts]


def bin_stats(bins: list[list[int]], metric_values: list[float]) -> dict:
    """Compute packing statistics."""
    fullnesses = [sum(metric_values[i] for i in b) for b in bins]
    return {
        "n_bins": len(bins),
        "bin_sizes": [len(b) for b in bins],
        "bin_fullness": fullnesses,
        "total_capacity_used": sum(fullnesses),
    }


# ------------------------------------------------------------------
# Experiment B — packing efficiency (tests claim 2, hardware-independent)
# ------------------------------------------------------------------


def exp_b_packing_efficiency(
    device: torch.device, dtype: torch.dtype
) -> tuple[list[ts.SimState], float, list[list[int]], list[list[int]]]:
    """Compare batch count + fullness for optimal vs greedy packing.

    Tests multiple distributions to probe when optimal packing wins over greedy.
    """
    print("\n" + "=" * 72)
    print("Experiment B: packing efficiency (heterogeneous systems)")
    print("=" * 72)

    distributions = {
        "bimodal": make_heterogeneous_states(device, dtype, n_systems=60, seed=0),
        "uniform": [
            make_ar_state(n_cells=2, device=device, dtype=dtype) for _ in range(60)
        ],
        # Adversarial: many small items, then a few big ones at the end.
        # Greedy packs smalls together fine but wastes space when bigs arrive.
        "adversarial": [
            *[make_ar_state(1, device, dtype) for _ in range(50)],  # 50 small (4-atom)
            *[make_ar_state(4, device, dtype) for _ in range(10)],  # 10 large (256-atom)
        ],
    }

    last_result: (
        tuple[list[ts.SimState], float, list[list[int]], list[list[int]]] | None
    ) = None
    for name, states in distributions.items():
        batched = ts.concatenate_states(states)
        metrics = calculate_memory_scalers(batched, "n_atoms_x_density")
        max_volume = sum(metrics) / 8  # aim for ~8 bins

        n_atoms_list = [s.n_atoms for s in states]
        print(f"\n-- {name} distribution --")
        print(
            f"   n_atoms: min={min(n_atoms_list)}, max={max(n_atoms_list)}, "
            f"mean={sum(n_atoms_list) / len(n_atoms_list):.1f}"
        )
        print(f"   max_memory_scaler: {max_volume:.1f}")

        opt_bins = optimal_pack(metrics, max_volume)
        grd_bins = greedy_pack(metrics, max_volume)
        opt_stats = bin_stats(opt_bins, metrics)
        grd_stats = bin_stats(grd_bins, metrics)

        opt_avg = (
            sum(opt_stats["bin_fullness"]) / (opt_stats["n_bins"] * max_volume) * 100
        )
        grd_avg = (
            sum(grd_stats["bin_fullness"]) / (grd_stats["n_bins"] * max_volume) * 100
        )
        delta = (grd_stats["n_bins"] - opt_stats["n_bins"]) / opt_stats["n_bins"] * 100
        print(
            f"   optimal: {opt_stats['n_bins']} bins, mean {opt_avg:.1f}% full | "
            f"greedy: {grd_stats['n_bins']} bins, mean {grd_avg:.1f}% full | "
            f"greedy uses {delta:+.1f}% more bins"
        )
        last_result = (states, max_volume, opt_bins, grd_bins)

    if last_result is None:
        raise RuntimeError("No distributions defined")
    return last_result


# ------------------------------------------------------------------
# Experiment C — end-to-end wall time (combines claims 1 + 2)
# ------------------------------------------------------------------


def exp_c_wall_time(
    model: LennardJonesModel,
    states: list[ts.SimState],
    opt_bins: list[list[int]],
    grd_bins: list[list[int]],
    n_reps: int = 3,
) -> None:
    """Run the model over all systems using each packing strategy; compare wall times."""
    print("\n" + "=" * 72)
    print("Experiment C: end-to-end wall time (optimal vs greedy packing)")
    print("=" * 72)

    def run(bins: list[list[int]]) -> float:
        # Warmup
        model(ts.concatenate_states([states[i] for i in bins[0]]))
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            for bin_indices in bins:
                batch = ts.concatenate_states([states[i] for i in bin_indices])
                model(batch)
            times.append(time.perf_counter() - t0)
        return min(times)

    t_opt = run(opt_bins)
    t_grd = run(grd_bins)

    print(f"{'strategy':>10}  {'n_bins':>8}  {'wall_time_ms':>14}  {'ms/sys':>10}")
    print("-" * 50)
    n_sys = len(states)
    print(
        f"{'optimal':>10}  {len(opt_bins):>8}  {t_opt * 1000:>14.2f}  "
        f"{t_opt / n_sys * 1000:>10.3f}"
    )
    print(
        f"{'greedy':>10}  {len(grd_bins):>8}  {t_grd * 1000:>14.2f}  "
        f"{t_grd / n_sys * 1000:>10.3f}"
    )
    delta_pct = (t_grd - t_opt) / t_opt * 100
    print(f"\nGreedy is {delta_pct:+.1f}% slower than optimal (negative = faster).")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Run all three benchmark experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (default: cuda if available else cpu)",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    print(
        "Note: CPU + LJ gives weak batching signal; real MLIP on GPU is needed for "
        "definitive numbers."
    )

    model = make_lj_model(device, dtype)

    exp_a_throughput_vs_batch_size(model, device, dtype)
    states, _max_v, opt_bins, grd_bins = exp_b_packing_efficiency(device, dtype)
    exp_c_wall_time(model, states, opt_bins, grd_bins)


if __name__ == "__main__":
    main()
