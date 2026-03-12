"""Benchmark script to compare JIT compiled vs eager performance for n-body functions."""

# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[test]"
# ]
# ///

import statistics
import time
from collections.abc import Callable

import torch

from torch_sim.neighbors.nbody import (
    build_mixed_triplets,
    build_quadruplets,
    build_triplets,
)


def benchmark_function(
    func: Callable,
    args: tuple,
    kwargs: dict | None = None,
    num_warmup: int = 3,
    num_runs: int = 20,
) -> dict[str, float]:
    """Benchmark a function with warmup runs.

    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        num_warmup: Number of warmup runs
        num_runs: Number of timing runs

    Returns:
        Dict with 'mean', 'median', 'std', 'min', 'max' times in seconds
    """
    if kwargs is None:
        kwargs = {}

    # Warmup runs
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)

    # Synchronize if CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timing runs
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


# %%
# Configuration
USE_LARGE = False  # Set to True for larger test cases

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def main() -> None:
    """Run benchmarks for all n-body functions."""
    print("=" * 80)
    print("N-Body Functions JIT Benchmark")
    print("=" * 80)
    print()

    print(f"Device: {device}")
    print()

    # Test 1: build_triplets
    print("-" * 80)
    print("Test 1: build_triplets")
    print("-" * 80)
    if USE_LARGE:
        # Larger case: many edges to one atom
        n_edges = 100
        edge_index = torch.stack(
            [
                torch.arange(1, n_edges + 1, device=device),
                torch.zeros(n_edges, device=device, dtype=torch.long),
            ]
        )
        n_atoms = n_edges + 1
    else:
        edge_index = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]], device=device)
        n_atoms = 7

    # Eager
    eager_stats = benchmark_function(build_triplets, (edge_index, n_atoms))
    print("Eager execution:")
    print(f"  Mean:   {eager_stats['mean'] * 1000:.3f} ms")
    print(f"  Median: {eager_stats['median'] * 1000:.3f} ms")
    print(f"  Std:    {eager_stats['std'] * 1000:.3f} ms")

    # JIT
    jit_func = torch.jit.script(build_triplets)
    jit_stats = benchmark_function(jit_func, (edge_index, n_atoms))
    print("JIT execution:")
    print(f"  Mean:   {jit_stats['mean'] * 1000:.3f} ms")
    print(f"  Median: {jit_stats['median'] * 1000:.3f} ms")
    print(f"  Std:    {jit_stats['std'] * 1000:.3f} ms")

    speedup = eager_stats["mean"] / jit_stats["mean"]
    print(f"Speedup: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
    print()

    # Test 2: build_mixed_triplets
    print("-" * 80)
    print("Test 2: build_mixed_triplets")
    print("-" * 80)
    if USE_LARGE:
        # Larger case: more edges
        n_in = 50
        n_out = 30
        edge_index_in = torch.stack(
            [
                torch.arange(n_in, device=device),
                torch.randint(4, 10, (n_in,), device=device),
            ]
        )
        edge_index_out = torch.stack(
            [
                torch.randint(0, 5, (n_out,), device=device),
                torch.randint(4, 10, (n_out,), device=device),
            ]
        )
        n_atoms = 15
    else:
        edge_index_in = torch.tensor([[0, 1, 3, 5], [4, 4, 5, 6]], device=device)
        edge_index_out = torch.tensor([[2, 2, 3], [4, 5, 6]], device=device)
        n_atoms = 7

    # Eager
    eager_stats = benchmark_function(
        build_mixed_triplets,
        (edge_index_in, edge_index_out, n_atoms),
        {"to_outedge": False},
    )
    print("Eager execution:")
    print(f"  Mean:   {eager_stats['mean'] * 1000:.3f} ms")
    print(f"  Median: {eager_stats['median'] * 1000:.3f} ms")
    print(f"  Std:    {eager_stats['std'] * 1000:.3f} ms")

    # JIT - need wrapper for keyword args
    def wrapper_mixed(
        edge_index_in: torch.Tensor,
        edge_index_out: torch.Tensor,
        n_atoms: int,
    ) -> dict[str, torch.Tensor]:
        return build_mixed_triplets(
            edge_index_in, edge_index_out, n_atoms, to_outedge=False
        )

    jit_func = torch.jit.script(wrapper_mixed)
    jit_stats = benchmark_function(jit_func, (edge_index_in, edge_index_out, n_atoms))
    print("JIT execution:")
    print(f"  Mean:   {jit_stats['mean'] * 1000:.3f} ms")
    print(f"  Median: {jit_stats['median'] * 1000:.3f} ms")
    print(f"  Std:    {jit_stats['std'] * 1000:.3f} ms")

    speedup = eager_stats["mean"] / jit_stats["mean"]
    print(f"Speedup: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
    print()

    # Test 3: build_quadruplets
    print("-" * 80)
    print("Test 3: build_quadruplets")
    print("-" * 80)
    if USE_LARGE:
        # Larger case: more edges
        n_main = 40
        n_qint = 15
        main_edge_index = torch.stack(
            [
                torch.randint(0, 10, (n_main,), device=device),
                torch.randint(0, 10, (n_main,), device=device),
            ]
        )
        qint_edge_index = torch.stack(
            [
                torch.randint(0, 10, (n_qint,), device=device),
                torch.randint(0, 10, (n_qint,), device=device),
            ]
        )
        n_atoms = 12
        main_cell_offsets = torch.zeros(n_main, 3, device=device)
        qint_cell_offsets = torch.zeros(n_qint, 3, device=device)
    else:
        main_edge_index = torch.tensor([[0, 2, 1, 1, 3], [1, 1, 3, 4, 5]], device=device)
        qint_edge_index = torch.tensor([[1, 3], [3, 5]], device=device)
        n_atoms = 6
        main_cell_offsets = torch.zeros(5, 3, device=device)
        qint_cell_offsets = torch.zeros(2, 3, device=device)

    # Eager
    eager_stats = benchmark_function(
        build_quadruplets,
        (
            main_edge_index,
            qint_edge_index,
            n_atoms,
            main_cell_offsets,
            qint_cell_offsets,
        ),
    )
    print("Eager execution:")
    print(f"  Mean:   {eager_stats['mean'] * 1000:.3f} ms")
    print(f"  Median: {eager_stats['median'] * 1000:.3f} ms")
    print(f"  Std:    {eager_stats['std'] * 1000:.3f} ms")

    # JIT
    jit_func = torch.jit.script(build_quadruplets)
    jit_stats = benchmark_function(
        jit_func,
        (
            main_edge_index,
            qint_edge_index,
            n_atoms,
            main_cell_offsets,
            qint_cell_offsets,
        ),
    )
    print("JIT execution:")
    print(f"  Mean:   {jit_stats['mean'] * 1000:.3f} ms")
    print(f"  Median: {jit_stats['median'] * 1000:.3f} ms")
    print(f"  Std:    {jit_stats['std'] * 1000:.3f} ms")

    speedup = eager_stats["mean"] / jit_stats["mean"]
    print(f"Speedup: {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
    print()

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


# %%
# Run benchmarks
if __name__ == "__main__":
    main()
