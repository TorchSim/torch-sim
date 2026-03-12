# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[test]",
# ]
# ///

import os
import statistics
import time
from collections.abc import Callable

import numpy as np
import torch

# Import torch_sim modules
from torch_sim.neighbors.torch_nl import torch_nl_linked_cell, torch_nl_n2


# Try to import vesin - the vesin module now properly detects vesin availability
try:
    from torch_sim.neighbors.vesin import VESIN_AVAILABLE, vesin_nl_ts
except ImportError:
    VESIN_AVAILABLE = False
    vesin_nl_ts = None  # type: ignore[assignment]


def generate_base_structure(
    n_atoms: int,
    device: torch.device,
    density: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a base structure for benchmarking.

    Args:
        n_atoms: Number of atoms in the base structure
        device: Device to create tensors on
        density: Approximate density (atoms per cubic Angstrom)
        seed: Random seed for reproducibility

    Returns:
        (positions, cell, pbc, cutoff)
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    cell_size = (n_atoms / density) ** (1 / 3)
    cell = torch.eye(3, device=device) * cell_size
    positions = torch.rand(n_atoms, 3, device=device, generator=generator) * cell_size
    pbc = torch.tensor([True, True, True], device=device)
    cutoff = torch.tensor(cell_size * 0.05, device=device)
    return positions, cell, pbc, cutoff


def rattle_positions(
    positions: torch.Tensor,
    cell: torch.Tensor,
    rattle_scale: float = 0.1,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Apply random displacements (rattle) to positions.

    Args:
        positions: Original positions [n_atoms, 3]
        cell: Unit cell [3, 3]
        rattle_scale: Scale of random displacements (fraction of cell size)
        seed: Random seed for reproducibility
        device: Device for tensors

    Returns:
        Rattled positions [n_atoms, 3]
    """
    if device is None:
        device = positions.device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    # Get cell size for scaling
    cell_size = torch.norm(cell, dim=0).mean().item()
    # Random displacements
    displacement = (
        torch.randn(positions.shape, device=device, generator=generator)
        * rattle_scale
        * cell_size
    )
    return positions + displacement


def generate_random_system(
    n_atoms: int,
    device: torch.device,
    density: float = 0.1,
    n_systems: int = 1,
    mode: str = "single_large",
    base_atoms: int = 128,
    rattle_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random atomic system(s) for benchmarking.

    Args:
        n_atoms: Total number of atoms
        device: Device to create tensors on
        density: Approximate density (atoms per cubic Angstrom)
        n_systems: Number of systems (for batched testing)
        mode: "single_large" for one large structure,
            "multiple_structures" for concatenated copies
        base_atoms: Base structure size for multiple_structures mode
        rattle_scale: Scale of random displacements for rattling
            (fraction of cell size)

    Returns:
        (positions, cell, pbc, cutoff, system_idx)
    """
    if mode == "single_large":
        # Single large structure - scale up one structure
        if n_systems == 1:
            positions, cell, pbc, cutoff = generate_base_structure(
                n_atoms, device, density, seed=42
            )
            system_idx = torch.zeros(n_atoms, dtype=torch.long, device=device)
        else:
            # Multiple independent systems
            atoms_per_system = n_atoms // n_systems
            all_positions = []
            all_system_idx = []
            cells = []

            for sys_idx in range(n_systems):
                pos, c, _, cut = generate_base_structure(
                    atoms_per_system, device, density, seed=42 + sys_idx
                )
                all_positions.append(pos)
                all_system_idx.append(
                    torch.full(
                        (atoms_per_system,), sys_idx, dtype=torch.long, device=device
                    )
                )
                cells.append(c)

            positions = torch.cat(all_positions, dim=0)
            system_idx = torch.cat(all_system_idx, dim=0)
            cell = torch.stack(cells, dim=0)  # [n_systems, 3, 3]
            pbc = torch.tensor([[True, True, True]] * n_systems, device=device)
            cutoff = torch.tensor(cut.item(), device=device)
    elif mode == "multiple_structures":
        # Multiple rattled copies of the same base structure
        # Create base structure
        base_positions, base_cell, _, base_cutoff = generate_base_structure(
            base_atoms, device, density, seed=42
        )

        # Calculate how many copies we need to reach n_atoms total
        n_copies = max(1, n_atoms // base_atoms)

        all_positions = []
        all_system_idx = []
        cells = []

        for copy_idx in range(n_copies):
            # Rattle each copy with different seed
            rattled_pos = rattle_positions(
                base_positions, base_cell, rattle_scale, seed=42 + copy_idx, device=device
            )
            all_positions.append(rattled_pos)
            all_system_idx.append(
                torch.full((base_atoms,), copy_idx, dtype=torch.long, device=device)
            )
            cells.append(base_cell)

        positions = torch.cat(all_positions, dim=0)
        system_idx = torch.cat(all_system_idx, dim=0)
        cell = torch.stack(cells, dim=0)  # [n_copies, 3, 3]
        pbc = torch.tensor([[True, True, True]] * n_copies, device=device)
        cutoff = base_cutoff
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Use 'single_large' or 'multiple_structures'"
        )

    return positions, cell, pbc, cutoff, system_idx


def benchmark_neighbor_list(
    nl_func: Callable,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    system_idx: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 20,
) -> dict[str, float]:
    """Benchmark a neighbor list function.

    Args:
        nl_func: Neighbor list function to benchmark
        positions: Atomic positions
        cell: Unit cell
        pbc: Periodic boundary conditions
        cutoff: Cutoff radius
        system_idx: System indices
        num_warmup: Number of warmup runs
        num_runs: Number of timing runs

    Returns:
        Dict with timing statistics
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = nl_func(positions, cell, pbc, cutoff, system_idx)

    # Synchronize if CUDA
    if torch.cuda.is_available() and positions.device.type == "cuda":
        torch.cuda.synchronize()

    # Timing runs
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available() and positions.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = nl_func(positions, cell, pbc, cutoff, system_idx)
        if torch.cuda.is_available() and positions.device.type == "cuda":
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


def run_benchmark_suite(
    device: torch.device,
    min_size: int,
    max_size: int,
    steps: int,
    num_warmup: int = 3,
    num_runs: int = 20,
    mode: str = "single_large",
    base_atoms: int = 128,
    rattle_scale: float = 0.1,
) -> None:
    """Run full benchmark suite on a device.

    Args:
        device: Device to run benchmarks on
        min_size: Minimum total atoms
        max_size: Maximum total atoms (shared by both modes)
        steps: Number of size steps (logarithmic spacing)
        num_warmup: Number of warmup runs per benchmark
        num_runs: Number of timing runs per benchmark
        mode: "single_large" for one large structure, "multiple_structures" for concatenated copies
        base_atoms: Base structure size for multiple_structures mode
        rattle_scale: Scale of random displacements for rattling
    """
    mode_str = (
        "Single Large Structure"
        if mode == "single_large"
        else "Multiple Structures (Rattled)"
    )
    print(f"\n{'=' * 100}")
    print(f"Neighbor List Benchmark - {str(device).upper()} - {mode_str}")
    if mode == "multiple_structures":
        print(f"  Base structure size: {base_atoms} atoms")
        print(f"  Rattle scale: {rattle_scale}")
    print(f"{'=' * 100}\n")

    # Generate system sizes (logarithmic spacing)
    if steps == 1:
        sizes = [min_size]
    else:
        sizes = np.logspace(
            np.log10(min_size), np.log10(max_size), steps, dtype=int
        ).tolist()
        sizes = sorted(set(sizes))  # Remove duplicates

    # Results storage
    results_n2 = []
    results_linked = []
    results_vesin = []

    # Build header - always include vesin columns (will show N/A if not available)
    header = (
        f"{'Size':<8} {'N² Mean (ms)':<15} {'N² Std (ms)':<15} "
        f"{'Linked Mean (ms)':<18} {'Linked Std (ms)':<18} "
        f"{'Vesin Mean (ms)':<18} {'Vesin Std (ms)':<18} "
        f"{'Winner':<15} {'Speedup':<15}"
    )
    print(header)
    print("-" * len(header))

    for n_atoms in sizes:
        # Generate system(s)
        positions, cell, pbc, cutoff, system_idx = generate_random_system(
            n_atoms, device, mode=mode, base_atoms=base_atoms, rattle_scale=rattle_scale
        )

        # Benchmark N²
        stats_n2 = None
        try:
            stats_n2 = benchmark_neighbor_list(
                torch_nl_n2,
                positions,
                cell,
                pbc,
                cutoff,
                system_idx,
                num_warmup=num_warmup,
                num_runs=num_runs,
            )
        except Exception as e:
            print(f"Error benchmarking N² for size {n_atoms}: {e}")

        # Benchmark linked cell
        stats_linked = None
        try:
            stats_linked = benchmark_neighbor_list(
                torch_nl_linked_cell,
                positions,
                cell,
                pbc,
                cutoff,
                system_idx,
                num_warmup=num_warmup,
                num_runs=num_runs,
            )
        except Exception as e:
            print(f"Error benchmarking linked cell for size {n_atoms}: {e}")

        # Benchmark vesin (always try if available)
        stats_vesin = None
        if VESIN_AVAILABLE and vesin_nl_ts is not None:
            try:
                stats_vesin = benchmark_neighbor_list(
                    vesin_nl_ts,  # type: ignore[arg-type]
                    positions,
                    cell,
                    pbc,
                    cutoff,
                    system_idx,
                    num_warmup=num_warmup,
                    num_runs=num_runs,
                )
            except Exception:
                # Vesin failed - will show as N/A in output
                pass

        # Collect all valid results
        all_stats = [
            ("N²", stats_n2),
            ("Linked", stats_linked),
        ]
        # Add vesin if it was successfully benchmarked
        if stats_vesin is not None:
            all_stats.append(("Vesin", stats_vesin))
        all_stats = [s for s in all_stats if s[1] is not None]

        if not all_stats:
            print(f"{n_atoms:<8} {'All failed':<15}")
            continue

        # Find fastest
        fastest_name = min(all_stats, key=lambda x: x[1]["mean"])[0]  # type: ignore[index]
        fastest_time = min(s[1]["mean"] for s in all_stats)  # type: ignore[index]

        # Format output
        mean_n2 = stats_n2["mean"] * 1000 if stats_n2 is not None else None
        std_n2 = stats_n2["std"] * 1000 if stats_n2 is not None else None
        mean_linked = stats_linked["mean"] * 1000 if stats_linked is not None else None
        std_linked = stats_linked["std"] * 1000 if stats_linked is not None else None
        mean_vesin = stats_vesin["mean"] * 1000 if stats_vesin is not None else None
        std_vesin = stats_vesin["std"] * 1000 if stats_vesin is not None else None

        # Calculate speedup relative to fastest
        speedup_str = "1.00x"
        if len(all_stats) > 1:
            slowest_time = max(s[1]["mean"] for s in all_stats)  # type: ignore[index]
            speedup = slowest_time / fastest_time
            speedup_str = f"{speedup:.2f}x"

        # Print row - always include vesin columns
        row = f"{n_atoms:<8} "
        row += f"{mean_n2:<15.3f} " if mean_n2 else f"{'N/A':<15} "
        row += f"{std_n2:<15.3f} " if std_n2 else f"{'N/A':<15} "
        row += f"{mean_linked:<18.3f} " if mean_linked else f"{'N/A':<18} "
        row += f"{std_linked:<18.3f} " if std_linked else f"{'N/A':<18} "
        row += f"{mean_vesin:<18.3f} " if mean_vesin else f"{'N/A':<18} "
        row += f"{std_vesin:<18.3f} " if std_vesin else f"{'N/A':<18} "
        row += f"{fastest_name:<15} {speedup_str:<15}"
        print(row)

        # Store results
        if stats_n2:
            results_n2.append((n_atoms, mean_n2))
        if stats_linked:
            results_linked.append((n_atoms, mean_linked))
        if stats_vesin:
            results_vesin.append((n_atoms, mean_vesin))

    print("-" * len(header))
    print()

    # Summary
    if len(results_n2) > 0 and len(results_linked) > 0:
        # Find crossover point between N² and linked cell
        crossover = None
        for i in range(min(len(results_n2), len(results_linked)) - 1):
            if i + 1 >= len(results_n2) or i + 1 >= len(results_linked):
                break
            n_atoms_curr = results_n2[i][0]
            n_atoms_next = results_n2[i + 1][0]
            time_n2_curr = results_n2[i][1]
            time_linked_curr = results_linked[i][1]
            time_n2_next = results_n2[i + 1][1]
            time_linked_next = results_linked[i + 1][1]

            if time_n2_curr < time_linked_curr and time_n2_next > time_linked_next:
                ratio = (time_linked_curr - time_n2_curr) / (
                    (time_n2_next - time_n2_curr) - (time_linked_next - time_linked_curr)
                )
                crossover = int(n_atoms_curr + ratio * (n_atoms_next - n_atoms_curr))
                break

        if crossover:
            print(f"Estimated N² ↔ Linked crossover: ~{crossover} atoms/system")
        print()


# %%
# Configuration
# Quick test mode - automatically enabled in CI
QUICK_TEST = os.getenv("CI") is not None

if QUICK_TEST:
    MIN_SIZE = 10
    MAX_SIZE = 100
    STEPS = 3
    NUM_WARMUP = 1
    NUM_RUNS = 3
else:
    MIN_SIZE = 10
    MAX_SIZE = 5000
    STEPS = 10
    NUM_WARMUP = 3
    NUM_RUNS = 20

# Benchmark mode: "single_large" or "multiple_structures"
MODE = "multiple_structures"  # "single_large" for one large structure, "multiple_structures" for concatenated copies
BASE_ATOMS = 128  # Base structure size for multiple_structures mode
RATTLE_SCALE = 0.1  # Scale of random displacements for rattling (fraction of cell size)
CPU_ONLY = False  # Set to True to only run CPU benchmarks
GPU_ONLY = False  # Set to True to only run GPU benchmarks


# %%
def main() -> None:
    """Main benchmark function."""
    print("=" * 100)
    print("Neighbor List Implementation Benchmark")
    if QUICK_TEST:
        print("QUICK TEST MODE - Fast verification only (CI detected)")
    print("=" * 100)
    implementations = (
        "torch_nl_n2 (O(n²)), torch_nl_linked_cell (optimized), vesin_nl_ts (Vesin)"
    )
    if not VESIN_AVAILABLE:
        implementations += " [vesin not available - will show N/A]"
    else:
        implementations += " [vesin available]"
    print(f"Comparing: {implementations}")
    print(f"Size range: {MIN_SIZE} - {MAX_SIZE} total atoms ({STEPS} steps)")
    print(f"Mode: {MODE}")
    if MODE == "multiple_structures":
        print(f"  Base structure: {BASE_ATOMS} atoms")
        print(f"  Rattle scale: {RATTLE_SCALE}")
    print()

    # Determine which devices to test
    devices = []
    if not GPU_ONLY:
        devices.append(torch.device("cpu"))
    if not CPU_ONLY and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif GPU_ONLY and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA not available. Skipping GPU benchmarks.")
        return

    # Run benchmarks on each device
    for device in devices:
        run_benchmark_suite(
            device,
            MIN_SIZE,
            MAX_SIZE,
            STEPS,
            num_warmup=NUM_WARMUP,
            num_runs=NUM_RUNS,
            mode=MODE,
            base_atoms=BASE_ATOMS,
            rattle_scale=RATTLE_SCALE,
        )

    print("=" * 100)
    print("Benchmark complete!")
    print("=" * 100)


# %%
# Run benchmarks
if __name__ == "__main__":
    main()
