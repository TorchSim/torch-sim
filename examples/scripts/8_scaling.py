"""Scaling benchmarks for static, relax, NVE, and NVT."""

# %%
# /// script
# dependencies = [
#     "torch_sim_atomistic[mace,test]"
# ]
# ///

import time
import typing

import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls


# Shared constants
N_STRUCTURES_STATIC = [1, 1, 1, 1, 10, 100, 500, 1000, 2500, 5000]
N_STRUCTURES_RELAX = [1, 10, 100, 500]
N_STRUCTURES_NVE = [1, 10, 100, 500]
N_STRUCTURES_NVT = [1, 10, 100, 500]
RELAX_STEPS = 10
MD_STEPS = 10
MAX_MEMORY_SCALER = 400_000
MEMORY_SCALES_WITH = "n_atoms_x_density"


def load_mace_model(device: torch.device) -> MaceModel:
    """Load MACE model for benchmarking."""
    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype="float64",
        device=str(device),
    )
    return MaceModel(
        model=typing.cast("torch.nn.Module", loaded_model),
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=torch.float64,
        enable_cueq=False,
    )


def run_torchsim_static(
    n_structures_list: list[int],
    base_structure: typing.Any,
    model: MaceModel,
    device: torch.device,
) -> list[float]:
    """Run static calculations for each n using batched path, return timings."""
    autobatcher = ts.BinningAutoBatcher(
        model=model,
        max_memory_scaler=MAX_MEMORY_SCALER,
        memory_scales_with=MEMORY_SCALES_WITH,
    )
    times: list[float] = []
    for n in n_structures_list:
        structures = [base_structure] * n
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ts.static(structures, model, autobatcher=autobatcher)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  n={n} static_time={elapsed:.6f}s")
    return times


def run_torchsim_relax(
    n_structures_list: list[int],
    base_structure: typing.Any,
    model: MaceModel,
    device: torch.device,
) -> list[float]:
    """Run relaxation with ts.optimize for each n; return timings."""
    autobatcher = ts.InFlightAutoBatcher(
        model=model,
        max_memory_scaler=MAX_MEMORY_SCALER,
        memory_scales_with=MEMORY_SCALES_WITH,
    )
    times: list[float] = []
    for n in n_structures_list:
        structures = [base_structure] * n
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ts.optimize(
            system=structures,
            model=model,
            optimizer=ts.optimizers.Optimizer.fire,
            init_kwargs={
                "cell_filter": ts.optimizers.cell_filters.CellFilter.frechet,
                "constant_volume": False,
                "hydrostatic_strain": True,
            },
            max_steps=RELAX_STEPS,
            convergence_fn=ts.runners.generate_force_convergence_fn(
                force_tol=1e-3,
                include_cell_forces=True,
            ),
            autobatcher=autobatcher,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  n={n} relax_{RELAX_STEPS}_time={elapsed:.6f}s")
    return times


def run_torchsim_nve(
    n_structures_list: list[int],
    base_structure: typing.Any,
    model: MaceModel,
    device: torch.device,
) -> list[float]:
    """Run NVE MD for MD_STEPS per n; return times."""
    times: list[float] = []
    for n in n_structures_list:
        structures = [base_structure] * n
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ts.integrate(
            system=structures,
            model=model,
            integrator=ts.Integrator.nve,
            n_steps=MD_STEPS,
            temperature=300.0,
            timestep=0.002,
            autobatcher=ts.BinningAutoBatcher(
                model=model,
                max_memory_scaler=MAX_MEMORY_SCALER,
                memory_scales_with=MEMORY_SCALES_WITH,
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  n={n} nve_time={elapsed:.6f}s")
    return times


def run_torchsim_nvt(
    n_structures_list: list[int],
    base_structure: typing.Any,
    model: MaceModel,
    device: torch.device,
) -> list[float]:
    """Run NVT (Nose-Hoover) MD for MD_STEPS per n; return times."""
    times: list[float] = []
    for n in n_structures_list:
        structures = [base_structure] * n
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ts.integrate(
            system=structures,
            model=model,
            integrator=ts.Integrator.nvt_nose_hoover,
            n_steps=MD_STEPS,
            temperature=300.0,
            timestep=0.002,
            autobatcher=ts.BinningAutoBatcher(
                model=model,
                max_memory_scaler=MAX_MEMORY_SCALER,
                memory_scales_with=MEMORY_SCALES_WITH,
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  n={n} nvt_time={elapsed:.6f}s")
    return times


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    base_structure = AseAtomsAdaptor.get_structure(atoms=mgo_ase)

    # Load model once
    model = load_mace_model(device)

    # Run all benchmarks
    print("=== Static benchmark ===")
    static_times = run_torchsim_static(N_STRUCTURES_STATIC, base_structure, model, device)

    print("\n=== Relax benchmark ===")
    relax_times = run_torchsim_relax(N_STRUCTURES_RELAX, base_structure, model, device)

    print("\n=== NVE benchmark ===")
    nve_times = run_torchsim_nve(N_STRUCTURES_NVE, base_structure, model, device)

    print("\n=== NVT benchmark ===")
    nvt_times = run_torchsim_nvt(N_STRUCTURES_NVT, base_structure, model, device)
