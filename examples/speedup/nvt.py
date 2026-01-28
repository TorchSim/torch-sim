"""Speedup comparison TorchSim vs ASE for NVT MD with Nose-Hoover thermostat (10 steps).

To run:
uv sync --extra mace --extra test
uv run nvt.py
"""

# pyright: basic

import io
import time
import typing
import warnings

import plotly.graph_objects as go
import torch
from ase import units as ase_units
from ase.build import bulk
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators.foundations_models import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor

import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls


warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level "
        "annotations on empty non-base types"
    ),
    category=UserWarning,
    module="torch.jit._check",
)

MD_STEPS = 10


def run_torchsim_nvt(
    n_structures_list: list[int],
    iterations: int,
    base_structure: typing.Any,
) -> list[float]:
    """Load TorchSim model, run NVT N-H MD for MD_STEPS per n/iter, return timings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype="float64",
        device=str(device),
    )
    max_memory_scaler = 400_000
    memory_scales_with = "n_atoms_x_density"
    model = MaceModel(
        model=typing.cast("torch.nn.Module", loaded_model),
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=torch.float64,
        enable_cueq=False,
    )
    times: list[float] = []
    for it in range(iterations):
        print("TorchSim iteration", it + 1)
        for n in n_structures_list:
            structures = [base_structure] * n
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
                    max_memory_scaler=max_memory_scaler,
                    memory_scales_with=memory_scales_with,
                ),
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  n={n} nvt_time={elapsed:.6f}s")
    return times


def run_ase_nvt(
    n_structures_list: list[int],
    iterations: int,
    mgo_ase: typing.Any,
) -> list[float]:
    """Load ASE calculator, run NVT N-H MD for MD_STEPS per n/iter, return timings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ase_calc = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        default_dtype="float64",
        device=str(device),
        enable_cueq=False,
    )
    dt_fs = 2.0
    tdamp_fs = 100.0 * dt_fs
    temperature_k = 300.0
    times: list[float] = []
    for it in range(iterations):
        print("ASE iteration", it + 1)
        for n in n_structures_list:
            ase_atoms_list = [mgo_ase.copy() for _ in range(n)]
            for at in ase_atoms_list:
                at.calc = ase_calc
                MaxwellBoltzmannDistribution(atoms=at, temperature_K=temperature_k)
            t0 = time.perf_counter()
            for at in ase_atoms_list:
                dyn = NoseHooverChainNVT(
                    at,
                    timestep=dt_fs * ase_units.fs,
                    temperature_K=temperature_k,
                    tdamp=tdamp_fs * ase_units.fs,
                    tchain=3,
                    tloop=1,
                    logfile=io.StringIO(),
                )
                dyn.run(MD_STEPS)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  n={n} ase_nvt_time={elapsed:.6f}s")
    return times


def plot_results(
    n_structures_list: list[int],
    iterations: int,
    ts_times: list[float],
    ase_times: list[float],
    output_path: str = "speedup_nvt.html",
) -> None:
    """Plot TorchSim and ASE NVT (Nose-Hoover) MD timings per iter and save to HTML."""
    n = len(n_structures_list)
    fig = go.Figure()
    for it in range(iterations):
        start, end = it * n, (it + 1) * n
        ts_slice = ts_times[start:end]
        ase_slice = ase_times[start:end]
        dash = "dash" if it == 0 else "solid"
        symbol = "square" if it == 0 else "circle"
        fig.add_trace(
            go.Scatter(
                x=n_structures_list,
                y=ts_slice,
                mode="lines+markers",
                marker={"size": 10, "symbol": symbol, "color": "red"},
                line={"width": 2, "dash": dash, "color": "red"},
                name=f"TorchSim (iter {it + 1})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=n_structures_list,
                y=ase_slice,
                mode="lines+markers",
                marker={"size": 10, "symbol": symbol, "color": "blue"},
                line={"width": 2, "dash": dash, "color": "blue"},
                name=f"ASE (iter {it + 1})",
            )
        )
    fig.update_layout(
        title=f"NVT (Nose-Hoover) MD timings ({MD_STEPS} steps)",
        xaxis_title="n_structures",
        yaxis_title="Time (s)",
        legend={"x": 0.01, "y": 0.99},
    )
    fig.write_html(output_path)


if __name__ == "__main__":
    # MgO rocksalt structure
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    base_structure = AseAtomsAdaptor.get_structure(atoms=mgo_ase)  # pyright: ignore[reportArgumentType]

    # Statistics sizes
    iterations = 2
    n_structures_list: list[int] = [1, 10, 100, 500]

    ts_times = run_torchsim_nvt(n_structures_list, iterations, base_structure)
    ase_times = run_ase_nvt(n_structures_list, iterations, mgo_ase)

    plot_results(
        n_structures_list,
        iterations,
        ts_times,
        ase_times,
        output_path="nvt.html",
    )
