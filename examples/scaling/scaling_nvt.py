"""Scaling for TorchSim NVT (Nose-Hoover)."""
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


N_STRUCTURES = [1, 1, 1, 10, 100, 500, 1000, 1500, 5000, 10000]


MD_STEPS = 10


def run_torchsim_nvt(
    n_structures_list: list[int],
    base_structure: typing.Any,
) -> list[float]:
    """Load model, run NVT MD for MD_STEPS per n; return times."""
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
                max_memory_scaler=max_memory_scaler,
                memory_scales_with=memory_scales_with,
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  n={n} nvt_time={elapsed:.6f}s")
    return times


if __name__ == "__main__":
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    base_structure = AseAtomsAdaptor.get_structure(atoms=mgo_ase)
    sweep_totals = run_torchsim_nvt(N_STRUCTURES, base_structure=base_structure)
