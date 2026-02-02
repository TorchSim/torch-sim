"""Scaling for TorchSim relax."""
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


N_STRUCTURES = [1, 1, 1, 10, 100, 500, 1000, 1500]


RELAX_STEPS = 10


def run_torchsim_relax(
    n_structures_list: list[int],
    base_structure: typing.Any,
) -> list[float]:
    """Load TorchSim model once, run 10-step relaxation with ts.optimize for each n;
    return timings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype="float64",
        device=str(device),
    )
    model = MaceModel(
        model=typing.cast("torch.nn.Module", loaded_model),
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=torch.float64,
        enable_cueq=False,
    )
    autobatcher = ts.InFlightAutoBatcher(
        model=model,
        max_memory_scaler=400_000,
        memory_scales_with="n_atoms_x_density",
    )
    times: list[float] = []
    for n in n_structures_list:
        structures = [base_structure] * n
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
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  n={n} relax_{RELAX_STEPS}_time={elapsed:.6f}s")
    return times


if __name__ == "__main__":
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    base_structure = AseAtomsAdaptor.get_structure(atoms=mgo_ase)
    sweep_totals = run_torchsim_relax(N_STRUCTURES, base_structure)
