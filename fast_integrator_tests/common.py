"""Shared constants and helpers for integrator validation scripts."""

import numpy as np
import torch
from ase.build import bulk
from pathlib import Path

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import MetalUnits

DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print(f"Using device: {DEVICE}")

# LJ Argon
SIGMA = 3.405
EPSILON = 0.0104
CUTOFF = 2.5 * SIGMA

DATA_DIR = Path(__file__).parent / "data"
torch.set_num_threads(4)

def make_lj_model(compute_stress=False):
    return LennardJonesModel(
        use_neighbor_list=False,
        sigma=SIGMA,
        epsilon=EPSILON,
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=compute_stress,
        cutoff=CUTOFF,
    )


def make_ar_supercell(repeat=(2, 2, 2)):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(repeat)
    return ts.io.atoms_to_state(atoms, DEVICE, DTYPE)


def to_kT(temperature_K):
    return temperature_K * float(MetalUnits.temperature)


def to_dt(timestep_ps):
    return timestep_ps * float(MetalUnits.time)
