# /// script
# requires-python = ">=3.12"
# dependencies = ["allegro-pol", "torch-sim-atomistic"]
# [tool.uv.sources]
# torch-sim-atomistic = { path = "../..", editable = true }
# allegro-pol = { git = "https://github.com/mir-group/allegro-pol" }
# ///
"""Calculate the static dielectric constant of alpha-quartz SiO2."""

from __future__ import annotations

from pathlib import Path

import cuequivariance_torch  # noqa: F401 — registers cuequivariance custom ops
import matplotlib
import numpy as np
import torch
from allegro_pol.integrations.torchsim import NequIPPolTorchSimCalc
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

import torch_sim as ts
from torch_sim.models.interface import SerialSumModel
from torch_sim.models.polarization import UniformPolarizationModel
from torch_sim.optimizers import Optimizer
from torch_sim.runners import generate_force_convergence_fn
from torch_sim.trajectory import TorchSimTrajectory, TrajectoryReporter


MODEL_PATH = "allegro-pol-SiO2.nequip.pt2"
STRUCTURE_XYZ_PATH = "SiO2.xyz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
EV_PER_ANG2_TO_INV_EPSILON0 = 5.5263499562e-3
MAX_STEPS = 500
FMAX_TOL = 1e-4
E_FIELD_V_PER_ANG = 1e-3  # V/Å along z
PLOT_PDF_PATH = "SiO2.pdf"
TRAJ_PATH = "efield_relax.h5"


# Read SiO2 structure from xyz file
atoms = ase_read(STRUCTURE_XYZ_PATH)
structure = AseAtomsAdaptor.get_structure(atoms)

# Load Allegro-pol model for SiO2
allegro_model = NequIPPolTorchSimCalc.from_compiled_model(
    MODEL_PATH, device=DEVICE, chemical_species_to_atom_type_map=True
)
field_model = UniformPolarizationModel(
    device=allegro_model.device,
    dtype=allegro_model.dtype,
    compute_forces=True,
    compute_stress=True,
)
combined_model = SerialSumModel(
    allegro_model,
    field_model,
    key_map={
        "polarization": "total_polarization",
        "born_charges": "born_effective_charges",
    },
)

# Relax at Efield = 0
print("Relaxing structure at Efield = [0.0, 0.0, 0.0]", flush=True)
state_e0 = ts.initialize_state([structure], combined_model.device, combined_model.dtype)
state_e0._system_extras["external_E_field"] = torch.zeros(
    1, 3, device=state_e0.device, dtype=state_e0.dtype
)
convergence_fn = generate_force_convergence_fn(force_tol=FMAX_TOL)
result_e0 = ts.optimize(
    state_e0,
    combined_model,
    optimizer=Optimizer.lbfgs,
    convergence_fn=convergence_fn,
    max_steps=MAX_STEPS,
)
P0 = result_e0.total_polarization.clone()
V0 = result_e0.volume[0].cpu().item()
alpha = result_e0.polarizability[0].cpu().numpy()
eps_electronic = np.eye(3) + alpha / (EV_PER_ANG2_TO_INV_EPSILON0 * V0)

# Relax at Efield = [0.0, 0.0, Ez]
print(f"Relaxing structure at Efield = [0.0, 0.0, {E_FIELD_V_PER_ANG}]", flush=True)
state_ef = ts.SimState(
    positions=result_e0.positions.clone(),
    masses=result_e0.masses.clone(),
    cell=result_e0.cell.clone(),
    pbc=result_e0.pbc,
    atomic_numbers=result_e0.atomic_numbers.clone(),
    system_idx=result_e0.system_idx.clone(),
)
state_ef._system_extras["external_E_field"] = torch.tensor(
    [[0.0, 0.0, E_FIELD_V_PER_ANG]], device=state_ef.device, dtype=state_ef.dtype
)
reporter = TrajectoryReporter(
    filenames=[TRAJ_PATH],
    state_frequency=0,
    prop_calculators={
        1: {
            "polarization": lambda state: state.total_polarization,
            "volume": lambda state: state.volume,
        }
    },
    trajectory_kwargs={"coerce_to_float32": False},
)
ts.optimize(
    state_ef,
    combined_model,
    optimizer=Optimizer.lbfgs,
    convergence_fn=convergence_fn,
    max_steps=MAX_STEPS,
    trajectory_reporter=reporter,
    init_kwargs={"alpha": 70.0, "step_size": 1.0},
)

# Calculate dielectric constant throughout relaxaton
P0_z = P0[0, 2].item()
with TorchSimTrajectory(TRAJ_PATH, mode="r") as traj:
    pol_history = traj.get_array("polarization")
    V_history = traj.get_array("volume")
pol_z = pol_history[:, 0, 2]
V_flat = V_history[:, 0]
denom_t = E_FIELD_V_PER_ANG * EV_PER_ANG2_TO_INV_EPSILON0 * V_flat
eps_static_z = 1.0 + (pol_z - P0_z) / denom_t

# Plot dielectric constant throughout relaxaton
steps = np.arange(len(eps_static_z))
plt.rcParams.update({"font.size": 24})
fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
fig.subplots_adjust(left=0.22, bottom=0.16, top=0.90, right=0.96)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_minor_locator(AutoMinorLocator(5))
    axis.set_tick_params(which="major", width=3.0, length=12, direction="in")
    axis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
ax.yaxis.set_ticks_position("both")
ax.plot(steps, eps_static_z, lw=2.5, color="#1f77b4", marker="o", markersize=8)
ax.annotate(
    r"$\varepsilon^{\infty}_{zz}$" + f" = {eps_electronic[2, 2]:.2f}",
    xy=(0, eps_static_z[0]),
    xytext=(8, 12),
    textcoords="offset points",
    fontsize=14,
    color="black",
)
ax.annotate(
    r"$\varepsilon^{0}_{zz}$" + f" = {eps_static_z[-1]:.2f}",
    xy=(steps[-1], eps_static_z[-1]),
    xytext=(-8, -18),
    textcoords="offset points",
    fontsize=14,
    color="black",
    ha="right",
)
ax.set_xlabel("Relaxation step")
ax.set_ylabel(r"$\varepsilon^{0}_{zz}$")
ax.set_title("Dielectric Constant")
with PdfPages(str(PLOT_PDF_PATH)) as pdf:
    pdf.savefig(fig)

# Print dielectric constants
eps_inf = float(eps_electronic[2, 2])
eps_0 = float(eps_static_z[-1])
print(f"eps_inf = {round(eps_inf, 2)}")
print(f"eps_0 = {round(eps_0, 2)}")
print(f"\nDielectric constant plot saved to {PLOT_PDF_PATH}")

# Delete trajectory file
Path(TRAJ_PATH).unlink()
