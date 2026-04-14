"""Calculate the ferroelectric hysteresis loop for BaTiO₃ with allegro-pol."""

from __future__ import annotations

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


MODEL_PATH = "allegro-pol-BaTiO3.nequip.pt2"
STRUCTURE_XYZ_PATH = "BaTiO3.xyz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
MAX_STEPS_INIT = 300
MAX_STEPS_SWEEP = 300
FMAX_TOL = 1e-4
E_MAX_V_PER_ANG = 0.20
N_FIELD_STEPS = 200
e2C = 1.602176634e-19
A2m = 1e-10
m2cm = 1e2
C2uC = 1e6
V2MV = 1e-6
P_CONV = e2C * C2uC * A2m * m2cm
OMEGA_CONV = (A2m * m2cm) ** 3
E_CONV = V2MV / (A2m * m2cm)
PLOT_PDF_PATH = "BaTiO3.pdf"

# Read BaTiO3 structure from xyz file
atoms = ase_read(STRUCTURE_XYZ_PATH)
structure = AseAtomsAdaptor.get_structure(atoms)

# Load Allegro-pol model for BaTiO3
allegro_model = NequIPPolTorchSimCalc.from_compiled_model(
    str(MODEL_PATH), device=DEVICE, chemical_species_to_atom_type_map=True
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
state = ts.initialize_state([structure], combined_model.device, combined_model.dtype)
state._system_extras["external_E_field"] = torch.zeros(
    1, 3, device=state.device, dtype=state.dtype
)
convergence_fn = generate_force_convergence_fn(force_tol=FMAX_TOL)
state = ts.optimize(
    state,
    combined_model,
    optimizer=Optimizer.lbfgs,
    convergence_fn=convergence_fn,
    max_steps=MAX_STEPS_INIT,
)
volume = state.volume[0].cpu().item()

# Relax at sinusoidal Efield
e_values: list[float] = []
pz_values: list[float] = []
phases = np.linspace(0, 2.5 * np.pi, N_FIELD_STEPS, endpoint=True)
e_field_schedule = E_MAX_V_PER_ANG * np.sin(phases)
for idx, e_z in enumerate(e_field_schedule):
    print(f"Step {idx + 1}/{N_FIELD_STEPS}: E_z = {e_z:.4e} V/Å", flush=True)

    state.external_E_field = torch.tensor(
        [[0.0, 0.0, e_z]], device=state.device, dtype=state.dtype
    )

    state = ts.optimize(
        state,
        combined_model,
        optimizer=Optimizer.lbfgs,
        convergence_fn=convergence_fn,
        max_steps=MAX_STEPS_SWEEP,
    )

    pz = state.total_polarization[0, 2].item()
    e_values.append(e_z)
    pz_values.append(pz)
    e_si = e_z * E_CONV
    p_si = pz * P_CONV / (volume * OMEGA_CONV)
    print(
        f"  → E = {e_si:+8.2f} MV/cm   P/Ω = {p_si:+8.2f} μC/cm²",
        flush=True,
    )

# Plot hysteresis loop
E_plot = np.array(e_values) * E_CONV
P_plot = np.array(pz_values) * P_CONV / (volume * OMEGA_CONV)
plt.rcParams.update({"font.size": 24, "legend.fontsize": 14, "legend.handlelength": 0.5})
fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
fig.subplots_adjust(left=0.22, bottom=0.16, top=0.90, right=0.96)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.xaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
ax.xaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
ax.yaxis.set_tick_params(which="major", width=3.0, length=12, direction="in")
ax.yaxis.set_tick_params(which="minor", width=3.0, length=6, direction="in")
ax.yaxis.set_ticks_position("both")
ax.axhline(0, color="black", lw=0.8, zorder=1)
ax.axvline(0, color="black", lw=0.8, zorder=1)
ax.plot(E_plot, P_plot, lw=2.5, color="#0055d4", zorder=100)
ax.set_xlabel(r"Efield (MV$\cdot$cm$^{-1}$)")
ax.set_ylabel(r"P/$\Omega$ ($\mu$C$\cdot$cm$^{-2}$)")
ax.set_title("hysteresis")
with PdfPages(str(PLOT_PDF_PATH)) as pdf:
    pdf.savefig(fig)

print(f"\nHysteresis plot saved to {PLOT_PDF_PATH}", flush=True)
