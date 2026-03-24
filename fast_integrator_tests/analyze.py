# %% [markdown]
# # Physical Validation of torch-sim Integrators
# 
# This notebook analyzes MD trajectory data produced by `run_nvt.py`, `run_npt.py`, and `run_nve.py`.
# 
# **Tests performed:**
# 1. **KE Distribution** — Does kinetic energy follow the Maxwell-Boltzmann (gamma) distribution?
# 2. **Ensemble Check** — Do energy distributions at two temperatures satisfy the Boltzmann weight relationship?
# 3. **Pressure Ensemble Check** — Do volume distributions at different pressures satisfy the expected Boltzmann weight relationship?
# 4. **NVE Convergence** — Does the energy drift RMSD scale as dt² (velocity Verlet)?
# 5. **Time Series** — Visual inspection of temperature, energy, and volume stability.

# %%
# Create a folder to save plot
import os
os.makedirs("plots", exist_ok=True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

DATA_DIR = Path("fast_integrator_tests/data")

def load(name):
    """Load an npz file from the data directory."""
    return dict(np.load(DATA_DIR / f"{name}.npz", allow_pickle=True))

# Check what data is available
available = sorted(p.stem for p in DATA_DIR.glob("*.npz"))
print("Available datasets:")
for a in available:
    print(f"  {a}")

# %%
import physical_validation
from torch_sim.units import MetalUnits

k_B_eV = float(MetalUnits.temperature)  # 8.617333e-5 eV/K
THRESHOLD = 3.0

# def make_unit_data():
#     return physical_validation.data.UnitData(
#         kb=k_B_eV,
#         energy_str="eV", energy_conversion=1.0,
#         length_str="Ang", length_conversion=1.0,
#         volume_str="Ang^3", volume_conversion=1.0,
#         temperature_str="K", temperature_conversion=1.0,
#         pressure_str="eV/Ang^3", pressure_conversion=1.0,
#         time_str="internal", time_conversion=1.0,
#     )
def make_unit_data():
    return physical_validation.data.UnitData(
        kb=k_B_eV,
        energy_str="eV", energy_conversion=96.485,  # Convert to kJ/mol
        length_str="Ang", length_conversion=1e-1,  # Convert to nm
        volume_str="Ang^3", volume_conversion=1e-3,  # Convert to nm^3
        temperature_str="K", temperature_conversion=1.0,
        # pressure_str="eV/Ang^3", pressure_conversion=1.6e6,  # Convert to bar
        pressure_str="bar", pressure_conversion=1,
        time_str="fs", time_conversion=1e-3,  # Convert to ps
    )

def build_sim_data(d, temperature, ensemble="NVT", pressure=None):
    """Build physical_validation.SimulationData from a loaded npz dict."""
    units = make_unit_data()
    natoms = int(d["natoms"])
    system = physical_validation.data.SystemData(
        natoms=natoms, nconstraints=0,
        ndof_reduction_tra=3, ndof_reduction_rot=0,
        mass=d["masses"],
    )
    ens_kw = dict(natoms=natoms, temperature=temperature)
    if ensemble == "NVT":
        ens_kw["ensemble"] = "NVT"
        ens_kw["volume"] = float(d.get("volume", np.mean(d.get("volumes", [0]))))
    else:
        ens_kw["ensemble"] = "NPT"
        ens_kw["pressure"] = pressure if pressure is not None else 0.0

    obs_kw = dict(
        kinetic_energy=d["kinetic_energy"],
        potential_energy=d["potential_energy"],
        total_energy=d["total_energy"],
    )
    if "volumes" in d:
        obs_kw["volume"] = d["volumes"]

    return physical_validation.data.SimulationData(
        units=units,
        dt=float(d["dt_internal"]),
        system=system,
        ensemble=physical_validation.data.EnsembleData(**ens_kw),
        observables=physical_validation.data.ObservableData(**obs_kw),
    )

print("physical_validation helpers loaded")

# %% [markdown]
# ## 1. NVT Time Series
# 
# Temperature and energy vs step for each NVT integrator at both temperatures.

# %%
NVT_INTEGRATORS = ["nvt_langevin", "nvt_nose_hoover", "nvt_vrescale"]
TEMPS = [88.0, 100.0]

fig, axes = plt.subplots(len(NVT_INTEGRATORS), 2, figsize=(14, 3.5 * len(NVT_INTEGRATORS)),
                         squeeze=False, sharex=True)
fig.suptitle("NVT Integrators — Time Series", fontsize=14, y=1.01)

for row, name in enumerate(NVT_INTEGRATORS):
    for temp in TEMPS:
        label = f"{name}_T{temp:.0f}K"
        try:
            d = load(label)
        except FileNotFoundError:
            continue

        steps = np.arange(len(d["temperature"]))
        target_T = float(d["target_temperature"])

        # Temperature
        ax = axes[row, 0]
        ax.plot(steps, d["temperature"], alpha=0.5, lw=0.5, label=f"T={target_T:.0f}K")
        ax.axhline(target_T, color="k", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel("Temperature (K)")
        ax.set_title(name)
        ax.legend(fontsize=8)

        # Total energy
        ax = axes[row, 1]
        ax.plot(steps, d["total_energy"], alpha=0.5, lw=0.5, label=f"T={target_T:.0f}K")
        ax.set_ylabel("Total Energy (eV)")
        ax.set_title(name)
        ax.legend(fontsize=8)

axes[-1, 0].set_xlabel("Step")
axes[-1, 1].set_xlabel("Step")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. KE Distribution — NVT Integrators
# 
# For each integrator at 100K, compare the observed KE distribution against the theoretical gamma distribution.
# The KE of an ideal gas with $N_f$ degrees of freedom at temperature $T$ follows $\text{Gamma}(N_f/2, k_BT)$.

# %%
fig, axes = plt.subplots(1, len(NVT_INTEGRATORS), figsize=(5 * len(NVT_INTEGRATORS), 4),
                         sharey=True)
fig.suptitle("KE Distribution vs Maxwell-Boltzmann (NVT, T=100K)", fontsize=13)

for ax, name in zip(axes, NVT_INTEGRATORS):
    label = f"{name}_T100K"
    try:
        d = load(label)
    except FileNotFoundError:
        ax.set_title(f"{name}\n(no data)")
        continue

    ke = d["kinetic_energy"]
    natoms = int(d["natoms"])
    target_T = float(d["target_temperature"])

    # Degrees of freedom: 3*N - 3 (COM removed)
    if name == "nvt_langevin":
        ndof = 3 * natoms
    else:
        ndof = 3 * natoms - 3
    # Theoretical gamma: shape=ndof/2, scale=k_B*T
    shape = ndof / 2
    scale = k_B_eV * target_T

    # Histogram of observed KE
    ax.hist(ke, bins=60, density=True, alpha=0.6, color="steelblue", label="Observed")

    # Theoretical curve
    x = np.linspace(ke.min(), ke.max(), 300)
    pdf = stats.gamma.pdf(x, a=shape, scale=scale)
    ax.plot(x, pdf, "r-", lw=2, label="Theory")

    # Stats
    d_mean = (ke.mean() - shape * scale) / (scale * np.sqrt(2 / ndof))
    d_width = (ke.std() - scale * np.sqrt(shape)) / (scale * np.sqrt(0.5))
    ax.set_title(f"{name}\nd_mean={d_mean:.2f}σ  d_width={d_width:.2f}σ", fontsize=10)
    ax.set_xlabel("Kinetic Energy (eV)")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Probability Density")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### physical_validation built-in KE distribution plots (NVT)
# 
# Uses `physical_validation.kinetic_energy.distribution(..., screen=True)` which overlays the observed and theoretical distributions with a KS-test or mean/width comparison.

# %%
for name in NVT_INTEGRATORS:
    label = f"{name}_T100K"
    try:
        d = load(label)
    except FileNotFoundError:
        print(f"{name}: no data")
        continue

    print(f"\n{'='*60}")
    print(f"  {name} — KE distribution (physical_validation)")
    print(f"{'='*60}")
    sd = build_sim_data(d, 100.0, ensemble="NVT")
    result = physical_validation.kinetic_energy.distribution(
        sd, strict=False, screen=True, verbosity=2, filename=f"plots/{name}_ke_distribution.png"
    )
    print(f"  Result: d_mean={result[0]:.3f}σ, d_width={result[1]:.3f}σ")

plt.show()

# %% [markdown]
# ## 3. Ensemble Check — NVT Integrators
# 
# For each integrator, compare total energy distributions at T_low=88K and T_high=100K.
# The log ratio of the energy histograms should be linear with slope $\Delta\beta = 1/k_BT_\text{low} - 1/k_BT_\text{high}$.

# %%
fig, axes = plt.subplots(1, len(NVT_INTEGRATORS), figsize=(5 * len(NVT_INTEGRATORS), 4),
                         sharey=True)
fig.suptitle("NVT Ensemble Check — Energy Distributions at T=88K vs T=100K", fontsize=13)

temp_low, temp_high = 88.0, 100.0
delta_beta = 1 / (k_B_eV * temp_low) - 1 / (k_B_eV * temp_high)

for ax, name in zip(axes, NVT_INTEGRATORS):
    try:
        d_lo = load(f"{name}_T{temp_low:.0f}K")
        d_hi = load(f"{name}_T{temp_high:.0f}K")
    except FileNotFoundError:
        ax.set_title(f"{name}\n(no data)")
        continue

    e_lo = d_lo["total_energy"]
    e_hi = d_hi["total_energy"]

    # Overlapping histogram bins
    e_min = min(e_lo.min(), e_hi.min())
    e_max = max(e_lo.max(), e_hi.max())
    bins = np.linspace(e_min, e_max, 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    h_lo, _ = np.histogram(e_lo, bins=bins, density=True)
    h_hi, _ = np.histogram(e_hi, bins=bins, density=True)

    # Plot overlapping distributions
    ax.hist(e_lo, bins=bins, density=True, alpha=0.5, color="blue", label=f"T={temp_low:.0f}K")
    ax.hist(e_hi, bins=bins, density=True, alpha=0.5, color="red", label=f"T={temp_high:.0f}K")

    # Inset: log ratio
    mask = (h_lo > 0) & (h_hi > 0)
    if mask.sum() > 2:
        log_ratio = np.log(h_lo[mask] / h_hi[mask])
        bc = bin_centers[mask]
        # Linear fit
        slope, intercept = np.polyfit(bc, log_ratio, 1)
        inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
        inset.scatter(bc, log_ratio, s=8, color="k", zorder=3)
        inset.plot(bc, slope * bc + intercept, "r-", lw=1.5,
                   label=f"fit: {slope:.1f}\ntheory: {delta_beta:.1f}")
        inset.set_xlabel("E (eV)", fontsize=7)
        inset.set_ylabel("ln(P_lo/P_hi)", fontsize=7)
        inset.legend(fontsize=6)
        inset.tick_params(labelsize=6)

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Total Energy (eV)")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Probability Density")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### physical_validation built-in ensemble check plots (NVT)
# 
# Uses `physical_validation.ensemble.check(..., screen=True)` which shows the forward/reverse work distributions and the linear fit of the log-likelihood ratio.

# %%
for name in NVT_INTEGRATORS:
    try:
        d_lo = load(f"{name}_T{temp_low:.0f}K")
        d_hi = load(f"{name}_T{temp_high:.0f}K")
    except FileNotFoundError:
        print(f"{name}: no data")
        continue

    print(f"\n{'='*60}")
    print(f"  {name} — Ensemble check (physical_validation)")
    print(f"{'='*60}")
    sd_lo = build_sim_data(d_lo, temp_low, ensemble="NVT")
    sd_hi = build_sim_data(d_hi, temp_high, ensemble="NVT")
    quantiles = physical_validation.ensemble.check(
        sd_lo, sd_hi,
        total_energy=True, data_is_uncorrelated=True,
        screen=True, verbosity=2, filename=f"plots/{name}_ensemble_check.png"
    )
    print(f"  Quantiles (σ): {[f'{q:.3f}' for q in quantiles]}")

plt.show()

# %% [markdown]
# ## 4. NPT Time Series
# 
# Temperature, energy, and volume vs step for each NPT integrator.

# %%
NPT_INTEGRATORS = ["npt_langevin", "npt_nose_hoover", "npt_isotropic_crescale", "npt_anisotropic_crescale"]

fig, axes = plt.subplots(len(NPT_INTEGRATORS), 3, figsize=(16, 3.5 * len(NPT_INTEGRATORS)),
                         squeeze=False, sharex=True)
fig.suptitle("NPT Integrators — Time Series", fontsize=14, y=1.01)

for row, name in enumerate(NPT_INTEGRATORS):
    for temp in TEMPS:
        label = f"{name}_T{temp:.0f}K"
        try:
            d = load(label)
        except FileNotFoundError:
            continue

        steps = np.arange(len(d["temperature"]))
        target_T = float(d["target_temperature"])
        tag = f"T={target_T:.0f}K"

        # Temperature
        axes[row, 0].plot(steps, d["temperature"], alpha=0.5, lw=0.5, label=tag)
        axes[row, 0].axhline(target_T, color="k", ls="--", lw=0.8, alpha=0.5)
        axes[row, 0].set_ylabel("Temperature (K)")
        axes[row, 0].set_title(name)
        axes[row, 0].legend(fontsize=7)

        # Total energy
        axes[row, 1].plot(steps, d["total_energy"], alpha=0.5, lw=0.5, label=tag)
        axes[row, 1].set_ylabel("Total Energy (eV)")
        axes[row, 1].set_title(name)
        axes[row, 1].legend(fontsize=7)

        # Volume
        axes[row, 2].plot(steps, d["volumes"], alpha=0.5, lw=0.5, label=tag)
        axes[row, 2].set_ylabel("Volume (Å³)")
        axes[row, 2].set_title(name)
        axes[row, 2].legend(fontsize=7)

for j in range(3):
    axes[-1, j].set_xlabel("Step")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. KE Distribution — NPT Integrators
# 
# Same gamma distribution check, but for NPT integrators at 100K.

# %%
fig, axes = plt.subplots(1, len(NPT_INTEGRATORS), figsize=(5 * len(NPT_INTEGRATORS), 4),
                         sharey=True)
fig.suptitle("KE Distribution vs Maxwell-Boltzmann (NPT, T=100K)", fontsize=13)

for ax, name in zip(axes, NPT_INTEGRATORS):
    label = f"{name}_T100K"
    try:
        d = load(label)
    except FileNotFoundError:
        ax.set_title(f"{name}\n(no data)")
        continue

    ke = d["kinetic_energy"]
    natoms = int(d["natoms"])
    target_T = float(d["target_temperature"])

    ndof = 3 * natoms - 3
    shape = ndof / 2
    scale = k_B_eV * target_T

    ax.hist(ke, bins=60, density=True, alpha=0.6, color="steelblue", label="Observed")

    x = np.linspace(ke.min(), ke.max(), 300)
    pdf = stats.gamma.pdf(x, a=shape, scale=scale)
    ax.plot(x, pdf, "r-", lw=2, label="Theory")

    d_mean = (ke.mean() - shape * scale) / (scale * np.sqrt(2 / ndof))
    d_width = (ke.std() - scale * np.sqrt(shape)) / (scale * np.sqrt(0.5))
    ax.set_title(f"{name}\nd_mean={d_mean:.2f}σ  d_width={d_width:.2f}σ", fontsize=9)
    ax.set_xlabel("Kinetic Energy (eV)")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Probability Density")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### physical_validation built-in KE distribution plots (NPT)

# %%
for name in NPT_INTEGRATORS:
    label = f"{name}_T100K"
    try:
        d = load(label)
    except FileNotFoundError:
        print(f"{name}: no data")
        continue

    print(f"\n{'='*60}")
    print(f"  {name} — KE distribution (physical_validation)")
    print(f"{'='*60}")
    sd = build_sim_data(d, 100.0, ensemble="NPT")
    result = physical_validation.kinetic_energy.distribution(
        sd, strict=False, screen=True, verbosity=2,
    )
    print(f"  Result: d_mean={result[0]:.3f}σ, d_width={result[1]:.3f}σ")

plt.show()

# %% [markdown]
# ## 6. Ensemble Check — NPT Integrators
# 
# Same Boltzmann weight ratio check at T=88K vs T=100K for NPT integrators.

# %%
fig, axes = plt.subplots(1, len(NPT_INTEGRATORS), figsize=(5 * len(NPT_INTEGRATORS), 4),
                         sharey=True)
fig.suptitle("NPT Ensemble Check — Energy Distributions at T=88K vs T=100K", fontsize=13)

for ax, name in zip(axes, NPT_INTEGRATORS):
    try:
        d_lo = load(f"{name}_T{temp_low:.0f}K")
        d_hi = load(f"{name}_T{temp_high:.0f}K")
    except FileNotFoundError:
        ax.set_title(f"{name}\n(no data)")
        continue

    e_lo = d_lo["total_energy"]
    e_hi = d_hi["total_energy"]

    e_min = min(e_lo.min(), e_hi.min())
    e_max = max(e_lo.max(), e_hi.max())
    bins = np.linspace(e_min, e_max, 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    h_lo, _ = np.histogram(e_lo, bins=bins, density=True)
    h_hi, _ = np.histogram(e_hi, bins=bins, density=True)

    ax.hist(e_lo, bins=bins, density=True, alpha=0.5, color="blue", label=f"T={temp_low:.0f}K")
    ax.hist(e_hi, bins=bins, density=True, alpha=0.5, color="red", label=f"T={temp_high:.0f}K")

    mask = (h_lo > 0) & (h_hi > 0)
    if mask.sum() > 2:
        log_ratio = np.log(h_lo[mask] / h_hi[mask])
        bc = bin_centers[mask]
        slope, intercept = np.polyfit(bc, log_ratio, 1)
        inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
        inset.scatter(bc, log_ratio, s=8, color="k", zorder=3)
        inset.plot(bc, slope * bc + intercept, "r-", lw=1.5,
                   label=f"fit: {slope:.1f}\ntheory: {delta_beta:.1f}")
        inset.set_xlabel("E (eV)", fontsize=7)
        inset.set_ylabel("ln(P_lo/P_hi)", fontsize=7)
        inset.legend(fontsize=6)
        inset.tick_params(labelsize=6)

    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Total Energy (eV)")
    ax.legend(fontsize=8)

axes[0].set_ylabel("Probability Density")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### physical_validation built-in ensemble check plots (NPT)

# %%
for name in NPT_INTEGRATORS:
    try:
        d_lo = load(f"{name}_T{temp_low:.0f}K")
        d_hi = load(f"{name}_T{temp_high:.0f}K")
    except FileNotFoundError:
        print(f"{name}: no data")
        continue

    print(f"\n{'='*60}")
    print(f"  {name} — Ensemble check (physical_validation)")
    print(f"{'='*60}")
    sd_lo = build_sim_data(d_lo, temp_low, ensemble="NPT")
    sd_hi = build_sim_data(d_hi, temp_high, ensemble="NPT")
    try:
        quantiles = physical_validation.ensemble.check(
            sd_lo, sd_hi,
            total_energy=False, # data_is_uncorrelated=True,
            screen=True, verbosity=2,
        )
        print(f"  Quantiles (σ): {[f'{q:.3f}' for q in quantiles]}")
    except Exception as e: #ConvergenceError
        print(f"  ConvergenceError: {e}")

plt.show()

# %% [markdown]
# ## 6b. NPT Pressure Ensemble Check
# 
# Compare volume distributions at two pressures (same temperature).
# For correct NPT sampling, `physical_validation` checks that volume distributions
# at different pressures satisfy the expected Boltzmann weight relationship.
# This is a 1D check on volumes only — no energy data needed.

# %%
from torch_sim.units import MetalUnits

P_BAR_CONVERSION = float(MetalUnits.pressure)  # 1 bar in eV/Ang^3

# Discover available pressure-sweep data files
pressure_files = sorted(DATA_DIR.glob("*_P*bar.npz"))
print("Pressure-sweep data files:")
for f in pressure_files:
    print(f"  {f.name}")

# Parse integrator -> {pressure_bar: filename} mapping
pressure_data = {}  # integrator_name -> [(p_bar, filename), ...]
for f in pressure_files:
    stem = f.stem  # e.g. npt_langevin_T100K_P0bar
    parts = stem.rsplit("_P", 1)
    if len(parts) != 2:
        continue
    prefix = parts[0]  # e.g. npt_langevin_T100K
    p_bar = float(parts[1].replace("bar", ""))
    int_prefix = prefix.rsplit("_T", 1)[0]  # e.g. npt_langevin
    pressure_data.setdefault(int_prefix, []).append((p_bar, f.stem))

for name, entries in pressure_data.items():
    entries.sort(key=lambda x: x[0])
    print(f"\n{name}: {[(p, fn) for p, fn in entries]}")

# %%
# Volume distributions at different pressures for each NPT integrator
integrators_with_pressure = [n for n in NPT_INTEGRATORS if n in pressure_data]

if integrators_with_pressure:
    fig, axes = plt.subplots(1, len(integrators_with_pressure),
                             figsize=(5 * len(integrators_with_pressure), 4), squeeze=False)
    axes = axes[0]
    fig.suptitle("NPT Pressure Check \u2014 Volume Distributions at Same T, Different P", fontsize=13)

    for ax, name in zip(axes, integrators_with_pressure):
        entries = pressure_data[name]
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(entries)))
        for (p_bar, fname), color in zip(entries, colors):
            d = load(fname)
            vols = d["volumes"]
            ax.hist(vols, bins=50, density=True, alpha=0.5, color=color,
                    label=f"P={p_bar:.0f} bar")
            ax.axvline(vols.mean(), color=color, ls="--", lw=1, alpha=0.7)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Volume (\u00c5\u00b3)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Probability Density")
    fig.tight_layout()
    plt.savefig("plots/npt_pressure_volume_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No pressure-sweep data found. Run: python run_npt.py --mode pressure")

# %% [markdown]
# ### physical_validation built-in pressure ensemble check
# 
# Uses `physical_validation.ensemble.check()` with two simulations at the same temperature
# but different pressures. The library auto-detects this case and performs a 1D volume-based check.

# %%
for name in integrators_with_pressure:
    entries = pressure_data[name]
    if len(entries) < 2:
        print(f"{name}: need at least 2 pressures, got {len(entries)}")
        continue

    p_lo_bar, fname_lo = entries[0]
    p_hi_bar, fname_hi = entries[-1]
    d_lo = load(fname_lo)
    d_hi = load(fname_hi)

    p_lo_eva3 = float(d_lo["external_pressure"])
    p_hi_eva3 = float(d_hi["external_pressure"])
    temp = float(d_lo["target_temperature"])
    print(f"{name}: p_lo={p_lo_bar:.1f} bar (eva3={p_lo_eva3:.3e}), "
          f"p_hi={p_hi_bar:.1f} bar (eva3={p_hi_eva3:.3e}), T={temp:.1f}K")

    print(f"\n{'='*60}")
    print(f"  {name} \u2014 Pressure ensemble check")
    print(f"  T={temp:.0f}K, P_lo={p_lo_bar:.0f} bar, P_hi={p_hi_bar:.0f} bar")
    print(f"{'='*60}")

    # sd_lo = build_sim_data(d_lo, temp, ensemble="NPT", pressure=p_lo_eva3)
    # sd_hi = build_sim_data(d_hi, temp, ensemble="NPT", pressure=p_hi_eva3)

    sd_lo = build_sim_data(d_lo, temp, ensemble="NPT", pressure=p_lo_bar)
    sd_hi = build_sim_data(d_hi, temp, ensemble="NPT", pressure=p_hi_bar)

    try:
        quantiles = physical_validation.ensemble.check(
            sd_lo, sd_hi,
            total_energy=False, #data_is_uncorrelated=True,
            screen=True, verbosity=2,
            filename=f"plots/{name}_pressure_ensemble_check.png",
        )
        print(f"  Quantiles (\u03c3): {[f'{q:.3f}' for q in quantiles]}")
    except Exception as e:
        print(f"  Error: {e}")

plt.show()

# %%
for name in integrators_with_pressure:
    entries = pressure_data[name]
    if len(entries) < 2:
        print(f"{name}: need at least 2 pressures, got {len(entries)}")
        continue

    p_lo_bar, fname_lo = entries[0]
    p_hi_bar, fname_hi = entries[-1]
    d_lo = load(fname_lo)
    d_hi = load(fname_hi)

    p_lo_eva3 = float(d_lo["external_pressure"])
    p_hi_eva3 = float(d_hi["external_pressure"])
    temp = float(d_lo["target_temperature"])
    print(f"{name}: p_lo={p_lo_bar:.1f} bar (eva3={p_lo_eva3:.3e}), "
          f"p_hi={p_hi_bar:.1f} bar (eva3={p_hi_eva3:.3e}), T={temp:.1f}K")

    print(f"\n{'='*60}")
    print(f"  {name} \u2014 Pressure ensemble check")
    print(f"  T={temp:.0f}K, P_lo={p_lo_bar:.0f} bar, P_hi={p_hi_bar:.0f} bar")
    print(f"{'='*60}")

    # sd_lo = build_sim_data(d_lo, temp, ensemble="NPT", pressure=p_lo_eva3)
    # sd_hi = build_sim_data(d_hi, temp, ensemble="NPT", pressure=p_hi_eva3)

    sd_lo = build_sim_data(d_lo, temp, ensemble="NPT", pressure=p_lo_bar)
    sd_hi = build_sim_data(d_hi, temp, ensemble="NPT", pressure=p_hi_bar)

    try:
        quantiles = physical_validation.ensemble.check(
            sd_lo, sd_hi,
            total_energy=False, data_is_uncorrelated=True,
            screen=True, verbosity=2,
            filename=f"plots/{name}_pressure_ensemble_check.png",
        )
        print(f"  Quantiles (\u03c3): {[f'{q:.3f}' for q in quantiles]}")
    except Exception as e:
        print(f"  Error: {e}")

plt.show()

# %% [markdown]
# ## 8. NVE Integrator Convergence
# 
# RMSD of the conserved quantity (total energy) vs timestep on a log-log scale.
# For velocity Verlet, the RMSD should scale as $\text{dt}^2$ (slope = 2 on log-log).

# %%
try:
    nve = load("nve_convergence")
    timesteps_ps = nve["timesteps_ps"]

    dts, rmsds, drifts = [], [], []
    for dt_ps in timesteps_ps:
        key = f"dt_{dt_ps}"
        com = nve[f"{key}_constant_of_motion"]
        dts.append(dt_ps)
        rmsds.append(com.std())
        drifts.append(com[-1] - com[0])

    dts = np.array(dts)
    rmsds = np.array(rmsds)
    drifts = np.array(drifts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("NVE Integrator Convergence (4-atom Ar, T=5K)", fontsize=13)

    # --- Panel 1: RMSD vs dt (log-log) ---
    ax1.loglog(dts, rmsds, "ko-", ms=8, lw=2, label="RMSD(E_tot)")

    # dt^2 reference line through middle point
    mid = len(dts) // 2
    ref = rmsds[mid] * (dts / dts[mid]) ** 2
    ax1.loglog(dts, ref, "r--", lw=1.5, alpha=0.7, label="$\\propto dt^2$ (reference)")

    # Fit slope
    log_dts = np.log(dts)
    log_rmsds = np.log(rmsds)
    # Only fit points where RMSD is clearly above noise floor
    mask = rmsds > 1.5 * rmsds.min()
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(log_dts[mask], log_rmsds[mask], 1)
        ax1.set_title(f"Log-log slope = {slope:.2f} (expected: 2.0)")
    else:
        ax1.set_title("RMSD vs dt")

    ax1.set_xlabel("Timestep (ps)")
    ax1.set_ylabel("RMSD of E_total (eV)")
    ax1.legend()

    # --- Panel 2: Energy time series for each dt ---
    for i, dt_ps in enumerate(timesteps_ps):
        key = f"dt_{dt_ps}"
        com = nve[f"{key}_constant_of_motion"]
        steps = np.arange(len(com))
        ax2.plot(steps, com - com[0], alpha=0.7, lw=0.8, label=f"dt={dt_ps:.4f} ps")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("E_total - E_total[0] (eV)")
    ax2.set_title("Energy drift per timestep")
    ax2.legend(fontsize=7, ncol=2)

    # --- Panel 3: RMSD ratio table ---
    ax3.axis("off")
    headers = ["dt1 (ps)", "dt2 (ps)", "dt1/dt2", "RMSD1/RMSD2", "(dt1/dt2)²"]
    rows = []
    for i in range(len(dts) - 1):
        dt_ratio = dts[i] / dts[i + 1]
        rmsd_ratio = rmsds[i] / rmsds[i + 1]
        expected = dt_ratio ** 2
        rows.append([f"{dts[i]:.4f}", f"{dts[i+1]:.4f}",
                     f"{dt_ratio:.3f}", f"{rmsd_ratio:.3f}", f"{expected:.3f}"])

    table = ax3.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax3.set_title("Convergence ratios", pad=20)

    fig.tight_layout()
    plt.show()

except FileNotFoundError:
    print("No NVE data found. Run: python run_nve.py")

# %% [markdown]
# ### physical_validation built-in integrator convergence plot
# 
# Uses `physical_validation.integrator.convergence(..., screen=True)` which plots RMSD vs dt with the expected dt^2 reference line.

# %%
try:
    nve = load("nve_convergence")
    timesteps_ps = nve["timesteps_ps"]
    natoms = int(nve["natoms"])
    masses = nve["masses"]
    volume = float(nve["volume"])

    # Pick 3 timesteps in the dt^2 regime (avoid noise floor and nonlinear regime)
    # Use [0.007, 0.005, 0.004] which showed good convergence
    selected_dts = [0.007, 0.005, 0.004]

    simulations = []
    for dt_ps in selected_dts:
        key = f"dt_{dt_ps}"
        com = nve[f"{key}_constant_of_motion"]
        dt_internal = float(nve[f"{key}_dt_internal"])

        system = physical_validation.data.SystemData(
            natoms=natoms, nconstraints=0,
            ndof_reduction_tra=3, ndof_reduction_rot=0, mass=masses)
        ensemble = physical_validation.data.EnsembleData(
            ensemble="NVE", natoms=natoms, volume=volume)
        obs = physical_validation.data.ObservableData(constant_of_motion=com)
        sd = physical_validation.data.SimulationData(
            units=make_unit_data(), dt=dt_internal,
            system=system, ensemble=ensemble, observables=obs)
        simulations.append(sd)

    result = physical_validation.integrator.convergence(
        simulations, verbose=True, screen=True,
    )
    print(f"\nConvergence deviation: {result:.3f}  ({'PASS' if result < 0.5 else 'FAIL'})")

except FileNotFoundError:
    print("No NVE data found. Run: python run_nve.py")

plt.show()

# %% [markdown]
# ## 9. Summary Table
# 
# Run `physical_validation` checks programmatically on all available data and collect pass/fail results.

# %%
# Collect results
results = []
all_integrators = NVT_INTEGRATORS + NPT_INTEGRATORS

for name in all_integrators:
    is_npt = name.startswith("npt")
    ensemble = "NPT" if is_npt else "NVT"

    # --- KE Distribution at 100K ---
    label_100 = f"{name}_T100K"
    ke_status = "no data"
    try:
        d100 = load(label_100)
        sd = build_sim_data(d100, 100.0, ensemble=ensemble)
        d_mean, d_width = physical_validation.kinetic_energy.distribution(
            sd, strict=False, verbosity=0)
        ke_pass = abs(d_mean) < THRESHOLD and abs(d_width) < THRESHOLD
        ke_status = f"PASS (\u03bc={d_mean:.2f}\u03c3, w={d_width:.2f}\u03c3)" if ke_pass else f"FAIL (\u03bc={d_mean:.2f}\u03c3, w={d_width:.2f}\u03c3)"
    except Exception as e:
        ke_status = f"ERROR: {e}"

    # --- Ensemble Check 88K vs 100K (temperature) ---
    ens_status = "no data"
    try:
        d88 = load(f"{name}_T88K")
        d100 = load(f"{name}_T100K")
        sd_lo = build_sim_data(d88, 88.0, ensemble=ensemble)
        sd_hi = build_sim_data(d100, 100.0, ensemble=ensemble)
        quantiles = physical_validation.ensemble.check(
            sd_lo, sd_hi, total_energy=True, data_is_uncorrelated=True, verbosity=0)
        max_q = max(abs(q) for q in quantiles)
        ens_pass = max_q < THRESHOLD
        ens_status = f"PASS (max |q|={max_q:.2f}\u03c3)" if ens_pass else f"FAIL (max |q|={max_q:.2f}\u03c3)"
    except Exception as e:
        ens_status = f"ERROR: {e}"

    # --- Pressure Ensemble Check (NPT only, same T, different P) ---
    press_status = "N/A"
    if is_npt and name in pressure_data:
        entries = pressure_data[name]
        if len(entries) >= 2:
            try:
                p_lo_bar, fname_lo = entries[0]
                p_hi_bar, fname_hi = entries[-1]
                d_plo = load(fname_lo)
                d_phi = load(fname_hi)
                p_lo_eva3 = float(d_plo["external_pressure"])
                p_hi_eva3 = float(d_phi["external_pressure"])
                temp_p = float(d_plo["target_temperature"])
                sd_plo = build_sim_data(d_plo, temp_p, ensemble="NPT", pressure=p_lo_eva3)
                sd_phi = build_sim_data(d_phi, temp_p, ensemble="NPT", pressure=p_hi_eva3)
                quantiles_p = physical_validation.ensemble.check(
                    sd_plo, sd_phi, total_energy=False, data_is_uncorrelated=True, verbosity=0)
                max_qp = max(abs(q) for q in quantiles_p)
                press_pass = max_qp < THRESHOLD
                press_status = f"PASS (max |q|={max_qp:.2f}\u03c3)" if press_pass else f"FAIL (max |q|={max_qp:.2f}\u03c3)"
            except Exception as e:
                press_status = f"ERROR: {e}"
        else:
            press_status = "need 2 pressures"

    results.append((name, ke_status, ens_status, press_status))

# Print table
print(f"{'Integrator':<30} {'KE Distribution':<40} {'Ensemble (T)':<35} {'Ensemble (P)':<35}")
print("-" * 140)
for name, ke, ens, press in results:
    print(f"{name:<30} {ke:<40} {ens:<35} {press:<35}")


