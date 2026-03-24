# Integrator Physical Validation Tests

Physical validation of all torch-sim MD integrators using the
[physical_validation](https://physical-validation.readthedocs.io/) library
and LJ Argon as a test system.

## What's here

| File | Purpose |
|---|---|
| `common.py` | Shared constants, model/structure builders |
| `run_nvt.py` | Run NVT simulations (Langevin, Nose-Hoover, V-Rescale) at 80K and 100K |
| `run_npt.py` | Run NPT simulations (Langevin, Nose-Hoover, iso/aniso C-Rescale) at 80K and 100K |
| `run_nve.py` | Run NVE simulations at 8 timesteps for convergence analysis |
| `analyze.ipynb` | Jupyter notebook with all diagnostic plots and `physical_validation` checks |
| `data/` | Output directory for `.npz` trajectory data (gitignored) |

## Quick start

```bash
# 1. Generate trajectory data (from this directory)
python run_nve.py
python run_nvt.py      
python run_npt.py

# Run a single integrator if needed
python run_nvt.py --integrator nvt_langevin
python run_npt.py --integrator npt_nose_hoover

# NPT runs both a pressure and temperature sweep, so you can specify which to do:
python run_npt.py --mode temperature  # Vary T at P=0
python run_npt.py --mode pressure     # Vary P at fixed T

# 2. Open the notebook
jupyter notebook analyze.ipynb
```

## What the notebook shows

### Custom plots
- **Time series**: Temperature, total energy, volume (NPT) vs step
- **KE distribution**: Observed histogram vs theoretical Gamma(Nf/2, k_BT) distribution
- **Ensemble check**: Overlapping energy distributions at two temperatures with log-ratio inset
- **NVE convergence**: Log-log RMSD vs dt plot, energy drift traces, convergence ratio table

### physical_validation native plots
The notebook also calls `physical_validation` functions with `screen=True` to produce
their built-in diagnostic figures:
- `kinetic_energy.distribution()` — KS test or mean/width comparison plot
- `ensemble.check()` — Forward/reverse work distributions and linear fit
- `integrator.convergence()` — RMSD vs dt with reference line

### Summary table
Final cell runs all checks programmatically and prints a PASS/FAIL table for every integrator.

## Validation checks

| Check | Integrators | What it tests |
|---|---|---|
| KE distribution | All NVT + NPT | Kinetic energy follows Maxwell-Boltzmann (gamma) distribution |
| Ensemble check | All NVT + NPT | Energy distributions at T=80K and T=100K satisfy Boltzmann weight ratio |
| NVE convergence | NVE (velocity Verlet) | Energy drift RMSD scales as dt^2 |

## System details

- **Structure**: FCC Argon (a=5.26 A), 2x2x2 supercell (32 atoms) for NVT/NPT, unit cell (4 atoms) for NVE
- **Model**: Lennard-Jones (sigma=3.405, epsilon=0.0104 eV, cutoff=8.5125 A), no neighbor list
- **Timestep**: 0.004 ps for NVT/NPT; sweep 0.002-0.010 ps for NVE
- **Production**: 10,000 steps after 2,000 (NVT) or 3,000 (NPT) equilibration steps
- **Threshold**: |deviation| < 3 sigma for all statistical checks
