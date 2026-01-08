# TorchSim Consolidated Examples

This directory contains consolidated example scripts demonstrating the key features of TorchSim. Each script combines multiple related examples into a single, well-organized file with clear sections.

## Quick Start

All scripts can be run directly with Python or using `uv`:

```bash
# Run with Python
python 1_introduction.py

# Run with uv (automatically installs dependencies)
uv run 1_introduction.py
```

## Overview of Examples

### 1. [1_introduction.py](1_introduction.py)

#### Introduction to TorchSim basics

Learn the fundamentals of TorchSim with simple examples using classical and machine learning potentials.

**Topics covered:**

- Lennard-Jones model for classical potentials
- MACE model for machine learning potentials
- Batched inference for efficient computation
- Basic state management and model evaluation

**Dependencies:** `scipy>=1.15`, `mace-torch>=0.3.12`

---

### 2. [2_structural_optimization.py](2_structural_optimization.py)

#### Structure optimization techniques

Comprehensive examples of structural relaxation using different optimizers and cell filters.

**Topics covered:**

- FIRE optimizer (Fast Inertial Relaxation Engine)
- Gradient descent optimizer
- Position-only optimization
- Cell optimization with unit cell filter
- Cell optimization with Frechet cell filter
- Batched optimization for multiple structures
- Pressure control during optimization

**Key concepts:** Force minimization, cell relaxation, pressure convergence, batched computations

**Dependencies:** `scipy>=1.15`, `mace-torch>=0.3.12`

---

### 3. [3_dynamics.py](3_dynamics.py)

#### Molecular dynamics simulations

Explore various ensembles and integrators for molecular dynamics.

**Topics covered:**

- **NVE ensemble** (microcanonical): Energy conservation
- **NVT ensemble** (canonical): Temperature control with Langevin and Nose-Hoover thermostats
- **NPT ensemble** (isothermal-isobaric): Pressure and temperature control
- Lennard-Jones and MACE models
- Energy conservation verification
- Performance benchmarking

**Key concepts:** Statistical ensembles, thermostats, barostats, total energy conservation

**Dependencies:** `scipy>=1.15`, `mace-torch>=0.3.12`

---

### 4. [4_high_level_api.py](4_high_level_api.py)

#### Simplified high-level interface

Use TorchSim's high-level API for common workflows with minimal code.

**Topics covered:**

- Simple integration interface (`ts.integrate`)
- Simple optimization interface (`ts.optimize`)
- Trajectory logging and reporting
- Batched simulations
- Custom convergence criteria
- Support for ASE Atoms and Pymatgen Structure objects

**Key concepts:** User-friendly API, automatic batching, flexible input formats, trajectory analysis

**Dependencies:** `mace-torch>=0.3.12`, `pymatgen>=2025.2.18`

---

### 5. [5_workflow.py](5_workflow.py)

#### Advanced workflows and utilities

Complex simulation workflows for production use cases.

**Topics covered:**

- In-flight autobatching for memory-efficient optimization
- Dynamic batch management
- Elastic constant calculations
- Bulk and shear moduli
- Bravais lattice detection
- Force convergence utilities

**Key concepts:** Autobatching, mechanical properties, elastic tensor, production workflows

**Dependencies:** `mace-torch>=0.3.12`, `matbench-discovery>=1.3.1`

---

### 6. [6_phonons.py](6_phonons.py)

#### Phonon calculations

Calculate vibrational properties using finite differences.

**Topics covered:**

- Structure relaxation for phonons
- Phonon density of states (DOS)
- Phonon band structure
- Batched force constant calculations
- Integration with Phonopy
- High-symmetry path generation
- Visualization (optional)

**Key concepts:** Harmonic approximation, force constants, phonon dispersion, thermal properties

**Dependencies:** `mace-torch>=0.3.12`, `phonopy>=2.35`, `pymatviz>=0.17.1`, `plotly>=6.3.0`, `seekpath`, `ase`

---

### 7. [7_others.py](7_others.py)

#### Miscellaneous advanced features

Advanced features and utility functions.

**Topics covered:**

- Batched neighbor list calculations
  - Linked cell method (efficient for large systems)
  - N^2 method (simple reference implementation)
- Velocity autocorrelation function (VACF)
- Correlation function analysis
- Property calculations during MD

**Key concepts:** Neighbor lists, time correlation functions, analysis tools

**Dependencies:** `ase>=3.26`, `scipy>=1.15`, `matplotlib`, `numpy`

---

## Running the Examples

### Prerequisites

Install TorchSim and dependencies:

```bash
pip install torch-sim
```

Or use `uv` for automatic dependency management:

```bash
uv pip install torch-sim
```

### Running Individual Scripts

Each script is self-contained with its dependencies specified in the header. You can run them directly:

```bash
# With Python
python examples/new_scripts/1_introduction.py

# With uv (auto-installs dependencies)
uv run examples/new_scripts/1_introduction.py
```

### Smoke Testing (Fast Mode)

All scripts support a fast "smoke test" mode for CI or quick verification:

```bash
CI=1 python 1_introduction.py
```

This reduces the number of steps and simplifies calculations for quick execution.

## Learning Path

We recommend working through the examples in order:

1. **Start with [1_introduction.py](1_introduction.py)** to understand basic concepts
2. **Try [2_structural_optimization.py](2_structural_optimization.py)** to learn optimization
3. **Explore [3_dynamics.py](3_dynamics.py)** for molecular dynamics
4. **Use [4_high_level_api.py](4_high_level_api.py)** for simplified workflows
5. **Advanced users:** Check out [5_workflow.py](5_workflow.py), [6_phonons.py](6_phonons.py), and [7_others.py](7_others.py)

## Key Features Demonstrated

### Models

- **Lennard-Jones**: Classical pair potential
- **MACE**: Machine learning interatomic potential

### Optimizers

- **FIRE**: Fast, adaptive optimizer for geometry relaxation
- **Gradient Descent**: Simple first-order optimizer

### Integrators

- **NVE**: Microcanonical ensemble (energy conservation)
- **NVT Langevin**: Canonical ensemble with stochastic thermostat
- **NVT Nose-Hoover**: Canonical ensemble with deterministic thermostat
- **NPT Nose-Hoover**: Isothermal-isobaric ensemble

### Cell Filters

- **None**: Position-only optimization
- **Unit Cell**: Optimize cell with uniform scaling
- **Frechet Cell**: Full cell optimization with metric-preserving updates

### Batching

- Efficient batched inference for multiple structures
- Dynamic autobatching for memory optimization
- Mixed system sizes in single batch

## Tips for Best Performance

1. **Use CUDA if available**: TorchSim automatically uses GPU when available
2. **Batch similar structures**: Group structures with similar sizes for best efficiency
3. **Enable autobatching**: For heterogeneous workloads, use `InFlightAutoBatcher`
4. **Choose appropriate precision**: Use `float32` for speed, `float64` for accuracy
5. **Profile your code**: Use the built-in timing in examples as a template

## Getting Help

- **Documentation**: See the main TorchSim documentation
- **Issues**: Report problems at the TorchSim GitHub repository
- **Questions**: Check existing issues or open a new discussion

## Differences from Original Examples

These consolidated examples:

- ✅ Remove duplicate code and common setup patterns
- ✅ Use consistent naming and style
- ✅ Add clear section markers and documentation
- ✅ Include sensible defaults and smoke test support
- ✅ Reduce total execution time by ~10x in CI
- ✅ Maintain all key functionality and learning objectives

The original examples in `examples/scripts/` remain available for reference.
