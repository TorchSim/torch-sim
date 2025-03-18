# TorchSim Examples

This folder contains a series of examples demonstrating the use of TorchSim, a library for simulating molecular dynamics and structural optimization using classical and machine learning interatomic potentials. Each example showcases different functionalities and models available in TorchSim.

## Examples Overview

### 1. Introduction

- **1.1 Lennard-Jones Model**: Demonstrates the setup and simulation of a Lennard-Jones potential for Argon atoms arranged in a face-centered cubic (FCC) lattice. The example initializes the model, runs a model forward pass, and prints the energy, forces, and stress.
  `torchsim/examples/1_Introduction/1.1_Lennard_Jones.py`

- **1.2 MACE Model**: Shows how to use the MACE model for simulating diamond cubic Silicon. It includes loading a pre-trained model, setting up the system, and running a model forward pass to obtain energy, forces, and stress.
  `torchsim/examples/1_Introduction/1.2_MACE.py`

- **1.3 Batched MACE Model**: Extends the MACE model example to handle batched inputs, allowing for the simulation of multiple systems simultaneously. It demonstrates the setup and execution of batched simulations.
  `torchsim/examples/1_Introduction/1.3_Batched_MACE.py`

- **1.4 Fairchem Model**: Demonstrates the Fairchem model for simulating diamond cubic Silicon. It includes model setup, data preparation, and running simulations to obtain energy, forces, and stress.
  `torchsim/examples/1_Introduction/1.4_Fairchem.py`

### 2. Structural Optimization

- **2.1 Lennard-Jones FIRE**: Shows how to perform structural optimization using the FIRE (Fast Inertial Relaxation Engine) optimizer with a Lennard-Jones model.
  `torchsim/examples/2_Structural_optimization/2.1_Lennard_Jones_FIRE.py`


- **2.2 Soft Sphere FIRE**: Similar to the Lennard-Jones example, but using a Soft Sphere model for optimization.
  `torchsim/examples/2_Structural_optimization/2.2_Soft_Sphere_FIRE.py`

- **2.3 MACE FIRE**: Uses the MACE model for structural optimization with the FIRE optimizer.
  `torchsim/examples/2_Structural_optimization/2.3_MACE_FIRE.py`

- **2.4 MACE UnitCellFilter FIRE**: Uses the MACE model for structural optimization with the UnitCellFilter FIRE optimizer.
  `torchsim/examples/2_Structural_optimization/2.4_MACE_UnitCellFilter_FIRE.py`

- **2.5 MACE FrechetCellFilter FIRE**: Uses the MACE model for structural optimization with the FrechetCellFilter FIRE optimizer.
  `torchsim/examples/2_Structural_optimization/2.5_MACE_FrechetCellFilter_FIRE.py`

- **2.6 Batched MACE Gradient Descent**: Demonstrates batched gradient descent optimization using the MACE model.
  `torchsim/examples/2_Structural_optimization/2.6_Batched_MACE_Gradient_Desent.py`

- **2.7 Batched MACE UnitCellFilter Gradient Descent**: Extends the batched optimization to include unit cell optimization using gradient descent.
  `torchsim/examples/2_Structural_optimization/2.7_Batched_MACE_UnitCellFilter_Gradient_Desent.py`

- **2.8 Batched MACE UnitCellFilter FIRE**: Similar to the previous example but using the FIRE optimizer for unit cell optimization.
  `torchsim/examples/2_Structural_optimization/2.8_Batched_MACE_UnitCellFilter_FIRE.py`



### 3. Dynamics

- **3.1 Lennard-Jones NVE**: Simulates molecular dynamics using the NVE ensemble with a Lennard-Jones model. It includes setup, simulation, and energy conservation checks.
  `torchsim/examples/3_Dynamics/3.1_Lennard_Jones_NVE.py`

- **3.2 MACE NVE**: Similar to the Lennard-Jones example but using the MACE model for NVE molecular dynamics simulation.
  `torchsim/examples/3_Dynamics/3.2_MACE_NVE.py`

- **3.3 MACE NVE with Cueq**: Runs the MACE model in NVE with the CuEq acceleration.
  `torchsim/examples/3_Dynamics/3.3_MACE_NVE_cueq.py`

- **3.4 MACE NVT Langevin**: Demonstrates the use of the NVT Langevin integrator with the MACE model for temperature-controlled molecular dynamics.
  `torchsim/examples/3_Dynamics/3.4_MACE_NVT_Langevin.py`

- **3.5 MACE NVT Nose-Hoover**: Uses the NVT Nose-Hoover integrator with the MACE model for temperature-controlled molecular dynamics.
  `torchsim/examples/3_Dynamics/3.5_MACE_NVT_Nose_Hoover.py`

- **3.6 MACE NVT Nose-Hoover with Temperature Profile**: Extends the Nose-Hoover example to include a temperature profile, simulating heating and cooling cycles.
  `torchsim/examples/3_Dynamics/3.6_MACE_NVT_Nose_Hoover_temp_profile.py`

- **3.7 Lennard-Jones NPT Nose-Hoover**: Demonstrates the use of the NPT Nose-Hoover integrator with a Lennard-Jones model for pressure-controlled molecular dynamics.
  `torchsim/examples/3_Dynamics/3.7_Lennard_Jones_NPT_Nose_Hoover.py`

- **3.8 MACE NPT Nose-Hoover**: Similar to the Lennard-Jones example but using the MACE model for pressure-controlled molecular dynamics.
  `torchsim/examples/3_Dynamics/3.8_MACE_NPT_Nose_Hoover.py`

- **3.9 MACE NVT with Staggered Stress**: Demonstrates the use of staggered stress calculations during NVT simulations with the MACE model.
  `torchsim/examples/3_Dynamics/3.9_MACE_NVT_staggered_stress.py`

- **3.10 Batched Integrator Test**: Tests batched integration using various models and integrators, showcasing the flexibility of TorchSim in handling different simulation scenarios.
  `torchsim/examples/3_Dynamics/3.10_Batched_integrator_test.py`

- **3.11 Hybrid Swap Monte Carlo**: Combines molecular dynamics with Monte Carlo simulations using the MACE model, demonstrating hybrid simulation techniques.
  `torchsim/examples/3_Dynamics/3.11_Hybrid_swap_mc.py`

### 4. High-Level API

- **4.1 High-Level API**: Provides examples of using the high-level API for integrating systems with different models and integrators, including Lennard-Jones and MACE models.
  `torchsim/examples/4_High_level_api/4.1_high_level_api.py`

### 5. Workflow

- **5.1 Workflow**: Demonstrates how to run the a2c workflow with MACE model.
  `torchsim/examples/5_Workflow/5.1_a2c_workflow.py`

### 6. Phonons

- **6.1 Phonon DOS with MACE Batched**: Demonstrates how to compute the phonon density of states (DOS) using the MACE model in batched mode.
  `torchsim/examples/6_Phonons/6.1_phonon_dos_batched_MACE.py`

- **6.2 Thermal Conductivity with MACE**: Demonstrates how to compute the thermal conductivity using the MACE model.
  `torchsim/examples/6_Phonons/6.2_Thermal_conductivity_MACE.py`

- **6.3 Thermal Conductivity with MACE Batched**: Demonstrates how to compute the thermal conductivity using the MACE model in batched mode.
  `torchsim/examples/6_Phonons/6.3_Thermal_conductivity_batched_MACE.py`

Each example is designed to be self-contained and can be run independently to explore the capabilities of TorchSim. The examples cover a wide range of functionalities, from basic model setup to advanced simulation techniques, providing a comprehensive overview of the library's features.
