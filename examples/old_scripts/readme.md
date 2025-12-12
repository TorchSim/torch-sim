# TorchSim Example Scripts

This folder contains a series of examples demonstrating the use of TorchSim, a library for simulating molecular dynamics and structural optimization using classical and machine learning interatomic potentials. Each example showcases different functionalities and models available in TorchSim.

1. **Introduction**

   1. **Lennard-Jones Model** - [`examples/1_Introduction/1.1_Lennard_Jones.py`](1_Introduction/1.1_Lennard_Jones.py): Simulate Argon atoms in an FCC lattice with a Lennard-Jones potential. Initialize the model, run a forward pass, and print energy, forces, and stress.

   1. **MACE Model** - [`examples/1_Introduction/1.2_MACE.py`](1_Introduction/1.2_MACE.py): Use the MACE model to simulate diamond cubic Silicon. Load a pre-trained model, set up the system, and calculate energy, forces, and stress.

   1. **Batched MACE Model** - [`examples/1_Introduction/1.3_Batched_MACE.py`](1_Introduction/1.3_Batched_MACE.py): Handle batched inputs with the MACE model to simulate multiple systems simultaneously.

   1. **Fairchem Model** - [`examples/1_Introduction/1.4_Fairchem.py`](1_Introduction/1.4_Fairchem.py): Simulate diamond cubic Silicon with the Fairchem model. Set up the model and calculate energy, forces, and stress.

1. **Structural Optimization**

   1. **Lennard-Jones FIRE** - [`examples/2_Structural_optimization/2.1_Lennard_Jones_FIRE.py`](2_Structural_optimization/2.1_Lennard_Jones_FIRE.py): Perform structural optimization using the FIRE optimizer with a Lennard-Jones model.

   1. **Soft Sphere FIRE** - [`examples/2_Structural_optimization/2.2_Soft_Sphere_FIRE.py`](2_Structural_optimization/2.2_Soft_Sphere_FIRE.py): Optimize structures with a Soft Sphere model using the FIRE optimizer.

   1. **MACE FIRE** - [`examples/2_Structural_optimization/2.3_MACE_FIRE.py`](2_Structural_optimization/2.3_MACE_FIRE.py): Optimize structures with the MACE model using the FIRE optimizer.

   1. **MACE UnitCellFilter FIRE** - [`examples/2_Structural_optimization/2.4_MACE_UnitCellFilter_FIRE.py`](2_Structural_optimization/2.4_MACE_UnitCellFilter_FIRE.py): Optimize structures with the MACE model using the UnitCellFilter FIRE optimizer.

   1. **MACE FrechetCellFilter FIRE** - [`examples/2_Structural_optimization/2.5_MACE_FrechetCellFilter_FIRE.py`](2_Structural_optimization/2.5_MACE_FrechetCellFilter_FIRE.py): Optimize structures with the MACE model using the FrechetCellFilter FIRE optimizer.

   1. **Batched MACE Gradient Descent** - [`examples/2_Structural_optimization/2.6_Batched_MACE_Gradient_Descent.py`](2_Structural_optimization/2.6_Batched_MACE_Gradient_Descent.py): Optimize multiple structures simultaneously using batched gradient descent with the MACE model.

   1. **Batched MACE FIRE** - [`examples/2_Structural_optimization/2.7_Batched_MACE_FIRE.py`](2_Structural_optimization/2.7_Batched_MACE_FIRE.py): Optimize multiple structures simultaneously using the batched FIRE optimizer with MACE.

   1. **Batched MACE UnitCellFilter Gradient Descent** - [`examples/2_Structural_optimization/2.8_Batched_MACE_UnitCellFilter_Gradient_Descent.py`](2_Structural_optimization/2.8_Batched_MACE_UnitCellFilter_Gradient_Descent.py): Optimize multiple structures and their unit cells using batched gradient descent with MACE.

   1. **Batched MACE UnitCellFilter FIRE** - [`examples/2_Structural_optimization/2.9_Batched_MACE_UnitCellFilter_FIRE.py`](2_Structural_optimization/2.9_Batched_MACE_UnitCellFilter_FIRE.py): Optimize multiple structures and their unit cells using the batched FIRE optimizer with MACE.

1. **Dynamics**

   1. **Lennard-Jones NVE** - [`examples/3_Dynamics/3.1_Lennard_Jones_NVE.py`](3_Dynamics/3.1_Lennard_Jones_NVE.py): Run molecular dynamics with the NVE ensemble using a Lennard-Jones model. Set up, simulate, and check energy conservation.

   1. **MACE NVE** - [`examples/3_Dynamics/3.2_MACE_NVE.py`](3_Dynamics/3.2_MACE_NVE.py): Run NVE molecular dynamics simulation with the MACE model.

   1. **MACE NVE with Cueq** - [`examples/3_Dynamics/3.3_MACE_NVE_cueq.py`](3_Dynamics/3.3_MACE_NVE_cueq.py): Run the MACE model in NVE with CuEq acceleration.

   1. **MACE NVT Langevin** - [`examples/3_Dynamics/3.4_MACE_NVT_Langevin.py`](3_Dynamics/3.4_MACE_NVT_Langevin.py): Run temperature-controlled molecular dynamics using the NVT Langevin integrator with MACE.

   1. **MACE NVT Nose-Hoover** - [`examples/3_Dynamics/3.5_MACE_NVT_Nose_Hoover.py`](3_Dynamics/3.5_MACE_NVT_Nose_Hoover.py): Run temperature-controlled molecular dynamics using the NVT Nose-Hoover integrator with MACE.

   1. **MACE NVT Nose-Hoover with Temperature Profile** - [`examples/3_Dynamics/3.6_MACE_NVT_Nose_Hoover_temp_profile.py`](3_Dynamics/3.6_MACE_NVT_Nose_Hoover_temp_profile.py): Simulate heating and cooling cycles using Nose-Hoover integrator with a temperature profile.

   1. **Lennard-Jones NPT Nose-Hoover** - [`examples/3_Dynamics/3.7_Lennard_Jones_NPT_Nose_Hoover.py`](3_Dynamics/3.7_Lennard_Jones_NPT_Nose_Hoover.py): Run pressure-controlled molecular dynamics using the NPT Nose-Hoover integrator with Lennard-Jones.

   1. **MACE NPT Nose-Hoover** - [`examples/3_Dynamics/3.8_MACE_NPT_Nose_Hoover.py`](3_Dynamics/3.8_MACE_NPT_Nose_Hoover.py): Run pressure-controlled molecular dynamics using the NPT Nose-Hoover integrator with MACE.

   1. **MACE NVT with Staggered Stress** - [`examples/3_Dynamics/3.9_MACE_NVT_staggered_stress.py`](3_Dynamics/3.9_MACE_NVT_staggered_stress.py): Use staggered stress calculations during NVT simulations with the MACE model.

   1. **Hybrid Swap Monte Carlo** - [`examples/3_Dynamics/3.10_Hybrid_swap_mc.py`](3_Dynamics/3.10_Hybrid_swap_mc.py): Combine molecular dynamics with Monte Carlo simulations using the MACE model.

1. **High-Level API**

   1. **High-Level API** - [`examples/4_High_level_api/4.1_high_level_api.py`](4_High_level_api/4.1_high_level_api.py): Integrate systems using the high-level API with different models and integrators.

1. **Workflow**

   1. **Workflow** - [`examples/5_Workflow/5.1_a2c_silicon.py`](5_Workflow/5.1_a2c_silicon.py): Run the a2c workflow with the MACE model.

   1. **Workflow** - [`examples/5_Workflow/5.4_Elastic.py`](5_Workflow/5.4_Elastic.py): Calculate elastic tensor, bulk modulus and shear modulus with MACE.

1. **Phonons**

   1. **Phonon DOS with MACE Batched** - [`examples/6_Phonons/6.1_Phonons_MACE.py`](6_Phonons/6.1_Phonons_MACE.py): Calculate DOS and band structure with MACE, batching over FC2 calculations.

   1. **Thermal Conductivity with MACE** - [`examples/6_Phonons/6.2_QuasiHarmonic_MACE.py`](6_Phonons/6.2_QuasiHarmonic_MACE.py): Calculates quasi-harmonic properties with MACE, batching over volumes and FC2 calculations.

   1. **Thermal Conductivity with MACE Batched** - [`examples/6_Phonons/6.3_Conductivity_MACE.py`](6_Phonons/6.3_Conductivity_MACE.py): Calculate the Wigner lattice conductivity with MACE, batching over FC2 and FC3 calculations.

Each example is self-contained and can be run independently to explore TorchSim capabilities. The examples cover basic model setup to advanced simulation techniques, providing a comprehensive overview of the library's features.
