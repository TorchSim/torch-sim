# Reproducibility

Molecular dynamics trajectories are often not exactly reproducible across runs, even when starting from the same initial structure and parameters.

Two common sources are:

- **Non-deterministic GPU operations**, where floating-point reductions may execute
  in different orders
- **Stochastic integrators** such as Langevin, which add random forces

For many MD tasks this is acceptable because sampling and ensemble statistics matter more than matching a step-by-step trajectory. If you need repeatable trajectories, use deterministic settings.

## Global Deterministic setup in PyTorch

Enable deterministic algorithms and seed random number generators:

```python
import os
import random

import numpy as np
import torch

# Required by CUDA/cuBLAS for some deterministic GEMM paths.
# Set this before any CUDA operations.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.use_deterministic_algorithms(True)
```

If deterministic mode raises a CuBLAS error, ensure `CUBLAS_WORKSPACE_CONFIG` is set before running your script.

One of the main reasons for seeding the global states is to control the determinism of any `ModelInterface` being used to make predictions. It should be noted that models may implement their own PRNG solutions that do not draw from global seeds. As such, setting the global seeds doesn't necessarily ensure deterministic behavior. Please check each model being used on a case-by-case basis if determinism is critical to your workflow.

## Seeding TorchSim SimStates

In PyTorch, `torch.manual_seed()` only seeds the default global generator, i.e. the generator used in `torch.rand(..., generator=None)` if the generator isn't specified. An explicit `torch.Generator()` is independent and is initialized at the time of calling from the OS entropy source. Since TorchSim internally uses a `torch.Generator()` in the form of `SimState.rng` for all algorithmic randomness in core code setting the global seed is insufficient to ensure determinism within TorchSim. **You must explicitly seed your SimStates to ensure determinism**.

```python
sim_state = ts.initialize_state(atoms, device, dtype)
sim_state.rng = 42  # required for reproducibility — torch.manual_seed() has no effect here
```

### Deterministic vs stochastic integrators in TorchSim

- `ts.Integrator.nvt_langevin` and `ts.Integrator.npt_langevin` include stochastic
  terms by design. When seeded via `state.rng`, they produce identical trajectories.
  The `rng` generator controls **both** the initial momenta sampling **and** all per-step stochastic noise (Langevin OU noise, V-Rescale draws, C-Rescale barostat noise, etc.). It is stored on the state and automatically advances on every step, so running the same seed twice produces identical trajectories.
- `ts.Integrator.nvt_nose_hoover` and `ts.Integrator.nve` are deterministic at the
  algorithmic level and require no seeding.

For the simplest path to reproducibility, use a deterministic integrator such as Nosé-Hoover:

```python
import torch_sim as ts

state = ts.integrate(
    system=atoms,
    model=model,
    n_steps=500,
    timestep=0.001,
    temperature=300,
    integrator=ts.Integrator.nvt_nose_hoover,
)
```

In practice, exact reproducibility also depends on hardware, driver/library versions, and precision choices.

### Batching and reproducibility

Because TorchSim runs batched simulations, all systems in a batch share a single `torch.Generator`. Random numbers are drawn in a fixed order each step, so **identical batch composition** is required for exact reproducibility. Changing which systems are in a batch (or their order) will consume random numbers differently and cause trajectories to diverge.

If strict reproducibility is required, keep your batching setup fixed.

### Serialising the RNG state

If you wish to be able to resume a session and ensure determinism you need to persist and reload the `torch.Generator` state. This can be done using `torch.save()` and `torch.Generator().set_state()`:

```python
# save
rng_state = state.rng.get_state()
torch.save(rng_state, "rng_state.pt")

# restore
gen = torch.Generator(device=state.device)
gen.set_state(torch.load("rng_state.pt"))
state.rng = gen
```
