# Reproducibility

Molecular dynamics trajectories are often not exactly reproducible across runs, even
when starting from the same initial structure and parameters.

Two common sources are:

- **Stochastic integrators** such as Langevin, which add random forces
- **Non-deterministic GPU operations**, where floating-point reductions may execute
  in different orders

For many MD tasks this is acceptable because sampling and ensemble statistics matter
more than matching a step-by-step trajectory. If you need repeatable trajectories, use
deterministic settings.

## Deterministic setup in PyTorch

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

If deterministic mode raises a CuBLAS error, ensure `CUBLAS_WORKSPACE_CONFIG` is set
before running your script.

## Deterministic vs stochastic integrators in TorchSim

- `ts.Integrator.nvt_langevin` and `ts.Integrator.npt_langevin` include stochastic
  terms by design.
- `ts.Integrator.nvt_nose_hoover` and `ts.Integrator.nve` are deterministic at the
  algorithmic level.

If you want to compare run-to-run determinism in TorchSim, use a deterministic
integrator such as Nosé-Hoover:

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

In practice, exact reproducibility also depends on hardware, driver/library versions,
and precision choices.

## Seeding stochastic integrators

Set `rng` on the state before calling any init function. You can pass an integer seed
or a `torch.Generator`:

```python
import torch_sim as ts

sim_state = ts.initialize_state(atoms, device, dtype)
sim_state.rng = 42  # or: sim_state.rng = ts.coerce_prng(42, device=device)

state = ts.nvt_langevin_init(state=sim_state, model=model, kT=kT)
```

The `rng` generator controls **both** the initial momenta sampling **and** all
per-step stochastic noise (Langevin OU noise, V-Rescale draws, C-Rescale barostat
noise, etc.). It is stored on the state and automatically advances on every step,
so running the same seed twice produces identical trajectories.

### Serialising the RNG state

`torch.Generator` state can be saved and restored for checkpoint/restart:

```python
# save
rng_state = state.rng.get_state()
torch.save(rng_state, "rng_state.pt")

# restore
gen = torch.Generator(device=state.device)
gen.set_state(torch.load("rng_state.pt"))
state.rng = gen
```

## Batching and reproducibility

Because TorchSim runs batched simulations, all systems in a batch share a single
`torch.Generator`. Random numbers are drawn in a fixed order each step, so
**identical batch composition** is required for exact reproducibility. Changing which
systems are in a batch (or their order) will consume random numbers differently and
cause trajectories to diverge.

If strict reproducibility is required, keep your batching setup fixed.

## Related issues

- [Issue #286](https://github.com/TorchSim/torch-sim/issues/286): seed/generator
  handling for stochastic step functions
- [Issue #51](https://github.com/TorchSim/torch-sim/issues/51): stable random
  behavior across batch sizes
- [Issue #314](https://github.com/TorchSim/torch-sim/issues/314): general
  reproducibility guidance
