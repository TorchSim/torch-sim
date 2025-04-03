## Introduction

TorchSim enables a lot of cool research! We wanted a place where the community to show off their work in an accessible and reproducible way so we created the workflows folder. Currently, this contains a reimplementation of the A2C method by [Aykol et al.](https://arxiv.org/abs/2310.01117) but we intend to expand it to include workflows for phonons, elastic properties, and more.


## Implemented Workflows

As a start, we implemented the A2C method created by [Aykol et al.](https://arxiv.org/abs/2310.01117) and originally [implemented in jax-md](https://github.com/jax-md/jax-md/blob/main/jax_md/a2c/a2c_workflow.py). The [a2c.py](/torch_sim/workflows/a2c.py) file contains many of the core operations in the paper, which are then linked together in the [a2c_silicon.py](/examples/scripts/5_Workflow/5.2_a2c_silicon_batched.py) file.
