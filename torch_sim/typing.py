"""Types used across torch-sim."""

from typing import Literal

import torch


MemoryScaler = Literal["n_atoms_x_density", "n_atoms"]


StateKey = Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "batch"]
StateDict = dict[StateKey, torch.Tensor]
