"""M3GNet model.

A wrapper class for the M3GNet force field model.
"""

from typing import Any

import torch
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.jit_compile_tools.jit import compile_mode
from torch import nn


@compile_mode("script")
class M3GnetModel(nn.Module):
    """A wrapper class for the force field model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allow_tf32: bool = False,
        compute_force: bool = False,
        compute_stress: bool = False,
        **_kwargs: Any,
    ) -> None:
        """Initialize the potential.

        Args:
            model: A force field model
            device: Device to run on
            allow_tf32: Whether to allow TF32 precision
            compute_force: Whether to compute forces
            compute_stress: Whether to compute stresses
            **kwargs: Additional arguments
        """
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

        self.compute_force = compute_force
        self.compute_stress = compute_stress

        self.model = M3Gnet(device=device, **model["model_args"]).to(device)
        self.model.load_state_dict(model["model"], strict=True)
        self.model.eval()
        self.device = device
        self.to(device)

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_dict: Dictionary containing necessary info.
                   The `batch_to_dict` method could convert a graph_batch from
                   pyg dataloader to the input dictionary.
            dataset_idx: Used for multi-head model, set to -1 by default

        Returns:
            Dictionary containing energies, forces and stresses
        """
        # Move input tensors to device
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)

        # Initialize strain and volume
        strain = torch.zeros_like(input_dict["cell"], device=self.device)
        volume = torch.linalg.det(input_dict["cell"])

        results = {}
        if self.compute_force:
            input_dict["atom_pos"].requires_grad_(True)  # noqa: FBT003
        if self.compute_stress:
            strain.requires_grad_(True)  # noqa: FBT003
            input_dict["cell"] = torch.matmul(
                input_dict["cell"],
                (torch.eye(3, device=self.device)[None, ...] + strain),
            )
            strain_augment = torch.repeat_interleave(
                strain, input_dict["num_atoms"], dim=0
            )
            input_dict["atom_pos"] = torch.einsum(
                "bi, bij -> bj",
                input_dict["atom_pos"],
                (torch.eye(3, device=self.device)[None, ...] + strain_augment),
            )
            volume = torch.linalg.det(input_dict["cell"])

        energies = self.model.forward(input_dict, dataset_idx)
        results["energy"] = energies

        # Only take first derivative if only force is required
        if self.compute_force and not self.compute_stress:
            grad_outputs: list[torch.Tensor | None] = [
                torch.ones_like(
                    energies,
                )
            ]
            grad = torch.autograd.grad(
                outputs=[
                    energies,
                ],
                inputs=[input_dict["atom_pos"]],
                grad_outputs=grad_outputs,
                create_graph=self.model.training,
            )

            # Dump out gradient for forces
            force_grad = grad[0]
            if force_grad is not None:
                forces = torch.neg(force_grad)
                results["forces"] = forces

        # Take derivatives up to second order
        # if both forces and stresses are required
        if self.compute_force and self.compute_stress:
            grad_outputs: list[torch.Tensor | None] = [
                torch.ones_like(
                    energies,
                )
            ]
            grad = torch.autograd.grad(
                outputs=[
                    energies,
                ],
                inputs=[input_dict["atom_pos"], strain],
                grad_outputs=grad_outputs,
                create_graph=self.model.training,
            )

            # Dump out gradient for forces and stresses
            force_grad = grad[0]
            stress_grad = grad[1]

            if force_grad is not None:
                forces = torch.neg(force_grad)
                results["forces"] = forces

            if stress_grad is not None:
                stresses = 1 / volume[:, None, None] * stress_grad
                results["stress"] = stresses

        return results
