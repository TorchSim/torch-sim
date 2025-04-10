"""TorchSim wrapper for MatterSim models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState, StateDict, state_to_atoms
from torch_sim.units import MetalUnits


try:
    from mattersim.datasets.utils.convertor import GraphConvertor
    from mattersim.forcefield.potential import batch_to_dict
    from torch_geometric.loader.dataloader import Collater

except ImportError:

    class MatterSimModel(torch.nn.Module, ModelInterface):
        """MatterSim model wrapper for torch_sim.

        This class is a placeholder for the MatterSimModel class.
        It raises an ImportError if sevenn is not installed.
        """

        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            """Dummy constructor to raise ImportError."""
            raise ImportError("sevenn must be installed to use this model.")


if TYPE_CHECKING:
    from mattersim.forcefield import Potential


class MatterSimModel(torch.nn.Module, ModelInterface):
    """Computes atomistic energies, forces and stresses using an MatterSim model.

    This class wraps an MatterSim model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, configuration, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    Examples:
        >>> model = MatterSimModel(model=loaded_matersim_model)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: Potential,
        *,  # force remaining arguments to be keyword-only
        stress_weight: float = MetalUnits.pressure * 1e4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the MatterSimModel with specified configuration.

        Loads an MatterSim model from either a model object or a model path.
        Sets up the model parameters for subsequent use in energy and force calculations.

        Args:
            model (Potential): The MatterSim model to wrap.
            stress_weight (float): Stress weight to use to scale the stress units.
                Defaults to value of ase.units.GPa to match MatterSimCalculator default.
            device (torch.device | str | None): Device to run the model on
            dtype (torch.dtype | None): Data type for computation
        """
        super().__init__()

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

        self._dtype = dtype or torch.float32
        self._memory_scales_with = "n_atoms"  # scale memory with n_atoms due to triplets
        self._compute_stress = True
        self._compute_forces = True

        self.stress_weight = stress_weight

        self.model = model.to(self._device)
        self.model = self.model.eval()

        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        model_args = self.model.model.model_args
        self.two_body_cutoff = model_args["cutoff"]
        self.three_body_cutoff = model_args["threebody_cutoff"]

        self.convertor = GraphConvertor(
            model_type="m3gnet",
            twobody_cutoff=self.two_body_cutoff,
            has_threebody=True,
            threebody_cutoff=self.three_body_cutoff,
        )

        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
        ]

    def forward(self, state: SimState | StateDict) -> dict[str, torch.Tensor]:
        """Perform forward pass to compute energies, forces, and other properties.

        Takes a simulation state and computes the properties implemented by the model,
        such as energy, forces, and stresses.

        Args:
            state (SimState | StateDict): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3],
                    if compute_stress is True

        Notes:
            The state is automatically transferred to the model's device if needed.
            All output tensors are detached from the computation graph.
        """
        if isinstance(state, dict):
            state = SimState(**state, masses=torch.ones_like(state["positions"]))

        if state.device != self._device:
            state = state.to(self._device)

        atoms_list = state_to_atoms(state)
        data_list = [self.convertor.convert(atoms) for atoms in atoms_list]
        batched_data = Collater([], follow_batch=None, exclude_keys=None)(data_list)
        output = self.model.forward(
            batch_to_dict(batched_data, device=self.device),
            include_forces=self.compute_forces,
            include_stresses=self.compute_stress,
        )

        results = {}
        results["energy"] = output["total_energy"].detach()
        results["forces"] = output["forces"].detach()
        results["stress"] = self.stress_weight * output["stresses"].detach()

        return results
