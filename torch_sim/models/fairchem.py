"""Wrapper for FairChem ecosystem models in TorchSim.

This module provides a TorchSim wrapper of the FairChem models for computing
energies, forces, and stresses of atomistic systems. It serves as a wrapper around
the FairChem library, integrating it with the torch_sim framework to enable seamless
simulation of atomistic systems with machine learning potentials.

The FairChemModel class adapts FairChem models to the ModelInterface protocol,
allowing them to be used within the broader torch_sim simulation framework.

Notes:
    This implementation requires FairChem to be installed and accessible.
    It supports various model configurations through configuration files or
    pretrained model checkpoints.
"""

from __future__ import annotations

import copy
import typing
from types import MappingProxyType

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState, StateDict


try:
    from fairchem.core.common.registry import registry
    from fairchem.core.common.utils import (
        load_config,
        setup_imports,
        setup_logging,
        update_config,
    )
    from fairchem.core.models.model_registry import model_name_to_local_file
    from torch_geometric.data import Batch

except ImportError:

    class FairChemModel(torch.nn.Module, ModelInterface):
        """FairChem model wrapper for torch_sim.

        This class is a placeholder for the FairChemModel class.
        It raises an ImportError if FairChem is not installed.
        """

        def __init__(self, *_args: typing.Any, **_kwargs: typing.Any) -> None:
            """Dummy init for type checking."""
            raise ImportError("FairChem must be installed to use this model.")


if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_DTYPE_DICT = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}


class FairChemModel(torch.nn.Module, ModelInterface):
    """Computes atomistic energies, forces and stresses using a FairChem model.

    This class wraps a FairChem model to compute energies, forces, and stresses for
    atomistic systems. It handles model initialization, checkpoint loading, and
    provides a forward pass that accepts a SimState object and returns model
    predictions.

    The model can be initialized either with a configuration file or a pretrained
    checkpoint. It supports various model architectures and configurations supported by
    FairChem.

    Attributes:
        neighbor_list_fn (Callable | None): Function to compute neighbor lists
        config (dict): Complete model configuration dictionary
        trainer: FairChem trainer object that contains the model
        data_object (Batch): Data object containing system information
        implemented_properties (list): Model outputs the model can compute
        pbc (bool): Whether periodic boundary conditions are used
        _dtype (torch.dtype): Data type used for computation
        _compute_stress (bool): Whether to compute stress tensor
        _compute_forces (bool): Whether to compute forces
        _device (torch.device): Device where computation is performed
        _reshaped_props (dict): Properties that need reshaping after computation

    Examples:
        >>> model = FairChemModel(model="path/to/checkpoint.pt", compute_stress=True)
        >>> results = model(state)
    """

    _reshaped_props = MappingProxyType(
        {"stress": (-1, 3, 3), "dielectric_tensor": (-1, 3, 3)}
    )

    def __init__(  # noqa: C901, PLR0915
        self,
        model: str | Path | None,
        neighbor_list_fn: Callable | None = None,
        *,  # force remaining arguments to be keyword-only
        config_yml: str | None = None,
        model_name: str | None = None,
        local_cache: str | None = None,
        trainer: str | None = None,
        cpu: bool = False,
        seed: int | None = None,
        dtype: torch.dtype | None = None,
        compute_stress: bool = False,
        pbc: bool = True,
        disable_amp: bool = True,
    ) -> None:
        """Initialize the FairChemModel with specified configuration.

        Loads a FairChem model from either a checkpoint path or a configuration file.
        Sets up the model parameters, trainer, and configuration for subsequent use
        in energy and force calculations.

        Args:
            model (str | Path | None): Path to model checkpoint file
            neighbor_list_fn (Callable | None): Function to compute neighbor lists
                (not currently supported)
            config_yml (str | None): Path to configuration YAML file
            model_name (str | None): Name of pretrained model to load
            local_cache (str | None): Path to local model cache directory
            trainer (str | None): Name of trainer class to use
            cpu (bool): Whether to use CPU instead of GPU for computation
            seed (int | None): Random seed for reproducibility
            dtype (torch.dtype | None): Data type to use for computation
            compute_stress (bool): Whether to compute stress tensor
            pbc (bool): Whether to use periodic boundary conditions
            disable_amp (bool): Whether to disable AMP
        Raises:
            RuntimeError: If both model_name and model are specified
            NotImplementedError: If local_cache is not set when model_name is used
            NotImplementedError: If custom neighbor list function is provided
            ValueError: If stress computation is requested but not supported by model

        Notes:
            Either config_yml or model must be provided. The model loads configuration
            from the checkpoint if config_yml is not specified.
        """
        setup_imports()
        setup_logging()
        super().__init__()

        self._dtype = dtype or torch.float32
        self._compute_stress = compute_stress
        self._compute_forces = True
        self._memory_scales_with = "n_atoms"
        self.pbc = pbc

        if model_name is not None:
            if model is not None:
                raise RuntimeError(
                    "model_name and checkpoint_path were both specified, "
                    "please use only one at a time"
                )
            if local_cache is None:
                raise NotImplementedError(
                    "Local cache must be set when specifying a model name"
                )
            model = model_name_to_local_file(
                model_name=model_name, local_cache=local_cache
            )

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or model is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(config_yml)
                if len(duplicates_warning) > 0:
                    print(
                        "Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        "Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml

            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(model, map_location=torch.device("cpu"))
            config = checkpoint["config"]

        if trainer is not None:
            config["trainer"] = trainer
        else:
            config["trainer"] = config.get("trainer", "ocp")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        self.neighbor_list_fn = neighbor_list_fn

        if neighbor_list_fn is None:
            # Calculate the edge indices on the fly
            config["model"]["otf_graph"] = True
        else:
            raise NotImplementedError(
                "Custom neighbor list is not supported for FairChemModel."
            )

        if "backbone" in config["model"]:
            config["model"]["backbone"]["use_pbc"] = pbc
            config["model"]["backbone"]["use_pbc_single"] = False
            if dtype is not None:
                try:
                    config["model"]["backbone"].update({"dtype": _DTYPE_DICT[dtype]})
                    for key in config["model"]["heads"]:
                        config["model"]["heads"][key].update(
                            {"dtype": _DTYPE_DICT[dtype]}
                        )
                except KeyError:
                    print(
                        "WARNING: dtype not found in backbone, using default model dtype"
                    )
        else:
            config["model"]["use_pbc"] = pbc
            config["model"]["use_pbc_single"] = False
            if dtype is not None:
                try:
                    config["model"].update({"dtype": _DTYPE_DICT[dtype]})
                except KeyError:
                    print(
                        "WARNING: dtype not found in backbone, using default model dtype"
                    )

        ### backwards compatibility with OCP v<2.0
        config = update_config(config)

        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = str(model)
        del config["dataset"]["src"]

        self.trainer = registry.get_trainer_class(config["trainer"])(
            task=config.get("task", {}),
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_functions=config["loss_functions"],
            evaluation_metrics=config["evaluation_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=False if dtype is not None else config.get("amp", False),
            inference_only=True,
        )

        if dtype is not None:
            # Convert model parameters to specified dtype
            self.trainer.model = self.trainer.model.to(dtype=self.dtype)

        if model is not None:
            self.load_checkpoint(checkpoint_path=model, checkpoint=checkpoint)

        seed = seed if seed is not None else self.trainer.config["cmd"]["seed"]
        if seed is None:
            print(
                "No seed has been set in model checkpoint or OCPCalculator! Results may "
                "not be reproducible on re-run"
            )
        else:
            self.trainer.set_seed(seed)

        if disable_amp:
            self.trainer.scaler = None

        self.implemented_properties = list(self.config["outputs"])

        self._device = self.trainer.device

        stress_output = "stress" in self.implemented_properties
        if not stress_output and compute_stress:
            raise NotImplementedError("Stress output not implemented for this model")

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: dict | None = None
    ) -> None:
        """Load an existing trained model checkpoint.

        Loads model parameters from a checkpoint file or dictionary,
        setting the model to inference mode.

        Args:
            checkpoint_path (str): Path to the trained model checkpoint file
            checkpoint (dict | None): A pretrained checkpoint dictionary. If provided,
                this dictionary is used instead of loading from checkpoint_path.

        Notes:
            If loading fails, a message is printed but no exception is raised.
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path, checkpoint, inference_only=True)
        except NotImplementedError:
            print("Unable to load checkpoint!")

    def forward(self, state: SimState | StateDict) -> dict:
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

        if state.batch is None:
            state.batch = torch.zeros(state.positions.shape[0], dtype=torch.int)

        if self.pbc != state.pbc:
            raise ValueError(
                "PBC mismatch between model and state. "
                "For FairChemModel PBC needs to be defined in the model class."
            )

        natoms = torch.bincount(state.batch)
        pbc = torch.tensor(
            [state.pbc, state.pbc, state.pbc] * len(natoms), dtype=torch.bool
        ).view(-1, 3)
        fixed = torch.zeros((state.batch.size(0), natoms.sum()), dtype=torch.int)
        self.data_object = Batch(
            pos=state.positions,
            cell=state.row_vector_cell,
            atomic_numbers=state.atomic_numbers,
            natoms=natoms,
            batch=state.batch,
            fixed=fixed,
            pbc=pbc,
        )

        if self.dtype is not None:
            self.data_object.pos = self.data_object.pos.to(self.dtype)
            self.data_object.cell = self.data_object.cell.to(self.dtype)

        predictions = self.trainer.predict(
            self.data_object, per_image=False, disable_tqdm=True
        )

        results = {}

        for key in predictions:
            _pred = predictions[key]
            if key in self._reshaped_props:
                _pred = _pred.reshape(self._reshaped_props.get(key)).squeeze()
            results[key] = _pred.detach()

        results["energy"] = results["energy"].squeeze(dim=1)
        if results.get("stress") is not None and len(results["stress"].shape) == 2:
            results["stress"] = results["stress"].unsqueeze(dim=0)
        return results
