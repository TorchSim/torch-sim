"""FairChem model for computing energies, forces and stresses.

This module provides a PyTorch implementation of the FairChem model.
"""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from fairchem.core.models.model_registry import model_name_to_local_file
from torch_geometric.data import Batch

from torchsim.models.interface import ModelInterface


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

DTYPE_DICT = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}


class FairChemModel(torch.nn.Module, ModelInterface):
    """Computes energies, forces and stresses using a FairChem model.

    Attributes:
        pbc (bool): Whether to use periodic boundary conditions
        neighbor_list_fn (Callable | None): The neighbor list function to use
        r_max (float): Maximum cutoff radius for atomic interactions
        config (dict): Model configuration dictionary
        trainer: The FairChem trainer object
        data_object (Batch): Data object containing system information
        implemented_properties (list): List of implemented model outputs
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
        pbc: bool = True,
        r_max: float | None = None,  # noqa: ARG002
        dtype: torch.dtype | None = None,
        compute_stress: bool = False,
    ) -> None:
        """Initialize the FairChemModel.

        Args:
            model: Path to model checkpoint
            atomic_numbers_list: List of atomic numbers for each system
            neighbor_list_fn: Neighbor list function (not currently supported)
            config_yml: Path to config YAML file
            model_name: Name of pretrained model
            local_cache: Path to local model cache
            trainer: Name of trainer to use
            cpu: Whether to use CPU instead of GPU
            seed: Random seed for reproducibility
            pbc: Whether to use periodic boundary conditions
            r_max: Maximum cutoff radius (overrides model default)
            dtype: Data type to use for the model
            compute_stress: Whether to compute stress
        """
        setup_imports()
        setup_logging()
        super().__init__()

        self._dtype = dtype or torch.float32
        self._compute_stress = compute_stress
        self._compute_force = True

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

        self.pbc = pbc
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
                    config["model"]["backbone"].update({"dtype": DTYPE_DICT[dtype]})
                    for key in config["model"]["heads"]:
                        config["model"]["heads"][key].update({"dtype": DTYPE_DICT[dtype]})
                except KeyError:
                    print("dtype not found in backbone, using default float32")
        else:
            config["model"]["use_pbc"] = pbc
            config["model"]["use_pbc_single"] = False
            if dtype is not None:
                try:
                    config["model"].update({"dtype": DTYPE_DICT[dtype]})
                except KeyError:
                    print("dtype not found in backbone, using default dtype")

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
            self.trainer.model.to(dtype=self.dtype)

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

        self.implemented_properties = list(self.config["outputs"])

        self._device = self.trainer.device

        stress_output = "stress" in self.implemented_properties
        if not stress_output and compute_stress:
            raise NotImplementedError("Stress output not implemented for this model")

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: dict | None = None
    ) -> None:
        """Load existing trained model.

        Args:
            checkpoint_path: string
                Path to trained model
            checkpoint: dict
                A pretrained checkpoint dict
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path, checkpoint, inference_only=True)
        except NotImplementedError:
            print("Unable to load checkpoint!")

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor | None = None,
        **_,
    ) -> dict:  # TODO: what are the shapes?
        """Forward pass of the model.

        Args:
            positions: Atomic positions tensor
            cell: Box vectors tensor
            batch: Batch tensor
            atomic_numbers: Atomic numbers tensor

        Returns:
            Dictionary of model predictions
        """
        if positions.device != self._device:
            positions = positions.to(self._device)
        if cell.device != self._device:
            cell = cell.to(self._device)

        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int)

        natoms = torch.bincount(batch)
        pbc = torch.tensor(
            [self.pbc, self.pbc, self.pbc] * len(natoms), dtype=torch.bool
        ).view(-1, 3)
        fixed = torch.zeros((batch.size(0), natoms.sum()), dtype=torch.int)
        self.data_object = Batch(
            pos=positions,
            cell=cell,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            batch=batch,
            fixed=fixed,
            pbc=pbc,
        )

        if self._dtype is not None:
            self.data_object.pos = self.data_object.pos.to(self._dtype)
            self.data_object.cell = self.data_object.cell.to(self._dtype)

        predictions = self.trainer.predict(
            self.data_object, per_image=False, disable_tqdm=True
        )

        results = {}

        for key in predictions:
            _pred = predictions[key]
            if key in self._reshaped_props:
                _pred = _pred.reshape(self._reshaped_props.get(key)).squeeze()
            results[key] = _pred

        results["energy"] = results["energy"].squeeze(dim=1)
        return results
