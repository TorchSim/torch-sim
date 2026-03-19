"""Wrapper for DeepMD-kit model in TorchSim."""

import traceback
import warnings
from pathlib import Path
from typing import Any

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState


try:
    from ase.data import atomic_numbers as ase_atomic_numbers
    from deepmd.infer import DeepPot
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(f"DeepMD import failed: {traceback.format_exc()}", stacklevel=2)

    class DeepMDModel(ModelInterface):  # type: ignore[no-redef]
        """DeepMD model wrapper for torch-sim.

        This class is a placeholder for the DeepMDModel class.
        It raises an ImportError if DeepMD is not installed.
        """

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            """Dummy init for type checking."""
            raise err


class DeepMDModel(ModelInterface):
    """Computes energies, forces, and stresses with a DeepMD-kit model.

    DeepMD models build their own neighbor lists internally via
    ``forward_common`` -> ``extend_input_and_build_neighbor_list``.

    Attributes:
        model_wrapper: The DeepMD ``ModelWrapper`` containing the loaded model.
        type_map (list[str]): Element symbols in the model's type ordering.
        rcut (float): Cutoff radius used by the model.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
        head: str | None = None,
    ) -> None:
        """Initialize the DeepMD model for energy, force, and stress calculations.

        Args:
            model: Path to the DeepMD model file (``.pt`` or ``.pth``).
            device: The device to run computations on.
                Defaults to CUDA if available, otherwise CPU.
            dtype: The data type for output tensors. Defaults to ``torch.float64``.
            compute_forces: Whether to compute forces. Defaults to True.
            compute_stress: Whether to compute stress. Defaults to True.
            head: Model head name for multi-task models. Defaults to None.
        """
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"

        dp = DeepPot(str(Path(model).resolve()), head=head)
        self.model_wrapper = dp.deep_eval.dp
        self.model_wrapper.to(self._device)
        self.model_wrapper.eval()

        self.type_map: list[str] = dp.get_type_map()
        self.rcut: float = dp.get_rcut()

        self._z_to_type = self._build_type_lookup()

    def _build_type_lookup(self) -> torch.Tensor:
        """Build a lookup tensor mapping atomic numbers to DeepMD type indices.

        Returns:
            A 1-D tensor where ``lookup[atomic_number] = type_index``.
            Unmapped atomic numbers have index -1.
        """
        z_to_idx: dict[int, int] = {}
        for type_idx, symbol in enumerate(self.type_map):
            z = ase_atomic_numbers[symbol]
            z_to_idx[z] = type_idx

        max_z = max(z_to_idx) if z_to_idx else 0
        lookup = torch.full((max_z + 1,), -1, dtype=torch.long, device=self._device)
        for z, t in z_to_idx.items():
            lookup[z] = t
        return lookup

    def _call_model(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Run the DeepMD ModelWrapper and return the prediction dict.

        Args:
            coord: Coordinates with shape ``(nframes, natoms, 3)``.
            atype: Type indices with shape ``(nframes, natoms)``.
            box: Cell vectors with shape ``(nframes, 3, 3)``, or None.

        Returns:
            Raw prediction dictionary from the model.
        """
        out = self.model_wrapper(coord, atype, box=box, do_atomic_virial=False)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _virial_to_stress(
        self, virial: torch.Tensor, box: torch.Tensor
    ) -> torch.Tensor:
        """Convert virial tensor to stress tensor (vectorized over frames).

        Args:
            virial: Raw virial with shape ``(nframes, 3, 3)``.
            box: Cell vectors with shape ``(nframes, 3, 3)``.

        Returns:
            Stress tensor with shape ``(nframes, 3, 3)``.
        """
        volumes = torch.det(box).abs().unsqueeze(-1).unsqueeze(-1)
        return -0.5 * (virial + virial.transpose(-2, -1)) / volumes

    def forward(
        self, state: SimState, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Systems with the same atom count are batched together into a single
        model call using DeepMD's native ``nframes`` batch dimension.

        Args:
            state: State object containing positions, cell, atomic_numbers,
                and system_idx.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            Dictionary of computed properties:
                - ``energy``: System energies with shape ``[n_systems]``
                - ``forces``: Atomic forces with shape ``[n_atoms, 3]``
                  (if ``compute_forces=True``)
                - ``stress``: System stresses with shape ``[n_systems, 3, 3]``
                  (if ``compute_stress=True``)
        """
        counts = torch.bincount(state.system_idx)
        n_systems = len(counts)
        total_atoms = state.positions.shape[0]

        atom_types = self._z_to_type[state.atomic_numbers]
        row_cell = state.row_vector_cell
        has_pbc = state.pbc.any()

        unique_counts = counts.unique()

        if unique_counts.numel() == 1:
            return self._forward_uniform(
                state, atom_types, row_cell, counts, n_systems, total_atoms,
                has_pbc=has_pbc,
            )

        return self._forward_mixed(
            state, atom_types, row_cell, counts, n_systems, total_atoms,
            has_pbc=has_pbc,
        )

    def _forward_uniform(
        self,
        state: SimState,
        atom_types: torch.Tensor,
        row_cell: torch.Tensor,
        counts: torch.Tensor,
        n_systems: int,
        total_atoms: int,
        *,
        has_pbc: bool,
    ) -> dict[str, torch.Tensor]:
        """All systems have the same atom count: single model call."""
        natoms = counts[0].item()

        coord = state.positions.reshape(n_systems, natoms, 3)
        atype = atom_types.reshape(n_systems, natoms)
        box = row_cell if has_pbc else None

        pred = self._call_model(coord, atype, box)

        results: dict[str, torch.Tensor] = {}
        results["energy"] = pred["energy"].reshape(n_systems).to(self._dtype).detach()

        if self.compute_forces:
            results["forces"] = (
                pred["force"].reshape(total_atoms, 3).to(self._dtype).detach()
            )

        if self.compute_stress:
            if box is not None:
                virial = pred["virial"].reshape(n_systems, 3, 3)
                results["stress"] = (
                    self._virial_to_stress(virial, box).to(self._dtype).detach()
                )
            else:
                results["stress"] = torch.zeros(
                    n_systems, 3, 3, device=self._device, dtype=self._dtype
                )

        return results

    def _forward_mixed(
        self,
        state: SimState,
        atom_types: torch.Tensor,
        row_cell: torch.Tensor,
        counts: torch.Tensor,
        n_systems: int,
        total_atoms: int,
        *,
        has_pbc: bool,
    ) -> dict[str, torch.Tensor]:
        """Group systems by atom count: one batched call per group."""
        counts_list = counts.tolist()
        pos_splits = state.positions.split(counts_list)
        type_splits = atom_types.split(counts_list)

        energies = torch.empty(n_systems, device=self._device, dtype=self._dtype)
        forces = (
            torch.empty(total_atoms, 3, device=self._device, dtype=self._dtype)
            if self.compute_forces
            else None
        )
        stresses = (
            torch.zeros(n_systems, 3, 3, device=self._device, dtype=self._dtype)
            if self.compute_stress
            else None
        )

        groups: dict[int, list[int]] = {}
        for i, c in enumerate(counts_list):
            groups.setdefault(c, []).append(i)

        offsets = torch.zeros(n_systems, device=self._device, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)[:-1]

        for natoms, indices in groups.items():
            nf = len(indices)
            idx = torch.tensor(indices, device=self._device)

            coord = torch.stack([pos_splits[i] for i in indices])
            atype = torch.stack([type_splits[i] for i in indices])
            box = row_cell[idx] if has_pbc else None

            pred = self._call_model(coord, atype, box)

            energies[idx] = pred["energy"].reshape(nf).to(self._dtype).detach()

            if self.compute_forces and forces is not None:
                f = pred["force"].reshape(nf, natoms, 3).to(self._dtype).detach()
                for j, sys_idx in enumerate(indices):
                    start = offsets[sys_idx].item()
                    forces[start : start + natoms] = f[j]

            if self.compute_stress and stresses is not None and box is not None:
                virial = pred["virial"].reshape(nf, 3, 3)
                stresses[idx] = (
                    self._virial_to_stress(virial, box).to(self._dtype).detach()
                )

        results: dict[str, torch.Tensor] = {"energy": energies}
        if self.compute_forces and forces is not None:
            results["forces"] = forces
        if self.compute_stress and stresses is not None:
            results["stress"] = stresses

        return results
