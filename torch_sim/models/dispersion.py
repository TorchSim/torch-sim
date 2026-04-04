"""DFT-D3(BJ) dispersion correction model.

Wraps the ``nvalchemiops`` Warp-accelerated DFT-D3 implementation as a
:class:`~torch_sim.models.interface.ModelInterface`, with full PBC, stress
(virial), and batched system support.

References:
    - Grimme et al., J. Chem. Phys. 132, 154104 (2010).
      https://doi.org/10.1063/1.3382344
    - Grimme et al., J. Comput. Chem. 32, 1456-1465 (2011).
      https://doi.org/10.1002/jcc.21759
    - nvalchemi-toolkit-ops: https://github.com/NVIDIA/nvalchemi-toolkit-ops
"""

from __future__ import annotations

import traceback
import warnings
from typing import TYPE_CHECKING, Any

import torch

from torch_sim._duecredit import dcite
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.units import UnitConversion


try:
    from nvalchemiops.torch.interactions.dispersion import D3Parameters
    from nvalchemiops.torch.interactions.dispersion import dftd3 as nvalchemiops_dftd3
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(f"nvalchemiops import failed: {traceback.format_exc()}", stacklevel=2)

    class D3Parameters:
        """Placeholder when nvalchemiops is not installed."""

        def __init__(self, *_a: Any, _err: Exception = exc, **_kw: Any) -> None:
            """Raise the original import error."""
            raise _err

    def nvalchemiops_dftd3(*_a: Any, _err: Exception = exc, **_kw: Any) -> Any:
        """Raise the original import error."""
        raise _err


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.state import SimState

_BOHR_TO_ANG = UnitConversion.Bohr_to_Ang
_ANG_TO_BOHR = UnitConversion.Ang_to_Bohr
_HARTREE_TO_EV = UnitConversion.Hartree_to_eV
_FORCE_CONV = _HARTREE_TO_EV / _BOHR_TO_ANG  # Hartree/Bohr -> eV/Ang


class D3DispersionModel(ModelInterface):
    """DFT-D3(BJ) dispersion correction as a :class:`ModelInterface`.

    Computes DFT-D3 energies, forces, and (optionally) stresses via the
    ``nvalchemiops`` Warp GPU/CPU kernels.  All user-facing quantities are in
    metal units (Angstrom / eV); unit conversion to and from atomic units
    (Bohr / Hartree) is handled internally.

    Args:
        a1: BJ damping parameter (dimensionless, functional-dependent).
        a2: BJ damping parameter (in Bohr, functional-dependent).
        s8: C8 scaling factor (dimensionless, functional-dependent).
        s6: C6 scaling factor. Defaults to 1.0.
        d3_params: Reference D3 parameters (rcov, r4r2, c6ab, cn_ref).
        cutoff: Neighbor-list cutoff in **Angstrom**.
        device: Compute device. Defaults to CUDA if available, else CPU.
        dtype: Floating-point dtype. Defaults to ``torch.float64``.
        compute_forces: Whether to return forces. Defaults to True.
        compute_stress: Whether to return stress. Defaults to True.
        neighbor_list_fn: Neighbor-list constructor. Defaults to ``torchsim_nl``.

    Example::

        model = D3DispersionModel(
            a1=0.4289,
            a2=4.4407,
            s8=0.7875,
            d3_params=params,
            cutoff=50.0,
        )
        results = model(sim_state)
    """

    @dcite("10.1063/1.3382344")
    @dcite("10.1002/jcc.21759")
    def __init__(
        self,
        a1: float,
        a2: float,
        s8: float,
        *,
        s6: float = 1.0,
        d3_params: D3Parameters | None = None,
        cutoff: float = 95.0 * _BOHR_TO_ANG,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
        neighbor_list_fn: Callable = torchsim_nl,
    ) -> None:
        """Initialize the D3 dispersion model."""
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = cutoff
        self.a1 = a1
        self.a2 = a2
        self.s8 = s8
        self.s6 = s6
        self.d3_params = d3_params

    def forward(self, state: SimState, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Compute D3 dispersion energy, forces, and stress.

        Args:
            state: Simulation state (positions in Angstrom, cell in Angstrom).
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict with ``"energy"`` [n_systems], ``"forces"`` [n_atoms, 3],
            and (if ``compute_stress``) ``"stress"`` [n_systems, 3, 3].
        """
        edge_index, _mapping_system, unit_shifts = self.neighbor_list_fn(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            self.cutoff,
            state.system_idx,
        )
        n_atoms = state.positions.shape[0]
        neighbor_ptr = torch.zeros(
            n_atoms + 1, dtype=torch.int32, device=state.positions.device
        )
        neighbor_ptr[1:] = (
            torch.bincount(edge_index[0], minlength=n_atoms).cumsum(0).to(torch.int32)
        )
        positions_bohr = state.positions * _ANG_TO_BOHR
        cell_bohr = state.row_vector_cell.contiguous() * _ANG_TO_BOHR
        numbers = state.atomic_numbers.to(torch.int32)
        unit_shifts_int = unit_shifts.to(torch.int32)
        edge_index_int = edge_index.to(torch.int32)
        d3_out = nvalchemiops_dftd3(
            positions=positions_bohr,
            numbers=numbers,
            a1=self.a1,
            a2=self.a2,
            s8=self.s8,
            s6=self.s6,
            d3_params=self.d3_params,  # type: ignore[arg-type]
            neighbor_list=edge_index_int,
            neighbor_ptr=neighbor_ptr,
            cell=cell_bohr,
            unit_shifts=unit_shifts_int,
            batch_idx=state.system_idx.to(torch.int32),
            compute_virial=self._compute_stress,
            num_systems=state.n_systems,
        )

        results: dict[str, torch.Tensor] = {
            "energy": (d3_out[0] * _HARTREE_TO_EV).to(self._dtype).detach(),
            "forces": (d3_out[1] * _FORCE_CONV).to(self._dtype).detach(),
        }
        if self._compute_stress:
            # d3_out[3] is only defined if compute_virial is True
            volumes = state.volume.unsqueeze(-1).unsqueeze(-1)
            stress = (d3_out[3] * _HARTREE_TO_EV) / volumes  # type: ignore[index]
            results["stress"] = stress.to(self._dtype).detach()
        return results
