"""Nudged Elastic Band (NEB) workflow.

This module implements the Nudged Elastic Band method for finding minimum energy
paths between two given atomic configurations.
"""

import inspect
import logging
import os  # Import os for fsync
import pickle  # Import pickle
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import (
    CellFireState,
    FireState,
    OptimState,
    fire_init,
    fire_step,
    gradient_descent_init,
    gradient_descent_step,
)
from torch_sim.optimizers.cell_filters import CellFilter
from torch_sim.state import SimState, concatenate_states, initialize_state
from torch_sim.trajectory import TorchSimTrajectory
from torch_sim.transforms import minimum_image_displacement
from torch_sim.typing import StateLike


logger = logging.getLogger(__name__)

# Add epsilon for numerical stability
_EPS = torch.finfo(torch.float64).eps


def _extract_kwargs_from_params(
    params: dict[str, Any], func: Callable[..., Any], exclude: set[str] | None = None
) -> dict[str, Any]:
    """Extract kwargs from params dict that match function signature.

    Args:
        params: Dictionary of parameters to filter
        func: Function to extract parameters for
        exclude: Set of parameter names to exclude (e.g., 'state', 'model')

    Returns:
        Dictionary of parameters that match the function signature
    """
    exclude = exclude or {"state", "model"}
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters and k not in exclude}


@dataclass
class _OptimizerConfig:
    """Configuration for an optimizer type."""

    init_fn: Callable[..., Any]
    step_fn: Callable[..., Any]
    state_type: type
    init_kwargs_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    step_kwargs_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None = None


# Registry of optimizer configurations
_OPTIMIZER_REGISTRY: dict[str, _OptimizerConfig] = {
    "fire": _OptimizerConfig(
        init_fn=fire_init,
        step_fn=fire_step,
        state_type=FireState,
    ),
    "frechet_cell_fire": _OptimizerConfig(
        init_fn=fire_init,
        step_fn=fire_step,
        state_type=CellFireState,
        init_kwargs_modifier=lambda kwargs: {**kwargs, "cell_filter": CellFilter.frechet},
    ),
    "gd": _OptimizerConfig(
        init_fn=gradient_descent_init,
        step_fn=gradient_descent_step,
        state_type=OptimState,
        step_kwargs_modifier=lambda kwargs: (
            kwargs if "pos_lr" in kwargs else {**kwargs, "pos_lr": kwargs.get("lr", 0.01)}
        ),
    ),
    "ase_fire": _OptimizerConfig(
        init_fn=fire_init,
        step_fn=fire_step,
        state_type=FireState,
        init_kwargs_modifier=lambda kwargs: (
            kwargs if "fire_flavor" in kwargs else {**kwargs, "fire_flavor": "ase_fire"}
        ),
        step_kwargs_modifier=lambda kwargs: (
            kwargs if "fire_flavor" in kwargs else {**kwargs, "fire_flavor": "ase_fire"}
        ),
    ),
}


@dataclass
class NEB:
    """Nudged Elastic Band (NEB) optimizer.

    Finds the minimum energy path (MEP) between an initial and final state using
    the NEB algorithm.

    Attributes:
        model: The energy/force model (e.g., MACE) wrapped in a ModelInterface.
        n_images: Number of intermediate images between initial and final states.
        spring_constant: Spring constant connecting adjacent images (eV/Ang^2).
        use_climbing_image: Whether to use a climbing image.
        optimizer_type: Type of optimizer to use.
        optimizer_params: Parameters for the chosen optimizer.
        trajectory_filename: Optional filename for saving the NEB trajectory.
        device: Computation device (e.g., 'cpu', 'cuda'). If None, uses model device.
        dtype: Computation data type (e.g., torch.float32). If None, uses model dtype.
    """

    model: ModelInterface
    n_images: int
    spring_constant: float = 0.1  # eV/Ang^2, typical ASE default
    use_climbing_image: bool = False
    optimizer_type: Literal["fire", "gd", "frechet_cell_fire", "ase_fire"] = "fire"
    optimizer_params: dict[str, Any] = field(default_factory=dict)
    trajectory_filename: str | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Initializes device, dtype, and optimizer functions after dataclass creation."""
        if self.device is None:
            self.device = self.model.device
        if self.dtype is None:
            self.dtype = self.model.dtype

        # Initialize variable to store step 0 debug output
        self._step0_debug_output = None

        # Get optimizer configuration from registry
        if self.optimizer_type not in _OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unsupported optimizer_type: {self.optimizer_type}. "
                f"Supported types: {list(_OPTIMIZER_REGISTRY.keys())}"
            )

        config = _OPTIMIZER_REGISTRY[self.optimizer_type]
        self._init_fn = config.init_fn
        self._step_fn = config.step_fn
        self._OptimizerStateType = config.state_type

        # Automatically extract kwargs from optimizer_params based on function signatures
        # For init: exclude 'state' and 'model' (positional args)
        # For step: exclude 'state' and 'model' (positional args)
        init_kwargs = _extract_kwargs_from_params(
            self.optimizer_params, config.init_fn, exclude={"state", "model"}
        )
        step_kwargs = _extract_kwargs_from_params(
            self.optimizer_params, config.step_fn, exclude={"state", "model"}
        )

        # Apply modifiers if provided (for special cases like cell_filter, defaults, etc.)
        if config.init_kwargs_modifier:
            init_kwargs = config.init_kwargs_modifier(init_kwargs)
        if config.step_kwargs_modifier:
            step_kwargs = config.step_kwargs_modifier(step_kwargs)

        self._init_kwargs = init_kwargs
        self._step_kwargs = step_kwargs

    def _interpolate_path(
        self, initial_state: SimState, final_state: SimState
    ) -> SimState:
        """Linearly interpolate the initial path between states using MIC.

        Generates `n_images` intermediate states between the initial and final states
        by linear interpolation of atomic positions, respecting periodic boundary
        conditions via the Minimum Image Convention (MIC).

        Args:
            initial_state (SimState): The starting SimState (must be single-batch).
            final_state (SimState): The ending SimState (must be single-batch).

        Returns:
            SimState: A single SimState containing all interpolated intermediate
                images, batched together. The batch index corresponds to the image
                index (0 to n_images-1).

        Raises:
            ValueError: If initial and final states are incompatible (e.g., different
                number of atoms, atom types, PBC settings, or if they are not
                single-batch states).
        """
        # --- Input Validation ---
        if initial_state.n_systems != 1 or final_state.n_systems != 1:
            raise ValueError("Initial and final states must be single-system SimStates.")
        if initial_state.n_atoms != final_state.n_atoms:
            raise ValueError(
                f"Initial ({initial_state.n_atoms}) and final ({final_state.n_atoms}) "
                "states must have the same number of atoms."
            )
        if not torch.equal(initial_state.atomic_numbers, final_state.atomic_numbers):
            # Comparing floats might be tricky, but atomic numbers should be exact
            raise ValueError("Initial and final states must have the same atom types.")
        # Compare PBC values properly (can be bool, list, or tensor)
        pbc_match = False
        if isinstance(initial_state.pbc, torch.Tensor) and isinstance(
            final_state.pbc, torch.Tensor
        ):
            pbc_match = torch.equal(initial_state.pbc, final_state.pbc)
        elif isinstance(initial_state.pbc, torch.Tensor) or isinstance(
            final_state.pbc, torch.Tensor
        ):
            # One is tensor, one is not - convert both to tensors for comparison
            initial_pbc_tensor = (
                initial_state.pbc
                if isinstance(initial_state.pbc, torch.Tensor)
                else torch.tensor(initial_state.pbc, device=initial_state.device)
            )
            final_pbc_tensor = (
                final_state.pbc
                if isinstance(final_state.pbc, torch.Tensor)
                else torch.tensor(final_state.pbc, device=final_state.device)
            )
            pbc_match = torch.equal(initial_pbc_tensor, final_pbc_tensor)
        else:
            # Both are bools or lists
            pbc_match = initial_state.pbc == final_state.pbc
        if not pbc_match:
            # TODO: Could potentially support different PBCs, but complex for NEB.
            raise ValueError("Initial and final states must have the same PBC setting.")
        # For fixed-cell NEB, cells should ideally be identical. Warn if not?
        # if not torch.allclose(initial_state.cell, final_state.cell):

        n_atoms_per_image = initial_state.n_atoms

        # --- Interpolation ---
        initial_pos = initial_state.positions
        final_pos = final_state.positions

        # Calculate displacement using Minimum Image Convention
        displacement = minimum_image_displacement(
            dr=final_pos - initial_pos,
            cell=initial_state.cell[0],  # Use cell from initial state
            pbc=initial_state.pbc,
        )
        # Ensure shape is correct [n_atoms, 3]
        displacement = displacement.reshape(n_atoms_per_image, 3)

        # Generate interpolation factors (e.g., for n_images=3: 0.25, 0.5, 0.75)
        factors = torch.linspace(
            0.0, 1.0, steps=self.n_images + 2, device=self.device, dtype=self.dtype
        )[1:-1]  # Exclude 0.0 and 1.0 # Ensure dtype
        factors = factors.view(-1, 1, 1)  # Shape: [n_images, 1, 1]

        # Calculate interpolated positions: initial + factor * displacement
        # Broadcasting: [N_atoms, 3] + [N_images, 1, 1] * [N_atoms, 3] -> [N_images, N_atoms, 3]
        interpolated_pos = initial_pos.unsqueeze(0) + factors * displacement.unsqueeze(0)

        # Reshape to [n_images * n_atoms_per_image, 3]
        all_positions = interpolated_pos.reshape(-1, 3)

        # --- Create Batched State ---
        # Repeat other attributes for each image
        all_atomic_numbers = initial_state.atomic_numbers.repeat(self.n_images)
        all_masses = initial_state.masses.repeat(self.n_images)
        # Use initial state's cell, repeated for each image
        all_cells = initial_state.cell.repeat(
            self.n_images, 1, 1
        )  # Shape: [n_images, 3, 3]

        # Create system_idx tensor: [0, 0, ..., 1, 1, ..., n_images-1, ...]
        system_indices = torch.arange(
            self.n_images, device=self.device, dtype=torch.int64
        )
        all_system_idx = torch.repeat_interleave(
            system_indices, repeats=n_atoms_per_image
        )

        return SimState(
            positions=all_positions,
            atomic_numbers=all_atomic_numbers,
            masses=all_masses,
            cell=all_cells,
            pbc=initial_state.pbc,
            system_idx=all_system_idx,
        )

    def _compute_tangents(
        self,
        all_pos: torch.Tensor,  # Shape: [n_total_images, n_atoms, 3]
        all_energies: torch.Tensor,  # Shape: [n_total_images]
        cell: torch.Tensor,  # Shape: [3, 3]
        *,  # Make pbc keyword-only
        pbc: bool,
    ) -> torch.Tensor:
        """Compute normalized tangent vectors for intermediate NEB images.

        Implements the improved tangent estimate of Henkelman and Jónsson (2000)
        to determine the local tangent direction at each intermediate image based
        on the positions and energies of its neighbors.

        Args:
            all_pos (torch.Tensor): Atomic configurations for all images in the path
                (initial + intermediate + final), shape [n_total_images, n_atoms, 3].
            all_energies (torch.Tensor): Potential energy of each image, shape
                [n_total_images].
            cell (torch.Tensor): Unit cell vectors (shape [3, 3]), assumed constant
                for the path.
            pbc (bool): Flag indicating if periodic boundary conditions are active.

        Returns:
            torch.Tensor: Normalized local tangent vectors for the intermediate
                images only, shape [n_images, n_atoms, 3]. Tangents are zero for
                numerically identical adjacent images.
        """
        n_total_images, n_atoms_per_image, _ = all_pos.shape
        n_intermediate_images = n_total_images - 2
        device = all_pos.device
        dtype = all_pos.dtype

        # Initialize tangents for intermediate images only
        tangents = torch.zeros(
            (n_intermediate_images, n_atoms_per_image, 3),
            device=device,
            dtype=self.dtype,  # Use self.dtype
        )

        # Calculate displacements between adjacent images using MIC
        # dR_forward[i] = R_{i+1} - R_i
        displacements = minimum_image_displacement(
            dr=all_pos[1:] - all_pos[:-1], cell=cell, pbc=pbc
        )
        # Ensure shape is correct after MIC if needed
        displacements = displacements.reshape(n_total_images - 1, n_atoms_per_image, 3)

        # Energy differences V_{i+1} - V_i
        dE_forward = all_energies[1:] - all_energies[:-1]  # Shape: [n_total_images - 1]

        # Compute tangents for intermediate images (indices 1 to N in all_pos)
        for i in range(n_intermediate_images):
            img_idx = i + 1  # Index in all_pos, all_energies

            # Displacements adjacent to image `img_idx`
            # Note: displacements[k] is R_{k+1} - R_k
            dR_plus = displacements[img_idx]  # R_{i+1} - R_i (where i = img_idx)
            dR_minus = displacements[img_idx - 1]  # R_i - R_{i-1} (where i = img_idx)

            # Energy differences adjacent to image `img_idx`
            dE_plus = dE_forward[img_idx]  # V_{i+1} - V_i
            dE_minus = dE_forward[img_idx - 1]  # V_i - V_{i-1}

            # Select tangent based on energy profile (Henkelman & Jónsson criteria)
            tangent_i = torch.zeros_like(dR_plus)

            # Condition 1: Ascending segment (minimum) V_{i+1}>V_i and V_i>V_{i-1} => dE_plus>0 and dE_minus>0
            if dE_plus > 0 and dE_minus > 0:
                tangent_i = (
                    dR_plus  # ASE uses forward difference (dR_plus = R[i+1] - R[i])
                )

            # Condition 2: Descending segment (maximum) V_{i+1}<V_i and V_i<V_{i-1} => dE_plus<0 and dE_minus<0
            elif (
                dE_plus < 0 and dE_minus < 0
            ):  # Check if dE_minus comparison is correct (<0 vs >0)
                # tangent_i = dR_plus if abs(dE_plus) < abs(dE_minus) else dR_minus # Old complex version
                # ASE logic: if E[i+1] < E[i] < E[i-1], tangent = dR_minus (spring1.t) -> Mismatch?
                # Let's assume torch-sim should match ASE exactly:
                tangent_i = (
                    dR_minus  # ASE uses backward difference (dR_minus = R[i] - R[i-1])
                )

            # Condition 3: Other cases (weighted average in ASE)
            else:
                # Implement ASE's weighting logic precisely
                # Note: ASE uses absolute values for deltavmax/min calculation
                abs_dE_plus = torch.abs(dE_plus)
                abs_dE_minus = torch.abs(dE_minus)

                deltavmax = torch.maximum(abs_dE_plus, abs_dE_minus)
                deltavmin = torch.minimum(abs_dE_plus, abs_dE_minus)

                # Check E[i+1] vs E[i-1]
                # E[i+1] - E[i-1] = dE_plus + dE_minus
                if (dE_plus + dE_minus) > 0:  # E[i+1] > E[i-1]
                    tangent_i = dR_plus * deltavmax + dR_minus * deltavmin
                else:  # E[i+1] <= E[i-1]
                    tangent_i = dR_plus * deltavmin + dR_minus * deltavmax

            # Normalize the tangent vector *within* the loop
            norm_i = torch.linalg.norm(tangent_i)
            if norm_i > _EPS:
                tangents[i] = tangent_i / norm_i
            # else: tangent remains zero if norm is too small

        return tangents

    def _calculate_neb_forces(
        self,
        path_state: SimState,
        true_forces: torch.Tensor,
        true_energies: torch.Tensor,
        initial_energy: torch.Tensor,
        final_energy: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, dict | None]:  # Return forces and optional debug data
        """Calculate the NEB forces for intermediate images.

        The NEB force is composed of the true force perpendicular to the path tangent
        and the spring force parallel to the path tangent. Handles climbing image
        force modification if enabled.

        Args:
            path_state (SimState): SimState containing the full path (initial +
                intermediate + final images). Batches are assumed to be ordered.
            true_forces (torch.Tensor): Forces from the potential energy model for
                the *intermediate* images only, shape [n_movable_atoms, 3].
            true_energies (torch.Tensor): Potential energies for the *intermediate*
                images only, shape [n_images].
            initial_energy (torch.Tensor): Potential energy of the initial state
                (scalar tensor).
            final_energy (torch.Tensor): Potential energy of the final state
                (scalar tensor).
            step (int): Current optimization step number (used for climbing image delay).

        Returns:
            torch.Tensor: Calculated NEB forces for the intermediate images, ready to
                be passed to the optimizer, shape [n_movable_atoms, 3].
        """
        n_total_images = path_state.n_systems
        n_intermediate_images = n_total_images - 2
        assert n_intermediate_images == self.n_images
        n_atoms_per_image = path_state.n_atoms // n_total_images

        # --- Reshape inputs ---
        # Positions for all images: [n_total_images, n_atoms, 3]
        all_pos = path_state.positions.reshape(n_total_images, n_atoms_per_image, 3)
        # True forces for intermediate images: [n_images, n_atoms, 3]
        true_forces_reshaped = true_forces.reshape(
            n_intermediate_images, n_atoms_per_image, 3
        )
        # Cell vectors (assuming fixed cell for now, take from first batch)
        cell = path_state.cell[0]  # Shape [3, 3]
        # Convert pbc to bool if it's a tensor (for _compute_tangents)
        if isinstance(path_state.pbc, torch.Tensor):
            pbc_bool: bool = bool(
                path_state.pbc.any().item()
            )  # True if any dimension has PBC
        elif isinstance(path_state.pbc, bool):
            pbc_bool = path_state.pbc
        elif isinstance(path_state.pbc, list):
            pbc_bool = bool(any(path_state.pbc))
        else:
            pbc_bool = True
        pbc = path_state.pbc  # Keep original for minimum_image_displacement

        # --- Get Energies for Tangent Calculation ---
        all_energies = torch.cat(
            [
                initial_energy.unsqueeze(0),
                true_energies,
                final_energy.unsqueeze(0),
            ]
        )

        # --- Setup for Debugging Step 0 ---
        log_step_0 = step == 0
        debug_img_idx = (
            n_intermediate_images // 2
        )  # Index within intermediates (0 to n_images-1)
        debug_img_idx_all = debug_img_idx + 1  # Index within all_pos (0 to n_images+1)
        debug_data_ts = {}  # Initialize debug dict

        if log_step_0:
            debug_data_ts = {
                "step": 0,
                "image_index_intermediate": debug_img_idx,
                "image_index_absolute": debug_img_idx_all,
                "inputs": {},
                "outputs": {},
                "error": None,
            }
            debug_data_ts["inputs"]["energies_all"] = all_energies  # Monty handles tensor
            debug_data_ts["inputs"]["cell"] = cell
            debug_data_ts["inputs"]["pbc"] = pbc_bool  # Store Python bool
            debug_data_ts["inputs"]["positions_image_minus_1"] = all_pos[
                debug_img_idx_all - 1
            ]
            debug_data_ts["inputs"]["positions_image"] = all_pos[debug_img_idx_all]
            debug_data_ts["inputs"]["positions_image_plus_1"] = all_pos[
                debug_img_idx_all + 1
            ]
            debug_data_ts["inputs"]["true_forces_image"] = true_forces_reshaped[
                debug_img_idx
            ]

        # --- Calculate Tangents (tau) using the improved method ---
        # tangents shape: [n_images, n_atoms, 3]
        tangents = self._compute_tangents(all_pos, all_energies, cell, pbc=pbc_bool)
        logger.debug(
            f"  Step {step}: Tangent norms per image: {torch.linalg.norm(tangents, dim=(-1, -2))}"
        )
        if log_step_0:
            # Note: ASE tangent might not be normalized if norm is ~0, TS tangent should be.
            tangent_img = tangents[debug_img_idx]
            tangent_norm_img = torch.linalg.norm(tangent_img)
            debug_data_ts["outputs"]["tangent_vector"] = tangent_img
            debug_data_ts["outputs"]["tangent_norm"] = tangent_norm_img

        # --- Calculate Displacements for Spring Force ---
        # Recalculate here or reuse from _compute_tangents if efficient
        displacements = minimum_image_displacement(
            dr=all_pos[1:] - all_pos[:-1], cell=cell, pbc=pbc
        )
        displacements = displacements.reshape(n_total_images - 1, n_atoms_per_image, 3)
        if log_step_0:
            # Save displacements relevant to the middle image's tangent/spring calculation
            debug_data_ts["outputs"]["mic_displacement_1"] = displacements[
                debug_img_idx_all - 1
            ]  # R(i) - R(i-1)
            debug_data_ts["outputs"]["mic_displacement_2"] = displacements[
                debug_img_idx_all
            ]  # R(i+1) - R(i)

        # --- Calculate NEB Force Components ---

        # 1. Perpendicular component of true force
        # F_perp = F_true - (F_true . tau) * tau
        # Dot product (sum over atoms and dims): [n_images]
        F_true_dot_tau = (true_forces_reshaped * tangents).sum(dim=(-1, -2), keepdim=True)
        F_perp = true_forces_reshaped - F_true_dot_tau * tangents
        logger.debug(
            f"  Step {step}: F_perp norms per image: {torch.linalg.norm(F_perp, dim=(-1, -2))}"
        )
        if log_step_0:
            f_perp_img = F_perp[debug_img_idx]
            f_perp_norm_img = torch.linalg.norm(f_perp_img)
            debug_data_ts["outputs"]["f_true_dot_tau"] = F_true_dot_tau[
                debug_img_idx
            ].item()  # scalar
            debug_data_ts["outputs"]["f_perp_vector"] = f_perp_img
            debug_data_ts["outputs"]["f_perp_norm"] = f_perp_norm_img

        # 2. Parallel component of spring force
        # F_spring_par = k * (|R_{i+1}-R_i| - |R_i-R_{i-1}|) * tau_i
        # Segment lengths (scalar magnitude per segment): [n_images+1]
        segment_lengths = torch.linalg.norm(
            displacements, dim=(-1, -2)
        )  # Cleaner way [n_total_images-1]
        # Spring force magnitude (scalar per intermediate image): [n_images]
        F_spring_mag = self.spring_constant * (segment_lengths[1:] - segment_lengths[:-1])
        # Project onto tangent: [n_images, 1, 1] -> [n_images, n_atoms, 3]
        F_spring_par = F_spring_mag.view(-1, 1, 1) * tangents
        logger.debug(
            f"  Step {step}: F_spring_par norms per image: {torch.linalg.norm(F_spring_par, dim=(-1, -2))}"
        )
        if log_step_0:
            f_spring_par_img = F_spring_par[debug_img_idx]
            f_spring_par_norm_img = torch.linalg.norm(f_spring_par_img)
            debug_data_ts["outputs"]["segment_lengths"] = segment_lengths  # Full list
            debug_data_ts["outputs"]["spring_force_magnitude_term"] = F_spring_mag[
                debug_img_idx
            ].item()  # scalar
            debug_data_ts["outputs"]["f_spring_par_vector"] = f_spring_par_img
            debug_data_ts["outputs"]["f_spring_par_norm"] = f_spring_par_norm_img

        # --- Combine Components for NEB Force ---
        neb_forces = F_perp + F_spring_par
        if log_step_0:
            # --- Direct Debug Logs for Step 0 ---
            f_perp_img = F_perp[debug_img_idx]
            f_spring_par_img = F_spring_par[debug_img_idx]
            neb_force_img = neb_forces[debug_img_idx]
            logger.debug("  --- DIRECT DEBUG LOG (TORCH-SIM STEP 0) ---")
            logger.debug(f"    f_perp_norm: {torch.linalg.norm(f_perp_img)}")
            logger.debug(f"    f_perp_vec[0]: {f_perp_img[0]}")
            # segment_lengths shape: [n_total_images - 1]
            # segment_lengths[debug_img_idx] corresponds to spring2 length
            # segment_lengths[debug_img_idx-1] corresponds to spring1 length
            len1 = segment_lengths[debug_img_idx - 1]
            len2 = segment_lengths[debug_img_idx]
            len_diff = len2 - len1
            logger.debug(
                f"    spring1_length (R[{debug_img_idx_all}]-R[{debug_img_idx_all - 1}]): {len1}"
            )
            logger.debug(
                f"    spring2_length (R[{debug_img_idx_all + 1}]-R[{debug_img_idx_all}]): {len2}"
            )
            logger.debug(f"    Length Diff (len2 - len1): {len_diff}")
            logger.debug(f"    f_spring_par_norm: {torch.linalg.norm(f_spring_par_img)}")
            logger.debug(f"    f_spring_par_vec[0]: {f_spring_par_img[0]}")
            logger.debug(
                f"    neb_force_before_climb_norm: {torch.linalg.norm(neb_force_img)}"
            )
            # --------------------------------------
            # Store a *copy* detached from the graph to prevent modification by climbing image logic
            debug_data_ts["outputs"]["neb_force_before_climb_vector"] = (
                neb_forces[debug_img_idx].clone().detach()
            )
            debug_data_ts["outputs"]["neb_force_before_climb_norm"] = torch.linalg.norm(
                neb_forces[debug_img_idx]
            )  # Norm calculation is fine

            # --- Log the vector right before it would be saved ---
            logger.debug(
                f"  Value assigned to debug_data[neb_force_before_climb_vector][0]: {neb_forces[debug_img_idx][0]}"
            )
            # -----------------------------------------------------

        # --- Handle Climbing Image ---
        climbing_delay_steps = 10  # Example value
        if (
            self.use_climbing_image and n_intermediate_images > 0
        ):  # and step >= climbing_delay_steps: # Check step number - REMOVED DELAY
            # Find index of highest energy image among intermediates
            climbing_image_idx = torch.argmax(
                true_energies
            ).item()  # Index from 0 to n_images-1
            # Calculate the climbing force: F_climb = F_true - 2 * (F_true . tau) * tau
            F_climb = true_forces_reshaped[climbing_image_idx] - (
                2 * F_true_dot_tau[climbing_image_idx] * tangents[climbing_image_idx]
            )
            # Replace the NEB force for the climbing image with F_climb
            # This overwrites the spring force component for this image, as required.
            neb_forces[climbing_image_idx] = F_climb
            logger.debug(
                f"  Step {step}: Climbing image index: {climbing_image_idx}, "
                f"Climbing Force Norm: {torch.linalg.norm(F_climb)}"
            )
            if log_step_0:
                debug_data_ts["outputs"]["is_climbing_image"] = (
                    climbing_image_idx == debug_img_idx
                )
                debug_data_ts["outputs"]["imax"] = climbing_image_idx
                debug_data_ts["outputs"]["climbing_force_vector"] = neb_forces[
                    climbing_image_idx
                ]
                debug_data_ts["outputs"]["climbing_force_norm"] = torch.linalg.norm(
                    neb_forces[climbing_image_idx]
                )

        # --- Logging (Optional) ---
        # logger.debug(
        #    "  Max True Force Mag: "
        #    f"{torch.linalg.norm(true_forces_reshaped, dim=(-1,-2)).max().item():.4f}"
        # )
        # logger.debug(
        #     "  Max F_perp Mag: "
        #     f"{torch.linalg.norm(F_perp, dim=(-1,-2)).max().item():.4f}"
        # )
        # logger.debug(
        #     "  Max F_spring_par Mag: "
        #     f"{torch.linalg.norm(F_spring_par, dim=(-1,-2)).max().item():.4f}"
        # )
        # logger.debug(
        #     "  Max NEB Force Mag: "
        #     f"{torch.linalg.norm(neb_forces, dim=(-1,-2)).max().item():.4f}"
        # )
        logger.debug(
            f"  Step {step}: NEB force norms per image: {torch.linalg.norm(neb_forces, dim=(-1, -2))}"
        )
        logger.debug(f"  Step {step}: Intermediate energies: {true_energies}")
        if log_step_0 and not (
            self.use_climbing_image and climbing_image_idx == debug_img_idx
        ):  # Avoid logging twice if climbing image was logged
            # If not the climbing image, the final force is the one before modification
            pass  # Already stored neb_force_before_climb

        if log_step_0:
            debug_data_ts["outputs"]["final_neb_force_vector"] = neb_forces[debug_img_idx]
            debug_data_ts["outputs"]["final_neb_force_norm"] = torch.linalg.norm(
                neb_forces[debug_img_idx]
            )

        # --- Reshape output ---
        final_neb_forces = neb_forces.reshape(-1, 3)  # [n_movable_atoms, 3]

        # Return forces and the debug dictionary if step 0
        return final_neb_forces, debug_data_ts if log_step_0 else None

    def run(
        self,
        initial_system: StateLike,
        final_system: StateLike,
        max_steps: int = 100,
        fmax: float = 0.05,
        # TODO: add convergence criteria, batching options, output frequency etc.
    ) -> SimState:
        """Run the Nudged Elastic Band optimization.

        Optimizes the path between the initial and final systems to find the
        Minimum Energy Path (MEP).

        Args:
            initial_system (StateLike): The starting configuration (can be ASE Atoms,
                SimState, or other compatible format recognized by initialize_state).
            final_system (StateLike): The ending configuration.
            max_steps (int): Maximum number of optimization steps allowed.
            fmax (float): Convergence criterion based on the maximum NEB force component
                acting on any single atom across all intermediate images (in eV/Ang).

        Returns:
            SimState: The final optimized NEB path, including the initial,
                intermediate, and final images, concatenated into a single SimState.
            SimState: The final optimized NEB path, including the initial,
                intermediate, and final images, concatenated into a single SimState.
        """
        logger.info("Starting NEB optimization")

        # Reset step 0 debug output storage for this run
        self._step0_debug_output = None

        # 1. Initialize initial and final states
        initial_state = initialize_state(initial_system, self.device, self.dtype)
        final_state = initialize_state(final_system, self.device, self.dtype)
        # TODO: Add checks (e.g., same number of atoms, atom types)
        # Ensure endpoints are single-system SimStates
        # (They should already be from initialize_state, but verify)
        if initial_state.n_systems != 1:
            raise ValueError("Initial state must be a single-system SimState")
        if final_state.n_systems != 1:
            raise ValueError("Final state must be a single-system SimState")

        # 1b. Calculate endpoint energies/forces (needed for tangent calculation)
        # Note: Forces aren't strictly needed here but model usually returns both
        logger.info("Calculating endpoint energies...")
        # Concatenate expects a list of SimStates (or subclasses)
        endpoint_states = concatenate_states([initial_state, final_state])
        endpoint_output = self.model(endpoint_states)
        initial_energy = endpoint_output["energy"][0]
        final_energy = endpoint_output["energy"][1]
        # Distribute model extras (e.g. interaction_energy) back onto the
        # endpoint states so that subsequent concatenate_states calls with
        # opt_state (which carries those extras) produce consistent leading dims
        n_init_atoms = initial_state.n_atoms
        n_final_atoms = final_state.n_atoms
        init_extras: dict[str, torch.Tensor] = {}
        final_extras: dict[str, torch.Tensor] = {}
        for key, val in endpoint_output.items():
            if key in {"energy", "forces", "stress"} or not isinstance(val, torch.Tensor):
                continue
            if val.shape[0] == 2:
                init_extras[key] = val[:1]
                final_extras[key] = val[1:]
            elif val.shape[0] == n_init_atoms + n_final_atoms:
                init_extras[key] = val[:n_init_atoms]
                final_extras[key] = val[n_init_atoms:]
        initial_state.store_model_extras(init_extras)
        final_state.store_model_extras(final_extras)
        logger.info(
            f"Initial Energy: {initial_energy:.4f}, Final Energy: {final_energy:.4f}"
        )

        # 2. Create initial interpolated path (movable images only)
        interpolated_images = self._interpolate_path(initial_state, final_state)

        # 3. Initialize optimizer state for the movable images
        # Use the generic initializer with model parameter
        opt_state = self._init_fn(interpolated_images, self.model, **self._init_kwargs)

        # 4. Optimization loop
        logger.info(f"Running NEB for max {max_steps} steps or fmax < {fmax} eV/Ang.")

        # Context manager for trajectory writing
        traj_context = (
            TorchSimTrajectory(self.trajectory_filename, mode="w")
            if self.trajectory_filename
            else nullcontext()  # Use a dummy context if no filename
        )

        def _opt_state_as_simstate(state: SimState) -> SimState:
            """Project an OptimState/FireState down to a plain SimState.

            Concatenating an OptimState/FireState with plain SimState endpoints
            collapses to the first state's class (SimState), causing optimizer-
            specific fields like velocities/forces/energy to be misrouted into
            extras with mismatched leading dims. We strip those here and
            preserve only model-derived extras (interaction_energy, etc.) that
            were also populated on the endpoints.
            """
            optim_only_atom = {"forces"}
            optim_only_system = {"energy", "stress", "dt", "alpha", "n_pos"}
            sys_extras = {
                k: v for k, v in state.system_extras.items() if k not in optim_only_system
            }
            atom_extras = {
                k: v for k, v in state.atom_extras.items() if k not in optim_only_atom
            }
            return SimState(
                positions=state.positions,
                masses=state.masses,
                cell=state.cell,
                pbc=state.pbc,
                atomic_numbers=state.atomic_numbers,
                system_idx=state.system_idx,
                _system_extras=sys_extras,
                _atom_extras=atom_extras,
            )

        with traj_context as traj:
            for step in range(max_steps):
                # a. Get current true forces and energies
                true_forces = opt_state.forces
                true_energies = opt_state.energy

                # b. Calculate NEB forces
                # Concatenate states - ensures consistent group ID (0 for single NEB)
                full_path_state_calc = concatenate_states(
                    [initial_state, _opt_state_as_simstate(opt_state), final_state]
                )
                # Store true forces *before* calculating NEB forces
                true_forces_for_traj = opt_state.forces.clone()

                # Get forces and potentially the step 0 debug data
                neb_forces, step0_debug_data = self._calculate_neb_forces(
                    full_path_state_calc,
                    true_forces,  # Pass the forces from the start of the step
                    true_energies,
                    initial_energy,
                    final_energy,
                    step=step,
                )

                # c. Update the forces in the FIRE state object with NEB forces
                opt_state.forces = neb_forces
                neb_forces_for_traj = neb_forces.clone()

                # d. Perform optimization step
                # Use the generic step function with model parameter
                opt_state = self._step_fn(opt_state, self.model, **self._step_kwargs)

                # *** Store Step 0 Debug Data AFTER optimizer step ***
                if step == 0 and step0_debug_data:
                    logger.info("Storing Step 0 TorchSim debug data.")
                    self._step0_debug_output = step0_debug_data
                # ***************************************************

                # e. Write to trajectory (if enabled)
                if self.trajectory_filename is not None:  # Use explicit check
                    # Create the full path state for writing (including endpoints)
                    current_full_path = concatenate_states(
                        [
                            initial_state,
                            _opt_state_as_simstate(opt_state),
                            final_state,
                        ]
                    )
                    # Write arrays directly using traj.write_arrays
                    data_to_write = {
                        "positions": current_full_path.positions,
                        # Add forces - Need to handle endpoints (no NEB forces)
                        # Pad NEB forces with zeros for endpoints
                        "neb_forces": torch.cat(
                            [
                                torch.zeros_like(initial_state.positions),
                                neb_forces_for_traj,
                                torch.zeros_like(final_state.positions),
                            ],
                            dim=0,
                        ),
                        # True forces are only calculated for intermediate images
                        # Need forces for endpoints from the initial calculation
                        "true_forces": torch.cat(
                            [
                                endpoint_output["forces"][
                                    : initial_state.n_atoms
                                ],  # Initial forces
                                true_forces_for_traj,  # Intermediate forces
                                endpoint_output["forces"][
                                    initial_state.n_atoms :
                                ],  # Final forces
                            ],
                            dim=0,
                        ),
                        "energies": torch.cat(
                            [
                                initial_energy.unsqueeze(0),
                                opt_state.energy,  # Energies *after* the step
                                final_energy.unsqueeze(0),
                            ],
                            dim=0,
                        ),
                    }
                    if step == 0:  # Write static data only on the first step
                        # Assuming fixed cell NEB, cell is static
                        data_to_write["cell"] = current_full_path.cell
                        # These should also be static for the whole band
                        data_to_write["atomic_numbers"] = current_full_path.atomic_numbers
                        data_to_write["masses"] = current_full_path.masses
                        # Convert bool to tensor for saving
                        data_to_write["pbc"] = torch.tensor(current_full_path.pbc)
                        # Save the system_idx tensor to map atoms to images
                        data_to_write["image_indices"] = current_full_path.system_idx

                    traj.write_arrays(data_to_write, steps=step)

                # f. Check convergence
                max_force_magnitude = torch.sqrt((neb_forces**2).sum(dim=-1)).max()
                max_intermediate_energy = opt_state.energy.max()
                logger.info(
                    f"Step {step + 1:4d}:  Max Force = {max_force_magnitude:.4f}   Max Energy = {max_intermediate_energy:.4f}"
                    # f"Energy = {fire_state.energy.mean():.4f} eV (mean per image), " # Removed mean energy for brevity
                )
                if max_force_magnitude < fmax:
                    logger.info("NEB optimization converged.")
                    break
            else:  # Loop finished without break
                logger.warning("NEB optimization did not converge within max_steps.")

        # 5. Return the final path (including endpoints)
        # --- Write Step 0 Debug Dictionary AFTER loop finishes ---
        if self._step0_debug_output:
            output_filename_ts = "torchsim_step0_debug.pkl"  # Change extension
            logger.info(
                f"Attempting to write final Step 0 TorchSim debug data to {output_filename_ts}"
            )
            try:
                with open(output_filename_ts, "wb") as f:  # Use 'wb' for pickle
                    pickle.dump(self._step0_debug_output, f)
                    f.flush()
                    os.fsync(f.fileno())
                logger.info(
                    f"--- TorchSim NEB Debug Info (Step 0) saved to {output_filename_ts} ---"
                )
            except Exception as e:
                logger.error(
                    f"ERROR WRITING FINAL TORCHSIM STEP 0 DEBUG PICKLE: {e}",
                    exc_info=True,
                )
        else:
            logger.warning("No Step 0 TorchSim debug data was stored to write.")
        # ----------------------------------------------------------

        return concatenate_states(
            [initial_state, _opt_state_as_simstate(opt_state), final_state]
        )
