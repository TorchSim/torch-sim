"""Tests for the FixSymmetry constraint."""

from typing import Literal

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixSymmetry as ASEFixSymmetry
from ase.spacegroup.symmetrize import refine_symmetry as ase_refine_symmetry
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import torch_sim as ts
from torch_sim.constraints import FixSymmetry
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.symmetrize import get_symmetry_datasets


# Skip all tests if moyopy is not available
moyopy = pytest.importorskip("moyopy")
# spglib still needed for ASE comparison tests
spglib = pytest.importorskip("spglib")


# =============================================================================
# Structure Definitions (Single Source of Truth)
# =============================================================================

# Expected space groups for each structure type
SPACEGROUPS = {
    "fcc": 225,  # Fm-3m
    "hcp": 194,  # P6_3/mmc
    "diamond": 227,  # Fd-3m
    "bcc": 229,  # Im-3m
    "p6bar": 174,  # P-6 (low symmetry)
}

# Default maximum optimization steps for tests
MAX_STEPS = 30

# Default dtype for tests (torch.float64 recommended for numerical precision)
DTYPE = torch.float64

# Default symmetry precision for spglib
SYMPREC = 0.01


def _make_p6bar() -> Atoms:
    """Create low-symmetry P-6 (space group 174) structure using pymatgen."""
    lattice = Lattice.hexagonal(a=3.0, c=5.0)
    structure = Structure.from_spacegroup(
        sg=174, lattice=lattice, species=["Si"], coords=[[0.3, 0.1, 0.25]]
    )
    return AseAtomsAdaptor.get_atoms(structure)


def make_structure(name: str) -> Atoms:
    """Create a standard test structure by name.

    This is the single source of truth for test structures.
    Use this instead of inline bulk() calls to avoid duplication.

    Args:
        name: One of "fcc", "hcp", "diamond", "bcc", "p6bar" with optional
              "_supercell" and/or "_rotated" suffix

    Returns:
        ASE Atoms object
    """
    base_name = name.replace("_supercell", "").replace("_rotated", "")
    structures = {
        "fcc": lambda: bulk("Cu", "fcc", a=3.6),
        "hcp": lambda: bulk("Ti", "hcp", a=2.95, c=4.68),
        "diamond": lambda: bulk("Si", "diamond", a=5.43),
        "bcc": lambda: bulk("Al", "bcc", a=2 / np.sqrt(3), cubic=True),
        "p6bar": _make_p6bar,
    }
    atoms = structures[base_name]()
    if "_supercell" in name:
        atoms = atoms * (2, 2, 2)
    if "_rotated" in name:
        # Apply 3 rotation matrices (matching ASE's test setup)
        F = np.eye(3)
        for k in range(3):
            L = list(range(3))
            L.remove(k)
            (i, j) = L
            R = np.eye(3)
            theta = 0.1 * (k + 1)
            R[i, i] = np.cos(theta)
            R[j, j] = np.cos(theta)
            R[i, j] = np.sin(theta)
            R[j, i] = -np.sin(theta)
            F = np.dot(F, R)
        atoms.set_cell(atoms.cell @ F, scale_atoms=True)
    return atoms


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def model() -> LennardJonesModel:
    """Create a LennardJonesModel for testing."""
    return LennardJonesModel(
        sigma=1.0,
        epsilon=0.05,
        cutoff=6.0,
        use_neighbor_list=False,
        compute_stress=True,
        dtype=DTYPE,
    )


class NoisyModelWrapper:
    """Wrapper that adds noise to forces and stress from an underlying model."""

    def __init__(
        self,
        model: LennardJonesModel,
        rng_seed: int = 1,
        noise_scale: float = 1e-4,
    ) -> None:
        self.model = model
        self.rng = np.random.default_rng(rng_seed)
        self.noise_scale = noise_scale

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def __call__(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        results = self.model(state)
        # Add noise to forces
        if "forces" in results:
            noise = self.rng.normal(size=results["forces"].shape)
            results["forces"] = results["forces"] + self.noise_scale * torch.tensor(
                noise,
                dtype=results["forces"].dtype,
                device=results["forces"].device,
            )
        # Add noise to stress
        if "stress" in results:
            noise = self.rng.normal(size=results["stress"].shape)
            results["stress"] = results["stress"] + self.noise_scale * torch.tensor(
                noise,
                dtype=results["stress"].dtype,
                device=results["stress"].device,
            )
        return results


@pytest.fixture
def noisy_lj_model(model: LennardJonesModel) -> NoisyModelWrapper:
    """Create a LJ model that adds noise to forces/stress.

    Similar to ASE's NoisyLennardJones.
    """
    return NoisyModelWrapper(model)


# =============================================================================
# Shared Helper Functions
# =============================================================================


def get_spglib_dataset_from_atoms(
    atoms: Atoms, symprec: float = SYMPREC
) -> spglib.SpglibDataset:
    """Get full symmetry dataset for an ASE Atoms object using spglib directly."""
    return spglib.get_symmetry_dataset(
        (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers),
        symprec=symprec,
    )


def run_optimization_check_symmetry(
    state: ts.SimState,
    model: LennardJonesModel,
    constraint: FixSymmetry | None = None,
    *,
    adjust_cell: bool = True,
    symprec: float = SYMPREC,
    max_steps: int = MAX_STEPS,
    force_tol: float = 0.001,
) -> dict[str, list[int | None]]:
    """Run FIRE optimization and return initial/final space group numbers.

    Args:
        state: torch-sim SimState (can be batched)
        model: torch-sim model for optimization
        constraint: Optional FixSymmetry constraint to apply. If None, no constraint.
        adjust_cell: Whether to enable cell optimization (with Frechet filter)
        symprec: Symmetry precision for spglib checks
        max_steps: Maximum optimization steps
        force_tol: Force convergence tolerance

    Returns:
        Dict with 'initial_spacegroups' and 'final_spacegroups' lists.
    """
    initial_datasets = get_symmetry_datasets(state, symprec)

    if constraint is not None:
        state.constraints = [constraint]

    init_kwargs = {"cell_filter": ts.CellFilter.frechet} if adjust_cell else None
    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol, include_cell_forces=adjust_cell
    )
    final_state = ts.optimize(
        system=state,
        model=model,
        optimizer=ts.Optimizer.fire,
        convergence_fn=convergence_fn,
        init_kwargs=init_kwargs,
        max_steps=max_steps,
        steps_between_swaps=1,
    )

    final_datasets = get_symmetry_datasets(final_state, symprec)

    return {
        "initial_spacegroups": [d.number if d else None for d in initial_datasets],
        "final_spacegroups": [d.number if d else None for d in final_datasets],
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestFixSymmetryCreation:
    """Tests for FixSymmetry constraint creation."""

    def test_from_state_batched(self):
        """Test creating FixSymmetry from batched SimState with different structures."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            torch.device("cpu"),
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        assert len(constraint.rotations) == 2
        assert len(constraint.symm_maps) == 2
        assert constraint.system_idx.shape == (2,)
        # Both have cubic symmetry (48 ops) but different number of atoms
        assert constraint.rotations[0].shape[0] == 48
        assert constraint.rotations[1].shape[0] == 48
        # Cu FCC has 1 atom, Si diamond has 2
        assert constraint.symm_maps[0].shape == (48, 1)
        assert constraint.symm_maps[1].shape == (48, 2)

    def test_p1_identity_only(self):
        """Test P1 (no symmetry) has only identity and doesn't change forces/stress."""
        atoms = Atoms(
            "SiGe",
            positions=[[0.1, 0.2, 0.3], [1.1, 0.9, 1.3]],
            cell=[[3.0, 0.1, 0.2], [0.15, 3.5, 0.1], [0.2, 0.15, 4.0]],
            pbc=True,
        )
        state = ts.io.atoms_to_state(atoms, torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        assert constraint.rotations[0].shape[0] == 1, "P1 should have 1 operation"

        # Forces should be unchanged
        forces = torch.randn(2, 3, dtype=DTYPE)
        original_forces = forces.clone()
        constraint.adjust_forces(state, forces)
        assert torch.allclose(forces, original_forces, atol=1e-10)

        # Stress should be unchanged (identity symmetrization)
        stress = torch.randn(1, 3, 3, dtype=DTYPE)
        # Make it symmetric (stress tensors are symmetric)
        stress = (stress + stress.transpose(-1, -2)) / 2
        original_stress = stress.clone()
        constraint.adjust_stress(state, stress)
        assert torch.allclose(stress, original_stress, atol=1e-10)

    def test_symmetry_datasets_match_spglib(self):
        """Test get_symmetry_datasets space groups match spglib."""
        atoms_list = [make_structure(name) for name in ["fcc", "diamond", "hcp"]]

        # Test batched state
        batched_state = ts.io.atoms_to_state(atoms_list, torch.device("cpu"), DTYPE)
        moyo_datasets = get_symmetry_datasets(batched_state, SYMPREC)
        assert len(moyo_datasets) == 3

        # Compare space group numbers with spglib
        for idx, atoms in enumerate(atoms_list):
            spglib_dataset = get_spglib_dataset_from_atoms(atoms, SYMPREC)

            assert moyo_datasets[idx].number == spglib_dataset.number, (
                f"Space group mismatch for {atoms.get_chemical_formula()}: "
                f"moyopy={moyo_datasets[idx].number} vs "
                f"spglib={spglib_dataset.number}"
            )
            # Both should find the same number of symmetry operations
            assert len(moyo_datasets[idx].operations) == len(spglib_dataset.rotations)


class TestFixSymmetryComparisonWithASE:
    """Compare TorchSim FixSymmetry with ASE's implementation."""

    def test_symmetrize_forces_batched(self):
        """Test force symmetrization for batched systems with different structures."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            torch.device("cpu"),
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Create asymmetric forces (1 atom for Cu FCC, 2 atoms for Si diamond)
        forces = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]], dtype=DTYPE
        )

        constraint.adjust_forces(state, forces)

        # First atom (Cu FCC) should have zero force due to cubic symmetry
        assert torch.allclose(forces[0], torch.zeros(3, dtype=DTYPE), atol=1e-10)

    def test_force_symmetrization_matches_ase(self):
        """Compare force symmetrization with ASE using a multi-atom structure."""
        atoms = make_structure("p6bar")

        # Create TorchSim state and constraint
        state = ts.io.atoms_to_state(atoms, torch.device("cpu"), DTYPE)
        ts_constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Set up ASE constraint
        ase_atoms = atoms.copy()
        ase_refine_symmetry(ase_atoms, symprec=SYMPREC)
        ase_constraint = ASEFixSymmetry(ase_atoms, symprec=SYMPREC)

        # Create random test forces
        rng = np.random.default_rng(42)
        forces_np = rng.standard_normal((len(atoms), 3))
        forces_ts = torch.tensor(forces_np.copy(), dtype=DTYPE)

        # Symmetrize with both
        ts_constraint.adjust_forces(state, forces_ts)
        ase_constraint.adjust_forces(ase_atoms, forces_np)

        # Compare results
        assert np.allclose(forces_ts.numpy(), forces_np, atol=1e-10)

    def test_stress_symmetrization_matches_ase(self):
        """Compare stress symmetrization with ASE's implementation."""
        atoms = make_structure("p6bar")

        # Create TorchSim state and constraint
        state = ts.io.atoms_to_state(atoms, torch.device("cpu"), DTYPE)
        ts_constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Set up ASE constraint
        ase_atoms = atoms.copy()
        ase_refine_symmetry(ase_atoms, symprec=SYMPREC)
        ase_constraint = ASEFixSymmetry(ase_atoms, symprec=SYMPREC)

        # Create asymmetric but symmetric (as a matrix) stress tensor
        stress_3x3 = np.array([[10.0, 1.0, 0.5], [1.0, 8.0, 0.3], [0.5, 0.3, 6.0]])

        # ASE uses Voigt notation
        stress_voigt = full_3x3_to_voigt_6_stress(stress_3x3)
        stress_voigt_copy = stress_voigt.copy()

        # TorchSim uses 3x3 tensor with batch dimension
        stress_ts = torch.tensor([stress_3x3.copy()], dtype=DTYPE)

        # Symmetrize with both
        ts_constraint.adjust_stress(state, stress_ts)
        ase_constraint.adjust_stress(ase_atoms, stress_voigt_copy)

        # Convert ASE result back to 3x3
        ase_result_3x3 = voigt_6_to_full_3x3_stress(stress_voigt_copy)

        # Compare results
        assert np.allclose(stress_ts[0].numpy(), ase_result_3x3, atol=1e-10), (
            f"Stress mismatch:\nTorchSim:\n{stress_ts[0].numpy()}\nASE:\n{ase_result_3x3}"
        )

    def test_cell_deformation_symmetrization_matches_ase(self):
        """Compare cell deformation symmetrization with ASE."""
        atoms = make_structure("p6bar")

        # Create TorchSim state and constraint
        state = ts.io.atoms_to_state(atoms, torch.device("cpu"), DTYPE)
        ts_constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Set up ASE constraint
        ase_atoms = atoms.copy()
        ase_refine_symmetry(ase_atoms, symprec=SYMPREC)
        ase_constraint = ASEFixSymmetry(ase_atoms, symprec=SYMPREC)

        # Create a small asymmetric deformation of the cell
        original_cell = ase_atoms.get_cell().copy()
        deformed_cell = original_cell.copy()
        deformed_cell[0, 1] += 0.05  # Small off-diagonal perturbation

        # TorchSim - need column vector convention for adjust_cell
        new_cell_ts = torch.tensor(
            [deformed_cell.copy().T],
            dtype=DTYPE,  # Transpose for column vectors
        )
        ts_constraint.adjust_cell(state, new_cell_ts)
        ts_result = new_cell_ts[0].mT.numpy()  # Back to row vectors

        # ASE
        ase_cell = deformed_cell.copy()
        ase_constraint.adjust_cell(ase_atoms, ase_cell)

        # Compare results
        assert np.allclose(ts_result, ase_cell, atol=1e-10), (
            f"Cell mismatch:\nTorchSim:\n{ts_result}\nASE:\n{ase_cell}"
        )


class TestFixSymmetryMergeAndSelect:
    """Tests for FixSymmetry.merge, select_constraint, select_sub_constraint."""

    def test_merge_two_constraints(self):
        """Test merging two FixSymmetry constraints."""
        state1 = ts.io.atoms_to_state(make_structure("fcc"), torch.device("cpu"), DTYPE)
        state2 = ts.io.atoms_to_state(
            make_structure("diamond"), torch.device("cpu"), DTYPE
        )
        c1 = FixSymmetry.from_state(state1, symprec=SYMPREC)
        c2 = FixSymmetry.from_state(state2, symprec=SYMPREC)

        merged = FixSymmetry.merge([c1, c2], state_indices=[0, 1], atom_offsets=None)

        assert len(merged.rotations) == 2
        assert len(merged.symm_maps) == 2
        assert merged.system_idx.tolist() == [0, 1]

    def test_merge_multi_system_constraints_no_duplicate_indices(self):
        """Regression: merging multi-system constraints must not produce duplicates."""
        # Create two batched states so each constraint covers multiple systems
        atoms_a = [
            make_structure("fcc"),
            make_structure("diamond"),
            make_structure("hcp"),
        ]
        atoms_b = [make_structure("bcc"), make_structure("fcc")]
        state_a = ts.io.atoms_to_state(atoms_a, torch.device("cpu"), DTYPE)
        state_b = ts.io.atoms_to_state(atoms_b, torch.device("cpu"), DTYPE)
        c_a = FixSymmetry.from_state(state_a, symprec=SYMPREC)  # 3 systems
        c_b = FixSymmetry.from_state(state_b, symprec=SYMPREC)  # 2 systems

        # Old bug: state_indices=[0, 1] was used as offsets → [0,1,2, 1,2] (duplicates)
        # Fix: cumulative offset → [0,1,2, 3,4]
        merged = FixSymmetry.merge([c_a, c_b], state_indices=[0, 1], atom_offsets=None)

        assert len(merged.rotations) == 5
        assert len(merged.symm_maps) == 5
        assert merged.system_idx.tolist() == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize("mismatch_field", ["adjust_positions", "adjust_cell"])
    def test_merge_mismatched_settings_raises(
        self, mismatch_field: Literal["adjust_positions", "adjust_cell"]
    ):
        """Test that merging constraints with different settings raises ValueError."""
        state1 = ts.io.atoms_to_state(make_structure("fcc"), torch.device("cpu"), DTYPE)
        state2 = ts.io.atoms_to_state(
            make_structure("diamond"), torch.device("cpu"), DTYPE
        )

        kwargs1 = {mismatch_field: True}
        kwargs2 = {mismatch_field: False}
        c1 = FixSymmetry.from_state(state1, symprec=SYMPREC, **kwargs1)
        c2 = FixSymmetry.from_state(state2, symprec=SYMPREC, **kwargs2)

        with pytest.raises(ValueError, match=f"different {mismatch_field} settings"):
            FixSymmetry.merge([c1, c2], state_indices=[0, 1], atom_offsets=None)

    def test_select_constraint_single_system(self):
        """Test selecting a single system from batched constraint."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            torch.device("cpu"),
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Create masks to select only first system
        atom_mask = torch.tensor(
            [True, False, False], dtype=torch.bool
        )  # 1 Cu + 2 Si atoms
        system_mask = torch.tensor([True, False], dtype=torch.bool)

        selected = constraint.select_constraint(atom_mask, system_mask)

        assert selected is not None
        assert len(selected.rotations) == 1
        assert len(selected.symm_maps) == 1
        assert selected.system_idx.shape == (1,)
        # Should have Cu's 48 symmetry operations
        assert selected.rotations[0].shape[0] == 48

    def test_select_sub_constraint(self):
        """Test selecting a specific system by index."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            torch.device("cpu"),
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Select second system (Si diamond)
        # Note: atom_idx is ignored for FixSymmetry
        selected = constraint.select_sub_constraint(
            atom_idx=torch.tensor([1, 2]), sys_idx=1
        )

        assert selected is not None
        assert len(selected.rotations) == 1
        # Si diamond has 2 atoms
        assert selected.symm_maps[0].shape[1] == 2
        # New system_idx should be 0 (renumbered)
        assert selected.system_idx.item() == 0


class TestFixSymmetryWithOptimization:
    """Test FixSymmetry with actual optimization routines.

    Uses the shared run_optimization_check_symmetry helper for most tests.
    """

    @pytest.mark.parametrize("structure_name", ["fcc", "hcp", "diamond", "p6bar"])
    @pytest.mark.parametrize(
        ("adjust_positions", "adjust_cell"),
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_distorted_structure_preserves_symmetry(
        self,
        noisy_lj_model: NoisyModelWrapper,
        structure_name: str,
        *,
        adjust_positions: bool,
        adjust_cell: bool,
    ):
        """Test that a distorted structure relaxes while preserving symmetry.

        All combinations of adjust_positions and adjust_cell should preserve symmetry
        because forces are always symmetrized (matching ASE's behavior).

        """
        atoms = make_structure(structure_name)
        expected_spacegroup = SPACEGROUPS[structure_name]

        state = ts.io.atoms_to_state(atoms, torch.device("cpu"), DTYPE)

        # Create constraint BEFORE distorting - captures ideal symmetry
        constraint = FixSymmetry.from_state(
            state,
            symprec=SYMPREC,
            adjust_positions=adjust_positions,
            adjust_cell=adjust_cell,
        )

        # Now distort the cell (uniform scaling preserves symmetry but creates forces)
        # Scale by 0.9 to compress - this creates repulsive forces
        scale_factor = 0.9
        state.cell = state.cell * scale_factor
        state.positions = state.positions * scale_factor

        result = run_optimization_check_symmetry(
            state,
            noisy_lj_model,
            constraint=constraint,
            adjust_cell=adjust_cell,
            max_steps=MAX_STEPS,
            force_tol=0.01,  # Looser tolerance to ensure movement
        )

        assert result["final_spacegroups"][0] == expected_spacegroup, (
            f"Space group changed from {expected_spacegroup} to "
            f"{result['final_spacegroups'][0]} with adjust_positions={adjust_positions}, "
            f"adjust_cell={adjust_cell}"
        )

    @pytest.mark.parametrize("cell_filter", [ts.CellFilter.unit, ts.CellFilter.frechet])
    def test_cell_filter_preserves_symmetry(
        self, model: LennardJonesModel, cell_filter: ts.CellFilter
    ):
        """Test that cell filters with FixSymmetry preserve symmetry."""
        state = ts.io.atoms_to_state(make_structure("fcc"), torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        state.constraints = [constraint]

        initial_datasets = get_symmetry_datasets(state, symprec=SYMPREC)

        final_state = ts.optimize(
            system=state,
            model=model,
            optimizer=ts.Optimizer.gradient_descent,
            convergence_fn=ts.generate_force_convergence_fn(force_tol=0.01),
            init_kwargs={"cell_filter": cell_filter},
            max_steps=MAX_STEPS,
        )

        final_datasets = get_symmetry_datasets(final_state, symprec=SYMPREC)
        assert initial_datasets[0].number == final_datasets[0].number

    @pytest.mark.parametrize("cell_filter", [ts.CellFilter.frechet, ts.CellFilter.unit])
    def test_lbfgs_cell_optimization_preserves_symmetry(
        self,
        noisy_lj_model: NoisyModelWrapper,
        cell_filter: ts.CellFilter,
    ):
        """Regression: LBFGS must use set_constrained_cell for FixSymmetry support."""
        state = ts.io.atoms_to_state(make_structure("bcc"), torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        state.constraints = [constraint]

        # Compress cell to create forces
        state.cell = state.cell * 0.95
        state.positions = state.positions * 0.95

        initial_datasets = get_symmetry_datasets(state, symprec=SYMPREC)
        assert initial_datasets[0].number == SPACEGROUPS["bcc"]

        final_state = ts.optimize(
            system=state,
            model=noisy_lj_model,
            optimizer=ts.Optimizer.lbfgs,
            convergence_fn=ts.generate_force_convergence_fn(
                force_tol=0.01, include_cell_forces=True
            ),
            init_kwargs={"cell_filter": cell_filter},
            max_steps=MAX_STEPS,
        )

        final_datasets = get_symmetry_datasets(final_state, symprec=SYMPREC)
        assert final_datasets[0].number == SPACEGROUPS["bcc"], (
            f"LBFGS+{cell_filter} lost symmetry: {SPACEGROUPS['bcc']} -> "
            f"{final_datasets[0].number}"
        )

    @pytest.mark.parametrize("rotated", [False, True])
    def test_noisy_model_loses_symmetry_without_constraint(
        self, noisy_lj_model: NoisyModelWrapper, *, rotated: bool
    ):
        """Test that WITHOUT FixSymmetry, optimization with noisy forces loses symmetry.

        This is a negative control - verifies that noisy forces will break symmetry
        if no constraint is applied. Mirrors ASE's test_no_symmetrization.
        """
        name = "bcc_rotated" if rotated else "bcc"
        bcc_atoms = make_structure(name)
        state = ts.io.atoms_to_state(bcc_atoms, torch.device("cpu"), DTYPE)
        result = run_optimization_check_symmetry(
            state, noisy_lj_model, constraint=None, max_steps=MAX_STEPS, symprec=SYMPREC
        )

        # Initial should be BCC (space group 229)
        assert result["initial_spacegroups"][0] == 229
        # Final should have lost symmetry (different space group)
        assert result["final_spacegroups"][0] != 229, (
            f"Symmetry should be lost without constraint, but final space group "
            f"is still {result['final_spacegroups'][0]}"
        )

    @pytest.mark.parametrize("rotated", [False, True])
    def test_noisy_model_preserves_symmetry_with_constraint(
        self, noisy_lj_model: NoisyModelWrapper, *, rotated: bool
    ):
        """Test that WITH FixSymmetry, optimization with noisy forces preserves symmetry.

        Mirrors ASE's test_sym_adj_cell.
        """
        bcc_atoms = make_structure("bcc_rotated" if rotated else "bcc")
        state = ts.io.atoms_to_state(bcc_atoms, torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        result = run_optimization_check_symmetry(
            state,
            noisy_lj_model,
            constraint=constraint,
            max_steps=MAX_STEPS,
        )

        assert result["initial_spacegroups"][0] == 229
        assert result["final_spacegroups"][0] == 229, (
            f"Symmetry should be preserved with constraint, but final spacegroup "
            f"changed to {result['final_spacegroups'][0]}"
        )


class TestFixSymmetryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_removed_dof_returns_zero(self):
        """Test get_removed_dof returns zero (constrains direction, not DOF count)."""
        state = ts.io.atoms_to_state(make_structure("fcc"), torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        dof = constraint.get_removed_dof(state)
        assert torch.all(dof == 0)

    def test_large_deformation_gradient_raises(self):
        """Test that large deformation gradient raises RuntimeError."""
        state = ts.io.atoms_to_state(make_structure("fcc"), torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Create a very large deformation (> 0.25)
        # FCC cell has zeros on diagonal, so modify all elements by a large factor
        new_cell_col = state.cell.clone()  # Column vector convention
        new_cell_col[0] *= 1.5  # 50% stretch of entire cell

        with pytest.raises(RuntimeError, match="large deformation gradient"):
            constraint.adjust_cell(state, new_cell_col)

    def test_medium_deformation_gradient_warns(self):
        """Test that medium deformation gradient emits warning."""
        state = ts.io.atoms_to_state(make_structure("fcc"), torch.device("cpu"), DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)

        # Create a medium deformation (> 0.15 but < 0.25)
        new_cell_col = state.cell.clone()  # Column vector convention
        new_cell_col[0] *= 1.2  # 20% stretch of entire cell

        with pytest.warns(UserWarning, match="may be ill-behaved"):
            constraint.adjust_cell(state, new_cell_col)

    @pytest.mark.parametrize("refine_symmetry_state", [True, False])
    def test_from_state_refine_symmetry(self, *, refine_symmetry_state: bool):
        """Test from_state with different refine_symmetry_state settings."""
        atoms = make_structure("fcc")
        # Add small perturbation
        perturbed = atoms.copy()
        rng = np.random.default_rng(42)
        perturbed.positions += rng.standard_normal(perturbed.positions.shape) * 0.001

        state = ts.io.atoms_to_state(perturbed, torch.device("cpu"), DTYPE)
        original_positions = state.positions.clone()
        original_cell = state.cell.clone()

        _ = FixSymmetry.from_state(
            state, symprec=SYMPREC, refine_symmetry_state=refine_symmetry_state
        )

        if not refine_symmetry_state:
            # State should not be modified
            assert torch.allclose(state.positions, original_positions)
            assert torch.allclose(state.cell, original_cell)
        else:
            # State may be modified (positions refined to ideal)
            # We just check the function runs without error
            assert state.positions.shape == original_positions.shape
