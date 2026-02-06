"""Tests for the FixSymmetry constraint."""

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
from torch_sim.constraints import FixCom, FixSymmetry
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.symmetrize import get_symmetry_datasets


pytest.importorskip("moyopy")
pytest.importorskip("spglib")  # needed by ASE's FixSymmetry

SPACEGROUPS = {"fcc": 225, "hcp": 194, "diamond": 227, "bcc": 229, "p6bar": 174}
MAX_STEPS = 30
DTYPE = torch.float64
SYMPREC = 0.01
CPU = torch.device("cpu")


# === Structure helpers ===


def _make_p6bar() -> Atoms:
    """Create P-6 (space group 174) structure."""
    lattice = Lattice.hexagonal(a=3.0, c=5.0)
    structure = Structure.from_spacegroup(
        sg=174, lattice=lattice, species=["Si"], coords=[[0.3, 0.1, 0.25]]
    )
    return AseAtomsAdaptor.get_atoms(structure)


def make_structure(name: str) -> Atoms:
    """Create a test structure by name (fcc/hcp/diamond/bcc/p6bar + _rotated suffix)."""
    base = name.replace("_rotated", "")
    builders = {
        "fcc": lambda: bulk("Cu", "fcc", a=3.6),
        "hcp": lambda: bulk("Ti", "hcp", a=2.95, c=4.68),
        "diamond": lambda: bulk("Si", "diamond", a=5.43),
        "bcc": lambda: bulk("Al", "bcc", a=2 / np.sqrt(3), cubic=True),
        "p6bar": _make_p6bar,
    }
    atoms = builders[base]()
    if "_rotated" in name:
        rotation_product = np.eye(3)
        for axis_idx in range(3):
            axes = list(range(3))
            axes.remove(axis_idx)
            row_idx, col_idx = axes
            rot_mat = np.eye(3)
            theta = 0.1 * (axis_idx + 1)
            rot_mat[row_idx, row_idx] = np.cos(theta)
            rot_mat[col_idx, col_idx] = np.cos(theta)
            rot_mat[row_idx, col_idx] = np.sin(theta)
            rot_mat[col_idx, row_idx] = -np.sin(theta)
            rotation_product = np.dot(rotation_product, rot_mat)
        atoms.set_cell(atoms.cell @ rotation_product, scale_atoms=True)
    return atoms


# === Fixtures ===


@pytest.fixture
def model() -> LennardJonesModel:
    """LJ model for testing."""
    return LennardJonesModel(
        sigma=1.0,
        epsilon=0.05,
        cutoff=6.0,
        use_neighbor_list=False,
        compute_stress=True,
        dtype=DTYPE,
    )


class NoisyModelWrapper:
    """Wrapper that adds noise to forces and stress."""

    def __init__(self, model: LennardJonesModel, noise_scale: float = 1e-4) -> None:
        self.model = model
        self.rng = np.random.default_rng(seed=1)
        self.noise_scale = noise_scale

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def __call__(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        """Forward pass with added noise."""
        results = self.model(state)
        for key in ("forces", "stress"):
            if key in results:
                noise = torch.tensor(
                    self.rng.normal(size=results[key].shape),
                    dtype=results[key].dtype,
                    device=results[key].device,
                )
                results[key] = results[key] + self.noise_scale * noise
        return results


@pytest.fixture
def noisy_lj_model(model: LennardJonesModel) -> NoisyModelWrapper:
    """LJ model with noise added to forces/stress."""
    return NoisyModelWrapper(model)


@pytest.fixture
def p6bar_both_constraints() -> tuple[ts.SimState, FixSymmetry, Atoms, ASEFixSymmetry]:
    """P-6 structure with both TorchSim and ASE constraints (shared setup)."""
    atoms = make_structure("p6bar")
    state = ts.io.atoms_to_state(atoms, CPU, DTYPE)
    ts_constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
    ase_atoms = atoms.copy()
    ase_refine_symmetry(ase_atoms, symprec=SYMPREC)
    ase_constraint = ASEFixSymmetry(ase_atoms, symprec=SYMPREC)
    return state, ts_constraint, ase_atoms, ase_constraint


# === Optimization helper ===


def run_optimization_check_symmetry(
    state: ts.SimState,
    model: LennardJonesModel | NoisyModelWrapper,
    constraint: FixSymmetry | None = None,
    *,
    adjust_cell: bool = True,
    max_steps: int = MAX_STEPS,
    force_tol: float = 0.001,
) -> dict[str, list[int | None]]:
    """Run FIRE optimization and return initial/final space group numbers."""
    initial = get_symmetry_datasets(state, SYMPREC)
    if constraint is not None:
        state.constraints = [constraint]
    init_kwargs = {"cell_filter": ts.CellFilter.frechet} if adjust_cell else None
    convergence_fn = ts.generate_force_convergence_fn(
        force_tol=force_tol,
        include_cell_forces=adjust_cell,
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
    final = get_symmetry_datasets(final_state, SYMPREC)
    return {
        "initial_spacegroups": [d.number if d else None for d in initial],
        "final_spacegroups": [d.number if d else None for d in final],
    }


# === Tests: Creation & Basics ===


class TestFixSymmetryCreation:
    """Tests for FixSymmetry creation and basic behavior."""

    def test_from_state_batched(self) -> None:
        """Batched state with FCC + diamond gets correct ops, atom counts, and DOF."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            CPU,
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        assert len(constraint.rotations) == 2
        assert constraint.rotations[0].shape[0] == 48  # cubic
        assert constraint.symm_maps[0].shape == (48, 1)  # Cu: 1 atom
        assert constraint.symm_maps[1].shape == (48, 2)  # Si: 2 atoms
        assert torch.all(constraint.get_removed_dof(state) == 0)

    def test_p1_identity_is_noop(self) -> None:
        """P1 structure has 1 op and symmetrization is a no-op for forces and stress."""
        atoms = Atoms(
            "SiGe",
            positions=[[0.1, 0.2, 0.3], [1.1, 0.9, 1.3]],
            cell=[[3.0, 0.1, 0.2], [0.15, 3.5, 0.1], [0.2, 0.15, 4.0]],
            pbc=True,
        )
        state = ts.io.atoms_to_state(atoms, CPU, DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        assert constraint.rotations[0].shape[0] == 1

        forces = torch.randn(2, 3, dtype=DTYPE)
        orig_forces = forces.clone()
        constraint.adjust_forces(state, forces)
        assert torch.allclose(forces, orig_forces, atol=1e-10)

        stress = torch.randn(1, 3, 3, dtype=DTYPE)
        stress = (stress + stress.mT) / 2
        orig_stress = stress.clone()
        constraint.adjust_stress(state, stress)
        assert torch.allclose(stress, orig_stress, atol=1e-10)

    @pytest.mark.parametrize("refine", [True, False])
    def test_from_state_refine_symmetry(self, *, refine: bool) -> None:
        """With refine=False state is unmodified; with refine=True it may change."""
        atoms = make_structure("fcc")
        rng = np.random.default_rng(42)
        atoms.positions += rng.standard_normal(atoms.positions.shape) * 0.001
        state = ts.io.atoms_to_state(atoms, CPU, DTYPE)
        orig_pos = state.positions.clone()
        _ = FixSymmetry.from_state(state, symprec=SYMPREC, refine_symmetry_state=refine)
        if not refine:
            assert torch.allclose(state.positions, orig_pos)

    @pytest.mark.parametrize("structure_name", ["fcc", "hcp", "diamond", "p6bar"])
    def test_refine_symmetry_produces_correct_spacegroup(
        self,
        structure_name: str,
    ) -> None:
        """Perturbed structure recovers correct spacegroup after refinement."""
        from torch_sim.symmetrize import refine_symmetry

        atoms = make_structure(structure_name)
        expected = SPACEGROUPS[structure_name]
        rng = np.random.default_rng(42)
        atoms.positions += rng.standard_normal(atoms.positions.shape) * 0.001
        state = ts.io.atoms_to_state(atoms, CPU, DTYPE)

        refined_cell, refined_pos = refine_symmetry(
            state.row_vector_cell[0],
            state.positions,
            state.atomic_numbers,
            symprec=SYMPREC,
        )
        state.cell[0] = refined_cell.mT
        state.positions = refined_pos

        datasets = get_symmetry_datasets(state, symprec=1e-4)
        assert datasets[0].number == expected

    def test_cubic_forces_vanish(self) -> None:
        """Asymmetric force on single cubic atom symmetrizes to zero."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            CPU,
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        forces = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
            dtype=DTYPE,
        )
        constraint.adjust_forces(state, forces)
        assert torch.allclose(forces[0], torch.zeros(3, dtype=DTYPE), atol=1e-10)

    def test_large_deformation_raises(self) -> None:
        """Deformation gradient > 0.25 raises RuntimeError."""
        state = ts.io.atoms_to_state(make_structure("fcc"), CPU, DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        new_cell = state.cell.clone()
        new_cell[0] *= 1.5
        with pytest.raises(RuntimeError, match="deformation gradient"):
            constraint.adjust_cell(state, new_cell)

    def test_init_mismatched_lengths_raises(self) -> None:
        """Mismatched rotations/symm_maps lengths raises ValueError."""
        rots = [torch.eye(3).unsqueeze(0)]
        smaps = [torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long)]
        with pytest.raises(ValueError, match="length mismatch"):
            FixSymmetry(rots, smaps)

    @pytest.mark.parametrize("method", ["adjust_positions", "adjust_cell"])
    def test_adjust_skipped_when_disabled(self, method: str) -> None:
        """adjust_positions=False / adjust_cell=False leaves data unchanged."""
        flag = method.replace("adjust_", "")  # "positions" or "cell"
        state = ts.io.atoms_to_state(make_structure("fcc"), CPU, DTYPE)
        constraint = FixSymmetry.from_state(
            state,
            symprec=SYMPREC,
            **{f"adjust_{flag}": False},
        )
        if method == "adjust_positions":
            data = state.positions.clone() + 0.1
        else:
            data = state.cell.clone() * 1.01
        expected = data.clone()
        getattr(constraint, method)(state, data)
        assert torch.equal(data, expected)


# === Tests: Comparison with ASE ===


class TestFixSymmetryComparisonWithASE:
    """Compare TorchSim FixSymmetry with ASE's implementation on P-6 structure."""

    def test_force_symmetrization_matches_ase(
        self,
        p6bar_both_constraints: tuple,
    ) -> None:
        """Force symmetrization matches ASE."""
        state, ts_c, ase_atoms, ase_c = p6bar_both_constraints
        rng = np.random.default_rng(42)
        forces_np = rng.standard_normal((len(ase_atoms), 3))
        forces_ts = torch.tensor(forces_np.copy(), dtype=DTYPE)
        ts_c.adjust_forces(state, forces_ts)
        ase_c.adjust_forces(ase_atoms, forces_np)
        assert np.allclose(forces_ts.numpy(), forces_np, atol=1e-10)

    def test_stress_symmetrization_matches_ase(
        self,
        p6bar_both_constraints: tuple,
    ) -> None:
        """Stress symmetrization matches ASE."""
        state, ts_c, ase_atoms, ase_c = p6bar_both_constraints
        stress_3x3 = np.array([[10.0, 1.0, 0.5], [1.0, 8.0, 0.3], [0.5, 0.3, 6.0]])
        stress_voigt = full_3x3_to_voigt_6_stress(stress_3x3).copy()
        stress_ts = torch.tensor([stress_3x3.copy()], dtype=DTYPE)
        ts_c.adjust_stress(state, stress_ts)
        ase_c.adjust_stress(ase_atoms, stress_voigt)
        assert np.allclose(
            stress_ts[0].numpy(),
            voigt_6_to_full_3x3_stress(stress_voigt),
            atol=1e-10,
        )

    def test_cell_deformation_matches_ase(
        self,
        p6bar_both_constraints: tuple,
    ) -> None:
        """Cell deformation symmetrization matches ASE."""
        state, ts_c, ase_atoms, ase_c = p6bar_both_constraints
        deformed = ase_atoms.get_cell().copy()
        deformed[0, 1] += 0.05
        new_cell_ts = torch.tensor([deformed.copy().T], dtype=DTYPE)
        ts_c.adjust_cell(state, new_cell_ts)
        ase_cell = deformed.copy()
        ase_c.adjust_cell(ase_atoms, ase_cell)
        assert np.allclose(new_cell_ts[0].mT.numpy(), ase_cell, atol=1e-10)

    def test_position_symmetrization_matches_ase(
        self,
        p6bar_both_constraints: tuple,
    ) -> None:
        """Position displacement symmetrization matches ASE."""
        state, ts_c, ase_atoms, ase_c = p6bar_both_constraints
        rng = np.random.default_rng(42)
        disp = rng.standard_normal((len(ase_atoms), 3)) * 0.01
        new_pos_ts = state.positions.clone() + torch.tensor(disp, dtype=DTYPE)
        new_pos_ase = ase_atoms.positions.copy() + disp
        ts_c.adjust_positions(state, new_pos_ts)
        ase_c.adjust_positions(ase_atoms, new_pos_ase)
        assert np.allclose(new_pos_ts.numpy(), new_pos_ase, atol=1e-10)


# === Tests: Merge, Select, Reindex ===


class TestFixSymmetryMergeSelectReindex:
    """Tests for reindex/merge API, select, and concatenation."""

    def test_reindex_preserves_symmetry_data(self) -> None:
        """reindex shifts system_idx but preserves rotations and symm_maps."""
        state = ts.io.atoms_to_state(make_structure("fcc"), CPU, DTYPE)
        orig = FixSymmetry.from_state(state, symprec=SYMPREC)
        shifted = orig.reindex(atom_offset=100, system_offset=5)
        assert shifted.system_idx.item() == 5
        assert torch.equal(shifted.rotations[0], orig.rotations[0])
        assert torch.equal(shifted.symm_maps[0], orig.symm_maps[0])

    def test_merge_two_constraints(self) -> None:
        """Merge two single-system constraints via reindex + merge."""
        s1 = ts.io.atoms_to_state(make_structure("fcc"), CPU, DTYPE)
        s2 = ts.io.atoms_to_state(make_structure("diamond"), CPU, DTYPE)
        c1 = FixSymmetry.from_state(s1)
        c2 = FixSymmetry.from_state(s2).reindex(atom_offset=0, system_offset=1)
        merged = FixSymmetry.merge([c1, c2])
        assert len(merged.rotations) == 2
        assert merged.system_idx.tolist() == [0, 1]

    def test_merge_multi_system_no_duplicate_indices(self) -> None:
        """Regression: multi-system constraints must use cumulative offsets."""
        atoms_a = [
            make_structure("fcc"),
            make_structure("diamond"),
            make_structure("hcp"),
        ]
        atoms_b = [make_structure("bcc"), make_structure("fcc")]
        c_a = FixSymmetry.from_state(ts.io.atoms_to_state(atoms_a, CPU, DTYPE))
        c_b = FixSymmetry.from_state(
            ts.io.atoms_to_state(atoms_b, CPU, DTYPE),
        ).reindex(atom_offset=0, system_offset=3)
        merged = FixSymmetry.merge([c_a, c_b])
        assert merged.system_idx.tolist() == [0, 1, 2, 3, 4]

    def test_system_constraint_merge_multi_system_via_concatenate(self) -> None:
        """Regression: merging multi-system FixCom via concatenate_states."""
        s1 = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            CPU,
            DTYPE,
        )
        s2 = ts.io.atoms_to_state(
            [make_structure("bcc"), make_structure("hcp")],
            CPU,
            DTYPE,
        )
        s1.constraints = [FixCom(system_idx=torch.tensor([0, 1]))]
        s2.constraints = [FixCom(system_idx=torch.tensor([0, 1]))]
        combined = ts.concatenate_states([s1, s2])
        assert combined.constraints[0].system_idx.tolist() == [0, 1, 2, 3]

    def test_concatenate_states_with_fix_symmetry(self) -> None:
        """FixSymmetry survives concatenate_states and still symmetrizes correctly."""
        s1 = ts.io.atoms_to_state(make_structure("fcc"), CPU, DTYPE)
        s2 = ts.io.atoms_to_state(make_structure("diamond"), CPU, DTYPE)
        s1.constraints = [FixSymmetry.from_state(s1, symprec=SYMPREC)]
        s2.constraints = [FixSymmetry.from_state(s2, symprec=SYMPREC)]
        combined = ts.concatenate_states([s1, s2])
        constraint = combined.constraints[0]
        assert isinstance(constraint, FixSymmetry)
        assert constraint.system_idx.tolist() == [0, 1]
        assert len(constraint.rotations) == 2
        # Forces on single FCC atom should still vanish
        forces = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
            dtype=DTYPE,
        )
        constraint.adjust_forces(combined, forces)
        assert torch.allclose(forces[0], torch.zeros(3, dtype=DTYPE), atol=1e-10)

    def test_select_sub_constraint(self) -> None:
        """Select second system from batched constraint."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            CPU,
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        selected = constraint.select_sub_constraint(torch.tensor([1, 2]), sys_idx=1)
        assert selected is not None
        assert selected.symm_maps[0].shape[1] == 2
        assert selected.system_idx.item() == 0

    def test_select_constraint_by_mask(self) -> None:
        """Select first system via system_mask."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            CPU,
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        selected = constraint.select_constraint(
            atom_mask=torch.tensor([True, False, False]),
            system_mask=torch.tensor([True, False]),
        )
        assert selected is not None
        assert len(selected.rotations) == 1
        assert selected.rotations[0].shape[0] == 48

    def test_select_returns_none_for_nonexistent(self) -> None:
        """select_sub_constraint and select_constraint return None when no match."""
        state = ts.io.atoms_to_state(
            [make_structure("fcc"), make_structure("diamond")],
            CPU,
            DTYPE,
        )
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        assert constraint.select_sub_constraint(torch.tensor([0]), sys_idx=99) is None
        assert (
            constraint.select_constraint(
                atom_mask=torch.zeros(3, dtype=torch.bool),
                system_mask=torch.zeros(2, dtype=torch.bool),
            )
            is None
        )


# === Tests: build_symmetry_map chunked path ===


def test_build_symmetry_map_chunked_matches_vectorized() -> None:
    """Per-op loop gives same result as vectorized path."""
    import torch_sim.symmetrize as sym_mod
    from torch_sim.symmetrize import (
        _extract_symmetry_ops,
        _moyo_dataset,
        build_symmetry_map,
    )

    state = ts.io.atoms_to_state(make_structure("p6bar"), CPU, DTYPE)
    cell = state.row_vector_cell[0]
    frac = state.positions @ torch.linalg.inv(cell)
    dataset = _moyo_dataset(cell, frac, state.atomic_numbers)
    rotations, translations = _extract_symmetry_ops(dataset, DTYPE, CPU)

    old_threshold = sym_mod._SYMM_MAP_CHUNK_THRESHOLD  # noqa: SLF001
    try:
        sym_mod._SYMM_MAP_CHUNK_THRESHOLD = len(state.positions) + 1  # noqa: SLF001
        vectorized = build_symmetry_map(rotations, translations, frac)
        sym_mod._SYMM_MAP_CHUNK_THRESHOLD = 0  # noqa: SLF001
        chunked = build_symmetry_map(rotations, translations, frac)
    finally:
        sym_mod._SYMM_MAP_CHUNK_THRESHOLD = old_threshold  # noqa: SLF001
    assert torch.equal(vectorized, chunked)


# === Tests: Optimization ===


class TestFixSymmetryWithOptimization:
    """Test FixSymmetry with actual optimization routines."""

    @pytest.mark.parametrize("structure_name", ["fcc", "hcp", "diamond"])
    @pytest.mark.parametrize(
        ("adjust_positions", "adjust_cell"),
        [(True, True), (False, False)],
    )
    def test_distorted_preserves_symmetry(
        self,
        noisy_lj_model: NoisyModelWrapper,
        structure_name: str,
        *,
        adjust_positions: bool,
        adjust_cell: bool,
    ) -> None:
        """Compressed structure relaxes while preserving symmetry."""
        atoms = make_structure(structure_name)
        expected = SPACEGROUPS[structure_name]
        state = ts.io.atoms_to_state(atoms, CPU, DTYPE)
        constraint = FixSymmetry.from_state(
            state,
            symprec=SYMPREC,
            adjust_positions=adjust_positions,
            adjust_cell=adjust_cell,
        )
        state.cell = state.cell * 0.9
        state.positions = state.positions * 0.9
        result = run_optimization_check_symmetry(
            state,
            noisy_lj_model,
            constraint=constraint,
            adjust_cell=adjust_cell,
            force_tol=0.01,
        )
        assert result["final_spacegroups"][0] == expected

    @pytest.mark.parametrize("cell_filter", [ts.CellFilter.unit, ts.CellFilter.frechet])
    def test_cell_filter_preserves_symmetry(
        self,
        model: LennardJonesModel,
        cell_filter: ts.CellFilter,
    ) -> None:
        """Cell filters with FixSymmetry preserve symmetry."""
        state = ts.io.atoms_to_state(make_structure("fcc"), CPU, DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        state.constraints = [constraint]
        initial = get_symmetry_datasets(state, symprec=SYMPREC)
        final_state = ts.optimize(
            system=state,
            model=model,
            optimizer=ts.Optimizer.gradient_descent,
            convergence_fn=ts.generate_force_convergence_fn(force_tol=0.01),
            init_kwargs={"cell_filter": cell_filter},
            max_steps=MAX_STEPS,
        )
        final = get_symmetry_datasets(final_state, symprec=SYMPREC)
        assert initial[0].number == final[0].number

    @pytest.mark.parametrize("cell_filter", [ts.CellFilter.frechet, ts.CellFilter.unit])
    def test_lbfgs_preserves_symmetry(
        self,
        noisy_lj_model: NoisyModelWrapper,
        cell_filter: ts.CellFilter,
    ) -> None:
        """Regression: LBFGS must use set_constrained_cell for FixSymmetry support."""
        state = ts.io.atoms_to_state(make_structure("bcc"), CPU, DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        state.constraints = [constraint]
        state.cell = state.cell * 0.95
        state.positions = state.positions * 0.95
        final_state = ts.optimize(
            system=state,
            model=noisy_lj_model,
            optimizer=ts.Optimizer.lbfgs,
            convergence_fn=ts.generate_force_convergence_fn(
                force_tol=0.01,
                include_cell_forces=True,
            ),
            init_kwargs={"cell_filter": cell_filter},
            max_steps=MAX_STEPS,
        )
        final = get_symmetry_datasets(final_state, symprec=SYMPREC)
        assert final[0].number == SPACEGROUPS["bcc"]

    @pytest.mark.parametrize("rotated", [False, True])
    def test_noisy_model_loses_symmetry_without_constraint(
        self,
        noisy_lj_model: NoisyModelWrapper,
        *,
        rotated: bool,
    ) -> None:
        """Negative control: without FixSymmetry, noisy forces break symmetry."""
        name = "bcc_rotated" if rotated else "bcc"
        state = ts.io.atoms_to_state(make_structure(name), CPU, DTYPE)
        result = run_optimization_check_symmetry(state, noisy_lj_model, constraint=None)
        assert result["initial_spacegroups"][0] == 229
        assert result["final_spacegroups"][0] != 229

    @pytest.mark.parametrize("rotated", [False, True])
    def test_noisy_model_preserves_symmetry_with_constraint(
        self,
        noisy_lj_model: NoisyModelWrapper,
        *,
        rotated: bool,
    ) -> None:
        """With FixSymmetry, noisy forces still preserve symmetry."""
        name = "bcc_rotated" if rotated else "bcc"
        state = ts.io.atoms_to_state(make_structure(name), CPU, DTYPE)
        constraint = FixSymmetry.from_state(state, symprec=SYMPREC)
        result = run_optimization_check_symmetry(
            state,
            noisy_lj_model,
            constraint=constraint,
        )
        assert result["initial_spacegroups"][0] == 229
        assert result["final_spacegroups"][0] == 229
