import pytest
import torch

import torch_sim as ts
from torch_sim.state import get_attrs_for_scope
from torch_sim.units import BaseConstant, UnitConversion


def _make_state_with_extras(n_atoms: int = 4, n_systems: int = 2) -> ts.SimState:
    """Helper: build a minimal multi-system state with both extras dicts populated."""
    system_idx = torch.cat(
        [
            torch.zeros(n_atoms // n_systems, dtype=torch.long),
            torch.ones(n_atoms // n_systems, dtype=torch.long),
        ]
    )
    return ts.SimState(
        positions=torch.randn(n_atoms, 3),
        masses=torch.ones(n_atoms),
        cell=torch.eye(3).unsqueeze(0).expand(n_systems, 3, 3).clone(),
        pbc=True,
        atomic_numbers=torch.ones(n_atoms, dtype=torch.int),
        system_idx=system_idx,
        _system_extras={"charge": torch.tensor([1.0, 2.0])},
        _atom_extras={"tags": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )


class TestExtras:
    def test_system_extras_construction(self):
        """Extras can be passed at construction time."""
        field = torch.randn(1, 3)
        state = ts.SimState(
            positions=torch.zeros(2, 3),
            masses=torch.ones(2),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
            external_E_field=field,
        )
        assert torch.equal(state.external_E_field, field)

    def test_atom_extras_construction(self):
        """Per-atom extras work at construction time."""
        tags = torch.tensor([1.0, 2.0])
        state = ts.SimState(
            positions=torch.zeros(2, 3),
            masses=torch.ones(2),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
            _atom_extras={"tags": tags},
        )
        assert torch.equal(state.tags, tags)

    def test_getattr_missing_raises_attribute_error(self, cu_sim_state: ts.SimState):
        with pytest.raises(AttributeError, match="nonexistent_key"):
            _ = cu_sim_state.nonexistent_key

    def test_post_init_validation_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="leading dim must be n_systems"):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
                _system_extras={"bad": torch.randn(5, 3)},
            )

    def test_construction_extras_cannot_shadow(self):
        with pytest.raises(ValueError, match="shadows an existing attribute"):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
                _system_extras={"cell": torch.zeros(1, 3)},
            )

    def test_store_model_extras_canonical_keys_not_stored(
        self, si_double_sim_state: ts.SimState
    ):
        """Canonical keys (energy, forces, stress) must not land in extras."""
        state = si_double_sim_state.clone()
        state.store_model_extras(
            {
                "energy": torch.randn(state.n_systems),
                "forces": torch.randn(state.n_atoms, 3),
                "stress": torch.randn(state.n_systems, 3, 3),
            }
        )
        for key in ("energy", "forces", "stress"):
            assert not state.has_extras(key)

    def test_store_model_extras_per_system(self, si_double_sim_state: ts.SimState):
        """Tensors with leading dim == n_systems go into system_extras."""
        state = si_double_sim_state.clone()
        dipole = torch.randn(state.n_systems, 3)
        state.store_model_extras(
            {"energy": torch.randn(state.n_systems), "dipole": dipole}
        )
        assert torch.equal(state.dipole, dipole)

    def test_store_model_extras_per_atom(self, si_double_sim_state: ts.SimState):
        """Tensors with leading dim == n_atoms go into atom_extras."""
        state = si_double_sim_state.clone()
        charges = torch.randn(state.n_atoms)
        density = torch.randn(state.n_atoms, 8)
        state.store_model_extras(
            {
                "energy": torch.randn(state.n_systems),
                "charges": charges,
                "density_coefficients": density,
            }
        )
        assert torch.equal(state.charges, charges)
        assert state.density_coefficients.shape == (state.n_atoms, 8)

    def test_store_model_extras_skips_scalars(self, si_double_sim_state: ts.SimState):
        """0-d tensors and non-Tensor values are silently ignored."""
        state = si_double_sim_state.clone()
        state.store_model_extras(
            {
                "scalar": torch.tensor(3.14),
                "string": "not a tensor",
            }
        )
        assert not state.has_extras("scalar")
        assert not state.has_extras("string")


class TestExtrasStateOps:
    """Tests for extras propagation through clone/split/concat/filter/to."""

    def test_clone_preserves_extras(self):
        state = _make_state_with_extras()
        cloned = state.clone()
        assert torch.equal(cloned.charge, state.charge)
        assert torch.equal(cloned.tags, state.tags)
        assert cloned.system_extras is not state.system_extras
        assert cloned.atom_extras is not state.atom_extras
        assert cloned.charge.data_ptr() != state.charge.data_ptr()

    def test_split_preserves_extras(self):
        state = _make_state_with_extras()
        splits = state.split()
        assert len(splits) == 2
        assert splits[0].charge.item() == 1.0
        assert splits[1].charge.item() == 2.0
        assert torch.equal(splits[0].tags, torch.tensor([10.0, 20.0]))
        assert torch.equal(splits[1].tags, torch.tensor([30.0, 40.0]))

    def test_concatenate_merges_extras(self):
        s1 = _make_state_with_extras()
        splits = s1.split()
        merged = ts.concatenate_states(splits)
        assert torch.equal(merged.charge, s1.charge)
        assert torch.equal(merged.tags, s1.tags)

    def test_filter_by_index_preserves_extras(self):
        state = _make_state_with_extras()
        sliced = state[0]
        assert sliced.charge.shape == (1,)
        assert sliced.charge.item() == 1.0
        assert sliced.tags.shape == (2,)
        assert torch.equal(sliced.tags, torch.tensor([10.0, 20.0]))

    def test_to_dtype_converts_extras(self):
        state = _make_state_with_extras()
        converted = state.to(dtype=torch.float64)
        assert converted.charge.dtype == torch.float64
        assert converted.tags.dtype == torch.float64

    def test_to_dtype_preserves_int_extras(self):
        state = ts.SimState(
            positions=torch.zeros(2, 3),
            masses=torch.ones(2),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
            _atom_extras={"int_tags": torch.tensor([1, 2], dtype=torch.int)},
        )
        converted = state.to(dtype=torch.float64)
        assert converted._atom_extras["int_tags"].dtype == torch.int  # noqa: SLF001


class TestExtrasSetattr:
    """Tests for __setattr__ routing writes into extras dicts."""

    def test_write_existing_system_extra(self):
        state = _make_state_with_extras()
        new_val = torch.tensor([99.0, 100.0])
        state.charge = new_val
        assert torch.equal(state.charge, new_val)

    def test_write_existing_atom_extra(self):
        state = _make_state_with_extras()
        new_val = torch.tensor([5.0, 6.0, 7.0, 8.0])
        state.tags = new_val
        assert torch.equal(state.tags, new_val)

    def test_delete_extra_by_setting_none(self):
        state = _make_state_with_extras()
        state.charge = None
        assert not state.has_extras("charge")

    def test_write_core_attr_unaffected(self):
        state = _make_state_with_extras()
        new_pos = torch.randn(4, 3)
        state.positions = new_pos
        assert torch.equal(state.positions, new_pos)


class TestExtrasInitRouting:
    """Tests for _wrap_init_for_extras auto-routing of unknown kwargs."""

    def test_atom_sized_kwarg_routes_to_atom_extras(self):
        n_atoms = 3
        charges = torch.randn(n_atoms)
        state = ts.SimState(
            positions=torch.zeros(n_atoms, 3),
            masses=torch.ones(n_atoms),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.ones(n_atoms, dtype=torch.int),
            partial_charges=charges,
        )
        assert state.has_extras("partial_charges")
        assert torch.equal(state.partial_charges, charges)

    def test_none_kwarg_skipped(self):
        state = ts.SimState(
            positions=torch.zeros(2, 3),
            masses=torch.ones(2),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
            extra_key=None,
        )
        assert not state.has_extras("extra_key")


class TestExtrasFromState:
    """Tests for from_state routing of unknown additional_attrs."""

    def test_atom_tensor_routes_to_atom_extras(self, si_sim_state: ts.SimState):
        charges = torch.randn(si_sim_state.n_atoms)
        new_state = ts.SimState.from_state(si_sim_state, partial_charges=charges)
        assert "partial_charges" in new_state.atom_extras
        assert torch.equal(new_state.partial_charges, charges)

    def test_system_tensor_routes_to_system_extras(self, si_sim_state: ts.SimState):
        dipole = torch.randn(si_sim_state.n_systems, 3)
        new_state = ts.SimState.from_state(si_sim_state, dipole=dipole)
        assert "dipole" in new_state._system_extras  # noqa: SLF001
        assert torch.equal(new_state.dipole, dipole)

    def test_invalid_leading_dim_raises(self, si_sim_state: ts.SimState):
        bad = torch.randn(999, 3)
        with pytest.raises(ValueError, match="invalid leading dimension"):
            ts.SimState.from_state(si_sim_state, bad_tensor=bad)

    def test_source_extras_preserved(self):
        state = _make_state_with_extras()
        new_state = ts.SimState.from_state(state)
        assert torch.equal(new_state.charge, state.charge)
        assert torch.equal(new_state.tags, state.tags)


class TestExtrasScope:
    """Test that get_attrs_for_scope yields extras."""

    def test_system_extras_in_per_system_scope(self):
        state = _make_state_with_extras()
        attrs = dict(get_attrs_for_scope(state, "per-system"))
        assert "charge" in attrs
        assert torch.equal(attrs["charge"], state.charge)

    def test_atom_extras_in_per_atom_scope(self):
        state = _make_state_with_extras()
        attrs = dict(get_attrs_for_scope(state, "per-atom"))
        assert "tags" in attrs
        assert torch.equal(attrs["tags"], state.tags)

    def test_extras_not_in_global_scope(self):
        state = _make_state_with_extras()
        attrs = dict(get_attrs_for_scope(state, "global"))
        assert "charge" not in attrs
        assert "tags" not in attrs


class TestExtrasValidation:
    """Validation edge cases for extras at construction time."""

    def test_atom_extra_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="leading dim must be n_atoms"):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
                _atom_extras={"bad": torch.randn(999)},
            )

    def test_non_tensor_extra_rejected(self):
        with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
                _system_extras={"bad": "string"},
            )

    def test_atom_extra_shadowing_rejected(self):
        with pytest.raises(ValueError, match="shadows an existing attribute"):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
                _atom_extras={"positions": torch.randn(2, 3)},
            )


class TestExtrasProperties:
    """Direct tests for system_extras/atom_extras properties and has_extras."""

    def test_system_extras_property(self):
        state = _make_state_with_extras()
        assert isinstance(state.system_extras, dict)
        assert "charge" in state.system_extras
        assert state.system_extras is state._system_extras  # noqa: SLF001

    def test_atom_extras_property(self):
        state = _make_state_with_extras()
        assert isinstance(state.atom_extras, dict)
        assert "tags" in state.atom_extras
        assert state.atom_extras is state._atom_extras  # noqa: SLF001

    def test_has_extras_positive(self):
        state = _make_state_with_extras()
        assert state.has_extras("charge")
        assert state.has_extras("tags")

    def test_has_extras_negative(self):
        state = _make_state_with_extras()
        assert not state.has_extras("nonexistent")


class TestUnitsEnum:
    """Smoke tests for BaseConstant/UnitConversion float-enum arithmetic."""

    def test_base_constant_arithmetic(self):
        assert float(BaseConstant.e) == pytest.approx(1.6021766208e-19)
        assert float(BaseConstant.e) * 2 == pytest.approx(2 * 1.6021766208e-19)
        assert isinstance(BaseConstant.e, float)

    def test_unit_conversion_arithmetic(self):
        result = UnitConversion.Ang_to_met * 1e10
        assert result == pytest.approx(1.0)
        assert isinstance(UnitConversion.eV_per_Ang3_to_GPa, float)

    def test_unit_conversion_in_expressions(self):
        gpa = UnitConversion.eV_per_Ang3_to_GPa
        assert gpa > 100  # ~160.2
        assert isinstance(gpa + 0.0, float)
