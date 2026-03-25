import pytest
import torch

import torch_sim as ts


DEVICE = torch.device("cpu")
DTYPE = torch.float64


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
        # Post-init validation should also catch shadowing during construction
        with pytest.raises(ValueError, match="shadows an existing attribute"):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
                _system_extras={"cell": torch.zeros(1, 3)},
            )

    # store_model_extras
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
        assert not state._system_extras  # noqa: SLF001
        assert not state._atom_extras  # noqa: SLF001

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


def test_system_extras_atoms_roundtrip():
    state = ts.SimState(
        positions=torch.zeros(2, 3),
        masses=torch.ones(2),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
        _system_extras={"external_E_field": torch.tensor([[1.0, 0.0, 0.0]])},
    )
    atoms_list = state.to_atoms()
    assert "external_E_field" in atoms_list[0].info
    restored = ts.io.atoms_to_state(
        atoms_list,
        system_extras_keys=["external_E_field"],
    )
    assert torch.allclose(restored.external_E_field, state.external_E_field)


def test_atom_extras_atoms_roundtrip():
    tags = torch.tensor([1.0, 2.0])
    state = ts.SimState(
        positions=torch.zeros(2, 3),
        masses=torch.ones(2),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1], dtype=torch.int),
        _atom_extras={"tags": tags},
    )
    atoms_list = state.to_atoms()
    assert "tags" in atoms_list[0].arrays
    restored = ts.io.atoms_to_state(
        atoms_list,
        atom_extras_keys=["tags"],
    )
    assert torch.allclose(restored.tags, state.tags)
