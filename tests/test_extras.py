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
            _system_extras={"external_E_field": field},
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

    def test_set_extras(self, cu_sim_state: ts.SimState):
        field = torch.randn(cu_sim_state.n_systems, 3, device=cu_sim_state.device)
        cu_sim_state.set_extras("E", field, scope="per-system")
        assert torch.equal(cu_sim_state.E, field)

    def test_set_extras_bad_shape(self, cu_sim_state: ts.SimState):
        bad = torch.randn(cu_sim_state.n_systems + 5, 3)
        with pytest.raises(ValueError, match="leading dim must be n_systems"):
            cu_sim_state.set_extras("bad", bad, scope="per-system")

    def test_clone_preserves_extras(self, cu_sim_state: ts.SimState):
        field = torch.randn(cu_sim_state.n_systems, 3, device=cu_sim_state.device)
        cu_sim_state.set_extras("E", field, scope="per-system")
        cloned = cu_sim_state.clone()
        assert torch.equal(cloned.E, field)
        # verify independence
        cloned.system_extras["E"].zero_()
        assert not torch.equal(cu_sim_state.E, cloned.E)

    def test_split_preserves_extras(self, si_double_sim_state: ts.SimState):
        field = torch.randn(
            si_double_sim_state.n_systems, 3, device=si_double_sim_state.device
        )
        si_double_sim_state.set_extras("H", field, scope="per-system")
        splits = si_double_sim_state.split()
        for i, s in enumerate(splits):
            assert torch.equal(s.H, field[i : i + 1])

    def test_getitem_preserves_extras(self, si_double_sim_state: ts.SimState):
        field = torch.randn(
            si_double_sim_state.n_systems, 3, device=si_double_sim_state.device
        )
        si_double_sim_state.set_extras("E", field, scope="per-system")
        sub = si_double_sim_state[[0]]
        assert torch.equal(sub.E, field[0:1])

    def test_concatenate_preserves_extras(self, cu_sim_state: ts.SimState):
        s1 = cu_sim_state.clone()
        s2 = cu_sim_state.clone()
        f1 = torch.randn(s1.n_systems, 3, device=s1.device)
        f2 = torch.randn(s2.n_systems, 3, device=s2.device)
        s1.set_extras("E", f1, scope="per-system")
        s2.set_extras("E", f2, scope="per-system")
        merged = ts.concatenate_states([s1, s2])
        assert torch.equal(merged.E, torch.cat([f1, f2], dim=0))

    def test_to_device_moves_extras(self, cu_sim_state: ts.SimState):
        field = torch.randn(cu_sim_state.n_systems, 3, device=cu_sim_state.device)
        cu_sim_state.set_extras("E", field, scope="per-system")
        moved = cu_sim_state.to(device=cu_sim_state.device)
        assert moved.E.device == cu_sim_state.device

    def test_pop_preserves_extras(self, si_double_sim_state: ts.SimState):
        field = torch.randn(
            si_double_sim_state.n_systems, 3, device=si_double_sim_state.device
        )
        si_double_sim_state.set_extras("E", field, scope="per-system")
        popped = si_double_sim_state.pop(0)
        assert popped[0].E.shape[0] == 1

    def test_has_extras(self, cu_sim_state: ts.SimState):
        assert not cu_sim_state.has_extras("E")
        cu_sim_state.set_extras(
            "E",
            torch.zeros(cu_sim_state.n_systems, 3, device=cu_sim_state.device),
            scope="per-system",
        )
        assert cu_sim_state.has_extras("E")

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

    def test_from_state_preserves_extras(self, cu_sim_state: ts.SimState):
        field = torch.randn(cu_sim_state.n_systems, 3, device=cu_sim_state.device)
        cu_sim_state.set_extras("E", field, scope="per-system")
        new = ts.SimState.from_state(cu_sim_state)
        assert torch.equal(new.E, field)

    def test_extras_cannot_shadow_declared_fields(self, cu_sim_state: ts.SimState):
        # set_extras should raise if attempting to shadow
        with pytest.raises(ValueError, match="shadows an existing attribute"):
            cu_sim_state.set_extras(
                "cell", torch.zeros(cu_sim_state.n_systems, 3), scope="per-system"
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
