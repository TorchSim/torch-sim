# Plan: Add `_system_extras` and `_atom_extras` to `SimState`

## Goal

Allow arbitrary per-system and per-atom tensors to be attached to any `SimState`
without modifying the class definition. The immediate use case is
`external_E_field` and `external_H_field` (both `[n_systems, 3]`), but the
mechanism should be fully general.

## Scope (from issue #463, trimmed to what we are doing)

- Add extensible per-system / per-atom storage to `SimState`
- Ensure all state operations (clone, split, slice, pop, concat, to, from_state)
  preserve extras automatically
- Round-trip extras through ASE IO
- Read extras in `MaceModel.forward` so models can consume them
- Tests

**Out of scope:** MACE-POLAR model support, model output propagation, POLAR
checkpoint URLs, optional dependency wiring.

---

## Design

### Two new private dataclass fields on `SimState`

```python
_system_extras: dict[str, torch.Tensor] = field(default_factory=dict)
_atom_extras: dict[str, torch.Tensor] = field(default_factory=dict)
```

- `_system_extras` values have leading dim `n_systems`
- `_atom_extras` values have leading dim `n_atoms`
- Both are private (`_`-prefixed) so they pass the existing
  `_assert_all_attributes_have_defined_scope` check unchanged
- No `_global_extras` — there is no concrete use case, and `pbc` (the only current
  global attr) is special enough to not warrant a generic bag

### Attribute-style access via `__getattr__`

```python
state.external_E_field  # reads from _system_extras["external_E_field"]
```

`__getattr__` is only invoked when normal lookup fails, so it never shadows
declared fields, properties, or methods.

### Extras flow through operations via `_get_all_attributes`, NOT `get_attrs_for_scope`

The extras dicts are included in `_get_all_attributes` (alongside `_constraints`)
so that `clone()` and `from_state()` copy them automatically. They are handled
explicitly in `_filter_attrs_by_index`, `_split_state`, `concatenate_states`, and
`_state_to_device` — the same pattern used for `_constraints`.

We do NOT modify `get_attrs_for_scope` because its yields are collected into a
flat dict and unpacked into `type(state)(**attrs)`. Extras keys are not dataclass
fields, so that would fail.

---

## Implementation

### 1. `torch_sim/state.py` — `SimState` class

#### 1.1 Add dataclass fields (after `_constraints`, line 94)

```python
_system_extras: dict[str, torch.Tensor] = field(default_factory=dict)
_atom_extras: dict[str, torch.Tensor] = field(default_factory=dict)
```

#### 1.2 Validate shapes in `__post_init__` (after device check, ~line 184)

```python
for key, val in self._system_extras.items():
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"System extra '{key}' must be a torch.Tensor")
    if val.shape[0] != n_systems:
        raise ValueError(
            f"System extra '{key}' leading dim must be "
            f"n_systems={n_systems}, got {val.shape[0]}"
        )
for key, val in self._atom_extras.items():
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"Atom extra '{key}' must be a torch.Tensor")
    if val.shape[0] != self.n_atoms:
        raise ValueError(
            f"Atom extra '{key}' leading dim must be "
            f"n_atoms={self.n_atoms}, got {val.shape[0]}"
        )
```

#### 1.3 Add `__getattr__`

```python
def __getattr__(self, name: str) -> Any:
    # Guard: don't look up private attrs in extras (avoids recursion during init)
    if name.startswith("_"):
        raise AttributeError(name)
    for extras_attr in ("_system_extras", "_atom_extras"):
        try:
            extras = object.__getattribute__(self, extras_attr)
        except AttributeError:
            continue
        if name in extras:
            return extras[name]
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
```

#### 1.4 Add `set_extras` and `has_extras` methods

```python
def set_extras(
    self,
    key: str,
    value: torch.Tensor,
    scope: Literal["per-system", "per-atom"],
) -> None:
    """Set an extras tensor with explicit scope and shape validation."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Extras value must be a torch.Tensor, got {type(value)}")
    if scope == "per-system":
        if value.shape[0] != self.n_systems:
            raise ValueError(
                f"System extras leading dim must be "
                f"n_systems={self.n_systems}, got {value.shape[0]}"
            )
        self._system_extras[key] = value
    elif scope == "per-atom":
        if value.shape[0] != self.n_atoms:
            raise ValueError(
                f"Atom extras leading dim must be "
                f"n_atoms={self.n_atoms}, got {value.shape[0]}"
            )
        self._atom_extras[key] = value
    else:
        raise ValueError(f"scope must be 'per-system' or 'per-atom', got {scope!r}")

def has_extras(self, key: str) -> bool:
    """Check if an extras key exists."""
    return key in self._system_extras or key in self._atom_extras
```

#### 1.5 Update `_get_all_attributes` (line 186-194)

```python
@classmethod
def _get_all_attributes(cls) -> set[str]:
    return (
        cls._atom_attributes
        | cls._system_attributes
        | cls._global_attributes
        | {"_constraints", "_system_extras", "_atom_extras"}
    )
```

This makes `clone()` and `from_state()` work with no other changes — they iterate
`self.attributes.items()` and the dicts get deep-copied.

#### 1.6 Update `_state_to_device` (line 708-739)

After the existing tensor move loop (line 730-732), add:

```python
for extras_key in ("_system_extras", "_atom_extras"):
    if extras_key in attrs and isinstance(attrs[extras_key], dict):
        attrs[extras_key] = {
            k: v.to(device=device) for k, v in attrs[extras_key].items()
        }
```

#### 1.7 Update `_filter_attrs_by_index` (line 768-826)

After the existing per-system loop (after line 824), add:

```python
filtered_attrs["_system_extras"] = {
    key: val[system_indices] for key, val in state._system_extras.items()
}
filtered_attrs["_atom_extras"] = {
    key: val[atom_indices] for key, val in state._atom_extras.items()
}
```

#### 1.8 Update `_split_state` (line 829-896)

After building `split_per_system` (~line 854), add:

```python
split_system_extras: dict[str, list[torch.Tensor]] = {}
for key, val in state._system_extras.items():
    split_system_extras[key] = list(torch.split(val, 1, dim=0))

split_atom_extras: dict[str, list[torch.Tensor]] = {}
for key, val in state._atom_extras.items():
    split_atom_extras[key] = list(torch.split(val, system_sizes, dim=0))
```

Inside the per-system loop, before `states.append`, add to `system_attrs`:

```python
system_attrs["_system_extras"] = {
    key: split_system_extras[key][sys_idx] for key in split_system_extras
}
system_attrs["_atom_extras"] = {
    key: split_atom_extras[key][sys_idx] for key in split_atom_extras
}
```

#### 1.9 Update `concatenate_states` (line 987-1129)

Add before the loop (line 1032):

```python
system_extras_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
atom_extras_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
```

Inside the loop (after per-system collection, ~line 1047):

```python
for key, val in state._system_extras.items():
    system_extras_tensors[key].append(val)
for key, val in state._atom_extras.items():
    atom_extras_tensors[key].append(val)
```

After the loop, before creating the final instance (~line 1117):

```python
concatenated["_system_extras"] = {
    key: torch.cat(tensors, dim=0)
    for key, tensors in system_extras_tensors.items()
}
concatenated["_atom_extras"] = {
    key: torch.cat(tensors, dim=0)
    for key, tensors in atom_extras_tensors.items()
}
```

---

### 2. `torch_sim/io.py` — ASE round-trip

#### 2.1 `state_to_atoms` (line 35-91)

After writing `charge` and `spin` to `atoms.info` (line 84-87):

```python
for key, val in state._system_extras.items():
    atoms.info[key] = val[sys_idx].detach().cpu().numpy()
for key, val in state._atom_extras.items():
    atoms.arrays[key] = val[mask].detach().cpu().numpy()
```

#### 2.2 `atoms_to_state` (line 217-291)

Add optional params to the signature:

```python
def atoms_to_state(
    atoms: "Atoms | list[Atoms]",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    system_extras_keys: list[str] | None = None,
    atom_extras_keys: list[str] | None = None,
) -> "ts.SimState":
```

Before the `return ts.SimState(...)`:

```python
_system_extras: dict[str, torch.Tensor] = {}
if system_extras_keys:
    for key in system_extras_keys:
        vals = [at.info.get(key) for at in atoms_list]
        if all(v is not None for v in vals):
            _system_extras[key] = torch.tensor(
                np.stack(vals), dtype=dtype, device=device
            )

_atom_extras: dict[str, torch.Tensor] = {}
if atom_extras_keys:
    for key in atom_extras_keys:
        arrays = [at.arrays.get(key) for at in atoms_list]
        if all(a is not None for a in arrays):
            _atom_extras[key] = torch.tensor(
                np.concatenate(arrays), dtype=dtype, device=device
            )
```

Pass `_system_extras=_system_extras, _atom_extras=_atom_extras` to the constructor.

---

### 3. `torch_sim/models/mace.py` — consume extras in forward

In `MaceModel.forward`, when building `data_dict` (line 329-341), read
`external_E_field` from the state if present:

```python
data_dict = dict(
    ptr=self.ptr,
    node_attrs=self.node_attrs,
    batch=sim_state.system_idx,
    pbc=sim_state.pbc,
    cell=sim_state.row_vector_cell,
    positions=wrapped_positions,
    edge_index=edge_index,
    unit_shifts=unit_shifts,
    shifts=shifts,
    total_charge=sim_state.charge,
    total_spin=sim_state.spin,
    external_field=getattr(sim_state, "external_E_field", None),
)
```

This is a minimal, backward-compatible change — `external_field` will be `None`
when no extra is set, which is what MACE expects by default.

---

### 4. Tests

#### `tests/test_state.py`

```python
class TestExtras:
    def test_system_extras_construction(self):
        """Extras can be passed at construction time."""
        field = torch.randn(1, 3)
        state = ts.SimState(
            positions=torch.zeros(2, 3),
            masses=torch.ones(2),
            cell=torch.eye(3).unsqueeze(0),
            pbc=True,
            atomic_numbers=torch.tensor([1, 1]),
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
            atomic_numbers=torch.tensor([1, 1]),
            _atom_extras={"tags": tags},
        )
        assert torch.equal(state.tags, tags)

    def test_getattr_missing_raises(self, sim_state):
        with pytest.raises(AttributeError):
            _ = sim_state.nonexistent_key

    def test_set_extras(self, sim_state):
        field = torch.randn(sim_state.n_systems, 3, device=sim_state.device)
        sim_state.set_extras("E", field, scope="per-system")
        assert torch.equal(sim_state.E, field)

    def test_set_extras_bad_shape(self, sim_state):
        bad = torch.randn(sim_state.n_systems + 5, 3)
        with pytest.raises(ValueError):
            sim_state.set_extras("bad", bad, scope="per-system")

    def test_clone_preserves_extras(self, sim_state):
        field = torch.randn(sim_state.n_systems, 3, device=sim_state.device)
        sim_state.set_extras("E", field, scope="per-system")
        cloned = sim_state.clone()
        assert torch.equal(cloned.E, field)
        # verify independence
        cloned._system_extras["E"].zero_()
        assert not torch.equal(sim_state.E, cloned.E)

    def test_split_preserves_extras(self, batched_state):
        field = torch.randn(batched_state.n_systems, 3, device=batched_state.device)
        batched_state.set_extras("H", field, scope="per-system")
        splits = batched_state.split()
        for i, s in enumerate(splits):
            assert torch.equal(s.H, field[i : i + 1])

    def test_getitem_preserves_extras(self, batched_state):
        field = torch.randn(batched_state.n_systems, 3, device=batched_state.device)
        batched_state.set_extras("E", field, scope="per-system")
        sub = batched_state[[0]]
        assert torch.equal(sub.E, field[0:1])

    def test_concatenate_preserves_extras(self, sim_state):
        s1 = sim_state.clone()
        s2 = sim_state.clone()
        f1 = torch.randn(s1.n_systems, 3, device=s1.device)
        f2 = torch.randn(s2.n_systems, 3, device=s2.device)
        s1.set_extras("E", f1, scope="per-system")
        s2.set_extras("E", f2, scope="per-system")
        merged = ts.concatenate_states([s1, s2])
        assert torch.equal(merged.E, torch.cat([f1, f2], dim=0))

    def test_to_device_moves_extras(self, sim_state):
        field = torch.randn(sim_state.n_systems, 3, device=sim_state.device)
        sim_state.set_extras("E", field, scope="per-system")
        moved = sim_state.to(device=sim_state.device)
        assert moved.E.device == sim_state.device

    def test_pop_preserves_extras(self, batched_state):
        field = torch.randn(batched_state.n_systems, 3, device=batched_state.device)
        batched_state.set_extras("E", field, scope="per-system")
        popped = batched_state.pop(0)
        assert popped[0].E.shape[0] == 1

    def test_has_extras(self, sim_state):
        assert not sim_state.has_extras("E")
        sim_state.set_extras(
            "E", torch.zeros(sim_state.n_systems, 3, device=sim_state.device),
            scope="per-system",
        )
        assert sim_state.has_extras("E")

    def test_post_init_validation_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            ts.SimState(
                positions=torch.zeros(2, 3),
                masses=torch.ones(2),
                cell=torch.eye(3).unsqueeze(0),
                pbc=True,
                atomic_numbers=torch.tensor([1, 1]),
                _system_extras={"bad": torch.randn(5, 3)},
            )

    def test_from_state_preserves_extras(self, sim_state):
        field = torch.randn(sim_state.n_systems, 3, device=sim_state.device)
        sim_state.set_extras("E", field, scope="per-system")
        new = ts.SimState.from_state(sim_state)
        assert torch.equal(new.E, field)

    def test_extras_dont_shadow_declared_fields(self, sim_state):
        sim_state._system_extras["cell"] = torch.zeros(sim_state.n_systems, 3)
        # __getattr__ is NOT called for 'cell' because it's a real attribute
        assert sim_state.cell.shape[-2:] == (3, 3)
```

#### `tests/test_io.py`

```python
def test_system_extras_atoms_roundtrip():
    state = ts.SimState(
        positions=torch.zeros(2, 3),
        masses=torch.ones(2),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1]),
        _system_extras={"external_E_field": torch.tensor([[1.0, 0.0, 0.0]])},
    )
    atoms_list = state.to_atoms()
    assert "external_E_field" in atoms_list[0].info
    restored = ts.io.atoms_to_state(
        atoms_list, system_extras_keys=["external_E_field"],
    )
    assert torch.allclose(restored.external_E_field, state.external_E_field)

def test_atom_extras_atoms_roundtrip():
    tags = torch.tensor([1.0, 2.0])
    state = ts.SimState(
        positions=torch.zeros(2, 3),
        masses=torch.ones(2),
        cell=torch.eye(3).unsqueeze(0),
        pbc=True,
        atomic_numbers=torch.tensor([1, 1]),
        _atom_extras={"tags": tags},
    )
    atoms_list = state.to_atoms()
    assert "tags" in atoms_list[0].arrays
    restored = ts.io.atoms_to_state(
        atoms_list, atom_extras_keys=["tags"],
    )
    assert torch.allclose(restored.tags, state.tags)
```

---

## Usage

```python
import torch
import torch_sim as ts

state = ts.initialize_state(atoms_list, device="cuda", dtype=torch.float64)

# Attach external fields
state.set_extras(
    "external_E_field",
    torch.tensor([[0.0, 0.0, 0.1]] * state.n_systems, device=state.device),
    scope="per-system",
)
state.set_extras(
    "external_H_field",
    torch.zeros(state.n_systems, 3, device=state.device),
    scope="per-system",
)

# Read via attribute access
print(state.external_E_field)  # [n_systems, 3]

# All state ops preserve extras
splits = state.split()
assert splits[0].has_extras("external_E_field")

merged = ts.concatenate_states(splits)
assert torch.equal(merged.external_E_field, state.external_E_field)

# Construction-time
state = ts.SimState(
    ...,
    _system_extras={"external_E_field": E_field, "external_H_field": H_field},
)
```

---

## Files touched

| File | Changes |
|------|---------|
| `torch_sim/state.py` | Add `_system_extras`/`_atom_extras` fields, `__post_init__` validation, `__getattr__`, `set_extras`/`has_extras`, update `_get_all_attributes`, `_state_to_device`, `_filter_attrs_by_index`, `_split_state`, `concatenate_states` |
| `torch_sim/io.py` | `state_to_atoms`: write extras. `atoms_to_state`: add `system_extras_keys`/`atom_extras_keys` params |
| `torch_sim/models/mace.py` | Read `external_E_field` from extras in `forward` data_dict |
| `tests/test_state.py` | `TestExtras` class (~15 tests) |
| `tests/test_io.py` | Two round-trip tests |

## Checklist

- [ ] Add `_system_extras` and `_atom_extras` dataclass fields to `SimState`
- [ ] Validate extras shapes in `__post_init__`
- [ ] Add `__getattr__` for attribute-style read access
- [ ] Add `set_extras` and `has_extras` methods
- [ ] Update `_get_all_attributes` to include `_system_extras` and `_atom_extras`
- [ ] Update `_state_to_device` to move extras tensors
- [ ] Update `_filter_attrs_by_index` to index extras
- [ ] Update `_split_state` to split extras
- [ ] Update `concatenate_states` to concatenate extras
- [ ] Update `state_to_atoms` to write extras to `atoms.info`/`atoms.arrays`
- [ ] Update `atoms_to_state` with `system_extras_keys`/`atom_extras_keys` params
- [ ] Update `MaceModel.forward` to pass `external_E_field` from extras
- [ ] Add extras tests in `tests/test_state.py`
- [ ] Add IO round-trip tests in `tests/test_io.py`
