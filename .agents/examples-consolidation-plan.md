# Examples Consolidation Plan

## Problem

There is significant duplication between `examples/scripts/` and `examples/tutorials/`. Both are run in CI (`find examples -name "*.py"`), but only tutorials are built into the docs (via jupytext → ipynb → sphinx). The scripts provide no pedagogical value beyond what tutorials offer, and maintaining both doubles the surface area for breakage.

## Current Inventory

### Tutorials (jupytext format, built into docs)

| File | Topics |
|------|--------|
| `high_level_tutorial.py` | `ts.integrate`, `ts.optimize`, `ts.static`, batching, reporting, autobatching, pymatgen |
| `state_tutorial.py` | SimState creation, batching, slicing/popping, conversion to ASE/pymatgen/phonopy |
| `reporting_tutorial.py` | TorchSimTrajectory (low-level), TrajectoryReporter (high-level), multi-batch |
| `autobatching_tutorial.py` | BinningAutoBatcher, InFlightAutoBatcher, memory scaling |
| `low_level_tutorial.py` | Direct model calls, FIRE init/step, NVT Langevin init/step, units |
| `hybrid_swap_tutorial.py` | Custom state objects, hybrid MD + swap Monte Carlo |
| `diff_sim.py` | Differentiable simulation, meta-optimization with soft spheres |

### Scripts (plain Python, CI-only)

| File | Topics | Overlap with tutorials |
|------|--------|----------------------|
| `1_introduction.py` | LJ model eval, MACE batched inference (raw dict API) | **Heavy** — covered by `low_level_tutorial` + `state_tutorial` |
| `2_structural_optimization.py` | FIRE, GD, L-BFGS, BFGS, unit cell filter, Fréchet filter, pressure | Partial — FIRE covered in `low_level_tutorial`, rest is unique |
| `3_dynamics.py` | NVE, NVT Langevin, NVT Nose-Hoover, NPT Nose-Hoover | Partial — NVT Langevin in `low_level_tutorial`, rest is unique |
| `4_high_level_api.py` | `ts.integrate`, `ts.optimize`, reporting, batching, pymatgen | **Full** — completely covered by `high_level_tutorial` |
| `5_workflow.py` | InFlight autobatching, elastic constants | **Heavy** — autobatching in `autobatching_tutorial`, elastic is unique |
| `6_phonons.py` | Phonon DOS, band structure, Phonopy integration | **Unique** — no tutorial coverage |
| `7_others.py` | Neighbor lists (linked cell, N²), VACF | **Unique** — no tutorial coverage |
| `8_benchmarking.py` | Scaling benchmarks for static/relax/NVE/NVT | **Unique** — not tutorial material |
| `7_Others/7.5_Batched_MACE_NEB.py` | NEB debugging script (hardcoded local paths) | N/A — dev script, not a real example |

## Plan

### Phase 1: Delete fully redundant scripts

- [ ] Delete `scripts/1_introduction.py`
  - §1 (LJ eval) → covered by `low_level_tutorial.py`
  - §2 (batched MACE) → covered by `state_tutorial.py` + `high_level_tutorial.py`
- [ ] Delete `scripts/4_high_level_api.py`
  - Every section is a less-documented version of `high_level_tutorial.py`

### Phase 2: Convert partially-overlapping scripts to tutorials

- [ ] Create `tutorials/dynamics_tutorial.py` from `scripts/3_dynamics.py`
  - Cover NVE, NVT Langevin, NVT Nose-Hoover, NPT Nose-Hoover
  - Add markdown explanations for each ensemble
  - Show energy conservation (NVE), thermostat behavior (NVT), barostat (NPT)
  - Delete `scripts/3_dynamics.py` after

- [ ] Create `tutorials/optimization_tutorial.py` from `scripts/2_structural_optimization.py`
  - Cover FIRE, gradient descent, L-BFGS, BFGS
  - Cover cell filters: none, unit cell, Fréchet
  - Show pressure convergence
  - Trim the 8 nearly-identical sections into a more concise comparison
  - Delete `scripts/2_structural_optimization.py` after

- [ ] Create `tutorials/elastic_tutorial.py` from `scripts/5_workflow.py` §2
  - Structure relaxation → Bravais type detection → elastic tensor → moduli
  - Delete `scripts/5_workflow.py` after (§1 autobatching is redundant with `autobatching_tutorial.py`)

### Phase 3: Convert unique scripts to tutorials

- [ ] Create `tutorials/phonons_tutorial.py` from `scripts/6_phonons.py`
  - Already structured like a tutorial, just needs jupytext format + markdown cells
  - Delete `scripts/6_phonons.py` after

- [ ] Create `tutorials/neighbor_lists_tutorial.py` from `scripts/7_others.py` §1
  - Neighbor list algorithms (linked cell vs N²) are worth documenting
  - VACF section could be folded in or kept separate
  - Delete `scripts/7_others.py` after

### Phase 4: Clean up remaining scripts

- [ ] Keep `scripts/8_benchmarking.py` as-is (not tutorial material, utility for perf testing)
- [ ] Remove `scripts/7_Others/7.5_Batched_MACE_NEB.py` — hardcoded local model paths, debugging script, not a portable example
- [ ] Remove `scripts/7_Others/neb_path_torchsim_fire_5im.hdf5` — binary artifact for the NEB script

### Phase 5: Update docs and CI

- [ ] Add new tutorials to `docs/tutorials/index.rst`
- [ ] Update `examples/readme.md` to reflect new structure
- [ ] Delete `scripts/readme.md` (will be mostly empty)
- [ ] Verify CI still discovers and runs all examples via `find examples -name "*.py"`

## Final Structure

```
examples/
├── readme.md
├── tutorials/
│   ├── high_level_tutorial.py      (existing)
│   ├── state_tutorial.py           (existing)
│   ├── reporting_tutorial.py       (existing)
│   ├── autobatching_tutorial.py    (existing)
│   ├── low_level_tutorial.py       (existing)
│   ├── hybrid_swap_tutorial.py     (existing)
│   ├── diff_sim.py                 (existing)
│   ├── dynamics_tutorial.py        (new, from scripts/3_dynamics.py)
│   ├── optimization_tutorial.py    (new, from scripts/2_structural_optimization.py)
│   ├── elastic_tutorial.py         (new, from scripts/5_workflow.py §2)
│   ├── phonons_tutorial.py         (new, from scripts/6_phonons.py)
│   └── neighbor_lists_tutorial.py  (new, from scripts/7_others.py)
└── scripts/
    └── 8_benchmarking.py           (kept, not tutorial material)
```

## Notes

- All new tutorials must follow jupytext percent format (`# %%` / `# %% [markdown]`)
- All tutorials must have exactly one top-level `#` header
- External model dependencies should be declared in `# /// script` blocks
- CI smoke test support (`SMOKE_TEST = os.getenv("CI") is not None`) should be preserved
- Tutorials should be trimmed vs the scripts — no need for 8 near-identical optimization sections when a well-explained comparison of 3-4 approaches is clearer
