# Restore PR #460 & #466 Changes Overwritten by PR #339

## Context

- **PR #466** (`795ef57..ad8624a`): Standardize parameter typing to `float | torch.Tensor`
- **PR #460** (`ad8624a..48dcfb1`): Introduce PRNG to SimState, remove `seed` params from integrators
- **PR #339** (`48dcfb1..ce9ac4c`): Run `ty` in lint CI — but also re-introduced `seed` params and added `calculate_momenta`, undoing #460's core design
- **Current HEAD**: `213e2ed` (a "maint" commit on top of #339)

## Problem

PR #339 re-introduced `seed: int | None = None` to every integrator init function and added a `calculate_momenta()` wrapper, undoing PR #460's design where `state.rng` is the sole source of randomness.

## Plan

### Step 1: Reset to before PR #339

```bash
# On branch restore-with-ty (already exists)
git stash drop          # drop the bad partial edits stash
git reset --hard 48dcfb1  # reset to just after #460, before #339
```

### Step 2: Identify file sets

```bash
# Files changed by #460
git diff --name-only ad8624a..48dcfb1

# Files changed by #466
git diff --name-only 795ef57..ad8624a

# Files changed by #339
git diff --name-only 48dcfb1..ce9ac4c

# Files in #339 that DON'T overlap with #460 or #466 (safe to take)
comm -23 \
  <(git diff --name-only 48dcfb1..ce9ac4c | sort) \
  <(cat <(git diff --name-only ad8624a..48dcfb1) <(git diff --name-only 795ef57..ad8624a) | sort -u)
```

### Step 3: Take safe (non-conflicting) files from #339

```bash
# Checkout the non-conflicting files from ce9ac4c
git checkout ce9ac4c -- <safe-files...>
git commit -m "Restore non-conflicting changes from #339 (ty linting setup)"
```

These are likely: `.pre-commit-config.yaml`, `pyproject.toml`, new test files, etc.

### Step 4: Manually review conflicting files

The conflicting files (touched by both #339 AND #460/#466) need manual review. These are primarily:

- `torch_sim/integrators/md.py` — #339 added `calculate_momenta()`, needs removal
- `torch_sim/integrators/nvt.py` — #339 added `seed` params to 3 init fns
- `torch_sim/integrators/nve.py` — #339 added `seed` param to `nve_init`
- `torch_sim/integrators/npt.py` — #339 added `seed` params to 3 init fns
- `torch_sim/state.py` — #339 renamed some vars (`new` → `new_generator`, etc.)

For each, compare:
```bash
git diff 48dcfb1..ce9ac4c -- <file>
```

**Keep from #339**: `ensure_sim_state()` usage, `require_system_idx()` calls, `torch.tensor()` over `torch.as_tensor()` changes, any `ty`-related fixes.

**Discard from #339**: `seed` params, `calculate_momenta` usage, removal of `state.rng` direct usage.

### Step 5: Update example callers

After restoring integrator APIs, update examples that used `seed=`:
- `examples/scripts/3_dynamics.py` — change `seed=1` to `state.rng = 1`
- `examples/tutorials/hybrid_swap_tutorial.py` — change `seed=42` to `state.rng = 42`

Note: `tests/workflows/test_a2c.py` uses `seed=42` but that's for the A2C workflow's own `_make_torch_generator`, not integrator init — leave it alone.

### Step 6: Also apply the `213e2ed` maint commit

```bash
git diff ce9ac4c..213e2ed  # check what the maint commit did
# Apply relevant parts
```
