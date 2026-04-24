## Examples

Tutorials are intended to provide pedagogical walkthroughs of TorchSim's core functionality

## Tutorial Formatting

All tutorials are built for the documentation and must follow some formatting rules:

1. They must follow the [jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format)
where code blocks are annotated with `# %%` and markdown blocks
are annotated with `# %% [markdown]`.
2. They must begin with a markdown block with a top level header
(e.g. #) and that must be the only top level header in the file.
This is to ensure documentation builds correctly.
3. If they use a external model, they should be placed in a separate
folder named after the model and CI should be updated to make sure
they are correctly executed.
4. Cells should return sensible values or None as they are executed
when docs are built.

Tutorials are converted to `.ipynb` files and executed when the docs are built. If you
add a new tutorial, add it to the
[/docs/tutorials/index.rst](/docs/tutorials/index.rst) file.

## Example Execution

Both scripts and tutorials are tested in CI, this ensures that all documentation stays
up to date and helps catch edge cases. To support this, all of the scripts and
tutorials have any additional dependencies included at the top.

If you'd like to execute the scripts or examples locally, you can run them with:

```sh
# if uv is not yet installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick any of the examples
uv run --with-editable . examples/scripts/1_introduction.py
uv run --with-editable . examples/scripts/2_structural_optimization.py
uv run --with-editable . examples/scripts/3_dynamics.py
uv run --with-editable . examples/scripts/4_high_level_api.py

# or any of the tutorials
uv run --with-editable . examples/tutorials/diff_sim.py
```

## Benchmarking Scripts

The `examples/benchmarking/` folder contains standalone benchmark scripts. They
declare their own dependencies via [PEP 723 inline script metadata](https://peps.python.org/pep-0723/)
and should be run with `uv run --with-editable .` so that the local `torch-sim` package
is available alongside the script's isolated dependency environment:

```sh
# Neighbor-list backend benchmark on WBM or MP structures
uv run --with-editable . examples/benchmarking/neighborlists.py \
    --source wbm --n-structures 100 --device cpu

# Scaling benchmark: static, relax, NVE, NVT
uv run --with ".[mace]" examples/benchmarking/scaling.py

# MD throughput: ASE Langevin vs torch-sim batched NVT-Langevin
uv run --with ".[mace]" examples/benchmarking/md-throughput.py --model mace

# Optimization throughput: ASE vs torch-sim LBFGS/FIRE on WBM structures
uv run --with ".[mace]" examples/benchmarking/opt-throughput.py \
    --model mace --optimizer lbfgs --n-structures 50
```

> **Note:** `--with .` installs the local editable `torch-sim` package into the
> script's isolated environment. Using `--no-project` instead would skip this
> and fail to find `torch_sim`.
