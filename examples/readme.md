## Types of Examples

Examples are provided in two foms:

* Tutorials are intended to provide pedagogical walkthroughs of TorchSim's core functionality
* Scripts are a holdover from early torch-sim development, they are not currently recommended as a learning resource. See issue [here](https://github.com/Radical-AI/torch-sim/issues/109).

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
uv run --with . examples/2_Structural_optimization/2.3_MACE_FIRE.py
uv run --with . examples/3_Dynamics/3.3_MACE_NVE_cueq.py
uv run --with . examples/4_High_level_api/4.1_high_level_api.py
```
