
Examples are provided as tutorials and scripts.
Tutorials are intended to provide pedagogical walkthroughs
of TorchSim's core functionality, whereas scripts are intended
to demonstrate a wide variety of use cases with few interspersed
explanation.

Tutorials must follow some formatting rules:
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

Both scripts and tutorials are tested in CI. Tutorials are
converted to `.ipynb` files and executed when the docs are built.

