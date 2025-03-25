# Developer Installation

You can install torch_sim with `pip` or from source.

## Install using pip

You can install the basic functionality of torch_sim using pip:

```bash
pip install torch_sim
```

If you are planning to use torch_sim with fireworks, you can install the optional
fireworks components:

## Install from source

To install torch_sim from source, clone the repository from [github](https://github.com/Radical-AI/torch-sim)

```bash
git clone https://github.com/Radical-AI/torch_sim
cd torch_sim
pip install .
```

Or do a developer install by using the `-e` flag:

```bash
pip install -e .
```

### Installing pre-commit

If you're planning on contributing to the torch_sim source, you should also install
the developer requirements with:

```bash
pip install -e .
pre-commit install
pre-commit run --all-files
```

The `pre-commit` command will ensure that changes to the source code match the
torch_sim style guidelines by running code linters such as `black` and `ruff` automatically with each commit.

## Running unit tests

Unit tests can be run from the source folder using `pytest`. First, the requirements
to run tests must be installed:

```bash
pip install .[test]
```

And the tests run using:

```bash
pytest
```

## Building the documentation

The torch_sim documentation can be built using the sphinx package. First, install the requirements:

```bash
pip install .[docs]
```

Next, the docs can be built to the `docs_build` directory:

```bash
sphinx-build docs docs_build
```

And launched with:

```bash
python -m http.server -d docs_build
```

To locally generate the tutorials, they must be copied to the docs folder,
converted to `.ipynb` files, and executed.
```bash
cp -r examples/tutorials docs/
jupytext --set-formats "py:percent,ipynb" docs/tutorials/*.py
jupytext --set-kernel python3 docs/tutorials/*.py
jupytext --to notebook --execute docs/tutorials/*.py
rm docs/tutorials/*.py
```

Documentation structure based on Alex Ganose (@utf) exceptional
[atomate2](https://materialsproject.github.io/atomate2/) package.

## Adding new models

We welcome the addition of new models to `torch_sim`. We want
easy batched simulations to be available to the whole community
of MLIP developers and users.

0. Open a PR or an issue to get feedback. We are happy to help,
even if you haven't finished your implementation yet.

1. Create a new model file in `torch_sim/models`. It should inherit
from `torch_sim.models.interface.ModelInterface` and `torch.nn.module`.

2. Write a test that runs `torch_sim.models.interface.validate_model_outputs`
on the model. This ensures the model adheres to the correct input and output formats.

3. Update test.yml to include proper installation and
testing of the relevant model.

4. Update .github/workflows/conf.py to include model in
autodoc_mock_imports = ['mace', 'fairchem']

[optional]

5. Write a tutorial or example showing off your model.

6. Update the .github/workflows/docs.yml to ensure your model
is being correctly included in the documentation.
