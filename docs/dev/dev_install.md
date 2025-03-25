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

To install torch_sim from source, clone the repository from [github](https://github.com/Radical-AI/torch_sim)

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