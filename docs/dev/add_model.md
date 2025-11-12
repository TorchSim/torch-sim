# Adding New Models

## How to add a new model to TorchSim

We welcome the addition of new models to `torch_sim`. We want
easy batched simulations to be available to the whole community
of MLIP developers and users.

1. Open a PR or an issue to get feedback. We are happy to take a look,
even if you haven't finished your implementation yet.

1. Create a new model file in `torch_sim/models`. It should inherit
from `torch_sim.models.interface.ModelInterface` and `torch.nn.module`.

1. Add `torch_sim.models.tests.make_validate_model_outputs_test` and
`torch_sim.models.tests.make_model_calculator_consistency_test` as
models tests. See any of the other model tests for examples.

1. Update `test.yml` to include proper installation and
testing of the relevant model.

1. Pull the model import up to `torch_sim.models` by adding import to
`torch_sim.models.__init__.py` in try except clause.

1. Update `docs/conf.py` to include model in `autodoc_mock_imports = [...]`

## Optional

1. Write a tutorial or example showing off your model.

1. Update the `.github/workflows/docs.yml` to ensure your model
is being correctly included in the documentation.

We are also happy for developers to implement model interfaces in their
own codebases. Steps 1 & 2 should still be followed to ensure the model
implementation is compatible with the rest of TorchSim.
