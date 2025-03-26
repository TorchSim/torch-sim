# Adding New Models

## How to add a new model to torchsim

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
