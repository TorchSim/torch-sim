# Adding New Models

We welcome new model integrations into the TorchSim ecosystem. Our preferred pattern is that developers implement the bulk of the TorchSim integration in their own model repository or package, and then TorchSim provides a thin wrapper or re-export for convenience. This keeps the model-specific logic close to the model itself while still making the integration easy to discover from `torch_sim.models`.

If you want TorchSim to expose your model, the recommended steps are:

1. Open a PR or issue early. We are happy to review incomplete work and discuss the best integration shape.
2. Prefer implementing the real integration in the upstream model package. The TorchSim-side file in `torch_sim/models` should ideally be a small wrapper or re-export, similar to other external integrations.
3. If some logic must live in TorchSim, keep it minimal and well-contained. The implementation should still follow the `torch_sim.models.interface.ModelInterface` contract.
4. Add model tests using `make_validate_model_outputs_test` and, where appropriate, `make_model_calculator_consistency_test`. See the existing model tests for examples.
5. Update `.github/workflows/test.yml` and `.github/workflows/model-tests.json` so the model installs and tests correctly in CI.
6. Re-export the model from `torch_sim.models.__init__.py` using the existing `try`/`except ImportError` pattern.
7. Update `docs/conf.py` so the docs can build without requiring the optional model dependency.

Optional follow-up work:

1. Add a tutorial or example showing how to use the model with TorchSim.
2. Update `.github/workflows/docs.yml` if the documentation build needs to exercise that integration.

We are also happy for developers to keep the entire integration in their own repository without adding a wrapper in TorchSim. But when TorchSim does expose a model, we prefer the bulk of the logic to live upstream and the TorchSim wrapper to stay thin.
