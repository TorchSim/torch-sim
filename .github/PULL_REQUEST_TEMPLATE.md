## Summary

* Feature 1
* Fix 1

## Checklist

Before a pull request can be merged, the following items must be checked:

* [ ] Doc strings have been added in the [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google).
* [ ] Run [ruff](https://beta.ruff.rs/docs/rules/#pydocstyle-d) on your code.
* [ ] Tests have been added for any new functionality or bug fixes.
* [ ] I have read and agree to the Contributor's Certification in [CONTRIBUTING.md](CONTRIBUTING.md).

We highly recommended installing the pre-commit hooks running in CI locally to speedup the development process. Simply run `pip install pre-commit && pre-commit install` to install the hooks which will check your code before each commit.
