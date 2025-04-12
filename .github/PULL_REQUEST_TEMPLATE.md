## Summary

Include a summary of major changes in bullet points:

* Feature 1
* Fix 1

## Checklist

Work-in-progress pull requests are encouraged, but please enable the draft status on your PR.

Before a pull request can be merged, the following items must be checked:

* [ ] Doc strings have been added in the [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google).
  Run [ruff](https://beta.ruff.rs/docs/rules/#pydocstyle-d) on your code.
* [ ] Tests have been added for any new functionality or bug fixes.
* [ ] All linting and tests pass.

Note that the CI system will run all the above checks. But it will be much more
efficient if you already fix most errors prior to submitting the PR. It is highly
recommended that you use the pre-commit hook provided in the repository. Simply run
`pre-commit install` and a check will be run prior to allowing commits.
