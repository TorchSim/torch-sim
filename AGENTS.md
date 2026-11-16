## Testing

### What to test

- Test the units that the change touches, and treat everything else as a dependency.
- Use real dependencies. Mock one only when the test cannot use it, for example because it needs hardware the tests do not run on, or when it is slow and its own tests cover it.
- Mock at the function or class your code calls, not something inside it.
- If an existing test of the same code path missed the bug, find out why and strengthen that test instead of adding a separate regression test.
- Do not weaken, skip or delete existing tests to make them pass.
- Add a new test only for behavior that no existing test covers.

### How to check

- For a bug fix, write tests that fail without the fix and pass with it.
- Assert what the change is about, not an indirect effect of it.

### Avoiding duplication

- Move a mock or setup that several tests use into a fixture.
- If tests differ only in their inputs and expected values, merge them with `pytest.mark.parametrize`.

## Pull requests

- Use a Conventional Commits title.
- Follow the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Keep the description concise. In the summary, cover:
  - What: one or two sentences on what changed.
  - Why: the problem it solves. Link the issue, if any.
  - Reproduction: for a bug fix, a minimal script that reproduces the bug on main, unless the linked issue already has one.
  - Tests: which tests you added or changed, and what each one checks.

### Self-review

Before opening a PR, offer the contributor a review by a separate agent. If it is skipped, say so in the summary.

1. Commit your changes, then start a reviewer that does not share your context, such as a subagent or a new session.
2. Give the reviewer only the issue the PR addresses, if any, and these instructions: review the diff of this branch against the main branch of TorchSim/torch-sim, check it against `AGENTS.md`, and read and report without editing. Do not explain the change or why you made it.
3. Show the findings to the contributor, and fix the ones they agree with. One or two rounds are enough.
4. Check the self-review item in the PR template.
