# Working on cleverly

`cleverly` is alpha software under heavy development. Treat the current architecture as
provisional: inspect the code and tests before changing it, and do not preserve an implementation
shape solely because this file once described it.

This file contains only repository-specific working agreements. Use `README.md` for setup,
development commands, and the public overview. `docs/README.md` routes technical documentation;
`docs/evidence.md` records the test-enforced correctness evidence. Current behavior is determined
by code and tests, not historical plans or investigation notes.

Cross-module constraints that are not derivable from one implementation live in
`docs/architecture-invariants.md`.

## Scientific changes

- Begin with the identified parameter and its paper or derivation. A reference implementation such
  as `tmle3` is useful for understanding an algorithm and localizing discrepancies, but agreement
  with it is not sufficient acceptance evidence.
- Validate mathematical changes independently. Prefer exact identities, finite-support laws,
  Gateaux derivatives, and remainder checks over output parity or a stochastic tolerance.
- Exact-law checks are blind to terms that vanish at the truth. When a sign, mask, guard, or
  counterfactual block can disappear, add a nonzero witness or a deliberate-mutation control that
  fails when that component is wrong.
- A registered estimand must remain covered in both directions by the oracle and evidence gates in
  `tests/unit/test_registry.py` and `docs/evidence.md`.
- Refuse unsupported but well-posed compositions explicitly and explain what is missing. Do not
  silently return a convenient approximation to a different estimand.

## Tests and tooling

- Use `tests.conftest.FAST_KWARGS` and parametric learners for estimator tests unless flexible
  learning is the behavior under test. Mark statistical studies that require many fits `slow`.
- Fenced documentation examples run in the nightly `docs` tier in reading order. Catalogue blocks
  use the tested `<!-- catalogue: reason -->` marker; see `tests/e2e/test_doc_snippets.py`.
- Ruff and mypy are pinned separately in `pyproject.toml`, `noxfile.py`, and CI. Update all copies
  together. Ruff checks Python examples in Markdown, so run it over the whole tree.
- Run the smallest relevant test while iterating, then use the current validation commands in
  `README.md` or the matching nox session before handing off a change.
