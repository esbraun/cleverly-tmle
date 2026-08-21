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
- Documentation examples are not statistical evidence. Cover behavior in the ordinary fast
  unit/integration/e2e tier or the named slow statistical tier. Reading the rendered prose is a
  manual act with no workflow behind it. What *is* automated stays in the fast tier:
  `tests/unit/test_documentation_links.py` resolves every link, including links naming a path in
  this repository; `tests/unit/test_documentation_examples.py` compiles every `python` fence — so a
  fence is still a promise that the block is Python, and prose belongs in a `text` fence; and
  `tests/unit/test_documentation_runtime.py` *executes* the registered reader-facing documents,
  asserting only that nothing raises. A reader-facing guide added under `docs/examples/`,
  `docs/getting-started/` or `docs/user-guide/` must be registered there or explicitly excluded.
- Ruff and mypy are pinned once in `pyproject.toml`'s `dev` extra and resolved by `uv.lock`.
  Nox and CI install that extra instead of restating versions. `dev` resolves to `cleverly[all]`
  plus tooling, so an optional extra kept out of *both* is installed by no session and its tests
  can only skip — and a skipped correctness check reads exactly like a passing one. Put a new
  extra in `dev`, or give it a dedicated job the way `bench`/`numba` has one. Ruff *formats* the
  Python examples in Markdown, so run it over the whole tree — but its linter does not read
  Markdown at all, and the formatter skips any block it cannot parse, so neither one sees a
  syntax error in an example.
- Run the smallest relevant test while iterating, then use the current validation commands in
  `README.md` or the matching nox session before handing off a change.
- The fast tier is the default handoff gate. Run a slow test only when all three conditions hold:
  it executes a changed runtime path, its assertion can observe the possible effect, and the
  evidence depends on repeated sampling, a large sample, or many expensive flexible fits that the
  fast tier cannot supply. Slow claims currently include coverage, type-I error, root-n and
  large-sample consistency, comparative variance, and flexible-learner bias. Inspect the relevant
  slow test before running it; file location or a broad label such as "estimator change" is not
  enough.
  Changes to an estimand, influence curve, variance or clustering calculation, targeting or
  cross-fitting behavior, nuisance predictions, randomization, weights, bounds, statistical DGP,
  or slow-study machinery normally satisfy this test. Documentation, formatting, type-only work,
  messages, serialization or presentation that preserves fitted arrays, and validation that exits
  before engine construction normally do not. For example, changing a configuration declaration
  from silent omission to a pre-construction refusal cannot affect a slow study's fitted sampling
  distribution and does not justify running it.
- When only one statistical family is reachable, run its named slow study rather than unrelated
  studies. Run the complete `pytest -m slow -q` tier when a shared estimation or inference path can
  affect several families, or when the applicable acceptance gate explicitly requires the whole
  tier. If no slow test can execute the changed path or observe its result, do not run slow tests;
  state that path analysis in the handoff. This is an evidence decision, not a time-budget waiver.
- **Never run fast and slow tests at the same time.** Both size themselves from
  `tests.parallel.available_cores()` and each expects the machine, so concurrent tiers oversubscribe
  every core and take longer than running sequentially. The complete slow tier takes about an hour;
  finish the fast tier before starting any relevant slow study.
- Run the relevant checks locally before a change is handed off, then treat the GitHub
  Actions jobs as the final merge signal. CI complements the local validation record; it does not
  replace the smallest relevant check while iterating or justify handing off a change known to
  fail locally.
