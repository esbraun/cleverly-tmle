# Working on cleverly

## Do not run the slow tests in the Claude Code cloud sandbox

The `slow` marker guards the statistical validation tier — coverage studies, root-n
consistency, type I error. Those tests fit thousands of models by design.

**Never run `pytest -m slow` (or an unmarked selection that includes those tests) in the
Claude Code cloud sandbox.** It is a small shared container (4 cores), and these runs:

- take tens of minutes and starve everything else on the box, which silently inflates
  every other timing measurement taken while they run;
- spawn joblib/loky worker processes that **survive a killed pytest**. Orphaned workers
  keep burning 100% CPU indefinitely and make later benchmarks meaningless. This has
  already happened once here and produced a 300x bogus timing.

The slow tier belongs in the nightly GitHub Actions workflow (`.github/workflows/nightly.yml`)
or on a developer machine with cores to spare.

```bash
pytest -m "not slow" -q        # the only tier to run in the sandbox
```

If you ever do kill a test run, clean up after it:

```bash
pkill -f pytest; sleep 2; pkill -f "joblib.externals.loky"
```

Then check `/proc/loadavg` before trusting any timing.

## Keep tests fast unless slowness is technically necessary

The fast tier is meant to stay in the low minutes. When adding a test, spend runtime only
where the claim genuinely requires it:

- **Default to `library="glm"`** for nuisance learners (`tests.conftest.fast_tmle` already
  does). The `"fast"`, `"default"` and `"rich"` presets include boosting and cost roughly
  10x, 20x and 60x as much per fit. Use them only when the test is *about* flexible
  learning — for example the double-robustness comparison — not when it merely needs a
  fitted model.
- **Use the smallest `n` and fewest replications that resolve the claim.** A coverage
  assertion with a ±0.05 window needs ~120 replications, not 400. State the reasoning in
  a comment so the budget is not later "optimised" away or inflated.
- **Prefer an exact check over a statistical one.** Verifying that targeting solves the
  score equation, that `IC_ate == IC_ey1 - IC_ey0`, or that the Newton solver matches a
  grid search costs milliseconds and fails deterministically. A simulation study should be
  the last resort, not the first.
- **Never assert coverage on a single fit.** A 95% interval misses 5% of the time by
  construction, so such a test is a coin flip that fails on a bad seed. Average over
  replications and compare against the Monte Carlo standard error.
- Scope expensive fixtures with `scope="module"` or `scope="class"` so a fit is shared
  across the tests that examine it.

## Layout

| directory | contents |
| --- | --- |
| `src/cleverly/data` | `CausalData` container and input validation |
| `src/cleverly/learners` | cross-fitting, screening, `SuperLearner`, thread limits |
| `src/cleverly/fluctuation` | clever covariates and the targeting step |
| `src/cleverly/estimators` | nuisance orchestration, `TMLE`, result objects |
| `src/cleverly/inference` | influence curves, clustering, bootstrap, simultaneous bands |
| `src/cleverly/sensitivity` | positivity, omitted-variable bias, E-values, MNAR tilt |
| `src/cleverly/validation` | score check, nuisance diagnostics, refutation, simulation |
| `src/cleverly/datasets` | synthetic processes with exactly known truth |

## Conventions

- **Dataframes**: everything user-facing goes through narwhals; results are returned in
  the backend the caller passed in. Never branch on pandas vs polars.
- **New estimands**: add the clever covariate to `fluctuation/submodel.py` and the
  influence curve to `inference/influence.py`. The variance, bands, delta method and score
  diagnostic then work without further changes.
- **Nuisance reuse**: `TMLE.retarget` re-runs only the targeting step against cached
  nuisance fits. Sensitivity analyses must use it rather than refitting.
- **Thread limits**: nuisance fits run single-threaded by default
  (`cleverly.learners.set_thread_limit`) so parallelism happens across folds and
  candidates. Do not add native threading inside a fit.

## Before committing

```bash
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q
```
