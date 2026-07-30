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
- **New estimands**: construct a `Target` and call `targets.register`. If it needs a score
  equation no existing group solves, write the clever-covariate builder and call
  `fluctuation.register_submodel` first — `register` refuses a target whose group has no
  builder. The influence curve goes in `inference/influence.py`; the variance, bands, delta
  method and score diagnostic then work without further changes.
- **Counterfactual arms**: `Submodel`, `InitialFit`, `Propensity` and
  `counterfactual_means` key their per-arm arrays by treatment level (`arms[1.0]`), not by
  `at_one` / `at_zero` fields, and `arm_columns` says which design column targets which arm.
  Use `map_arms` rather than writing a triple, so a helper does not silently assume there
  are two arms. A treatment may have up to 20 levels; `data.arm_codes` is the internal
  coding and `data.arm_label` maps back to what the user passed, which is what every
  reported name, table and error message must use.
- **The binary path is a regression surface.** Multi-arm support was built so that a
  two-armed fit stays bit-for-bit identical, and several choices exist only for that:
  `predict_probabilities` takes the complement rather than reading `predict_proba`'s zero
  column, `Propensity.bounded` clips `g1` and complements it rather than clipping both
  columns, and the `K-1` indicator design collapses to the old single column. Before
  changing any of them, check the claim still holds — the fixtures in `tests/unit` and the
  oracle laws are what enforce it.
- **Binary-only by declaration, not by accident.** A target that names an arm declares
  `requires_binary_treatment=True`; C-TMLE, the omitted-variable bound and the MNAR tilt
  raise on a multi-arm fit. Prefer refusing with a message that says what the derivation
  would need over quietly reporting arms 0 and 1.
- **Nuisance reuse**: `TMLE.retarget` re-runs only the targeting step against cached
  nuisance fits. Sensitivity analyses must use it rather than refitting.
- **New estimator variants**: a variant that only changes *which* nuisance estimate is
  targeted should override `TMLE._nuisances`, return a `dataclasses.replace`d
  `NuisanceEstimates` plus its diagnostics, and let the inherited `retarget` do the rest.
  `CTMLE` (`estimators/ctmle.py`) is the worked example: because it swaps one array,
  every influence curve, sensitivity analysis and validation diagnostic keeps working
  untouched, and the bootstrap repeats the selection for free.
- **Thread limits**: nuisance fits run single-threaded by default
  (`cleverly.learners.set_thread_limit`) so parallelism happens across folds and
  candidates. Do not add native threading inside a fit.

## Before committing

```bash
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q
```
