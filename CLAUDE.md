# Working on cleverly

## Do not run the slow tests in the Claude Code cloud sandbox

The `slow` marker guards the statistical validation tier — coverage studies, root-n
consistency, type I error. Those tests fit thousands of models by design.

**Never run `pytest -m slow` (or an unmarked selection that includes those tests) in the
Claude Code cloud sandbox.** It is a small shared container (4 cores), and these runs:

- take tens of minutes and starve everything else on the box, which silently inflates
  every other timing measurement taken while they run;
- spawn joblib/loky worker processes that **survive a `SIGKILL`ed pytest**. Orphaned
  workers keep burning 100% CPU indefinitely and make later benchmarks meaningless. This
  has already happened once here and produced a 300x bogus timing.

The slow tier belongs in the nightly GitHub Actions workflow (`.github/workflows/nightly.yml`)
or on a developer machine with cores to spare.

```bash
pytest -m "not slow" -q        # the only tier to run in the sandbox
```

**Interrupt a test run with `Ctrl-C`, not `kill -9`.** Which signal you use is the whole
difference: joblib registers an `atexit` handler that shuts its worker pool down on every
path where Python still runs, so a `SIGINT` — and an ordinary failure, and a clean exit —
leaves nothing behind. Measured here at zero survivors in each case. Only `SIGKILL`
orphans them, because nothing in-process runs: a killed run leaves `LokyProcess` workers
at ~75% CPU each, reparented to init, still going a minute later.

No fixture or `n_jobs` setting can prevent that, so do not go looking for a code fix —
the fix is the signal. If a run *was* `SIGKILL`ed, clean up after it:

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
  across the tests that examine it. Then *use* the shared fit: a parametrized case that
  refits a configuration the fixture already holds is the commonest waste here, and a
  result object usually carries more than the one case reads — a CDE fit carries every
  level of the intermediate, so parametrizing over the level and refitting inside each
  case does the same work twice.
- **Spell the fold counts out**, or build on `tests.conftest.FAST_KWARGS`. Writing
  `outcome_learner="glm"` by hand and leaving `n_folds` off silently takes the `TMLE`
  defaults of `n_folds=10, learner_folds=5` — twice the fast tier's budget, and the `glm`
  in the constructor makes it read as though the fast-tier rules were being followed.

**The `n_jobs=2` on the simulation studies is deliberate — leave it.** It looks like
oversubscription under CI's `pytest -n auto`, and it is not: xdist parallelises *between*
tests and cannot split one, so the handful of 6–13s `CoverageStudy` tests are the critical
path and the inner pool halves each of them. Measured over three paired runs on four
cores, dropping it to `n_jobs=1` made the e2e tier **35% slower** (75.7s → 102.3s), with
three xdist workers idling while the longest test ran twice as long. Nesting here is
load-balancing the tail, not contending for cores.

## Layout

| directory | contents |
| --- | --- |
| `src/cleverly/data` | `CausalData` container and input validation |
| `src/cleverly/learners` | cross-fitting, screening, `SuperLearner`, thread limits |
| `src/cleverly/interventions` | regimes: static arms, dynamic rules, stochastic assignments; shifts of a continuous dose |
| `src/cleverly/msm.py` | the working model a fit projects the counterfactual means onto |
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
  method and score diagnostic then work without further changes. Every registered target
  also needs a longhand branch in one of the oracle laws — `tests/discrete_law.py` for the
  arm- and regime-indexed estimands, `tests/discrete_law_shift.py` for the shift-indexed
  ones — and `tests/unit/test_registry.py` checks the two cover each other in *both*
  directions, so a target with no oracle and an oracle branch with no target both fail.
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
- **A regime is a density over arms.** `interventions=` makes the parameter axis the
  *regime* rather than the arm: an `(n, K, R)` density, keyed by code with labels carried
  separately exactly as arms are. The `regime` fluctuation is a separate group rather than
  a generalised `mean`, because the two axes come apart — a fluctuation still updates
  `Qbar` at every arm, but the score equations are one per regime — and because the arm
  path is a regression surface that must not move. The evaluated densities live on
  `NuisanceEstimates`, so everything reached through `retarget` targets the declared
  regimes without the caller's rules being callable again.
- **A shift moves the dose the unit received.** `shifts=` is a third parameter axis, not a
  kind of regime: a regime assigns an arm from `W` alone, an MTP reads `A`. `Target`
  declares which of `arm` / `regime` / `shift` / `ipsi` / `msm` it belongs to via
  `parameter_axis`, and the five partition the registry — a fit reporting parameters from two
  of them would be putting two score equations under one heading. `shifts=` also declares the treatment
  continuous, since a dose has no arms to index by. The mechanism is a `ConditionalDensity`
  rather than a `Propensity`, and it lives on `NuisanceEstimates` beside `regimes` for the
  same reason; the `ShiftSet` is built *inside* `fit_nuisances`, because `g(A|W)` and
  `g(A-δ|W)` must come from one out-of-fold model and evaluating them there makes that
  structural rather than an invariant to maintain. A shift's mean equals that of the
  stochastic regime at the induced density; its influence curve does not, and
  `tests/unit/test_influence_gateaux_shift.py` keeps the negative control that fails if
  someone delegates one to the other.
- **An incremental intervention tilts the mechanism, so the estimator targets it.**
  `incremental=` multiplies the odds of treatment by `δ` (Kennedy 2019). Because `q_δ` is built
  out of `g`, three things differ from every other axis and each is easy to get wrong.
  *No positivity assumption*: the covariate is `δ/D` and `1/D` with `D = δg + 1 − g`, bounded by
  `max(δ, 1/δ)` however small `g` is — so `g_bounds=` is **refused**, since `g` is inside the
  estimand and truncating it would move `Ψ(δ)` rather than regularise a denominator, and the
  `ipsi` builder ignores its `propensity` argument outright to keep that structural.
  *Not doubly robust* — the only estimand here that is not: every term of the remainder carries
  `(ĝ − g₀)`, so a consistent mechanism is required and a consistent `Q̄` cannot substitute.
  *Two score equations*: the `∂m/∂g` term lives in the mechanism's tangent space, so `g` gets a
  logistic submodel of its own and the two alternate (`fluctuation/mechanism.py`,
  `targeting.solve_with_mechanism`). The alternation is coordinate ascent on one joint
  likelihood, so it converges — but *linearly*, at rates measured between 0.15 and 0.52 per
  round, which is why the stall threshold sits at 0.95 and the outer cap at 50. The targeted `g`
  lives on `Fluctuation.mechanism`, never on `NuisanceEstimates.propensity`, which stays the
  initial cross-fitted mechanism exactly as `outcome` stays the initial regression. `ψ(δ=1)`
  equals `mean(Y)` row by row whatever the nuisances are; keep that test, it is the canary that
  catches an alternation exiting with one equation open.
- **A working model summarises the arms; it does not replace them.** `msm=` is a fourth
  parameter axis and the one that is *not* about what "counterfactual" means: the
  counterfactuals are still the arms and the fluctuation still updates `Qbar` at every one
  of them. What moves is the report — `p` score equations, one per term, in place of `K`,
  one per arm — which is enough to need its own group and its own axis. `beta` is a
  *projection* under a known weight `h(a, V)`, so it is well defined whether or not the
  working model is right, and the interval is not secretly one for a misspecified
  regression; say so wherever a reader could assume otherwise. Two consequences are easy
  to get wrong. The projection is solved on the **raw** outcome scale, unlike every other
  estimand, because a coefficient vector has no single `Scale` to map back with — use
  `TargetContext.finish_unscaled`, not `finish`. And a *saturated* working model must
  reproduce the per-arm report exactly; `tests/unit/test_msm_submodel.py` and
  `tests/e2e/test_msm.py` are what enforce that, at the covariate and at the estimate.
- **Which truncation bound a group gets is a statement about its covariate.**
  `utils.bounds.CONDITIONAL_GROUPS` names the groups whose clever covariate is a propensity
  *odds* (`att`, `atc`) and so needs the tighter bound; everything else divides by `g` once
  and takes the ordinary one. Do not reintroduce the `group == "mean"` test this replaced —
  it was written when those were the only three groups, and every group added since
  inherited the ATT bound silently, as would any group a caller registers.
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
