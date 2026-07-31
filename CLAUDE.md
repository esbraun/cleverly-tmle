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
| `src/cleverly/longitudinal` | the time-ordered container, regimens, sequential regression, `LTMLE` |
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
  **`delta=` is supported and `intermediate=`, multi-arm and C-TMLE are not**, and the
  asymmetry is not caution — it is that only the first has an oracle law. `π(A,W)` divides the
  *outcome* half of the covariate and Kennedy's `∂m/∂g` term is untouched, because `q_δ` is a
  functional of `P(A|W)` and both `A` and `W` are recorded whatever happens to `Y`;
  `tests/discrete_law_mar.py` carries the `ey_ipsi`/`ate_ipsi` branches and
  `tests/unit/test_influence_gateaux_ipsi_mar.py` checks the composition against a complex-step
  Gateaux derivative rather than arguing it. Two things change with it and both are easy to get
  backwards. The guarantee *tightens* to "`ĝ` right **and** one of `π̂`, `Q̄` right" — the
  `(ĝ − g₀)²` term is π-free and survives everything else — and the two mechanisms cannot trade
  off the way they do on the arm path, since `ĝ` is in the estimand. And the `ψ(δ=1)` canary
  above holds **absent `delta=`** only: with it, `ψ(1)` is the MAR-identified `E[Y]`, the curve
  is `Δ/π·(Y − Q̄(A,W)) + Q̄(A,W) − ψ`, and the complete-case mean is the wrong answer.
  Before refusing a further combination here, check whether a law in `tests/` already covers
  it — this refusal was written on the belief that none did.
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
- **A regimen is a plan over nodes, and it is not a `Target`.** `cleverly.longitudinal` is a
  separate estimator with its own container and result object, not a fifth parameter axis:
  a `Target` is indexed by an arm, a regime, a shift, a tilt or an MSM coefficient, and the
  point-treatment pipeline is built around one `Qbar(a, W)` and one `g(a | W)`. What it
  *does* reuse is everything below the estimand — `cross_fit_predictions`, `Submodel` and
  `solve_fluctuation` (with `group="sequential"`, which needs no `register_submodel` since
  `TargetGroup` is a plain `str`), `make_estimate`, `delta_method`, `influence_covariance`
  — so do not fork any of those. Three things are easy to get wrong here and each has a
  test that fails when it is. The recursion's masks must line up: `at_risk(t+1)` **is**
  `following(t)`, the set the previous node's regression was fitted on, and if the two come
  apart the pseudo-outcome is regressed on a population it was not computed for. The update
  is applied at the *counterfactual* covariate (`1/∏g`, no indicator) while the score uses
  the observed one (`1{followed}/∏g`), exactly as the arm path takes `Q*(1, W)` from
  `submodel.arms[1.0]` — read `fluctuation.targeted.arms[a]`, never `.observed`, or the
  nodes after the first stop being updated at all. And the recursion carries the *targeted*
  prediction forward, not the initial one. `tests/discrete_law_longitudinal.py` catches all
  three: on a law the sample realises exactly, a saturated learner makes every score zero,
  so `epsilon` must come back zero and the reported curve must equal the complex-step
  Gateaux derivative to `1e-14` **absolute, with `rtol=0`** — pass it explicitly as every
  sibling module does, since these curves reach order 20 and numpy's default relative
  tolerance would loosen the check to ~`1e-6` while still reading as exact.
  Two further things it does *not* reuse, and both are refusals rather than gaps.
  `res.sensitivity` and the targeted bootstrap need a refit against re-truncated or
  resampled nuisances, and `g_bounds` enters the *pseudo-outcome* of every earlier node
  through the recursion — so there is no `retarget` that re-solves the fluctuation alone,
  and `LongitudinalData` has no `subset`. Both say so by name; do not "fix" either by
  wiring the point-treatment path to a longitudinal result. And `LTMLE`'s `alpha` /
  `alpha_sig` mean exactly what they mean on `TMLE` (shrink, then significance level) —
  they were once the other way round here, which made `LTMLE(alpha=0.9995)` a silent
  0.05 %-level interval.
- **A dynamic rule is a regimen, not a fifth axis, and three invariants keep it that way.**
  Any node of a plan may be a rule `d_t(H_t)` instead of an arm. Everything downstream reads
  one `(n, T)` **assignment matrix**, which a static regimen fills by `np.broadcast_to` — a
  view, producing the same float64 the old scalar path did, which is why a static fit is
  bit-for-bit unchanged and why there is one code path rather than two. Do not reintroduce a
  scalar `regimen.at(time)` read anywhere in `sequential.py`.
  *One*: the matrix is built **once**, in `LTMLE.fit` via `resolve_plans`, and `fit_mechanism`
  and `fit_regimen` see `Plan` objects that hold arms, never callables. Evaluating lazily
  would call each user rule a further `T` times per regimen and — if a rule is not
  deterministic — let the follower masks disagree with the designs the mechanism was
  evaluated at, so the fit would answer for no single regimen at all.
  *Two*: a rule sees `history_frame(t)` — `[W, L_1, ..., L_t]` — and nothing else, enforced by
  what the frame contains rather than by a check, exactly as `interventions/base._covariate_frame`
  does at one node. Passing the earlier `A_s` would let a rule read the treatment of a
  *deviator*, which is a different object from the one it assigns. Off the recorded set the
  arm is coerced to zero rather than validated, because such a row is masked out of
  everything and the only way it could still matter is a `nan` reaching a learner.
  *Three*: the outcome regression's design stays `covariate_history(t)` with **no** treatment
  columns. The old justification — "a follower's past treatment is constant" — is false under
  a rule; the true one is that among the followers `A_s = d_s(W, L_1, ..., L_s)` is a
  deterministic function of columns the design already carries, since that is the frame the
  rule was handed. **No fit here comes out differently if this is changed**, and that was
  measured, not assumed: on the exact law the saturated learner partitions by distinct design
  row, so a column that is a function of the others is invisible; and under `glm` the natural
  comparison — a rule that ignores the history against the constant plan it equals — adds a
  *constant* column, which `StandardScaler` zeroes, to both sides at once. So the statistical
  claim is still an argument and not a test: change it only with an argument, and do not add a
  test claiming to guard *that* without checking the test fails. What is pinned is the **call
  site** — `tests/unit/test_sequential_design.py` asserts the matrix `fit_regimen` passes is
  `covariate_history(t)` bit for bit, and was mutated to `history_design(...)` and seen to
  fail. A structural pin is the right instrument here precisely because both designs are
  consistent: it catches the silent edit without pretending to adjudicate the statistics.
  The oracle for all of this is `tests/discrete_law_longitudinal.py`, where `W` and `L2` are
  binary so a rule is a lookup over cells: `REGIMEN_ARMS` states the plans for the oracle and
  `REGIMEN_SPEC` states the same plans as callables for the estimator, deliberately in two
  representations so a slip is a wrong number rather than one that cancels on both sides. Keep
  `functional` selecting arms by **cell index** — a rule written as an indicator *of the
  probabilities* makes the complex step come back real, silently.
- **A survival outcome is a population, not a parameter axis.** `outcome=[...]` — a sequence
  rather than a name — puts an absorbing `Y_t` into the ordering and makes the report the
  cumulative risk at every horizon. It is *not* a fifth axis beside `arm`/`regime`/`shift`/
  `ipsi`/`msm`: the counterfactual is still a regimen, and `LTMLE` is still not a `Target`.
  What changes is which rows each node's regression is fitted on, and there are three ways to
  get that backwards, each with a test that fails when you do.
  *The event node is one earlier than the censoring node.* `following(t)` reads
  `event_free_through(t - 1)` while `uncensored_through` reads `t`, so `at_risk(t+1) ==
  following(t) & event-free at t` — the closure identity generalised, not broken. A unit that
  has the event at `t` **is** in node `t`'s regression, because it is the observation that the
  event happened. "Tidying" that `t - 1` to a `t` reads like a correction and turns 26 of
  `tests/unit/test_influence_gateaux_survival.py`'s 30 tests red; do not.
  *The mechanism's fit mask is event-aware too.* A unit that has already had the event has no
  `A_t`, and `history_design` fills a missing arm with zero — so leaving it in `fit_mechanism`'s
  mask trains it as an untreated observation and biases `g`. That mutation is **silent**: every
  point estimate stays green, since with an exact initial fit `epsilon` is zero and no error in
  `g` can move `psi`. Only the Gateaux comparison catches it.
  *The event contributes no factor to the clever covariate.* Being event-free is `H_t`-measurable
  — part of the history, not an intervened node — so it enters the *indicator* of `h_t` and never
  the denominator. `g_bounds` and the per-factor truncation mean exactly what they meant.
  Two further things. Each horizon is its own backward pass (`T(T+1)/2` regressions per regimen,
  mechanism fitted once and shared); `horizons=` is the lever, and is refused on an end-of-study
  fit rather than ignored. And `fits` is keyed by `(label, horizon)` **composed forward** in
  `LTMLE.fit`, never parsed back out of a report name — `diagnostics()` and `summary()` read
  `fit.regimen.label` and `fit.horizon`, which is why the `regimen` column is still a regimen.
  The report is the **risk**; `res.curve(scale="survival")` is a derived view, and it branches on
  `ParameterEstimate.scale` because `S = 1 - F` is right for a level and *wrong* for a contrast,
  where the map is `-(F_a - F_b)`. A fit whose event can only happen at the last node reproduces
  the end-of-study fit bit for bit — `psi`, the whole curve, every `epsilon` — which is the pin
  that says this is a generalisation and not a second estimator. The oracle is
  `tests/discrete_law_survival.py`, a **new** law rather than a wider `discrete_law_longitudinal`,
  because that one has to go on proving the end-of-study derivation unchanged.
- **A competing cause is a population, not a denominator.** `outcome={cause: [...]}` — a
  *mapping* rather than a list — declares more than one absorbing state per node and makes the
  report the cause-specific cumulative incidence, `cif_regimen[always, relapse @ t=2]`. A
  mapping with one cause is competing risks **by declaration**; a fit reports what it said it
  would, not what its sample happened to contain. The estimand is the incidence with the
  competing causes *left alone*, and that choice is what makes almost all of this a
  generalisation: a competing event is `H_t`-measurable, so it enters the clever covariate's
  indicator and never its denominator, and positivity is still about the same `2T` factors.
  `eliminate=` — intervening on the competing events — is refused by name, because it is a
  different parameter needing a further factor per node and its own identification, not a
  setting on this one.
  One line differs from the single-event recursion and it is the only thing that does:
  `Z_t = 1{cause j at t} + 1{no event at t}·Q̄*_{t+1}`, a **cause-specific numerator against an
  all-cause survival factor**. Write `1 - failed` there — the cause's own survival — and you
  are wrong by exactly the mass that left through the other causes; it was applied and takes 21
  of `tests/unit/test_influence_gateaux_competing.py`'s 130 tests, every one at `t = 2`, since
  the first horizon has no survival factor to get wrong. So `event` stays the **all-cause**
  matrix that `at_risk`, `following` and `fit_mechanism`'s mask read, `cause_event` sits beside
  it, and only the pseudo-outcome is cause-specific — which is also why the causes share every
  nuisance fit and `J` causes cost `J` backward passes and one mechanism.
  Two consequences to leave alone. `Σ_j F_j(k) + S(k) = 1` holds of the *parameters* and not of
  the estimates, since each cause is its own backward pass; `incidence_total()` reports the sum
  and its excess over one, exactly as `simplex_deviation` reports a multi-arm row's departure
  rather than rescaling it — do not renormalise. And `curve()` reads a `name -> (regimen, cause,
  horizon)` index **composed forward** in `_estimates`; with two indexes inside one pair of
  brackets there is no split that is right in general, and a regimen called `"a, b"` would be
  filed under one that does not exist rather than failing. The oracle is
  `tests/discrete_law_competing.py`, a **third** law rather than a wider survival one, since
  that has to go on proving the single-event derivation unchanged.
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
  `LTMLE` is the counter-example, and the contrast is the point: it answers a parameter
  the point-treatment pipeline cannot express, so it needs its own container and result
  object — and then every subsystem keyed to `TMLEResult` has to be either reused
  deliberately (`make_estimate`, `delta_method`, `influence_covariance`,
  `simultaneous_bands`, `CoverageStudy`) or refused by name. An `AttributeError` from a
  subsystem that was never taught about a new result type is not a refusal; nor is a
  replicate loop's blanket `except Exception` turning a missing method into "the fit is
  too unstable to bootstrap". Prefer overriding `_nuisances` wherever the parameter
  allows it, precisely so this does not arise.
- **Thread limits**: nuisance fits run single-threaded by default
  (`cleverly.learners.set_thread_limit`) so parallelism happens across folds and
  candidates. Do not add native threading inside a fit.

## Before committing

```bash
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q
```

**The ruff version is pinned exactly**, in `pyproject.toml`'s `dev` extra and in
`.github/workflows/ci.yml`, and the two must move together. CI used to run `ruff@latest`,
which meant a formatter release could turn the lint job red with no commit to blame — and
did: 0.16 began formatting the Python blocks inside Markdown, so `README.md` failed there
while a local 0.15 said everything was clean. When bumping the pin, run `ruff format .`
over the *whole* tree rather than `src` and `tests`, because the README is now formatted
code too, and expect hand-aligned trailing comments in its examples to be collapsed.
