# Working on cleverly

This file is durable guidance for maintainers and coding agents. `cleverly` is actively developed
and is not feature complete. Preserve its mathematical and API invariants while extending it; do
not treat current implementation choices as permanent when new evidence supports a better design.

## Start here

- `README.md` summarizes the public API and development commands.
- `docs/user-guide.md` is the capability reference.
- `docs/methodology.md` states each estimand, influence function, remainder, and validation
  strategy.
- `docs/drtmle.md` is the production contract for `DRTMLE`.
- `docs/roadmap.md` records current limitations, candidate features, and engineering decisions
  with explicit reopening conditions.
- `docs/benchmarks/README.md` explains the performance instruments and current evidence.

Historical investigation notes are not specifications. The archived DR-TMLE validation programme
is recoverable from the `drtmle-validation-archive-2026-08` tag; current behavior is specified and
tested on `main`.

## Development commands

```bash
uv venv
uv pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/cleverly
pytest -m "not slow and not docs" -q -n auto
```

The fast test tier is the default local and pull-request check. The `slow` marker covers coverage,
root-n consistency, type-I error, and other statistical studies that intentionally fit many
models. Run it through the nightly workflow or on a suitable development machine:

```bash
pytest -m slow -q -n auto
```

Interrupt parallel test runs normally so pytest and joblib can shut workers down. After a forced
process kill, confirm no worker pool remains before trusting test or benchmark timings.

Ruff and mypy are pinned in `pyproject.toml`, `noxfile.py`, and the CI workflow. Update every copy
together. Ruff formats Python examples in Markdown, so formatting checks always run over the whole
tree.

## Repository layout

| path | responsibility |
| --- | --- |
| `src/cleverly/data` | `CausalData`, treatment encoding, weights, and input validation |
| `src/cleverly/learners` | cross-fitting, screening, Super Learner, and thread limits |
| `src/cleverly/interventions` | deterministic and stochastic regimes, shifts, and incremental interventions |
| `src/cleverly/targets` | estimand declarations and registry |
| `src/cleverly/fluctuation` | clever covariates and targeting submodels |
| `src/cleverly/estimators` | nuisance orchestration, point-treatment estimators, and result objects |
| `src/cleverly/longitudinal` | longitudinal data, regimens, sequential regression, and `LTMLE` |
| `src/cleverly/inference` | influence curves, covariance, clustering, bootstrap, and simultaneous bands |
| `src/cleverly/sensitivity` | positivity, omitted-variable bias, E-values, and MNAR analysis |
| `src/cleverly/validation` | scores, nuisance diagnostics, refutation, and simulation |
| `src/cleverly/datasets` | synthetic data-generating processes with known truth |
| `tests/discrete_law*.py` | finite-support oracle laws independent of estimator internals |
| `benchmarks` | end-to-end and post-nuisance performance instruments |

## How to add a feature

1. **Name the parameter or behavior.** State its identification assumptions, estimand, influence
   function, remainder, and unsupported compositions before choosing an implementation shape.
2. **Choose the extension point.** Prefer a new registered target or intervention when the shared
   point-treatment pipeline can express the parameter. Prefer an estimator variant only when
   nuisance selection or targeting behavior changes. Use a separate pipeline when the data
   ordering or parameter cannot be represented by `TMLE`.
3. **Define public behavior.** Specify constructor arguments, result names, diagnostics,
   serialization, backend behavior, failure messages, and compatibility with weights,
   cross-fitting, sensitivity analysis, and bootstrap.
4. **Build independent evidence.** Add an oracle or analytic identity, component tests, mutation
   tests for load-bearing signs or masks, end-to-end behavior, and statistical validation only
   where deterministic evidence cannot resolve the claim.
5. **Document the capability and its limits.** Update the user guide, methodology, roadmap, and
   references as appropriate. A well-posed unsupported composition must be refused by name with a
   useful reason, not left to an incidental exception.
6. **Measure before optimizing.** Profile the supported workflow with a realistic learner, improve
   the numpy baseline first, and report the region as a share of an end-to-end fit.

New functionality must not silently change existing default reports, simultaneous-band families,
parameter names, serialized formats, or binary-treatment results. If a compatibility change is
intentional, make it explicit in the public documentation and tests.

## Correctness evidence

Correctness is established against a derivation, not by matching another implementation.
Cross-language output can help localize a discrepancy only when a named question cannot be
answered by existing instruments; it is never the oracle or a release criterion.

Use independent checks in this order:

1. exact algebraic and structural identities;
2. finite-support laws with analytically known truth;
3. componentwise comparisons with the source theorem at values where the component is nonzero;
4. Gateaux derivatives and remainder decompositions;
5. simulation with a stated Monte Carlo standard error.

An exact-law check can be blind where a correction vanishes at the truth. For signs, guards, and
other components that can cancel, construct a law where the relevant term is nonzero and mutate
the implementation to confirm the test fails. A test written after a fix but never observed to
fail under the relevant mutation is not sufficient evidence.

Score convergence is numerical evidence only. It does not prove that the implemented score is the
derived one, that nuisance-rate conditions hold, or that an interval has nominal finite-sample
coverage.

## Testing policy

- Default to `library="glm"` and the shared fast-test configuration unless flexible learning is
  itself under test.
- Use the smallest sample and replication count that resolves the asserted gap. State the sizing
  argument for statistical tests.
- Prefer deterministic identities to stochastic tolerances. Never assert coverage from a single
  fit.
- Share expensive fitted fixtures at class or module scope, and do not refit configurations a
  fixture already contains.
- Spell out fold counts or use the shared fast-test defaults; learner names alone do not reduce
  cross-fitting work.
- Keep intentionally expensive statistical tests marked `slow`. The nightly workflow is their
  validation tier, not an optional duplicate of fast CI.
- The `docs` marker is the second nightly tier: it executes every fenced Python block in the
  documentation, one namespace per document, in reading order. Keep the examples at the sizes the
  guides quote — an example shrunk to make a test cheap is no longer the example. A block that
  enumerates an API rather than working an example carries a `<!-- catalogue: -->` marker and is
  checked statically instead; a marker the parser misses fails open, so its attachment is itself
  asserted.
- When adding a target, update the appropriate discrete-law oracle. Registry tests require targets
  and oracle branches to cover one another in both directions.
- When a test protects a mathematical or architectural invariant, explain the invariant in the
  test rather than citing an investigation number or a closed defect.

## Stable architectural invariants

### Dataframes and labels

User-facing frames go through narwhals and results return in the caller's dataframe library. Keep
backend names, not whole input frames, in data and report objects so results remain serializable
and do not pin input memory. Cast numeric roles through narwhals before conversion; preserve the
treatment's logical type until treatment encoding. Detect null labels through the logical column
type, never by branching on pandas or polars.

Per-arm arrays are keyed by treatment level. Use shared arm-mapping helpers and carry original arm
labels into parameter names, tables, and errors. Do not parse user-visible names to recover arms;
compose the known names forward and retain a structured index.

The binary path is a regression surface for generalized treatment support. A multi-arm change must
leave binary results bit-for-bit stable unless a documented compatibility change is intended.

### Targets, interventions, and variants

A new point-treatment estimand is a `Target` registered through `targets.register`. If it needs a
new score group, register the fluctuation submodel first. Put its influence curve in the shared
inference layer so covariance, bands, delta methods, and score diagnostics remain reusable.

A regime is a density over arms and is a parameter axis distinct from arms, shifts, incremental
interventions, and MSM coefficients. Keep those axes explicit; a fit must not mix incompatible
definitions of its counterfactual under one result namespace.

An estimator variant that only changes which nuisance estimate is targeted should override the
nuisance hook, return a replaced `NuisanceEstimates` with diagnostics, and inherit targeting and
result behavior. `CTMLE` is the reference pattern. A method with different data ordering or
recursion, such as `LTMLE`, should use a separate container and result type and must explicitly
reuse or refuse each shared subsystem.

`TMLE.retarget` operates on cached point-treatment nuisances. Do not assume this contract for a
variant whose derived equations require nuisance refits at the targeted state; document and test
the cost and persistence behavior instead.

### Longitudinal, survival, and competing risks

Resolve dynamic rules once into an assignment matrix before fitting. A rule receives only the
history available at its node, and downstream mechanism, follower, and outcome-regression logic
must read the same resolved plan. Outcome designs contain covariate history, not redundant past
treatment columns that are deterministic under the resolved plan.

Longitudinal MSMs are projections over regimen/horizon cells. Their fluctuation is pooled and the
backward recursion proceeds in lockstep over nodes; the horizon belongs in the design and each
cause receives its own projection while sharing nuisance fits.

For survival, a unit experiencing the event at node `t` belongs in node `t`'s event regression.
The event-free state is part of history, not an intervened mechanism factor. Reports store risk;
survival is a derived scale, with contrasts transformed by sign rather than by `1 - estimate`.

For competing risks, use a cause-specific event numerator with the all-cause survival factor.
Competing events affect the at-risk population but do not add a denominator factor unless a new
estimand explicitly intervenes on them. Cause-specific estimates need not be renormalized to a
simplex.

### Weights, bounds, and sensitivity

Observation weights define the target population and must flow through nuisance loss, score
equations, influence functions, covariance, and effective-sample-size calculations. Estimand or
MSM weights answer a different question; do not merge the two.

Choose mechanism bounds from the clever covariate's algebra. Conditional-effect groups use
propensity odds and need their corresponding bound regardless of arm count. A binary-only method
must declare and validate that restriction rather than indexing two arms by accident.

Sensitivity analyses operate on one named parameter or contrast. Resolve its arms from structured
parameter metadata, not string splitting, and preserve the distinction between assumptions shared
across arms and explicitly arm-specific assumptions.

### Parallelism and performance

Nuisance fits are single-threaded by default so parallelism occurs across folds and learner
candidates. Callers can change the process-level limit through the public learner controls. Do not
add nested native threading without an end-to-end measurement and an oversubscription plan.

Concurrency is `outer × inner × threads-per-fit`, the third is pinned to one, and the split of the
first two belongs to the tier. The fast tier is thousands of short tests, so xdist balances it and
the inner `n_jobs` stays at its default of one. The `docs` tier is the mirror image — one test per
document, and the long one is a single sequential namespace — so xdist has one useful worker and
the budget goes inward, by raising the `n_jobs` default for the run. That is sound only because
`n_jobs` invariance is pinned bit for bit; if that test fails, the injection comes out.

Size every layer from `tests/parallel.available_cores()`, which reads a container's CPU quota and
affinity mask through joblib. Neither `os.cpu_count()` nor xdist's `-n auto` does, so CI and the
noxfile export `PYTEST_XDIST_AUTO_NUM_WORKERS` from it. Nesting pools is sometimes right and is
never assumed: both the configuration in use and the alternative that measured worse are recorded
with their numbers where the defaults are set.

Before adding compiled code, compare against a competent numpy implementation and include the
real learner workload. Track compile time, memory, core count, numerical equivalence, and the
kernel's share of a fit. Generated benchmark output is not documentation; preserve environment
metadata and summarize durable conclusions in `docs/benchmarks/`.

## Documentation and maintenance

- Public behavior belongs in the user guide; derivations and validation arguments belong in the
  methodology; current priorities and limitations belong in the roadmap.
- Keep comments and docstrings about the behavior they protect. Do not cite ephemeral task IDs,
  work-package names, agent sessions, or line numbers in historical plans.
- Every relative documentation link and heading anchor is tested. Update references in the same
  change as a rename or deletion.
- Keep generated benchmark results, caches, local environments, and agent scratch directories out
  of Git.
- Add dependencies only for a supported runtime capability. Development and benchmark tools stay
  in their dedicated extras.
- Bump serialized formats deliberately, test old-format behavior where supported, and report
  unsupported operations explicitly after loading.

## Before committing

```bash
ruff check .
ruff format --check .
mypy src/cleverly
pytest -m "not slow and not docs" -q -n auto
```

Also run the smallest relevant slow test, backend smoke test, or benchmark correctness tier when
the change touches statistical validation, optional dataframe backends, or benchmark kernels.
