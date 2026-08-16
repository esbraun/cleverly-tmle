# Roadmap

This document contains only proposed work and the evidence required to begin it. Implemented
capabilities belong in the [user guide](user-guide.md), scientific contracts in the
[technical appendix](methodology.md) and [DR-TMLE contract](drtmle.md), validation results in
[`docs/evidence.md`](evidence.md), and performance findings in [the benchmark reports](benchmarks/).

What is still *open* in a released estimator is a limitation of that estimator rather than a
roadmap item, so it is not listed here. `DRTMLE`'s are in
[what the validation programme established](drtmle.md#what-the-validation-programme-established),
which states what is not established and why the release claim is conditional.

## Eligibility

`cleverly` implements established statistical methods; it does not use a package feature as the
place to invent one. A scientific feature enters implementation only when a published derivation
covers the estimand and the requested inference regime. For a new estimator or composition, that
means an identified parameter, its influence function, the targeting or estimating equations, and
the remainder and rate conditions needed for the interval being claimed. If one of those is
absent, the roadmap item is to locate published theory, not to create it here.

A canonical public implementation is valuable implementation provenance. It can settle intended
control flow, data layout, and named conventions, and it can expose cases the paper leaves easy to
misread. It is not acceptance evidence by itself. Where code and paper appear to disagree, the
published derivation governs and the discrepancy becomes a nonzero regression or mutation test.
The independent evidence requirements in [the technical appendix](methodology.md) and
[`docs/evidence.md`](evidence.md) still apply.

The status labels below rate **published-method support**, not programming effort:

- **published support** — a paper derives the requested method and inference claim;
- **source audit** — published theory and/or canonical code appear to cover it, but the exact
  construction must be matched and any discrepancy resolved before implementation;
- **theory-neutral** — an engineering capability that preserves an already-derived estimator;
- **waiting on published theory** — related estimators or code exist, but not the requested
  composition or inference claim. This is not an active research assignment for this project.

A label may add **pending source read**, meaning the governing result is published and identified
but has not yet been read first-hand into this package's contract. An item is not started on the
strength of a result nobody here has read.

## Definition of done

An item is finished when all of these hold, not when the estimator runs and returns a plausible
number:

- the estimand is registered and covered **in both directions** by the oracle and evidence gates in
  `tests/unit/test_registry.py`, with a row in [`docs/evidence.md`](evidence.md) naming which
  instruments check its influence curve and which mistakes none of them would see;
- a well-posed composition this package still refuses has a test pinning the refusal and its
  message, so the refusal cannot decay into a silent approximation of a different estimand;
- wherever a sign, mask, guard, or counterfactual block can vanish at the truth, an exact-law check
  is accompanied by a nonzero witness or a deliberate-mutation control that fails when that
  component is wrong. Exact-law checks alone are blind to terms that disappear at the truth;
- a cross-module change satisfies [the architecture invariants](architecture-invariants.md), which
  also hold the standing decisions and the condition that would reopen each one;
- **every check has been run locally.** GitHub Actions is out of budget: jobs currently fail at
  startup in seconds with no steps run, which looks identical to a red build. A pull request's
  checks are not a verdict on its code, and pushing does not test anything.

## Ordered priorities

### 1. Extend the published DR-TMLE surface

Work these in order. The canonical implementation reference is the MIT-licensed
[`benkeser/drtmle`](https://github.com/benkeser/drtmle) R package, read at the commit
[the references](references.md#doubly-robust-inference-drtmle) pin. Its source is implementation
provenance; van der Laan (2014), Benkeser et al. (2017), and Benkeser & Hejazi (2023) supply the
statistical claims.

1. **Cross-validated DR-TMLE — source audit.** Benkeser & Hejazi (2023, §4.7) and the canonical
   package's `cvFolds` path provide published and executable guidance. First map that construction
   to this package's distinct nuisance cross-fitting, `targeting_scheme="fold"`, and
   `cv_evaluation=True` semantics. Implement only the modes for which the published parameter,
   corrected influence curve, and fold aggregation coincide; a shared name such as “CV-TMLE” is
   not proof that they do.
2. **Bivariate reduction — published support, pending source read.** van der Laan (2014),
   Theorem 3, supplies the regularity conditions; Benkeser et al. (2017) supplies the bivariate
   expansion; and the canonical package implements and tests
   `reduction="bivariate"`. Read Theorem 3 into the
   contract before transcribing the single two-column reduced mechanism and its different extra
   score equation. This is an alternative to the implemented univariate reduction, not an assumed
   improvement over it.

The remaining DR-TMLE refusals are **waiting on published theory** for this package's requested
composition. Public implementations of ordinary TMLE for `att`/`atc`, stochastic interventions,
continuous shifts, incremental interventions, marginal structural models, mediation, and C-TMLE
do not establish an interval that remains valid when one primary nuisance is inconsistent.
Estimated weights likewise need a published influence contribution for estimating the weights.
Missing treatment also remains refused: canonical smoke tests do not supply this package's
required identification, corrected curve, remainder, and rate conditions.
Keep these refusals until a paper supplies the missing reduced regressions, corrected influence
curve, remainder, and rate conditions; do not generalize the mean construction by analogy.

### 2. Complete the LTMLE implementation surface

Work these in order, retaining the same oracle and evidence gates as the implemented longitudinal
estimands.

1. **Multi-valued treatment nodes — published support, pending source audit.** Poulos et al.
   (2024) study longitudinal TMLE with multi-valued treatments and provide public MIT-licensed
   [`multi-ltmle`](https://github.com/jvpoulos/multi-ltmle) reproduction code. Confirm the
   cumulative treatment mechanism, regimen indexing, influence curve, and working-MSM map against
   the paper and code before extending the binary-node implementation.
2. **Targeted bootstrap — waiting on a citable construction.** Keep this second in the LTMLE
   queue, but do not infer a procedure from the name. Implementation begins when a published
   source states what is held fixed, what is resampled, which nuisance and targeting steps are
   rerun, and what sampling law the resulting interval estimates. Resampling a stored influence
   curve, retargeting cached arrays, and refitting the complete estimator are distinct procedures.
3. **Persistence and serialization — theory-neutral.** Preserve the complete fitted recursion,
   regimen and node metadata, targeting state, diagnostics, and enough learner state to
   distinguish operations that can be replayed from those that require a refit. Round trips must
   leave estimates, curves, scores, and refusals unchanged.
4. **Sensitivity analysis — source audit for each operation.** A sweep over prespecified nuisance
   bounds may refit the already-derived estimator and report diagnostics without defining a new
   estimand. Any operation that changes the intervention, missingness law, or reported parameter
   needs its own published identification and influence-function result. In every case the full
   backward recursion must rerun when the bound changes an earlier pseudo-outcome.

### 3. Add published longitudinal estimands

Additional longitudinal estimands, including interventions on competing events, are **waiting on
published theory** until their separate identification assumptions, influence functions, targeting
construction, and inference conditions are available. Once a source supplies those objects, add
the estimand in both directions to the oracle registry and evidence gates rather than treating it
as another option on the existing cause-specific estimand.

### 4. Add time-respecting cross-fitting

Blocked-temporal and rolling-origin cross-fitting are next after the longitudinal estimands. Treat
each as a **source audit**: select a published sample-splitting result whose dependence assumptions
match the supported data structure, then record which rows may train each prediction and which
asymptotic argument licenses the reported influence-curve interval. Reusing the iid fold machinery
with ordered indices is not sufficient.

## Later candidates

These remain worthwhile but are not ahead of the ordered work above:

- replicate-weight designs such as BRR and jackknife, once the published variance construction is
  matched to the package's weighted-law estimands;
- an MNAR tilt for continuous-dose shifts, and intermediate variables with incremental
  interventions, only when published identification and influence-function results cover those
  exact compositions;
- HAL and undersmoothed HAL learners, following their published loss, basis, optimization, and
  undersmoothing criteria. Consider native implementation only if profiling shows that their
  package-owned workload materially dominates end-to-end time.

## Reading a gap correctly

Not every absence on this page is the same kind of absence, and the full refusal taxonomy is in
[How to read a refusal](methodology.md#how-to-read-a-refusal). It distinguishes missing package
functionality from a different causal question and from a method that would be wrong by
construction. Only the first is a roadmap item.
