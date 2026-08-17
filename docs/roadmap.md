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

DR-TMLE has no active item. Its cross-validated source audit and bivariate reduction are complete.
The latter follows van der Laan (2014), Theorem 3 and the pinned R `estimategrn`, `fluctuateQ2`,
and `eval_Dstar_Q` branches; those branches apply the construction armwise to a discrete
multi-level treatment, while univariate remains the default. The multi-arm surface is therefore
an implementation-backed extension of the binary theorem and is labelled as such.

One DR-TMLE refusal is narrower than the rest and is stated separately because a single source
would close it. **Multi-arm missing-outcome DR-TMLE — waiting on published theory.** `delta=`
under `guard=("Q", "g")` refuses a treatment with more than two arms: Díaz and van der Laan's
missing-outcome theorem is stated for a binary randomized treatment, and the per-arm multi-level
assembly of its observation, treatment and outcome correction blocks is not in it. This is not
the armwise situation of the complete-outcome reductions above — there, a pinned canonical
implementation applies the construction arm by arm and the extension is implementation-backed.
Here nothing states what the arm-indexed blocks are, so an analogy would be inventing the
construction rather than matching one. Implementation begins when a source gives the multi-arm
corrected influence curve, its remainder, and the rate conditions for the interval; the existing
binary evidence in [`docs/evidence.md`](evidence.md) is then the regression surface it must not
disturb.

The remaining DR-TMLE refusals are **waiting on published theory** for this package's requested
composition. Public implementations of ordinary TMLE for `att`/`atc`, stochastic interventions,
continuous shifts, incremental interventions, marginal structural models, mediation, and C-TMLE
do not establish an interval that remains valid when one primary nuisance is inconsistent.
Estimated weights likewise need a published influence contribution for estimating the weights.
Missing treatment also remains refused: canonical smoke tests do not supply this package's
required identification, corrected curve, remainder, and rate conditions.
Keep these refusals until a paper supplies the missing reduced regressions, corrected influence
curve, remainder, and rate conditions; do not generalize the mean construction by analogy.

### 1. Complete the LTMLE implementation surface

Deterministic multi-valued treatment nodes are no longer listed here: they are implemented, so
by this document's own scope rule they belong in the [user guide](user-guide.md#treatment-given-over-time),
[the technical appendix](methodology.md#treatment-given-over-time-the-sequential-regression),
and [`docs/evidence.md`](evidence.md), with the source audit that preceded them recorded in
[`docs/references.md`](references.md). What the audit narrowed is worth keeping in one line,
because it governs what a future item may cite: Poulos et al. (2024) is a **point-treatment**
multinomial TMLE paper and R `ltmle` 1.3-0 is binary implementation provenance, so neither is
acceptance evidence for a longitudinal categorical extension.

Work these in order, retaining the same oracle and evidence gates as the implemented longitudinal
estimands.

1. **Stochastic categorical policies at a node — waiting on published theory.** The implemented
   surface is deterministic: a static or dynamic rule assigns one label per unit, and the clever
   covariate selects that label's conditional probability. A policy assigning a *distribution*
   over the labels replaces the intervention density, so the cumulative product carries a ratio
   rather than a selected column and the parameter is the mean under that density. Implementation
   begins when a source supplies that parameter's identification, its longitudinal influence
   function, and the remainder and rate conditions for the interval — not by analogy with the
   point-treatment regime path, which is a single node.
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

### 2. Add published longitudinal estimands

Additional longitudinal estimands, including interventions on competing events, are **waiting on
published theory** until their separate identification assumptions, influence functions, targeting
construction, and inference conditions are available. Once a source supplies those objects, add
the estimand in both directions to the oracle registry and evidence gates rather than treating it
as another option on the existing cause-specific estimand.

### 3. Add time-respecting cross-fitting

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
