# Method benchmarking strategy

Method benchmarking has two complementary parts: bounded comparisons with an external R
implementation and independent statistical studies against properties implied by the derivation.
Neither replaces the package's exact-law, Gateaux, remainder, identity, and deliberate-mutation
checks. Use this strategy when adding or changing a method-level row in the
[implementation validation grid](../technical-reference/method-evidence/validation-grid.md).

## Begin with the scientific claim

Record the identified parameter, estimator construction, governing source, supported
compositions, and inference rule before choosing a comparator. An R package is an implementation
witness: agreement can localize a discrepancy, but it cannot prove that the parameter, influence
curve, or theorem is correct.

Give materially different estimators separate evidence rows. Ordinary TMLE, stacked CV-TMLE,
fold-evaluated CV-TMLE, and fold-specific extensions may share a limit while differing in finite
samples; one method does not inherit another's parity or coverage claims.

## Reference implementation comparisons

- Pin the runtime base image and packages immutably. Record hashes for the Dockerfile, runner, and
  reference sources in the study manifest.
- Give both implementations the same realized datasets, covariates, contrasts, nuisance-model
  families, bounds, targeting controls, and interval scale. Declare any setting that cannot be
  matched and exclude only the affected comparison.
- Supply and assert the exact row-to-fold assignment for cross-fitted methods. A common seed or
  fold count is not enough when splitters or dependency versions differ.
- Preserve native inference scales, including log-scale risk-ratio and odds-ratio intervals.
- Fail the run on a dropped or unsuccessful replication. Do not replace samples or summarize a
  shorter run than the one declared.
- Retain paired replication-level estimates, standard errors, interval endpoints, truth,
  coverage, initial estimates where exposed, and pairing keys. Require a nonzero targeting
  witness so parity cannot be explained by both implementations returning their initial fits.

Test each implementation against known truth before comparing them. Use predeclared equivalence
and non-inferiority margins for the paired comparison rather than a null test of exact equality.

### Choose the comparator before you write the runner

Search for every maintained implementation of the parameter. Record what you found. Record why
you rejected each candidate you did not use. A study that reports no comparator makes a claim
about the field, and a reader cannot check that claim unless the study names what it examined.

Prefer the implementation that the governing source publishes. That implementation reports the
influence curve the source derives. A separate package that reproduces the point estimate does
not have to report the same influence curve, and one such package does not.

Match the comparator's nuisance specification. Give both implementations the same regression
family over the same design. An asymmetric pair measures two model choices instead of two
targeting steps. The asymmetry then fails a calibration comparison for a reason that belongs to
neither estimator.

This survey covers the intervention family. Each verdict names the evidence behind it.

| candidate | parameter it reaches | verdict |
| --- | --- | --- |
| R `lmtp` 1.5.4 with R `ife` 0.2.3 | clustered point-treatment effects, deterministic regimes, categorical longitudinal regimes, known stochastic regimes, modified treatment policies | Used by point and longitudinal studies. The adapters supply analytic density ratios and exact rowwise folds. The clustered study also supplies identifiers and forms contrasts from joint influence curves. |
| R `npcausal` at `56a5ac1` | point-treatment effects, counterfactual densities, continuous-treatment curves, and incremental propensity-score interventions | Used by the incremental study. Its public estimator list has no deterministic categorical longitudinal regimen. |
| R `stremr` | static, dynamic, and stochastic longitudinal regimes with categorical exposures | Not used for categorical parity. Its required long-form data introduces a second representation, while pinned `lmtp` accepts the study's node columns directly. |
| Poulos `multi-ltmle` companion | simulation code for longitudinal multi-valued treatment | Retained as supporting source provenance. It is a simulation repository, not a versioned package entry point for rowwise paired studies. |
| R `imtp` 0.1.0 at `d4b5204` | incremental odds curve | Rejected. Its influence curve omits that derivative, so it witnesses the point curve and cannot gate inference. |
| R `ltmle` 1.3-0 | deterministic point-treatment regimes | Available and not used. `abar` accepts a single treatment node, which `ltmle_regimen_adapter.R` already reaches at `horizon = 1`. One comparator per study is the framework limit today. |
| R `txshift` 0.3.8 | continuous shift interventions | Not used. It estimates the exposure density through a second density path, which requires a separate study rather than serving as a second opinion on this one. |
| R `tmle3` at `ed72f8a` | static and dynamic point-treatment regimes | Rejected for stochastic regimes. `Param_TSM` evaluates a counterfactual at one treatment value and does not integrate over a declared density. |

The missing-outcome survey is separate because the response mechanism changes the observed-data
likelihood and the comparator boundary.

| candidate | parameter it reaches | verdict |
| --- | --- | --- |
| R `tmle` 2.1.1 | ordinary point-treatment TMLE with MAR outcomes | Used for the observational missing-outcome row. It accepts separate treatment and response nuisance predictions and reports arm means and their contrast. |
| R `drtmle` 1.1.2 at `538a3a2` | corrected randomized point-treatment means with missing outcomes | Used only in the both-correct limit. Its `gn` is the joint treatment-response mechanism, so it cannot witness `cleverly`'s separate five-reduction cycle or either component-specific drift direction. |

The fixed-weight survey is separate because observation weights change the target law and every
estimation stage.

| candidate | weighted construction it reaches | verdict |
| --- | --- | --- |
| R `tmle` 2.1.1 | ordinary point-treatment TMLE through `obsWeights=` | Used by both weighted point-treatment rows. The fixed-nuisance runner supplies the same weights and exact predictions to both implementations. The learned-nuisance runner fits matching weighted main-effects Gaussian and binomial GLMs. In the fixed-nuisance row, R `tmle` reports a marginal log-odds-ratio standard error below the exact bound. That comparison therefore witnesses the point estimate rather than the inference. The learned-nuisance row reports two arm means and their difference, and no ratio. |
| R `ltmle` 1.3-0 | ordinary weighted longitudinal TMLE through `observation.weights=` | Used by the registered ordinary weighted longitudinal reporting study. Its sequential data layout adds no witness to the point-treatment row. |
| R `lmtp` 1.5.4 | cross-fitted weighted longitudinal TMLE through `weights` | Used by the registered cross-fitted weighted longitudinal reporting study. It is not ordinary point-treatment TMLE. |

The cross-fitting survey is separate because the question is how an implementation aggregates over
folds, not which parameter it reaches. Every candidate below describes itself as cross-fitted or
cross-validated, and the candidates do not agree on what that means. Read the aggregation line in
the source before you accept one.

| candidate | how it aggregates over folds | verdict |
| --- | --- | --- |
| R `tmle3` at `ed72f8a` | stacks the validation rows, targets once, and evaluates on the whole sample | Used by the stacked CV-TMLE row. It is not the fold-evaluated construction. |
| R `drtmle` 1.1.2 | pools the out-of-fold predictions and forms one estimate on the whole sample | Rejected for the fold-evaluated row. |
| R `lmtp` 1.5.4 | takes `weighted.mean` of the pooled shifted regression column | Rejected for the fold-evaluated row. Used elsewhere for its intervention family. |
| R `medoutcon` at `nhejazi/medoutcon` | binds the per-fold results, then targets over the pooled validation rows | Rejected. Its estimand is a mediation effect, and its aggregation is pooled. |
| R `npcausal` at `56a5ac1` | averages the influence-function values over all rows | Rejected. It is a one-step AIPW estimator, and its aggregation is pooled. |
| Julia `TMLE.jl` v0.20.4 | takes the mean of the counterfactual aggregate over all rows, after one fluctuation | Rejected for the fold-evaluated row. The multi-arm selector survey below also rejects it as a canonical comparator. |
| Python `zEpid` 0.9.1 at [`16a0f96`](https://github.com/pzivich/zEpid/blob/16a0f96f8b2c65df8715085801f21757d1478e1e/zepid/causal/doublyrobust/crossfit.py#L976-L1048) | `SingleCrossfitTMLE` targets separately by validation split, then takes the mean over the stacked targeted rows, which size-weights the folds. It uses a within-fold sample variance with `ddof=1`; `cleverly` uses an equal $1/V$ fold average and the raw influence-curve second moment instead | Used by the fold-targeted CV-TMLE row at two equal folds and one partition. The two point-estimate weightings coincide only at these equal sizes. At two folds, each nuisance training split is the complete validation-fold complement. It also corroborates repeated-report aggregation, but it does not compare with the stacked pooled update. |
| R `Crossfit` at `momenulhaque/Crossfit` | takes the median of the split estimates under a double cross-fit | Rejected. Double cross-fitting fits each nuisance on a separate split, which is a different estimator. |
| Chernozhukov et al. (2018), Definition 3.3 and equation (3.13) | explicitly define median aggregation for fixed repeated partitions with within-partition variance plus squared split displacement | Governing source for the median reporting rule and fixed-repeat first-order validity. It does not make the DML score a TMLE or establish full-method numerical parity. |
| current R and Python `DoubleML` | uses median aggregation with a between-partition dispersion adjustment | Further corroboration for the reporting rule, but rejected as a comparator because it is DML rather than a targeted estimator and does not expose `cleverly`'s stacked update. |

The multi-arm collaborative survey is separate because the selector, not the parameter, is what a
comparator has to reach.

| candidate | selector it reaches | verdict |
| --- | --- | --- |
| R `ctmle` 0.1.2 | greedy and pre-ordered selectors for a binary treatment | Rejected. `ctmleDiscrete` and `ctmleGeneral` both document the treatment as a binary indicator. |
| R `ctmle` 0.1.2 through `ctmleGeneral`, one arm against the rest | greedy and pre-ordered selectors on `1{A = a}` | Rejected, and not because the parameter differs. An arm mean under the indicator is the same parameter. Per-arm selection is a different mechanism from this package's joint multiclass selection, and the return value carries no influence curve, so a contrast would have no covariance. |
| archived R `ctmle3` at `a4ea77b` | the outcome-adaptive treatment model | Used by the outcome-adaptive rows. It ships no greedy, ordered, or discrete selector. |
| Julia [`TMLE.jl` v0.20.4](https://doi.org/10.21105/joss.08446) | componentwise `GreedyStrategy` and dynamic `AdaptiveCorrelationStrategy` paths | Rejected as a canonical comparator and not planned as a roadmap priority. Its [`JointEstimand` method](https://github.com/TARGENE/TMLE.jl/blob/dacc908df9addb174e24d4a7ec61a9a26ad46914/src/estimators.jl#L294-L306) fits each requested component separately. The [adaptive strategy](https://github.com/TARGENE/TMLE.jl/blob/dacc908df9addb174e24d4a7ec61a9a26ad46914/src/counterfactual_mean_based/covariate_based_strategies.jl#L46-L92) reorders covariates from the latest targeted residual. That differs from the fixed published preorder that `cleverly` uses. It has no native ratio target, so ratios require composition across separately selected components. The survey found no published multi-arm selector-aware theorem for either construction. The v0.20.4 [candidate fluctuation](https://github.com/TARGENE/TMLE.jl/blob/dacc908df9addb174e24d4a7ec61a9a26ad46914/src/counterfactual_mean_based/collaborative_template.jl#L195-L220) receives the complete data. Its [held-out loss](https://github.com/TARGENE/TMLE.jl/blob/dacc908df9addb174e24d4a7ec61a9a26ad46914/src/counterfactual_mean_based/collaborative_template.jl#L290-L314) is evaluated afterward, so the implementation is not suitable for a numerical comparison without a fold audit. |

### Multi-arm selector recommendation

Keep the shipped joint selector. Do not replace it with the componentwise `TMLE.jl` path.
The library review records the alternative and its limitations; implementing it is not a current
priority and it is not tracked on the roadmap.

[van der Laan and Gruber (2010)](https://doi.org/10.2202/1557-4679.1181) derive greedy
collaborative selection for one target parameter. [Ju et al.
(2019)](https://doi.org/10.1177/0962280217729845) develop the scalable pre-ordered approach.
Neither paper supplies a theorem for separately selected multi-arm components and their contrast
covariance. The [`TMLE.jl` JOSS paper](https://doi.org/10.21105/joss.08446) documents the software,
but it does not add that theorem.

Two limits of the framework shape these tables. A study record names one `reference`, so a second
comparator needs a second registered study. A comparator that fits its own nuisances also fixes
the specification the subject must match.

## Independent statistical evidence

Each property cell declares an exact-truth law, fixed estimator configuration, replication budget,
margin, and seed rule before the full run. Size the budget against the binding confidence bound,
not the most favorable point estimate.

Primary replications use the study record's `replicate_seed`; property cells use fixed shared
seeds so methods see identical draws and paired controls merge by replication index. Resampling
uses independently labelled `stream_seed` streams rather than adjacent integer offsets.

Use interval-shaped verdicts: bias intervals must fit inside a practical equivalence margin,
coverage lower bounds must clear a validity floor, and one-sided type-I bounds must rule out
material over-rejection. Positive claims need a control that makes the same instrument fail:

- double robustness includes a both-wrong nuisance control;
- type-I error includes a nonzero-effect power control;
- coverage or standard-error calibration includes deliberately invalid inference; and
- convergence uses at least three sample sizes and excludes a predeclared slower rate.

Repeated-sampling, large-sample, and flexible-learner claims belong in the named slow study and
committed artifacts. Documentation examples never count as statistical evidence.

## Registration and acceptance

Register the scenarios, estimands, sample size, replication count, margins, runner and property
modules, expected cells, artifact directory, document anchor, and every result-determining module.
A study without an external comparator records a valid empty equivalence artifact rather than a
surrogate reference.

Set `publication_policy="reporting"` when a red scientific result is part of the declared
evidence. This policy writes failed statistical verdicts. It does not permit missing replications,
invalid schemas, non-finite estimates, or incomplete provenance. Existing studies use
`publication_policy="gated"` unless they declare otherwise.

Published studies retain `replicates.csv.gz`, `property-replicates.csv.gz`, `summary.csv`,
`performance-tests.csv`, `equivalence.csv`, `properties.csv`, and a provenance- and hash-complete
`manifest.json`. Run a disposable smoke study first, then the declared study without permitting
failed replications or tuning margins after seeing the result. A regeneration expects the whole
machine: do not run the Python and R full-core phases concurrently, and do not run the fast and
slow test tiers concurrently. Documentation quotes measured values through
`tests/studies/evidence/claims.py` so tests can check them against the artifacts.

Every evidence row states what it does not cover, including relevant outcome and treatment types,
missingness, weights, clusters, fold repeats, learner class, truncation, interval type, and
unsupported estimands. Validate changes with the targeted evidence and documentation tests, the
complete fast tier, and only the named slow study whose path and assertion can observe the change.

## What makes a study stale

A study's artifacts are evidence about one run. The `manifest.json` records three groups of
hashes, and each group answers a different question.

| hash group | what it covers | what the fast tier does | what a difference means |
| --- | --- | --- | --- |
| `sha256` | the six published artifacts | refuses every difference | somebody edited the committed evidence |
| `reference_sha256` | Dockerfiles, R runners, shared R harnesses | refuses an undeclared difference | the comparator's source moved after the run |
| `study_module_sha256` | the Python modules the record names | records the hash and gates nothing | a Python source moved after the run |

Ask one question of any change to a hashed source. Does it change what the study computes?

A **result-determining** change makes the artifacts stale, so regenerate the study. Laws, margins,
cells, seeds, learners, estimator arguments, the published schema, and package pins are all
result-determining.

A **result-neutral** change does not. A rename, a comment, a type annotation, an extracted helper,
and a script moved from an image into a mount are all result-neutral. A regeneration would spend
hours to write identical bytes.

No tool separates the two, so each hash group takes its own position.

The published artifacts are the evidence itself. Any difference fails, and nothing is declarable.

Python modules are not gated. Re-execution is what keeps the artifacts honest, and a hash gate
would mean regenerating twenty studies for a docstring. Refactor the shared machinery in
`tests/studies/evidence/` freely. Regenerate only the studies whose results the change moves.
`pytest -m slow` re-executes each study, so a change that was not result-neutral appears as a
changed artifact.

Reference sources sit between the two. They are gated, because a pinned container is the one input
a reader cannot re-derive from this repository. Declare a result-neutral difference in
[`tests/canonical/provenance-revisions.md`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/provenance-revisions.md).

Do not rewrite a recorded hash. The manifest keeps the hash of the bytes that ran. A rewritten
hash makes the manifest claim that bytes which did not exist produced the result.

## Adding a method row

A row is not written, it is earned by registering a study. The machinery in
`tests/studies/evidence/` is method-agnostic, so a new method supplies only what is genuinely its
own and inherits every gate, every negative control and the checks over the validation grid.
Follow the rest of this guide rather than designing new margins, fold-matching rules,
artifacts, or provenance conventions for each comparison.

1. **Declare it.** Add a `StudyRecord` to `tests/studies/evidence/registry.py`'s register: the
   name the validation grid's first cell must carry, the artefact directory, the document and
   section anchor, the scenarios and their estimands, the replication count and sample size, and a
   `Margins`. The `Margins` carries every acceptance margin, declared before the run rather
   than chosen after it.
2. **Write only the method-specific half.** A law to sample from with an exact parameter oracle,
   a fit function, and a reference implementation pinned by digest where one exists. Seeds
   come from `replicate_seed` *applied to this study's own record*, so a replication is a fixed
   sample whatever the study's size and a short probe redraws the published one. Reusing another
   study's ready-made sampler inherits its seed silently while publishing your own, which a test
   now refuses: two registered studies may share a law but not a set of datasets.
3. **Say what a failure would look like.** Property cells are only evidence if a claim can fail:
   pair each positive cell with a control that must fail the same instrument in the opposite
   direction, and pair any rejection-rate cell with a power cell so an inert test cannot pass.
4. **Quote nothing by hand.** Numbers in the section go in its measured-values table by quantity
   name; the gate resolves each against the artefacts and checks it at the precision printed.
   The per-test tables are generated between sentinel comments by
   `python -m tests.studies.evidence.document`, so the section carries three empty
   `<!-- generated: ... -->` blocks and nothing typed into them.
5. **Name every key in plain English.** Add each new scenario, estimand, property family and
   cell to `tests/studies/evidence/descriptions.py`. A key the module does not describe fails
   the gate, so a study cannot publish a row whose meaning a reader cannot recover.
6. **Declare the blind spots.** The limitations cell is the column the grid exists for. A
   composition the study does not reach is a separate row, not an implied one. Cross-fitting,
   weights, clustering, missing outcomes, and a flexible learner are each such a composition.

If a registered study has no canonical comparator, its cross-implementation cell says so and its
row rests on the other two columns; a study with no property cells is not a validation row at
all, because matching another implementation is not evidence that either one is right.
