# Method benchmarking strategy

Method benchmarking has two complementary parts: bounded comparisons with an external R
implementation and independent statistical studies against properties implied by the derivation.
Neither replaces the package's exact-law, Gateaux, remainder, identity, and deliberate-mutation
checks. Use this strategy when adding or changing a method-level row in the
[implementation validation grid](../technical-reference/method-evidence/validation-grid.md).

## Begin with the scientific claim

Record the identified parameter, estimator construction, governing source, supported
compositions, and inference rule before choosing a comparator. Agreement with an R package is
acceptance evidence for the claims the comparison covers, as
[Theory and comparators](../architecture-invariants.md#theory-and-comparators) states.
[RM42](../roadmap.md#rm42-acceptance-status-and-refusal-reasons) part (e) adds the comparator-backed
label to the evidence pages. Agreement does not prove that the parameter, influence curve, or
theorem is correct.

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
- Record each fitting failure of either implementation, and account for it as
  [Comparator fitting failures](#comparator-fitting-failures) states. The run fails on a
  missing replication that has no record. Do not replace samples or summarize a shorter run
  than the one declared.
- Resume a reference phase that a killed worker or container stopped. The phase refits only the
  groups without a checkpoint, as the [testing strategy](testing-strategy.md#resume-a-study-run) describes.
- Retain paired replication-level estimates, standard errors, interval endpoints, truth,
  coverage, initial estimates where exposed, and pairing keys. Require a nonzero targeting
  witness so parity cannot be explained by both implementations returning their initial fits.

Test each implementation against known truth before comparing them. Use predeclared equivalence
and non-inferiority margins for the paired comparison rather than a null test of exact equality.

### Comparator fitting failures

The R harness (`tests/canonical/study_harness.R`) runs each comparator call inside
`study_stage`. An error in that call becomes a failure record, and the run continues. The
Python phase fits each `cleverly` replicate through `fit_replicates`
(`tests/studies/evidence/replication.py`), and a recordable error there also becomes a record.
The driver publishes both sides' records as `failures.csv` and the accounting as
`failure-accounting.csv`. Both files exist only when a fit failed.

| rule | statement |
| --- | --- |
| bound | `StudyRecord.failure_bound` declares a bound `b` in [0, 0.05] per implementation. A cell may lose `floor(b * R)` replicates. An undeclared implementation has bound 0. Every registered study declares 0.01 for each implementation. The value governs the study's next run |
| bound 0 | One failure makes the cell red. Diagnose the red cell, fix a genuine defect, and re-run under a fresh declaration. Otherwise report it |
| raising a bound | Declare a bound before the study runs under it. A study whose run has shown failures may not raise its bound. The driver refuses such a run before any fit (`failures.raised_after_failures`) |
| manifest and claims | A run writes `configuration.failure_bound`. The fast tier fails a stated key that differs from the record, and committed failures without the key. The claims quote `margin:<side>_failure_bound` from the manifest, so a page quotes the bound only after a run applied it |
| fallback fit | Do not replace a failed replicate by a fallback fit or a new sample |
| invalid result | A comparator row with a non-finite value, a non-positive standard error, an inverted interval, or a `covered` flag that disagrees with its interval becomes an `invalid_result` record |
| worst case | Each verdict leg must pass under its least favourable candidate fill of the lost values. The bias, coverage, and similarity candidates lie inside the surviving range. The RMSE and coverage-difference legs fill the reference at its best case: the estimate at the truth and the interval covering it. They fill the subject at its worst case: each lost estimate at the surviving estimate farthest from the truth, and each lost interval missing the truth. The calibration leg tries every pair of the two sides' candidate fills. The range is an assumption, and the accounting records it |
| enumeration | The SE-ratio and calibration candidates are a finite set: edge, mean, and root fills. They are not a proof that no other fill is less favourable |
| routes | A paired cell passes by equivalence or by superiority. A route survives when each of its legs passes its worst case. A `superior` cell that keeps only equivalence reads `equivalent`. A cell that keeps no route reads `unaccounted` |
| pairing | A paired cell drops the replicates that either side lost |
| witness | The accounting reports the subject's mean error on the replicates only the reference lost and on the shared ones. It reports the reference's mean error on the replicates only the subject lost and on the shared ones. The witness is descriptive. No gate reads it |
| hooks | A study hook must define `account_failures`, or its accounting row reads `failures not accountable`, which is red. A probe that refits a replication outside `study_stage` stops the run when that refit fails |
| calibration | A group that fails during memory calibration is skipped, and calibration takes the next group of its scenario. When 20 groups of one scenario fail, the run stops as systematic |
| recording off | `CLEVERLY_R_NO_RECORD=1` turns recording off. Every error in a stage then stops the run, as it did before this rule |
| subject failures | The Python phase records a `DataError`, a `LongitudinalError`, a `numpy.linalg.LinAlgError`, a `FloatingPointError`, a `ZeroDivisionError`, an `OverflowError`, and a plain `ValueError` or `RuntimeError` that `numpy`, `scipy`, `sklearn` or `statsmodels` raised (`cleverly.validation._failures`). Any other error stops the run. An error in the draw stops the run |
| subject rows | A subject row that breaks a row rule of the `invalid result` rule becomes an `invalid_result` record |
| side bounds | Each implementation's failures count against its own bound. A paired cell is red when either side exceeds its bound |
| paired input | A study whose reference input does not need the subject's fit still writes the sample of a replicate that the subject lost, so the reference fits it. A study whose reference input comes from the subject's fit writes no such sample. The driver then writes a `paired_input_missing` reference record, which counts against the subject's bound |
| Python checkpoints | Under the driver, each Python replicate runs in a worker and leaves a checkpoint. A killed worker costs the replicates in flight. Each one runs once more alone, and a second kill stops the run. A `MemoryError` in a worker's fit is a kill. A `MemoryError` elsewhere in the worker stops the run |
| zEpid | The zEpid runner records failures with a copy of the Python rule that adds `zepid` to the libraries. It runs in one container, so the driver's stored reference result is its resume |
| property phase | A recordable error in a property fit costs the cells that the fit writes one replicate each. See [Property fitting failures](#property-fitting-failures) |

The stage covers the comparator call only. These conditions stay fatal and stop the run:

- an error outside a stage, such as a data read or an output write,
- a `study_fatal_error`, such as a promoted warning,
- a C stack overflow,
- the first 20 groups, or Python replicates, of one cell all fail (a cell is one scenario, or the set of scenarios one key spans), and
- every group, or every Python replicate, of the run fails.

A memory-exhaustion error ends the worker as a kill. The same run then refits that group with
half the workers.
A failure message can carry a container path. The record keeps it as written.

### Property fitting failures

A property loop fits through `fit_properties` (`tests/studies/evidence/replication.py`). A
recordable error in one fit costs the cells that the fit writes one replicate each. The run
continues. The driver opens `replication.recording()` around the property phase and publishes the
records. These files exist only when a property fit failed. A study with no failure publishes the
bytes it published before.

| file | content |
| --- | --- |
| `property-failures.csv` | one record for each failed fit, with the columns of `failures.csv`. The `scenarios` field lists the `property/cell` names that the fit writes |
| `property-failure-accounting.csv` | for each summary row that read a lost replicate: the bound, then one row for each declared clause. Each clause row gives the complete-case slack, the worst-case slack, the label, and the gate |
| `property-failure-witnesses.csv` | the witness of each draw in each cell that lost a replicate |

#### Record and stamp rules

| rule | statement |
| --- | --- |
| multi-cell key | One fit can write several cells. A lost fit costs every cell it writes one replicate. `cells_of` lists those cells. A fit whose rows differ from `cells_of` stops the run |
| `SUMMARY_ROWS` | A module whose `summarize_properties` first rescales its rows declares that function as `SUMMARY_ROWS`. The accounting's candidates then read the same scale, so a `k = 0` candidate equals the complete case. `ctmle_oat_per_arm_strata_missing_properties` puts a ratio cell's standard error on the scale of `psi` |
| `DERIVED_CELLS` | A module declares each base cell's derived controls (`shrunken_se_control`, `noise_control`). A record of the base names its controls, because they lose the same replicate |
| stamp | The driver is the only writer of `failed_replicates`. It sets each cell's count from the records. `reconcile` refuses a cell whose missing ids differ from its recorded ids |
| `require_accounted` | A cell may be short by exactly its stamped failures and must keep two replicates. Any other gap stops the summary |
| pairing | A paired statistic reads the replicates that both arms kept (`paired`). It refuses a misalignment that no stamp explains. The rule covers cross-key pairs, the selector arms, and `projection_sd_ratio` |
| derived noise | `calibration_controls` draws `noise_control` over the requested range and indexes the noise by replicate. A lost base replicate changes no other control row |
| verdict budget | `_at_verdict_budget` keeps the ids below the verdict budget. It never refills a lost id from a later one. It sets `failed_replicates` to the ids lost below the budget. A loss at a higher id shows on the slope row only |
| rate rows | `fitted_rate_row` stamps the sum of its rungs' failures |
| draw split | Each property loop draws its sample in `draw=`, outside the classifier. An error in the draw stops the run. `WORKER_EXCEPTIONS` in `tests/canonical/python_phase_identity.py` lists the modules whose worker source changed with the split |
| byte identity | `tests/canonical/python_phase_identity.py` runs the base and the head in one environment and compares their rows. It does not compare a rerun with the committed rows. Those carry float noise of 1e-16 to 3e-12 from the environment that wrote them, and a rerun of the base shows the same noise |
| no-draw loop | `repeated_crossfit_properties` passes no `draw=`. Its stability trial draws one sample before the loop, and each key only varies the fold seed. An error in that draw stops the run before any fit. `UNSPLIT_LOOPS` in `tests/unit/test_property_loops.py` pins this list |
| `known_mechanism` | A bootstrap that lost draws writes the side column `bootstrap_failed_draws`. The row check records that row as an `invalid_result`, and the published row writes 0. This extends the row rules of the `invalid result` rule |
| memory cap | `longitudinal_mtp` caps its workers for each size. The pool keeps at least two workers, so a size with a cap of 1 can run two fits. The driver retries a worker killed for memory alone |
| classifier | The classes and the stop conditions of the `subject failures` rule apply to the property phase |
| no collector | Outside the driver, a recorded property failure raises `RuntimeError`. Tests and diagnostics keep the all-fatal behavior |
| checkpoints | The property phase checkpoints each fit under `checkpoints/<implementation>/properties[/<name>]/`. A rerun refits only the unfinished fits |

#### The bound

| item | rule |
| --- | --- |
| declaration | Every property module declares `FAILURE_BOUND`. The value is 0.01 in every module. The range is 0 to 0.05 |
| allowance | A cell may lose `floor(b * R)` replicates. The smallest committed cell (`R` = 400) allows 4. A contraction rung of 73,000 replicates allows 730 |
| gate | A cell above its allowance reads `bound exceeded` before any clause runs. A cell inside it reads `within bound` or `worst case fails`, by its clauses |
| declared before the run | The value governs a study's next property run. No committed property run lost a replicate, so the value rescues no committed result |
| raising | Raise the bound only while no committed run of the study shows a property failure. The driver refuses a property run whose `FAILURE_BOUND` exceeds the bound of a committed run that recorded failures (`property_failures.raised_after_failures`) |
| lowering | Lowering is allowed. It changes the `bound` and `allowed` columns, so the property phase reruns in the same change |
| manifest | A property run writes `configuration.property_failure_bound`. A `--skip-properties` run copies the committed manifest's key, or its absence. The fast tier fails a manifest whose key differs from the module, and failure files without the key |
| primary phase | Each implementation of each registered study declares 0.01 for its next run. No committed primary run lost a replicate, so the value moves no committed verdict. The primary bound has the same facts as the property bound (see the next table) |

The value 0.01 follows from these facts.

| fact | consequence |
| --- | --- |
| Seeds are deterministic, so a pathological replicate fails again on every rerun | At bound 0 the cell stays red. At 0.01 the worst case of the claim decides |
| At bound 0 the gate returns `bound exceeded` before it reads a clause | At 0.01 every declared clause can act in a real run |
| A lost fit is plausibly an extreme draw, so its estimate is the least likely to lie in the survivor range | For an unbounded statistic, the worst-case mean shift is about `b * sqrt(R) * range / sd` standard errors. This is about 1 at `R` = 400 and about 3 at `R` = 2,400. Large cells turn red through their clauses before they reach the allowance |
| A bounded statistic fills its full support | The range assumption does not apply to it |

The primary bound has two more facts.

| fact | consequence |
| --- | --- |
| The accounting only turns a cell red. A paired leg fills the reference at its best case and the subject at its worst case | A lost replicate on either side never rescues a verdict. At 0.01 the reference side's losses are judged by the same legs |
| No committed run has a `failures.csv` | No manifest needs the key, no claim artifact changes, and no study reruns for this declaration |

#### Accounting statistics

Each clause reads one statistic. Each statistic has a candidate set for the `k` lost values. A
cell is red when a clause that holds on the complete cases fails under one candidate. A
`support` clause fills a bounded variable over its full support and needs no assumption. A `range`
clause assumes that a lost value lies inside the survivor range and that the scale of the margin
stays at its complete-case value. Every bootstrap in a candidate uses the verdict's own seed, so
the case `k = 0` reproduces the complete case.

| statistic | verdicts it judges | candidate fills | tier | label |
| --- | --- | --- | --- | --- |
| S1 bias interval | equivalence, discrimination, and a fixed floor on the bias interval | errors at every vertex split between the survivor minimum and maximum, with the scale held. A sided clause holds on its declared side | exact | `range` |
| S2 coverage rate | coverage floors, bands, and ceilings | `covered = 1` for `j` of `k` lost values, for every `j` | exact | `support` |
| S3 rejection rate | type-I error, power, warning rate | `rejected = 1` for `j` of `k`, for every `j` | exact | `support` |
| S4 SE ratio, point | the SE sanity band | the SEs at the survivor extremes against the estimates at the survivor mean and at each vertex split | exact | `range` |
| S5 SE and efficiency ratio intervals | calibration bands and controls | the S4 set, through `ratio_intervals` | enumerated | `range` |
| S6 rate slope | root-n rate, contraction slopes | for each lost rung, every vertex split and one fill at the survivor mean, with exact pruning. For a bias slope, the fill nearest a zero mean replaces the survivor-mean fill | point exact, interval enumerated | `range` |
| S7 coverage gain | paired overfitting and clustered gain | a lost positive arm at `covered = 0`, a lost control arm at `covered = 1`, the surviving arms at their real values | exact | `support` |
| S9 paired spread ratio | randomizer projection, repeat stability | the numerator at each vertex split and the denominator at the survivor mean, or the reverse for a lower bound | point exact, interval enumerated | `range` |
| S10 extra column mean | a cap or a floor on a published column | the lost rows at the adverse support bound for a share, otherwise at the survivor maximum or minimum | exact | `support` for a share, else `range` |
| S12 paired displacement | the 16 necessity families | the attainable difference of the arm means, with `sd` of the left arm held | exact under the held scale | `range` |
| S13 RMSE ratio, point | selector necessity | the numerator errors at the farther survivor end, the denominator estimates at the truth | exact when the truth lies in the survivor range | `range` |
| S14 two-size absolute-bias bound | single robustness, sequential robustness | the larger size at every vertex split, the smaller size at equal fills | exact | `range` |
| S15 standardized-bias growth | `one_wrong_inference` rule G | every vertex split, the equal fills that maximize and minimize the ratio, and the fill nearest a zero mean | point exact, interval enumerated | `range` |

"Exact" means that the candidate set reaches the worst value over every completion that the
label allows. "Enumerated" means that the set is a finite list. The verdict's own bootstrap
evaluates it. The list does not prove that no other fill is worse.

#### Rule per family

A module declares its rules in `PROPERTY_RULES`, beside `summarize_properties`. A module rule
takes precedence over the shared `FAMILY_RULES` entry. The fast tier fails a verdict family that
resolves to no rule. The `group` of a rule names the cells that share a joint verdict: `row`,
`pair`, `label`, or `family`. A module declares its pairs and labels in `PROPERTY_GROUPS`. The
rule never infers a pair from a cell name.

| families | clauses | group |
| --- | --- | --- |
| `double_robustness`, `cde_robustness`, `mar_robustness`, `robustness_contract` | positive: S1 equivalent, S4 in the union band. control: S1 discriminated, S4 | `row` |
| `root_n_and_efficiency` | positive: S2 floor, S4 sanity, S1 equivalent. control: S2 resolve rule | `row` |
| `root_n_rate`, `double_robust_contraction` slopes | S6 slope inside the band, or the sign of the bias slope | `row` |
| `type_i_error`, `power`, `warning_rate` | S3, with the S2 floor for `type_i_error` | `row` |
| `interval_calibration` and its controls | S5 in the band, S2 in the band, S5 efficiency where a band is declared | `row` |
| `simultaneous_coverage` | S2 in the band, or at the ceiling for the control | `row` |
| `crossfit_overfitting`, `clustered_inference` | S5 and S2 on each cell, S7 gain on the pair | `pair` |
| `few_cluster_reference` | S1 equivalent, S2 floor, S5 ceiling | `row` |
| the 16 displacement families | S1 on each cell, S12 minimum over the labels | `family` |
| `selector_necessity` | S1 on each arm, S13 maximum over the paths | `family` |
| `generated_design` | S5 and S2 on each cell, then every cell passes | `family`, or `label` in `ctmle_oat_per_arm` |
| `treatment_score_necessity`, `mechanism_term_necessity` | S5 on each cell, then every cell passes | `family` |
| `mechanism_requirement` | S1 on each cell, then every cell passes | `family` |
| `estimand_weighting` | S2 on each cell, then every cell passes | `family` |
| `known_mechanism_accuracy`, `ordinary_targeting`, `treatment_complete_case` | S1, S2, S4 as the verdict applies them | `row` |
| `bootstrap_coverage` | S2 floor | `row` |
| `randomizer_projection` | S1 on each arm, S9 on the pair of projections | `family` |
| `repeat_stability` | S9 | `row` |
| `single_robustness` | S14 over the two sizes | `family` |
| `sequential_double_robustness` | S14 over the two sizes, with S1, S2, S4 on the both-correct pattern | `pair`, `row` |
| `clustered_calibration`, `weighted_calibration` | S5 and S2 in the band | `row` |
| `one_wrong_inference` | rules C, V and F with S1, S2, S4, S5, and rule G with S15 | `row` |
| `corrected_mar_inference`, `correction_necessity` initial score | S1 equivalent or beyond a floor, S2, S4 | `row` |

Three clauses are not accountable, because no sound candidate set exists. A lost replicate in
one of them turns the cell red at any bound, and the reason reads `failures not accountable`.
Diagnose the cell, fix the cause, and rerun.

| clause | reason |
| --- | --- |
| `natural_course_identity` and `static_reduction` | The clause is a supremum of a difference with a fixed tolerance. One lost value moves a supremum by an order-one amount. A range fill leaves it unchanged, so the leg would pass whenever the complete case passes. A wider fill fails every time |
| the closed-score clause of `correction_necessity` | The claim is that the fit's score closes. A recorded fit failure is a counterexample, and a range fill assumes it away. No range completion fails the clause, so the leg is vacuous |

Rows with no verdict (`fold_policy`, `cluster_aggregation_rule`, `in_sample_agreement`,
`variance_direction`, `one_wrong_reported`, `construction_diagnostic`, `sdr_tmle_agreement`,
`censoring_complete_case`, `in_sample_fit`, and the diagnostic rows of `few_cluster_reference` and
`single_robustness`) get no accounting row. The stamp is their record.

#### The informative-failure witness

Each module declares `PROPERTY_WITNESS`, a statistic of the draw that needs no fit. A lost
replicate's draw is regenerated from its deterministic seed. The accounting reports the statistic
and its quantile among the cell's survivors. The report is not a gate, because no validated
threshold exists. A draw that does not separate from the fit reports `nan` with the reason
`draw not separable`.

#### The property differential

A rule restates a verdict that the study's own `summarize_properties` computes. A rule that drops
a conjunct would let a lost replicate turn a cell green. Two checks guard the rules.

| check | where | content |
| --- | --- | --- |
| differential | the `property-differential` job of `.github/workflows/ci.yml`, from `tests/canonical/property_differential.py` | The job perturbs the property rows of every committed study. It also perturbs synthetic rows of the two uncommitted modules and of `longitudinal_sdr_properties`. The committed `longitudinal-sdr` rows give no witness for one clause. It summarizes them with the study's own code. The declared rules must give the same `passed` and `property_passed` on every verdict row. The perturbations scale `std_error`, shift `estimate`, flip indicators, push a slope across zero, collapse one arm of a pair, apply a seeded affine map, and scale or shift each extra column |
| clause binding | the same job | Each clause that a subject uses must decide a verdict alone under a perturbation of that subject. Many clauses read a threshold of their own module, so a witness in another study does not bind them. `tests/canonical/clause-bindings.csv` names one perturbation for each subject and clause. The search adds a two-point grid for a module's own band, and role-restricted and composed kinds, where the shared kinds bind nothing |
| binding reasons | the same job | A pair that its subject cannot decide alone has a row in `tests/canonical/clause-binding-reasons.csv`. The job verifies each reason. The reasons follow this table |
| no-loss candidates | the same job | At `k = 0`, each candidate of each clause of each subject must give the complete-case value and verdict (F18). This covers every statistic type |
| fast tier | `tests/unit/test_method_evidence.py` | The job's parametrization lists every committed study. Every verdict family resolves to a rule. A module that writes `passed` outside a shared function declares `PROPERTY_RULES`. Each shared rule reproduces the committed verdicts. The bound and the manifest key agree |

| reason | check | where it applies |
| --- | --- | --- |
| `fixed at size` | The S2 or S3 clause gives one verdict at every success count from 0 to `R` on each of its rows. No perturbation changes `R` | the root-n control's `coverage resolves` at `R` of 800 or more: the 99 per cent interval is narrower than the gap between the floor and nominal |
| `same module as: <subject>` | The other subject runs the same module and record, and its witness decides the pair | the committed and the synthetic rows of `longitudinal_sdr_properties` |
| `implied by: <clause>` | On every row, the other clause fails wherever this one fails. The check sweeps each role's mean error from -3 to 3 times, scales its spread, and reads every shared perturbation | `bias discriminated` under `alternative bias equivalent` in the two weighted studies and in `repeated_outcome_necessity` |

The job runs 5,892 cases in about 7,600 CPU-seconds, about 35 minutes on four workers. So it does not run in the fast tier, and its timeout is 105 minutes, three times the measured run. It runs on every pull
request that touches `tests/studies/`, `tests/canonical/`, `src/cleverly/`, `pyproject.toml`,
`uv.lock` or the workflow. A failed `git diff` fails the job instead of skipping it. Make it a
required check. It always reports, and it skips its steps on other pull requests.

### Choose the comparator before you write the runner

Search for every maintained implementation of the parameter. Record what you found. Record why
you rejected each candidate you did not use. A study that reports no comparator makes a claim
about the field, and a reader cannot check that claim unless the study names what it examined.
A study with no comparator is still eligible to register. It records the independent evidence
that replaces the comparison, as [Theory and comparators](../architecture-invariants.md#theory-and-comparators)
requires.

A comparator phase can finish after the method merges. The method merges when its Python phases
pass with their expected properties. Until the comparator results are registered, the evidence
page labels the evidence "comparator runs pending". A later change
registers the comparator results and fixes each defect that they show.

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
| R `lmtp` 1.5.4 with R `ife` 0.2.3 | clustered point-treatment effects, clustered cross-fitted longitudinal regimen effects at equal cluster sizes, deterministic regimes, categorical longitudinal regimes, known stochastic regimes, modified treatment policies | Used by point and longitudinal studies. The adapters supply analytic density ratios and exact rowwise folds. The clustered study also supplies identifiers and forms contrasts from joint influence curves. Its longitudinal TMLE fits each fold's fluctuation on training rows, and `cleverly` fits one pooled fluctuation per node, so each cross-fitted longitudinal paired verdict compares two constructions under the registered margin. |
| R `npcausal` at `56a5ac1` | point-treatment effects, counterfactual densities, continuous-treatment curves, and incremental propensity-score interventions | Used by the incremental study. Its public estimator list has no deterministic categorical longitudinal regimen. |
| R `stremr` | static, dynamic, and stochastic longitudinal regimes with categorical exposures | Not used for categorical parity. Its required long-form data introduces a second representation, while pinned `lmtp` accepts the study's node columns directly. |
| Poulos `multi-ltmle` companion | simulation code for longitudinal multi-valued treatment | Retained as supporting source provenance. It is a simulation repository, not a versioned package entry point for rowwise paired studies. |
| R `imtp` 0.1.0 at `d4b5204` | incremental odds curve | Rejected. Its influence curve omits that derivative, so it witnesses the point curve and cannot gate inference. |
| R `ltmle` 1.3-0 | deterministic point-treatment regimes | Available and not used. `abar` accepts a single treatment node, which `ltmle_regimen_adapter.R` already reaches at `horizon = 1`. One comparator per study is the framework limit today. |
| R `txshift` 0.3.8 | continuous shift interventions | Not used. It estimates the exposure density through a second density path, which requires a separate study rather than serving as a second opinion on this one. |
| R `tmle3` at `ed72f8a` | static and dynamic point-treatment regimes | Rejected for stochastic regimes. `Param_TSM` evaluates a counterfactual at one treatment value and does not integrate over a declared density. |

The point-treatment survival survey covers one baseline treatment and an event over visits.

| candidate | parameter it reaches | verdict |
| --- | --- | --- |
| R `survtmle` 1.1.1 | `method="mean"`: the iterated-mean TMLE of the cumulative incidence of Benkeser, Carone and Gilbert (2018), for two arms and competing causes | Used by `point-treatment-survival`. It is the governing source's own implementation. It refuses more than two arms and has no RMST, so those are property cells. Its fluctuation fits one logistic coefficient per arm's clever covariate, which separates by arm, and this package fluctuates each node's intercept |
| R `lmtp` 1.5.4 | survival with a baseline treatment, a known policy or a modified treatment policy | Used by `point-treatment-survival-policies` with K copied treatment columns and the identity policy after node 1. With `trt` of length one it evaluates every node's regression at the shifted treatment, which applies a policy that is not idempotent once per node, a different estimand |
| R `tmle3` at `ed72f8a` | `tmle_survival`: a whole-curve hazard fluctuation on `k_grid <- 1:max(T_tilde)`, binary treatment | Not used. It belongs to the hazard-based targeting of [X14](../roadmap.md#x14-hazard-based-and-monotone-survival-curves), whose pairing must pass the same grid |
| MOSS, `concrete` | monotone and continuous-time survival | Not used. They belong to X14 and to continuous-time intensities |
| R `ltmle` 1.3-0 | a point treatment with survival nodes | Not used. It adds no witness that `survtmle` does not, for one comparator per study |

The cluster survey records how each pinned comparator aggregates a clustered curve. At unequal
cluster sizes the cluster sum is the delta-method variance of the row mean, and the cluster mean
is a different variance ([references](../references.md#grouped-folds-and-clustered-cross-fitting)
gives the source locators).

| candidate | aggregation | verdict |
| --- | --- | --- |
| R `ltmle` 1.3-0 | the cluster sum, $t$ with $J-1$ degrees of freedom below 100 clusters | Used by `clustered-few-cluster-tmle` in sample. Its point and standard error pair with this package. Its interval uses $J-1$ degrees of freedom and this package uses $J-2$, a declared difference |
| R `tmle` 2.1.1 | the cluster mean, normal reference | Not used at unequal sizes. Its variance is a different quantity there |
| R `tmle3` at `ed72f8a` | the cluster mean, normal reference | Not used at unequal sizes, for the same reason |
| R `lmtp` 1.5.4 with R `ife` 0.2.3 | the cluster mean, normal reference | Used by `clustered-tmle` at equal sizes only, where the two aggregations agree |
| R `drtmle` 1.1.2 at `538a3a2` | no cluster argument | Not applicable |

The missing-outcome survey is separate because the response mechanism changes the observed-data
likelihood and the comparator boundary.

| candidate | parameter it reaches | verdict |
| --- | --- | --- |
| R `tmle` 2.1.1 | ordinary point-treatment TMLE with MAR outcomes | Used for the observational arm-indexed row. It accepts separate treatment and response nuisance predictions and reports arm means and their contrast. |
| R `tmle` 2.1.1 population-mean path | missing-outcome natural-course mean | Used for the stacked natural-course row, conditional on supplied stitched outcome and response predictions. Used for both laws of the ordinary natural-course row, conditional on the supplied in-sample predictions. A continuous fit plants the two `q_bounds` on two `Delta = 0` rows, and a committed probe checks that workaround on every continuous replication. Neither row compares nuisance training, and the stacked row does not compare fold generation. |
| R `tmle` 2.1.1 two-arm and population-mean paths | stacked arm-indexed missing-outcome means and contrasts | Used for the stacked arm-indexed row, conditional on supplied stitched outcome, treatment, and response predictions. L1 and L2 use the two-arm path. L3 and L4 run the population-mean path once for each arm, and the runner builds the joint covariance from the per-arm curves. A continuous fit plants the two `q_bounds` on two `Delta = 0` rows, because R takes the scale from every non-`NA` outcome, and a committed probe checks that workaround. |
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
| R `tmle` 2.1.1 population-mean path | accepts stitched out-of-fold outcome and response predictions, targets once, and evaluates on the whole sample | Used by the stacked missing-outcome natural-course row and, once for each arm, by the stacked arm-indexed row. Each comparison is conditional on the supplied predictions. |
| R `drtmle` 1.1.2 | pools the out-of-fold predictions and forms one estimate on the whole sample | Rejected for the fold-evaluated row. |
| R `lmtp` 1.5.4 | takes `weighted.mean` of the pooled shifted regression column. Its TMLE fits each fold's fluctuation on that fold's training rows. After upstream commit `9996b04`, it fits the fluctuation on validation rows. Neither version fits the pooled fluctuation of Section 5.2 of Díaz, Williams, Hoffman and Schenck (2023) | Rejected for the fold-evaluated row. Used elsewhere for its intervention family. |
| R `medoutcon` at `nhejazi/medoutcon` | binds the per-fold results, then targets over the pooled validation rows | Rejected. Its estimand is a mediation effect, and its aggregation is pooled. |
| R `npcausal` at `56a5ac1` | averages the influence-function values over all rows | Rejected. It is a one-step AIPW estimator, and its aggregation is pooled. |
| Julia `TMLE.jl` v0.20.4 | takes the mean of the counterfactual aggregate over all rows, after one fluctuation | Rejected for the fold-evaluated row. The multi-arm selector survey below also rejects it as a canonical comparator. |
| Python `zEpid` 0.9.1 at [`16a0f96`](https://github.com/pzivich/zEpid/blob/16a0f96f8b2c65df8715085801f21757d1478e1e/zepid/causal/doublyrobust/crossfit.py#L976-L1048) | `SingleCrossfitTMLE` targets separately by validation split, then takes the mean over the stacked targeted rows, which size-weights the folds. It uses a within-fold sample variance with `ddof=1`; `cleverly` uses an equal $1/V$ fold average and the raw influence-curve second moment instead | Used by the fold-targeted CV-TMLE row at two equal folds and one partition. The two point-estimate weightings coincide only at these equal sizes. At two folds, each nuisance training split is the complete validation-fold complement. It also corroborates repeated-report aggregation, but it does not compare with the stacked pooled update. |
| R `Crossfit` at `momenulhaque/Crossfit` | takes the median of the split estimates under a double cross-fit | Rejected. Double cross-fitting fits each nuisance on a separate split, which is a different estimator. |
| Chernozhukov et al. (2018), Definition 3.3 and equation (3.14) | explicitly define median aggregation for fixed repeated partitions with within-partition variance plus squared split displacement | Governing source for the median reporting rule and fixed-repeat first-order validity. It does not make the DML score a TMLE or establish full-method numerical parity. |
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

Repeated-sampling, large-sample, and flexible-learner claims belong in a registered study and its
committed artifacts. Documentation examples never count as statistical evidence.

### A truth that varies by replication

Set `truth_varies_by_replicate=True` on a `StudyRecord` when the target is a data-adaptive
parameter. The value of the rules that each training fold learns is an example: each replication
has its own truth. The spread of the estimate then differs from the spread of the error, so every
statistic must read the error. The flag defaults to `False`. Every other record keeps its
arithmetic byte for byte, because subtracting a constant truth moves the last bits.

| statistic | a record with one truth | a record that sets the flag |
| --- | --- | --- |
| schema check of the truth | constant within `TRUTH_TOLERANCE` | finite on every row. `covered` is checked against the row's own truth |
| inference scale | any declared scale | the level scale only |
| bias, its interval, and the standardized bias | the mean estimate minus the truth, over the SD of the estimate | the mean error, over the SD of the error |
| empirical SE, RMSE, SE ratio and its bootstrap | from the estimates | from the errors. The bootstrap resamples the error and the SE together |
| the empirical efficiency ratio and the root-n `spread` slope | the SD of the estimate | the SD of the error |
| paired displacement | on the estimates | on the errors. Each arm subtracts its own truth |
| `summary.csv` and `properties.csv` | `truth` | `truth` holds the mean truth, beside `truth_min` and `truth_max` |

Such a record cannot declare a reference, because the paired comparison reads one truth per cell.
`tests/unit/test_per_replicate_truth.py` holds the witnesses. The learned-rule studies set the
flag.

## Registration and acceptance

Register the scenarios, estimands, sample size, replication count, margins, runner and property
modules, expected cells, artifact directory, document anchor, and every result-determining module.
A study without an external comparator records a valid empty equivalence artifact rather than a
surrogate reference.

The manifest must record a hash under `study_module_sha256` for every study-specific module. List
each one in the `StudyRecord` `modules` field, which supplies that list.
`tests/unit/test_study_provenance.py` checks the manifest against the rule below.

| rule | detail |
| --- | --- |
| which modules count | every `tests` module that the runner module or the properties module imports, directly or transitively within `tests/` |
| which imports count | every import in a file, including an import inside a function and a relative import. The test reads imports with `ast` and imports no study module |
| what is excluded | the shared framework under `tests/studies/evidence/` and `tests/parallel.py`. The walk does not follow imports inside them. Some framework modules feed results, for example `pairing.py`, but the rule still excludes them. A package `__init__.py` is never required |
| older gaps | `KNOWN_GAPS` in the test file lists them. The test fails when a listed gap is repaired but still listed, so the list can only shrink |
| gaps from a result-neutral refactor | the table with the `study`, `source`, and `judgement` columns in `provenance-revisions.md` declares them. The test fails when a declared module is no longer a gap |

The test names both routes when it finds a new gap. Choose the route by the kind of change.

| change | action |
| --- | --- |
| result-determining | Add the module to the `StudyRecord` `modules` field and regenerate the study. The manifest list must equal the `modules` field, so name the module in both |
| result-neutral, for example a helper moved into a new module | Declare the study slug and the module path in [`tests/canonical/provenance-revisions.md`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/provenance-revisions.md). Do not add a hash, because the new bytes never ran |

A gap leaves `KNOWN_GAPS` in one of two ways. Regenerate the study, or record the hash of a file
that git shows unchanged since the run and declare it in the same ledger. A declared gap leaves
the ledger when you regenerate the study.

Set `publication_policy="reporting"` when a red scientific result is part of the declared
evidence. This policy writes failed statistical verdicts. It does not permit an unrecorded missing
replication, invalid schemas, non-finite estimates, or incomplete provenance. Existing studies use
`publication_policy="gated"` unless they declare otherwise.

Published studies retain `replicates.csv.gz`, `property-replicates.csv.gz`, `summary.csv`,
`performance-tests.csv`, `equivalence.csv`, `properties.csv`, and a provenance- and hash-complete
`manifest.json`. A run with a recorded comparator failure also retains `failures.csv` and
`failure-accounting.csv`. A run with a recorded property failure also retains
`property-failures.csv`, `property-failure-accounting.csv`, and `property-failure-witnesses.csv`.
Run a disposable smoke study first, then the declared study without permitting
failed replications. The [red cells](#red-cells) section states what may change after a result.

A regeneration owns its declared core budget. Do not run the Python and R full-core phases concurrently. Do not run the fast tests
beside a study regeneration. Documentation quotes measured values through
`tests/studies/evidence/claims.py` so tests can check them against the artifacts.

Every evidence row states what it does not cover, including relevant outcome and treatment types,
missingness, weights, clusters, fold repeats, learner class, truncation, interval type, and
unsupported estimands. Validate changes with the complete fast suite and every registered study
whose evaluated path a result-determining change can affect. Record each selected study and its
regeneration command in the pull request.

### Comparator runs pending

A method merges when its Python studies pass. The comparator study does not block the merge or a
dependent row, and a later change registers it. This is the owner's rule, and it applies
[Theory and comparators](../architecture-invariants.md#theory-and-comparators).

| step | what to do |
| --- | --- |
| register | Set `pending_reference` to the declared comparator name and leave `reference=None`. Set `pending_reason` to the reason and the item that owns the runs. Keep passing the `Reference` to `regenerate.main` |
| regenerate | `regenerate.main` runs no reference process for a pending record. The Python phase, the independent tests, the failure accounting, the property families and every subject gate run as for any study. `equivalence.csv` has no rows |
| manifest | `generated_with.reference` is `null` and `reference_sha256` is empty. The `comparator` block has `status: pending`, the comparator name, the reason, the reference metadata, and `declared_sources_sha256`. No test compares those digests with the tree |
| document | The grid's canonical-implementation cell names the pending comparator. That cell, the paired cell and the study section carry the label "Python property evidence, with comparator runs pending". The paired cell reads 0/0 |
| add the comparator | Move the name from `pending_reference` to `reference`, clear `pending_reason`, and regenerate the study with its reference |

`tests/unit/test_method_evidence.py` asserts the pending state for every registered study through
`manifest.comparator_findings`. It is a check, not a skip. A pending study that publishes a
comparator row or a paired row fails, because an empty paired table would otherwise read as passed.
`tests/unit/test_comparator_pending.py` holds the controls for each part.

To rerun the property families against the committed Python rows, pass `--reuse-subject`. The
driver offers the flag only when no reference process runs. It checks the committed rows against
their manifest hash and the declared budget, and then it fits no subject model.

## Red cells

Diagnose every red cell before you route it. The table gives the route for each diagnosis.

| diagnosis | route |
| --- | --- |
| a genuine problem: an algorithm defect, an inconsistent estimator, or a test or design bug that makes the cell measure the wrong thing | fix the code or the test. Then re-run under a fresh declaration commit that states what changed and why. This is allowed after a verdict |
| no defect, such as a finite-sample limit | publish the cell red under `reporting`, with an owner row. Declare the route before the repeat run |

Never game a test. A change whose effect is to make a test less meaningful or easier to pass is
refused, before or after a verdict. Examples are moving a margin, adding replications to cross a
bound, weakening the law, dropping or relabelling a cell, and substituting an oracle.

## What makes a study stale

A study's artifacts are evidence about one run. The `manifest.json` records three groups of
hashes, and each group answers a different question.

| hash group | what it covers | what the fast tier does | what a difference means |
| --- | --- | --- | --- |
| `sha256` | every published artifact | refuses every difference | an artifact differs from the bytes its manifest records |
| `reference_sha256` | Dockerfiles, R runners, shared R harnesses | refuses an undeclared difference | the comparator's source moved after the run |
| `study_module_sha256` | the Python modules the record names | checks that the list is complete, and gates no hash | a Python source moved after the run |

Ask one question of any change to a hashed source. Does it change what the study computes?

A **result-determining** change makes the affected evidence stale. Regenerate that evidence under the
[testing strategy](testing-strategy.md#targeted-reruns). Laws, margins, cells, seeds, learners,
estimator arguments, the published schema, and package pins are all result-determining.

A **result-neutral** change does not. A rename, a comment, a type annotation, an extracted helper,
and a script moved from an image into a mount are all result-neutral. A regeneration would spend
hours to write identical bytes.

No tool separates the two, so each hash group takes its own position.

The published artifacts are the evidence itself. A difference from their recorded hashes fails.
An approved regeneration records the hashes of its new artifacts. A targeted record also retains every baseline hash.

Python module hashes are not gated. Selective regeneration keeps the artifacts honest, and a hash gate
would mean regenerating twenty studies for a docstring. Refactor the shared machinery in
`tests/studies/evidence/` freely. Regenerate only the studies whose results the change moves. A
wrong result-neutral judgment appears as a changed artifact at the next required regeneration.

Reference sources sit between the two. They are gated, because a pinned container is the one input
a reader cannot re-derive from this repository. Declare a result-neutral difference in
[`tests/canonical/provenance-revisions.md`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/provenance-revisions.md).

Do not rewrite a recorded hash. The manifest keeps the hash of the bytes that ran. A rewritten
hash makes the manifest claim that bytes which did not exist produced the result.

### Targeted regeneration record

A targeted rerun combines inherited replications with new fits under one unchanged study design.
Its manifest must identify both sources. It cannot describe every row as a new execution.

| record | required contents |
| --- | --- |
| baseline | immutable artifact commit, original manifest and its hash, artifact hashes, source identification, runtime, and original clean-state flag |
| selection | declared reason, complete affected replication keys, and old and new seeds for a seed repair |
| replacement fits | source commit, clean-state flag, result-determining module hashes, runtime, command, and fit failures |
| inherited evidence | preserved files and rows, their baseline attribution, and exact comparison results |
| combined analysis | complete replication counts, analysis source and runtime, unchanged design, recomputed summaries, bootstrap intervals, controls, and verdicts |
| final artifacts | hashes of every data artifact and replacement record; the completed run log records the manifest hash |

Preserve the original manifest metadata, including an original dirty-worktree flag.
Record replacement and analysis provenance separately. Do not overwrite historical module hashes with current hashes.
The record must expose a runtime difference even when the published verdict stays unchanged.

Derive the selection before fitting. Validate the complete combined key set and all dependent comparisons.
Require an independent review of the reuse boundary and final artifacts.
If verification fails, investigate the difference or use a complete rerun.
Verify the committed run log against the completed execution record.

A committed notebook follows the same rule under a different name. `scripts/execute_notebook.py`
stamps it, and `tests/notebooks.py` splits the stamp in two.

| stamp half | holds | behaves like |
| --- | --- | --- |
| gated | the notebook's code cells and stored outputs | the published artifacts. Any difference fails |
| recorded | the package source, the dependency lock, and the generator | the module hashes. Nothing is asserted equal |

The recorded half needs no declared exception, because no edit to your tree can fail it. Carry it
forward when you restamp a notebook you did not re-execute. Recomputing it would name a checkout
that never produced those outputs, which is the error the paragraph above describes.

Carry a value forward only while the digest that produced it is unchanged. A carried value that
predates a change to `tests/notebooks.py` matches no checkout at all, because no commit ever
computed it that way. Re-execute the notebook instead when a digest definition moves.

The recorded half names its generator file set. `generator_files` lists the files the generator
digest folded, and the fast tier compares that list against the current `GENERATOR_MODULES`. A
file-set change therefore fails with an instruction. A semantic digest change still needs a
schema-version change and review.

`cleverly_commit` and `cleverly_version` place the run in repository history. A digest reports
that two trees differ and never reports where. Read `cleverly_worktree_clean` before you interpret
the commit. A notebook runs before the commit that lands it, so the recorded commit is the parent.
A `false` value means that the commit does not reconstruct the executed tree.

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
row rests on the other two columns. A comparison is acceptance evidence for the claims it covers.
Today the validation grid lists only a study with property cells.
[RM42](../roadmap.md#rm42-acceptance-status-and-refusal-reasons) part (e) owns the label and the
grid row for a comparator-backed study.

A study with no property cells can still register as a reporting study. It declares
`publication_policy="reporting"` and an empty `property_cells`. It keeps its own page, with no
properties block, and the red-cell ledger reads its verdicts. The validation grid does not list
it. `tests/unit/test_method_evidence.py` checks each of these facts. The learned-rule boundary
study is the example.
