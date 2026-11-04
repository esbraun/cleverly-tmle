# cleverly 0.2.0

This release extends the learned-rule value, `LearnedRuleValue`, to the fit settings that other
point-treatment targets accept, and adds the value of the optimal rule, `OptimalRuleValue`.

## Upgrade notes

- Read the fold fields of a learned-rule fit from `result.extra["learned_rule"].splits`, which
  holds one `LearnedRuleSplit` per split, in the order of `result.repeats`. `LearnedRuleRecord` no
  longer has `fold_sizes`, `fold_estimates`, `treated_shares` or `blip_quantiles`. On a split,
  `arm_shares` replaces `treated_shares`: each fold gives the share of each arm, in the order of
  `record.arms`, and at two arms the second share is the treated share. `gap_quantiles` replaces
  `blip_quantiles`. ([#530](https://github.com/esbraun/cleverly-tmle/pull/530),
  [#547](https://github.com/esbraun/cleverly-tmle/pull/547))
- `FrozenLearnedRule.treat` now holds an `int8` arm code per row instead of a boolean, and the new
  `levels` field holds the treatment levels of the fit. A refit with other levels refuses.
  ([#547](https://github.com/esbraun/cleverly-tmle/pull/547))

Refit analyses when you move saved results to 0.2.0. Loading another version's result emits
`VersionMismatchWarning`. The loader does not migrate the result, and the load can fail.

## Added

### Learned-rule settings

Each fit below estimates a data-adaptive target, the value of the rules that the fit learns. A
fit reads supported only inside the setting that its registered study measured, and reads
experimental elsewhere. The interval can under-cover when the treatment effect is zero or near
zero for a share of units.

- The fold-evaluated learned-rule fit, `CrossFitting(fold_evaluation=True)`, now accepts
  `targeting_scheme="fold"`, known observation weights (`weights=`) and clusters (`id=`). At
  n = 2,000 its study measured coverage of 0.9427 under fold targeting, 0.9537 with known weights
  and 0.9445 at 200 equal clusters of 10. Unequal cluster sizes, and weights with clusters, read
  experimental. Estimated weights (`weights_estimated=True`) refuse.
  ([#529](https://github.com/esbraun/cleverly-tmle/pull/529))
- `CrossFitting(repeats=R)` with `R` above 1 reports the mean of the `R` split estimates, each
  split with its own fold rules, and not their median. The variance reads the mean of the split
  curves. An unweighted fit with pooled targeting and no `id=` reads supported from a study of
  five splits of ten folds at n = 2,000. `repeats` above 1 with `id=` refuses, and
  `freeze_learned_rule`, the support report and `truncation_curve` refuse a fit of several
  splits. ([#530](https://github.com/esbraun/cleverly-tmle/pull/530))
- `LearnedRule(loss="blip", learner=...)` and `LearnedRule(loss="weighted_classification",
  learner=...)` learn each fold's rule from a doubly robust pseudo-outcome (Luedtke and van der
  Laan 2016) instead of the outcome regression. Both compare two arms. An unweighted fit on rows,
  with pooled targeting, `repeats=1` and no `id=`, reads supported. Its study calibrates the
  interval at n = 4,000. At n = 1,000 and 2,000 the standard error was 3.5% to 5% too small.
  ([#531](https://github.com/esbraun/cleverly-tmle/pull/531))
- `strata=` on a learned-rule fit also reports the value inside each baseline stratum, with one
  rule per fold read stratum by stratum. Under fold evaluation, the plug-in rule with
  `repeats=1`, pointwise intervals, and rows or known weights reads supported from a study at
  n = 1,000 with five folds and three strata. `strata=` with `id=` refuses.
  ([#535](https://github.com/esbraun/cleverly-tmle/pull/535))
- The default `CrossFitting()`, the stacked report, now runs for `LearnedRuleValue`, and 0.1.7
  refused it. Its target weights each fold by its share of the rows, or of the weight mass under
  `weights=`. The plug-in rule with pooled targeting and `repeats=1` reads supported on rows, with
  known weights, and with `id=` at 40 clusters or more. Over 8,800 samples of each setting at
  n = 2,000, the 95% interval covered 0.943 to 0.949 of the time.
  ([#543](https://github.com/esbraun/cleverly-tmle/pull/543))
- `CrossFitting(enabled=False)` learns one rule on all rows and estimates the value of that rule
  on the same rows, a different target from the cross-fitted value. Its interval needs models of
  fixed form. `LogisticRegression`, `LinearRegression`, `GLM` and a class that sets
  `cleverly_learner_family = "parametric"` count as fixed, and any other learner raises the new
  `InSampleLearnerWarning`. With fixed-form models, the plug-in rule, a binary outcome and
  treatment, and no weights, clusters or strata, the fit reads supported. Its point leaned
  optimistic by 0.16 to 0.21 standard errors at n = 2,000.
  ([#545](https://github.com/esbraun/cleverly-tmle/pull/545))
- `TMLE(learned_rule=LearnedRule(), interventions=[...])` reports the learned-rule value, each
  known regime's value (`ey_known_regime`) and their differences (`ate_learned_rule`). The
  interval of a difference uses the correlation of the two estimates on the same rows. With two
  arms, the plug-in rule, pooled targeting and `repeats=1` on rows, the three rows read supported
  under either report. `simultaneous_bands()` refuses these rows.
  ([#546](https://github.com/esbraun/cleverly-tmle/pull/546))
- A learned rule on a treatment with three or more arms gives each unit the arm with the highest
  predicted outcome, and a tie goes to the lowest arm code. Three arms under fold evaluation read
  supported. In its study, at n = 2,000 with ten folds, the 95% interval covered 0.9425 of the
  time over 3,200 samples. Four or more arms read experimental, and the rule learners refuse
  more than two arms. ([#547](https://github.com/esbraun/cleverly-tmle/pull/547))
- `TMLE(learned_rule=LearnedRule()).fit(..., intermediate=...)` returns one result per value z of
  a binary intermediate. Each result estimates the mean outcome had every unit followed the rule
  learned for that value, with the intermediate set to z. That rule is not the best joint choice
  of treatment and intermediate. The plug-in rule on two arms under fold evaluation on rows reads
  supported at each value, and the stacked report reads experimental. `CausalStudy` refuses this
  request.
  ([#549](https://github.com/esbraun/cleverly-tmle/pull/549))

### Value of the optimal rule

- `OptimalRuleValue` estimates the mean outcome had every unit received the arm with the best mean
  outcome given its covariates, a fixed parameter. It uses the same fit as `LearnedRuleValue`
  and reports `ey_optimal_rule`. The estimate targets this value only when the outcome model, or
  the rule learner, can learn the true pattern of benefit, the effect is zero for almost no unit,
  and the learned rules approach the optimal rule fast enough (van der Laan and Luedtke 2015,
  Section 7.2). The fold-evaluated fit on rows with known treatment probabilities
  (`treatment_probabilities=`) reads supported, and every other fit reads experimental. The fit
  refuses `weights=`, `strata=`, `intermediate=`, a known regime beside it and the in-sample fit.
  ([#548](https://github.com/esbraun/cleverly-tmle/pull/548))

### Sensitivity of a learned-rule value

- `result.sensitivity.omitted_confounding()` gives a point bound for a learned-rule fit of one
  split under fold evaluation, with each fold's rule held fixed. Its limits and `benchmark()`
  refuse, and no study measures the bound. On the in-sample fit, the bound also gives its
  limits. The bound refuses a fit with strata, several splits, the stacked report, known-regime
  contrasts, more than two arms or an intermediate.
  ([#532](https://github.com/esbraun/cleverly-tmle/pull/532),
  [#545](https://github.com/esbraun/cleverly-tmle/pull/545))

### Fold-evaluated CV-TMLE with baseline strata

- `CrossFitting(fold_evaluation=True)` with `strata=` now runs for the arm means and the ATE
  (`ey`, `ate`) and for the regime means and their contrasts, and 0.1.7 refused it. Each stratum
  estimate averages the stratum plug-ins of the validation folds, and the marginal estimate mixes
  them by the stratum shares. The arm means and the ATE with `repeats=1` and pointwise intervals
  read supported from a study at n = 1,000 with five folds and three strata. The regime targets,
  `repeats` above 1 and the simultaneous band read experimental. Other targets, missing outcomes,
  `intermediate=` and `id=` refuse. ([#533](https://github.com/esbraun/cleverly-tmle/pull/533))

### Comparator evidence

- These studies are now paired with their R comparators on the same draws: the repeated-outcome
  `LTMLE` study with `ltmle` 1.3-0, the vector-treatment policy study with `lmtp` 1.5.4, the
  point-treatment survival node-bounds study with `survtmle` 1.1.1, the point survival-curve
  study with `tmle3` and `survtmle`, and `AdaptiveIPTW` with `drtmle` 1.1.2 `adaptive_iptw`. No
  shipped estimate changes. ([#536](https://github.com/esbraun/cleverly-tmle/pull/536),
  [#538](https://github.com/esbraun/cleverly-tmle/pull/538),
  [#540](https://github.com/esbraun/cleverly-tmle/pull/540),
  [#541](https://github.com/esbraun/cleverly-tmle/pull/541),
  [#550](https://github.com/esbraun/cleverly-tmle/pull/550))

## Changed

- A point-treatment policy fit with a cap tuned on the data now reads experimental, because no
  point study fits a tuned cap. A longitudinal fit with a tuned cap reads supported only inside
  the scope of the `mtp-tuning` study, and a vector fit with a tuned cap reads experimental.
  ([#527](https://github.com/esbraun/cleverly-tmle/pull/527))

## Fixed

- The summary note of a fit below the cluster floor now names that fit's floor, 20 clusters for a
  longitudinal fit. Earlier versions named 10, the point-treatment floor, for every fit.
  ([#527](https://github.com/esbraun/cleverly-tmle/pull/527))

## Packaging and compatibility

The version series moves from `0.1.N` to `0.2.N`, and release tags now take the form `v0.2.N`.
The package does not promise compatibility for public APIs, `CausalStudy` objects, pickled
objects, or saved results. Follow the upgrade notes when you move an analysis to 0.2.0.

**Full changelog:** [v0.1.7...v0.2.0](https://github.com/esbraun/cleverly-tmle/compare/v0.1.7...v0.2.0)
