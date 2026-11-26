# Implementation validation studies

Each page here reports one study of a method that `cleverly` implements. A page exists only for
a study registered in `tests/studies/evidence/registry.py`. The shared machinery in
`tests/studies/evidence/` computes every verdict, and `tests/unit/test_method_evidence.py` checks
these pages against the committed results. Every table is generated from those results, so a stale
number is a test failure and not a reading error.

Read the pages in this order.

| page | what it gives you |
| --- | --- |
| [Implementation validation grid](validation-grid.md) | every validation study in one table, with the counts and the declared limits |
| [How to read these studies](how-to-read.md) | the three questions, the verdict rules, the comparators declared and not yet run, and the terms every study below applies |
| [The red-cell ledger](red-cells.md) | every red verdict the studies publish, and the roadmap ask that owns each one |
| the study pages | one row per committed test, with what it checked and the verdict its own endpoints produced |

A validation study declares property cells or names a reference. A study that names a reference
is comparator-backed, and its agreement block names the comparison kind of each row. These
studies declare neither, so each has a page and no grid row:

<!-- generated: off-grid-studies -->
- [CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws](learned-rule-cvtmle-boundary.md): no property cell and no comparator in its record; published under the `reporting` policy.
- [collaborative TMLE of a risk curve](survival-ctmle.md): no property cell and no comparator in its record; published under the `reporting` policy.
- [continuous-time survival switched variance](continuous-survival-variance.md): no property cell and no comparator in its record; published under the `gated` policy; the reason is recorded in `GATED_OFF_GRID` in `tests/unit/test_method_evidence.py`.
- [Adaptive TMLE with external data](adaptive-tmle-trial.md): no property cell and no comparator in its record; published under the `gated` policy; the reason is recorded in `GATED_OFF_GRID` in `tests/unit/test_method_evidence.py`.
- [Adaptive TMLE under selective enrollment](adaptive-tmle-selective.md): no property cell and no comparator in its record; published under the `gated` policy; the reason is recorded in `GATED_OFF_GRID` in `tests/unit/test_method_evidence.py`.
<!-- /generated -->

The study pages are grouped by method family below, and the grid's `family` column uses the same
names. A study of a cross-cutting option sits with that option, a time-to-event study sits under
survival whatever its estimator, and a conditional-effect, rule or score target sits with the
learned rules. To register a new study and place its page, follow
[adding a method row](../../development/method-benchmarking.md#adding-a-method-row).

```{toctree}
:maxdepth: 2

validation-grid
how-to-read
red-cells
```

```{toctree}
:caption: Point-treatment TMLE
:maxdepth: 1

canonical-point-treatment-tmle
controlled-direct-effect-tmle
deterministic-point-treatment-regimes
known-mechanism-sample-weights
known-score-att
known-treatment-mechanism
learned-weighted-point-treatment-tmle
missing-outcome-attributable-effects-tmle
ordinary-missing-outcome-natural-course-tmle
ordinary-missing-outcome-tmle
ordinary-multi-arm-tmle
point-treatment-msm-projection
stratified-msm-identity
stratified-point-treatment-tmle
weighted-point-treatment-tmle
```

```{toctree}
:caption: Cross-fitting and CV-TMLE
:maxdepth: 1

fold-evaluated-point-treatment-cv-tmle
fold-evaluated-strata
fold-targeted-point-treatment-cv-tmle
repeated-cross-fitted-longitudinal
repeated-cross-fitting
stacked-arm-indexed-missing-outcome-cvtmle
stacked-missing-outcome-natural-course-cvtmle
stacked-point-treatment-cv-tmle
```

```{toctree}
:caption: Clustered data
:maxdepth: 1

clustered-cross-fitted-longitudinal-tmle
clustered-few-cluster-tmle
clustered-point-treatment-cv-tmle
clustered-unequal-cv-tmle
few-cluster-adaptive-learners
few-cluster-cross-fitted-longitudinal-tmle
few-cluster-floor-cross-fitted-ltmle
few-cluster-floor-tmle
few-cluster-fold-variance
```

```{toctree}
:caption: Collaborative TMLE
:maxdepth: 1

c-tmle-candidate-sequences
ctmle-derivative
outcome-adaptive-multi-arm-c-tmle
outcome-adaptive-per-arm-c-tmle
outcome-adaptive-per-arm-c-tmle-parametric
outcome-adaptive-per-arm-c-tmle-parametric-crossfit
outcome-adaptive-per-arm-c-tmle-parametric-extensions
outcome-adaptive-per-arm-c-tmle-strata-missing
outcome-adaptive-point-treatment-c-tmle
selector-based-multi-arm-c-tmle
selector-based-point-treatment-c-tmle
```

```{toctree}
:caption: DR-TMLE
:maxdepth: 1

canonical-dr-tmle
dr-tmle-foldwise
dr-tmle-foldwise-n1500
dr-tmle-one-inconsistent-nuisance
known-mechanism-dr-tmle
multi-arm-dr-tmle
multi-arm-dr-tmle-foldwise
observational-missing-data-dr-tmle
randomized-missing-outcome-dr-tmle
randomized-multi-arm-missing-outcome-dr-tmle
stratified-dr-tmle
stratified-dr-tmle-foldwise
```

```{toctree}
:caption: Adaptive IPTW and adaptive TMLE
:maxdepth: 1

adaptive-iptw
adaptive-iptw-pairing
adaptive-tmle-selective
adaptive-tmle-trial
```

```{toctree}
:caption: Policies, incremental interventions and dose-response
:maxdepth: 1

categorical-incremental
continuous-modified-treatment-policies
dose-response
dose-response-npcausal
incremental-propensity-interventions
longitudinal-incremental-interventions
longitudinal-incremental-interventions-dropout
longitudinal-modified-treatment-policies
mtp-boundaries
mtp-density-tails
mtp-policy-variance
mtp-pricing
mtp-pricing-censored
mtp-tuning
point-modified-treatment-policies
stochastic-point-treatment-regimes
stratified-incremental-msm
vector-modified-treatment-policies
```

```{toctree}
:caption: Learned rules, conditional effects and scores
:maxdepth: 1

conditional-projection
ep-assessment
ep-cate
ep-crr
learned-regimen-ltmle
learned-rule-arms
learned-rule-compositions
learned-rule-contrast
learned-rule-contrast-comparators
learned-rule-cvtmle
learned-rule-cvtmle-boundary
learned-rule-ep
learned-rule-ep-declared
learned-rule-in-sample
learned-rule-intermediate
learned-rule-learners
learned-rule-optimal-compositions
learned-rule-optimal-value
learned-rule-split-mean
learned-rule-stacked
learned-rule-strata
projection-compositions
projection-evidence
projection-methods
score-assessment
score-assessment-compositions
targeting-curves
```

```{toctree}
:caption: Longitudinal TMLE, SDR and baselines
:maxdepth: 1

boundary-node-ltmle
cross-fitted-categorical-longitudinal-tmle
cross-fitted-end-of-study-longitudinal-tmle
cross-fitted-longitudinal-msm-projection
cross-fitted-weighted-end-of-study-longitudinal-tmle
known-node-mechanisms-ltmle
long-horizon-categorical-longitudinal-tmle
long-horizon-longitudinal-tmle
long-horizon-sequentially-doubly-robust-estimation
ordinary-categorical-longitudinal-tmle
ordinary-end-of-study-longitudinal-tmle
ordinary-longitudinal-msm-projection
ordinary-weighted-end-of-study-longitudinal-tmle
parametric-gformula
repeated-outcome-ipsi
repeated-outcome-longitudinal-tmle
repeated-outcome-sdr
sequential-baselines
sequentially-doubly-robust-estimation
stochastic-categorical-longitudinal-tmle
structural-nested-mean
```

```{toctree}
:caption: Survival and competing risks
:maxdepth: 1

continuous-survival-causes
continuous-survival-functionals
continuous-survival-variance
continuous-time-survival
cross-fitted-competing-risk-longitudinal-tmle
cross-fitted-point-treatment-survival
cross-fitted-survival-curve-longitudinal-tmle
eliminated-competing-events
hazard-msm
hazard-mtp
hazard-multinode
hazard-time-varying
hazard-time-varying-multinode
ordinary-competing-risk-longitudinal-tmle
ordinary-survival-curve-longitudinal-tmle
point-survival-curve
point-survival-curve-crossfit
point-survival-hazard
point-treatment-survival
point-treatment-survival-policies
separable-effects
survival-ctmle
survival-dr-tmle
survival-node-bounds
```

```{toctree}
:caption: Targeting and fitting options
:maxdepth: 1

isotonic-binding
option-markov-order
option-outcome-by-arm
option-patience
option-ratio-trim
option-stopping-ltmle
option-stopping-point
option-treatment-adjustment
```

```{toctree}
:caption: Intervals, bands and the bootstrap
:maxdepth: 1

asymmetric-wald-intervals
band-t-reference
band-t-reference-held-out
default-simultaneous-bands
full-refit-bootstrap-and-derived-contrasts
small-sample-wald-intervals
```

```{toctree}
:caption: Sensitivity, refutation and diagnostics
:maxdepth: 1

calibration-slope-warning
evalue-limit-coverage
marginal-bound-limit-coverage
missingness-tilt
omitted-variable-bound-plugin-standard-error
omitted-variable-bound-standard-error
ovb-limit-coverage
refuter-calibration-band
refuter-calibration-continuous-survival
refuter-calibration-longitudinal
refuter-calibration-point
refuter-sampled
```
