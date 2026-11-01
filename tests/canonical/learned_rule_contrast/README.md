# Learned-rule contrasts with known regimes

This directory holds the gated repeated-sampling study of X11 part (c) (X11 PR G). The subject
is `TMLE(learned_rule=LearnedRule(), interventions=[Static(1), Static(0)], n_folds=10,
targeting_scheme="pooled", g_bounds="auto")` with RM30's interaction logistic outcome learner and
logistic treatment learner. One fit reports five parameters: the learned-rule value
`ey_learned_rule[learned rule]`, the two known-regime values `ey_known_regime[always 1]` and
`ey_known_regime[always 0]`, and the two contrasts `ate_learned_rule[learned rule vs always 1]`
and `ate_learned_rule[learned rule vs always 0]`. The learned value's target is the
data-adaptive fold average of `Psi_{d_v}(P0)` (van der Laan and Luedtke 2015); a known
value's is `E_0 Qbar0(a, W)`; a contrast's is their difference, with curve `D_L - D_K` on the
same rows. Both reports are read on the same draws: the fold-evaluated one (`cv_evaluation=True`)
and the stacked one (`cv_evaluation=False`, X11 (e)).

| part | declaration |
| --- | --- |
| learners | outcome: RM30's `Pipeline(PolynomialFeatures(2, interaction_only=True), LogisticRegression(C=1e6))` on `(A, W1, W2)`; treatment: RM30's logistic regression on `(W1, W2)`, a correct model fitted by maximum likelihood |
| laws | RM30 `non_exceptional` (correctly specified; calibration bands) and `misspecified_limit` (the outcome model cannot represent the blip; bias leg and coverage floor only), one size n = 2,000 |
| primary | 1,200 replications of `non_exceptional`, fold-evaluated report, all five parameters. The bias legs would pass at 800 with predictive probability 0.999; the study's floor of 1,000 draws for a gated cell sets 1,200 |
| truth | per replication: refit each fold's outcome learner on the fit's training rows, require the fit's rule on every validation row (rule L4), integrate each fold rule's value on 4,001 points of `W1` for each `W2` with the known `Qbar0`, and weight fold `v` by the report's weight (`1/V` fold-evaluated, `n_v/n` stacked), checked against the record's `fold_weights`. A known regime's truth is `E Qbar0(a, W)`; a contrast's is the difference. `oracle_se` is descriptive |
| `interval_calibration` | `non_exceptional`, both reports, all five parameters: 10 positives, each with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit, the per-parameter bounds in `TAG_BOUNDS`). 5,600 draws: the arm-pooled rule on the five fold-evaluated positives of a 4,000-draw pilot (family-wise 0.951). Each draw fits both reports |
| `conservative_coverage` | `misspecified_limit`, both reports, all five parameters. A positive passes on its bias leg and on the lower bound of its exact 99% coverage interval at or above 0.92; its SE ratio is published and no verdict reads it. The control is the same estimate with its SE times 0.60, and its coverage interval must lie wholly below 0.92. 3,200 draws: every cell passes with predictive probability 0.964 on the pilot |
| `contrast_covariance` | every contrast of both laws and reports with SE `abs(se_L - se_K)`, the SE if the two values had correlation one. Its coverage interval must lie wholly below 0.80. The pilot read 0.05 to 0.09 |
| `independent_se_diagnostic` | role `diagnostic`, no verdict: every contrast with SE `sqrt(se_L^2 + se_K^2)`, the SE if the two values were independent. The pilot read SE ratios of 1.36 and 1.49 and coverage above 0.99 |
| `targeting_necessity` | RM30's `drop_w1_learner` on `non_exceptional`, fold-evaluated, contrast against `Static(0)`, 1,200 draws (the floor). The positive arm is the targeted contrast; the control is the untargeted difference of the initial plug-ins, which must sit 0.5 positive-arm SDs or more from it. The contrast against `Static(1)` is not read: that learner's limit rule treats every row |
| verdicts | the framework defaults on the error `estimate - truth` of each row, and the two families' own rules in `tests/studies/learned_rule_contrast_properties.py` |
| comparator | none: no external package fits a cross-validated TMLE of a learned rule with known regimes |

## The support rows

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_rule_contrast.py`). The
registering change adds two P11 rows, each with the qualifier `known_regimes`, covering all
three parameter names a contrast fit reports.

- The `fold_evaluated` row is added if and only if these verdicts pass as declared: the primary
  performance tests of all five parameters (bias leg, coverage floor, SE sanity), and every
  `interval_calibration`, `conservative_coverage` and `contrast_covariance` verdict of the
  `fold_evaluated` labels, and the `targeting_necessity` verdicts.
- The `stacked` row is added if and only if the same verdicts pass with the `stacked` labels in
  place of the `fold_evaluated` ones, and the primary and necessity verdicts pass too. At ten
  equal folds without weights the stacked point is the fold-evaluated point, so those verdicts
  read the stacked point as well; only the variance rules differ.

The route for a red primary bias leg, a red positive cell, or a control that does not fail is
declared now. The cell is diagnosed before a route is chosen. With no defect found, it is
published red under `reporting` with an owner row, and the row of that report is not added.
Under `gated` a red primary gate still publishes the run. Contrast fits outside the rows stay
experimental.

## What registration must state

- The `misspecified_limit` cells claim coverage at or above the floor, not SE calibration. The
  reported curves omit the treatment model's projection term. Corollary 2 of van der Laan and
  Luedtke (2015) makes that variance conservative for a maximum-likelihood `g` in a correct
  model, and the paper extends the claim to the cross-validated TMLE by analogy only
  (`reviews/x11/X11-G-corollary3-reading.md` in the plans repository). The claim rests on that
  analogy. The pilot did not show the predicted over-coverage there: two SE ratios read 0.977.
  Quote each positive's realized SE ratio beside its coverage.
- The claim holds for a correctly specified treatment model fitted by maximum likelihood. A
  known `g` and a super-learner `g` are not measured; a super-learner `g` is report only.
- The stacked report was measured at ten equal folds without weights, where its point is the
  fold-evaluated one. Unequal folds and weighted fits under the stacked report are not measured.
- The necessity pair reads the contrast against `Static(0)` only; quote why.
- Intervals are pointwise; strata, weights, clusters and rule learners with known regimes are
  not measured here.

The declarations are in `tests/studies/learned_rule_contrast.py`,
`tests/studies/learned_rule_contrast_properties.py` and
`tests/studies/_learned_rule_contrast_law.py`. The run form is the X11 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. It draws 11,200 samples (1,200 primary, 5,600 calibration, 3,200 coverage and 1,200
necessity). The pilot read about 0.38 core-s per property draw (both reports and the refit
harness) and 0.29 core-s per necessity draw. That is about 4,000 core-s, under 20 minutes on 4
cores; allow an hour with the summaries and bootstraps:

```bash
python -u -m tests.canonical.learned_rule_contrast.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 3` also fits three draws of every property cell and prints the rule
agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_contrast.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
