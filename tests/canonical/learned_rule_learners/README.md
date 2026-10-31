# Learned-rule value, rule learners

This directory holds the gated repeated-sampling study of X11 part (g). The subject is
`TMLE(learned_rule=LearnedRule(loss=..., learner=...), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`
with `repeats=1`. Each fold's rule comes from a rule learner fitted on the fold's training rows
to the doubly robust blip pseudo-outcome, whose outcome and treatment nuisances are cross-fitted
in five inner folds of those rows. The target is the fold-average value of the rules the fit
learned.

| part | declaration |
| --- | --- |
| scenarios | `blip` (`PolynomialFeatures(2)` and `LinearRegression()`) and `weighted_classification` (`PolynomialFeatures(2)` and `LogisticRegression(C=1e6, max_iter=5000)`, sample weights the size of the pseudo-outcome), each on RM30 `misspecified_limit` with the main-terms outcome learner and the logistic treatment learner, rows only (no weights, no clusters). The primary phase fits n = 2,000; the calibration cells fit n = 4,000, and a diagnostic cell per loss fits n = 1,000. At n = 1,000 and 2,000 the error variance of a rule-learner fit exceeds the fixed-rule variance by about 7% to 13% (pilot SE ratios blip 0.946, 0.955, 1.001 and weighted classification 0.959, 0.971, 0.993 at n = 1,000, 2,000, 4,000), through the covariance between folds whose rules are fitted on each other's validation rows, a second-order remainder of relative order about 1/n that is larger for a pseudo-outcome learner than for the plug-in rule. The paired error-variance ratio of the rule learner to the plug-in rule on the same draws is about 1.05, 1.03 to 1.06 and 0.97 to 0.99 at the three sizes, so the excess declines with n and is within about 3% of zero at n = 4,000; the pilot does not pin a rate. `CALIBRATION_N` stays 4,000 whatever the result: a red cell is diagnosed under the red-cell table, a finite-sample diagnosis goes to `reporting` with an owner row, and the size never moves again. R = 7,200 per calibration cell (the pooled pilot would allow 4,400) |
| truth | each replication refits each fold's rule through `fit_fold_rule` with the fold's recorded inner seed and the fit's resolved `g_bounds`, requires the fit's rule on every validation row, integrates the value of each rule on 4,001 points of `W1` for each `W2`, and averages over the folds. The refit reproduces the record through the package's code; the construction rests on the unit witnesses W-g2 and W-g4 |
| properties | `interval_calibration` for each loss with the shrunken-SE and noise controls, and the fold-locality control `validation_rule_control`: the same sample and fold seed with each fold's rule learner fitted on that fold's validation rows (`ValidationRowRule`). Its estimate is optimistic for the value of its own rules, and its coverage interval must lie below 0.92. The control resolving below the band shows that the instrument sees a gross fold-locality violation; the subtle hazard of pseudo-outcome nuisances read from the outer cross-fitted predictions at the rows of the validation fold is invisible to any calibration cell and rests on the unit witnesses W-g3 and W-g4. `small_sample_diagnostic` fits each loss at n = 1,000 for 2,000 replications from its own sample and fold streams (role `diagnostic`, no control, no verdict) and publishes the SE ratio with its bootstrap interval (about 0.04 on each side at 99%), the coverage and the bias with their intervals. Every row of the primary phase, the calibration positives, the controls and the diagnostic cells also carries the plug-in fit of the same sample and folds (`plug_in_estimate`, `plug_in_std_error`, `plug_in_truth`); the value gain over the plug-in rule is descriptive and no verdict reads it. The claims vocabulary reads the primary value gain from `harness.csv.gz` (`value_gain_mean_<scenario>`, `value_gain_sd_<scenario>`, `value_gain_positive_share_<scenario>`) |
| verdicts | the framework defaults, read on the error `estimate - truth` of each row |
| comparator | none |

The declarations are in `tests/studies/learned_rule_learners.py`,
`tests/studies/learned_rule_learners_properties.py` and
`tests/studies/_learned_rule_learners_law.py`. The run form is the RM30 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library:

```bash
python -u -m tests.canonical.learned_rule_learners.regenerate --jobs 4 --output <empty scratch directory>
```

Cost: 4,000 primary fits at n = 2,000 (about 1.25 CPU-s each), 14,400 calibration fits at n = 4,000 (about 1.71 CPU-s each) and 4,000 diagnostic fits at n = 1,000 (about 1.2 CPU-s each), about 2.7 hours (3.5 allowed) on 4 cores with the summaries and bootstraps.

A smoke run passes another `--replicates` and writes outside the repository. `--property-harness-check 3` also fits three draws of every property cell, the diagnostic cells included, and prints the rule agreement and solver warnings:

```bash
python -u -m tests.canonical.learned_rule_learners.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 3
```
