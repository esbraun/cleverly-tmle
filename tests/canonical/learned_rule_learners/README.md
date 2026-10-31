# Learned-rule value, rule learners

This directory holds the gated repeated-sampling study of X11 part (g). The subject is
`TMLE(learned_rule=LearnedRule(loss=..., learner=...), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`
with `repeats=1`. Each fold's rule comes from a rule learner fitted on the fold's training rows
to the doubly robust blip pseudo-outcome, whose outcome and treatment nuisances are cross-fitted
in five inner folds of those rows. The target is the fold-average value of the rules the fit
learned.

| part | declaration |
| --- | --- |
| scenarios | `blip` (`PolynomialFeatures(2)` and `LinearRegression()`) and `weighted_classification` (`PolynomialFeatures(2)` and `LogisticRegression(C=1e6, max_iter=5000)`, sample weights the size of the pseudo-outcome), each on RM30 `misspecified_limit` with the main-terms outcome learner and the logistic treatment learner, rows only (no weights, no clusters). The primary phase fits n = 2,000; the calibration cells fit n = 4,000, because at n = 2,000 the error spreads 3% to 5% more than the reported SE (the rule's own sampling variability), which leaves the SE-ratio band's lower end within the pilot's noise |
| truth | each replication refits each fold's rule through `fit_fold_rule` with the fold's recorded inner seed and the fit's resolved `g_bounds`, requires the fit's rule on every validation row, integrates the value of each rule on 4,001 points of `W1` for each `W2`, and averages over the folds. The refit reproduces the record through the package's code; the construction rests on the unit witnesses W-g2 and W-g4 |
| properties | `interval_calibration` for each loss with the shrunken-SE and noise controls, and the fold-locality control `validation_rule_control`: the same sample and fold seed with each fold's rule learner fitted on that fold's validation rows (`ValidationRowRule`). Its estimate is optimistic for the value of its own rules, and its coverage interval must lie below 0.92. Each row also carries the plug-in fit of the same sample and folds (`plug_in_estimate`, `plug_in_truth`); the value gain over the plug-in rule is descriptive and no verdict reads it |
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

A smoke run passes another `--replicates` and writes outside the repository:

```bash
python -u -m tests.canonical.learned_rule_learners.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch>
```
