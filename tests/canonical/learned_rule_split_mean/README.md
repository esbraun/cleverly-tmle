# Learned-rule value, mean over splits

This directory holds the gated repeated-sampling study of X11 part (i). The subject is
`TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, g_bounds="auto", repeats=5)`.
The fit draws five independent ten-fold splits, learns each split's fold rules on its training
rows, and reports the mean of the five split estimates. Its variance is the second moment of the
mean of the five split curves. The target is the mean over the five splits of each split's
fold-average value of its rules.

| part | declaration |
| --- | --- |
| scenarios | RM30 `non_exceptional` and RM30 `misspecified_limit`, rows only (no weights, no clusters), at n = 2,000 |
| truth | each replication refits, for each split, each fold's outcome learner on the split's training rows, requires the split's rule on every validation row, integrates the value of each rule on 4,001 points of `W1` for each `W2`, and averages over the folds and then over the five splits |
| properties | `interval_calibration` for each law with the shrunken-SE and noise controls, and the independent-splits control: the fit's estimate with the mean single-split SE over the square root of 5, which must fall below the SE-ratio band. The median of the split estimates is a descriptive column that no verdict reads |
| verdicts | the framework defaults, read on the error `estimate - truth` of each row |
| comparator | none |

The declarations are in `tests/studies/learned_rule_split_mean.py`,
`tests/studies/learned_rule_split_mean_properties.py` and
`tests/studies/_learned_rule_split_mean_law.py`. The run form is the RM30 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library:

```bash
python -u -m tests.canonical.learned_rule_split_mean.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository:

```bash
python -u -m tests.canonical.learned_rule_split_mean.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch>
```
