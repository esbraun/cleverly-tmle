# Learned-rule value with fold targeting, weights and clusters

This directory holds the gated repeated-sampling study of X11 parts (h), (k) and (l). The subject
is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, g_bounds="auto")` with one of
three settings: `targeting_scheme="fold"`, known observation weights `weights=`, or clusters
`id=`. The target is the average, over the ten outer folds, of the value of the plug-in rule that
each fold learns on its training rows, in the weighted population under `weights=`.

| part | declaration |
| --- | --- |
| scenarios | `fold_targeted` (RM30 `non_exceptional`, fold targeting), `weighted` (RM30 `misspecified_limit`, weight `1 + 1.5 * 1{W1 > 0}`), `clustered` (RM30 `non_exceptional` plus a cluster effect of SD 1.5 on the logit scale, 200 clusters of 10), at n = 2,000 |
| truth | each replication refits each fold's outcome learner on the fit's training rows with its weights, requires the fit's rule on every validation row, and integrates the weighted value of the rules on 4,001 points of `W1` for each `W2` |
| properties | `interval_calibration` for each scenario with the shrunken-SE and noise controls, the clustered row-variance control, and a few-cluster cell (30 clusters of 34); `targeting_necessity` on the fold-targeted fit with an outcome learner that drops `W1` |
| verdicts | the framework defaults, read on the error `estimate - truth` of each row |
| comparator | none |

The declarations are in `tests/studies/learned_rule_compositions.py`,
`tests/studies/learned_rule_compositions_properties.py` and
`tests/studies/_learned_rule_compositions_law.py`. The run form is the RM30 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library:

```bash
python -u -m tests.canonical.learned_rule_compositions.regenerate --jobs 16 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository:

```bash
python -u -m tests.canonical.learned_rule_compositions.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch>
```
