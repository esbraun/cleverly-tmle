# Fold-evaluated CV-TMLE with baseline strata

This directory holds the gated repeated-sampling study of roadmap X28 (X11 PR E1). The subject is
`TMLE(cross_fit=True, n_folds=5, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto", q_bounds=(0, 1), estimands=("ate", "ey"), simultaneous=False)`
fitted with `strata=["S"]`, rows as the unit, complete data and `repeats=1`. Each stratum
estimate is the fold average of the stratum plug-ins, each fold's stratum curve divides by the
fold's own stratum share, and the marginal estimate is the share mixture of the stratum
estimates. Each fit reports 12 parameters: `ate`, `ey[0]` and `ey[1]` in strata 0, 1 and 2 and
marginally.

| part | declaration |
| --- | --- |
| law | the RM30 family with the blip `0.1 + 1.5 W1` and a proportion outcome: `W1 ~ U(-1, 1)`, `W2 ~ Bernoulli(0.5)`, `logit g0 = 0.3 W1 - 0.2 W2`, `Y ~ Beta(12 m, 12 (1 - m))` with mean `m = Qbar0(A, W)`, `logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + a (0.1 + 1.5 W1)`. `S` cuts `W1` at -0.6 and 0.1 (shares 0.2, 0.35, 0.45; stratum ATEs -0.220, -0.064, 0.184). The share term carries about 27% of the marginal `ate` variance |
| scenarios | `unweighted` (rows only) and `weighted` (known weight `w = 1 + 1.5 W2`, which varies inside every stratum), each at n = 1,000 with the pairwise-interaction quasibinomial outcome learner on `(A, W1, W2, S)` and the logistic treatment learner on `(W1, W2, S)`, both correctly specified. Primary phase 2,000 replications per scenario |
| truth | fixed: each stratum mean by the trapezoid rule on 4,001 points of `W1` per stratum for each `W2`, weighted by `w(W2)`; the marginal truth is the share mixture (checked against an adaptive integral to 1e-8) |
| properties | `interval_calibration`: one positive cell per setting and parameter (24), each with the shrunken-SE and noise controls, plus `unnormalised_curve_control` on every stratum parameter (the stratum curve without its `1 / P_{n,v}(S = s)` normalisation) and `share_dropped_control` on the marginal `ate` and `ey[1]` of each setting (the marginal curve without its share term; asymptotic SE ratio 0.854 and 0.656 to 0.660). `targeting_necessity`: the rows-only fit with RM30's `drop_w1_learner()` on the quasibinomial solver, which keeps `A` and `W2` only, against its untargeted fold plug-in, on the nine stratum parameters |
| verdicts | the framework defaults; the two construction controls are below-band SE controls |
| comparator | none: no pinned comparator reports a fold-averaged stratified estimate |

The declarations are in `tests/studies/fold_evaluated_strata.py`,
`tests/studies/fold_evaluated_strata_properties.py` and
`tests/studies/_fold_evaluated_strata_law.py`. The run form is the X11 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library:

```bash
python -u -m tests.canonical.fold_evaluated_strata.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository. `--property-harness-check 3` also fits three draws of every property cell and prints the solver warnings:

```bash
python -u -m tests.canonical.fold_evaluated_strata.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 3
```
