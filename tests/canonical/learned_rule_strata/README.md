# Learned-rule value inside baseline strata

This directory holds the gated repeated-sampling study of X11 part (m) (X11 PR E2, roadmap X28's
learned-rule cell). The subject is
`TMLE(learned_rule=LearnedRule(), cross_fit=True, n_folds=5, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto", simultaneous=False)`
fitted with `strata=["S"]`, rows as the unit, complete data and `repeats=1`, with
`q_bounds=(0, 1)` on a proportion outcome and no `q_bounds` on a binary one. Fold `v`'s plug-in
rule `d_v` comes from its training rows. Each stratum estimate is the fold average of the
stratum plug-ins at `d_v`, each fold's stratum curve divides by the fold's own stratum share, and
the marginal estimate is the share mixture of the stratum estimates. Each fit reports four
parameters: `ey_learned_rule[learned rule]` in strata 0, 1 and 2 and marginally.

| part | declaration |
| --- | --- |
| law | E1's (`tests/studies/_fold_evaluated_strata_law.py`, read only): `W1 ~ U(-1, 1)`, `W2 ~ Bernoulli(0.5)`, `logit g0 = 0.3 W1 - 0.2 W2`, `logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + a b(W1)`, `S` cut from `W1` at -0.6 and 0.1 (shares 0.2, 0.35, 0.45). Bounded: `Y ~ Beta(12 m, 12 (1 - m))`, `b = 0.1 + 1.5 W1`, rule boundary `W1 = -1/15` in stratum 1. Binary: `Y ~ Bernoulli(m)`, `b = -1.2 + 3.0 W1`, boundary `W1 = 0.4` in stratum 2 |
| why this binary blip | E1's binary blip `0.1 - 0.3 W1` (slope 0.3) and then `-0.6 + 1.5 W1` were piloted first. The fold rules disagreed on 45% and 26% of the boundary stratum's rows, and the boundary stratum read SE ratios 0.88 to 0.89 and 0.94 to 0.95: the weak-margin finite-sample limit of RM30's `weak_blip` law (F27), not the stratum construction. At slope 3.0 they disagree on 14% and every ratio is 0.97 to 1.02 |
| scenarios | `unweighted` and `weighted` (bounded; E1's quasibinomial pairwise-product outcome learner on `(A, W1, W2, S)`), `binary_steep` and `binary_steep_weighted` (binary; RM30's logistic learner on the same products). The weight is `w = 1 + 1.5 W2`. The treatment learner is logistic on `(W1, W2, S)`; every learner is correctly specified. n = 1,000 and 2,000 primary replications per scenario |
| truth | per replication: each fold's rule refit from its training rows with the fit's learner, task and scaled outcome, checked against the fit's rule on every validation row (rule L4) and the stored fold rule on every row; each stratum value by the trapezoid rule on 4,001 points of `W1` per stratum for each `W2`, weighted by `w(W2)`, averaged over the folds; the marginal truth is the share mixture |
| `interval_calibration` | one positive cell per setting and parameter (16), 14,800 draws per setting, each with the shrunken-SE and noise controls; `unnormalised_curve_control` on every stratum parameter (12); `share_dropped_control` on the marginal value of the two bounded settings (2) |
| `targeting_necessity` | the bounded rows-only fit with E1's `drop_w1_learner()` against its untargeted fold plug-in at its own rule, on the three stratum parameters, 8,000 draws; its truth is refit with that learner |
| verdicts | the framework defaults; the two construction controls are below-band SE controls |
| comparator | none: no pinned comparator reports a fold-averaged stratified learned-rule value |

## What each control shows

`unnormalised_curve_control` fails by construction: its SE ratio is about the stratum share,
0.2 to 0.45. It shows that the instrument resolves a gross normalisation error, not that the
fold-local share is the right one, which unit tests witness. `share_dropped_control` tests the
marginal derivation for the learned rule: its asymptotic ratio is 0.778 rows only and 0.773
weighted on the bounded outcome. On the binary outcome the ratio is 0.98, which the band cannot
resolve, so no binary label carries it. The targeting control's learner sees `A` and `W2` only,
so it learns a rule of `W2` alone and its plug-in is nearly the same in every stratum. The cell
shows that targeting repairs a stratum-blind initial fit at a learned rule, not a subtle
misspecification within a stratum.

## What registration must state

- Support rows over the outcome kind: `binary` from the binary cells and `continuous` (bounded)
  from the beta cells, each with `weighting="fixed"` rows, since each outcome has a weighted
  cell.
- Measured at n = 1,000 with 5 folds, three strata whose smallest share is 0.2, and blips whose
  slope at the rule boundary is 1.5 (bounded) and 3.0 (binary). With a weaker margin the fold
  rules disagree on much of the boundary stratum and the interval under-covers (pilot: 0.911 to
  0.933 at E1's binary blip), as RM30's `weak_blip` row does.
- The boundary stratum's target varies by replication; a stratum the rules treat alike has a
  fixed target.
- Intervals are pointwise (`simultaneous=False`); the band and `repeats > 1` are not measured.
- The two construction controls are counted separately, and the targeting control is described
  as stratum-blind.

The declarations are in `tests/studies/learned_rule_strata.py`,
`tests/studies/learned_rule_strata_properties.py` and `tests/studies/_learned_rule_strata_law.py`.
The run form is the X11 one in `tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library. It
draws 75,200 samples (8,000 primary, 59,200 calibration and 8,000 necessity), each one fit and
its harness, at about 0.10 core-s per bounded draw and 0.22 per binary draw (timed on a shared
machine). That is about 11,600 core-s, under an hour on 4 cores; allow three hours with the
summaries:

```bash
python -u -m tests.canonical.learned_rule_strata.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository. `--property-harness-check 3` also fits three draws of every property cell and prints the rule agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_strata.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 3
```
