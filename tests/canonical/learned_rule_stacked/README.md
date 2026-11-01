# Stacked report of the learned-rule value

This directory holds the gated repeated-sampling study of X11 part (e) (X11 PR F1). The subject
is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=False, targeting_scheme="pooled", g_bounds="auto")`
with `repeats=1`. One pooled fluctuation runs over every validation row with the observation
weights. The fit reports `sum_v (W_v / W) psi*_v`, where `W_v` is fold `v`'s weight mass (its row
count without weights), and the target of each replication is `sum_v (W_v / W) Psi_{d_v}(P0)`,
the value of the rules the fit learned, weighted the same way.

| part | declaration |
| --- | --- |
| scenarios | `rows`: RM30 `non_exceptional`, n = 2,000, ten equal folds, so `W_v / W = 1/V` and the cell measures the pooled centring. `weighted`: the `learned-rule-compositions` weighted law (`misspecified_limit` blip, `w = 1 + 1.5 * 1{W1 > 0}`), n = 2,000, unequal fold masses. `clustered`: the `non_exceptional` logit plus `1.0 z_j` per cluster, size `5 + Binomial(10, expit(1.5 z_j))`, 200 clusters, about 2,000 rows; rows publish the nominal n and a `rows` column |
| why the cluster law | the size rises with the cluster's outcome effect (size-curve correlation 0.891, curve ICC 0.077 under the row law). From the law alone, before any run, the fold-evaluated point carries a ratio bias of -0.120 stacked SDs and the stacked point about -0.01 (`ratio_bias_oracle`, the method of `tests/unit/test_fold_evaluated_ratio_bias.py`) |
| stored variance | `rows`, `weighted`: `{n (n - 1)}^-1 sum_i (phi_i - phibar)^2` with `phi_i = w_i (Z_i - psi)` on mean-one weights and `phibar = 0` to solver tolerance. `clustered`: `J var(T_j; ddof=1) / n^2` of the cluster totals, normal reference (the t rule applies below 40 clusters). The interval is `psi +/- 1.96 SE` |
| truth | per replication: refit each fold's outcome learner on the fit's training rows with its weights and clusters, require the fit's rule on every validation row (rule L4), integrate each rule's value on 4,001 points of `W1` for each `W2` (and 64 Hermite nodes of the size-biased cluster effect), and weight fold `v` by `W_v / W` computed from the fit's data, checked against the record's `fold_weights` |
| `interval_calibration` | one positive per scenario, 8,800 draws each (the per-arm rule on a 4,000-draw pilot at the declared size), with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit); `clustered__row_variance_control`, the same estimate with the row-level centred SE of the same curve (limit ratio 0.771) |
| `fold_evaluated_reporting` | role `diagnostic`, no verdict: the fold-evaluated fit (`cv_evaluation=True`) of every clustered calibration draw, same fold seed, so the same folds and rules (checked). Its truth weights the same fold values by `1/V`. The summary publishes each report's bias and the paired error difference |
| diagnostics | `cross_term` and `between_term`: the two extra terms of the stacked variance identity over the fold-evaluated second moment, on every positive and primary row; descriptive |
| verdicts | the framework defaults on the error `estimate - truth` of each row |
| comparator | none: R `polle`'s Algorithm 4 is the construction's source, not a pinned comparator of the learned-rule value |

## The `unequal_clusters` rule

Declared before any run (`UNEQUAL_CLUSTERS_RULE` in `tests/studies/learned_rule_stacked.py`). The
registering change drops the `unequal_clusters` qualifier from fits keyed `stacked` if and only
if `clustered__declared` passes the calibration rule, establishes its bias inside the margin, all
three of its controls fail as declared, no more clustered replications were lost than the
failure bound allows, and the reporting row's paired error difference (fold-evaluated minus
stacked) is negative with its upper 99% bound below zero. Otherwise the qualifier stays. A
dropped qualifier claims only the measured scope: 200 clusters in 10 folds, sizes 5 to 15
rising with the outcome effect. The fold-evaluated report keeps it either way, by theory: its
ratio bias is about -0.12 SD at 200 clusters and -0.29 SD at 40 by the law's oracle.

## What registration must state

- The stacked variance is not guaranteed to be at least the fold-evaluated second moment,
  because the cross term of the variance identity can be negative. Quote the realized mean
  `cross_term` and `between_term`; on the pilot they move the SE by under 0.1%, far below the
  study's resolution, so no calibration reading is attributed to them.
- The calibration verdicts cannot tell the `W_v / W` target from the fold average in these
  cells; the unit witnesses and the harness's fold-weight check establish the weighting.
- The fold-evaluated report keeps `unequal_clusters` by theory (its ratio bias grows as
  `V / sqrt(J)` in SDs), not by a failed verdict: at 200 clusters it would pass the same bands.
- The target is data-adaptive and weights each fold by its share of the rows (of the weight
  mass under weights), so at unequal folds it is not the fold-evaluated target.
- Intervals are pointwise; strata, fold targeting, rule learners and `repeats > 1` under the
  stacked report are not measured here.

The declarations are in `tests/studies/learned_rule_stacked.py`,
`tests/studies/learned_rule_stacked_properties.py` and
`tests/studies/_learned_rule_stacked_law.py`. The run form is the X11 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library. It
draws 32,400 samples (6,000 primary and 26,400 calibration); each is one fit and its harness,
and each clustered calibration draw also fits the fold-evaluated report. The pilot read about
0.21 core-s per rows draw, 0.23 per weighted draw and 0.9 per clustered pair: about 13,600
core-s, an hour on 4 cores; allow three hours with the summaries and bootstraps:

```bash
python -u -m tests.canonical.learned_rule_stacked.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository. `--property-harness-check 3` also fits three draws of every property cell and prints the rule agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_stacked.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 3
```
