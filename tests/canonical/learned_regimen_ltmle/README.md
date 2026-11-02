# Learned two-node regimens

This directory holds the gated repeated-sampling study of X11 part (b) (X11 PR K). The subject
is `LTMLE(LearnedRegimen(), n_folds=10)` on a censored two-node panel with a binary treatment at
each node and an end-of-study binary outcome. Each training fold `v` learns the rule pair
`d_nv = (d_nv1, d_nv2)` by backward plug-in Q-learning inside its own untargeted recursion, and
each validation row is evaluated under its own fold's pair. The fit reports
`ey_regimen[learned regimen]`, whose target is the data-adaptive
`psi~_0n = sum_v (n_v / n) Psi_{d_nv}(P_0)`. One pooled logistic fluctuation targets each node,
and the variance is the stacked curve centred at the pooled estimate.

| part | declaration |
| --- | --- |
| law | `W1 ~ U(-1, 1)`, `W2 ~ Bernoulli(0.5)`; `logit g1 = 0.8 W1 - 0.3 W2`; `logit P(C1 = 1) = 2.5 + 0.3 W1`; `L2 ~ N(0.5 W1 + 0.2 W2 - 0.8 A1, 1)`; `logit g2 = -0.2 + 0.3 L2 + 0.4 A1`; `logit P(C2 = 1) = 2.5 - 0.2 L2`; `logit Qbar0 = -0.2 + 0.6 W1 - 0.2 W2 + 0.3 L2 + a2 (-0.1 + 1.2 L2 - 0.6 a1) + a1 (0.3 + 1.2 W1)` |
| optimal pair | node 2 treats where `L2 > (0.1 + 0.6 a1) / 1.2` (34% of rows that follow node 1); node 1 is a threshold in `W1` for each `W2` (36% treated). The optimal value is 0.557923 and its efficiency bound 0.894934; the best static regimen is worth 0.4600 |
| margins | 5.6% of rows have a node-2 blip within 0.1 logit of zero, and 4.0% a node-1 value gap below 0.01 (the weak-margin diagnostic, disclosed, not gated) |
| truth | each row's stacked value of the fit's own fold rule pairs: each fold's Q-learning is refit with scikit-learn alone, both rules must equal the stored ones on every row (rule L4), and each pair is integrated by the trapezoid rule on 1,001 points of `W1` for each `W2` and by Gauss-Legendre over `L2`, split at the fold's node-2 threshold (`truth_varies_by_replicate`) |
| exactness gate | the harness rebuilds the stacked curve from the stored node steps and raises if its variance differs from the reported one by more than `1e-8` relatively |
| learners | node 2: `LogisticRegression(C=1e6)` per stratum `(A1, A2)` on `(W1, W2, L2)`, a correct model. Node 1 `declared`: a cubic polynomial in `(W1, W2)` by least squares. Treatment and censoring: main-terms logistic, correct at both nodes, in every cell |
| primary | `two_node`, 1,200 replications of the declared fit. It reads the bias leg, the coverage floor and the SE sanity band |
| `interval_calibration` | `two_node`, 4,000 draws: the positive, the shrunken-SE control (x0.80) and the noise control (one bound unit). Each control must fall below the SE-ratio band |
| `conservative_coverage` | `squared_w1` (node 1 by least squares on `(W1^2, W2)`, `g` correct), 2,400 draws. Its limit rule treats 50% at node 1 and is worth 0.5046, 0.053 below the optimum, with a margin. The positive is held to the bias leg and a coverage floor of 0.92, with no SE-ratio gate; the control (SE x 0.80) must cover wholly below 0.92 |
| `targeting_necessity` | `drop_w1` (node 1 on `W2` alone), 1,200 draws: the targeted value against the untargeted plug-in, the mean of the stitched node-1 initial values. The plug-in's limit bias is 0.0596, 2.65 SEs. The displacement must be at least 0.5 positive-arm SDs |
| `fold_locality` | the declared learners twice on one sample and fold seed, 1,200 draws: the fit against the same fit with each fold's rules learned on its validation rows (the `ValidationRowRule` pattern), each read against its own pairs' value. The control's bias must clear the margin, and the displacement must be at least 0.5 |
| verdicts | the framework defaults on the error `estimate - psi~_0n`, `calibration_rule` without an efficiency band, `learned-rule-contrast`'s coverage rule, and the two necessity rules |
| comparator | none: no external package fits a learned two-node regimen by cross-validated TMLE |

Every cell is at n = 2,000.

## The support row

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_regimen_ltmle.py`). The key
reads experimental now: regimen kind `learned`, cross-fitted, censoring, two binary nodes, an
end-of-study binary outcome, no weights, no clusters, every option at its default.

- Row `G47`, `long_row("G47", ["learned-regimen-ltmle"], regimens=("learned",),
  splitting=("cross_fitted",))`, licensing `point` and `wald`. The registering change adds it if
  and only if every declared verdict passes: the primary (bias leg, coverage floor, SE sanity),
  `interval_calibration` (positive and both controls), `conservative_coverage` (positive and
  control), `targeting_necessity` and `fold_locality`.
- The row would not cover a continuous outcome with `q_bounds=`, declared node bounds,
  `ratio_trim=`, a stopping rule or a known mechanism (admitted experimental and keyed apart),
  or a misspecified treatment or censoring model: no cell measures them. Every refused
  composition stays refused.

A red cell is diagnosed before a route is chosen. With no defect found, it is published red
under `reporting` with an owner row, and the row is not added.

## Sizing

Each count is the smallest multiple of 400, at least 1,200, whose predictive pass probability
reaches 0.95 on the pilot, with the pilot's error counted once. `test_learned_regimen_ltmle_design`
recomputes every count from the pinned pilot readings.

- Primary: 1,200.
- Calibration: 4,000, a known risk. The per-cell rule gives 9,200 on the first 4,000-draw pilot
  and on the 8,000 draws pooled with a probe. That is over the budget the probe rule fixed (75
  minutes of fitting at four jobs on the pilot timings), so the rule's fallback declares the
  largest multiple of 400 the other counts leave, 4,000. Its pooled predictive pass is 0.72.
- Coverage: 2,400, on the redesign pilot of `squared_w1`.
- Necessity: 1,200, the floor. Fold locality: 1,200, the floor.
- Controls: each lies wholly below the SE-ratio band at its count with probability above 0.999.

## What registration must state

- The interval targets the stacked value of the fit's own learned rule pairs, not the value of
  the optimal pair.
- The claim holds on one law with correct node-2, treatment and censoring models, binary
  nodes and censoring; `conservative_coverage` adds a misspecified node-1 learner with `g`
  correct, and claims coverage only.
- The `drop_w1` learner's interval under-covers (its limit arm contrast is 0.015 to 0.027, a
  weak margin); its cell reads only the bias legs and the displacement.
- The fold-locality control shows that the instrument resolves rules learned on the rows that
  evaluate them.

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. The run fits 11,200 draws (fold locality fits twice per draw): about 17,800
core-seconds on the pilot timings, which were taken on a shared machine, or about 74 minutes at
four jobs; about half that on an idle machine. With the summary bootstraps, allow 80 to 90
minutes.

```bash
python -u -m tests.canonical.learned_regimen_ltmle.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 3` also fits three draws of every property cell and prints the rule
agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_regimen_ltmle.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
