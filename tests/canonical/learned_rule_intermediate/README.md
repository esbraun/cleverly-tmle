# The learned rule with an intermediate held at each level

This directory holds the gated repeated-sampling study of X11 part (j) (X11 PR J). The subject
is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled",
g_bounds="auto").fit(..., intermediate="Z")`. It reports one result per level `z` of a binary
intermediate `Z`, each `ey_learned_rule[learned rule]`. Each level learns its own fold rules
`d^z_nv(w) = 1{Qbar_nv(1, z, w) > Qbar_nv(0, z, w)}`, and the target is the data-adaptive
`psi~^z_0n = V^-1 sum_v E_0 Qbar0(d^z_nv(W), z, W)`. The clever covariate is
`1{A = d^z} 1{Z = z} / (g(A | W) q(z | A, W))`.

| part | declaration |
| --- | --- |
| law | RM30's covariates and `logit g0 = 0.3 W1 - 0.2 W2`; `logit P(Z = 1 \| A, W) = 0.2 + 0.6 A - 1.2 W1`; `logit Qbar0(a, z, W) = 0.2 + 0.5 W1 - 0.3 W2 + 0.3 z + a (0.1 + W1 - 0.6 z)`. At `z = 0` the outcome regression is RM30's `non_exceptional` one |
| optimal rules | `1{W1 > -0.1}` at `z = 0` (55% treated) and `1{W1 > 0.5}` at `z = 1` (25%). They differ on 30% of the population. The smallest `g0 q0` on the support is 0.045 |
| truth | each row's level value of the fit's own fold rules, refit with rule L4 and integrated by the trapezoid rule on 4,001 points of `W1` for each `W2` (`truth_varies_by_replicate`). The optimal values are 0.577115 and 0.597071, and the efficiency bounds 1.120917 and 0.882374 |
| exactness gate | the harness rebuilds each level's variance longhand from the fit's curve and raises if it differs from the reported one by more than `1e-8` relatively |
| learners | outcome: RM30's logistic regression on every pairwise product of `(A, W1, W2, Z)`, a correct model of the law. treatment: RM30's logistic regression on `(W1, W2)`; intermediate: a main-terms logistic regression on `(A, W1, W2)`, correct for the law. Both are fitted in every cell |
| primary | `cde_z0` and `cde_z1`, 1,200 replications at n = 2,000. One fit writes both levels' rows from one sample. Each level reads the bias leg, the coverage floor and the SE sanity band |
| `interval_calibration`, learned | `learned_z0` and `learned_z1`, 20,000 draws: the positive, the shrunken-SE control (x0.80), the noise control (one bound unit) and `term_dropped_control`, the same estimate with the SE of the curve rebuilt without `1{Z = z} / q`. Each control must fall below the SE-ratio band |
| `interval_calibration`, known rule | `known_z0` and `known_z1`, 10,000 draws of the known rule `1{W1 > 0}` through the same fold-evaluated path: the positive, the shrunken-SE, the noise and the term-dropped controls. Witness W-j1 reduces the learned fit to this path bit for bit. The known-rule values are 0.576491 and 0.581904, the bounds 1.105274 and 0.855360 |
| `targeting_necessity` | the `drop_w1` learner (no `W1` in the outcome model) at each level, 1,200 draws: the targeted value against the untargeted fold average of the initial plug-ins. The plug-in's limit bias is 0.098 at `z = 0` (3.27 SEs) and -0.025 at `z = 1` (-1.17 SEs) by the trapezoid rule. The displacement must be at least 0.5 positive-arm SDs at both levels |
| verdicts | the framework defaults on the error `estimate - psi~^z_0n`, `calibration_rule` without an efficiency band, and the necessity rules |
| comparator | none: no external package fits a learned rule with an intermediate held fixed |

Every cell is at n = 2,000.

## The support rows

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_rule_intermediate.py`). Both
keys read experimental now, with splitting `fold_evaluated` and missing `intermediate`.

- Learned rule, family P11. The registering change adds this row if and only if both primary
  levels (bias leg, coverage floor, SE sanity) and every `learned_z0` and `learned_z1`
  `interval_calibration` verdict (positive, shrunken, noise and term-dropped controls) pass,
  and the `targeting_necessity` verdict passes.
- Known rule, family P5. The registering change adds this row if and only if every `known_z0`
  and `known_z1` `interval_calibration` verdict (positive and the same three controls) passes.
  It is the measured path that W-j1 reduces the learned fit to. The study measures one dynamic
  rule's value, `1{W1 > 0}`, but the P5 key does not tell a value from a contrast or a static
  regime from a dynamic one. Once added, the key also covers static regimes, contrasts between
  known regimes and any deterministic rule with a binary intermediate on the fold-evaluated
  path, and none of those is measured. T19, where one dynamic-rule cell backs P5, is the
  precedent.

A red cell is diagnosed before a route is chosen. With no defect found, it is published red
under `reporting` with an owner row, and its row is not added. Neither row covers weights,
clusters, strata, fold targeting, repeats, the stacked or in-sample report, more than two arms
or a rule learner. A non-binary intermediate is refused at fit (X63).

## Sizing

The plan's rule is the per-cell predictive rule: each count is the smallest multiple of 400, at
least 1,200, whose predictive pass probability reaches 0.95, with each calibration draw's two
levels at a product of at least 0.975. `test_learned_rule_intermediate_design` recomputes every
count from the pinned pilot readings.

- Primary: 1,200, both levels' bias legs times their coverage floors.
- Learned calibration: 20,000. The per-cell rule gives no count on the first 4,000-draw pilot.
  The probe rule, fixed before the probe, then reads the 8,000 draws pooled with a second pilot
  and sets the count at the run-time budget. The per-cell rule alone gives 7,200 on the pooled
  draws.
- Known calibration: 10,000, the per-cell rule on the pilot.
- The arm-pooled rule (`replicates_from_pilot`) was declared beside the per-cell rule. With two
  arms it returns no count on a 4,000-draw pilot even at a mean SE ratio of exactly 1, so it is
  withdrawn as a sizing rule on that condition. The two levels are different settings, which is
  the case `per_arm_replicates` is for.
- Controls: each lies wholly below the SE-ratio band at its count with probability above 0.999.
- Necessity: 1,200, the floor.

## What registration must state

- Each level's interval targets the value of that level's own learned rule, not the value of
  the optimal rule at that level.
- The claim holds on one law with correct outcome, treatment and intermediate models and a
  binary intermediate.
- The term-dropped controls show that the factor `1{Z = z} / q` is needed for valid intervals,
  on the learned and the known cells. The necessity cell shows that targeting is needed when the
  outcome model leaves out a confounder.
- Quote the published SE ratio of each learned level with its 99% interval. The pilot readings
  are in the design record, not on the page.

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. The run fits 32,400 draws: about 11,600 core-seconds on the pilot timings, or roughly
48 minutes of fitting at four jobs. With the summary bootstraps, allow 60 to 70 minutes.

```bash
python -u -m tests.canonical.learned_rule_intermediate.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 3` also fits three draws of every property cell and prints the rule
agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_intermediate.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
