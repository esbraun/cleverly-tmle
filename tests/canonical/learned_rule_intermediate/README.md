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
| `interval_calibration`, known rule | `known_z0` and `known_z1`, 20,000 draws of the known rule `1{W1 > 0}` through the same fold-evaluated path: the positive, the shrunken-SE and the noise controls. Witness W-j1 reduces the learned fit to this path bit for bit. The known-rule values are 0.576491 and 0.581904, the bounds 1.105274 and 0.855360 |
| `targeting_necessity` | the `drop_w1` learner (no `W1` in the outcome model) at each level, 1,200 draws: the targeted value against the untargeted fold average of the initial plug-ins. The plug-in's limit bias is 0.098 at `z = 0` (3.27 SEs) and -0.025 at `z = 1` (-1.17 SEs) by quadrature. The displacement must be at least 0.5 positive-arm SDs at both levels |
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
  and `known_z1` `interval_calibration` verdict passes. It is the measured path that W-j1
  reduces the learned fit to.

A red cell is diagnosed before a route is chosen. With no defect found, it is published red
under `reporting` with an owner row, and its row is not added. One law, ten folds, no weights,
two arms, a binary intermediate, the plug-in rule and `repeats=1` are what either row covers.

## Sizing

Each gated count is the smallest multiple of 400, at least 1,200, whose predictive pass
probability under the declared rule reaches 0.95. The rules were fixed before the pilot, and the
probe rule before the probe (`test_learned_rule_intermediate_design` recomputes every count).

- Primary: both levels' bias legs times their coverage floors. The 4,000-draw pilot gives 1,200
  (0.957), and so do the 8,000 draws pooled with the probe.
- Calibration: each draw's two levels at a product of at least 0.975, read by the per-cell rule
  (SE-ratio leg times coverage band) and by the arm-pooled rule over the two levels, taking the
  larger. The per-cell rule gives no learned count on the pilot (SE ratio 0.959 at `z = 1`),
  7,200 on the pooled draws, and 10,000 for the known rule. The arm-pooled rule gives no count
  up to 40,000 for either draw, so the declared fallback holds both at the run-time budget:
  20,000 each.
- Controls: each lies wholly below the SE-ratio band at its count with probability above 0.999.
- Necessity: 1,200, the floor. The pilot's displacements are 3.04 and 1.16 SDs.

## What registration must state

- Each level's interval targets the value of that level's own learned rule, not the value of
  the optimal rule at that level.
- The claim holds on one law with a correct outcome model and the intermediate binary.
- The term-dropped control's failure shows that the factor `1{Z = z} / q` is needed for valid
  intervals, and the necessity cell that targeting is needed when the outcome model leaves out a
  confounder.
- Quote the first pilot's low SE ratio at `z = 1` (0.959) beside the probe's (1.003).

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. The run fits 42,400 draws: about 13,600 core-seconds on the pilot timings, or roughly
57 minutes at four jobs.

```bash
python -u -m tests.canonical.learned_rule_intermediate.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 3` also fits three draws of every property cell and prints the rule
agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_intermediate.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
