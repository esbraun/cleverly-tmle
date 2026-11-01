# Learned treatment rules over three arms

This directory holds the gated repeated-sampling study of X11 part (d) (X11 PR H). The subject
is `TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, targeting_scheme="pooled",
g_bounds="auto")` on a treatment with three arms. Each fold's rule is the plug-in argmax of its
outcome model over the arms, a tie going to the lowest arm code. The fit reports the
fold-evaluated value `ey_learned_rule[learned rule]`, whose target is the data-adaptive fold
average of `Psi_{d_v}(P0)` (van der Laan and Luedtke 2015; polle's argmax rule).

| part | declaration |
| --- | --- |
| law | `three_arms`: `W1 ~ U(-1, 1)`, `W2 ~ Bernoulli(0.5)`; multinomial logit `g0` with scores `0`, `0.7 W1 + 0.2 W2` and `-0.4 W1 + 0.2 W2` (every arm probability at least 0.166); binary outcome with `logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + b_a(W)`, `b_0 = 0`, `b_1 = -1 + 6 W1`, `b_2 = -1.75 - 6 W1 - W2 - 4 (W1^2 - 1/3)`. One size, n = 2,000 |
| learners | outcome: `ArmLogistic`, a logistic regression on fixed products of the design `(A1, A2, W1, W2)`, marked parametric. `declared` has every arm's intercept and `W1`, `W2` slopes and `A2 W1^2` (a correct model); `missing_quadratic` drops `A2 W1^2`; `drop_w1` keeps the arm indicators, `W2` and their products. Treatment: `LogisticRegression(C=inf, max_iter=5000)` on `(W1, W2)`, the multinomial maximum-likelihood fit of a correct model |
| margin condition | a law cell gates when, at its learner's population limit, every arm's share is at least 0.10 and the top-two gap is below 0.01 on at most 2% of the population. `declared`: shares 0.175, 0.417, 0.408 and 1.59%. `missing_quadratic`: shares 0.246, 0.417, 0.338 and 1.83%, and its limit rule differs from the optimal rule on 7.05% |
| primary | 1,200 replications of the `declared` fit (bias leg, coverage floor, SE sanity) |
| truth | per replication: refit each fold's outcome learner on the fit's training rows, take the argmax at the three arms, require the fit's rule on every validation row and the stored fold rule on every row (rule L4), and integrate `Qbar0(d_v(W), W)` on 4,001 points of `W1` for each `W2`; weight `1/V`. `oracle_se` is descriptive |
| `interval_calibration` | `declared`: the positive with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit, 0.670220). 3,200 draws |
| `conservative_coverage` | `missing_quadratic`: the positive passes on its bias leg and on the lower bound of its exact 99% coverage interval at or above 0.92; its SE ratio is published and no verdict reads it. The control is the same estimate with its SE times 0.60, whose coverage interval must lie wholly below 0.92. 1,200 draws |
| `targeting_necessity` | `drop_w1`: the targeted value against the untargeted fold average of the initial plug-ins, which must sit 0.5 positive-arm SDs or more from it. The learner's limit rule is arm 1 everywhere, and its plug-in limit is 9.4 SEs above that rule's value. 1,200 draws |
| `rule_margin_diagnostic` | role `diagnostic`, no verdict: for each law cell, the mean share of rows whose fitted rule differs from the cell's limit rule, the mean share whose own fold's top-two gap is below 0.01, the mean smallest arm share, and the share of draws in which some arm gets no row |
| verdicts | the framework defaults on the error `estimate - truth` of each row, and the coverage family's own rule in `tests/studies/learned_rule_arms_properties.py` |
| comparator | none: no external package fits a cross-validated TMLE of a learned multi-arm rule |

## The support row

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_rule_arms.py`). The
registering change adds one P11 row: the fold-evaluated learned rule with
`treatment=("arms3",)`. It is added if and only if these verdicts pass as declared: the primary
performance tests (bias leg, coverage floor, SE sanity), every `interval_calibration` verdict
(positive and both controls), both `conservative_coverage` verdicts, and the
`targeting_necessity` verdicts.

The route for a red primary bias leg, a red positive cell, or a control that does not fail is
declared now. The cell is diagnosed before a route is chosen. With no defect found, it is
published red under `reporting` with an owner row, and the row is not added. Under `gated` a
red primary gate still publishes the run. Four or more arms (`arms4` and above) are another key
and stay experimental, and so do weights, clusters, strata, fold targeting, repeats, the stacked
report and contrasts on three arms.

## Sizing

Each count is the smallest multiple of 400, at least 1,000, whose predictive pass probability on
a 4,000-draw pilot of every cell (throwaway seeds, no failed fit, rule L4 on every row) is at
least 0.95. The primary bias leg and the necessity bias leg give the floor, 1,200. The
calibration positive (pilot SE ratio 0.9986) gives 3,200 by the per-arm rule (0.952). The
coverage cells (pilot coverage 0.9515, control 0.751) give 1,200 (0.953). The design review ran
a second 4,000-draw pilot of the two law cells. On it alone the declared rules give the same
counts. Pooled with the first pilot, the coverage reads 0.9525, and with the coverage known at
one pooled SE lower (0.950) the 1,200 draws pass with 0.953. `test_learned_rule_arms_design`
recomputes every number.

## What registration must state

- The `missing_quadratic` cells claim coverage at or above the floor, not SE calibration. The
  reported curve omits the treatment model's projection term, which Corollary 2 of van der Laan
  and Luedtke (2015) makes conservative for a maximum-likelihood `g` in a correct model,
  extended to the cross-validated TMLE by analogy only. On this law the predicted SE ratio of
  that term is 1.0004, so the cells measure coverage under an outcome model misspecified in one
  arm's regression (arm 2's quadratic; arms 0 and 1 are exact at the limit), not
  conservativeness. Quote each positive's realized SE ratio beside its coverage.
- `drop_w1` is the targeting-necessity pair and reads bias only. Its fold rules have not
  settled at n = 2,000 (arm 1 leads arm 0 by 0.0245 at `W2 = 0` at the limit), so quote the
  targeted arm's SE ratio and coverage as a weak-margin reading, not as calibration evidence.
- The claim holds for a correctly specified multinomial treatment model fitted by maximum
  likelihood. A known `g` and a super-learner `g` are not measured.
- One law, one size, ten folds, no weights; the rule's margin condition holds at both
  learners' limits. Quote the `rule_margin_diagnostic` readings, including the share of fitted
  rules that lose an arm.

The declarations are in `tests/studies/learned_rule_arms.py`,
`tests/studies/learned_rule_arms_properties.py` and `tests/studies/_learned_rule_arms_law.py`.
The run form is the X11 one in `tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. The run fits 6,800 draws at about 0.22 to 0.30 core-seconds each, truth included: about
2,000 core-seconds, or roughly ten minutes at four jobs.

```bash
python -u -m tests.canonical.learned_rule_arms.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 3` also fits three draws of every property cell and prints the rule
agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_arms.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
