# Projection methods

This study will validate two estimators of the difference-scale projection of a
`ConditionalContrast` on `Linear(["W1"])`: the DR-TMLE projection with `guard=("Q", "g")`, and the
per-arm outcome-adaptive C-TMLE projection.
[Conditional effects](../conditional-effects.md#the-dr-tmle-projection) gives the targets, the
estimators and their conditions. The study is declared and not registered. Its first declared run
failed two gated cells, and its second declared run has not been made, so no row of this study
supports a claim yet. This page publishes the two red rows of the first run and the redesign they
led to. The full evidence tables follow the second run.

## The re-declared cells

The first run was declared at commit `32f9d8ac` and ran at commit `c566a4c`. Two gated cells were
red. A diagnosis found no defect in either estimator: each cell measured something other than its
claim. The study re-declared both groups on fresh seed streams before any new draw, and moved no
margin, band, rule, law, offset or threshold. The second run replaces every artifact of the
first, and its unchanged groups must reproduce the first run's rows and verdicts.
`tests/studies/projection_methods_limit.csv` holds every figure below, copied from the first run's
artifacts or computed by the declared-stream probes of `tests/studies/projection_methods_limit.py`.

### The conservative interval at a correct parametric mechanism

The `dr_glm_4k` group fits the DR-TMLE projection with the correct logistic mechanism and a wrong
outcome regression at n = 4,000. The interval of its intercept is predicted to be conservative,
and the cell requires the 99% interval of the SE ratio to lie wholly above the calibration band's
upper end of 1.07.

| first run, 2,400 replicates | value |
| --- | --- |
| SE ratio (99% interval) | 1.1076 (1.0661 to 1.1514) |
| coverage (99% interval) | 0.9675 (0.957 to 0.976) |
| bias | 0.00075 |
| verdict | **fail** |

The interval is wider than the spread, and its coverage is above 0.95. The ratio's interval
reaches below 1.07, so the cell did not show the excess it is named for. The cell had been sized
from the quadrature of an estimator that solves the reduced equations (8) and (9) only, whose SE
ratio is 1.189. The declared estimator also solves equation (10), whose fluctuation moves the
outcome regression's limit by an amount that does not shrink with n. Its ratio is lower, and it
varies from fit to fit even at n = 200,000.

| fit | SE ratio of the intercept |
| --- | --- |
| quadrature of the limit, equations (8) and (9) | 1.189 |
| n = 200,000, `guard=("Q",)`, replicates 0 and 1 | 1.190 and 1.187 |
| n = 200,000, declared `guard=("Q", "g")`, replicates 0 and 1 | 1.133 and 1.108 |
| first run and the 1,000-replicate declared-stream probe at n = 4,000, pooled | 1.108 |

The redesigned cell reads the pooled finite-sample ratio, rounded down to 1.105, as its planning
value. At that value and a spread constant of 0.74, the declared sizing rule gives R = 17,600
replicates on a fresh stream. The claim that the interval is conservative at a correct
parametric mechanism is withheld for the first run, and the redesigned cell decides it.

### The first-order ladder at 1,000 rows

The `dr_slow` ladder fits the DR-TMLE projection under a mechanism with a constructed logit error
$n^{-1/3} W_2 (1 + W_1)$. Each rung must show a negative bias outside the margin of 0.25 SD. In the
first run the 1,000-row rung read a standardized bias of -0.26 on the intercept and -0.44 on the
slope, and the intercept stayed inside the margin.

| fit at n = 1,000 | intercept (SD) | slope (SD) |
| --- | --- | --- |
| first run, declared offset, 1,000 replicates | -0.26 | -0.44 |
| probe, 400 replicates each: no offset | +0.12 | -0.06 |
| declared offset | -0.19 | -0.31 |
| twice the declared offset | -0.22 | -0.53 |
| three times the declared offset | -0.16 | -0.62 |

Without the offset, the intercept carries a positive finite-sample bias of +0.0036 (SE 0.0015), which works
against the negative first-order term. Larger offsets move the intercept little. At n = 1,000 on
this law, the first-order bias of the intercept under a slow mechanism is therefore masked by an
opposing finite-sample bias, and the rung measured that sum rather than the first-order term. The
ladder now runs at 4,000, 16,000 and 64,000 rows, with the factor-4 spacing and factor-16 span of
the first ladder. The claim that the projection's first-order bias under a slow mechanism is
visible and grows with n is withheld until the redesigned ladder passes.
