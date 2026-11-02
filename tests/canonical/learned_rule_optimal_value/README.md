# The value of the optimal rule

This directory holds the gated repeated-sampling study of X11 part (f) (X11 PR I). The subject
is `TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"], n_folds=10,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`. It reports
`ey_optimal_rule[learned rule]`, whose target is the fixed parameter
`Psi_d0(P0) = E Qbar0(d0(W), W)` with `d0(w) = 1{Qbar0(1, w) > Qbar0(0, w)}`, a tie going to
arm 0. The estimator is the shipped fold-evaluated CV-TMLE of the learned-rule value (van der
Laan and Luedtke 2015, Section 7.2, last paragraph). Its licence needs the learner's limit rule
to be `d0` (C3f), no unit with a zero blip (C7, the source's condition (5)), and a regret
`Psi_d0 - psi~0n` of order `o_P(n^-1/2)` (C8, Lemma 1), with the treatment mechanism known.

| part | declaration |
| --- | --- |
| laws | RM30's covariates, `logit g0 = 0.3 W1 - 0.2 W2` and `logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a b(W)`, with three blips: `non_exceptional`, `b = 0.1 + W1`; `wrong_limit`, `b = 3 (W1^2 - 0.5)`; `partially_exceptional`, `b = (W1 - 0.2)_+` |
| truth | `Psi_d0` by the trapezoid rule on 4,001 points of `W1` for each `W2`, one number per law: 0.577115, 0.557421 and 0.547362. The record reads one truth per cell |
| learners | outcome: RM30's logistic regression on every pairwise product of `(A, W1, W2)`; on `partially_exceptional` the same six products and `A (W1 - 0.2)_+`, so both are correct models of their laws. Treatment: RM30's logistic regression on `(W1, W2)`, fitted only in the estimated-g cell |
| known g | every cell but one passes `treatment_probabilities=[1 - g0(1 \| W), g0(1 \| W)]`, so its support key reads `known_mechanism` |
| gate | a law cell gates when its learner's population limit has no tie mass (C7), differs from `d0` off the tie set on at most 0.001 of the population (C3f), and has `max_t P(0 < \|b0\| <= t) / t <= 10` on the probability scale (C8 with exponent 1). `non_exceptional` reads 0, 0 and 4.07 and gates; `wrong_limit` differs on 0.293 and is the control; `partially_exceptional` has tie mass 0.6 and reports |
| primary | 1,600 replications of the known-g `non_exceptional` fit at n = 2,000 (bias leg, coverage floor, SE sanity), against `Psi_d0` |
| `interval_calibration` | `non_exceptional`, known g, n = 2,000: the positive with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit, 0.664438). 13,200 draws |
| `optimal_target_control` | `wrong_limit`, known g, n = 2,500: the fit's own estimate and SE. The declared learner's limit rule treats nobody while `d0` treats 29.3%, so the target converges to a value 0.0452 below `Psi_d0`, 3.29 SEs at n = 2,500. The control passes when its exact 99% coverage interval lies wholly below 0.92. 1,200 draws |
| `estimated_g_report` | `non_exceptional` with the treatment model fitted, n = 2,000: role `diagnostic`, no verdict. 1,200 draws, the study's floor of 1,000 as a multiple of 400, as for the exceptional report. The cell can never become gated evidence: a support row for estimated `g` needs a gated cell with its own positive and controls, declared before its run. At 1,200 draws its coverage reads to about ±0.014 (95%), enough to show a gross failure |
| `exceptional_law_report` | `partially_exceptional`, known g, n = 2,000: role `diagnostic`, no verdict. 1,200 draws |
| descriptive columns | each row carries the data-adaptive `psi~0n` of the fit's own fold rules (refit with rule L4, integrated by the same quadrature), the regret `Psi_d0 - psi~0n` and the record's `rule_disagreement`. Each cell's summary publishes the mean regret, the mean, smallest and largest disagreement, and the error `estimate - psi~0n` read as a standardized bias, a coverage and an SE ratio (`rule_value_standardized_bias`, `rule_value_coverage`, `rule_value_se_ratio`). No verdict reads them |
| verdicts | the framework defaults on the error `estimate - Psi_d0`, and the control's rule in `tests/studies/learned_rule_optimal_value_properties.py` |
| comparator | none: no external package fits a CV-TMLE of the value of the optimal rule |

The record's `n` is the primary's, 2,000. Each property row carries its own `n`, so the control
runs at 2,500 while every other cell keeps 2,000. That size is W-f2's: the smallest multiple of
500 at which the quadrature gap is at least three SEs of the limit rule's curve (2,085 needed;
2,000 gives 2.94).

## The support row

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_rule_optimal_value.py`). The
registering change adds one row: family P12, the fold-evaluated CV-TMLE of the value of the
optimal rule, with the qualifier `known_mechanism`. It is added if and only if these verdicts
pass as declared: the primary performance tests (bias leg, coverage floor, SE sanity), every
`interval_calibration` verdict (positive and both controls), and the `optimal_target_control`
verdict.

No row is added for estimated `g`, whose cell reports; a row for it waits for a gated cell of its
own. An exceptional law gets no row. The route for a red primary bias leg, a red positive cell,
or a control that does not fail is declared now. The cell is diagnosed before a route is
chosen. With no defect found, it is published red under `reporting` with an owner row, and the
row is not added.

## Sizing

Each gated count is the smallest multiple of 400, at least 1,000, whose predictive pass
probability under the declared rule reaches 0.95, read on a 4,000-draw pilot and on the 8,000
draws pooled with a second 4,000-draw probe of the gated cell, taking the larger count. That
rule was fixed before the probe ran.

- Primary: the bias leg times the coverage floor. The first pilot (bias -0.104 SDs, coverage
  0.936) gives 1,600 (0.987; 1,200 gives 0.944). The pooled draws give 1,200.
- Calibration: the SE-ratio leg times the coverage leg. The first pilot (SE ratio 0.971,
  coverage 0.936) gives 13,200 (0.953); the SE ratio alone would give 11,200. The pooled draws
  (0.984, 0.943) give 4,400, and the probe alone 3,200.
- Control: its pilot coverage, 0.092, clears 0.92 with probability one, so the floor gives 1,200.
- Reports: both take the floor, 1,200. No verdict reads them, so no reading sets their count.

## What the regret does to each verdict

Every verdict reads the error against the fixed `Psi_d0`, and the realized rule value `psi~0n`
falls short of it by the regret. At n = 2,000 the pilots put the mean regret near 0.10 estimate
SDs. Against `psi~0n` the readings are published beside each cell, with no verdict.

- Primary bias leg: the 99% interval of the bias should sit near -0.17 to -0.04 SDs. It is
  expected to exclude zero, because of the regret, and to pass the 0.25 SD leg.
- Coverage floor and coverage band: a bias of 0.10 SD costs about 0.001 of coverage at an SE
  ratio of 1, so a gap from 0.95 comes mostly from the SE ratio.
- SE sanity and the calibration SE band: the regret's own spread widens the error's SD by about
  0.9% at n = 2,000. The pooled pilots read an SE ratio of 0.984 against `Psi_d0` and 0.993
  against `psi~0n`. Both effects are of order `n^-1/2` relative to the SE.
- Calibration controls: they inherit the positive's regret, and fail by construction far outside
  both bands.
- `optimal_target_control`: its miss is the limit gap, a regret of about 3.3 SEs. Against its own
  `psi~0n` it covers (pilot 0.946, SE ratio 1.00), which shows it fails for the right reason.
- `exceptional_law_report`: the regret is small (pilot 0.0009). Its under-coverage comes from fold
  rules that do not settle on the tie set, not from the regret.

`test_learned_rule_optimal_value_design` recomputes every number.

## What registration must state

- The bias leg reads the regret. Against its own `psi~0n` the gated cell is centred (pilot bias
  -0.01 SDs), and against `Psi_d0` it reads -0.104 SDs, the mean regret of 0.0014 at n = 2,000.
  Lemma 1 makes the regret `O_P(1/n)` here, and the study measures one size, so it does not
  show the rate. Quote the published mean regret and the readings against `psi~0n`.
- The primary bias interval is expected to exclude zero because of the regret, and the SE ratio
  against `Psi_d0` carries the regret's spread, about 0.9% at n = 2,000.
- The claim holds with the treatment mechanism known, on a law where the learner is correct
  and the blip has a bounded density at 0. Quote the estimated-g cell's bias, SE ratio and
  coverage as a reading for decision E2, not as a validated claim.
- The partially exceptional cell shows how the interval behaves where C7 fails (pilot: coverage
  0.908, SE ratio 0.882, mean disagreement 0.29); no claim is made there.
- One law per cell, ten folds, no weights, two arms, the plug-in rule, `repeats=1`. Clusters,
  fold targeting, repeats, rule learners, the stacked report and three arms stay experimental.

The declarations are in `tests/studies/learned_rule_optimal_value.py`,
`tests/studies/learned_rule_optimal_value_properties.py` and
`tests/studies/_learned_rule_optimal_law.py`. The run form is the X11 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. The run fits 18,400 draws at about 0.20 to 0.22 core-seconds each, truth included:
about 3,800 core-seconds, or roughly 16 minutes of fitting at four jobs.

```bash
python -u -m tests.canonical.learned_rule_optimal_value.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 3` also fits three draws of every property cell and prints the rule
agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_optimal_value.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
