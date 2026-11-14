# Compositions of the value of the optimal rule

This directory will hold the gated repeated-sampling study of X11 part (f3) (X11 PR 2). The
declaration is committed before any run; no declared run has been made, and the study is not yet
in the evidence register. The subject is `TMLE(learned_rule=LearnedRule(),
estimands=["ey_optimal_rule"], n_folds=10, cv_evaluation=True, targeting_scheme="pooled",
g_bounds="auto")` with the known treatment mechanism, under the five compositions PR 2 admits.
Each composition's estimator is the learned-rule fit with the same setting, so its error is the
learned-rule claim with that setting minus the regret (JCI Lemma 1, Theorems 5 and 6).

| part | declaration |
| --- | --- |
| laws | RM30's `non_exceptional` law (`logit g0 = 0.3 W1 - 0.2 W2`, `logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a (0.1 + W1)`) for four settings; the `learned-rule-intermediate` law (binary `Z` after `A`, level blips `0.1 + W1 - 0.6 z`) for the intermediate setting |
| settings | `weighted` (weight `1 + 1.5 * 1{W1 > 0}`), `strata` (`strata=["W2"]`), `known_regimes` (beside `Static(1)` and `Static(0)`), `in_sample` (`cross_fit=False`, RM30's learner marked parametric), `intermediate` (both levels) |
| truths | fixed, by the trapezoid rule on 4,001 points of `W1` for each `W2`: weighted 0.630774; strata 0.577115 (marginal), 0.610609 and 0.543621; contrasts 0.045499 (vs always 1) and 0.064936 (vs always 0); in sample 0.577115; intermediate 0.577115 (`z = 0`) and 0.597071 (`z = 1`) |
| known g | every cell passes `treatment_probabilities=[1 - g0(1 \| W), g0(1 \| W)]`. The weight depends on `W` alone, so `g0` is the tilt's own mechanism |
| primary | 3,200 replications of each setting at n = 2,000, nine parameters (bias leg, coverage floor, SE sanity) against the fixed truths |
| `interval_calibration` | nine labels, each with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit). 8,400 draws per setting, each draw serving all of its setting's labels |
| `weighted_target_control` | its own draws of the weighted setting, read against the unweighted `Psi_d0 = 0.577115`, 3.40 SEs below the target. It passes when its exact 99% coverage interval lies wholly below 0.92. 1,200 draws |
| in-sample lean | the `in_sample` cell's standardized bias against the value of its own rule (`rule_value_standardized_bias`), reported with no verdict |
| descriptive columns | each row carries the value of the fit's own fold rules (refit with rule L4), the regret and the record's `rule_disagreement` |
| comparator | none: no external package fits these compositions |

## Sizing

The sizing pilot (`tests/studies/learned_rule_optimal_compositions_pilot.py`, 2,000 draws per
setting on its own seed stream, committed as `learned_rule_optimal_compositions_pilot.csv`) reads
SE ratios of 0.980 to 1.028 against the fixed truths, coverages of 0.9415 to 0.9565 and
standardized biases of -0.145 to 0.138. The arm-pooled predictive rule gives 8,400 calibration
draws per setting (family-wise 0.955 over the nine labels). The primary count is the smallest
multiple of 400 at which the nine parameters pass together with probability at least 0.95
(3,200). `tests/unit/test_learned_rule_optimal_compositions_design.py` recomputes both from the
pilot file.

## The support rows

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_rule_optimal_compositions.py`):
one P12 row per composition with the qualifier `known_mechanism`, each added only if its
setting's primary tests, every calibration verdict of its labels, and, for the weighted row, the
target control pass as declared. A red calibration row is a finite-sample red only when the 99%
interval of its SE ratio contains 1, its coverage is consistent with 0.95 and diagnosis finds no
defect; it then goes to `reporting` with an owner row. Any other red row is a defect suspect, and
a control that passes is a design defect.

## Running

    python -m tests.canonical.learned_rule_optimal_compositions.regenerate --output <scratch dir>

A smoke run adds `--replicates <k> --allow-failures --property-harness-check <k>`.
