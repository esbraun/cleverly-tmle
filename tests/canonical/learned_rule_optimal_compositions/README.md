# Compositions of the value of the optimal rule

This directory holds the gated repeated-sampling study of X11 part (f3) (X11 PR 2). The
declaration was committed before any run. The declared run passed every primary test and every
property cell, and the study is registered with the support rows `T55`
(`docs/technical-reference/method-evidence/learned-rule-optimal-compositions.md`).
The subject is `TMLE(learned_rule=LearnedRule(),
estimands=["ey_optimal_rule"], n_folds=10, cv_evaluation=True, targeting_scheme="pooled",
g_bounds="auto")` with a known treatment mechanism, under the five compositions PR 2 admits.
Each composition's estimator is the learned-rule fit with the same setting, so its error is the
learned-rule claim with that setting minus the regret (JCI Lemma 1, Theorems 5 and 6).

| part | declaration |
| --- | --- |
| laws | RM30's `non_exceptional` law (`logit g0 = 0.3 W1 - 0.2 W2`, `logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a (0.1 + W1)`) for four settings; the `learned-rule-intermediate` law (binary `Z` after `A`, level blips `0.1 + W1 - 0.6 z`) for the intermediate setting |
| settings | `weighted` (weight `1 + A Y`), `strata` (`strata=["W2"]`), `known_regimes` (beside `Static(1)` and `Static(0)`), `in_sample` (`cross_fit=False`, RM30's learner marked parametric), `intermediate` (both levels) |
| weighted target | the weight moves the optimal rule to `1{W1 > -0.1 - log 2}`, so the target is the weighted population's optimal value 0.690108, 0.0277 above the weighted value of the unweighted rule (0.662385) |
| truths | fixed, by `scipy.integrate.quad` over `W1` for each `W2`, split at the rule's threshold: weighted 0.690108; strata 0.577115 (marginal), 0.610609 and 0.543621; contrasts 0.045499 (vs always 1) and 0.064936 (vs always 0); in sample 0.577115; intermediate 0.577115 (`z = 0`) and 0.597071 (`z = 1`). The design test checks each against a second quadrature written independently |
| known g | every cell passes `treatment_probabilities=`: RM30's `g0` in four settings, and in the weighted one the weighted population's own mechanism `g_w = g0 E[w \| A, W] / E[w \| W]` |
| primary | 4,400 replications of each setting at n = 2,000, nine parameters (bias leg, coverage floor, SE sanity) against the fixed truths |
| `interval_calibration` | nine labels, each with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit); each stratum label also with the `unnormalised_curve_control` (its curve without the fold share, SE about half). 10,000 draws per setting, each draw serving all of its setting's labels |
| `weighted_target_control` | 1,200 draws of the weighted setting, each read against the unweighted optimal value 0.577115 (`target_gap_control`, 8.36 SEs below the target) and against the weighted value of the unweighted rule 0.662385 (`unweighted_rule_control`, 2.05 SEs below). Each passes when its exact 99% coverage interval lies wholly below 0.92 |
| in-sample lean | the `in_sample` cell's standardized bias against the value of its own rule (`rule_value_standardized_bias`), reported with no verdict |
| descriptive columns | each row carries the value of the fit's own fold rules (refit with rule L4), the regret, the record's `rule_disagreement` and the untargeted plug-in (`initial_estimate`) |
| comparator | none: no external package fits these compositions |

## Sizing

The sizing pilot (`tests/studies/learned_rule_optimal_compositions_pilot.py`, 2,000 draws per
setting on its own seed streams, committed as `learned_rule_optimal_compositions_pilot.csv`)
reads SE ratios of 0.980 to 1.028 against the fixed truths, coverages of 0.9415 to 0.9565 and
standardized biases of -0.160 to 0.138. The weighted setting was re-piloted under `1 + A Y` on
its own streams (`se_ratio_pilot_ay`). A positive needs both its SE-ratio leg and its coverage
leg, so the calibration count is sized on the joint family: the arm-pooled SE-ratio family times
the nine coverage legs. It first reaches 0.95 at 9,600 (0.952), which a stress like T46's (the
pooled mean one pooled pilot SE further from 1, every coverage one pilot SE toward its nearer
edge) leaves at 0.949, so the count is 10,000 (joint 0.957, stressed 0.954). The SE-ratio rule
alone would give 8,400. The primary count is the smallest multiple of 400 at which the nine
parameters pass together with probability at least 0.95 (4,400; 0.949 at 4,000).
`tests/unit/test_learned_rule_optimal_compositions_design.py` recomputes each count from the
pilot file.

The arm-pooled rule pools labels whose SE constructions differ (stratum curves, contrasts, the
in-sample and level curves), which is optimistic for the label furthest from 1: `strata_s1`
(pilot 1.028) passes its SE-ratio leg with about 0.79 at 10,000 if its true ratio is one pilot
SD higher. A red there is read under the red-row rule below; the design does not size for it.

The declared run is 5 × (4,400 + 10,000) + 1,200 = 73,200 fits. At the pilot's CPU seconds per
draw (weighted 0.17, strata 0.40, known regimes 0.20, in sample 0.03, intermediate 0.35, fit and
harness refit together) that is about 4.7 CPU-hours: roughly 1.2 hours on 4 cores and 20 minutes
on 16, plus overhead.

## What no cell sees

The weighted row covers fixed weights with the weighted population's mechanism supplied.
Supplying the design mechanism `g0` beside weights that depend on the treatment or the outcome is
the Fix-16 defect, which the support key cannot see and no cell measures.

The intermediate cell cannot see a fit that ignores `Z`: the total-effect optimum (0.588) is 0.44
and 0.46 SEs from the two level truths, and the level-blind rule 0.27 and 0.22 SEs. That rests on
W-f3-id (each composition is the learned-rule fit with the same setting, bit for bit), on witness
W-f3z, and on the harness, which stops a draw whose level rules differ from the refit's. The
stratum cells' `unnormalised_curve_control` resolves a gross normalisation error only; the
fold-local share itself rests on W-f3-id and on the witnesses of `test_learned_rule_strata`.

## The support rows

Declared before any run (`SUPPORT_ROWS` in `tests/studies/learned_rule_optimal_compositions.py`,
as `point_row` arguments): one P12 row per composition with the qualifier `known_mechanism`, the
in-sample row with the learner families of condition C5a (`outcome_learner_family="parametric"`,
`treatment_learner_family="known"`). The design test requires every fitted key of every setting,
both intermediate levels included, to be covered by its row with equal options and qualifiers.
Each row is added only if its setting's primary tests, every calibration verdict of its labels,
and, for the weighted row, both target controls pass as declared. A red calibration row is a
finite-sample red only when the 99% interval of its SE ratio contains 1, its coverage is
consistent with 0.95 and diagnosis finds no defect; it then goes to `reporting` with an owner row.
Any other red row is a defect suspect, and a control that passes is a design defect.

## Running

    python -u -m tests.canonical.learned_rule_optimal_compositions.regenerate --jobs 4 --output <empty scratch directory>

A smoke run, on the throwaway record, before the declared run:

    python -u -m tests.canonical.learned_rule_optimal_compositions.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3

That smoke ran on the declared code before any declared run (2 jobs, 32 s of wall time). It exited
cleanly, wrote the seven artefacts and the manifest, filled `initial_estimate` for every
parameter, and wrote all 31 property cells three times each. The refit fold rules agreed with
the fits on every row, with no solver warnings. Its performance tests fail at 4 replicates, as
`--allow-failures` expects.
