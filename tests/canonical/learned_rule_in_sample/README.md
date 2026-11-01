# In-sample learned-rule value

This directory holds the gated repeated-sampling study of X11 part (a) (X11 PR F2). The subject
is `TMLE(learned_rule=LearnedRule(), cross_fit=False, cv_evaluation=False, g_bounds="auto")`
with a parametric outcome learner and RM30's logistic treatment learner. The fit learns one
plug-in rule `d_n` from the outcome regression fitted on every row and runs one regime TMLE at
`d = d_n` on the same rows. The target of each replication is `Psi_{d_n}(P0)`, the value of that
one rule (van der Laan and Luedtke 2015, Theorem 5, conditions C3a, C4a and C5a).

| part | declaration |
| --- | --- |
| learners | outcome: `InteractionLogistic`, RM30's interaction logistic regression on `(A, W1, W2)` in a class marked `cleverly_learner_family = "parametric"`; treatment: RM30's logistic regression on `(W1, W2)`. Every fit's support key is checked: `splitting ("in_sample",)`, both learner families `parametric` |
| scenarios | RM30 `non_exceptional` (correctly specified) and `misspecified_limit` (the outcome model cannot represent the blip; the correct treatment model carries the remainder), one size n = 2,000 |
| primary | 14,400 replications of each law. The count is set by the bias leg: the readings of the optimism bias are +0.12 error SDs (`non_exceptional`, 4,000 pilot draws) and +0.21 (`misspecified_limit`, 8,000 pilot and design-review probe draws pooled), and 14,400 is the smallest multiple of 400 at which both 99% intervals of the mean error fall inside 0.25 SD with predictive probability 0.95 (0.9529). The first declaration read the pilot alone and set 8,400; the pooled reading moved it before any run (design record, D6) |
| stored variance | `{n (n - 1)}^-1 sum_i (phi_i - phibar)^2` with `phi_i = Z_i - psi`, `phibar = 0` to solver tolerance; a factor `n / (n - 1) = 1.0005` over `n^-2 sum_i phi_i^2`. The interval is `psi +/- 1.96 SE` |
| truth | per replication: refit the outcome learner on every row, require the fit's rule on every row (rule L4), and integrate the rule's value on 4,001 points of `W1` for each `W2` with the known `Qbar0`. `oracle_se` is `sqrt(Var_P0 D*(d_n, Qbar0, g0) / n)` at that rule, descriptive |
| `interval_calibration` | one positive per law, 7,200 draws each (the per-arm rule on a 4,000-draw pilot at the declared size), with the shrunken-SE control (factor 0.80) and the noise control (one efficiency-bound unit). `non_exceptional__interpolating_control`: an unmarked interpolating tree clipped to [0.01, 0.99], fitted in sample on the same draw and read against its own rule's value; C5a fails, and its SE ratio must fall below the band |
| `targeting_necessity` | `WithoutW1MainLogistic` (logistic on `(A, W2, A W1, A W2)`, marked parametric) on `non_exceptional`, 1,200 draws. The positive arm is the fit; the control is its initial plug-in at the fit's own rule. The control's mean must sit 0.5 positive-arm SDs or more from the positive arm's |
| `weak_limit_necessity` | role `diagnostic`, no verdict: the same pair with RM30's `drop_w1_learner`, whose limit blip (0.037 and 0.049) is too weak at n = 2,000 for one rule to settle: on the exact law its rule differs from the limit rule in about 17.5% of samples. The summary publishes `weak_limit_displacement`, the flip share `weak_limit_flip_share`, and the targeted arm's mean error on settled and on flipped draws, in its error SD, as `weak_limit_settled_bias` and `weak_limit_flipped_bias` |
| `target_distinction` | role `diagnostic`, no verdict: every calibration fit of each law (`non_exceptional__fold_average`, `misspecified_limit__fold_average`) read against the fold-average target `(1/10) sum_v Psi_{d_v}(P0)` of the same draw's 10-fold rules. The summary publishes the mean truth gap, its SD over the error SD, and the mean grid disagreement of the fold rules with `d_n` |
| verdicts | the framework defaults on the error `estimate - truth` of each row |
| comparator | none: no external package fits the in-sample learned-rule TMLE |

## The support row

Declared before any run (`SUPPORT_OPTIONS` in `tests/studies/learned_rule_in_sample.py`). The
registering change adds a P11 row with `splitting` `in_sample` and the options
`outcome_learner_family = "parametric"` and `treatment_learner_family = "parametric"` if and only
if these verdicts pass as declared: the primary performance tests of both laws (bias leg,
coverage floor, SE sanity), every `interval_calibration` verdict, and the `targeting_necessity`
verdicts. No cell fits a rule learner or passes `treatment_probabilities=`, so the row declares
no `rule_learner_family` and no `known` treatment family. In-sample fits outside the row stay
experimental.

The route for a red primary bias leg or a red positive calibration cell is declared now. The
diagnosis is finite-sample in-sample optimism, not a defect, unless the investigation finds a
defect. The cell is then published red under `reporting` with an owner row, and no P11 row is
added. Under `gated` a red primary gate still publishes the run.

## What registration must state

- The in-sample estimate leans optimistic: the pilot read +0.12 (`non_exceptional`) and the
  pooled readings +0.21 (`misspecified_limit`) error SDs at n = 2,000. Quote the realized
  standardized bias of each positive. A passed bias leg means "within 0.25 SD at n = 2,000",
  not "unbiased". The bias was measured at n = 2,000 only; theory makes it vanish beside the SE
  as n grows once the rule settles (C3a), and smaller samples lean further.
- The calibration verdicts cannot tell the full-sample target from the fold-average one at this
  size: the pilot's truth gap had an SD of 0.007 error SDs on `non_exceptional`, and the design
  review's probe read 0.013 on `misspecified_limit`. Quote both `target_distinction` cells; the
  unit witnesses, not this study, establish which rule the truth reads.
- The `targeting_necessity` learner replaced the plan's `drop_w1_learner` before the run,
  because that learner's rule does not settle at n = 2,000 (C3a): on the exact law it flips in
  about 17.5% of samples (design record, D2). The weak-limit pair is published as a diagnostic;
  quote its flip share and the targeted arm's bias on settled and flipped draws.
- Adaptive outcome learners (`InSampleLearnerWarning`) are outside the row; the interpolating
  control shows why.
- Intervals are pointwise; strata, weights, clusters, rule learners and known treatment
  probabilities under the in-sample fit are not measured here.

The declarations are in `tests/studies/learned_rule_in_sample.py`,
`tests/studies/learned_rule_in_sample_properties.py` and
`tests/studies/_learned_rule_in_sample_law.py`. The run form is the X11 one in
`tests/canonical/learned_rule_run.py`.

## Run

A declared run takes `--output`, `--jobs` and `--fresh` only, on the Python and package versions
that rule L7 of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical
library. It draws 45,600 samples (28,800 primary, 14,400 calibration and 2,400 necessity). The
pilot read about 0.03 core-s per declared fit. A `non_exceptional` calibration draw also fits the
tree and the ten fold rules (about 0.15 core-s in all), and a `misspecified_limit` one the ten
fold rules (about 0.12 core-s). That is about 2,900 core-s, under 15 minutes on 4 cores; allow an
hour with the summaries and bootstraps:

```bash
python -u -m tests.canonical.learned_rule_in_sample.regenerate --jobs 4 --output <empty scratch directory>
```

A smoke run passes another `--replicates` and writes outside the repository. `--property-harness-check 3` also fits three draws of every property cell and prints the rule agreement and the solver warnings:

```bash
python -u -m tests.canonical.learned_rule_in_sample.regenerate --replicates 4 --jobs 2 --allow-failures --output <scratch> --property-harness-check 3
```
