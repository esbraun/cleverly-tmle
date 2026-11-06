# Learned-rule value, EP rule learner

This directory holds the declared repeated-sampling study of P1 part (c). The subject is
`TMLE(learned_rule=LearnedRule(loss="ep", modifiers=Modifiers(["W1", "W2"], domains={"W1": (-1, 1)}, levels={"W2": (0, 1)})), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`
with `repeats=1`. Each fold's rule is the sign of the EP curve of the effect given `(W1, W2)`,
fitted on the fold's training rows from the shipped 5-fold inner cross-fit. The default sieve's
size is chosen by cross-validation over the inner folds, with the nuisances refitted inside each.
The target is the fold-average value of the rules the fit learned. The study is declared, not
registered: no support row cites it until its run lands.

| part | declaration |
| --- | --- |
| scenarios | RM30 `non_exceptional` and `misspecified_limit` with the main-terms outcome learner and the logistic treatment learner (Q12: the RM30 GLMs), rows only. The primary phase fits n = 2,000 for 2,000 replications per law; the calibration cells fit n = 16,000 (plan rev 2.6) |
| limit | with the inner treatment learner correct, the EP risk is doubly robust and each fold curve converges to the additive projection `E[tau \| W1] + E[tau \| W2] - E[tau]` of the effect. Its sign changes once in `W1` for each `W2` (thresholds -0.0797 and -0.1203 at `non_exceptional`, 0.2900 and 0.2662 at `misspecified_limit`), with a share near zero linear in the distance, so C3 holds. Against `d0` it differs on 1.0% of the population and gives up 2.5e-5 of value, 0.005 SE at n = 16,000 |
| truth | rule L4: each replication refits each fold's rule through `fit_fold_rule` with the fold's recorded inner seed and the fit's resolved `g_bounds` and `alpha`, requires the fit's rule on every validation row, integrates each rule's value on the RM30 grid, and averages over the folds. The construction rests on the unit witnesses W-c1 to W-c3 of `tests/unit/test_learned_rule_ep.py` |
| properties | `interval_calibration`: `<law>__declared` for both laws and `optimal_known_g__declared` (the value of the optimal rule with `g0` supplied at `non_exceptional`, truth `Psi_d0 = 0.577115`), 7,200 draws each at n = 16,000, with the shrunken-SE (0.70) and noise controls of each, and the fold-locality control `non_exceptional__validation_rule_control` (each fold's EP curve fitted on its own validation rows, 500 draws at n = 2,000), whose coverage interval must lie below 0.92. `small_sample_diagnostic` fits each law at n = 1,000 for 2,000 draws and `exceptional_law_report` fits the RM30 `exceptional` law at n = 2,000 for 2,400 draws; both are role `diagnostic` and state no verdict. Every gated and diagnostic row also carries the paired plug-in fit on the same sample and folds |
| verdicts | the framework defaults, read on the error `estimate - truth` of each row: coverage floor 0.90 and SE screen 0.80 to 1.20 on the primary phase, SE-ratio band 0.93 to 1.07 and coverage band 0.92 to 0.98 on the calibration cells, 99% intervals |
| comparator | none |

The declarations are in `tests/studies/learned_rule_ep.py`,
`tests/studies/learned_rule_ep_properties.py` and `tests/studies/_learned_rule_ep_law.py`. The
red-cell readings are in the docstring of `regenerate.py`, declared before any run and before the
first pilot's summary was read. Plan rev 2.6 moved the calibration cells from n = 4,000 to 16,000
after both pilots were read and before any declared draw; it changed the size the readings name
and nothing else in them.

## Sizing

Both pilots ran on their own record seed 20261131, which no declared draw reads.
`tests/unit/test_learned_rule_ep_design.py` checks their seeds against every declared seed and
recomputes every reading below from the committed summaries.

The first pilot (`tests/studies/learned_rule_ep_pilot.py`, streams `se_ratio_pilot` and
`se_ratio_pilot_folds`, summary `tests/studies/learned_rule_ep_pilot.csv`) sized the study at
n = 4,000:

| pilot cell | n | draws | SE ratio | oracle-SE ratio | coverage | standardized bias | CPU-s per fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `non_exceptional__declared` | 4,000 | 2,000 | 0.970 | 0.969 | 0.9365 | -0.021 | 5.3 |
| `misspecified_limit__declared` | 4,000 | 600 | 0.990 | 0.987 | 0.963 | -0.015 | 5.0 |
| `optimal_known_g__declared` | 4,000 | 600 | 0.999 | 0.998 | 0.957 | -0.097 | 4.3 |
| `non_exceptional__primary` | 2,000 | 300 | 0.943 | 0.940 | 0.930 | -0.024 | 4.1 |
| `non_exceptional__validation_rule_control` | 2,000 | 60 | 0.915 | 0.926 | 0.017 | 3.74 | 0.2 |

At n = 4,000 the `non_exceptional` ratio less one pilot SE is 0.970 - 0.017 = 0.953, below the
plan's 0.96, so the `T49` rows were out of reach. The oracle-SE ratio sits at the SE ratio: the
reported SE is the fixed-rule SE of the realized rules, and rule learning adds about 6.5% to the
error variance.

The size pilot (`tests/studies/learned_rule_ep_size_pilot.py`, streams `se_ratio_size_pilot_1`
and `se_ratio_size_pilot_1_folds`, summary `tests/studies/learned_rule_ep_size_pilot.csv`) fits
the same cells at n = 8,000 and 16,000. Each draw also fits the *limit-rule* fit: the same
estimator, sample and folds with the rule fixed at the EP limit rule. Its error correlates with
the learned fit's at 0.95 to 0.97, so their variance ratio resolves the excess about 2.5 times
more tightly than the SE ratio alone.

| pilot cell | n | draws | SE ratio | oracle-SE ratio | coverage | variance ratio to the limit-rule fit | paired SE ratio | CPU-s per fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `non_exceptional__n8000` | 8,000 | 600 | 1.018 | 1.018 | 0.950 | 0.974 (SD 0.023) | 1.014 (0.012) | 5.5 |
| `non_exceptional__n16000` | 16,000 | 1,400 | 0.997 | 0.996 | 0.953 | 0.993 (0.015) | 1.004 (0.008) | 8.7 |
| `misspecified_limit__n16000` | 16,000 | 1,400 | 1.011 | 1.010 | 0.942 | 1.008 (0.014) | 0.997 (0.007) | 7.2 |
| `optimal_known_g__n16000` | 16,000 | 1,400 | 1.009 | 1.008 | 0.955 | 1.034 (0.015) | 0.983 (0.007) | 6.0 |

The excess error variance of rule learning at `non_exceptional` is 0.13 (SD 0.09) at n = 2,000
and 0.065 (0.033) at 4,000, read from the oracle-SE ratio, and -0.026 (0.023) at 8,000 and
-0.007 (0.015) at 16,000, read from the paired variance ratio. The drop from 4,000 to 16,000 is
0.072 (0.037), about two SDs. Four sizes do not pin a rate: a weighted fit of `a n^b` leaves `b`
anywhere from -4 to 1 inside its 95% profile interval, and a constant excess fits with p = 0.06.
The excess is consistent with a second-order remainder of order `1/n` or faster.

At n = 16,000 the plan's rule reads 0.997 - 0.020 = 0.977, at least 0.975, so it gives R = 4,800
and admits the `T49` rows. At 4,800 the `misspecified_limit` cell's assurance over its pilot error
is 0.85, below the 0.9 target of rev 2.5 (B2), so R is the plan's other count, 7,200. There every
gated cell reaches the target on the unpaired reading (0.965, 0.917, 0.953; family 0.84) and on
the paired reading (1.000, 0.972, 0.996; family 0.97). The paired reading equals the SE ratio when
the limit-rule fit, a CV-TMLE of a fixed rule, has an SE ratio of 1; its pilot reads 0.99 to 1.03.
The primary floor passes with probability 0.99 at the pilot coverage and 0.77 over its 300-draw
pilot; a red floor with the 16,000-row cells green reads as finite-sample. The control's interval
lies below 0.92 with assurance above 0.999. The cells run as declared and are never downgraded.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m tests.canonical.learned_rule_ep.regenerate --jobs 4 --output <empty scratch directory>
```

Cost from the pilots' CPU seconds, with the harness refit of each fold rule and the paired
plug-in fit: the calibration cells about 89 CPU-hours (35, 29 and 24), the primary phase 9, the
diagnostic cells 7, the exceptional cell 5.5 and the control 1, about 112 CPU-hours. That is
about 29 hours on 4 cores or 7 on 16; the run is one run, and a stopped run resumes from its
checkpoints with a new empty `--output`.

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 2` also fits two draws of every property cell and prints the rule
agreement and solver warnings:

```bash
python -u -m tests.canonical.learned_rule_ep.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 2
```
