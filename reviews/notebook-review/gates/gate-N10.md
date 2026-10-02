# Gate N10: `docs/examples/twins-causal-inference.ipynb` at `aaf4c484`

Scope: the notebook (cells and stored outputs), `tests/unit/tutorial_semantics/twins_causal_inference.py`,
`docs/references.md`, `reviews/notebook-review/probes/twins-causal-inference-final/`, ledger rows TW-01 to
TW-N3, the TWINS verification file, `docs/development/example-notebooks.md`. Read-only on tracked files.

## Runs

| command | result |
| --- | --- |
| `scripts/execute_notebook.py docs/examples/twins-causal-inference.ipynb --check` | exit 0; "every cell reproduced its stored non-image output" (networked, Python 3.11.13 venv) |
| `pytest -q -p no:cacheprovider tests/unit/test_documentation_runtime.py -k twins` | 5 passed |

## OK

- Estimand and population. The protocol names the population as children from same-sex US twin births,
  the two threshold strategies, time zero, the pair as the interference unit, and the reasons
  exchangeability and consistency fail. The page calls the target illustrative and does not generalize
  beyond twins. No real-data truth is claimed.
- Hand-built TMLE (cell 18). Package bounds (g at 5/(sqrt(n) log n) = 0.004859 at n = 12000; Q in
  [0.0005, 0.9995]), two arm clever covariates, two-dimensional Newton step, counterfactual-update
  identity asserted, nonzero witnesses (scores 4.79e-04 and -3.27e-05, epsilon nonzero, Q bound active
  on 88 children). Package and hand both print 0.057003; `|gap|/SE < 0.001` asserted. `summary.log`:
  max gap/SE 1.01e-04, below 0.001 on 31/31 pair samples. The prose claim matches.
- Booster. `max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0`. The sentence on
  scikit-learn early stopping (only above 10,000 rows) is correct; every booster fit here sees fewer.
  `summary.log` (shallow): sensitivity ran 31/31, nu^2 4.091 to 4.224, cf_y <= 0.0019, cf_d 0.0023 to
  0.0225, calibration flags 14/31, all as narrated. The "earlier booster" account (negative nu^2 on 2 of
  31: seed 11 main fit, seed 5 benchmark refit) matches `sweep-current.csv`.
- Omitted-variable readings recompute. Bias scale sqrt(0.033464 x 4.1746) = 0.3738; bound at
  cf_y = cf_d = 0.05, rho = 1: 0.3738 x sqrt(0.0025/0.95) = 0.01917; RV solves cf^2/(1 - cf) =
  (0.057448/0.37376)^2 to 0.1423; implied cf_y = (0.033482 - 0.033464)/0.033482 = 0.0006 and
  cf_d = (4.1746 - 4.141)/4.1746 = 0.0081. The cf_d wording (share of the Riesz representer's second
  moment the confounder adds) is the general definition, not the PLM special case. Stable claims
  (cf_y at or below 0.0019) are separated from sample-specific ones (cf_d 0.0081 "on this pair sample").
  E-values 26.41 and 17.17 follow from RR 13.4607 and 8.8426.
- Stress test. All 14,870 discordant pairs of the 71,345-pair file; twin 0 asserted lighter and
  therefore the sub-threshold twin in every discordant pair; the multinomial-count bootstrap is exactly
  pair resampling for differences in {-1, 0, 1}. The reading says it has another target.
- LTMLE truth. `exact_regime_risk` integrates L2 and averages over the empirical W2, which is the only
  baseline node Y and L2 use; asserted against -0.06464265455218184.
- Trust rows. The clustered row matches `clustered-point-treatment-cv-tmle.md`: binary law, clusters of
  ten, five grouped folds, exact propensity, unpenalized logistic (`C=1e6`), cell
  `clustered_inference/cluster_robust` exists (line 105). The uncovered list includes risk-ratio
  intervals (TW-N2). The LTMLE row names the correct study and its law.
- Citations. The published NeurIPS 2017 PDF has "4.3 Binary treatment outcome on Twins" (verified from
  the proceedings PDF). Almond, Chay and Lee, QJE 120(3):1031-1083, DOI 10.1162/003355305774268228.
  `docs/references.md` lines 1810-1827 agree with the notebook.
- Formats and pins. Every narrated decimal is either printed with an explicit format or listed in
  `UNPRINTED_DECIMALS` with its probe source. No raw solver residual is printed; `_MACHINE_NOISE`
  guards three outputs. `check_stored()` pins the fingerprint flow, the two bounds, the witnesses, the
  gap relation, the clever-covariate identity, the sign-survives line, sigma^2 ordering, log-symmetry of
  the ratio interval, the E-value relation, and the LTMLE containment lines.

## REQUIRED FIXES

1. Refusal showcase in section 8 (cells 26 and 27). `assessment.summary()` prints the `Not run` block
   with `unavailable 1 sensitivity.simulated_confounding`, and the reading's table narrates it
   ("`simulated_confounding` is `unavailable` for a clustered fit, so section 9 uses the
   omitted-variable bound"). `docs/development/example-notebooks.md` line 74: a step that prints a
   capability list with unavailable rows prints only the rows that run. The refusal inventory applies
   the same rule to the interventions `assessment` cell. Replace: print `assessment.to_frame()`
   filtered to rows whose status is not `unavailable` or `not_applicable` (keep `needs attention` and
   the nuisance and score lines), drop the `unavailable` narration from the table, and open section 9
   positively ("Section 9 reads the omitted-variable bound and its benchmark"). Update the
   `check_stored()` pin `"passed  1      validation.score_equations" in diagnostics` to the new print.
   The underlying stop is roadmap F9 (`_simulated_confounding_request.py:151`), not a property of the
   data, so it does not belong in a real-data tutorial.
2. Section 11 reading (cell 40) draws a one-sided conclusion from the probe. It reports that the
   interval contains the naive contrast on 56 of 60 draws but omits the same probe's truth coverage,
   55 of 60 (`summary.log`, `ltmle-sweep.csv`), which is the statistic the section's title and the
   learn-table row ("check a longitudinal estimator against an exact truth") promise. State both
   counts in one sentence, and reconcile the next paragraph: "does not establish repeated-sampling
   coverage" now sits beside a 60-draw sweep. Say instead that the sweep is conditional on the 6,000
   sampled baseline rows and gives 55 of 60 at a nominal 95% level. Optional, larger: strengthen the
   A1 -> L2 and L2 -> Y paths so the naive contrast leaves the interval on most draws; that moves the
   truth assertion and every stored section-11 output.

No other finding. The remaining TW rows (TW-01 to TW-12, TW-N1 to TW-N3) are closed by the commit as
described in its message.
