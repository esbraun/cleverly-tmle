# Gate N1: `docs/examples/point-treatment-tmle.ipynb` at a9684c17

Reviewed the notebook (every cell and stored output), the callback
`tests/unit/tutorial_semantics/point_treatment_tmle.py`, the probe directory
`reviews/notebook-review/probes/point-treatment-tmle-final/`, ledger rows PT-01 to PT-09, PT-N1,
PT-N2, plan row N1 with rulings R1 to R8, and `docs/development/example-notebooks.md`.
Scratch for this gate: `.tmp/notebook-review/gate-N1/` (not tracked).

## Verdict

The page is a working end-to-end example with no refusal cell. One paragraph needs a wording fix.
No library change is required.

## The nu^2 concern

| question | finding | evidence |
| --- | --- | --- |
| is 4.83 the right target? | yes. `truth.py` computes `E[1/g + 1/(1 - g)]` on the untruncated true `g`. The estimator's representer is built from the truncated fitted `g`, but the default bound truncates 0 rows on 200 of 200 draws, so the two targets coincide here | `truth.log` 4.8286 (MC SE 4.4e-4); `summary.log`; `omitted_variable.py:645-694` |
| why does it read low on 197 of 200? | by construction, not by accident. The default estimator is `E_n[2 m(alpha_hat) - alpha_hat^2]`, whose expectation is `nu_0^2 - E[(alpha_hat - alpha_0)^2]` (the code's own comment at `omitted_variable.py:695-700`). It reads below `nu_0^2` whenever the fitted `g` has any error, and the fit's regularized booster underfits the `-0.4 W2^2` term | identity verified: the sample squared error of the representer equals the shortfall at every `n` in the gate probe |
| how large? | mean 4.479 against 4.829, a 7.2% shortfall in `nu^2`, so the bias scale `sqrt(sigma^2 nu^2)` is 3.7% low on average. Per-draw SD of `nu^2` is 0.22, so the shortfall is about 1.6 per-draw SDs | `sweep.csv`, recomputed in this gate |
| does it shrink with `n`? | slowly, and it does not vanish for this learner. Shortfall 0.80 at n = 1000, 0.30 at 3000, 0.24 at 12000 (16 seeds each; 48 of 48 draws below 4.83). The plug-in `E_n[alpha_hat^2]` is no substitute: 5.66 at n = 1000, 4.45 at 12000 | `.tmp/notebook-review/gate-N1/nu2_scaling.{py,log,csv}` |
| what does it do to the page's readings? | on the shown draw, at the law's `nu^2` the bias scale is 0.29124 not 0.27859, the bound on the bias is 0.0244 not 0.0233, the lower bound is 0.1422 not 0.1433, and RV is 0.431 not 0.445. Across the 200 draws the mean RV moves from 0.427 to 0.415. Every repeated-draw claim survives: the one-sided limits exclude zero on 200 of 200 at the law's `nu^2` too | recomputed from `sweep.csv` with the page's bound formula (RV recomputation matches the probe's `rv` to 4 decimals) |
| library issue? | no. The estimator is the cited paper's, it is consistent, and its influence curve is derived. Optional: the `nu2_estimator` docstring at `omitted_variable.py:539` says "less sensitive to error" and could state the sign of the second-order bias, which the code comment already does | not required for N1 |

## OK

- `scripts/execute_notebook.py docs/examples/point-treatment-tmle.ipynb --check`: "every cell reproduced its stored non-image output", exit 0 (`.tmp/notebook-review/gate-N1/check.log`).
- `pytest tests/unit/test_documentation_runtime.py tests/unit/test_documentation_api.py -k point_treatment`: 5 passed, including the semantics callback and the narrated-decimal test. `ruff check` and `ruff format --check` pass on the notebook. `test_documentation_prose.py`: 16 passed.
- Every printed number the readings quote matches the stored output (Steps 2, 3, 6, 7, 8, 9, truncation curve, 10, 11), including 0.0114, 0.070385, 0.9564, 0.0027, 0.0417, 0.1674, 4.418, 0.445, 0.425, and the benchmark and bound values.
- Every repeated-draw claim matches `summary.log`: 189 of 200, SE/SD 0.98, 190 and 195, 192, 30, default bound clips none on 200 of 200, 0.1 clips on 200 of 200, `cf_d` 0 to 0.456, limits exclude zero on 200 of 200, 197 of 200 below 4.83, about 1.9% of true `g` below 0.1 (0.0188).
- R3: no multiplier on a single-draw relation. The both-linear exclusion (holds on 85% of draws) quotes printed numbers only. "On this draw" tags are present where needed.
- Truth recomputation: `truth.py` types the structural equations without importing `cleverly`; its five means agree with the package truth within 1 MC SE. The `0.05 + 0.90 expit(u)` propensity matches `synthetic.py:638`.
- Trust table: `crossfit_overfitting/stacked_cvtmle` 0.945 and `in_sample_control` 0.5225 match `tests/canonical/tmle3_cvtmle/properties.csv`. Confirmed in `tests/studies/cvtmle_properties.py`: `nonlinear_dgp()` through `bounded_twin` at concentration 12.0 (the page's law; truth 0.16286 identical), `DecisionTreeRegressor(min_samples_leaf=1)` for Q, `LogisticRegression()` for g, n = 500, 400 replicates, `n_folds=10`, `G_BOUNDS=(0.025, 0.975)`. The five R5 items are named.
- Truncation witness: 0 rows truncated at every bound up to 0.05; 0.1 clips 0.0027 with no four-decimal move; 0.2 clips 0.0417 and moves the estimate by +0.0008. The callback asserts positive truncation at 0.1, monotone truncation at 0.2, and a move above 5e-4 (a nonzero witness). The default-bound rule `5 / (sqrt(n) log n)` is `utils/bounds.py:122`.
- Learner rationale matches `example-notebooks.md:147` (bare booster below 10,000 rows) and `learner-experiment.md` (slope 0.96, coverage 0.960). The sweep confirms the calibration slope 5% to 95% range [0.91, 1.02].
- No refusal cell. Step 10 runs `robustness_value()`, both benchmarks, and `omitted_confounding(rho=1.0)` with no `try`. The `discharge_risk` clip is disclosed and the bound uses the second covariate.
- `UNPRINTED_DECIMALS` lists each quoted sweep and study number with its source (R8). All twelve link targets exist.

## REQUIRED FIXES

1. `sensitivity-reading`, the `nu^2` paragraph. Replace

   > The assessment in Step 9 printed the doubly robust $\nu^2$ as 4.418, below the law's 4.83. The estimate fell below 4.83 on 197 of the 200 probe draws. Step 9 shows that the fitted propensities miss part of the law's tail. These bounds can therefore be slightly narrower than bounds built on the law's $\nu^2$.

   with text that states the mechanism, the size, and the direction for both outputs. Suggested:

   > The assessment in Step 9 printed the doubly robust $\nu^2$ as 4.418, below the law's 4.83. This is the estimator's design, not an accident of the draw. The doubly robust form equals the law's $\nu^2$ minus the squared error of the fitted Riesz representer, so it reads low whenever the fitted propensity has any error. The shallow booster underfits the law's quadratic term, and Step 9 shows that it misses part of the tail. On the 200 probe draws the estimate averaged 4.48, about 7% below 4.83, and fell below 4.83 on 197 of them. The bias scale is therefore about 4% low, and the robustness values read slightly high. At the law's $\nu^2$ the robustness value on this draw would be 0.431 rather than 0.445, and the lower bound 0.142 rather than 0.143. The one-sided limits excluded zero on all 200 draws under either $\nu^2$.

   Then add to `UNPRINTED_DECIMALS`: `"4.48"` (mean DR `nu^2` over 200 draws, `summary.log`), `"0.431"` and `"0.142"` (RV and lower bound of the shown draw at the law's `nu^2`; put the two-line recomputation in `summarize.py` and `summary.log`, or cite this gate's recomputation after moving it into the probe directory). Keep the callback's existing `0.0 < elements.nu2 < law["nu2"]` assertion; add `assert elements.nu2 > 0.9 * law["nu2"]` so a shortfall above 10% fails loudly.

   Do not add a comparison that implies the shortfall vanishes with `n`. The gate probe shows it falls from 0.80 to 0.24 between n = 1000 and 12000 and then flattens for this learner.

## Not required, recorded

- `omitted_variable.py:539` docstring: state that the doubly robust estimator's expectation is `nu_0^2` minus the squared error of the fitted representer (second-order, downward). The code comment at lines 695-700 already says this; the public text says only "less sensitive to error".
