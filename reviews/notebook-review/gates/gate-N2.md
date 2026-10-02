# Gate N2: `docs/examples/cross-fitting.ipynb` at 5d5e08a0

Reviewed the notebook (every cell and stored output), the callback
`tests/unit/tutorial_semantics/cross_fitting.py`, the probe directory
`reviews/notebook-review/probes/cross-fitting-final/`, ledger rows CF-01 to CF-06 and CF-N1, plan
row N2 with rulings R1 to R8, `docs/development/example-notebooks.md`, and `gates/gate-N1.md`.
Scratch for this gate: `.tmp/notebook-review/gate-N2/` (not tracked).

## Verdict

The page is a working end-to-end example with no refusal cell. One sentence in Step 5 states a
behaviour the library does not have and must be corrected. No library change is required.

## The failure-mode step on a draw that covers

| question | finding | evidence |
| --- | --- | --- |
| does the page say that this draw does not miss? | yes, twice. The table prints `covers truth True` for both fits, and the reading says "Both intervals contain the true ATE" before it quotes the probe | cell `failure-mode` output; cell `failure-mode-reading` |
| does the step still show the named mechanism? | yes. The heading names "an interval that is too narrow", and the narrowness is a per-draw observable: the in-sample SE was smaller on 200 of 200 draws, between 0.68 and 0.73 times the cross-fitted one on 90% of them, and 0.73 on this draw. The coverage consequence (153 vs 188 of 200) is correctly placed in the probe, not in the draw | `summary.log`; `sweep.csv` recomputed |
| is "too narrow" the right diagnosis, rather than "too variable"? | yes. The two estimators have nearly equal empirical SDs (0.0057 in-sample, 0.0060 cross-fitted); the in-sample mean SE is 0.0040 against 0.0056. The reading's two causes (SE too small, estimate -0.0024 low, 6 MC SEs from zero) are both real and both quoted with their source | `summary.log` |
| is a different presentation needed? | no. A draw that misses would show the consequence but prove nothing more than this one does, and choosing it would be seed-shopping. The present form (relation on the draw, rate from the probe, "nearly misses" at 1.9 SE) is honest and useful. R3 and R4 are satisfied | plan rulings |

## OK

- `scripts/execute_notebook.py docs/examples/cross-fitting.ipynb --check`: "every cell reproduced its stored non-image output", exit 0 (`.tmp/notebook-review/gate-N2/check.log`).
- Every printed number the readings quote matches a stored output: Steps 2 (0.163), 3 (`623fc2c240615d26`), 5 (0.15731, 0.005444, [0.14664, 0.16798], both fingerprints), 6 (0.1553, 0.0040, 0.1573, 0.0054, 0.73, 1.9, 1.0, the four interval ends), 7 (0.1237, 0.0195, 0.1256, 0.0301, 1.54, 0.104, `7df34d840b4e8d1a`), 8 (1, 4, 60), 9 (fingerprints, 0.15795, 0.15721), 10 (fifth-decimal differences), 11 (0.9611, 0.0239, 82.5%, 89.6%, 0.0063), 12 (0.1567, 0.0055, 0.0004, 0.0679), 13 (0.018093, 4.5363, 0.419, 0.398, 0.28648, 0.0087264, both bound pairs).
- Every repeated-draw claim matches `summary.log` or `truth.log`: 188 and 153 of 200; SE/SD 0.93 and 0.70; biases +0.0003 and -0.0024; SE-ratio band 0.68 to 0.73 (5% and 95% quantiles); 199 of 200 with no attention row; slope band 0.90 to 1.01; 196 of 200 below the law's share; in-sample ESS ratio above on 200 of 200; nu^2 mean 4.504 ("4.50"), below 4.83 on 195 of 200, shortfall 6.7% ("about 7%") and 3.4% in the bias scale ("about 3%"); limits exclude zero on 200 of 200; 1.9% of true g below 0.1 (0.0188); 46% of teams negative (0.4602).
- The probe reproduces the page's configuration: `sweep.py` sets the data seed, `Runtime`, and both learners' `random_state` to the sweep seed, with the same booster hyperparameters, five folds, `q_bounds=(0.0, 1.0)`, and `n_jobs=1`.
- Truth recomputation: `truth.py` types the structural equations without importing `cleverly`. Its propensity `0.05 + 0.90 expit(u)` and outcome mean match `synthetic.py:638` and `:713-723`; the navigation ATE 0.162860 (MC SE 1.9e-5) and the team ATE 0.104123 (MC SE 9.5e-5) agree with the package truths 0.1628580 and 0.1040497 within one MC SE. The team mean `expit(-0.4 + 0.8a + 0.5W1 + 0.3W2 + 0.6u + 8.0au)` matches `synthetic.py:1268-1270`, and the propensity `expit(0.3W1 + 0.6W2)` (`:1260`) excludes the team factor, so "does not change who receives an offer" and "main-terms logistic model in the two covariates" are both true.
- Step 13 nu^2 reading follows the N1 ruling: mechanism (expected value is the law's nu^2 minus the squared representer error), size (7%, 3%), direction for both outputs (bounds narrow, RV high), the shown-draw recomputation (RV 0.409, bounds [0.1483, 0.1663]) is in `summarize.py` and `summary.log`, and the callback asserts `0.9 * law_nu2 < elements.nu2 < law_nu2` and reproduces 0.409 from the fit's own formula. The identity `E[1/(g(1-g))]` = mean inverse conditional treatment variance is correct.
- Trust table versus artifacts: `crossfit_overfitting/stacked_cvtmle` 0.945 and `in_sample_control` 0.5225 match `tests/canonical/tmle3_cvtmle/properties.csv` (n = 500, 400 replicates); `clustered_inference/cluster_robust` 0.9475 and `iid_control` 0.84125 match `tests/canonical/lmtp_clustered_tmle/properties.csv` (n = 2000, 2400 replicates). Study designs confirmed in `tests/studies/cvtmle_properties.py` (`DecisionTreeRegressor(min_samples_leaf=1)`, `LogisticRegression`, ten folds, `Q_BOUNDS=(0.0, 1.0)`, `G_BOUNDS=(0.025, 0.975)`) and `tests/studies/canonical_clustered_tmle.py` (`clustered_dgp(cluster_size=10, family="binomial")`, `OracleTreatment`, unpenalized main-effects logistic Q, `N_FOLDS=5`). The five R5 items are named in every row.
- Cluster statements: the formula sentence matches `inference.md:169-170` and reduces to the row formula at `inference.md:15` with singleton clusters; the 40-cluster rule and the equal-size requirement for a cross-fitted interval match the status table (`inference.md:194-195`); 200 teams of 15 are equal, so `inference_status == "influence_curve"` is the right witness.
- Step 9 replaces the other-data refusal with a logistic g on the same plan. The table shows the plan reproducing the Step 5 point and influence curve exactly under a new seed, and the reading says the comparison "does not show which treatment model is right". Reproducible wiring, correctly scoped.
- Step 7 states the team law as a separate law with its own truth (CF-04), the binary support positively (CF-06), the 40-team and equal-size requirements positively, and prints the team protocol fingerprint; the callback asserts `changed_fields == {"outcome"}`.
- CF-02: "addresses data reuse only" in `plan` and `where-next`; no claim that the in-sample estimate is not more biased. CF-03: the ESS reading names the fitted model and what it cannot see. CF-05: `cf_d` in the share form that the output prints. CF-N1: the trust row leads with 0.5225 versus 0.945.
- No refusal cell. Step 13 runs `robustness_value()` and `omitted_confounding()` with no `try`. The callback has no refusal pins.
- `UNPRINTED_DECIMALS` lists every quoted sweep, truth, and study decimal with its source (R8). `pytest tests/unit/test_documentation_runtime.py tests/unit/test_documentation_links.py -k "cross and fitting"`: 12 passed, exit 0, including the semantics callback and the narrated-decimal test (`.tmp/notebook-review/gate-N2/pytest.log`).

## REQUIRED FIXES

1. `estimate-heading` (Step 5), the `q_bounds` paragraph. The sentences

   > Without it, the fit reads the outcome scale from every observed score. The held-out rows of each fold then help set the scale of their own predictions.

   describe a behaviour the library does not have. `TMLE._outcome_scale_refusal` (`src/cleverly/estimators/tmle.py`, the block ending at line 1780) raises `CapabilityError` for a cross-fitted continuous outcome with `q_bounds=None`; confirmed by a direct fit in this gate (`.tmp/notebook-review/gate-N2/`). A reader who omits `q_bounds` gets a refusal, not a fit on a leaked scale. Replace with text that names the requirement and the reason, for example:

   > Without it, the fit refuses to run. The alternative would read the outcome scale from every observed score, so the held-out rows of each fold would help set the scale of their own predictions.

   Keep the rest of the paragraph. No callback change is needed, and the Markdown edit does not move the stamp.

## Not required, recorded

- `team-clusters-reading`: "The interval uses a normal reference with no small-sample cluster correction." The variance does carry the $m/(m-1)$ factor (`inference.md:169`). The sentence is read here as "no $t$ reference", which is what `inference.md:195` says. Writing "a normal reference, not a $t$ reference" would remove the ambiguity.
- `failure-mode-reading`: "so that interval nearly misses" is a one-draw relation at 1.9 SE and is tagged by its own sentence. No change.
