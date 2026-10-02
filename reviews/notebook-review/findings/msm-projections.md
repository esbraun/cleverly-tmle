# Review: `docs/examples/msm-projections.ipynb`

| item | value |
| --- | --- |
| notebook | `docs/examples/msm-projections.ipynb` (callback `tests/unit/tutorial_semantics/msm_projections.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), `cleverly.__file__` checked under this worktree's `src` |
| `--check` | exit 0, wall 9 s; "every cell reproduced its stored non-image output" (`.tmp/notebook-review/msm-projections/check.log`) |
| seed sweep | `.tmp/notebook-review/msm-projections/sweep.py` over seeds 1..600 at n = 3000, the notebook's exact method (log `sweep.log`, rows `sweep.csv`); seed 61 alone reproduces every printed number (`sweep61.log`) |
| n probe | `.tmp/notebook-review/msm-projections/probe_n.py 12000 1001 1301` (300 seeds at n = 12000, log `probe_n12000.log`) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `n_jobs=1`; total sweep compute about 4 minutes |

Independent truth (`sweep.py` lines 30-45, not the library helper): from `multi_arm_dgp`
(`src/cleverly/datasets/synthetic.py:1602-1668`), Q(a, W) = 0.6 step[a] + W1 - 0.5 W2 + 0.2 W3
with step = (0, 1, 2.4) and W ~ N(0, I), so E[Y(a)] = (0, 0.6, 1.44) exactly; a 2e6-draw Monte
Carlo with an independent RNG gives (-0.0008, 0.5992, 1.4392). The uniform-weight projection onto
[1, contacts] with contacts (1, 2, 6) is intercept -0.117143, slope 0.265714 = 3.72/14, and the
population miss at `medium` is -0.185714. The true arm probabilities P(A = a) are
(0.2777, 0.3599, 0.3625), and the P(A = a)-weighted slope is 0.26002.

## Findings

### MS-01 — `failure-reading`: "Its estimate differs from the Step 7 miss in the fourth decimal"

- Problem: On the shown draw the two numbers are -0.2084 and -0.2092. They differ in the third
  decimal (difference 0.00079). Over seeds the difference has no fixed size.
- Severity: **wrong**
- Evidence: [executed] seed 61: `mis_psi` -0.208388, Step 7 miss -0.209174, difference 0.000786
  (`sweep61.log`). [recomputed] over 600 seeds, |difference| < 0.0005 (a true fourth-decimal
  difference) in 52.2%; max 0.0067 (`sweep.log`). [read] the callback pins only
  `0.0 < abs(...) < 0.005` (`tests/unit/tutorial_semantics/msm_projections.py:199-201`), so it
  accepts a third-decimal difference and does not test the sentence.
- Proposed fix: "Its estimate differs from the Step 7 miss by 0.0008 on this draw, because the arm
  fit and the MSM fit target separately." Tighten the callback to the printed magnitude, or drop the
  decimal-place claim. Files: notebook cell `failure-reading`, callback. Library bug: no.
  Registered study can move: no.

### MS-02 — `trust` / `estimate-arms-heading`: "Correctly specified parametric learners are what makes that condition plausible on this page." / "correctly specified parametric learners need no cross-fitting"

- Problem: The page presents its 95% intervals as resting only on the in-sample (Donsker) condition,
  which correct parametric learners satisfy. At n = 3000 the slope interval covers the projection
  truth 92.5% of the time, with the reported SE about 8% too small. The simultaneous band and the
  `medium` arm interval under-cover too. The cause is finite-sample practical positivity
  (fitted g down to 0.0006, ESS/n 0.48 for `high` on the shown draw), not the cross-fitting
  condition the page names. The page never tells the reader that the nominal level is not reached
  at this design.
- Severity: **misleading**
- Evidence: [recomputed] 600 seeds at n = 3000 with the notebook's method: slope pointwise coverage
  0.925 (binomial MC SE 0.009, so 2.8 SE below 0.95); slope SE ratio (mean SE / empirical SD)
  0.00906 / 0.00984 = 0.921; joint simultaneous-band coverage 0.927; intercept coverage 0.945;
  arm coverage low 0.948, medium 0.923, high 0.937; slope bias -0.00003 (negligible). [recomputed]
  300 seeds at n = 12000: slope coverage 0.947, SE ratio 1.001, so the shortfall is finite-sample.
  [executed] shown draw support table: g[high] 0% quantile 0.0006, 1.20% truncated.
- Proposed fix: In the trust cell, add one measured sentence: "Over 600 draws of this law at
  n = 3000, the slope interval covered the projection 92.5% of the time, and 94.7% at n = 12000.
  Small fitted probabilities of `high` and `medium` cause the shortfall." Keep the sweep as a
  committed script or note. Files: notebook `trust` cell (and the callback if the sentence is
  pinned). Library bug: not shown (consistent with near-positivity finite-sample behavior; the
  in-sample IC variance ignores truncation and fitted-g variability). Registered study can move: no.

### MS-03 — `share-weight-reading`: "an estimated function declared as known still fits. That fit reports a standard error that is too small."

- Problem: The direction is not universal. The library's own refusal text says the reported
  standard error "can be too small ... its direction is not universal", and the cited witness has
  one coefficient whose ratio is exactly 1.
- Severity: **misleading**
- Evidence: [read] `src/cleverly/msm.py:323` ("its direction is not universal") and
  `src/cleverly/msm.py:331-336` ("its standard error can be wrong");
  `docs/technical-reference/msm-projections.md:213-222` (ratios 0.742, 0.913, and 1.000 for
  `msm[a]`).
- Proposed fix: "That fit reports a standard error that omits the weight's term. It can be too
  small: in `tests/unit/test_msm_projection_weights.py` it is 0.742 of the correct value for one
  coefficient." Files: notebook cell `share-weight-reading`. Library bug: no. Study: no.

### MS-04 — `share-weight` / `share-weight-reading`: "The uniform weight gives a population slope of 0.2657. The share weight gives 0.2593. ... only the weight moved the slope."

- Problem: The identity is correct, but the illustration moves the slope by less than one standard
  error. On the shown draw 0.2593 lies inside the fitted slope interval (0.2519, 0.2856), so the
  data cannot tell the two estimands apart. A reader can conclude the weight choice is immaterial,
  which is the opposite lesson. The step also mixes population means with sample shares; the
  population share weight P(A = a) gives 0.2600. The page says "only an illustration", so this is
  presentation, not error.
- Severity: **weak**
- Evidence: [recomputed] |share slope - uniform slope| / slope SE averages 0.63 over 600 seeds; the
  fitted slope interval covers the share-weight target in 88.5% of seeds; P(A = a) = (0.2777,
  0.3599, 0.3625) gives slope 0.26002. [recomputed] the callback's own 1:10:1 weight gives slope
  0.240, about 3 SE from 0.2657 (`msm_projections.py:120-140` uses it as the witness).
- Proposed fix: State that the share-weight shift is smaller than the slope's standard error here
  because the shares are near equal, and add the 1:10:1 weight (slope 0.2400) as the contrast that
  shows a material move. Files: notebook `share-weight`, `share-weight-reading`; callback numbers.
  Library bug: no. Study: no.

### MS-05 — `assessment-reading`: "one fluctuation equation and one influence-curve row per coefficient, not one per arm"

- Problem: Step 7 says the fluctuation "solves one score equation per coefficient". The `msm
  fluctuation` row is the norm of that two-component score, not one equation. The two sentences
  read as contradictory.
- Severity: **weak**
- Evidence: [read] `src/cleverly/validation/score.py:587-592` (`score = fluctuation.score_norm`,
  one row per fluctuation group); [executed] the score table shows one `msm` fluctuation row and
  two influence-curve rows.
- Proposed fix: "one row for the fluctuation's two-coefficient score, reported as its norm, and one
  influence-curve row per coefficient". Files: notebook `assessment-reading`. Library bug: no.

### MS-06 — `trust`: the registered study's nuisance construction is not named

- Problem: The trust cell lists the study's limits (ordinary targeting, three terms, two arms,
  pointwise intervals). It does not say the study's `cleverly` arm uses the exact (oracle) outcome
  regression and treatment mechanism with fixed bounds [0.01, 0.99]. This page fits
  `LinearRegression` and a multinomial `LogisticRegression` with the data-adaptive bound 0.0114.
  The study therefore says nothing about fitted-nuisance or multinomial-g behavior, which is where
  MS-02's shortfall arises.
- Severity: **weak**
- Evidence: [read] `tests/studies/canonical_point_msm.py:186-190` (`OracleOutcomeContinuous`,
  `OracleTreatment`, `g_bounds=G_BOUNDS`); `tests/canonical/tmle3_msm/manifest.json`
  (`g_bounds: [0.01, 0.99]`); `docs/technical-reference/method-evidence/point-treatment-msm-projection.md`
  "nuisance fits | exact conditional means and treatment probabilities".
- Proposed fix: Add "with exact nuisances and fixed bounds" to the trust cell's list of study
  limits. Files: notebook `trust`. Library bug: no. Study: no.

### MS-07 — callback comment drift

- Problem: The callback comment quotes "1.23% of the units"; the notebook prints and narrates 1.20%.
- Severity: **weak**
- Evidence: [read] `tests/unit/tutorial_semantics/msm_projections.py:214`; [executed] notebook
  output "truncated: 36 unit(s) (1.20%)".
- Proposed fix: update the comment to 1.20%. Files: callback. Library bug: no.

## Checked and sound

- [recomputed] True means 0, 0.6, 1.44 and gains per contact 0.60 and 0.21: exact from the
  structural equations; Monte Carlo agrees to 0.001.
- [recomputed] Projection intercept -0.1171, slope 0.2657, and the fixed contrast
  (-2, -1, 3)/14 (contacts centred at 3, sum of squares 14). Population line 0.1486, 0.4143, 1.4771;
  misses 0.1486, -0.1857, 0.0371.
- [read] The estimand in code matches the prose and Rosenblum & van der Laan (2010):
  beta = argmin E sum_a h(a,V){Qbar(a,W) - phi(a,V)'b}^2 with h = 1, the sum over the three arms
  rather than over P(A = a) (`src/cleverly/msm.py:1020-1021`, Gram `P_n sum_a h phi phi'`;
  `docs/technical-reference/msm-projections.md:33-41`). `weights=None` equals an explicit all-ones
  weight (callback (a), executed in the fast tier).
- [recomputed] The working model misses the means: the population miss at `medium` is -0.1857, and
  the `contrast` interval excluded zero in 600/600 seeds and covered -0.1857 in 93.8%.
- [recomputed] The reported slope targets the projection, not a naive slope: mean slope over 600
  seeds 0.26625 vs truth 0.26571 (bias -0.00003); the observed-means slope on seed 61 would be
  0.090.
- [recomputed] "Observed means put medium above high", and both confounders push that way: 600/600
  seeds for each of the three inequalities.
- [read] Confounder signs: W1 enters Q with +1 and W2 with -0.5; W1 and W2 enter the arm logits;
  W3 enters Q only (`synthetic.py:1647-1658`).
- [read] "Correctly specified" learners: Q is additive in arm indicators and W, and the outcome
  learner sees K-1 indicators plus W (`src/cleverly/data/causal_data.py:741-755`); g is a softmax
  linear in W, which multinomial logistic regression represents (with a negligible default L2
  penalty at n = 3000).
- [recomputed] Positivity: softmax probabilities are strictly positive but not bounded away from
  zero (minimum over 2e6 draws: 0.0099, 0.00018, 0.00014 for low, medium, high). The page's
  "partly, through the support report" is accurate; MS-02 records the finite-sample consequence.
- [recomputed] IPW with stabilized weights corresponds to h(a) = P(A = a); the population
  P(A = a)-weighted slope is 0.2600, distinct from 0.2657.
- [executed] Saturated control: intercept and both contrasts equal `ey[low]` and the two ATEs, with
  influence curves equal within 1e-12 (callback asserts rtol/atol 1e-12).
- [recomputed] Arm-mean point estimates are unbiased (biases -0.0024, 0.0022, -0.0011 against SDs
  0.036 to 0.044).
- [recomputed] Seed-tagged readings in Step 12 are labelled "on this draw" and are correct for
  seed 61. Over 600 seeds: the support warning appears in 82% (attention is empty in 18%); the
  narrowest arm is `high` in 55% and `medium` in 45%; the largest clever covariate comes from `high`
  in 98.7% and always equals contact / min clipped g (600/600). The callback pins the seed-61 values
  (`attention == {"support"}`, narrowest `high`); that is acceptable only because the prose is
  seed-tagged.
- [read] The support warning threshold is a truncated share above 1% (`src/cleverly/sensitivity/positivity.py:717-723`).
- [recomputed] Truncation curve: max |slope movement| averages 0.0042 over seeds (max 0.0157); the
  seed-61 values 0.2654 to 0.2700 and 0.0033 match.
- [recomputed] Robustness values: `ate[medium vs low]` below `ate[high vs low]` in 600/600 seeds
  (means 0.204 and 0.417); rva < rv in the shown draw.
- [executed] The `MSM.linear` refusal and the MSM sensitivity refusal print the quoted tokens.
- [read] The trust cell's study claims match the page and the artifact: identity link, fixed
  nonuniform weights `1 + 0.5 A + 5 W`, three terms, two arms, pointwise only, ordinary
  targeting; `tests/canonical/tmle3_msm/summary.csv` gives `msm[a]` coverage 0.94625 and SE ratio
  0.997, matching the generated table.
- [read] Literature: the notebook cites none directly. `docs/references.md:1401-1406` gives the
  published DOIs (JSPI 10.1016/j.jspi.2005.12.008; IJB 10.2202/1557-4679.1238), which match the
  published articles from reviewer knowledge; not web-verified.
- [read] Practicality: per-assigned-contact framing, explicit contact mapping, refusal of label
  coding, the "do not extrapolate to ten contacts" warning, and the saturated fallback are what a
  regional plan's program board needs. The slope is read as a projection, never as a
  dose-response effect.
- [executed] Seeds: `make_multi_arm(seed=61)`, `Runtime(random_state=61)` (multiplier bands), and
  `LogisticRegression(random_state=61)`; `--check` reproduces every output.

## Patterns for siblings

- A "differs in the Nth decimal" sentence pinned by a loose callback bound (`< 0.005`) is untested
  in practice. Check the printed digits and the bound together.
- "Correct parametric learners make the in-sample condition plausible" does not imply nominal
  coverage under weak practical positivity. A 300-600 seed sweep at the page's n is cheap (0.25 s
  per fit here) and exposed a 92.5% coverage that the trust cell does not mention.
- Prose that states a direction ("too small") for an omitted influence-curve term should match the
  library's own refusal text, which here says the direction is not universal.
- An illustration of "X is part of the estimand" should move the number by more than its standard
  error, or say that it does not.
- Trust cells should name whether the cited study used oracle nuisances.
