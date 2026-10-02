# Gate N3: `docs/examples/dr-tmle.ipynb` at 215545d2

Reviewed the notebook (every cell and stored output), the callback
`tests/unit/tutorial_semantics/dr_tmle.py`, the probe directory
`reviews/notebook-review/probes/dr-tmle-final/`, ledger rows DR-01 to DR-10 and DR-N1/N2, plan
row N3, `docs/development/example-notebooks.md`, `docs/technical-reference/dr-tmle/theorem.md`
and `targeting.md`, `tests/studies/canonical_drtmle.py`, `tests/canonical/drtmle/properties.csv`,
the pinned `drtmle` vignette, and `gates/gate1-findings.md` decision 3.
Scratch for this gate: `.tmp/notebook-review/gate-N3/` (not tracked).

## Verdict

The page is a working end-to-end example with no refusal cell, and every number it quotes is
reproduced. It is acceptable to demonstrate DR-TMLE on a law where the probe found no advantage,
and the page reads that evidence honestly. Two things need a fix before the gate passes: the page
describes the no-advantage result but does not explain it, and the examples index now contradicts
the page's premise. No library change is required.

## The practicality question

| question | finding | evidence |
| --- | --- | --- |
| is a tutorial on a law with no measured advantage acceptable? | yes. The method's claim is conditional (interval protection when g is wrong and the outcome fit is slow), and the page states that the probe did not show the failure the method guards against. A page that manufactured an advantage by seed or learner choice would be worse than this one | Step 7 and "How far to trust this" |
| does the page say why no advantage appears? | no. It says "the ordinary interval covered near the nominal rate, so this law does not show the failure", which restates the result. A hostile reader asks what was small | cell `compare-reading` |
| what was small? | the ordinary remainder. It is the integral of the product of the assignment-model error and the outcome-fit error (`dr-tmle/index.md`, "What this solves"). On 20 probe seeds I recomputed it from cross-fitted copies of the page's primary nuisances: RMS outcome-fit error 0.063 per arm, mean absolute propensity error 0.095, arm remainders -0.0016 and +0.0004 that partly cancel, ATE remainder -0.0019 on average. That equals the ordinary mean error -0.0020 in `summary.log`, and it is 0.3 of the mean standard error 0.0068. An interval loses little coverage to a bias of 0.3 standard errors | `.tmp/notebook-review/gate-N3/remainder.py`; `summary.log` (bias -0.00201, mean SE 0.00677) |
| what did DR-TMLE cost? | its extra fluctuations moved the estimate 0.0011 further below the truth on average (MC SE 0.00009), and the constant-reduction fit, which solves the same equations with reductions that carry no information, had the ordinary fit's bias (-0.0020) and RMSE (0.00725 against 0.00758). The fitted reductions, not the equations, carry the finite-sample cost | `summary.log` |
| is option (b), a configuration change, warranted? | no. Changing the law, the learner, or n to make the ordinary interval fail would be tuning to manufacture an advantage, and the paired coverage difference here (DR-TMLE alone covered on 1 draw, the ordinary TMLE alone on 6) is not significant (exact McNemar p = 0.125) | `sweep.csv` recomputed |

## REQUIRED FIXES

1. Step 7 reading, after "this law does not show the failure that DR-TMLE guards against": add the
   mechanism with its evidence. Minimal form: "The ordinary remainder is an integral of the product
   of the assignment-model error and the outcome-fit error. On this law the ordinary estimate ran
   0.0020 below the truth on average, 0.3 of its mean standard error 0.0068, so the term the extra
   equations remove was already small at n = 2,000. The extra fluctuations moved the estimate a
   further 0.0011 from the truth on average; that is the finite-sample cost of solving equations
   built from fitted reductions." Both numbers are in `summary.log` (mean SE 0.00677 is new, so
   add "0.0068" to `UNPRINTED_DECIMALS`). Keep the statement that the theory describes when the
   method can help and the probe did not test another sample size. Do not add a larger-n sweep to
   manufacture a gap.
2. Step 8 reading says "The constant fit passed both checks on every draw" and stops. The spline
   fit failed the score or correction check on 6 of 200 draws (`attention` nonempty on those
   seeds). Say so in the same sentence; it is the one repeated-draw fact in `summary.log` that
   cuts against the fitted reductions and the page omits it.
3. `docs/examples/index.md:16`: "a recorded assignment rule that is difficult to model" now
   contradicts the page, which says nobody recorded how the teams combined the logged variables
   (DR-08). Replace with wording such as "an assignment model whose functional form is unknown".

## Advisory, not gating

- `contract theorem` held on 180 of 200 draws; a truncation was active on 20. The Step 9 reading
  is scoped to this draw and is at the 90% threshold of `example-notebooks.md` rule 175, so it
  passes. One sentence noting the 20 bound-active draws would help the trust table's score-report
  row.
- The support reading could carry the numbers behind "a wrong assignment model can miss a region of
  poor overlap": the smallest fitted g had median 0.127 across draws while the smallest true g had
  median 0.051 (`summary.log`).

## OK

- `scripts/execute_notebook.py docs/examples/dr-tmle.ipynb --check`: "every cell reproduced its
  stored non-image output", exit 0 (`.tmp/notebook-review/gate-N3/check.log`).
  `pytest tests/unit/test_documentation_runtime.py -k dr_tmle`: 6 passed, exit 0.
- Every printed number the readings quote matches a stored output: Step 2 (0.456, 0.567, 0.404,
  0.163), Step 3 (0.195, 0.273, -0.189), Step 4 (`623e603910316e9c`), Step 6 (0.16283, 0.00676,
  (0.14958, 0.17609), [0.01471, 0.9853], the guard line, no `outcome scaled` line), Step 7
  (`True`, 0.16302, 0.00677, (0.14975, 0.17629), -0.03, 1.00), Step 8 (0.16287, 0.00678), Step 9
  (0.0%, 0.9471, spline best 3/2/1 of six for `gr1`/`qr`/`gr2`, fourth `qr` fit linear-best with
  spline weight 1.0, `theorem`, the two contract phrases), Step 10 (3.1539, 2.9596, 0.2346, 1.874;
  sqrt(exp(1.81 x 0.16283 / 0.2346)) = 1.874).
- Every repeated-draw claim matches `summary.log`: -0.0020 and -0.0031; 0.0070 and 0.0069; 0.97
  and 0.98; 187 and 182 of 200; -0.0011 (-0.00113); 83 of 200 twice (DR vs ordinary, spline vs
  constant, both 83); constant fit passed on 200 of 200; medians 0.05 and 0.17; `gr1` spline best
  in at least four fits on 138 of 200; E-value limit quantiles 2.75 and 3.09. The probe's
  `TRUTH` 0.1628580 agrees with `truth.log` (0.162827, MC SE 2.3e-5) within 1.4 MC SE.
- The probe reproduces the page's configuration (`sweep.py`: data seed, `Runtime`, both primary
  learners, both Super Learners set to the sweep seed; three folds; `q_bounds=(0.0, 1.0)`;
  `n_jobs=1`), and ran against this worktree's `src` (`sweep.log`).
- Theorem scope: "Theorem 1 makes the non-cross-fitted estimator asymptotically linear", the rate
  conditions on the outcome fit and the reduced regressions (the g-wrong branch, which is the
  page's case), the Donsker condition, and condition (S) open for a reduction that selects structure
  all follow `theorem.md` and `targeting.md`, and the stored contract line says the same. The
  reading "rate conditions for the outcome fit and the reduced regressions" is correct for a
  wrong g; the Q-wrong branch (DR-04's "both" guard) would need the propensity rate instead, and
  the page's premise excludes it.
- The ordinary-TMLE row ("remainder first order in the outcome-fit error; the interval needs that
  error to vanish faster than root-n") matches DR-03's corrected wording and the signed remainder in
  `dr-tmle/index.md`.
- Trust table vs `tests/canonical/drtmle/properties.csv`: `double_robustness/outcome_correct`
  coverage 0.9475, bias 99% interval 0.002611 to 0.007540 against margin 0.006749,
  `bias_equivalent` False; contraction rows 0.945, 0.94, 0.94375 with `bias_equivalent` True.
  `canonical_drtmle.py` confirms the construction: binary law of Benkeser et al. (2017), outcome
  GLM with the `W1:W2` term, main-effects logistic treatment, `LinearRegression` and GLM
  reductions, ten pooled folds. "It does not test a flexible outcome fit" is correct.
- Vignette claim: `using_drtmle.Rmd` at `538a3a2` lines 528 to 529 set
  `SL_gr = SL_Qr = c("SL.glm", "SL.gam")`.
- E-value per gate1 decision 3: labelled approximate, the conversion chain printed and named
  (marginal SD including the between-arm shift, Chinn 1.81, square root), and the reading states
  that it reads the estimate and interval only and does not test the doubted g.
- DR-09 support reading: the row is attributed to the fitted model, 0% below the bound agrees with
  a true g in [0.05, 0.95] (`summary.log`: smallest true g below the bound on 0 of 200 draws), and
  the sentence that the report describes the fitted model only is present.
- Callback: no old-law residue (the nu^2 witness, the catalog refusal, and the ATT refusal are
  gone). Nonzero witnesses present: shift in (-0.1, -0.01); `crude.psi != guarded.psi` beside the
  three-decimal agreement; fourth `qr` fit linear-best with spline weight 1.0; E-value chain
  recomputed from `sd(Y)`; true propensity of the draw's rows inside [0.05, 0.95];
  omitted-variable operations `unavailable`. `UNPRINTED_DECIMALS` covers every quoted sweep and
  study number.
- Ledger rows DR-01 to DR-10, DR-N1, DR-N2 are each addressed as the plan row N3 specifies.
