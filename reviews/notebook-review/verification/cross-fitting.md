# Verification: docs/examples/cross-fitting.ipynb

- Verifier scratch: `.tmp/notebook-review/verify-cross-fitting/` (`truths.py`, `sweep.py`, `analyze.py`, `s[abc].csv`, `lss_v6.txt`).
- `cleverly.__file__` asserted under this worktree's `src` inside `sweep.py`. Single-threaded, `n_jobs=1`.
- Own sweep: seeds 400 to 459 (60 seeds, disjoint from the reviewer's 100 to 159), n = 3000, the notebook's
  folds, support and runtime. Three learner pairs, each cross-fitted and in-sample:
  `default` (the notebook's bare HGB pair), `hgbreg` (HGB Q; HGB g with `max_leaf_nodes=6,
  learning_rate=0.05, max_iter=120, min_samples_leaf=60`, a different regularisation from the reviewer's),
  `poly2g` (HGB Q; degree-2 polynomial logistic g, close to the true propensity's form).
- Seed 34 through my script reproduces the notebook: 0.15177 / 0.01215 cross-fitted, 0.1548 / 0.0033 in-sample.
- Independent truth `[recomputed]` from my own transcription of the structural equations (N = 4e6):
  ATE 0.162868 +- 0.000033. Efficient SE at n = 3000: 0.00666 (reviewer 0.00661).

## CF-01
verdict: CONFIRMED (magnitude slightly smaller on my seeds)
severity: misleading

Notebook text checked `[read]`: Step 6 heading "the failure mode, an interval that is too narrow";
reading "The in-sample standard error is 0.27 times the cross-fitted one, which is less than a third.";
trust table "without splitting, the boosted fit reported a standard error 0.27 times the cross-fitted one".
The callback asserts `in_sample.std_error < cross_fitted.std_error / 3`
(`tests/unit/tutorial_semantics/cross_fitting.py`, Step 6 block).

Evidence `[executed]`, 60 seeds:

| learners | fit | bias | emp. SD | mean SE | SE/SD | coverage | OOF calibration slope (mean, range) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default HGB | cross-fitted | -0.0010 | 0.0070 | 0.0113 | **1.62** | 1.000 | 0.47 (0.37-0.54) |
| default HGB | in-sample | -0.0007 | 0.0061 | 0.0033 | 0.54 | 0.733 | |
| regularised HGB g | cross-fitted | -0.0000 | 0.0060 | 0.0056 | 0.94 | 0.917 | 0.97 (0.86-1.02) |
| regularised HGB g | in-sample | -0.0029 | 0.0059 | 0.0040 | 0.67 | 0.783 | |
| degree-2 logistic g | cross-fitted | +0.0006 | 0.0065 | 0.0062 | 0.94 | 0.917 | 0.95 (0.93-0.97) |
| degree-2 logistic g | in-sample | -0.0028 | 0.0060 | 0.0044 | 0.73 | 0.817 | |

In-sample / cross-fitted SE ratio: default median 0.30 (0.19-0.43, below 1/3 on 83% of seeds);
regularised g 0.70 (0.68-0.73); polynomial g 0.71 (0.59-0.76). Never below 1/3 with a calibrated g.

- Reviewer: cross-fitted SE/SD 2.12; mine 1.62. Both show the cross-fitted yardstick is conservative
  (mean SE 0.0113 is 1.7x the efficient SE 0.00666; coverage 60/60 in both sweeps). "About twice" is
  slightly high; "1.6 to 2.1 times" is what two independent sweeps support.
- Cause confirmed: with either calibrated g the cross-fitted SE matches the SD (0.94), and the slope
  goes from 0.47 to about 0.96. The inflation belongs to the out-of-fold bare HGB propensity.
- Answer to the focus question: the page's lesson survives a calibrated learner. In-sample still
  undercovers (0.78 / 0.82) and cross-fitting restores near-nominal coverage (0.917, 55/60; within
  about one binomial SE of 0.95). But the 0.27 / "less than a third" magnitude does not survive: it
  is about 0.7 with a calibrated g. Roughly half of the page's headline ratio is the conservative
  denominator, not the in-sample failure.

Fix recommendation: (a) is right and better than (b), which would need a registered source for scratch
numbers. Under (a), the callback's `< cross_fitted.std_error / 3` assertion and "less than a third" must
change (to about 0.7). Seed 34 still shows the failure under a regularised g: in-sample 0.1536, SE
0.00396, interval (0.1458, 0.1614) misses 0.163 (my `hgbreg` row; the polynomial g covers on seed 34).
Cost of (a): Step 11's calibration warning and Step 13's doubly robust refusal will likely disappear, so
the page loses its demonstration of the F26 refusal. If the authors want to keep that demonstration,
an acceptable alternative is to keep the bare HGB g in Steps 11 to 13 only, and state there that its
slope of 0.49 also makes the Step 5 interval conservative. No registered study moves under either fix.

## CF-02
verdict: CONFIRMED
severity: weak

`[read]` plan: "It addresses no other condition."; where-next: "Cross-fitting addresses only the Donsker
condition." `[executed]` default learners: in-sample bias -0.0007 against -0.0010 cross-fitted, with
similar SDs (0.0061 / 0.0070); the failure is in the reported SE (0.0033 against SD 0.0061). Note that
with a calibrated g the in-sample point does pick up a little bias (-0.0028/-0.0029), so the proposed
sentence "The estimate itself is not more biased" is true only for the notebook's learners; keep it
scoped to "on this configuration". The proposed `where-next` wording is acceptable. Theoretically the
in-sample variance shortfall is itself a data-reuse (empirical-process) effect, so the cleaner fix is to
say "addresses only data reuse" rather than add a second named condition; align
`docs/technical-reference/cv-tmle.md` lines 21-28 only if the page wording changes the four-row claim.

## CF-03
verdict: CONFIRMED
severity: weak

`[recomputed]` true-g quantities: P(g < 0.01) = 0.0028 (reviewer 0.28%). I did not recompute the
treated-arm ESS ratio (reviewer 0.389). `[executed]` the OOF slope is 0.37 to 0.54 on 60/60 of my seeds,
so the learner over-disperses g on every draw. The page sentence "The 22.9% describes the estimated
mechanism, not the population" is true; the gap is real. The proposed sentence is correct and becomes
moot under CF-01 (a).

## CF-04
verdict: CONFIRMED (as weak; the page does not explicitly claim a mild effect)
severity: weak

`[read]` `src/cleverly/datasets/synthetic.py:1227-1230`: `expit(-0.4 + 0.8a + 0.5W1 + 0.3W2 + 0.6u + 8.0 a u)`,
`u ~ N(0,1)` shared per cluster (`_latent`, lines 366-377). `[recomputed]` team-mean effect over the
covariate law, 20,000 latent draws: 45.5% negative; 5th/50th/95th percentiles -0.318 / 0.199 / 0.512
(reviewer 46%, -0.32 / +0.52). Analytically the sign flips at u < -0.1, P = 0.46.
The page does not say teams all benefit; it says "A shared difference that changes how much navigation
helps", which implies benefit throughout but is not a false statement. Points 2 and 3 are real:
`0.6 * u` shifts the baseline; and `navigation.py:39` maps `W3` to `medication_burden` in the main law,
while the team cell renames the team law's `W2` to `medication_burden`, and the team ATE is 0.104 against
0.163, so "the team law measures the same program on a coarser scale" is inaccurate.
Fix: the proposed replacement sentences are correct. Do not change the coefficient here (it is set by
the clustered study's design-effect rule, `synthetic.py:1201-1205`).

## CF-05
verdict: CONFIRMED
severity: weak

`[read]` `omitted_variable.py:1196-1202` uses `|rho| sqrt(cf_y cf_d / (1 - cf_d))`; module docstring
lines 21-22: "the corresponding gain in the Riesz representer, i.e. how much the confounder would improve
prediction of treatment". The notebook row paraphrases that docstring's second half, so the library's own
wording shares the looseness. The proposed fix is correct; apply it to the sibling notebooks and
consider the module docstring in the same change.

## CF-06
verdict: CONFIRMED
severity: weak

`[read]` `src/cleverly/estimators/tmle.py:1771-1772`: the refusal applies only when `cross_fit` and the
family is not binomial and `q_bounds is None`. The notebook sentence is accurate but implies a continuous
outcome would force cross-fitting off, whereas a declared support would also do. The proposed wording is
correct.

## Literature locator: "Lemma 3 and Theorem 4 of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026)"
verdict: locators VERIFIED against arXiv:2112.13398v6 (21 Sep 2026, journal ref REStat 2026, DOI 10.1162/REST.a.1705)

`[read]` from the v6 PDF (`lss_v6.txt`), Section 4 "Statistical inference under omitted variable bias":
"Lemma 3 (DML for Bound Components)" gives asymptotic linearity of the DML estimators of theta, sigma^2 and
nu^2, with the nu^2 score `(2m(W, alpha) - alpha^2) - nu^2` under DML Assumptions 3.1-3.2 of Chernozhukov
et al. (2018a). "Theorem 4 (DML Confidence Bounds for Bounds)" gives the influence function of the bounds
and the one-sided limits. Neither covers a plug-in `E_n[alpha_hat^2]`, so the refusal's claim "give the
limits for the doubly robust estimator only" is correct. The appendix heading "C.10. Proof of Lemma 3
and Theorem 4" is present. The published main text was not read (the package's references entry says the
same). The refusal's identity `E[2 m(alpha_hat) - alpha_hat^2] = nu_0^2 - ||alpha_hat - alpha_0||^2` is
correct algebra given `E[m(alpha_hat)] = E[alpha_0 alpha_hat]`.

## New findings

- (weak) The trust table's "why split the sample" answer quotes only the 0.27 ratio. Even after CF-01, it
  should cite the coverage gap (the registered study's 48.8% against 93.3%) as the answer, and the ratio
  as one draw's illustration.
- No other material problem found.
