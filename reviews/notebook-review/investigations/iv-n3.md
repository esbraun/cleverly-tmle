# IV-N3: cross-fitted shift SE several times the empirical SD

Worktree at cd2a1452 (`shift_trim=0.999` default in place). `cleverly.__file__` asserted under the
worktree `src` in every probe (`probe_ic.log`). Probes and outputs are in
`reviews/notebook-review/probes/iv-n3/`:

| file | what it does |
| --- | --- |
| `probe_ic.py` | 60 seeds (9000 to 9059), `make_shift_dose(n=3000)`, page's four policies, `q_bounds=(-30, 40)`, ten configurations; records the reported SE, an independent IC, the one-step estimator, ICs with the exact ratio, ratio moments per fold |
| `probe_ic.csv`, `summary_ic.log`, `summary_extra.log`, `se_spread.log` | raw rows and the summaries quoted below (`summarize_ic.py`, `summarize_extra.py`) |
| `fold_check.py`, `fold_check.log` | seed 9000: independent per-fold refit of the pooled-hazard density, independent ratio lookups, the targeted score with trimmed and raw ratios |
| `oracle_rows.py` | seed 9000: the exact law discretized at 40 and 320 equal-mass bins, against the continuous ratio |

Configuration names: `boost` = bare `HistGradientBoostingRegressor`/`Classifier` (the page's);
`rboost` = the regularized density booster from gate N4; `quad` = `PolynomialFeatures(2)` +
`LinearRegression` Q; `oracle` = `OracleShiftDensity` on the exact law at 40 bins; `oracle320` =
the same at 320 bins; `xfit` = 3 folds, pooled targeting; `notrim` = `shift_trim=1`.

## Verdict

**Not a library bug. An inherent property of influence-curve variance with a poor density ratio,
made visible by cross-fitting.** The reported IC is exactly the efficient influence function at the
fit's own nuisances. The held-out density ratio from the bare booster is far from the true ratio
(mean squared error 24.9 against a true second moment of 2.77). The IC variance inherits that
noise in full. The TMLE point estimate does not: its single-epsilon fluctuation regresses the
residual on the noisy ratio and applies only about a quarter of the one-step correction. The
reported SE is therefore the SD of the one-step estimator built from the same nuisances, not of
the TMLE. With the exact law at 320 bins the cross-fitted SE/SD is 0.98 (boosted Q) and 0.96
(quadratic Q).

The 0.999 trim of cd2a1452 is correct lmtp parity but addresses only the extreme tail: the whole
held-out ratio distribution is dispersed, so SE/SD falls from 3.91 to 2.46, no further.

## Findings

### F1. The IC is computed correctly from the fit's nuisances (hypothesis 2 rejected)

| check | result | tag |
| --- | --- | --- |
| IC re-implemented as `(h_r - h_0)(Y - Q*(A,W)) + Q*(d_r) - Q*(d_0) - psi`, from `fluctuations["mtp"].targeted`, `nuisance.shifts.design[:, 0, :]` (trimmed) and the scaler | max abs difference to `influence_curve` 4.5e-12 over 600 fits; psi difference 0 | [executed] `summary_ic.log` column `icdiff` |
| density bin probabilities per fold against an independent long expansion and a fresh `HistGradientBoostingClassifier` trained on that fold's training rows | own fold max abs diff 1.9e-16 to 2.2e-16; every other fold 0.18 to 0.31 | [executed] `fold_check.log` |
| `ShiftSet.ratio` and `ratio_at[:, r, r]` against an independent lookup of `g(a - delta)/g(a)` and `g(a)/g(a + delta)` | max abs diff 0 | [executed] `fold_check.log` |
| targeted score with the trimmed covariate (what the IC reads) | 1e-18 per column; with the raw ratio 2.1e-4 for +1.0, so targeting used the trimmed covariate | [executed] `fold_check.log` |
| Q* in the IC is post-fluctuation | `targeted.observed` differs from `nuisance.outcome.observed` (median abs move 0.29 on the top 1% ratio rows); the recomputed IC uses `targeted` and matches | [executed] `summary_extra.log` `one_qmove_top1pct` |

Code read: fold loop `src/cleverly/learners/density.py:462-470`; ratio
`src/cleverly/interventions/shift.py:441-461`; trimmed design `shift.py:349-365`; targeting reads
`nuisance.shifts.design` at `src/cleverly/estimators/targeting.py:392`; the `mtp` covariate blocks
`src/cleverly/fluctuation/submodel.py:944-975`; the IC `src/cleverly/inference/influence.py:1337-1346`.
[read]

### F2. The inflation tracks the error in the estimated ratio, not cross-fitting as such

`+1.0 uncapped vs current practice`, 60 seeds. SD is the empirical SD of the TMLE estimate; SE is
the mean reported SE; the efficient SE (true h, true Q) is 0.0224. [executed] `summary_ic.log`,
`summary_extra.log`.

| configuration | SE/SD | coverage | mean E_n[h] | median E_n[h^2] | median E_n[(h_n - h_0)^2] |
| --- | --- | --- | --- | --- | --- |
| insample_boost | 0.76 | 0.867 | 0.92 | 2.94 | 1.27 |
| xfit_boost_notrim | 3.91 | 1.000 | 1.66 | 32.3 | 24.9 |
| xfit_boost (page, current default) | 2.46 | 1.000 | 1.66 | 32.3 | 24.9 |
| xfit_quad_boost | 2.43 | 1.000 | 1.66 | 32.3 | 24.9 |
| xfit_rboost | 1.36 | 0.917 | 1.19 | 8.08 | 4.29 |
| xfit_oracle (exact law, 40 bins) | 1.36 | 0.967 | 1.12 | 5.62 | 2.07 |
| insample_oracle (same ratio) | 1.02 | 0.933 | 1.12 | 5.62 | 2.07 |
| xfit_oracle320 | 0.98 | 0.933 | 1.02 | 3.07 | 0.30 |
| xfit_quad_oracle320 | 0.96 | 0.950 | 1.02 | 3.07 | 0.30 |
| insample_quad_oracle320 | 0.94 | 0.933 | 1.02 | 3.07 | 0.30 |
| truth | | | 1.00 | 2.77 | 0 |

`+0.5 uncapped` shows the same ordering: 2.11 (xfit_boost), 2.36 (notrim), 1.57 (rboost), 1.15
(oracle320), 1.04 (quad oracle320).

- The correct quadratic Q does not remove the inflation (2.43). The density ratio does.
- Cross-fitting with an accurate ratio is calibrated (0.96 to 0.98). No fold-specific defect exists.
- In sample, the overfit booster puts high density on its own training rows. That shrinks the
  ratio (E_n[h^2] 2.94) and the overfit Q shrinks the residual (SD 0.60 against noise 0.80). Both
  push SE/SD below 1 (0.76). Cross-fitting removes both deflations and exposes the ratio error.

### F3. The held-out ratio is mis-normalized per fold (hypothesis 4 confirmed as a symptom)

h is the density of the shifted dose law with respect to the observed one, so E_0[h] = 1 inside
the support for every shift, capped or not. Per fold, the bare booster's held-out mean ratio for
+1.0 ranges 1.19 to 4.81 (mean 1.66); for +0.5, 1.09 to 1.42. The exact law at 40 bins gives 1.12,
and 1.02 at 320 bins. [executed] `summary_ic.log`. The library reports no mean ratio: the
`ShiftSupport` fields are quantiles, maximum, ESS, the ceiling and the trimmed count
(`shift.py:500-565`). [read]

### F4. Forty equal-mass bins add a discretization error even with the exact law

The outer bins of 40 equal-mass bins are wide. On seed 9000 the top bin is [4.46, 7.02] and the
bottom one is 1.66 wide. A piecewise-constant density over a 2.5-unit bin underestimates
g(a | w) near the bin's lower edge. Every large ratio of the exact 40-bin law sits in that bin
(maximum 63.6 against a continuous 10.9 for the same row). E_n[h^2] is 7.3 against 2.95. At 320
bins it is 3.07. [executed] `oracle_rows.py` output. Edges are full-sample marginal quantiles by
design (`density.py:88-118`). [read] The effect is modest next to F2's learner error (SE/SD 1.36
against 0.98 at 320 bins, boosted Q, cross-fitted).

### F5. Why the TMLE does not inherit the ratio noise (hypotheses 1 and 3)

| quantity, +1.0, xfit_boost, 60 seeds | value | tag |
| --- | --- | --- |
| one-step estimator `plug-in + mean(h_n (Y - Q))` from the same nuisances: SD, mean IC SE | 0.101, 0.082 | [executed] |
| one-step bias, coverage | -0.149, 0.600 | [executed] |
| plug-in SD, bias | 0.027, +0.0099 | [executed] |
| TMLE SD, bias | 0.030, -0.016 | [executed] |
| median (TMLE - plug-in) / one-step correction | 0.18 (0.99 for xfit_quad_oracle320) | [executed] |
| median kappa = sum(h_0 h_n)/sum(h_n^2) | 0.24 (0.98 for oracle320) | [executed] |
| corr(TMLE move, kappa times one-step correction) | 0.76 | [executed] |
| share of IC variance from the top 1% of rows | 0.82 (0.35 in sample; 0.45 quad oracle320 xfit) | [executed] |
| corr(per-fit SE, abs error of the estimate) | -0.025; per-fit SE 0.038 to 0.127 | [executed] `se_spread.log` |

The pooled fluctuation solves `P_n h_n (Y - Q*) = 0` with one epsilon per shift. When h_n is
heavy-tailed, epsilon fits the few dominant rows and is small. The plug-in then moves by about
kappa times the one-step correction, where kappa is the regression slope of the true ratio on the
estimated one. [recomputed] In the expansion psi* - psi_0 = (P_n - P_0) D(h_n, Q*) + R_2, the
term R_2 = -P_0 (h_n - h_0)(Q* - Q_0) is no longer second order. Q* carries epsilon h_n, so R_2
cancels most of the linear term. The reported SE estimates the variance of the linear term,
which equals the one-step's spread (SD/SE 1.24) and not the TMLE's. The TMLE behaves like a
damped plug-in. The per-fit SE varies across seeds with the ratio noise and carries no
information about the error of that fit.

This is the standard condition of IC-based inference: the variance estimator is consistent only
when ||h_n - h_0|| tends to zero (van der Laan and Rose 2011, ch. 5; Díaz, Williams, Hoffman and
Schenck 2023 for MTPs). At n_train = 2000 the bare booster's held-out ratio is not in that regime.

### F6. The gate N4 Q2 mechanism needs correcting

Gate N4 attributed the inflation to the unbounded held-out ratio and proposed the trim (fix 3).
The trim binds on 3 rows of 3000 (`fold_check.log`: trimmed rows 3) and leaves SE/SD at 2.46.
The cause is the dispersion of the whole held-out ratio (F2, F3). The trim stays correct as lmtp
parity. Gate fix 2's notebook sentence "because the held-out density ratio is unbounded (fix 3)"
states the wrong mechanism.

### F7. No registered study covers an estimated or cross-fitted shift density

`canonical_shift_policies.py:188,215-219` fits in sample with `OracleShiftDensity` and
`QuadraticShiftOutcome` at 320 bins. `shift_policy_properties.py:87-103` fits in sample with an
oracle density. The method-evidence page states the limit
(`docs/technical-reference/method-evidence/continuous-modified-treatment-policies.md:120`). [read]
Nothing proposed here changes a result-determining path, so no study or `tests/canonical/lmtp_shift`
artifact moves.

## Recommendation

1. **No fix to the IC, the fold mapping or the targeting.** F1 rules each out.
2. **Technical reference** (`docs/technical-reference/point-treatment-tmle.md`, after the trim
   table at lines 598-604; mirror one line in the method-evidence Limits). Proposed text:

   > The standard error reads the estimated density ratio. When that ratio is far from the true
   > one, the standard error does not describe the TMLE. Held out, a flexible density overstates
   > the spread. In sample, an overfit density understates it. In a 60-seed probe on
   > `make_shift_dose(n=3000)` with a +1.0 shift, the ratio of the mean standard error to the
   > empirical SD was 2.46 for a cross-fitted, unregularized booster density, 0.76 in sample, and
   > 0.98 cross-fitted with the exact density at 320 bins. The 0.999 trim does not correct this.
   > Under the true law the mean ratio at the observed dose is 1. A mean far from 1 in any fold
   > shows that the ratio, and so the standard error, is unreliable.

   Cite `reviews/notebook-review/probes/iv-n3/` as the probe until a registered study exists.
3. **Optional library diagnostic, not a correctness fix.** Add `mean_ratio` (all rows) and
   `fold_mean_ratio` (one per fold) to `ShiftSupport` (`shift.py:500-565`), filled in
   `check_shift_support` (`shift.py:596-678`) from `ShiftSet.ratio` before the trim, and print it
   in `summary()`. Failing-first test: on `make_shift_dose(n=3000, seed=9000)` cross-fitted with
   bare `HistGradientBoostingClassifier(random_state=9000)` at 40 bins, the `+1.0 uncapped`
   report exposes a fold mean ratio above 1.3. The same fit with `OracleShiftDensity` at 320
   bins reads within 0.05 of 1 in every fold. A warning threshold needs its own calibration and
   is not proposed here. Diagnostic only, so no registered study moves.
4. **Notebook (IV-N3 wording, gate fix 2).** Replace "because the held-out density ratio is
   unbounded (fix 3)" with: a cross-fitted shift fit with a flexible density reported standard
   errors 1.4 to 2.5 times the empirical SD in the review probe, because the held-out density
   ratio is far from the true one. The page therefore targets in sample, as the registered study
   does.
5. **Roadmap candidate.** A registered cross-fitted, estimated-density shift study is the missing
   evidence for any claim about cross-fitted shift intervals (F7).
