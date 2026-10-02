# Gate S6: revert of the shift ratio trim (19e91585) and the per-fold mean ratio (9ae18141)

Reviewed read-only at 9ae18141. Probes: `git diff`, pandas re-read of
`reviews/notebook-review/probes/iv-n3/probe_ic.csv` (`float_precision="round_trip"`), the new
unit test, every `tests/unit/test_shift*.py`, the prose and API-doc gates.

## OK

1. **Investigation conclusion is sound.** Recomputed from `probe_ic.csv`: mean SE / empirical SD
   for `+1.0` is 3.91 (`xfit_boost_notrim`), 2.46 (`xfit_boost`), 0.76 (`insample_boost`),
   0.98 (`xfit_oracle320`), as the investigation and the doc table state. Seed 9000 per-fold
   `E_n[h]`: 1.462, 1.200 (boost), 1.033, 0.983 (oracle 320), matching the commit message. The IC
   recomputation column `icdiff` is at most 4.5e-12. `fold_check.log` shows own-fold bin
   probabilities agree to 2e-16 and other-fold to 0.18 to 0.31, so held-out predictions are the
   fold's own model and the fold mapping is correct.
   - On "SE matches the one-step spread, not the TMLE's": in the probe this makes the
     cross-fitted interval conservative (coverage 1.000, SE 2.5 to 3.9 times SD), but that is not
     a property to rely on. The per-fit SE spans 1.3 to 4.3 times the SD and has correlation
     -0.025 with the fit's own error (`se_spread.log`), and in sample the same mechanism
     understates (0.76, coverage 0.867). The TMLE in this regime moves only 18% of the one-step
     correction (`move tmle/os`), so it is a damped plug-in, not an asymptotically linear
     estimator; "conservative" would overstate what is known. The doc text says the SE "does not
     describe the spread of the TMLE" and gives both directions with the table. That wording is
     correct and does not claim conservativeness.
2. **Revert is complete and byte-exact.** `git diff cd2a1452~1 19e91585` is empty. No
   `shift_trim` residue in `src`, `tests`, or `docs`. The only remaining "trim" mentions are in
   the new doc paragraph, which describes the probe build and states that the shipped estimator
   applies no bound.
3. **`ShiftSet.ratio` docstring is true of the code.** `_ratio` applies no bound.
   `submodel.py` `mtp` builder divides the stacked ratio by `pi * pz` only (missingness and
   intermediate mechanisms); `truncation_curve(mechanism=True)` sweeps those bounds. No option
   bounds the ratio.
4. **`mean_ratio` / `fold_mean_ratio` are correct.** Computed from the untruncated ratio at the
   observed dose; `held_out = [arange(n)]` when `folds is None`, else each `(train, test)` pair's
   `test` from `Folds.__iter__`. In sample the tuple holds one entry equal to `mean_ratio` (test
   passes). `diagnostics.support()` is the only caller and now passes `nuisance.folds`. Numpydoc
   entries are `name : type` form under `Parameters`. Witness and control are meaningful: fold
   means 1.46/1.20/1.87 vs 1.03/0.98/1.07 with a `max > 1.5` / `|x - 1| < 0.1` split; a
   train-rows mis-mapping would fail the independent per-fold recomputation.
   - **Identity check.** Uncapped: `E_0[h | W] = ∫ g(a - δ | w) da over supp(g(· | w))
     = P(A + δ ∈ supp | W)`, so `E[h] = 1` exactly when the shifted dose stays inside the support,
     and below 1 otherwise. Capped (`_ratio`): `E_0[h | W] = ∫_{a ≤ cap} g(a - δ) da + P(A > cap - δ)
     = P(A ≤ cap - δ) + P(A > cap - δ) = 1` exactly, with no support condition, because units the
     cap holds at their own dose contribute the indicator term. The docstring's qualifier is
     correct for both cases.
5. **Technical reference** matches the probe numbers and the code; sentences are within the
   25-word description limit; prose ledger and API-doc gates pass (135 passed). The
   method-evidence Limits already states "non-cross-fitted targeting", so no inconsistency.
6. **No numeric output changed.** `git diff f62ed489 HEAD -- src` is additive only: two
   `ShiftSupport` fields, the `folds=` keyword, and the docstring. No estimate, SE, or
   targeting path is touched. `tests/unit/test_shift_ratio_mean.py` plus every
   `tests/unit/test_shift*.py`: 66 passed.

## REQUIRED FIXES

1. **`ShiftSupport.summary()` drops the support qualifier.** The printed line reads
   "(1 under the true density; a mean far from 1 marks a mis-estimated ratio ...)". For an
   uncapped shift on a dose with a bounded support, `E_0[h] = P(A + δ ∈ supp) < 1` under the true
   density, which is exactly the case `_warn_outside_support` fires on. A user would read a true
   mean of 0.9 as mis-estimation. Add the qualifier the docstring and doc carry ("while the
   shifted dose stays inside the support"), or point at `unsupported` and the support warning.
   `src/cleverly/interventions/shift.py`, `summary()`.

## Advisory, not gating

- Doc sentences "A flexible density fitted out of fold overstates the spread. An overfit density
  fitted in sample understates it." read as general laws; the evidence is one DGP. Prefix with
  "In the probe below," or move them after the table.
- The doc could add one sentence that in this regime the TMLE applies only part of the one-step
  correction (`move tmle/os` 0.18), so the estimate is also not asymptotically linear; the
  current text addresses the standard error alone.
- `n_repeats > 1`: `fold_mean_ratio` describes draw 1's folds, like the retained weights. The
  `n_repeats` docstring sentence covers weights only; extending it to the fold means is one
  clause.
