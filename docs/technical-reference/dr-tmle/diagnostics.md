# Diagnostics to inspect

In cost order. The first two are free.

| call | what it answers |
| --- | --- |
| `res.score_verdict` | the score check's verdict, carried whether it passed or not. `summary()` prints it whenever it **fails**; a passing fit says nothing extra. Derived from the fluctuations rather than stored, so a reloaded fit recomputes it. |
| `res.diagnostics.score_equations()` | the same score object, asked for directly |
| `res.validate()` | the default assessment; reports `warning` when equation (10) had numerically difficult inner solves even if the returned score equations pass, with the affected round count and fraction |
| `res.diagnostics.corrections()` | the low-level doubly-robust rows: per arm, per equation. Empty unless the fit is a guarded `DRTMLE`. |
| `res.diagnostics.nuisance_models()` | the primary fits' held-out risk and diagnostics |
| `res.diagnostics.refute()` | negative controls; costs refits |

`correction_check()` recomputes each arm's `P_n[w D*_g]` and `P_n[w D*_Q]` **from the exact
returned state** and reports the residual against the score the targeting step recorded. Five
conditions govern how it does so. Each one rules out a way of passing for the wrong reason.

1. Per arm, and never only on the ATE. Arm-specific errors cancel in a difference.
2. **Before** the contrast is constructed.
3. With the row weights included.
4. On **one outcome scale**, which is the outcome's own. A correction score and `se/√n` are then
   comparable numbers rather than two quantities a factor of `range` apart.
5. In the tests, on a fixture where the truncation binds.

**Two failures, and they are not the same failure.**

- *An identity residual*, meaning `Δ_g` or `Δ_Q` above `IDENTITY_TOLERANCE = 1e-12`, is a
  **software defect**. The fit solved one expression and reported another, and no amount of
  further iteration would fix it, because the loop is not posing the equation the curve needs.
  The tolerance sits seven orders above the arithmetic and four below the smallest observed real
  failure.
- *A correction score* above the inferential tolerance is a **fit that did not solve its
  equations**. That is the ordinary failure, reported per arm and per equation so a reader can
  see which.

*And a row that is neither.* A fit guarding one nuisance solves one of the two extra equations, so
its curve subtracts one correction; the other term is still reported, marked `solved=False`, as the
diagnostic saying what is **not** in this curve. Such a row is not a failure and cannot be one:
nothing subtracts it.

Read `CorrectionCheck.contract` alongside `passed`, never folded into it. `passed` answers *did
this fit solve what it reports*; `contract` answers *which estimator the numbers are evidence
about*.

## Solved scores do not establish nuisance consistency

This is the one thing to take away from the page.

The score equations being solved is a statement about **numerical targeting**. It says nothing
about whether the method-specific functions of
[section 4](nuisance-conditions.md) are adequately estimated. The two are independent,
and the independence is easy to get backwards. A fit with badly wrong reductions returns a `psi`,
an `se` and a confidence interval formatted exactly like a good one, with every score green.

**The evidence is `tests/unit/test_oracle_reductions.py`**, and it is worth stating as a result
rather than a caveat. On an exact law, with **exact** reduced regressions handed to a real
alternation, the estimator recovers the truth *despite misspecified primary nuisances*, which is
the whole point of the variant. With **wrong** reductions, the estimate moves, and **every score
equation still passes**. Nothing on the face of such a fit distinguishes it from the good one.

Three consequences for practice:

1. **The score check is necessary, not sufficient.** Treat a failing score check as
   disqualifying and a passing one as saying nothing about the nuisances.
2. **Inspect the reduced-regression fits themselves**, not just the equations built from them.
   Their diagnostics are on `result.extra["drtmle"].diagnostics`, keyed `"qr"`, `"gr1"`, `"gr2"`
   on the univariate reduction, `"qr"`, `"gr1"` on the bivariate reduction, and `"gamma_a"`,
   `"gamma_m"`, `"r_a"`, `"r_m"`, `"e"` on the missing-outcome one. The constructions do not
   fit the same regressions, so they cannot report under the same names.
   `result.extra["drtmle"].reduction` says which ran, read off the
   set that was fitted rather than off the `reduction=` setting, and `.missingness_bound` records
   the bound the two observation reductions were formed at.
3. **Where you cannot argue the rate conditions, do not treat the interval as settled.** Use this
   estimator where you have a reason to think one primary nuisance is badly estimated; that is the
   regime it was derived for and the regime the evidence covers.

The same distinction, once more, in the theorem's own terms: Theorem 1 licenses an interval
*conditional on* the three empirical scores being `o_p(n^(−1/2))` **and** the two second-order
remainder terms being `o_p(n^(−1/2))`. A fit can report on the first. Nothing reports on the
second.
