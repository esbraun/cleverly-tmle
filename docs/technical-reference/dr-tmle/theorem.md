# The theorem-backed contract

This page carries what Theorem 1 states, the objects it is stated over, and the conditions it
rests on. [Targeting and cross-fitting](targeting.md) carries what the implementation chooses where
the theorem is silent. Every source and locator is in
[references](../../references.md#doubly-robust-inference-drtmle).

## The objects

For treatment level `a`, with `Q̄_0(a, w) = E_0(Y | A = a, W = w)`, the target is
`ψ_0(a) = E_0{Q̄_0(a, W)}` and `ATE = ψ_0(1) − ψ_0(0)`. The ordinary efficient influence function
is `D*(Q, g)(O) = A/g(W)·{Y − Q̄(W)} + Q̄(W) − Ψ(Q)`.

The default construction uses three univariate reduced regressions however many covariates the fit
adjusted for:

```text
Q_r    := Q̄_{0,r}(Q̄, g)(w)      = E_0[ Y − Q̄(W) | A = 1, g(W) = g(w) ]
g_{r,1} := g_{1,0,r}(Q̄)(w)       = E_0[ A | Q̄(W) = Q̄(w) ]
g_{r,2} := g_{2,0,r}(Q̄, g)(w)    = E_0[ (A − g(W))/g(W) | Q̄(W) = Q̄(w) ]
```

The two corrections, in the orientation this package computes and the appendices derive:

```text
D_A = D*_g = (Q_r/g)·{A − g}
D_Y = D*_Q = 1_a·(g_{r,2}/g_{r,1})·{Y − Q̄}
```

The bivariate alternative retains `Q_r` and replaces both reduced mechanisms by one probability:

```text
g_r(a|w) = P_0{A=a | Qbar(a,W)=Qbar(a,w), g(a|W)=g(a|w)}
D_Y      = 1_a·{g_r(a|W)-g(a|W)}/[g(a|W)g_r(a|W)]·{Y-Qbar(a,W)}
```

This is exactly the pinned R source's two-column `estimategrn` branch, `fluctuateQ2` clever
covariate, and `eval_Dstar_Q` correction. Those branches loop over every requested level of a
discrete treatment, so cleverly uses the same one-versus-rest probability and correction for each
arm. In the stored state `gr1` is that probability and
`gr2` is `NaN`, because the latter regression does not exist on this path; using zero would make
an accidental univariate formula look valid.

and the limiting influence function

```text
D^{*,#}(Q, g) = D*(Q, g) − I(g = g_0)·D_A − I(Q̄ = Q̄_0)·D_Y
```

The indicators contain the doubly-robust claim. At the true primary nuisance functions, both
corrections vanish and the ordinary efficient influence function is recovered.

The two extra score equations, as the software article states them:

```text
(9)   P_n[ Q_r(a,W)/g*(a|W) · {1_a − g*(a|W)} ]                    = 0
(10)  P_n[ 1_a · g_{r,2}(a|W)/g_{r,1}(a|W) · {Y − Q̄*(a,W)} ]        = 0
```

## Theorem 1

**Suppose** either `Q̄ = Q̄_0` or `g = g_0`, and let the targeted collection
`(Q̄*_n, Q̄*_{n,r}, g*_n, g*_{1,n,r}, g*_{2,n,r})` satisfy the three empirical score conditions

```text
B_n     = P_n D*(Q*_n, g*_n)                            = o_p(n^(−1/2))
B_{A,n} = P_n D_A(Q̄*_{n,r}, g*_n)                       = o_p(n^(−1/2))
B_{Y,n} = P_n D_Y(Q̄*_n, g*_{1,n,r}, g*_{2,n,r})         = o_p(n^(−1/2))
```

**and** suppose the appendix second-order terms satisfy `R_{Q,n} = o_p(n^(−1/2))` and
`R_{g,n} = o_p(n^(−1/2))`. **Then** the targeted plug-in is asymptotically linear with influence
function `D^{*,#}`, and `√n(ψ̂ − ψ_0) ⇝ N(0, σ²)` with

```text
σ̂²_n = P_n [ D*(Q*_n, g*_n) − D_A − D_Y ]²
```

Three things to read off it:

- the score conditions are `o_p(n^(−1/2))`, **not** exact zeros. This implementation's exact-zero
  ambition is stricter than the theorem, and its numerical stopping rule is not obviously either;
- the conditions are on the **targeted** collection, including the starred *reduced* nuisances,
  which is why `retarget` on a `DRTMLE` result costs a fit rather than arithmetic on cached
  arrays. That cost follows the source, and is not a design slip;
- the variance is `P_n[·]²` of the **rowwise** corrected curve, so nothing that summarises the
  curve before squaring is computing this. The ATE curve is the **rowwise difference** of the two
  arm curves, which is what makes an ATE-only diagnostic insufficient: arm-specific errors cancel
  in a difference.

`tests/unit/test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s` pins the last property.
The interval built from the package's own corrections is the one Theorem 1's terms give. The
uncentred `P_n{D}²` differs from the reported variance by exactly `(P_n D)²`, and the contrast
reads the covariance rather than the sum of the arms.

## A known treatment mechanism

Data can declare its treatment mechanism with `treatment_probabilities=` (the
[point-treatment contract](../point-treatment-tmle.md#known-treatment-mechanism) gives the forms).
A complete-data fit is then Theorem 1 at the degenerate estimator `g_n ≡ g_0`. The branch
`g = g_0` of the theorem's hypothesis holds by declaration, and `R_{g,n}` is trivial at `g_0`.

| guard | what the fit does with the declared `g_0` |
| --- | --- |
| `()` | the ordinary TMLE at `g_0`, which is exact for any `Q̄` |
| `("g",)` | equation (10) fluctuates `Q̄` only, so the mechanism stays `g_0` |
| `("Q",)` and `("Q", "g")` | equation (9) fluctuates `g*` along `Q_r/g` (step 6 of the algorithm in Benkeser et al. 2017, Section 3.2), so `g*` moves off `g_0` on purpose |

Under the `"Q"` guard the reported curve `D* − D_A − D_Y` is the curve of the estimator that
fluctuates a known mechanism. It is not the curve of the known-mechanism TMLE. R `drtmle` 1.1.2
does the same with a supplied `gn`: its `gnStar` starts at `gn`, and `fluctuateG` moves it. The
fluctuated `g*` is no longer the declaration, so it keeps the shipped truncation. The declaration
itself answers to the bound rule of the point-treatment contract before any learner.

At an exact law the `"Q"` guards are blind. There `P_n D_A = 0` at `g_0` before any
fluctuation, so equation (9) solves at `ε_g = 0` whether the fluctuation exists or not.
`tests/unit/test_known_treatment_mechanism.py` therefore checks the fluctuation on a finite sample:
the initial `P_n D_A` exceeds `1e-3`, the final score is solved, and `max |g* − g_0|` exceeds
`1e-4`.

Every reduction and both `reduced_crossfit` settings admit a declared mechanism. The nested
construction reuses `g_0` in every inner fold, because a declared mechanism does not depend on the
rows a model saw. An `evaluation=` companion must declare the same columns. The registered
`known-treatment-mechanism-drtmle` study pairs each guard with R `drtmle` (`gn=`) at a wrong
outcome regression.

## Randomized trials with missing outcomes

For observed data `O=(W,A,Delta,Delta Y)`, write `g_A(a|W)=P(A=a|W)`,
`g_Delta(a,W)=P(Delta=1|A=a,W)`, and `g=g_A g_Delta`. Díaz & van der Laan's
efficient influence function is

```text
D* = 1(A=a, Delta=1)/g · {Y − Qbar(a,W)} + Qbar(a,W) − psi(a).
```

The implementation keeps the five one-dimensional regressions and the three correction
blocks in the paper separate:

```text
gamma_A = P(A=a | Qbar_a)
gamma_Delta = P(Delta=1 | A=a,Qbar_a)
r_A = E[{1(A=a)-g_A}/g_A | Qbar_a]
r_Delta = E[{Delta-g_Delta}/(g_A g_Delta) | A=a,Qbar_a]
e = E[Y-Qbar_a | A=a,Delta=1,g_A g_Delta]

D_A = e/g_A · {1(A=a)-g_A}
D_Delta = 1(A=a)e/(g_A g_Delta) · {Delta-g_Delta}
D_Y = 1(A=a,Delta=1) · {r_A/(gamma_A gamma_Delta)+r_Delta/gamma_Delta}
      · {Y-Qbar_a}
```

The targeting cycle jointly updates the ordinary outcome and `D_Y` covariates, updates
`g_Delta` within each arm, updates `g_A`, refits all five reductions, and repeats until all four
score blocks settle. The `g_A` update takes one of two routes, and both solve the equation
`P_n[e_a/g_a · {1(A=a) − g_a}] = 0` for every arm.

| arms | `g_A` update | covariate |
| --- | --- | --- |
| two | one two-parameter logistic tilt of `g_1`, with `g_0 = 1 − g_1` | `(−e_0/g_0, +e_1/g_1)` |
| three or more | one logistic tilt of each column `g_a`, with the response `1(A=a)`, on all rows | `+e_a/g_a` |

At two arms the arm-0 column carries a minus sign, because its score `−(e_0/g_0)(A − g_1)` equals
`(e_0/g_0){1(A=0) − g_0}`. Above two arms the tilted columns are not renormalized. They are nuisance
denominators and not an intervention. `correction_check` reports `D_A`, `D_Delta` and `D_Y`
separately; checking only `D_A + D_Delta` would be blind to equal and opposite score errors.
Missing-outcome fits therefore require `guard=("Q", "g")`.

Treatment probabilities and observation probabilities retain their own bounds: `g_bounds`
applies to `g_A` and `gamma_A`, while `nuisance_bound` applies to `g_Delta` and
`gamma_Delta`. Their product is derived for the ordinary outcome clever covariate and the
positivity report, never stored as a third nuisance. The treatment truncation curve moves only
the treatment bound; `mechanism=True` moves only the observation bound.

The positivity report's `P(A=a,Delta=1|W)` row is derived, so it **has no bound of its own**.
Reading it as though it did is a mistake worth naming. Each factor is truncated and the two are
then multiplied, so the product is never compared against `g_bounds[0] × nuisance_bound`. That row's `clipped` therefore counts the cells where either factor's
truncation moved the product. Its `ess_ratio` weights by `clip(g)·clip(π)`, so the denominator
the equation forms. Counting against the product of the floors instead reports a strict subset,
since a small factor beside a large one leaves the product above it: measured at **1.1%** against
a true **20.1%** on the pinched fixture in `tests/unit/test_drtmle_missing.py`.

The shipped scope follows the paper rather than the broader canonical package: randomized treatment
at any number of arms, MAR and positivity, no cross-fitting, and no weights, repeats, fold
targeting, or evaluation companion. `randomized=True` estimates `g_A`. Data that declares its
mechanism ([a known treatment mechanism](#a-known-treatment-mechanism)) supplies `g_A` and fits no
treatment learner. An observational missing outcome and a missing treatment take the construction
of [the composite indicator](#observational-missing-data-the-composite-indicator) instead.

### More than two arms

Díaz and van der Laan (2017) state their result for one arm indicator. Above two arms, the fit
applies it to each indicator `1(A=a)` and stacks the arm estimators. The contract below gives the
argument in five parts. Page numbers are those of the arXiv v1 author manuscript that
[the references](../../references.md) cite.

| part | content |
| --- | --- |
| base result | Theorem 2 (p. 20) under Condition 2 (Donsker, p. 13) and Condition 3 (p. 15). It gives `n^(1/2)(psi_dtmle − psi_0) → N(0, Var D_dr)` with `D_dr` of Theorem 1 (p. 16), for one indicator `A` and the target `E(Y_1)`. Section 2.1 (p. 6) applies it to "four such indicators" in its application |
| steps | indicator reduction: arm `a` uses `(W, 1(A=a), Delta, Delta Y)` and its own `g_A(a|W)` and `g_Delta(a,W)`. Fixed-dimension stack: the `K` arm estimators are asymptotically linear on the same rows, so their joint covariance is `P_0[D_a D_b]`. Linearity gives each `ate`, and the delta method gives `rr` and `or` on the log scale. The simultaneous band is the multiplier band over the stacked curves |
| objection search | p. 25 rejects a composite `T = AM` reduction. The armwise construction keeps `g_A` and `g_Delta` apart, so the objection does not apply. p. 26 discusses cross-fitting, not arms. No source records the indicator reduction or the stack as open |
| conditions | Assumptions 1 to 4 (pp. 6-7) at every arm, with positivity `g_A(a|W) g_Delta(a,W) > 0`. Randomization by design: `randomized=True`, or a declared known mechanism, which can depend on `W`. Conditions 2 and 3 for every arm, hence `cross_fit=False`. Every arm's four score equations solved to `o_P(n^(−1/2))` |
| evidence | the exact-law checks, nonzero witnesses, mutation controls and independent reference of `tests/unit/test_drtmle_missing_multi_arm.py`, and the registered [multi-arm missing-outcome study](../method-evidence/randomized-multi-arm-missing-outcome-dr-tmle.md) |

**No fluctuation parameter is shared across arms above two arms.** This condition is what lets
each arm's expansion hold inside the joint loop. Theorem 2's proof uses only that arm's solved
equations and Conditions 2 and 3. Four places carry the condition.

| block | where it is per arm |
| --- | --- |
| outcome tilt | `missing_outcome_outcome_submodel` builds a zero block for every other arm |
| observation tilt | `_solve_missing_observation_mechanism` tilts `g_Delta` within each arm |
| reductions | `fit_missing_outcome_reduced` fits each arm's five regressions on that arm's columns |
| treatment tilt | `solve_armwise_bounded_mechanism` tilts each column `g_a` alone |

The loop stops on the largest score over all arms, and so does the final check. That rule is
stricter than Theorem 2's per-arm requirement. At two arms the one shared tilt of `g_1` is a
different path to the same two solved equations. The proof reads only the solved equations, so
the result holds there too.

## Observational missing data: the composite indicator

An observational fit with `delta=` and a fit with a missing treatment use one construction. The
fit declares a missing treatment with `treatment_delta=<column>`, the 0/1 column that is 1 where
the treatment is recorded. The package never infers a missing treatment from a missing value, for
the reason it never infers a missing outcome: an accidental gap must not start a missing-data
analysis. The roadmap words for this item read "a missing treatment coded as a missing value", and
the declaration is a deliberate deviation from them.

For arm `a`, write `Delta_A` for the treatment indicator and define the composite indicator and its
mechanism:

```text
C_a = Delta_A · Delta · 1(A=a)
g_c,a(W) = P(C_a=1 | W) = pi_A(W) · g(a | Delta_A=1, W) · pi(a, W)
pi_A(W) = P(Delta_A=1 | W),    pi(a, W) = P(Delta=1 | A=a, Delta_A=1, W)
```

The product is the law of total probability, so it holds for every law. No factor needs a causal
reading. The fit estimates each factor on the rows it conditions on, and the outcome regression
on the rows where `C_a` can be 1.

**The route.** `cleverly.estimators.composite.missing_data_route` chooses the construction, and
the fit records it under `result.extra["missing_data"]`.

| `delta=` | missing treatment | `randomized=True`, or a declared mechanism on `DRTMLE` | guard | route |
| --- | --- | --- | --- | --- |
| no | no | any | any | `complete` |
| yes | no | yes | non-empty | `randomized_missing_outcome` (Díaz and van der Laan above) |
| yes | no | no | non-empty | `composite` |
| yes | no | any | `()`, or `TMLE` | `missing_outcome` (the shipped missing-outcome TMLE) |
| any | yes | no | any, or `TMLE` | `composite` |
| any | yes | yes | any | refused: `randomized=True` selects a construction that observes the treatment on every row, and a declared mechanism is not the mechanism of the recorded rows |

**The contract** has five parts.

| part | content |
| --- | --- |
| base result | Benkeser, Carone, van der Laan and Gilbert (2017), Section 3.2, Theorem 1: on `O = (W, A, Y)` with a binary `A`, the estimator that solves the targeting equation and the equations of the reduced regressions is asymptotically linear with curve `D* − D*_Q − D*_g` when either `Qbar` or `g` is consistent |
| steps | indicator reduction: Theorem 1 applies as stated to `O'_a = (W, C_a, C_a Y)`, with the binary treatment `C_a`, the mechanism `g_c,a` and the regression `E(Y | C_a=1, W)`. Identification: `E{Y(a)} = E[E{Y | A=a, Delta_A=1, Delta=1, W}]` under the conditions below. Fixed-dimension stack: the `K` arm estimators are asymptotically linear on the same rows, with joint covariance `P_0[D_a D_b]`. Linearity gives `ate`, and the delta method gives `rr` and `or` on the log scale. Fixed weights tilt the law and clusters are the unit, as in both parents |
| objection search | Benkeser et al. (2017) mention no missing data and no caveat about coarsening. Díaz and van der Laan (2017, p. 25) reject a composite `T = AM` for a randomized trial, because it discards known design information. On an observational law nothing about the treatment mechanism is known, so the composite discards nothing. R `drtmle` 1.1.2 implements this construction (`R/drtmle.R` lines 207-209, `R/fluctuate.R` lines 28, 98 and 169-172, `R/estimate.R` lines 116-200) |
| conditions | the four conditions below; Theorem 1's conditions for each arm on `O'_a`, which include the Donsker condition, hence `cross_fit=False`; the solved equations of every arm at `o_P(n^(−1/2))`. The loop stops on the largest score over the arms, which is stricter than the per-arm requirement |
| evidence | the exact-law checks, nonzero witnesses, mutation controls and per-arm reference of `tests/unit/test_composite_missing_data.py`, and the registered composite study |

**The conditions** are these.

| condition | statement |
| --- | --- |
| consistency and no unmeasured confounding | `Y = Y(a)` when `A = a`, and `Y(a)` is independent of `A` given `W` |
| treatment missing at random | `Y(a)` is independent of `Delta_A` given `(A, W)` |
| outcome missing at random | `Y` is independent of `Delta` given `(A, Delta_A=1, W)` |
| composite positivity | `g_c,a(W) > 0` for every arm, and `W` complete on every row |

**What the data do not identify.** The conditions let `Delta_A` depend on `A`. Then
`P(A = a | W)` is not identified, and only `P(A = a | Delta_A = 1, W)` is. A target or a clever
covariate that reads the treatment law of every row is therefore not identified with a missing
treatment. That covers `att`, `atc`, `ey_obs`, `par`, `paf`, `incremental=` and `policies=`. The
shared preflight refuses each one by name, and each refusal repeats this statement. A regime mean
and an arm-indexed `msm=` coefficient read only the outcome regression and the law of `W`, so the
composite identifies them.

No observed-data check can detect a violation of the treatment condition.

**The carrier.** `g_c,0 + g_c,1 < 1` whenever a row can be unrecorded, so the composite is not a
distribution over the arms. It always travels as an `(n, K)` mechanism off the simplex. Every site
that tilts or reads a two-arm mechanism uses one rule: the mechanism is one column exactly when
the two-arm complement form applies. The composite therefore tilts each arm's column alone at
every arm count, two included. No fluctuation parameter is shared across arms.

**The bounds.** Each factor is bounded as the shipped missing-outcome TMLE bounds it: the
treatment factor by `g_bounds`, and each observation factor below by `nuisance_bound`. The
composite is then tilted and clipped inside `[g_lo · nb^k, g_hi]`, where `k` counts the
observation factors. That floor is the smallest product of the bounded factors, so the initial
clip moves no value. At `g_bounds=(0.01, 0.99)` and `nuisance_bound=0.01` with both indicators,
the floor is `0.01 · 0.01² = 1e−6`, against `1e−4` on a missing outcome alone. The default
`g_bounds="auto"` sets `g_lo = 5 / (sqrt(n) ln n)` instead. The positivity report adds the composite row with its minimum
and the share of unit-arm cells below `g_lo`.

The rejected alternative floors the composite at `g_lo` after the tilt, as R `drtmle`'s `tolg`
floors its product. That floor clips legitimate products of three probabilities. It also breaks an
exact reduction: without a missing treatment the composite's initial covariate is the shipped
missing-outcome covariate `1(A=a) Delta/(g pi)` bit for bit, and that covariate is not floored at
`g_lo`.

## The sign of the mechanism correction

**The implementation's sign is the derived one, on the working paper's own appendices.** Only a
variance check could have caught this. All three empirical means are driven to zero, so a flipped
sign moves the variance and leaves the point estimate alone.

**The discrepancy.** The §3.1 display defines `D_A := −(Q_r/g)(A − g)`, with a leading minus.
Theorem 1 then reports `D^{*,#} = D* − D_A − D_Y`. Read off those two displays, the theorem's
mechanism contribution is `+u` where the code's is `−u`.

**What settles it.** The same paper prints the object twice with two signs. §3.2 redefines `D_Y`
for the univariate construction, which is the default here, with **no** leading minus. That is
twelve lines after printing the bivariate one with one. Two displays of one object with opposite
signs is already a reason not to settle the question from a display.

**Appendices A and B decide it.** Each derives one term, and each derivation fixes the orientation.
Each reads `P_0[term] = −(P_n − P_0)·D + B_n + (second order)` with `B_n := P_n·D`. For any `u`
whatever, `P_0[u] = P_n[u] − (P_n − P_0)[u]` is an identity. The decomposition is therefore
satisfiable **only** with `D` equal to the positive term. Appendix A's opening step names the
quantity being decomposed, and it is checkable rather than interpretable:

```text
−P_0{ (Q_r/g_0)·(g_n − g_0) } = P_0{ (Q_r/g_0)·(A − g_n) }      since E_0[A | W] = g_0
```

The right-hand side is positive. `tests/unit/test_theorem_drtmle.py` checks that identity on the
exact law. It also checks that the correction's mean is materially nonzero there, because a zero
mean would make both readings agree and the question unanswerable. The test then checks the
consequence. The asymptotic-linearity representation closes to `1e-12` with the corrections
**subtracted**, and fails by **exactly twice the correction** when they are added. Three mutations
are watched to fail it, one of them the flipped sign in the library itself.

The leading minus in the §3.1 display is therefore not a rival convention to match. It contradicts
the derivation in the same document, Theorem 1's own variance formula, and the exact-law arithmetic
here. The published 2017 article confirms the corrected constructions. It does not replace that
algebraic sign witness. The code follows the identity rather than a display in either edition.

Two further sign slips sit in the same document. Equation (2) prints `+(P_n − P_0)D − B_n` where
its appendices derive `−(P_n − P_0)D + B_n`, which is the same slip under `D → −D`. Equation (2)
also crosses its second-order labels: appendix A's block is the `D_A` one and is collected into
`R_{Q,n}`, while (2) pairs `D_A` with `R_{g,n}`. Neither affects the implementation, which reads
no label. Both are worth knowing before you quote (2).

## Appendix C: a correction that is not needed costs nothing

The paper's own account of why solving an equation you did not need is asymptotically free. Both
answers rest on a vanishing:

```text
D_A = 0  for every g,   because Q_r    = 0 when Q̄ = Q̄_0
D_Y = 0  for every Q̄,   because g_{r,2} = 0 when g  = g_0
```

Each `B` then decomposes into an empirical-process term plus a second-order one, both
`o_p(n^(−1/2))` under the appendices' rate conditions.

**This is also the source of the blindness that shapes every test on this estimator.** `Q_r` and
`g_{r,2}` are zero *row by row* at correct nuisances. Any check taken at the truth is therefore
blind to a flipped sign, to an update order, and to a reduction vintage alike. That is why
`tests/unit/test_theorem_drtmle.py` and `tests/unit/test_influence_gateaux_drtmle.py` are taken at
values where the corrections do **not** vanish, and why their fixtures are misspecified on
purpose.

## The remainder terms, and the rate conditions

The mechanism-misspecification branch, appendix B:

```text
R̃_{5,n} = P_0[ { (A/g_{1,0n,r})·g_{2,0n,r} − (A/g_{1,0,r})·g_{2,0,r} } · (Y − Q̄_n) ]
R̃_{6,n} = P_0[ { (A/g_{1,0,r})·g_{2,0,r}   − (A/g_{1,n,r})·g_{2,n,r} } · (Y − Q̄_n) ]
M̃_{2,n} = (P_n − P_0)[ D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r}) − D_Y(Q̄_0, g_{1,0,r}, g_{2,0,r}) ]
R_{g,n} = R̃_{5,n} + R̃_{6,n} + M̃_{2,n}
```

and the outcome-misspecification branch, appendix A, `R_{Q,n} = R_{3,n} + R_{4,n} + M_{1,n}`:

```text
R_{3,n} = P_0[ { (Q̄_{0n,r} − Q̄_{0,r}) / g_0 } · (g_0 − g_n) ]
R_{4,n} = P_0[ { Q̄_{0,r}/g_0 − Q̄_{n,r}/g_n }  · (g_0 − g_n) ]
M_{1,n} = (P_n − P_0)[ D_A(Q̄_{n,r}, g_n) − D_A(Q̄_{0,r}, g_0) ]
```

with `Q̄_{0n,r}(w) := E_0{Y − Q̄(W) | g_n(W) = g_n(w), g_0(W) = g_0(w)}` evaluated at the
*estimated* propensity as well as the true one, which is what makes `R_{3,n}` an approximation
error rather than a fitted one. Appendix A also carries `R*_n = R_{1,n} + R_{2,n}`, the part that
is second order whichever nuisance is right.

**The paper's rate conditions are illustrative, not necessary.** It states that it *generally
suffices* that, for `R_{g,n}`:

```text
‖Q̄_n − Q̄_0‖_2 = o_p(n^(−1/4))
‖g_{2,0n,r} − g_{2,0,r}‖_2 = o_p(n^(−1/4))
‖g_{2,n,r}  − g_{2,0,r}‖_2 = o_p(n^(−1/4))
```

and for `R_{Q,n}`, in the same "if, for example" form, `o_p(n^(−1/4))` on the reduced regression's
approximation error `‖Q̄_{0n,r} − Q̄_{n,r}‖_2`, on its fitted error `‖Q̄_{n,r} − Q̄_{0,r}‖_2`, and on
the primary propensity's error `‖g_n − g_0‖_2`. Both appendices add an empirical-process pair in
one shape: a `P_0`-Donsker class containing the *estimated* curve, plus `L_2(P_0)` convergence of
the estimated curve to its limit.

**These are the conditions the release claim is conditional on**, and they are conditions on
estimated functions that no fit can check for itself.
