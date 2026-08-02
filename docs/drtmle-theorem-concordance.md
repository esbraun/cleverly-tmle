# DRTMLE: the theorem concordance

What the sources derive, what this package computes, and where the two disagree — with what
`drtmle` computes recorded beside them as **provenance**, since that is where several of these
formulae were read from. It is provenance and not a target: comparing against that package's
numbers is [retired by decision](roadmap.md#closed-since-this-list-opened) and no R enters this
repository or CI. This is [piece A1](roadmap.md#a1--the-theoretical-audit)'s artefact and it is
**open**,
though less so than it was: the working paper is now in the repository and read first-hand, and
the stop-ship discrepancy it appeared to carry — [the sign of the mechanism
correction](#4-the-sign-discrepancy-item-21--resolved) — is **resolved in favour of the implementation**, on
that paper's own appendices.

Its purpose is to make the next reader's audit cheap rather than a re-derivation, so every row
says where it came from. A row with no source is not a row.

## 0. Source inventory

| document | in hand | what it supplies |
| --- | --- | --- |
| Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using drtmle*, Observational Studies 9(2):43–78 | **yes**, `docs/pdf.pdf` | equations (5)–(10), both reduced-regression constructions, the package workflow, the qualitative doubly-robust claims, the multi-level and cross-validation claims |
| Benkeser, Carone, van der Laan & Gilbert (2016), the Berkeley working paper, UCB Biostatistics paper 356 | **yes**, `docs/viewcontent.cgi.pdf` — it was transcribed by the second review before the document itself was in hand | Theorem 1, `D_A`, `D_Y`, `D^{*,#}`, the variance formula, the recursive algorithm, appendix A's bivariate remainder, appendix B's univariate remainder and its sufficient conditions, appendix C on unnecessary one-step corrections |
| Benkeser, Carone, van der Laan & Gilbert (2017), Biometrika 104(4):863–880, PMC5793673 | **no** | the *published* Theorem 1, which is authoritative wherever the working paper and it differ |
| van der Laan (2014), IJB 10(1):29–57, Theorem 3 | **no** | the bivariate construction's regularity conditions |
| `benkeser/drtmle` 1.1.2 source | **yes**, read | where several formulae here were transcribed from, and the names they carry there — provenance, not a check |

Two consequences, and they are the reason this file exists rather than a paragraph:

- **The 2023 article cannot close A1 and never could.** It refers the reader to van der Laan
  (2014) Theorem 3 and Benkeser et al. (2017) Theorem 1 for the regularity conditions and the
  corrected influence function respectively, and reproduces neither, nor appendix B. It is the
  equation-level specification and the software-paper claim set; that is all.
- **The working paper closes most of the rest, and what looked worse turned out to be internal.**
  It supplies every theorem object A1 needed. Its §3.1 *display* also disagrees with both
  implementations on a sign — and with its own appendices, which is what settles it. See
  [§4](#4-the-sign-discrepancy-item-21--resolved).

```text
Benkeser et al., Theorem 1
- theorem statement available:                  YES (2016 working paper, docs/viewcontent.cgi.pdf)
- corrected influence function available:       YES
- appendix A and B remainder terms available:   YES
- the sign question (item 21):                  RESOLVED, on the paper's own appendices
- published (2017) version checked against it:  NO   <-- no longer a gate; see below
```

**Reading the document rather than a transcription of it changed the answer.** The transcription
was faithful — the §3.1 display really does carry a leading minus on `D_A` — but a display is not
a derivation, and the derivation is twenty pages later. This is
[lesson 6](drtmle-investigation-log.md#what-the-sizings-got-wrong) again: the cheapest instrument
for a claim about a source is a reader with the source open, and second-hand is not open.

## 1. The target and the ordinary score

*Source: Benkeser & Hejazi (2023), and standard.*

For treatment level `a`, with `Q̄_0(a, w) = E_0(Y | A = a, W = w)`,

```text
psi_0(a) = E_0{ Q̄_0(a, W) },      ATE = psi_0(1) − psi_0(0)
```

under the usual conditional-randomisation and positivity assumptions for the causal reading. The
working paper states the univariate construction for `Ψ(P) = ∫ Q̄(w) dQ_W(w)` at `A = 1`; the
`a = 0` and multi-level cases replace `A` and `g(1|W)` by the arm indicator and arm probability
throughout.

The ordinary efficient influence function is

```text
D*(Q, g)(O) = A/g(W) · {Y − Q̄(W)} + Q̄(W) − Ψ(Q)
```

and the ordinary TMLE score equation, the article's **equation (5)**, is

```text
(1/n) Σ_i 1(A_i = a)/g_n(a|W_i) · {Y_i − Q̄*_n(a, W_i)} = 0
```

**Three different things wear this equation's clothes and A1 must keep them apart**: the exact
finite-sample equality the software article writes, the theorem-level requirement that the
empirical score be `o_p(n^(−1/2))`, and the implementation's numerical stopping criterion. They
are not interchangeable, and [item 12](roadmap.md#closed-since-this-list-opened) is what happens
when the last stands in for the second.

The ordinary second-order remainder is

```text
R(Q_1, Q_2, g_1, g_2) = P_0[ (g_1 − g_2)(Q̄_1 − Q̄_2) / g_1 ]
```

and the theorem assumes the **true** mechanism is bounded away from zero, `g_0(w) > δ > 0`.

## 2. The theorem's objects

*Source: the 2016 working paper, univariate construction.*

**Reduced outcome regression.**

```text
Q̄_{0,r}(Q̄, g)(w) = E_0[ Y − Q̄(W) | A = 1, g(W) = g(w) ]
```

**Univariate reduced propensity regressions.** Two of them, both conditioning on `Q̄(W)` alone —
which is what makes this construction univariate where van der Laan (2014)'s is bivariate:

```text
g_{1,0,r}(Q̄)(w)      = E_0[ A | Q̄(W) = Q̄(w) ]
g_{2,0,r}(Q̄, g)(w)   = E_0[ (A − g(W))/g(W) | Q̄(W) = Q̄(w) ]
```

**The two corrections**, as the §3.1 and §3.2 *displays* print them. The leading minus on the
first is [§4](#4-the-sign-discrepancy-item-21--resolved)'s whole subject, and the answer there is that the
paper's own appendices contradict it — so these two lines are a transcription of the displays and
**not** what the theorem derives:

```text
D_A(Q̄_{0,r}, g)(O)              = − Q̄_{0,r}(W)/g(W) · {A − g(W)}      <-- display; see §4
D_Y(Q̄, g_{1,0,r}, g_{2,0,r})(O) =   A/g_{1,0,r}(W) · g_{2,0,r}(W) · {Y − Q̄(W)}
```

Note that §3.1 prints the *bivariate* `D_Y` with a leading minus as well, and §3.2 reprints the
same object without one. One of the two is wrong on its face, whatever is decided about `D_A`.

**The limiting influence function.**

```text
D^{*,#}(Q, g) = D*(Q, g) − I(g = g_0)·D_A(Q̄_{0,r}, g)
                         − I(Q̄ = Q̄_0)·D_Y(Q̄, g_{1,0,r}, g_{2,0,r})
```

The indicators are the doubly-robust content in one line: when both primary nuisances are
correct, both corrections vanish and the ordinary efficient influence function is recovered — the
same degeneracy that makes the exact-law instrument blind here
([lesson 2](drtmle-investigation-log.md#what-the-sizings-got-wrong)).

## 3. Theorem 1

*Source: the 2016 working paper, notation adapted to this file's.*

**Suppose** either `Q̄ = Q̄_0` or `g = g_0`, and let the targeted collection
`(Q̄*_n, Q̄*_{n,r}, g*_n, g*_{1,n,r}, g*_{2,n,r})` satisfy the three empirical score conditions

```text
B_n     = P_n D*(Q*_n, g*_n)                                = o_p(n^(−1/2))
B_{A,n} = P_n D_A(Q̄*_{n,r}, g*_n)                           = o_p(n^(−1/2))
B_{Y,n} = P_n D_Y(Q̄*_n, g*_{1,n,r}, g*_{2,n,r})             = o_p(n^(−1/2))
```

**and** suppose the appendix second-order terms satisfy `R_{Q,n} = o_p(n^(−1/2))` and
`R_{g,n} = o_p(n^(−1/2))`. **Then** the targeted plug-in `ψ^{*,c}_n = Ψ(Q*_n)` is asymptotically
linear with influence function `D^{*,#}(Q, g)`, and `√n(ψ^{*,c}_n − ψ_0) ⇝ N(0, σ²)`, with

```text
σ̂²_n = P_n [ D*(Q*_n, g*_n) − D_A(Q̄*_{n,r}, g*_n) − D_Y(Q̄*_n, g*_{1,n,r}, g*_{2,n,r}) ]²
```

Three things to read off it, because each is a row in the assumption matrix below:

- the score conditions are `o_p(n^(−1/2))`, **not** exact zeros. The implementation's exact-zero
  ambition is stricter than the theorem and its numerical stopping rule is not obviously either;
- the conditions are on the **targeted** collection, including the *starred reduced* nuisances,
  which is why [limitation 8](roadmap.md#limitations-recorded-rather-than-fixed) — that `retarget`
  is no longer arithmetic on cached arrays — is a cost of following the source rather than a
  design slip;
- the variance is `P_n[·]²` of the **rowwise** corrected curve, so nothing that summarises the
  curve before squaring is computing this.

**Arms and the contrast.** The theorem is stated for one treatment-specific mean. For `a = 0`,
replace `A` and `g(1|W)` with the arm indicator and arm probability. The ATE influence function is
the **rowwise difference** of the two arm-level curves — which is what this package computes and
what makes an ATE-only diagnostic insufficient, since arm-specific errors cancel in a difference.

## 4. The sign discrepancy (item 21) — resolved

**Resolved in favour of the implementation, on the working paper's own appendices.** It was the
highest-priority open question on this variant; what follows is the charge, what settles it, and
what is left.

Write `u` and `v` for the two **positive** quantities the software computes:

```text
u := (Q_r/g)·{A − g}                     inference/influence.py::reduced_correction_parts, d_g
v := 1_a·(g_{r,2}/g_{r,1})·{Y − Qbar}                                                     d_q
```

### The charge

The §3.1 display defines `D_A := −u`, with a leading minus, and Theorem 1 reports
`D^{*,#} = D* − D_A − D_Y`. Read off those two, the theorem's mechanism contribution is `+u` while
the code's is `−u` — and the disagreement is with `drtmle` as well. The item was filed on `D_A`
alone, correctly: §3.2's `D_Y`, which is the one the univariate construction uses, carries no
leading minus, so `− D_Y = −v` agrees with the code and was never in dispute. §3.1's *bivariate*
`D_Y` does carry one, which is where the second thread below starts.

### What settles it

**The same paper prints the object twice with two signs.** §3.2 redefines `D_Y` for the
univariate construction — the one this package implements — as `a·(g_{2,r}/g_{1,r})·{y − Qbar}`,
with **no** leading minus. Two displays of one object with opposite signs is already a reason not
to settle the question from a display.

**Appendices A and B derive both terms, and each derivation fixes the orientation.** Each reads

```text
P_0[ term ] = −(P_n − P_0)·D + B_n + (second order),        with B_n := P_n·D
```

and `P_0[u] = P_n[u] − (P_n − P_0)[u]` is an identity for any `u` whatever. So the decomposition
is satisfiable **only** with `D` equal to the positive term: appendix A forces `D_A = +u`, and
appendix B forces `D_Y = +v`. Then Theorem 1's own `D^{*,#} = D* − D_A − D_Y` is `D* − u − v`,
which is what the code and `drtmle` compute, and its `σ²_n = P_n{D* − D_A − D_Y}²` is the variance
they report.

Appendix A's opening step is the one that says *which* quantity is being decomposed, and it is
checkable rather than interpretable:

```text
−P_0{ (Qbar_r/g_0)·(g_n − g_0) } = P_0{ (Qbar_r/g_0)·(A − g_n) }      since E_0[A | W] = g_0
```

The right-hand side is `u`, positive. `tests/unit/test_theorem_drtmle.py` checks that identity on
the exact law, checks that the correction's mean is materially non-zero there (without which both
readings would agree and the question would be unanswerable), and checks the consequence: the
asymptotic-linearity representation closes to `1e-12` with the corrections **subtracted** and
fails by **exactly twice the correction** when they are added. It also pins the arrays
`reduced_correction_parts` builds against `u` and `v`, so the adjudication is a statement about
this package and not only about the paper. Watched to fail against three mutations, one of them
the flipped sign in the library itself.

So the leading minus in the §3.1 display is not a rival convention to be matched. It is
inconsistent with the derivation in the same document, with Theorem 1's own variance formula, and
with the exact-law arithmetic in this repository.

### What is left, and it is not a gate

The **published 2017** article is still unread. It no longer blocks anything: the adjudication is
on internal consistency plus arithmetic, and neither depends on the edition. If the published
version prints the same §3.1 display, the display is wrong there too; if it prints `D_A = +u`, the
working paper's display was a typo and the published version says so. Either way the code does not
move. Should someone obtain it, the row to check is the §3.1 display and nothing else.

Two further sign observations from the same reading, recorded because they are the kind of thing a
later reader will otherwise re-derive:

- **Equation (2) is printed with `+(P_n − P_0)D − B_n`, and its appendices derive `−(P_n − P_0)D +
  B_n`.** The two are the same statement under `D → −D`, which is the same slip as the display's.
- **Equation (2) also crosses its second-order labels**: appendix A's block, which is the `D_A`
  one, is collected into `R_{Q,n}`, while (2) pairs `D_A` with `R_{g,n}`. Harmless for the
  implementation — nothing here reads those labels — and worth knowing before quoting (2).

## 5. The remaining remainder terms

*Source: the 2016 working paper, appendix A (bivariate) and appendix B (univariate).*

The mechanism-misspecification branch is built from three terms:

```text
R̃_{5,n} = P_0[ { (A/g_{1,0n,r})·g_{2,0n,r} − (A/g_{1,0,r})·g_{2,0,r} } · (Y − Q̄_n) ]
R̃_{6,n} = P_0[ { (A/g_{1,0,r})·g_{2,0,r}   − (A/g_{1,n,r})·g_{2,n,r} } · (Y − Q̄_n) ]
M̃_{2,n} = (P_n − P_0)[ D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r}) − D_Y(Q̄_0, g_{1,0,r}, g_{2,0,r}) ]

R_{g,n} = R̃_{5,n} + R̃_{6,n} + M̃_{2,n}
```

and the outcome-misspecification branch retains appendix A's `R_{Q,n} = R_{3,n} + R_{4,n} + M_{1,n}`.

**The paper's rate conditions are illustrative, not necessary.** For `R_{g,n} = o_p(n^(−1/2))` it
states that it *generally suffices* that

```text
‖Q̄_n − Q̄_0‖_2 = o_p(n^(−1/4))
‖g_{2,0n,r} − g_{2,0,r}‖_2 = o_p(n^(−1/4))
‖g_{2,n,r} − g_{2,0,r}‖_2 = o_p(n^(−1/4))
```

together with a `P_0`-Donsker condition on the estimated `D_Y` class and `L_2(P_0)` convergence of
the estimated `D_Y` to its limit. Appendix A gives the analogous examples for `R_{Q,n}`, involving
the reduced outcome regression's approximation error, its fitted error, the primary propensity
error, and a Donsker plus `L_2` condition for `D_A`.

**Consequence for item 13**, which [A1](roadmap.md#a1--the-theoretical-audit) opens and
[C](roadmap.md#c-the-demonstration) closes. The single diagnostic

```text
R_remaining = ψ̂ − ψ_0 − (P_n − P_0) D̂_DR
```

stays useful and is not sufficient: a total trending to zero can conceal cancellation between
`R_Q` and `R_g`. Where the DGP permits, [the study](drtmle-validation-plan.md#5-the-controlled-study-piece-c)
must report `R_Q` and `R_g` **separately**, their component products, their signs, and the total.

## 6. The recursive algorithm (item 22)

*Source: the 2016 working paper.* With

```text
H_1(g)(a, w)       = a/g(w)
H_2(g_1, g_2)(a, w) = a·g_2(w)/g_1(w)
H_3(Q̄_r, g)(w)     = Q̄_r(w)/g(w)
```

the procedure is:

1. initialise `Q̄⁰_n` and `g⁰_n`;
2. fluctuate `Q̄^k_n` along `H_1(g^k_n)`;
3. estimate `g^k_{1,n,r}` and `g^k_{2,n,r}` using `g^k_n` and the **once-updated** outcome
   regression;
4. fluctuate the outcome regression along `H_2`;
5. estimate `Q̄^k_{n,r}` using `g^k_n` and the **twice-updated** outcome regression;
6. fluctuate `g^k_n` along `H_3(Q̄^k_{n,r}, g^k_n)`;
7. iterate until the empirical means of `D*`, `D_A` and `D_Y` are approximately zero;
8. return the final targeted collection.

The transcription above is confirmed against `docs/viewcontent.cgi.pdf` first-hand. Two details it
did not carry, both worth having: the outcome fluctuations at steps 2 and 4 are fitted **using
only data points with `A = 1`**, which is the package's per-arm indicator design; and the
procedure has no closing pass and no truncation anywhere in it
([§7](#7-truncation-is-not-in-the-theorems-algorithm)).

**The Python iteration is not an exact transcription of this order, and that is item 22 — which
the source narrows to a numerical question.** Step 7 states the exit condition as *exactly* the
three empirical means being approximately zero. So the paper asks for a fixed point satisfying
the three equations and prescribes the order only as one way of reaching it; an order that
reaches a point satisfying step 7 has done what the theorem requires, and Theorem 1's own
hypotheses are conditions on the returned collection rather than on the route. That is what makes
the difference licensed rather than merely unchecked.

What is *not* settled by reading is whether the two orders reach the same fixed point on real
data, which is a numerical claim and belongs to [A1](roadmap.md#a1--the-theoretical-audit):
implement the paper's order beside this one, **both here and against the same nuisances**, and
compare the fixed point each reaches and the final three theorem-defined scores at each. The
instrument for the last of those now exists — `res.validation.correction_check()` reports each
score at the returned state, per arm — so the comparison is a run rather than a construction.
A third implementation reaching a third fixed point would have answered a different question, which
is one reason this stopped being the parity piece's.

**Do not compare fluctuation coefficients across algorithms unless the submodels and the update
order are identical.** R tilts each arm's mechanism in its own one-column `glm` where this package
solves one two-column tilt, so its `epsilon` and this package's are different quantities already
([§12](#12-multi-valued-treatment-and-the-simplex)); adding an order difference makes a coefficient
comparison meaningless twice over. The scores are what must agree — and since the two orders being
compared are now **both run here**, the rule bites on the comparison this file actually asks for.

## 7. Truncation is not in the theorem's algorithm

The theorem assumes `g_0 > δ > 0` and defines every mechanism-side object at the **same** `g`. Its
mechanism update is

```text
g^{k+1}_n(w) = expit[ logit{g^k_n(w)} + ε_{3,n,k}·H_3(Q̄^k_{n,r}, g^k_n)(w) ]
```

with **no hard clipping after the fluctuation**. So the theorem as written supports *neither* of
the conventions this package and R respectively use:

- not the current hybrid — bounded denominator, raw residual — which is
  [item 20](drtmle-investigation-log.md#item-20-from-discovery-to-cause);
- not R's post-fit clipping, as an exact theorem step.

That is a stronger statement than "the two conventions disagree", and it is what makes
[B1b](roadmap.md#b1b--the-theorem-conforming-targeting-decision) a derivation rather than a taste
question. Three things follow:

1. one mechanism must appear consistently in the score and in the influence function;
2. if a finite bound is required in practice, it wants a **bounded submodel or a constrained
   estimating equation whose final score is the theorem-defined score** for the estimator being
   declared — not a projection applied after an unconstrained optimisation;
3. theorem validity does not follow from R carrying its clipped result consistently.

The unbounded `expit` update is the closest literal implementation, subject to the boundedness and
rate conditions actually holding. A **smooth** bounded submodel is likely preferable to hard
clipping wherever practical positivity control is needed, because hard clipping is a non-smooth
projection performed after the optimisation and the unconstrained first-order condition is not the
first-order condition of the clipped state.

## 8. Cross-fitting is not covered (item 15)

The theorem uses Donsker and `L_2`-convergence conditions for the empirical-process terms. **It
presents no cross-fitted version**, so the working paper does not close item 15. The 2023
article's statement that cross-validated nuisance estimates weaken the entropy conditions — made
for ordinary TMLE/AIPTW and for the doubly-robust TML estimator alike — is supporting evidence
that cross-validated DRTMLE is intended and implemented, and it is not the missing proof.

What is specifically unaddressed is this package's **pooled** construction, in which an
observation influences other rows' generated regressors and then returns to its own
reduced-regression prediction through those rows. Generic cross-fitting results do not
automatically cover that, because the conditional independence they turn on is exactly what the
generated design breaks. `fit_reduced`'s docstring reaches the right conclusion for the wrong kind
of reason: it shows an independent split removes *none* of the induced dependence (the
contamination is in the design values, not in which rows train) and that per-fold designs would
trade a second-order dependence for a first-order covariate shift. Both are sound and neither
establishes that the induced dependence is higher order in the DRTMLE expansion, which is what the
theorem needs.

```text
cross-validation claimed applicable by Benkeser & Hejazi (2023):  YES
exact pooled construction analysed anywhere in the sources:       NO
theorem or proof for that construction:                           ABSENT
status:                                                           UNVERIFIED
```

Both tracks stay live: a proof or expansion for the pooled construction, **and** a
nested/per-outer-fold reference estimator. The nested version need not become the default; it
provides a construction with clearer conditional independence, an empirical comparator for the
cheap one, and a way to see whether the pooled dependence changes bias, variance or remainder
rates. **Agreement with R would not have been evidence here** — that package predates this
construction — which is one of several reasons the parity piece was never going to earn its keep.
This is A1's work.

## 9. What was read out of the R source, and what is still owed

This section used to be *six traps for reading the R source alongside the paper* — advice for a
parity run. There will be no parity run: it is [retired by
decision](roadmap.md#closed-since-this-list-opened), no R enters this repository or CI, and four of
the six traps were only ever about how to compare two implementations without fooling yourself.

Two of the six are not advice. They are **facts recorded once** and they survive here, because
nothing else in this repository writes them down. Both were read out of `benkeser/drtmle` at
**version 1.1.2** (`R/drtmle.R`, `R/estimate.R`, `R/fluctuate.R`, `R/inf_functions.R`) rather than
recalled; the naming inversion the deleted first trap described lives on in
[§13](#13-the-object-concordance)'s `(swapped)` markers and in
`src/cleverly/estimators/reduced.py`.

- **`_NEGLIGIBLE / n` is R-shaped and not an invention.** R's stopping rule is `tolIC = 1/n` on the
  mean of the *reported* correction terms, tested against `max|c(PnDnoStar, PnDQnStar, PnDgnStar)|`
  where each is `mean()` of the array `eval_Dstar*` returns, and capped at `maxIter = 3`. The
  absolute branch [item 7](roadmap.md#closed-since-this-list-opened) added here is the same shape.
  Worth keeping precisely because it is the *weaker* claim: it says where the constant came from,
  not that it is right — and [piece B2](roadmap.md#b2--the-sweep-on-the-corrected-implementation)
  is where the loop's bar stops being a proxy for the reported one at all. Note also that R's
  convergence test is defined *on the curve it reports* where this package's is defined on what the
  solver recorded, which is the difference
  [item 20](drtmle-investigation-log.md#item-20-from-discovery-to-cause) lives in.
- **`D*_g` and the missing-outcome indicator, which is an open derivation question and not a
  difference.** R's `eval_Dstar_g` is `Qr/g · (1{A = a, DeltaA = 1, DeltaY = 1} − g)`;
  `reduced_corrections` applies `observed` to `D*_Q` and **not** to `D*_g`. It is not live —
  `DRTMLE` refuses `delta=`, so no fit it accepts has a missing outcome — but it is the thing to
  settle *before* that refusal is lifted, and it has to be settled from the derivation. This is the
  cleanest small example of why the parity piece was retired: every fixture would have agreed, on
  every draw, because the quantity that distinguishes the two conventions is identically absent
  from the fits either package accepts. A check that cannot fail is not a check
  ([lesson 8](drtmle-investigation-log.md#what-the-sizings-got-wrong)). It is carried as a row in
  the assumption matrix below.

The other three deleted traps were: that R's internal signs are confirmed and are what this package
computes (now [§4](#4-the-sign-discrepancy-item-21--resolved)'s subject and settled from the
appendices, not from R); that R tilts each arm's mechanism separately, so its `epsilon` and this
package's are different quantities and only the *scores* were ever comparable — kept in
[§12](#12-multi-valued-treatment-and-the-simplex) where the simplex question lives; and that
`nuisance_drtmle$grnStar` holds the *initial* `grn`, which was a warning about how to read a
returned object in a fixture nobody will now write.

## 10. The bivariate construction

*Source: Benkeser & Hejazi (2023), attributing van der Laan (2014).* Carried here because
[piece D](roadmap.md#d-widen-the-scope-to-what-the-sources-derive) is gated on it and because
"derived in the sources" was being read as "transcription".

```text
Q̄_{r,0n}(a, w) = E_0[ Y − Q̄_n(W) | A = a, g_n(W) = g_n(w) ]              (6)
g_{r,0n}(a|w)  = P_0[ A = a | Q̄_n(W) = Q̄_n(w), g_n(W) = g_n(w) ]         (7)
```

and the three score equations the targeted OR, PS, R-OR and bivariate R-PS satisfy:

```text
(1/n) Σ 1(A_i=a)/g*_n(a|W_i) · {Y_i − Q̄*_n(a,W_i)}                       = 0   (8)
(1/n) Σ Q̄*_{r,n}(a,W_i)/g*_n(a|W_i) · {1(A_i=a) − g*_n(a|W_i)}           = 0   (9)
(1/n) Σ 1(A_i=a)/g*_{r,n}(a|W_i)
       · {(g*_{r,n}(a|W_i) − g*_n(a|W_i))/g*_n(a|W_i)}
       · {Y_i − Q̄*_n(a,W_i)}                                            = 0   (10-biv)
```

The article states, in substance, that under the regularity conditions of van der Laan (2014)
Theorem 3 a G-computation estimator based on a `Q̄*_n` satisfying these is doubly robust with
respect to both consistency and its limiting distribution, and that those conditions include
convergence-rate assumptions for the OR, the PS, the R-OR and the bivariate R-PS.

```text
source status: equations reproduced in Benkeser & Hejazi (2023)
theorem status: requires van der Laan (2014), Theorem 3 — NOT IN HAND
```

Missing with it: the formal statement, its complete assumptions, the asymptotic linear expansion,
its influence function, its remainder decomposition, any treatment of truncation, and any theorem
for a cross-fitted version. One implementation detail to carry over from R rather than rediscover:
its bivariate branch of `eval_Dstar_Q` is `1{A=a}/grn2 · (grn2 − g)/g · (Y − Q)` and the `g` there
is the **initial** mechanism, not the targeted one — `drtmle.R` passes `gn = gn` into that call in
both the loop and the covariance block. On the univariate branch the argument is unused, so this
is a difference that only appears when the bivariate reduction is written.

## 11. The univariate equations, as the software article states them

The article replaces the bivariate R-PS with two univariate regressions,

```text
g_{r,0,1}(a|w) = P_0[ A = a | Q̄_n(W) = Q̄_n(w) ]                          (R-PS1)
g_{r,0,2}(a|w) = E_0[ {1(A=a) − g_n(a|W)}/g_n(a|W) | Q̄_n(W) = Q̄_n(w) ]   (R-PS2)
```

and states the third score equation as

```text
(1/n) Σ 1(A_i=a)/g*_{r,n,1}(a|W_i) · g*_{r,n,2}(a|W_i) · {Y_i − Q̄*_n(a,W_i)} = 0   (10-uni)
```

which is the equation this package's reduced-outcome fluctuation solves. The article's stated
consequence is that a G-computation estimator based on such a `Q̄*_n` has a doubly-robust limiting
distribution, that Benkeser et al. (2017) supplied both the algorithm and closed-form doubly
robust variance estimates, and — separately worth keeping — that **adding the extra score terms to
an AIPTW-style estimator does not generally produce the same property**; the targeted construction
is what earns it.

## 12. Multi-valued treatment and the simplex

The 2023 article states that `drtmle` handles an arbitrary finite number of discrete levels and
describes a sequential-binomial construction for the *initial* propensity that ensures
`Σ_a g_n(a|w) = 1`. It works a three-level example and builds a covariance matrix for the
treatment-specific means and their contrasts.

```text
software support in the article:                       YES
formal multi-arm theorem reproduced anywhere in hand:  NO
simplex-compatible initial PS construction described:  YES
simplex compatibility of the TARGETED PS established:  NO
```

**The non-normalisation of the targeted mechanism is an unasked question, not a known defect.** An
algorithm may use arm-indexed nuisance updates as working objects for separate estimating
equations without claiming that the updated values jointly define one conditional treatment
distribution. What the theorem requires — a single valid joint `g*(·|W)`, or only arm-specific
functions satisfying the stated equations — is exactly the sort of thing a theorem says and an
example does not, and it is unread. For the binary implementation the honest description is *an
arm-specific parameterisation whose joint-distribution interpretation must be checked against the
theorem*. For a new multi-arm implementation, prefer a **simplex-preserving multinomial
fluctuation** unless the theorem explicitly licenses independent armwise updates; contrasts need a
coherent joint covariance construction whether or not each arm mean is targeted separately.

## 13. The object concordance

The permanent table. **Rows marked `TODO` are what A1 has left to do**, and a status column with
no `unverified` in it has been filled in from the code rather than from the paper.

The **R** column is *provenance*: it records where each formula in this package was read from, and
the two `(swapped)` markers are the single easiest thing here to transcribe backwards. It is not a
target. Comparing against that package's numbers is [retired by
decision](roadmap.md#closed-since-this-list-opened) and there is no R in this repository or in CI.

The **evidence** column is what that retirement was traded for: which test pins the row *against
its derivation*. It is the checklist item 2 used to be, restated as tests to write. A row reading
`TODO` has no such test — not "probably fine"; and a column with no `TODO`s on first pass has been
filled in from optimism, exactly as §15's `unverified` column says of itself.

| theorem object | Python | R (provenance) | conditions on | sign | denominator / truncation | initial or starred | arm-specific | consumed by | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Ψ` | `counterfactual_means` | GCOMP estimate | — | + | — | starred `Q̄` | yes | the report | `test_reduction_alternation.py` (at the truth it is plain `TMLE` array for array) |
| `D*` | ordinary curve, `influence.py` | `eval_Dstar` | `A`, `W` | + | `g*`, bounded | starred | yes | eq (8), `D_DR` | the `test_influence_gateaux*` modules, for the *plain* curve |
| `Q̄_n`, `g_n` | initial cross-fitted predictions | `estimateQ`, `estimateG` | `W` | — | `g` bounded at use | initial | yes | every reduction's design | `TODO` — no component check at the initial predictions |
| `Q̄_{0,r}` / `Q_r` | `ReducedSet.qr` | `estimateQrn` | `A = a`, `g_n(W)` | + | — | starred in eqs | yes | eq (9), `D_A`/`D*_g` | `test_reduced_regressions.py`, against `test_remainder_drtmle.py`'s longhand |
| `g_{1,0,r}` | `ReducedSet.gr1` | `grn2` **(swapped)** | `Q̄_n(W)` | + | `bounded_gr1` | starred in eqs | yes | eq (10-uni), `D_Y` | `test_reduced_regressions.py`; the inversion trap at `test_reduced_submodel.py::test_but_gr1_does_not_vanish` |
| `g_{2,0,r}` | `ReducedSet.gr2` | `grn1` **(swapped)** | `Q̄_n(W)` | signed | fixed at fit time (limitation 9) | starred in eqs | yes | eq (10-uni), `D_Y` | as above |
| `D_A` | `D*_g` | `eval_Dstar_g` | `A`, `W` | + ([§4](#4-the-sign-discrepancy-item-21--resolved)) | `g*`; truncation convention open | starred | yes | `D_DR`, eq (9)'s check | `test_theorem_drtmle.py`, **at nonzero `Q_r`** — the one instrument here an exact law cannot supply |
| `D_Y` | `D*_Q` | `eval_Dstar_Q` univariate branch | `A`, `W`, `Y` | + | `g_{r,1}` bounded | starred | yes | `D_DR`, eq (10)'s check | `test_theorem_drtmle.py`; `test_influence_drtmle.py` for the longhand |
| `D^{*,#}` | `D = D* − D*_Q − D*_g` | `DnoStar − DnQoStar − DngoStar` | — | + ([§4](#4-the-sign-discrepancy-item-21--resolved)) | — | starred | rowwise per arm; ATE is the rowwise difference | the variance | `test_influence_drtmle.py` (difference not sum; per-guard membership). **`TODO`**: no Gateaux-style pin of the decomposition against a perturbation of the law |
| `B_n`, `B_{A,n}`, `B_{Y,n}` | the three recorded scores | `PnDnoStar` etc. | — | — | **the identity B1a pins** | starred | yes | the stopping rule and `score_check` | `test_drtmle_fit.py` and `validation/drtmle.py`'s `correction_check` |
| `R_{Q,n}`, `R_{g,n}` | not computed | not computed | — | — | — | — | — | item 13, `TODO` | `TODO` — piece C's column |
| `σ̂²_n` | `influence_covariance` | `drtmle` covariance block | — | — | — | — | — | the interval | `TODO` — pinned only through the curve it is built from |
| the probability limits `Q̄_1`, `g_1` | only in tests | — | — | — | — | — | — | `TODO` | `TODO` |

## 14. What the sources supply and what they do not

| required result | in hand | from where | still missing |
| --- | --- | --- | --- |
| ordinary TMLE score, usual DR consistency claim | yes | 2023 art., eq (5) | full standard regularity theory if needed |
| bivariate reduced regressions | yes | 2023 art., (6)–(7) | — |
| bivariate targeting equations | yes | 2023 art., (8)–(10) | — |
| **bivariate theorem** | **no** | qualitative consequence only | van der Laan (2014), Thm 3 |
| univariate R-PS1 / R-PS2 | yes | 2023 art. | — |
| univariate third score | yes | 2023 art., (10-uni) | — |
| **Theorem 1 statement** | yes | 2016 working paper | the **published 2017** version |
| **corrected influence function** | yes | 2016 working paper | nothing — the display/appendix sign conflict is resolved in [§4](#4-the-sign-discrepancy-item-21--resolved) |
| appendix B remainder terms | yes | 2016 working paper | — |
| reduced-regression rate conditions | yes, as *sufficient* examples | 2016 working paper, app. A/B | necessary conditions, if wanted |
| empirical-process conditions | yes | 2016 working paper | — |
| recursive algorithm | yes | 2016 working paper | — |
| **truncation theorem** | **no** | not stated anywhere in hand | an original derivation or a new proof — [§7](#7-truncation-is-not-in-the-theorems-algorithm) |
| **pooled cross-fitting theorem** | **no** | general CV claim only | a new argument or a reference construction — [§8](#8-cross-fitting-is-not-covered-item-15) |
| **multi-arm theorem** | **no** | software example only | the 2017 paper's multi-arm case, or a derivation |
| weights, estimated or fixed | **no** | — | item 17 closed the transport on the exact law; the theorem says nothing |
| repeated sample splitting | **no** | — | item 18 closed the arithmetic; the theorem says nothing |

## 15. Assumptions, and which the implementation meets

One row per condition. **`unverified` is a permitted answer and is the point of the column.** The
statuses below are the state on the day this file was seeded, not a result.

| condition | source | required for | what the implementation does | evidence | status |
| --- | --- | --- | --- | --- | --- |
| `Q̄ = Q̄_0` **or** `g = g_0` | Thm 1 | the whole conclusion | assumed, not checked | — | met by assumption; the union model is the point |
| `g_0 > δ > 0` (true mechanism) | Thm 1 | boundedness | assumed; a *fitted* `g` is truncated instead | positivity warning | **unverified** — the theorem bounds `g_0`, the code bounds `ĝ` |
| `B_n = o_p(n^(−1/2))` | Thm 1 | eq (8) | solved to `1e-11` relative or `_NEGLIGIBLE/n` absolute | sweep | met, under a numerical proxy for `o_p` |
| `B_{A,n} = o_p(n^(−1/2))` | Thm 1 | eq (9) | solved at the **raw** residual; the curve reads the truncated one | item 20 | **violated** until B1 lands |
| `B_{Y,n} = o_p(n^(−1/2))` | Thm 1 | eq (10) | solved exactly | tests | met |
| `R_{Q,n} = o_p(n^(−1/2))` | app. A | asymptotic linearity | unmeasured | — | **unverified** — item 13 |
| `R_{g,n} = o_p(n^(−1/2))` | app. B | asymptotic linearity | unmeasured | — | **unverified** — item 13 |
| Donsker / `L_2` for `D_A`, `D_Y` | app. A/B | the empirical-process terms | cross-fitting, pooled | `fit_reduced` docstring | **unverified** — item 15 |
| reduced regressions consistent | Thm 1 | the corrections' limits | estimated, unmeasured rates | — | **unverified** |
| exact zeros vs `o_p(n^(−1/2))` | Thm 1 | the stopping rule | numerical criterion | item 12 | met under a stated restriction |
| arm-level means / ATE contrast | Thm 1 + adaptation | the reported parameters | rowwise difference of arm curves | — | met; the adaptation is stated, not cited |
| hard truncation of `ĝ` | **nowhere** | the implementation as written | applied, inconsistently | item 20 | **not covered by the source** — [§7](#7-truncation-is-not-in-the-theorems-algorithm) |
| the mechanism correction's sign | Thm 1 | the variance | the appendices' orientation | [§4](#4-the-sign-discrepancy-item-21--resolved), `test_theorem_drtmle.py` | **met**; the §3.1 display disagrees and its own appendices contradict it — item 21, closed |
| the update order | Thm 1's algorithm | nothing, if the fixed point is the same | different order | [§6](#6-the-recursive-algorithm-item-22) | **met under a stated restriction**: the paper's step 7 states its own exit as the three scores, so the order is not prescriptive; whether the fixed points coincide numerically is A1's, both orders run here — item 22 |
| fixed weights | **nowhere** | item 17's claim | weighted loss throughout | `test_remainder_drtmle.py` | met for a **fixed** weight; estimated weights not covered |
| repeated sample splitting | **nowhere** | item 18's claim | mean over draws | `test_drtmle_fit.py` | met arithmetically; not covered by the source |
| `K` arms | **nowhere** | piece D | binary only | — | **not covered by the source** — [§12](#12-multi-valued-treatment-and-the-simplex) |
| missing outcomes | **nowhere**; `drtmle` masks `D*_g` and this package does not | a lifted `delta=` | refused, so the two conventions never differ on a fit either package accepts | [§9](#9-what-was-read-out-of-the-r-source-and-what-is-still-owed) | **not covered by the source** — settle from the derivation *before* lifting the refusal; no run could ever have settled it |
| composition with `CTMLE` | **nowhere** | — | refused | — | **not covered by the source** |
