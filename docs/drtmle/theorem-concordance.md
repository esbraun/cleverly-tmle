# DRTMLE: the theorem concordance

What the sources derive, what this package computes, and where the two disagree — with what
`drtmle` computes recorded beside them as **provenance**, since that is where several of these
formulae were read from. It is provenance and not a target: comparing against that package's
numbers is [retired by decision](../roadmap.md#closed-since-this-list-opened) and no R enters this
repository or CI. This is [piece A1](../roadmap.md#a1a--the-theoretical-audit)'s artefact and it is
**open**,
though less so than it was: the working paper is now in the repository and read first-hand, and
the stop-ship discrepancy it appeared to carry — [the sign of the mechanism
correction](#4-the-sign-discrepancy-item-21--resolved) — is **resolved in favour of the implementation**, on
that paper's own appendices.

Its purpose is to make the next reader's audit cheap rather than a re-derivation, so every row
says where it came from. A row with no source is not a row.

## 0. Source inventory

| document | read first-hand | what it supplies, and where |
| --- | --- | --- |
| Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using `drtmle`*, Observational Studies 9(2):43–78 | **yes** | equations (5)–(10) and both reduced-regression constructions; the package workflow; the qualitative doubly-robust claims; multi-level treatments (§4.6, pp. 66–67) and cross-validation (§4.7, p. 69) |
| Benkeser, Carone, van der Laan & Gilbert (2016), *Doubly-robust Nonparametric Inference on the Average Treatment Effect*, U.C. Berkeley Division of Biostatistics Working Paper Series, paper 356 | **yes** | §3.1's bivariate construction and its `D_A`/`D_Y` displays (p. 9); equation (2) (p. 9); §3.2's univariate `D_Y`, Theorem 1, `D^{*,#}`, `σ̂²_n` and the recursive algorithm (pp. 10–11); appendix A's bivariate remainder and rate conditions (pp. 19–20); appendix B's univariate remainder (p. 21); appendix C on unnecessary correction terms (pp. 21–22) |
| Benkeser, Carone, van der Laan & Gilbert (2017), *Doubly robust nonparametric inference on the average treatment effect*, Biometrika 104(4):863–880, PMC5793673 | **no** | the *published* Theorem 1, which is authoritative wherever the working paper and it differ |
| van der Laan (2014), *Targeted estimation of nuisance parameters to obtain valid statistical inference*, International Journal of Biostatistics 10(1):29–57, Theorem 3 | **no** | the bivariate construction's regularity conditions |
| `benkeser/drtmle` 1.1.2 source, and its own reference documentation | **yes**, read | where several formulae here were transcribed from, and the names they carry there — provenance, not a check. The `Qsteps = 2` backfitting that `fluctuation/reduced.py` and `estimators/targeting.py` quote as "found to be more stable in simulations" is that documentation's phrase, not either paper's |

**Everything the two papers are cited for is transcribed below, and neither file is kept in the
repository.** Both were, and were cited by path from here and from `methodology.md`,
`user-guide.md`, `roadmap.md`, `investigation-log.md`, `fluctuation/reduced.py` and
`test_theorem_drtmle.py`. What replaced them is a **page number on every locator**, which is the
work those paths were doing badly: a path resolves for a reader who already has the file, and a
page number resolves for one who does not. The transcriptions that closed the gap are §4's
equation (2) and §3.1 displays, §4a's appendix C, §5's appendix A remainder terms and rate
conditions, and §12's sequential-binomial construction.

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
- theorem statement available:                  YES (2016 working paper, pp. 10-11)
- corrected influence function available:       YES
- appendix A and B remainder terms available:   YES
- the sign question (item 21):                  RESOLVED, on the paper's own appendices
- published (2017) version checked against it:  NO   <-- no longer a gate; see below
```

**Reading the document rather than a transcription of it changed the answer.** The transcription
was faithful — the §3.1 display really does carry a leading minus on `D_A` — but a display is not
a derivation, and the derivation is twenty pages later. This is
[lesson 6](investigation-log.md#what-the-sizings-got-wrong) again: the cheapest instrument
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
are not interchangeable, and [item 12](../roadmap.md#closed-since-this-list-opened) is what happens
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
([lesson 2](investigation-log.md#what-the-sizings-got-wrong)).

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
  which is why [limitation 8](../roadmap.md#limitations-recorded-rather-than-fixed) — that `retarget`
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

Both observations are about a display, so here it is, transcribed from p. 9 rather than paraphrased
— the warning above is unusable without it:

```text
(2)   R(Q̄_n, Q̄, g_n, g)
        = R*_n
        + I(g = g_0)  { (P_n − P_0) D_A(Q̄_{0,r}, g) − B_{A,n}(Q̄_{n,r}, g_n) + R_{g,n} }
        + I(Q̄ = Q̄_0) { (P_n − P_0) D_Y(Q̄, g_{0,r}) − B_{Y,n}(Q̄_n, g_{n,r}) + R_{Q,n} }

with   B_{A,n}(Q̄_{n,r}, g_n) := P_n D_A(Q̄_{n,r}, g_n)
       B_{Y,n}(Q̄_n, g_{n,r}) := P_n D_Y(Q̄_n, g_{n,r})
```

and `R*_n`, `R_{g,n}`, `R_{Q,n}` second-order. Read it beside §5: the `I(g = g_0)` line is
appendix A's `D_A` derivation, whose second-order remainder that appendix names `R_{Q,n}` — which
is the crossing, visible in one place once both are on the page.

And the §3.1 bivariate `D_Y`, the other half of the internal-inconsistency argument, which §4 above
asserts the sign of and which is otherwise nowhere in this repository. Both displays, p. 9,
verbatim in orientation:

```text
D_A(Q̄_{0,r}, g)(o) := − ( Q̄_{0,r}(w) / g(w) ) · { a − g(w) }

D_Y(Q̄, g_{0,r})(o) := − ( a / g_{0,r}(w) )
                        · { (g_{0,r}(w) − g(w)) / g(w) }
                        · { y − Q̄(w) }
```

Both carry the leading minus. §3.2's redefinition — the univariate one this package implements —
does not:

```text
D_Y(Q̄, g_{1,0,r}, g_{2,0,r})(o) := ( a / g_{1,0,r}(w) ) · g_{2,0,r}(w) · { y − Q̄(w) }
```

So the two `D_Y`s differ by a sign *within the paper*, twelve lines apart, and Theorem 1 subtracts
both under one name. That is the internal inconsistency; the appendices are what break the tie.

## 4a. Appendix C: what happens to a correction that is not needed

*Source: the 2016 working paper, appendix C, pp. 21–22.* Cited in §0 as one of the things that
paper supplies, and until now transcribed nowhere — which mattered, because it is the paper's own
statement of the property `tests/unit/test_influence_gateaux*` relies on and cannot see.

The question is what `B_{A,n}` does when `Q̄ = Q̄_0`, and what `B_{Y,n}` does when `g = g_0` — that
is, what the *unnecessary* correction costs. Both answers rest on a vanishing:

```text
D_A(Q̄_{0,r}, g) = 0  for every g,   because Q̄_{0,r} = 0 when Q̄ = Q̄_0
D_Y(Q̄, g_{1,0,r}, g_{2,0,r}) = 0  for every Q̄,   because g_{2,0,r} = 0 when g = g_0
```

and each then decomposes into an empirical-process term plus a second-order one:

```text
B_{A,n}(Q̄_{n,r}, g_n) = P_0 D_A(Q̄_{n,r}, g_n) + M_{A,n},
    M_{A,n} := (P_n − P_0){ D_A(Q̄_{n,r}, g_n) − D_A(Q̄_{0,r}, g) }
    P_0 D_A(Q̄_{n,r}, g_n) = P_0[ (Q̄_{n,r}/g)·(g_0 − g) ] + R_{A,n}
    R_{A,n} := P_0[ Q̄_{n,r}·((g − g_n)/(g_n·g))·(g_0 − g_n) ] + P_0[ (Q̄_{n,r}/g)·(g − g_n) ]

B_{Y,n}(Q̄_n, g_{1,n,r}, g_{2,n,r}) = P_0 D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r}) + M_{Y,n},
    M_{Y,n} := (P_n − P_0){ D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r}) − D_Y(Q̄, g_{1,0,r}, g_{2,0,r}) }
    P_0 D_Y(…) = P_0[ (A/g_{1,0,r})·(Q̄_0 − Q̄)·g_{2,n,r} ] + R_{Y,n}
    R_{Y,n} := P_0[ A·((g_{1,0,r} − g_{1,n,r})/(g_{1,0,r}·g_{1,n,r}))·g_{2,n,r}·(Q̄_0 − Q̄_n) ]
             + P_0[ (A/g_{1,0,r})·g_{2,n,r}·(Q̄ − Q̄_n) ]
```

with `R_{A,n}` and `R_{Y,n}` `o_p(n^(−1/2))` under the rate conditions of appendices A and B, and
`M_{A,n}`, `M_{Y,n}` "reasonably assumed" to be so.

**Why this is worth having on the page.** It is the paper's own account of why solving an equation
you did not need does not cost you anything asymptotically — the guard that is not required
contributes a term that is already `o_p(n^(−1/2))` at the truth. It is also the source of the trap
CLAUDE.md records: `Q̄_r` and `g_{r,2}` are **zero row by row** at correct nuisances, so any check
taken there is blind to a flipped sign. Appendix C is where that vanishing is stated, and it is why
[`test_theorem_drtmle.py`](../../tests/unit/test_theorem_drtmle.py) has to be taken at values where
the corrections do not vanish.

## 5. The remaining remainder terms

*Source: the 2016 working paper, appendix A (bivariate) and appendix B (univariate).*

The mechanism-misspecification branch is built from three terms:

```text
R̃_{5,n} = P_0[ { (A/g_{1,0n,r})·g_{2,0n,r} − (A/g_{1,0,r})·g_{2,0,r} } · (Y − Q̄_n) ]
R̃_{6,n} = P_0[ { (A/g_{1,0,r})·g_{2,0,r}   − (A/g_{1,n,r})·g_{2,n,r} } · (Y − Q̄_n) ]
M̃_{2,n} = (P_n − P_0)[ D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r}) − D_Y(Q̄_0, g_{1,0,r}, g_{2,0,r}) ]

R_{g,n} = R̃_{5,n} + R̃_{6,n} + M̃_{2,n}
```

and the outcome-misspecification branch retains appendix A's `R_{Q,n} = R_{3,n} + R_{4,n} + M_{1,n}`,
which that appendix defines as

```text
R_{3,n} = P_0[ { (Q̄_{0n,r} − Q̄_{0,r}) / g_0 } · (g_0 − g_n) ]
R_{4,n} = P_0[ { Q̄_{0,r}/g_0 − Q̄_{n,r}/g_n }  · (g_0 − g_n) ]
M_{1,n} = (P_n − P_0)[ D_A(Q̄_{n,r}, g_n) − D_A(Q̄_{0,r}, g_0) ]
```

with `Q̄_{0n,r}(w) := E_0{ Y − Q̄(W) | g_n(W) = g_n(w), g_0(W) = g_0(w) }` — the reduced outcome
regression evaluated at the *estimated* propensity as well as the true one, which is what makes
`R_{3,n}` an approximation error rather than a fitted one.

Appendix A's first branch also carries `R^*_n = R_{1,n} + R_{2,n}`, the part that is second-order
whichever nuisance is right:

```text
R_{1,n} = P_0[ (Q̄_n − Q̄_0)·(g_n − g_0)·(g − g_n) / (g_n·g) ]
R_{2,n} = P_0[ (Q̄_n − Q̄)·(g_n − g) / g ]
```

**The paper's rate conditions are illustrative, not necessary.** For `R_{g,n} = o_p(n^(−1/2))` it
states that it *generally suffices* that

```text
‖Q̄_n − Q̄_0‖_2 = o_p(n^(−1/4))
‖g_{2,0n,r} − g_{2,0,r}‖_2 = o_p(n^(−1/4))
‖g_{2,n,r} − g_{2,0,r}‖_2 = o_p(n^(−1/4))
```

together with a `P_0`-Donsker condition on the estimated `D_Y` class and `L_2(P_0)` convergence of
the estimated `D_Y` to its limit. Appendix A's analogous conditions for `R_{Q,n}` are, in the same
"if, for example" form:

```text
‖Q̄_{0n,r} − Q̄_{n,r}‖_2 = o_p(n^(−1/4))     the reduced regression's approximation error
‖Q̄_{n,r}  − Q̄_{0,r}‖_2 = o_p(n^(−1/4))     its fitted error
‖g_n      − g_0‖_2      = o_p(n^(−1/4))     the primary propensity's error
```

which give `R_{3,n}` and `R_{4,n}`; and for `M_{1,n}`, that `D_A(Q̄_{n,r}, g_n)` falls in a
`P_0`-Donsker class with probability tending to one, together with

```text
P_0{ D_A(Q̄_{n,r}, g_n) − D_A(Q̄_{0,r}, g_0) }² = o_p(1)
```

The empirical-process pair is stated in exactly that shape in both appendices — a Donsker class
containing the *estimated* curve, and `L_2(P_0)` convergence of the estimated curve to its limit —
and appendix B's `M̃_{2,n}` takes the same form with `D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r})` in place of
`D_A`.

**Consequence for item 13**, which [A1](../roadmap.md#a1a--the-theoretical-audit) opens and
[C](../roadmap.md#c-the-demonstration) closes. The single diagnostic

```text
R_remaining = ψ̂ − ψ_0 − (P_n − P_0) D̂_DR
```

stays useful and is not sufficient: a total trending to zero can conceal cancellation between
`R_Q` and `R_g`. Where the DGP permits, [the study](validation-plan.md#5-the-controlled-study-piece-c)
must report `R_Q` and `R_g` **separately**, their component products, their signs, and the total.

**C2 computes all of this and the arithmetic above is what made the branches reachable.** Two
things fall out of writing the sums rather than the terms, and one does not.

*The branch sums lose their univariate limits.* Adding the pairs above,

```text
R_{3,n} + R_{4,n}         = P_0[ { Q̄_{0n,r}/g_0 − Q̄_{n,r}/g_n } (g_0 − g_n) ]
R̃_{5,n} + R̃_{6,n}         = P_0[ { (1_a/g_{1,0n,r})g_{2,0n,r} − (1_a/g_{1,n,r})g_{2,n,r} } (Y − Q̄_n) ]
```

— `Q̄_{0,r}`, `g_{1,0,r}` and `g_{2,0,r}` **cancel**. What is left is the *fitted* reductions,
which `DRTMLE(evaluation=…)` supplies exactly at an independent draw, and the two `0n` limits.

*A `0n` limit is a quadrature.* `Q̄_{0n,r}` is a population conditional mean of a computable
quantity given two computable scalars, so it needs no model — `benchmarks/drtmle_remainder.py`
estimates it by a binned average over the evaluation draw at two bin counts and reports their
difference as the column's own error.

*The `M` terms do not fall out.* `M_{1,n}` and `M̃_{2,n}` are `(P_n − P_0)` of a difference of
estimated curves, and a cross-fitted estimator has no single nuisance function: `P_n` reads the
out-of-fold arrays and `P_0` reads the fold-conditional ones, so no single-sample expression is
both. They are **refused by name** rather than approximated, and what is reported is each branch's
second-order half — which is what the cancellation question is about, since an empirical-process
term is `o_p(n^(−1/2))` under the conditions above and carries no product of nuisance errors to
cancel against.

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

The transcription above is confirmed against the working paper (pp. 10-11) first-hand. Two details it
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
data, which is a numerical claim and belongs to [B2](../roadmap.md#b2--the-sweep-on-the-corrected-implementation):
implement the paper's order beside this one, **both here and against the same nuisances**, and
compare the fixed point each reaches and the final three theorem-defined scores at each. The
instrument for the last of those now exists — `res.validation.correction_check()` reports each
score at the returned state, per arm — so the comparison is a run rather than a construction.
A third implementation reaching a third fixed point would have answered a different question, which
is one reason this stopped being the parity piece's.

**The second route is now written and the comparison is a dispatch.** `DRTMLE(update_order="paper")`
is the six steps above, sharing this package's stopping rule, stall test and closing pass — the
question is the route, and a comparison in which two things differ answers nothing.
`benchmarks/bench_drtmle.py --order paper` fits every draw both ways and reports `|Δψ|/se` and the
ratio of the two standard errors, paired on the draw.

**On the two draws it has been run on, the routes agree on `ψ` and *not quite* on `σ²_n`**, and the
second half of that is worth having in advance of the sweep. At `n = 600` on `nonlinear_dgp` the
`ate` estimates differ by `9e-03` of a standard error while the standard errors themselves differ
by 2.3% — `0.13231` against `0.12929` — with both fits solving all three equations at their
returned state (`1e-09` and `6e-10`), so neither is unconverged. At `n = 400` the same ratio was
`1.0006`. **This is not a contradiction of step 7 and it is what step 7 does not say**: the exit
condition constrains the three empirical *means*, and the reported variance is the second moment of
a curve built from the reductions, which the two routes leave at different vintages by construction
— measured at `sd(g_{r,2})` of `0.024` against `0.031` and `sd(g_{r,2}/g_{r,1})` of `0.058` against
`0.042` on the same draw. Whether a couple of per cent is what that gap always is, or whether it
opens up under weak overlap, is a distribution over draws and is the sweep's.
`tests/unit/test_drtmle_fit.py::TestBothUpdateOrdersReachTheTheoremsExit` pins the one-draw
statement, including the control that the two are genuinely two routes.

**Do not compare fluctuation coefficients across algorithms unless the submodels and the update
order are identical.** R tilts each arm's mechanism in its own one-column `glm` where this package
solves one two-column tilt, so its `epsilon` and this package's are different quantities already
([§12](#12-multi-valued-treatment-and-the-simplex)); adding an order difference makes a coefficient
comparison meaningless twice over. The scores are what must agree — and since the two orders being
compared are now **both run here**, the rule bites on the comparison this file actually asks for.

## 7. Truncation is not in the theorem's algorithm

**This section is A1a's answer to the question [B1b](../roadmap.md#b1b--the-theorem-conforming-targeting-decision)
waited on, and it is stated here as a finding rather than left as a reading.** The roadmap sent
B1b here "once A1 has said which mechanism the theorem's `D_g` is evaluated at". It says:

> **The theorem's `D_g` is evaluated at the same `g*` its score is solved at, and that `g*` is
> not truncated anywhere.** There is one mechanism in the theorem, produced by an unconstrained
> `expit` fluctuation, appearing identically in equation (9)'s covariate, in equation (9)'s
> residual and in `D_A`. Boundedness is an *assumption about `g_0`*, not an operation on `ĝ`.

Two consequences, and the second is the one that changes what B1b is:

1. the current hybrid — bounded denominator, raw residual — is **not** a convention the theorem
   offers, and neither is R's post-fit clip. The theorem has no third array to choose between:
   it has one;
2. so B1b was not adjudicating between two readings of the source. It chose a **finite-sample
   rendering of a step the theorem states without one**, on the criterion of which rendering
   leaves a final score that is the theorem-defined score of the estimator being declared. That is
   a design decision with a stated bar, not a transcription question, and nothing in the sources
   would have settled it — which is why no further document was owed here.

**What it chose, and how this section's own preference fared.** B1b landed the constrained
estimating equation — `clip` inside `F`, solved for a root, so the final score *is* the declared
estimator's — rather than the smooth bounded submodel the last paragraph of this section reads as
preferring. The preference is stated *against a projection applied after an unconstrained
optimisation*, which is a different candidate, and the smooth submodel was fitted before being set
aside: it is a different submodel on **every** fit rather than on the clipping ones, so at inert
bounds it moves a fit whose bound never binds. The measurements are in [the investigation
log](investigation-log.md#what-the-b1b-prototype-measured).

The reasoning follows.

The theorem assumes `g_0 > δ > 0` and defines every mechanism-side object at the **same** `g`. Its
mechanism update is

```text
g^{k+1}_n(w) = expit[ logit{g^k_n(w)} + ε_{3,n,k}·H_3(Q̄^k_{n,r}, g^k_n)(w) ]
```

with **no hard clipping after the fluctuation**. So the theorem as written supports *neither* of
the conventions this package and R respectively use:

- not the current hybrid — bounded denominator, raw residual — which is
  [item 20](investigation-log.md#item-20-from-discovery-to-cause);
- not R's post-fit clipping, as an exact theorem step.

That is a stronger statement than "the two conventions disagree", and it is what makes
[B1b](../roadmap.md#b1b--the-theorem-conforming-targeting-decision) a derivation rather than a taste
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

### The scope decision (item 25)

**A finding that a step is not in the source is not by itself a statement about which fits the
theorem covers, and this section stopped one sentence short of one.** B1b's bar was that the final
score be the theorem-defined score of the estimator *declared*, and it is met; what was never
written down is which estimator is declared. The three available answers are: derive the expansion
for the constrained mechanism, restrict the guarantee to fits where the truncation is
asymptotically inactive, or hold bound-active fits as empirically supported and outside the
theorem. **The second and third are taken**, and [the roadmap's
contract](../roadmap.md#the-supported-contract-and-item-25) is where they are stated for a reader.
Four things belong here rather than there, because they are properties of the derivation:

1. **Where nothing clips there is nothing to render.** `solve_bounded_mechanism` returns the
   unconstrained solve untouched when the tilt stays interior, so such a fit *is* the estimator
   Theorem 1 is stated for — not an approximation to it. This is the whole reason the second option
   is available at all, and it is pinned by
   `tests/unit/test_bounded_mechanism.py::TestTheFastPathIsTheOldSolver`.
2. **The asymptotic half needs three conditions and two of them are not Theorem 1's.** With
   `g_0 ∈ [δ, 1 − δ]` — which *is* Theorem 1's — a bound sequence eventually below `δ`, and `ĝ`
   consistent in **sup** norm, the clipping event has probability tending to zero and the two
   estimators coincide with probability → 1, which is all asymptotic linearity needs. The sup norm
   is stronger than the `L₂` conditions the theorem assumes, and `g_bounds="auto"` supplies the
   bound sequence (`5/(√n·log n) → 0`) while a user-set fixed bound above `ess inf g_0` does not
   supply it at all. Both are rows in [§15](#15-assumptions-and-which-the-implementation-meets)
   now rather than a paragraph here.
3. **`ĝ` is not the only truncated mechanism-side object.** Equation (10)'s covariate divides by
   `ReducedSet.bounded_gr1`, and `g_{r,1}` is a regression of an arm indicator on `Q̂` whose
   boundedness away from zero has **no counterpart in the theorem's assumption list** — `g_0 > δ`
   does not imply it, since the conditioning variable is an estimate rather than `W`. So the
   condition is *none of the three truncations active*, and the third is an assumption about an
   estimated object. This is the part no document had; the matrix's `hard truncation of ĝ` row
   named one of two.
4. ~~**The residue is a witness, not a derivation.**~~ **The witness exists**, and it changed the
   answer. It used to read: `CorrectionRow.margin` reports the targeted mechanism's distance from
   the nearer bound, nothing on a fit reports the initial mechanism's or `g_{r,1}`'s, and
   `benchmarks/bench_drtmle.py` is the only place either is computed. Piece
   [C1](../roadmap.md#what-c1-landed) put all three on the fit — `CorrectionCheck.contract`,
   `initial_clip_share`, `margin`, `gr1_margin` — and asking a fit the question rather than
   inferring it from a sweep's medians produced a finding: **one to two of six well-overlapped
   draws are bound-active**, through the exit margin alone, with the initial mechanism never
   clipping and `g_{r,1}` interior. The cause is item 4 with the nuisances swapped — equation (9)'s
   covariate is `Q_r/g*` and `Q_r` vanishes where the outcome regression is right, so its score's
   root is an `epsilon` of order 20 on the logit scale and pins rows to both bounds. [The
   measurement](investigation-log.md#what-c1s-witness-measured-on-its-first-run).

   So the condition is checkable *and* checked, and what it says is that it is **not** the ordinary
   case at the exit. Item 25's third option is therefore the operative one on a share of ordinary
   fits rather than only on weak overlap, which is a matter of how a study reports its cells rather
   than of anything being wrong: every one of those fits satisfies every identity and every score
   bound. The per-cell column is on
   [the harness](https://github.com/esbraun/cleverly-tmle/blob/main/benchmarks/drtmle_coverage.py);
   how a *mixed* cell is read is [piece C3](../roadmap.md#c-the-demonstration)'s to freeze before
   its dispatch.

Option one stays open to anyone who wants the bound-active regime inside the theorem rather than
beside it, and nothing here argues a bound-active fit is *wrong*: B1b's measurements — identities
at `1e-17`, final scores at `1e-10`, `check fails` flat zero across 96 fits including both
`weak-overlap` cells — say the opposite. They say it about an estimating equation this file's own
finding puts outside the theorem, which is exactly the distinction the third option exists to keep.

## 8. Cross-fitting is not in the sources, and the argument for it (item 15)

The theorem uses Donsker and `L_2`-convergence conditions for the empirical-process terms. **It
presents no cross-fitted version**, so the working paper does not close item 15. The 2023
article's statement that cross-validated nuisance estimates weaken the entropy conditions — made
for ordinary TMLE/AIPTW and for the doubly-robust TML estimator alike — is supporting evidence
that cross-validated DRTMLE is intended and implemented, and it is not the missing proof.

What is specifically unaddressed is this package's **pooled** construction, in which an
observation influences other rows' generated regressors and then returns to its own
reduced-regression prediction through those rows. Generic cross-fitting results do not
automatically cover that, because the conditional independence they turn on is exactly what the
generated design breaks.

```text
cross-validation claimed applicable by Benkeser & Hejazi (2023):  YES
exact pooled construction analysed anywhere in the sources:       NO
theorem or proof for that construction:                           ABSENT
argument for it, with its conditions stated:                      BELOW — A1b's
status:                                                           MET UNDER A STATED
                                                                  ENTROPY CONDITION, WITH
                                                                  ONE CONDITION UNVERIFIED
```

### 8.1 The decomposition, and which term needs what

Write `ĥ_k = Q̂_r^(−k) ∘ ĝ^(−k)` for what fold `k`'s reduced regression contributes, and `h̃_k` for
the same object built **nested** — trained on designs *and targets* from models that left fold `k`
out too. Then

```text
(P_{n,k} − P_0) ĥ_k  =  (P_{n,k} − P_0) h̃_k          [A]
                      + (P_{n,k} − P_0) Δ_k ∘ ĝ^(−k)  [B],   Δ_k = Q̂_r^(−k) − Q̃_r^(−k)
```

**[A] is the ordinary cross-fitting argument.** Conditional on the fold-`k` complement, `ĝ^(−k)` is
a fixed function and `h̃_k` is a fixed function, so the term has conditional mean zero and
conditional variance `‖h̃_k − h̄‖²/|V_k|` — `O_p(n^(−1/2)‖h̃_k − h̄‖)`, which is `o_p(n^(−1/2))` under
the `L_2` convergence Theorem 1 already assumes. This is Zheng & van der Laan (2011)'s argument
composed with the theorem rather than replacing it, exactly as
[the methodology page](../methodology.md)'s package-wide one is. Note what it needs and the
theorem supplies: a **deterministic limit**, and convergence in `L_2` of the *pushforward*
`P_0 ∘ ḡ^(−1)` — a measure that moves with `n`, which is the non-obvious part.

**[B] is the whole of item 15**, because `Δ_k` is precisely the object that depends on fold `k`. It
is not conditionally mean zero and no cross-fitting lemma reaches it. What does reach it is
asymptotic equicontinuity, which needs two things.

### 8.2 The two conditions, and the structural fact that makes the first available

**The reductions are univariate.** That is the fact the argument turns on, and it is already the
first thing `_reduced_column`'s docstring says: the regression is on one scalar however many
covariates the fit adjusted for. Composition with the fixed map `ĝ^(−k)` transports brackets
exactly — `‖u∘T − l∘T‖_{L_2(P_0)} = ‖u − l‖_{L_2(Q)}` for `Q = P_0 ∘ T^(−1)` — so the entropy
requirement falls entirely on a class of functions of **one variable**, and not at all on the
primary nuisances' complexity. *The Donsker condition cross-fitting exists to avoid is available
here, because the reduction is one-dimensional whatever the primary nuisances did.*

One subtlety decides how the condition must be phrased. `T` is fixed only *conditionally*, so `Q`
is a **random** measure, and a bound holding at `P_0` says nothing. The condition is therefore:

> **(E)** the reduction learner's fitted functions of one scalar lie, with probability tending to
> one, in a class whose bracketing entropy is bounded **uniformly in the underlying measure**, and
> whose difference class inherits that bound.

That is not a technicality dressed up: bounded variation with a non-growing bound, monotone
functions, and a fixed-dimension bounded sieve all satisfy it *for every* probability measure,
which is why those are the examples and "a fixed `P_0`-Donsker class" is not the right phrase.

> **(S)** `‖Δ_k‖_{L_2} = o_p(1)` — the reduction fit is `L_2`-continuous in the design and target
> columns it is handed.

Given (E) and (S), `(P_n − P_0)Δ_k = o_p(n^(−1/2))` and the pooled construction satisfies what
Theorem 1 asks for.

### 8.3 Which learners are inside (E), and it is better than it looks

Every hyperparameter below is a **hard-coded constant** in `learners/library.py`, and that is
load-bearing rather than incidental — a CV-selected round count would take boosting out of the
class.

| candidate | its fitted function of one scalar | inside (E)? |
| --- | --- | --- |
| `mean` | a constant | yes, trivially |
| `glm`, `glmnet` | linear in one scalar | yes — a fixed-dimension sieve |
| `gam` | a fixed-knot penalised spline | yes — knots are not chosen from the design |
| `boost` | a step function, at most `max_iter × max_leaf_nodes` jumps, total variation bounded by `max_iter × learning_rate × range` | yes — `max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=15`, `early_stopping=False` are all constants |
| `forest` | a step function with `≈ n/min_samples_leaf` pieces per tree | **no** — the class grows with `n` |
| a nearest-neighbour or saturated interpolator | `n` pieces | **no**, badly |

So `library="glm"`, `"fast"` and `"default"` are inside — a Super Learner is a convex combination
of its candidates and the convex hull of a class with entropy exponent below two is Donsker — and
**only `"rich"` steps outside, via `forest`**. It is not refused, because a caller may want it and
the estimator still computes something; it is *scoped*, in the matrix rows below.

### 8.4 What is not settled, and what measures it

**(S) is the open condition.** It splits in two and only one half is free. That
`ĝ^(−k)` and the fold-free `ĝ` converge to a common limit is implied by the limit assumption [A]
already needs, so it costs nothing. That the *fit* moves continuously with its design column does
not: for a fixed-basis linear smoother it follows from a matrix perturbation bound, and for
anything that **selects** structure from the data — a split point, a bandwidth, a neighbour, a
CV-chosen candidate — an arbitrarily small design perturbation can move the selection discretely
and leave `Δ_k` of order one on a region. **Boosting is entropy-safe and design-continuity-unsafe**,
and it is the default reduction learner whenever the primary one is boosting, since
`reduced_*_learner` falls back to it.

That is the point at which an argument stops and an instrument starts, and **`Δ_k` is a computable
array**: it is exactly the difference between `reduced_crossfit="pooled"` and `"nested"`. So the
open condition of the pooled proof is the thing the reference construction estimates, which is why
the two tracks this section used to hold open are one track — see
[the validation plan's §7](validation-plan.md#7-the-cross-fitting-construction-piece-a1b) for the
rule that reads it.

**It has been read once, and the answer is *supported, not shown*.** [The A1b
dispatch](investigation-log.md#what-the-a1b-dispatch-measured) — 216 fits over two processes and
three sizes — puts the construction difference at or below what a redrawn fold split moves in every
cell, with the two shrinking together on the process whose differences are large enough to carry a
trend. Three of the rule's four clauses pass; the primary one passes on `nonlinear` and fails on
`linear`, where the quantity it tests is already 3 to 7 times below its own control. So this row
stays `unverified` rather than moving: what the run measures is `Δ_k`'s **consequence** on `ψ`, and
a consequence can hold by cancellation. The measurement that would settle it is the paired `L₂`
distance between the two arms' reduced arrays, which is `‖Δ_k‖` itself.

**A further condition sits beside (E) and is a rate rather than an entropy bound.** `g_{r,2}`'s
target is `(1_a − ĝ)/ĝ` at the *bounded* mechanism, so its envelope is `1/lo − 1`; equation (10)'s
covariate is `g_{r,2}/g_{r,1}` with `g_{r,1}` truncated at the same bound, so that envelope is
`O(1/lo²)`. Under `g_bounds="auto"`, `lo = 5/(√n·log n) → 0` and the envelope **grows with `n`** —
so "a fixed ball with a non-growing bound" is false for `g_{r,2}` by construction under the shipped
default. It pulls against [§7's](#7-truncation-is-not-in-the-theorems-algorithm) *bound sequence
eventually below δ* row, which wants exactly that shrinkage; both are now matrix rows rather than a
paragraph here.

**And neither construction makes the *targeted* collection fold-independent.** `epsilon` is solved
on all `n` rows — `targeting_scheme="fold"` is refused by name — so a nested fit is *nested in the
nuisance models and pooled in the tilt*. That residual dependence is finite-dimensional and is
handled by the expansion [the methodology page](../methodology.md) already gives for pooled
targeting; it is a row below so that "nested" is not read as "independent".

### 8.5 What A1a settled, and the reason it gave was wrong

The comparison in `tests/unit/test_influence_gateaux_drtmle.py` is silent about this construction.
That much stood and stands. **The reason recorded here was wrong twice over**, and A1b found it by
building the thing the reason was about.

It said the module runs at *saturated* reductions, "where each conditioning cell is a singleton, so
there is no fold-borrowing left to differ about". On this law the design takes **three** values over
a thousand rows, so the cells are not singletons — `_saturated_reductions` computes a value per cell
precisely because they are not. And saturation of the *reduction* is not what decides it: under a
primary learner that learns, an inner model and an outer model disagree, so the reduced regressions
are fitted on different designs and any learner returns different arrays — a saturated one more so,
since it keys on exact design values.

What actually makes that module silent is `cross_fit=False` **and** oracle primary learners: one
fold has no complement to nest inside, and a learner that ignores its training rows returns the same
function whichever rows it saw. `tests/unit/test_nested_reductions.py` asserts that corrected
statement rather than describing it, and keeps it as a mutation watched to **pass**.

The correction matters beyond tidiness. The false reason would have licensed reading a
*cross-fitted* saturated fit as evidence about fold reuse, which is the shape of mistake
[stop-ship 14](../roadmap.md#stop-ship) exists to prevent — so the stop-ship's own wording was
carrying the error it was written to catch.

**Agreement with R would not have been evidence here** either — that package predates this
construction — which is one of several reasons the parity piece was never going to earn its keep.

## 9. What was read out of the R source, and what is still owed

This section used to be *six traps for reading the R source alongside the paper* — advice for a
parity run. There will be no parity run: it is [retired by
decision](../roadmap.md#closed-since-this-list-opened), no R enters this repository or CI, and four of
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
  absolute branch [item 7](../roadmap.md#closed-since-this-list-opened) added here is the same shape.
  Worth keeping precisely because it is the *weaker* claim: it says where the constant came from,
  not that it is right — and [piece B2](../roadmap.md#b2--the-sweep-on-the-corrected-implementation)
  is where the loop's bar stops being a proxy for the reported one at all. Note also that R's
  convergence test is defined *on the curve it reports* where this package's is defined on what the
  solver recorded, which is the difference
  [item 20](investigation-log.md#item-20-from-discovery-to-cause) lives in.
- **`D*_g` and the missing-outcome indicator, which is an open derivation question and not a
  difference.** R's `eval_Dstar_g` is `Qr/g · (1{A = a, DeltaA = 1, DeltaY = 1} − g)`;
  `reduced_corrections` applies `observed` to `D*_Q` and **not** to `D*_g`. It is not live —
  `DRTMLE` refuses `delta=`, so no fit it accepts has a missing outcome — but it is the thing to
  settle *before* that refusal is lifted, and it has to be settled from the derivation. This is the
  cleanest small example of why the parity piece was retired: every fixture would have agreed, on
  every draw, because the quantity that distinguishes the two conventions is identically absent
  from the fits either package accepts. A check that cannot fail is not a check
  ([lesson 8](investigation-log.md#what-the-sizings-got-wrong)). It is carried as a row in
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
[piece D](../roadmap.md#d-widen-the-scope-to-what-the-sources-derive) is gated on it and because
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

The 2023 article (§4.6, pp. 66–67) states that `drtmle` handles an arbitrary finite number of
discrete levels and describes a sequential-binomial construction for the *initial* propensity that
ensures `Σ_a g_n(a|w) = 1` — which the article calls the estimates being **compatible**. It works a
three-level example and builds a covariance matrix for the treatment-specific means and their
contrasts.

The construction itself, since it is short and is the thing a multi-arm implementation here would
have to match. Rather than one multinomial regression — which would put the PS outside the reach of
every binary-outcome learner the package can use, and that is the article's stated motivation — a
series of *binomial* regressions is fitted. At `A ∈ {0, 1, 2}` with no missingness:

```text
g_n(0|w)   estimates P_0(A = 0 | W = w)                  a binary regression on all rows
g̃_n(1|w)   estimates P_0(A = 1 | A > 0, W = w)           I(A = 1) on W, among rows with A > 0
g_n(1|w) = g̃_n(1|w) · { 1 − g_n(0|w) }                   by the chain rule
g_n(2|w) = 1 − g_n(0|w) − g_n(1|w)                       by subtraction
```

Compatibility is then true by construction rather than by a projection afterwards, and the pattern
generalises to any number of levels: regress each level against the ones above it, multiply by the
mass not yet allocated, and take the last level as the remainder. Every regression in it has a
binary outcome, which is the whole point.

Two things this does **not** settle, and both are why the block below still reads NO. It is the
*initial* propensity only — the article says nothing about whether the **targeted** `g*` retains
compatibility, which is the open question §12 is really about. And a compatible PS is not a
multi-arm theorem: the article reproduces none.

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

The permanent table. **`TODO` is gone from the `evidence` column and that is A1a's deliverable**,
so what a row now records is which test pins it and — where nothing here does — which piece owns
it and why. A status column with no `unverified` in it has been filled in from the code rather
than from the paper.

Two rows are still open and neither is a gap in this piece. `R_{Q,n}`/`R_{g,n}` is item 13, which
A1 *opens* and [piece C](../roadmap.md#c-the-demonstration) closes, because only that study knows
`ψ_0`. And the reduced regressions' cross-fitting is [§8](#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15),
which is [A1b](../roadmap.md#a1b--the-cross-fitting-construction). Writing "open, owned by C" where a
row used to read `TODO` is not a downgrade of the bar: `TODO` said *nobody has looked*, and these
two say *this is whose it is*.

The **R** column is *provenance*: it records where each formula in this package was read from, and
the two `(swapped)` markers are the single easiest thing here to transcribe backwards. It is not a
target. Comparing against that package's numbers is [retired by
decision](../roadmap.md#closed-since-this-list-opened) and there is no R in this repository or in CI.

The **evidence** column is what that retirement was traded for: which test pins the row *against
its derivation*. It is the checklist item 2 used to be, restated as tests to write. A row reading
`TODO` has no such test — not "probably fine"; and a column with no `TODO`s on first pass has been
filled in from optimism, exactly as §15's `unverified` column says of itself.

| theorem object | Python | R (provenance) | conditions on | sign | denominator / truncation | initial or starred | arm-specific | consumed by | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Ψ` | `counterfactual_means` | GCOMP estimate | — | + | — | starred `Q̄` | yes | the report | `test_reduction_alternation.py` (at the truth it is plain `TMLE` array for array) |
| `D*` | ordinary curve, `influence.py` | `eval_Dstar` | `A`, `W` | + | `g*`, bounded | starred | yes | eq (8), `D_DR` | the `test_influence_gateaux*` modules, for the *plain* curve |
| `Q̄_n`, `g_n` | initial cross-fitted predictions | `estimateQ`, `estimateG` | `W` | — | `g` bounded at use | initial | yes | every reduction's design | `test_influence_gateaux_drtmle.py`, whose `Misspecified` shim *declares* both and asserts one is exact in the sample and the other off by `>0.1`; `test_drtmle_fit.py::TestTheReportedNuisancesAreTheFittedOnes` for a fitted pair |
| `Q̄_{0,r}` / `Q_r` | `ReducedSet.qr` | `estimateQrn` | `A = a`, `g_n(W)` | + | — | starred in eqs | yes | eq (9), `D_A`/`D*_g` | `test_reduced_regressions.py`, against `test_remainder_drtmle.py`'s longhand |
| `g_{1,0,r}` | `ReducedSet.gr1` | `grn2` **(swapped)** | `Q̄_n(W)` | + | `bounded_gr1` | starred in eqs | yes | eq (10-uni), `D_Y` | `test_reduced_regressions.py`; the inversion trap at `test_reduced_submodel.py::test_but_gr1_does_not_vanish` |
| `g_{2,0,r}` | `ReducedSet.gr2` | `grn1` **(swapped)** | `Q̄_n(W)` | signed | fixed at fit time (limitation 9) | starred in eqs | yes | eq (10-uni), `D_Y` | as above |
| `D_A` | `D*_g` | `eval_Dstar_g` | `A`, `W` | + ([§4](#4-the-sign-discrepancy-item-21--resolved)) | the theorem's is untruncated ([§7](#7-truncation-is-not-in-the-theorems-algorithm)); the rendering is B1b's | starred | yes | `D_DR`, eq (9)'s check | `test_theorem_drtmle.py`, **at nonzero `Q_r`**; `test_influence_gateaux_drtmle.py`'s `g_right` cell, where it is the live correction |
| `D_Y` | `D*_Q` | `eval_Dstar_Q` univariate branch | `A`, `W`, `Y` | + | `g_{r,1}` bounded | starred | yes | `D_DR`, eq (10)'s check | `test_theorem_drtmle.py`; `test_influence_drtmle.py` for the longhand; `test_influence_gateaux_drtmle.py`'s `q_right` cell, where it is the live correction |
| `D^{*,#}` | `D = D* − D*_Q − D*_g` | `DnoStar − DnQoStar − DngoStar` | — | + ([§4](#4-the-sign-discrepancy-item-21--resolved)) | — | starred | rowwise per arm; ATE is the rowwise difference | the variance | `test_influence_drtmle.py` (difference not sum; per-guard membership) **and `test_influence_gateaux_drtmle.py`**, which is the Gateaux pin of the decomposition this row wanted: in each off-diagonal cell the corrected curve equals `tests/discrete_law.py`'s complex-step EIF, row for row, from a real fit as well as longhand |
| `B_n`, `B_{A,n}`, `B_{Y,n}` | the three recorded scores | `PnDnoStar` etc. | — | — | **the identity B1a pins and B1b makes hold** | starred | yes | the stopping rule and `score_check` | `test_drtmle_fit.py`, `test_bounded_mechanism.py`, and `validation/drtmle.py`'s `correction_check` |
| `R_{Q,n}`, `R_{g,n}` | `branch_products` | not computed | — | — | — | — | — | item 13 | **computed, in [piece C2](../roadmap.md#c-the-demonstration)** — a column on that study rather than a test here, since only it knows `ψ_0`. `benchmarks/drtmle_remainder.py` reports each branch's **second-order half** off `DRTMLE(evaluation=…)`, with its own binning error beside it; the `M` terms are refused by name. [§5](#5-the-remaining-remainder-terms) has the arithmetic |
| `σ̂²_n` | `influence_covariance` | `drtmle` covariance block | — | — | — | — | — | the interval | `test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s` — the interval built from the package's own corrections is the one Theorem 1's terms give, the uncentred `P_n{D}²` differs from the reported variance by exactly `(P_n D)²`, and the contrast reads the covariance rather than the sum |
| the probability limits `Q̄_1`, `g_1` | declared in tests; a fit has estimates, not limits | — | `W` | — | — | the limit the starred arrays converge to | yes | Theorem 1's conclusion, and `D^{*,#}` is evaluated **at them** | `test_influence_gateaux_drtmle.py`, the first fixture here to *have* limits: the misspecified nuisance is a constant, so it is its own limit, and the union model is entered by construction rather than by a rate |

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
| **truncation theorem** | **no** | not stated anywhere in hand | an original derivation, and it is **no longer owed**: [§7's scope decision](#the-scope-decision-item-25) restricts the guarantee to the inactive-bound regime, where the estimator is the theorem's, and puts the active one beside the theorem rather than inside it. Wanted only by someone who wants that regime covered |
| **pooled cross-fitting theorem** | **no** | general CV claim only | **supplied here** rather than found: [§8](#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15)'s argument, whose entropy half turns on the reductions being univariate and whose stability half `reduced_crossfit="nested"` measures |
| **multi-arm theorem** | **no** | software example only | the 2017 paper's multi-arm case, or a derivation |
| weights, estimated or fixed | **no** | — | item 17 closed the transport on the exact law; the theorem says nothing |
| repeated sample splitting | **no** | — | item 18 closed the arithmetic; the theorem says nothing |

## 15. Assumptions, and which the implementation meets

One row per condition. **`unverified` is a permitted answer and is the point of the column.**

The statuses below were the state on the day this file was seeded; **A1a filled the six empty
`evidence` cells and corrected one status, and the count of `unverified` rows did not fall** —
which is what the column is for. What changed is that a row now says *what would settle it* rather
than nothing at all, and one row that read **violated** now reads violated-and-measured, because
[B1a](../roadmap.md#b1a--the-identity-and-safety-patch) landed an instrument for it.

| condition | source | required for | what the implementation does | evidence | status |
| --- | --- | --- | --- | --- | --- |
| `Q̄ = Q̄_0` **or** `g = g_0` | Thm 1 | the whole conclusion | assumed, not checked | `test_influence_gateaux_drtmle.py` enters each half of the union **by construction** — one nuisance exact in the sample, the other a declared constant | met by assumption; the union model is the point, and it is now the fixture rather than a hope |
| `g_0 > δ > 0` (true mechanism) | Thm 1 | boundedness | assumed; a *fitted* `g` is truncated instead | positivity warning | **unverified** — the theorem bounds `g_0`, the code bounds `ĝ`. What the two have to do with each other is [§7's scope decision](#the-scope-decision-item-25) and not this row |
| `ĝ` consistent in **sup** norm | not Thm 1's — item 25's | the truncation being asymptotically inactive | not checked | — | **unverified**, and stronger than the `L₂` conditions the theorem assumes. It is the price of the second option: without it, "the bound stops binding" is a hope rather than a consequence |
| a bound sequence eventually below `δ` | not Thm 1's — item 25's | as above | `g_bounds="auto"` is `5/(√n·log n) → 0` and satisfies it; a user-set fixed bound above `ess inf g_0` does not | `resolve_g_bounds` | **met for `"auto"`**, and a stated restriction otherwise — a fixed bound is a choice a caller can make that puts their fit outside the guarantee, which is worth saying plainly rather than pricing into the default |
| `B_n = o_p(n^(−1/2))` | Thm 1 | eq (8) | solved to `1e-11` relative or `_NEGLIGIBLE/n` absolute | sweep | met, under a numerical proxy for `o_p` |
| `B_{A,n} = o_p(n^(−1/2))` | Thm 1 | eq (9) | solved at the **truncated** residual, which is the one the curve reads | item 20; `correction_check`'s `identity` rows, per arm, on four fixtures including one where 375 rows clip | **met**, under the stated restriction that the mechanism is the truncated one and not the theorem's untruncated `g*` — [§7](#7-truncation-is-not-in-the-theorems-algorithm) is why that is a rendering rather than a departure. It read *violated wherever the bound binds*: B1a made it visible and [B1b](../roadmap.md#b1b--the-theorem-conforming-targeting-decision) closed it, at `1e-17` on the identity and `1e-10` on the score |
| `B_{Y,n} = o_p(n^(−1/2))` | Thm 1 | eq (10) | solved exactly | tests | met |
| `R_{Q,n} = o_p(n^(−1/2))` | app. A | asymptotic linearity | measured, not shown | `test_remainder_drtmle.py` has the *arithmetic* at saturated reductions; C2 built the instrument — `√n R_remaining` off an independent draw through the fit's own nuisances, plus each branch apart — and [piece C3](../roadmap.md#c-the-demonstration)'s dispatch is what reads the rate. C3c read it and it does not vanish — flat at `1.43 / 1.26 / 1.25` in `q-drift` — **at `glm` reductions**, which is the row three below and is why that is a measurement of a configuration; [E1](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) then removed most of the instrument as a candidate explanation of the flatness, by integrating `P₀D̂` deterministically; how much of it is a measurement E1b makes rather than one E1's ladder bounded | **unverified, and now measurable** — item 13. The status does not move on an instrument landing, which is the same rule [§8](#8-cross-fitting-is-not-in-the-sources-and-the-argument-for-it-item-15)'s stability row is held to — **and it does not move on a sharper instrument either**: E1 makes the reading believable and does not make the condition met |
| `R_{g,n} = o_p(n^(−1/2))` | app. B | asymptotic linearity | measured, not shown | as above, and the two branches are reported **apart** — [§5](#5-the-remaining-remainder-terms) — with the caveat that what is reported is each branch's second-order half and the `M` terms are refused. Their own inter-bin *movement* is now a reported column at every rung of E1's ladder rather than a field recorded and read by nothing, which is what `branches settled` falling to `192/250` arrived without -- a stability diagnostic and not an error bound, so the branches' error stays unestablished until E2 | **unverified, and now measurable** — item 13 |
| Donsker / `L_2` for `D_A`, `D_Y` | app. A/B | the empirical-process terms | cross-fitting, pooled | [§8.1](#81-the-decomposition-and-which-term-needs-what) splits it: the nested term is conditionally mean zero by the ordinary argument, and what is left is `(P_n − P_0)Δ_k` | **met for term [A]**; term [B] is the four rows below, which is what item 15 became |
| **(E)** the reductions' univariate fitted class has a **measure-free** bracketing-entropy bound | not Thm 1's — A1b's | term [B]'s equicontinuity | fixed by the learner: `mean`/`glm`/`glmnet`/`gam` are bounded fixed-dimension sieves and `boost` is a fixed bounded-variation ball, since `max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=15` and `early_stopping=False` are constants in `learners/library.py` | [§8.3](#83-which-learners-are-inside-e-and-it-is-better-than-it-looks)'s table | **met for `library` in `glm`/`fast`/`default`**. This is the row the whole argument buys: the reduction is univariate, so the entropy condition falls on a one-dimensional class and *not* on the primary nuisances' complexity |
| …and `library="rich"`, a nearest-neighbour or a saturated candidate is **outside** it | as above | as above | not refused and not warned — `reduced_*_learner` falls back to the primary spec | `_forest(min_samples_leaf=10)` gives `≈ n/10` pieces on one column | **outside the guarantee by declaration** where a caller asks for it. Scope, not a defect, and the same shape as [§7's](#the-scope-decision-item-25) truncation rows |
| **(S)** `‖Δ_k‖ = o_p(1)` — the reduction fit is `L_2`-continuous in its design and target columns | not Thm 1's — A1b's | term [B]'s equicontinuity | not checked. Free for a fixed-basis smoother; **not** free for anything selecting a split, a bandwidth or a candidate from the data, which includes the default `boost` | **`Δ_k` is exactly the pooled-minus-nested difference**, so `DRTMLE(reduced_crossfit="nested")` computes it; [the A1b dispatch](investigation-log.md#what-the-a1b-dispatch-measured) puts it at or below a redrawn split's effect in all six cells, three of [§7](validation-plan.md#7-the-cross-fitting-construction-piece-a1b)'s four clauses passing and the primary one passing on one process of two | **unverified, and supported**. It is the one condition of the argument a run rather than a reading settles, and this run measures its *consequence* on `ψ` rather than `‖Δ_k‖` itself — which is why the status does not move |
| `g_{r,2}`'s envelope is `1/lo` and `lo → 0` under `g_bounds="auto"` | not Thm 1's — A1b's | equation (10)'s block | the fit-time quotient by `bounded_propensity`, and `bounded_gr1` at the same bound, so the covariate's envelope is `O(1/lo²)` | `min gr1` is `0.000` on both `weak-overlap` cells | **unverified**, and a **rate** condition rather than an entropy one. It pulls *against* the `bound sequence eventually below δ` row above: the shrinkage that makes the truncation asymptotically inactive is what makes this envelope grow |
| pooled targeting: no fold's arrays are conditionally independent of `epsilon` | **nowhere** | both constructions | `targeting_scheme="fold"` refused by name, so `epsilon` is solved on all `n` rows | [the methodology page](../methodology.md)'s finite-dimensional expansion, which this composes with rather than replaces | **not covered by the source**, and the reason a nested fit is *nested in the nuisance models and pooled in the tilt* rather than independent |
| reduced regressions consistent | Thm 1 | the corrections' limits | estimated, unmeasured rates | `test_reduced_regressions.py` shows a **saturated** learner recovers them exactly on the exact law; that is consistency at one learner on one law and not a rate | **unverified** |
| exact zeros vs `o_p(n^(−1/2))` | Thm 1 | the stopping rule | numerical criterion | item 12 | met under a stated restriction |
| arm-level means / ATE contrast | Thm 1 + adaptation | the reported parameters | rowwise difference of arm curves | `test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s` — the contrast's variance is the difference's, not the sum of the arms' | met; the adaptation is stated, not cited |
| hard truncation of `ĝ` | **nowhere** | the implementation as written | applied consistently since B1b: one array in the score and in the curve | item 20; [§7's scope decision](#the-scope-decision-item-25) | **not covered by the source**, and the guarantee is now **scoped around it** rather than assumed through it — Theorem 1 is claimed for a fit on which the bound is inactive, where the estimator is the unconstrained one bit for bit; a bound-active fit is empirically supported and outside the theorem (item 25) |
| hard truncation of `g_{r,1}` | **nowhere** | equation (10)'s covariate | `ReducedSet.bounded_gr1`, at the same `g_bounds` | `min gr1` is `0.000` on both `weak-overlap` cells and `0.117`–`0.426` elsewhere ([the sweep](investigation-log.md#where-weak-overlap-enters-now-that-it-does-not-fail)) | **not covered by the source** — and unlike `ĝ` it has no assumption to lean on: `g_0 > δ` says nothing about a regression on `Q̂`. Same scope as the row above, and it is the half item 25 added |
| the mechanism correction's sign | Thm 1 | the variance | the appendices' orientation | [§4](#4-the-sign-discrepancy-item-21--resolved), `test_theorem_drtmle.py` | **met**; the §3.1 display disagrees and its own appendices contradict it — item 21, closed |
| the update order | Thm 1's algorithm | nothing, if the fixed point is the same | different order | [§6](#6-the-recursive-algorithm-item-22) | **met under a stated restriction**: the paper's step 7 states its own exit as the three scores, so the order is not prescriptive; whether the fixed points coincide numerically is B2's, and both orders are now **runnable** here (`update_order=`), agreeing on `ψ` and differing by 2.3% in `se` on the one draw compared so far — item 22 |
| fixed weights | **nowhere** | item 17's claim | weighted loss throughout | `test_remainder_drtmle.py` runs the whole expansion at two tilted laws, with the wrong transport kept as a control that fails | met for a **fixed** weight, by a **transport argument** and not merely by arithmetic: the reductions are `P_w`-conditional expectations because they are fitted by weighted loss, and the mechanism they condition on and divide by is the `P_w` one. Inside [the contract](../roadmap.md#the-supported-contract-and-item-25); estimated weights are refused |
| repeated sample splitting | **nowhere** | item 18's claim | mean over draws | `test_drtmle_fit.py` | met, and it needs no source and no DRTMLE-specific derivation: every row is out of fold in every draw, so each draw is asymptotically linear with the same `D` and `mean_r ψ_r` is asymptotically linear with `mean_r IC_r`. That is the package-wide argument composed with Theorem 1; inside [the contract](../roadmap.md#the-supported-contract-and-item-25) |
| `K` arms | **nowhere** | piece D | binary only, refused by name | `reduced_mechanism_covariate` raises above two arms rather than generalising the tilt | **not covered by the source** — [§12](#12-multi-valued-treatment-and-the-simplex) |
| missing outcomes | **nowhere**; `drtmle` masks `D*_g` and this package does not | a lifted `delta=` | refused, so the two conventions never differ on a fit either package accepts | [§9](#9-what-was-read-out-of-the-r-source-and-what-is-still-owed) | **not covered by the source** — settle from the derivation *before* lifting the refusal; no run could ever have settled it |
| composition with `CTMLE` | **nowhere** | — | refused | `test_drtmle_fit.py::TestTheRefusals` | **not covered by the source** |

**Six rows read *not covered by the source*** — the truncation of `ĝ`, the truncation of
`g_{r,1}`, pooled targeting's `epsilon`, `K` arms, missing outcomes and composition with
`CTMLE` — and eight read `unverified`. **A1b split the cross-fitting row into six and the count of
`unverified` rows went up rather than down**, which is what this column is for and is the same thing
A1a's revision recorded: the old single row said *nobody has an argument*, and the new ones say what
the argument is, which of its conditions the learner settles, and which one a run has to. Two of
them read **met under a stated restriction** — the first time this cell has moved on item 15.
The two truncation rows were one row until item 25, and splitting them is the substance rather than
bookkeeping: `ĝ`'s truncation has an assumption in the theorem to be scoped against (`g_0 > δ`) and
`g_{r,1}`'s has none.

**A row reading *not covered by the source* is not the same as a row outside the guarantee**, and
conflating the two is what item 25 corrected. `K` arms, missing outcomes and `CTMLE` are refused,
so no fit reaches them. The truncation rows are reached by every fit, which is why they needed a
scope decision rather than a refusal — and why the two weight and split rows, which are *also*
covered nowhere in the sources, are inside the contract on arguments written down here rather than
outside it on the sources' silence.
