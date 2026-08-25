# DR-TMLE

This entry says what applied problem the variant solves, how the algorithm is built, and what is
special about validating it. The pages beneath it carry the production contract: the
[supported estimands and refusals](supported-estimands.md), the
[theorem](theorem.md), the [targeting and cross-fitting choices](targeting.md), the
[nuisance conditions](nuisance-conditions.md) the interval is conditional on, the
[diagnostics](diagnostics.md) to inspect, and
[what the validation programme established](validation-programme.md).

## The release claim, in one paragraph

**Conditional validity.** The default univariate algorithm computes what Benkeser, Carone, van der
Laan & Gilbert's Theorem 1 derives. `reduction="bivariate"` computes van der Laan (2014), Theorem
3's earlier binary construction. Checks compare both constructions with their remainder derivations
and with the parameter's Gateaux derivative. The exact finite-support laws and the remainder
identities run at three arms as well as two, in the union-model cells where exactly one correction
survives.

The reported interval is valid **conditional on** the practitioner obtaining adequate primary and
reduced-regression fits. Those are rate conditions on estimated functions. They are not verifiable
from a fit's own output, and **numerical score convergence does not verify them**. Read
[solved scores do not establish nuisance consistency](diagnostics.md#solved-scores-do-not-establish-nuisance-consistency),
which is the single most important page in this section.

What this is *not*: a better point estimate. The three empirical means are all driven to zero, so
the extra terms cannot move `Ψ̂` and only move its variance. Read a `DRTMLE` fit as the same
estimate with an interval entitled to be believed under weaker conditions.

And it is *not* the efficient estimator. Under misspecification the canonical gradient at `P_0` is
still `D*`. What the three equations leave is `D = D* − D*_Q − D*_g`, the estimator's asymptotic
influence function at the nuisance limits, and it is generally not efficient there. When both
nuisances are consistent, the corrections converge to zero and the curve approaches the ordinary
efficient curve. At the true nuisance functions, the corrections vanish row by row.

## What this solves

Ordinary TMLE is **doubly robust for consistency and singly robust for inference**, and that
distinction is the whole of what this variant is for.

Without outcome missingness, the treatment-specific mean under a binary treatment has the exact
signed remainder

$$
R_{2,a}=\int \frac{\hat g_a-g_{0,a}}{\hat g_a}
(\hat Q_a-Q_{0,a})\,dP_0.
$$

Here $g_a(W)=P(A=a\mid W)$, and the norms below are $L^2(P_0)$. The ATE remainder is the difference
between the two arm remainders. With outcome missingness, the complete treatment-and-observation
mechanism replaces $g_a$. If $\hat g_a\geq\epsilon>0$, Cauchy-Schwarz gives the separate bound

$$
|R_{2,a}|\leq\epsilon^{-1}
\lVert\hat g_a-g_{0,a}\rVert_2\lVert\hat Q_a-Q_{0,a}\rVert_2.
$$

That bound is what double robustness rests on. One error converges, the other stays bounded, and
the product goes to zero.

The point estimate needs $R_2 \to 0$. Influence-curve inference needs the stronger condition
$\sqrt{n}R_2 \to 0$. Nuisance errors of order $o(n^{-1/4})$ satisfy it, and errors of exactly
$n^{-1/4}$ leave $\sqrt{n}R_2$ bounded away from zero. If one error does not shrink, the remainder
is first order in the other. The estimator's bias then dominates the $n^{-1/2}$ standard-error
scale, and coverage decays as the sample grows.

DR-TMLE solves two further score equations, built from reduced-dimension regressions, so that valid
inference can survive one inconsistent primary nuisance.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| you doubt one nuisance and still want an interval | an interval entitled to be believed under weaker conditions | the reduced regressions are refitted inside the alternation, so a fit costs several rounds of several regressions |
| both nuisances are consistent | nothing you need. The corrections converge to zero and the curve converges to the efficient curve | the extra cost, for no gain. This is the case the variant is not for |
| you want a narrower interval | this is not that. The union model it stays valid over is larger than the model it is efficient in | the corrected curve is the estimator's own influence function and not the efficient one |
| you want `retarget` to be cheap | it is not. A truncation curve on a `DRTMLE` fit costs about a fit per point | a plain `TMLE` handed these nuisances refuses rather than re-solving against arrays it cannot refresh |

`DRTMLE` ships under **conditional validity**. Read
[the release claim](#the-release-claim-in-one-paragraph) before you rely on it.

A worked applied analysis is in the [DR-TMLE tutorial](../../examples/dr-tmle.md). It shows why
one fit cannot display what this variant buys.

An inconsistent nuisance model is not unmeasured confounding. This variant still requires the
recorded history to satisfy exchangeability.

## The algorithm as implemented

Writing $1_a$ for $\mathbb{1}\{A=a\}$, the three reduced regressions are

$$
\begin{aligned}
Q_r(a, w) &= E[\,Y - \hat Q(a, W) \mid A = a,\ \hat g(a\mid W) = \hat g(a\mid w)\,] \\
g_{r1}(a \mid w) &= P(\,A = a \mid \hat Q(a, W) = \hat Q(a, w)\,) \\
g_{r2}(a \mid w) &= E\!\left[\left.\frac{1_a - \hat g(a\mid W)}{\hat g(a\mid W)}
                    \;\right|\; \hat Q(a, W) = \hat Q(a, w)\right].
\end{aligned}
$$

On the default construction each one is **univariate however many covariates the fit adjusted
for**. That fixed dimension is the whole argument: the reduced regressions can be estimated fast
enough whether or not the primary nuisances can.

The three equations, in the software paper's numbering, are

| | equation | what it fluctuates |
| --- | --- | --- |
| (8) | $P_n[\,1_a / g^*(a\mid W) \cdot (Y - Q^*(a, W))\,] = 0$ | the outcome regression, along the ordinary covariate |
| (9) | $P_n[\,Q_r(a, W) / g^*(a\mid W) \cdot (1_a - g^*(a\mid W))\,] = 0$ | **the mechanism** |
| (10) | $P_n[\,1_a \cdot g_{r2}(a\mid W) / g_{r1}(a\mid W) \cdot (Y - Q^*(a, W))\,] = 0$ | the outcome regression, along a second covariate |

The reported influence curve is $D = D^* - D^*_Q - D^*_g$, with the two corrections being the
left-hand sides above, row by row. There is **one term per equation the guard asked for**, so a
single-guard fit reports one of the two and a shorter curve. All three empirical means are zero
after targeting, so the subtraction cannot move the point estimate. It moves only the variance.

**`guard=` is crossed, and the crossing runs through to the curve.** `guard="Q"` guards against a
misspecified *outcome regression* and adds equation (9), which fluctuates the mechanism.
`guard="g"` guards against a misspecified *mechanism* and adds equation (10), which fluctuates the
outcome regression. The keyword names the nuisance you are worried about, not the one that the
equation it adds moves. An empty guard fits no reduced regression at all and is bit for bit a plain
TMLE.

**The three equations are solved at the arrays the curve is built from.** That is not automatic.
The alternation on its own leaves both extra equations solved somewhere the curve is not built
from, which is harmless on a converged fit and worth percents of a standard error on one that stops
early. A closing pass re-solves all three at the reductions the fit reports, refitting nothing.

**One guard removes the whole first-order remainder; two over-correct.** Each extra equation
subtracts a projection of $R_2$. Where both projections are onto all of $\sigma(W)$, either one
recovers the whole of it, so the pair leaves exactly $-R_2$. That is arithmetic on a finite-support
law rather than a defect. Asymptotically at most one of the two errors fails to vanish, so at most
one projection is non-negligible, which is why both are solved by default.

Implementation:
[`estimators/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/drtmle.py),
[`estimators/reduced.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/reduced.py),
and
[`validation/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/drtmle.py).
The source-to-equation map is in [the contract](theorem.md#the-objects).

## Variations

| option | what it does | is it a different estimator? |
| --- | --- | --- |
| `guard=("Q", "g")` | which extra equations are posed. The default is both | **yes**. An empty guard is a plain TMLE, and each single guard is its own construction |
| `reduction="univariate"` | the three regressions above, from Benkeser et al. (2017). The default, because its reduced regressions can achieve faster rates | |
| `reduction="bivariate"` | van der Laan (2014), Theorem 3: one two-column probability $P(A=a \mid \hat Q(a,W), \hat g(a\mid W))$, and equation (10)'s covariate replaced by $1_a(g_r - g)/(g\,g_r)$ | **yes**. `gr2` is `NaN` on this path by design, so accidental use of the absent regression cannot silently return zero |
| `update_order="drtmle"` | the canonical R sequence. The default. The round refits only the mechanism reductions after equation (9), and only the outcome reduction after equation (8) | no. A diagnostic keyword. Both source-specific solve orders are pinned |
| `update_order="benkeser"` | the published six-step recursion | no |
| `reduced_crossfit="pooled"` | out-of-fold reduced fits sharing the primary split. The default | no. A diagnostic keyword. Refused below three folds, under `cross_fit=False`, and with `algorithm="one_step"` |
| `reduced_crossfit="nested"` | measures the generated-regressor dependence rather than assuming it away | no |
| `randomized=`, `treatment_probabilities=` | the Diaz and van der Laan (2017) randomized missing-outcome surface: five reductions, and three separate corrections for treatment, observation, and outcome | **yes**. It requires `cross_fit=False` and both guards. See [the contract](theorem.md#randomized-trials-with-missing-outcomes) |
| `evaluation=` | an independent-draw evaluation set carried through targeting | no. A remainder diagnostic |
| multiple treatment levels | each reduction and correction is indexed by a free level, and equation (9) is solved by independent one-versus-rest fluctuations | this follows the published R workflow. The cited theorem is binary, so this is an implementation-backed armwise extension |

The targeted margins deliberately do not renormalise, because a simplex projection would reopen the
armwise equations. They are not inert either. Equation (8) divides by them, so the estimate does
read a mechanism that no longer sums to one.

Refusals by name are listed in [the contract](supported-estimands.md#refused-by-name) and summarised in
[scope and refusals](../scope-and-refusals.md#not-written-yet).

## Validation issues special to this method

**The exact-law instrument is blind here, and derivably so.** Under a law the sample realises
exactly with a saturated learner, both nuisances are exact, so $Q_r$ and $g_{r2}$ have identically
zero targets and vanish *row by row*. Both extra coefficients are then zero, and the estimator
reproduces ordinary TMLE. The Gateaux modules therefore supply a degeneracy check, and they would
pass against a wrong sign, an omitted term, or a wrong $g_{r1}$. That last one is a probability. It
does not vanish, and it sits in a denominator whose numerator does.

This is the reason the only **theorem** instrument in the tree belongs to this variant.
`tests/unit/test_theorem_drtmle.py` checks against Benkeser et al.'s Theorem 1 *at values where the
correction does not vanish*. The rest of the estimand catalogue does not need one, because its
influence curves do not vanish at the truth. The argument for that empty column is written down in
[the evidence manifest](../evidence.md#what-this-table-says-is-missing).

**Four nonzero instruments carry the multi-arm extension**, because the exact multi-arm law makes
every new term vanish and so cannot fail. They are an independent `brentq` solve of the canonical
package's own mechanism score equation arm by arm, a misspecified fit where $Q_r$ is nonzero and
the mechanism visibly leaves the simplex, a column-permutation witness on the exit state, and an
armwise covariate formula check on a nonzero $Q_r$. They are tabulated in
[estimator variants over registered targets](../evidence.md#estimator-variants-over-registered-targets).

**Bounding the two mechanisms separately is what the scope label had to learn.** The contract once
measured its truncation witnesses on the treatment mechanism alone, which is blind in exactly the
regime this construction is for. A randomized trial's mechanism is flat by design and cannot clip,
so a fit whose observation mechanism was pinned on a fifth of its rows was certified as
theorem-backed. The fixture is now a pair of fits, and the second is asserted to leave every
pre-existing column inactive, so a bound-active verdict there can come only from the two new
witnesses.

**Numerical agreement with the canonical R package is not acceptance evidence, and never will be.**
The influence curve's *provenance* is that its form was read off that package. Its *evidence* is
that it has since been checked against Theorem 1's own appendices at a nonzero $Q_r$, and against a
perturbation of the law where in each half of the union model the corrected curve is the efficient
influence function row for row. Agreement between two transcriptions of one source is evidence
about the transcription. This is a
[standing decision](../../architecture-invariants.md#validation-and-evidence) rather than a gap.

**Solved scores do not establish nuisance consistency.** `score_check` used to sign a doubly-robust
fit off with one verdict over three rows, two of which are the corrections. It now branches on
whether the fit is corrected, and says which equations it solved. See
[the contract](diagnostics.md#solved-scores-do-not-establish-nuisance-consistency).

The registered [canonical DR-TMLE study](../method-evidence/canonical-dr-tmle.md) reports rather than
gates: it publishes failed cells instead of hiding them.
[What the validation programme established](validation-programme.md)
is the full list of what is and is not settled.

```{toctree}
:maxdepth: 2

supported-estimands
theorem
targeting
nuisance-conditions
diagnostics
validation-programme
```
