# Marginal structural model projections

## What this solves

You have four arms, or six regimens, or a dose you want to read as a trend. Reporting every
counterfactual mean separately is honest and unreadable. You want one summary: a slope in the dose,
or an effect that varies with one baseline characteristic.

A marginal structural model in `cleverly` is a **projection**. It is the best approximation of the
true counterfactual surface within a working model you declared. It is not a claim that the working
model is the true causal response surface, and the estimand is well defined whether or not the
working model fits.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| many arms, or many regimens | one coefficient vector instead of one mean per level | the coefficients mean what the working model says they mean. Read them as a projection |
| a dose you want to summarise as a trend | a slope, with an influence curve and an interval | a model linear in the arm reads the arm as a dose, so non-numeric labels are refused |
| effect modification by a baseline variable | an interaction term in the working design | the design must be full rank on the *realized* cells |
| repeated treatment over time | the same projection over regimen and horizon cells | the node fluctuations are pooled, and under a link one round of the alternation is a whole backward pass |

The projection is a fourth parameter axis. `msm=` cannot be combined with `interventions=` or
`shifts=`, because one fluctuation solves one set of score equations, and a fit reporting
parameters from two axes would put two of them under one heading.

A worked applied analysis is in the
[MSM projections tutorial](../examples/msm-projections.md). It checks the saturated case
against the per-arm report.

## The algorithm as implemented

For design vector $\phi(a,V)$, nonnegative projection weight $h(a,V)$, and the identity link,

$$
\beta(P)=\arg\min_b E_P\left[\sum_a h(a,V)\{\bar Q_P(a,W)-\phi(a,V)^\top b\}^2\right].
$$

With $M=E\sum_a h\phi\phi^\top$ and $r=E\sum_a h\phi\,\bar Q_P(a,W)$, the coefficient is
$\beta=M^{-1}r$. The working design and the weights are known functions, and $M$ must be full rank.

With the identity link the clever covariate is $h(a,V)\,\phi(a,V)/g(a\mid W)$, one column per term,
so the score equation is one per coefficient rather than one per arm. The counterfactuals are still
the arms. What changed is the summary. A **saturated** working model, with one indicator per arm,
reproduces the per-arm report exactly, at the point estimate and at the influence curve.

Theory: Neugebauer and van der Laan (2007), Rosenblum and van der Laan (2010), and chapter 12 of
van der Laan and Rose (2011). Petersen et al. (2014) gives the longitudinal construction. See the
[reference list](../references.md#longitudinal-survival-and-marginal-structural-models).
Implementation:
[`msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/msm.py),
[`longitudinal/msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/msm.py),
and
[`fluctuation/submodel.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/submodel.py).

### Under a link

Three things change inside, and each is a place where the obvious generalisation is wrong.

**The clever covariate reads the coefficient.** It becomes
$h(a,V)\,(dm/d\eta)\,\phi(a,V)/g(a\mid W)$, and $dm/d\eta$ is a function of $m(a,V;\beta)$. So the
fluctuation and the projection are solved *together*, alternating until both settle. It converges
fast, because the coefficient reaches the covariate only through that smooth factor. The shift falls
by a factor of $10^{-3}$ to $10^{-4}$ per round, and `res.fluctuations["msm"].projection` carries
the per-round trace.

**The matrix the influence curve is premultiplied by is no longer the Gram matrix.** It is

$$
M = E\left[\sum_a h\left\{(dm/d\eta)^2 - (\bar Q_P-m)\,d^2m/d\eta^2\right\}\phi\phi^\top\right],
$$

and the second term vanishes only where the working model *fits*, which is exactly what a
projection does not promise. Dropping it is wrong in a way that no saturated-model check can see.

**The remainder is second order without being zero.** With the identity link a correct mechanism
drives the remainder to exactly zero, and that exactness is the *linearity* of the estimating
equation in the coefficient rather than a stronger form of double robustness. Under a link what is
left is quadratic in the coefficient error. The other half is untouched: a correct outcome
regression still gives exactly zero, under every link.

A saturated working model still reproduces the per-arm report through the link. The inverse link of
the coefficient is the counterfactual mean to machine precision, with the influence curves related
by the delta method. That is what says a link is a reparameterisation of the same counterfactual
means rather than a second estimator.

### The longitudinal projection

The longitudinal MSM projects regimen, horizon, and cause-specific means onto a pooled working
model. Three things differ, and each is again a place where the obvious generalisation is wrong.

**The node fluctuation is pooled across the regimens.** At one node the design's columns get their
rank by summing over the arms *within a row*, because a unit contributes its design at the arm it
received. A regimen is a plan and not a value some unit took, so there is nothing to sum over within
a row. Whenever the working model has no effect modifier, the design is constant down the rows,
which makes a per-regimen design rank one and collapses its score equations into one. Each node
therefore solves a **single** fluctuation over the regimens stacked, with one shared coefficient.
The backward recursion is lockstep: outer over the nodes, inner over the regimens, one update,
all carried forward together.

**A saturated working model reproduces the per-regimen report at `n_folds=1`**, and not bit for bit
on the estimate. One indicator per regimen makes the stacked design exactly block diagonal, and each
block carries the loss weight the plain recursion uses. The pooled Newton convergence test and line
search are taken over all the stacked rows, so the two can stop on different iterates. On a law the
sample realises exactly, no step is taken at all and the agreement is exact. Elsewhere it is `1e-11`.

**Above one fold the identity does not hold, because the two paths run different constructions.**
Both fit nuisances out of fold. The regimen-mean path then runs one complete backward recursion per
outer fold and stitches held-out rows. This path targets pooled over the whole sample, because a
fold-specific pooled-regimen recursion is not implemented.

Both paths estimate the same parameter. Across five seeds at $n = 1500$ the largest disagreement was
0.69 standard errors. `TestTheTwoCrossFittedConstructionsAreNotTheSameArithmetic` in
[`tests/e2e/test_ltmle_msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/e2e/test_ltmle_msm.py)
measures it. One difference is worth stating plainly: this path solves its pooled score equation and
the regimen-mean path does not. See
[longitudinal TMLE](longitudinal-tmle.md#cross-fitting-runs-two-constructions).

**Under a link, one round of the alternation is a whole backward pass.** The coefficient enters the
covariate through the derivative of the inverse link, so each targeted regression moves with it. A
targeted regression is the previous node's regression *target*, so every earlier node's learner is
refit. There is no fixed initial state to restart from, and the fixed point is stated over the whole
pass. It costs four or five passes in practice. The mechanism is free of the coefficient and is
fitted once.

**The projection weight and the observation weight are different objects and must stay so.** The
first says how the regimens are traded off inside the projection. The second tilts the *population*
the projection is taken over. Both multiply the node's loss weight, and only the observation weight
multiplies the finished influence curve row by row. The projection is solved on the **raw** outcome
scale, because a coefficient vector has no single scale to map back with.

## Variations

| option | what it does |
| --- | --- |
| `MSM(...)` design | the working design vector, including effect modifiers and interactions |
| projection weights | how the arms or regimens are traded off inside the projection |
| `link="identity"` | the clever covariate is free of the coefficient, and a correct mechanism drives the remainder to exactly zero |
| `link="log"`, `link="logit"` | the covariate reads the coefficient, so the fluctuation and the projection alternate. `res.coefficients(scale="ratio")` exponentiates them |
| `MSM.linear` | a model linear in the arm. **Refused on non-numeric labels**, because it would read the sort order as a dose scale nobody chose |
| longitudinal `msm=` | the same projection over regimen, horizon, and cause cells, with rank checked on the actual realized design |
| `targeting_scheme="fold"` | each fold solves its own coefficient, since the coefficient is something the covariate reads. This removes coupling *between* folds, and the rows inside a fold still fit both the coefficient and the fluctuation used for that fold. The pooled score is exactly zero because each fold's is zero at its own coefficient. This is a package extension and not the common-update CV-TMLE of Zheng and van der Laan |

One thing is **refused rather than approximated**, in the sense
[scope and refusals](scope-and-refusals.md#how-to-read-a-refusal) sets out.

| refused | kind | what it would need |
| --- | --- | --- |
| weights derived from the estimated mechanism, a "stabilised" MSM | wrong by construction | the weight would be a functional of $P$, so the influence curve carries a further term for the pathwise derivative through the estimated mechanism. This is the same argument that gives an incremental intervention its own axis. Supplying such weights anyway does not fail. It reports a standard error that is too small |

A one-shot non-identity-link fit is also refused. The derivative of the inverse link depends on the
coefficient, so a single pass would report a standard error for an equation it did not solve. The
link is supported. What is refused is skipping the alternation it needs.

## Validation issues special to this method

**A saturated oracle proves nothing about a projection.** A saturated working model *fits*, which
is exactly what a projection does not promise, so it agrees with the per-arm means whatever the
projection code does. The oracle's working model is therefore deliberately **not** saturated: three
coefficients against six covariate-and-arm cells at one node, and three coefficients against twelve
regimen cells longitudinally. Both choices are asserted on the law itself, so they are shown to be
load-bearing rather than claimed to be.

**Uniform weights are equally disqualifying.** With a constant weight the design is orthogonal and
the coefficient collapses to the marginal ATE *identically*, so code that reported the ATE under
the name of an MSM coefficient would pass every check. The oracle's weights are deliberately not
uniform.

**A comparison is not analytic.** Under a link the oracle solves its own normal equations by a
*fixed* number of Newton steps with no convergence test, because a functional that branched on a
convergence test could not be differentiated by a complex step at all. A separate check requires
that doubling the step count moves neither the value nor the curve.

**Coefficient parity with R `ltmleMSM` is not an acceptance check.** That package uses a
quasibinomial working-model projection. `cleverly` declares an outcome-scale weighted least-squares
projection. Raw coefficient parity would compare different estimands.

**Seven mutations were applied to the longitudinal projection and the tests watched.** Three passed
on the first try and exposed real gaps. That is why deliberate mutation is part of the development
guidance rather than an optional extra.

| where to read the evidence | what is there |
| --- | --- |
| [the `msm` row](evidence.md#the-table) | the Gateaux comparison, the nonzero remainder, the saturated reduction, the link-specific submodel tests, and the continuous-dose density-ratio scores |
| [working model over regimen and horizon cells](evidence.md#longitudinal-estimands-outside-the-target-registry) | the non-saturated, nonuniform projection law, the exact pooled-design and loss-weight checks, and the rank-refusal tests |

There is no registered repeated-sampling study for this axis. The continuous-dose test uses a
linear truth, and a nonlinear continuous-dose Gateaux oracle remains absent.
