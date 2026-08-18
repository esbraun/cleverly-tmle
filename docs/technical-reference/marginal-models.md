# Marginal structural model projections

## Parameter

An MSM in `cleverly` is a projection, not a claim that the working model is the true causal response
surface. For design vector $\phi(a,V)$, nonnegative projection weight $h(a,V)$, and identity link,

$$
\beta(P)=\arg\min_b E_P\left[\sum_a h(a,V)
    \{\psi_P(a,W)-\phi(a,V)^\top b\}^2\right].
$$

With $M=E\sum_a h\phi\phi^\top$ and $r=E\sum_a h\phi\psi_P(a,W)$, the coefficient is
$\beta=M^{-1}r$. The working design and weights are known functions and $M$ must be full rank.
A link-scale model adds the derivative of the inverse link to the targeted score and influence
curve.

Theory is given by Neugebauer & van der Laan (2007), Rosenblum & van der Laan (2010), and chapter
12 of van der Laan & Rose (2011). The [reference list](../references.md#longitudinal-survival-and-marginal-structural-models)
also identifies the longitudinal construction of Petersen et al. (2014).

## Point-treatment implementation

`MSMProjection` combines an `MSM` working design with the relevant arm, regime, shift, or other
supported counterfactual mean surface. The targeter solves the projection score; the result reports
one structured key per term.

Implementation: [`msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/msm.py),
[`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py),
and [`fluctuation/submodel.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/submodel.py).
The pinned `tmle3` `Param_MSM` source is implementation provenance for parameter indexing.

The [`msm` evidence row](../evidence.md#the-table) includes a Gateaux comparison, a nonzero
remainder, a saturated-model reduction to arm means, link-specific submodel tests, and continuous-
dose density-ratio scores. The saturated identity alone is insufficient because it cannot detect a
link-specific mistake in a genuinely nonsaturated projection.

## Longitudinal implementation

The longitudinal MSM projects regimen/horizon/cause-specific means onto a pooled working model.
Targeting uses the declared cell weights and a shared coefficient vector across sequential
regressions. Rank is checked on the actual regimen-cell design.

Implementation: [`longitudinal/msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/msm.py)
and [`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py).
Evidence includes a non-saturated finite law, the longitudinal Gateaux comparison, exact pooled-
design and loss-weight checks, and rank-refusal tests. R `ltmleMSM` uses a quasibinomial working-
model projection; raw coefficient parity would therefore compare different estimands and is not an
acceptance check.
