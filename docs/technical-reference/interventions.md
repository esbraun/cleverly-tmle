# Stochastic interventions

## Known regimes

For a known stochastic intervention $g^*(a\mid w)$,

$$
\psi_{g^*}(P)=E_P\left\{\sum_a Q_P(a,W)g^*(a\mid W)\right\}.
$$

Its efficient influence function has residual weight $g^*(A\mid W)/g_P(A\mid W)$ and a
centered plug-in term. Deterministic static and dynamic rules are degenerate $g^*$ distributions.
Identification requires consistency, conditional exchangeability, and positivity only where the
regime assigns mass. Because a known rule does not depend on $P$, its influence function has no
pathwise-derivative term for estimating the rule.

`Static`, `Rule`, and `Stochastic` implement the intervention protocol;
`RegimeMean` and `RegimeContrast` define levels and contrasts. See Robins (2004), Díaz Muñoz &
van der Laan (2012), and Díaz & van der Laan (2013). The pinned `tmle3` `LF_static` and
`Param_TSM` source confirms static-intervention indexing, while local exact-law and Gateaux checks
establish the parameter.

Implementation: [`interventions/base.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/base.py),
[`interventions/support.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/support.py),
and [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py).
Evidence: [`ey_regime` and `ate_regime`](../evidence.md#the-table), including nonzero dynamic-rule
and regime-support witnesses.

## Modified treatment policies

For a continuous exposure and an invertible shift $d(a,w)$, a modified treatment policy targets

$$
\psi_d(P)=E_P\{Q_P(d(A,W),W)\}.
$$

The residual clever covariate is a conditional-density ratio. For a simple additive shift by
$\delta$ away from a boundary it has the form

$$
H_d(A,W)=\frac{g_P(A-\delta\mid W)}{g_P(A\mid W)},
$$

with the inverse-map and Jacobian terms used for the declared policy. Identification requires the
shifted dose to remain inside observed conditional support. A fixed `cap=` is part of the policy;
estimating the cap from the same data would define a different pathwise-dependent intervention.

The implementation fits a conditional density, targets the outcome regression as a function of
dose, and evaluates it at $d(A,W)$. Missingness and intermediate mechanisms multiply the density
ratio when those roles are declared. The estimator is doubly robust in the outcome regression and
the complete density/mechanism product.

Theory: Díaz Muñoz & van der Laan (2012), Haneuse & Rotnitzky (2013), and Díaz, Williams,
Hoffman & Schenck (2023). Implementation:
[`interventions/shift.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/shift.py),
[`learners/density.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/density.py),
and [`fluctuation/submodel.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/submodel.py).
Evidence: [`ey_shift` and `ate_shift`](../evidence.md#the-table), continuous-density Gateaux tests,
shift submodel tests, and a nonzero remainder for the CDE/missingness composition.

## Incremental propensity-score interventions

For binary treatment, an incremental intervention multiplies the observed odds by $\delta$:

$$
q_\delta(1\mid W)=\frac{\delta g_P(1\mid W)}
                         {\delta g_P(1\mid W)+1-g_P(1\mid W)}.
$$

The denominator keeps the clever covariate bounded even when the observed propensity approaches
zero, so the parameter does not require conventional treatment positivity. The intervention is,
however, a functional of $P$. Its efficient influence function includes the derivative through
$g_P$, and the implementation targets the treatment mechanism as well as the outcome regression.

This distinction controls the robustness claim: consistency of $g$ is required because $g$ defines
the target. The second-order remainder vanishes when the mechanism is correct regardless of $Q$,
but a correct $Q$ cannot rescue an inconsistent mechanism. Kennedy (2019) is the primary theory
reference.

Implementation: [`interventions/incremental.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/incremental.py),
[`fluctuation/mechanism.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/mechanism.py),
and [`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py).
Evidence: [`ey_ipsi` and `ate_ipsi`](../evidence.md#the-table), mechanism-score identities,
nonzero Gateaux terms, and remainder controls that fail under a falsely two-sided robustness claim.

## Support reports

Arm positivity, regime support, shift support, and incremental support answer different questions.
The intervention classes therefore expose distinct reports rather than reducing all overlap to a
single propensity histogram.
