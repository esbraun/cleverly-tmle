# Point-treatment TMLE

## Counterfactual means and arm contrasts

For observed data $O=(W,A,Y)$ and treatment level $a$, the counterfactual mean is

$$
\psi_a(P) = E_P\{Q_P(a,W)\}, \qquad Q_P(a,w)=E_P(Y\mid A=a,W=w).
$$

Under consistency, conditional exchangeability, and treatment positivity, this observed-data
functional identifies $E(Y^a)$. Its efficient influence function is

$$
D_a(P)(O)=\frac{\mathbb{1}(A=a)}{g_P(a\mid W)}\{Y-Q_P(A,W)\}
            + Q_P(a,W)-\psi_a(P).
$$

`CounterfactualMean` reports one or all $\psi_a$. `ATE` reports $\psi_a-\psi_{a_0}$ for each
non-reference arm. `RiskRatio` and `OddsRatio` apply smooth transformations and use delta-method
influence curves. The implementation supports binary and multi-valued discrete treatments; the
arm label remains a structured parameter-key field.

The ordinary TMLE fits $Q$ and $g$, bounds predictions when required, fluctuates $Q$ along a
least-favorable logistic submodel, evaluates the targeted counterfactual means, and constructs
inference from the targeted influence curve. See van der Laan & Rubin (2006), Gruber & van der
Laan (2010), and the [targeted-learning references](../references.md#targeted-learning-in-general).

Local implementation: [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py),
[`estimators/targeting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/targeting.py),
and [`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py).
R `tmle` and the pinned `tmle3` update/parameter source are implementation references, not oracles.

## Conditional-population effects

`ATT` and `ATC` condition the effect on the population that received one observed arm. Their
influence curves include the randomness of that conditioning event; they are not ATE curves with a
different summary label. Positivity is needed at the counterfactual reference arm within the
conditioning population.

The estimators use `att_estimate` and `atc_estimate` in
[`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py).
The [`att` and `atc` evidence rows](../evidence.md#the-table) exercise the extra influence term and
target-aware truncation behavior.

## Population interventions

`NaturalCourseMean` reports $E(Y)$. With a declared reference intervention $a_0$,

$$
\operatorname{PAR}=E(Y)-E(Y^{a_0}), \qquad
\operatorname{PAF}=1-\frac{E(Y^{a_0})}{E(Y)}.
$$

The intervention mean uses the targeted arm mean; the complete-data natural-course mean uses the
empirical distribution. PAF is undefined when the observed outcome risk is zero. The construction
follows Díaz Muñoz & van der Laan (2012) and is implemented in
[`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py).

## Missing outcomes and controlled direct effects

With an observation indicator $\Delta$, the outcome residual is additionally weighted by the
inverse observation mechanism $\pi(A,W)=P(\Delta=1\mid A,W)$. The missing-at-random composition
requires positivity of $\pi$ where the intervention places mass. Missingness is modeled as a
design role; missing adjustment or treatment values are not implicitly covered.

For a declared intermediate $Z$, `ControlledDirectEffect(intermediate=z)` targets the treatment
contrast with $Z$ fixed at $z$. Its clever covariate composes the treatment, intermediate, and
observation mechanisms. This is a controlled direct effect at a specified level, not a mediation
decomposition.

Local implementations are in
[`estimators/direct_effect.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/direct_effect.py)
and [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py).
The missing-outcome and CDE exact laws, Gateaux comparisons, and remainder tests are indexed from
the [evidence manifest](../evidence.md#the-table). Díaz & van der Laan (2017) supplies the
randomized-trial missing-outcome construction; broader compositions are claimed only where their
own local derivation and evidence exist.

## Weights, strata, and clustering

Fixed observation weights define the tilted target law $dP_w=w\,dP/E_P(w)$ and are used in
nuisance losses, targeting, plug-in averaging, and influence-curve covariance. `strata=` produces
stratum-specific parameters. `cluster=` changes the independent unit for covariance and fold
construction, not the estimand. These paths have separate weighted-law, stratified-target,
cluster-integration, and parallel-invariance tests.
