# Longitudinal TMLE

## End-of-study regimen means

Let $L_t$ be time-varying history, $A_t$ treatment, $C_t$ censoring, and $Y$ the final outcome.
For a regimen $g^*$, sequential regression works backward from $Q_{T+1}=Y$:

$$
Q_t(h_t)=E\{Q_{t+1}(H_{t+1})\mid H_t=h_t,
            A_t\sim g_t^*, C_t=0\}.
$$

The target is $\psi_{g^*}=E\{Q_0(W)\}$. Its efficient influence function is a telescoping sum

$$
D(P)(O)=\sum_{t=0}^{T} H_t(P)(O)\{Q_{t+1}(O)-Q_t(O)\}
          +Q_0(W)-\psi_{g^*},
$$

where $H_t$ is the cumulative product of regimen-to-observed treatment and uncensoring density
ratios through node $t$. Each regression and fluctuation uses only its at-risk, uncensored follower
population.

Bang & Robins (2005) supplies the sequential-regression foundation; van der Laan & Gruber (2012)
gives longitudinal TMLE for multiple intervention points; Chaffee & van der Laan (2012) covers
dynamic rules. See the [longitudinal references](../references.md#longitudinal-survival-and-marginal-structural-models).

Implementation: [`longitudinal/sequential.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/sequential.py),
[`longitudinal/regimen.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/regimen.py),
and [`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py).

## Finite-sample implementation choices

The engine bounds the cumulative product after multiplying node mechanisms, rather than bounding
each factor separately. Targeting proceeds backward, and the updated pseudo-outcome feeds the next
earlier regression. Dynamic rules receive only the history available at their node. Categorical
treatments use one probability per declared level rather than a binary complement shortcut.

R `ltmle` 1.3-0 is pinned as a bounded implementation witness for one static binary regimen with
fixed mechanism predictions, intercept-only outcome regressions, active cumulative truncation,
nonzero fluctuation, and censoring. Its `FixedTimeTMLE`, `CalcCumG`, and `UpdateQ` source locators
are recorded in [references](../references.md#longitudinal-survival-and-marginal-structural-models).
This witness is deliberately narrow and does not establish cross-fitting, weights, dynamic rules,
or categorical nodes.

Independent evidence covers exact finite laws, Gateaux derivatives, nonzero remainders, dropped-
censoring and reversed-stage mutations, history visibility, and multi-value arm selection. See
[longitudinal estimands outside the target registry](../evidence.md#longitudinal-estimands-outside-the-target-registry).

## Survival and competing risks

An outcome sequence represents an absorbing event process. At horizon $t$, outcome regression is
fit among units event-free and uncensored before $t$; a `t-1` risk-set mask error changes the target
and is caught by a deliberate mutation.

For competing risks, the outcome is a mapping from cause to event sequence. Cause-specific
recursion removes every prior event from later risk sets while accumulating incidence only for the
requested cause. A one-cause process reduces exactly to survival. The result key preserves regimen,
horizon, and cause separately.

Stitelman, De Gruttola & van der Laan (2012) is the survival implementation reference. The R
fixture covers survival at a named horizon. Competing-risk correctness rests on the independent
finite law, Gateaux comparison, all-cause-versus-cause-specific mutation, and one-cause reduction;
no redundant external parity claim is made.

## Longitudinal MSMs and refusals

Longitudinal MSM projections are described in [marginal structural models](marginal-models.md).
Continuous longitudinal doses, stochastic categorical policies, and method variants without a
published sequential derivation are refused rather than mapped to a point-treatment formula.
