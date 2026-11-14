# Treatment at several nodes on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` when a binary treatment is
decided again at every node. The slots of the walk are full treatment paths: a regimen
contributes each path it can follow, a policy node enumerates both arms with its own
probabilities, and the copies of one path prefix share one update. The risk of a regimen is the
plug-in sum over its paths of the path mass times the targeted path risk. The section on
[a treatment at several nodes on the hazard path](../longitudinal-tmle.md#a-treatment-at-several-nodes-on-the-hazard-path)
states the estimand, the influence curve and the exact-law evidence.

The declared run at commit `5ba870d` passed every gated row: 51 of 51 primary tests against
known truth and 39 of 39 property cells with a verdict. The 73 property rows also hold the
agreement with the iterated-mean fit and the one-step walk, reported with no verdict. Four fits held a treatment prefix cell and were recorded
as lost replicates, inside the failure bound of 0.01: two `base` calibration draws, one
`iterated_mean_difference` draw and one draw of the 1,600-row rung. No other fit failed. The
support rows `G62` license the fits it covers with `targeting="iterative"`, and the working model
with `targeting="one_step"`.

## The estimand and the laws

For a regimen $r$ with node probabilities $q_{r,k}(a \mid \bar a_{k-1}, W)$ the target is
$\psi_r(t) = E \sum_{\bar u} q_r(\bar u \mid W)\, F(t \mid \bar u, W)$, the risk of the event by
node $t$ when every unit follows $r$. Here $q_r(\bar u \mid W)$ is the product of the node
probabilities along the path $\bar u$. The regimen fits report every $\psi_r(t)$ and the
contrast of each regimen with `never` at each node. Identification needs sequential
randomization of each $A_k$ given $(W, \bar A_{k-1})$ among the units at risk, censoring at random
given $(W, \bar A_k)$, and positivity of every path the regimen puts mass on: the g-formula of a
longitudinal design with no covariate after baseline.

$W_1 \in \{0, 1\}$ and $W_2 \in \{0, \dots, 3\}$ are independent and uniform. At node $k$ the
treatment is logistic in $(W_1, W_2, A_{k-1})$, the unit then drops out with a probability
logistic in $(W_1, W_2, A_k, A_{k-1})$, and if it stays it has the event with a hazard logistic in
$(k, W_1, W_2, A_k, A_{k-1})$, with $A_0 = 0$.

| law | nodes | treatment $P(A_k = 1)$ | event hazard | smallest cumulative $\prod g_s c_s$ |
| --- | --- | --- | --- | --- |
| `base` | 3 | 0.21 to 0.68 at node 1, and 0.20 to 0.83 later, where the previous arm adds 1.3 on the logit scale | 0.04 to 0.37, lowered by $A_k$ and by $A_{k-1}$ | 0.020 |
| `four` | 4 | close to balanced: 0.47 to 0.55 | 0.04 to 0.09 at node 1, rising to 0.24 to 0.48 at node 4, with smaller arm effects | 0.042 |

`base` is the law of plan X70 section 4.3. `four` has the same form with its own coefficients.
The shipped fit holds the cell of the units at risk with one treatment prefix at zero when none
of them has the event, or at one when all of them do. A regimen that reads a held cell reports
no interval. The policy reads all 16 prefixes at node 4, and the expected count of held cells
in a `four` draw of 2,000 rows is about $2 \times 10^{-9}$. The plan asked for five nodes, but at
five nodes the policy reads 32 prefixes, and a held cell was expected in about one fit in 500.

A held fit is never read as coverage. The study turns every fit with an estimate that lacks the
influence-curve interval into a lost replicate under the failure bound of 0.01, on the primary,
property and pilot paths alike. The design test sums the expected held fits over every fit set
and pilot label at its own size: about 6.6 over the study and its pilot, all on `base`, whose
expected count per draw is $1.5 \times 10^{-4}$ at 2,000 rows and $10^{-3}$ at 1,600. The chance
that any fit set loses more than its allowance to held fits is below 0.01. The committed pilot
held 4 of the 8,000 `calibration_base` draws, against an allowance of 80 and an expectation of
1.2, and no draw of any other label.

Both laws keep every path's cumulative treatment and retention probability ten times above the
declared bound of $10^{-3}$. On `base` the shipped default bound of 0.01 lies within a factor of
two of the smallest, so the study declares its own. The calibration pilot records each fit's
smallest estimated cumulative probability. Its median is 0.019 on `base` and `msm` and 0.036 on
`four`, and the default of 0.01 would have clipped 4 of the 7,996 kept `base` draws (0.05%),
1 of the 8,000 `msm` draws and none of the `four` draws. The smallest estimate in any draw is
0.0092, nine times the declared bound.

| regimen | node $k$ |
| --- | --- |
| `always`, `never` | treat at every node, or at none |
| `rule` | treat when $W_2 \ge 2$, a known rule |
| `policy` | treat with probability $0.3 + 0.4 A_{k-1} + 0.1 W_1$, a known policy |
| `stay` | treat at node 1 with probability $0.3 + 0.2 W_1$, then keep the previous arm |
| `pXYZ` | the static path $(X, Y, Z)$, for the working model |

Every truth is an exact sum over the paths and every efficiency bound the exact standard
deviation of the efficient influence function over the support of the observed law, written in
its iterated-mean form (Bang and Robins 2005), which shares no step with the estimator. The laws
are defined in `tests/studies/hazard_multinode_law.py`.
`tests/unit/test_hazard_multinode_study_design.py` checks each truth against inverse weighting
over the support and against a fit on the law's exact sample, within $10^{-7}$. It checks each
bound against the influence curve that the exact-sample fit reports, row by row.

## The fits

| fit label | law | regimens | targeting |
| --- | --- | --- | --- |
| `base` | `base` | `always`, `never`, `rule`, `policy`, `stay`; reference `never` | per horizon (`targeting="iterative"`) |
| `four` | `four` | `always`, `never`, `policy`; reference `never` | per horizon |
| `msm` | `base` | the eight static paths, under the logit working model $\beta_0 + \beta_1 d + \beta_2 t + \beta_3 d W_1$, with $d$ the path's cumulative dose through $t$ | one step (`targeting="one_step"`) |
| `onestep` | `base` | `always`, `never`, `policy`; reference `never`; reported only | one step, paired with a per-horizon fit of the same draw |

| nuisance | correct learner | wrong learner |
| --- | --- | --- |
| hazard | unpenalized logistic regression on `W1`, `W2`, `A1`, ..., `Ak` at each node | logistic regression on `Ak` alone (`LastArmLogistic`); on `four` also the event share at each node, which reads no arm |
| treatment | unpenalized logistic regression on `W1`, `W2`, `A1`, ..., `A(k-1)` at each node | the prior share |
| censoring | unpenalized logistic regression on `W1`, `W2`, `A1`, ..., `Ak` at each node | the prior share |

Each factor of both laws depends on $W$ and on the last one or two arms, so the correct learners
are correctly specified. They hold no interaction, so they are not saturated and targeting
moves every fit. Every `base`, `four` and `onestep` fit raises `SparseInfluenceWarning`: the risk of a static
arm by the last node reads its influence curve from the few units that followed the arm, each
weighted by the inverse of a cumulative probability as small as 0.02. The study declares the
warning as expected on those fit labels, and the calibration rows read the resulting intervals.

The regimen fits use the per-horizon walk, the shipped default. On one `base` draw of 2,000 rows the one-step walk took 0.6 seconds over
`always` and `never`, 33.5 over `policy` and `stay`, and 96 over `always`, `never` and `policy`.
Over the five `base` regimens it took 250 seconds, and 10,776 steps over `always`, `never`,
`rule` and `policy`. The enumerated policy slows the walk, not the rule. The reported `onestep`
fit label measures the one-step walk on regimen curves at 200 draws; the working model needs the
one-step walk, the only algorithm that targets its scores.

Every fit of the committed pilot converged. The pilot records, per draw, the largest iteration
count of a walk and the draw's wall time on one core at n = 2,000.

| pilot label | draws kept | iterations: mean (most) | mean seconds a draw |
| --- | --- | --- | --- |
| `calibration_base` | 7,996 | 3.7 (5) | 1.42 |
| `calibration_four` | 8,000 | 3.3 (6) | 1.76 |
| `calibration_msm` | 8,000 | 30.2 (70) | 0.97 |
| `wrong_learner_base` | 800 | 4.0 (5) | 1.14 |
| `wrong_learner_four` | 800 | 3.4 (5) | 1.34 |
| `wrong_learner_msm` | 800 | 120.8 (173) | 2.43 |
| `wrong_learner_node` | 800 | 3.9 (4) | 1.50 |
| `both_wrong_base` | 20 | 3.5 (4) | 0.98 |
| `fits_differ_base` | 200 | 3.7 (5) | 1.33 |
| `fits_differ_four` | 200 | 3.3 (5) | 1.62 |
| `fits_differ_mechanism_correct` | 200 | 4.0 (5) | 1.26 |

A `calibration` draw of `base` and `four` fits the simultaneous band as well. A `fits_differ`
draw's time includes its iterated-mean fits. The `msm` fits count the steps of the one-step
walk, so their iteration counts do not compare with those of the per-horizon walk.

No maintained package targets discrete-time hazards over treatment paths, so the study names no
comparator and no canonical implementation is compared.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[always vs never @ t=1]` | the risk by node 1 under treatment at every node minus the risk under no treatment | `cleverly` | -0.000708 to 0.0011 | 0.9479 | 0.9977 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[always vs never @ t=2]` | the risk by node 2 under treatment at every node minus the risk under no treatment | `cleverly` | -0.000479 to 0.0021 | 0.9492 | 0.9916 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[always vs never @ t=3]` | the risk by node 3 under treatment at every node minus the risk under no treatment | `cleverly` | -0.000836 to 0.0026 | 0.9471 | 0.9729 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[policy vs never @ t=1]` | the risk by node 1 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.000221 to 0.000436 | 0.9463 | 0.9984 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[policy vs never @ t=2]` | the risk by node 2 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.000276 to 0.000977 | 0.9475 | 0.9990 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[policy vs never @ t=3]` | the risk by node 3 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.000681 to 0.0013 | 0.9425 | 0.9743 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[rule vs never @ t=1]` | the risk by node 1 under the known rule that treats at every node when W2 is at least 2 minus the risk under no treatment | `cleverly` | -0.000720 to 0.000724 | 0.9487 | 0.9889 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[rule vs never @ t=2]` | the risk by node 2 under the known rule that treats at every node when W2 is at least 2 minus the risk under no treatment | `cleverly` | -0.000475 to 0.0016 | 0.9417 | 0.9871 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[rule vs never @ t=3]` | the risk by node 3 under the known rule that treats at every node when W2 is at least 2 minus the risk under no treatment | `cleverly` | -0.000748 to 0.0019 | 0.9367 | 0.9743 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[stay vs never @ t=1]` | the risk by node 1 under a draw at node 1 with probability 0.3 + 0.2 W1, then the previous arm kept minus the risk under no treatment | `cleverly` | -0.000234 to 0.000542 | 0.9446 | 0.9986 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[stay vs never @ t=2]` | the risk by node 2 under a draw at node 1 with probability 0.3 + 0.2 W1, then the previous arm kept minus the risk under no treatment | `cleverly` | -0.000143 to 0.000956 | 0.9475 | 0.9992 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `ate_regimen[stay vs never @ t=3]` | the risk by node 3 under a draw at node 1 with probability 0.3 + 0.2 W1, then the previous arm kept minus the risk under no treatment | `cleverly` | -0.000283 to 0.0012 | 0.9479 | 0.9755 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[always @ t=1]` | the risk of the event by node 1 under treatment at every node | `cleverly` | -0.000267 to 0.000956 | 0.9496 | 0.9883 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[always @ t=2]` | the risk of the event by node 2 under treatment at every node | `cleverly` | -0.000502 to 0.0012 | 0.9379 | 0.9767 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[always @ t=3]` | the risk of the event by node 3 under treatment at every node | `cleverly` | -0.000593 to 0.0018 | 0.9337 | 0.9654 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[never @ t=1]` | the risk of the event by node 1 under no treatment at any node | `cleverly` | -0.000535 to 0.000818 | 0.9492 | 1.0114 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[never @ t=2]` | the risk of the event by node 2 under no treatment at any node | `cleverly` | -0.0014 to 0.000521 | 0.9483 | 1.0075 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[never @ t=3]` | the risk of the event by node 3 under no treatment at any node | `cleverly` | -0.0016 to 0.000963 | 0.9421 | 0.9845 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[policy @ t=1]` | the risk of the event by node 1 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.000237 to 0.000735 | 0.9529 | 1.0089 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[policy @ t=2]` | the risk of the event by node 2 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.000709 to 0.000482 | 0.9479 | 1.0129 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[policy @ t=3]` | the risk of the event by node 3 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.000697 to 0.000637 | 0.9413 | 0.9876 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[rule @ t=1]` | the risk of the event by node 1 under the known rule that treats at every node when W2 is at least 2 | `cleverly` | -0.000414 to 0.000702 | 0.9533 | 0.9922 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[rule @ t=2]` | the risk of the event by node 2 under the known rule that treats at every node when W2 is at least 2 | `cleverly` | -0.000672 to 0.000835 | 0.9550 | 0.9990 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[rule @ t=3]` | the risk of the event by node 3 under the known rule that treats at every node when W2 is at least 2 | `cleverly` | -0.000692 to 0.0012 | 0.9425 | 0.9883 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[stay @ t=1]` | the risk of the event by node 1 under a draw at node 1 with probability 0.3 + 0.2 W1, then the previous arm kept | `cleverly` | -0.000178 to 0.000769 | 0.9517 | 1.0058 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[stay @ t=2]` | the risk of the event by node 2 under a draw at node 1 with probability 0.3 + 0.2 W1, then the previous arm kept | `cleverly` | -0.000734 to 0.000621 | 0.9513 | 1.0066 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; five regimens against never treating | `risk_regimen[stay @ t=3]` | the risk of the event by node 3 under a draw at node 1 with probability 0.3 + 0.2 W1, then the previous arm kept | `cleverly` | -0.000759 to 0.0010 | 0.9450 | 0.9901 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[always vs never @ t=1]` | the risk by node 1 under treatment at every node minus the risk under no treatment | `cleverly` | -0.000443 to 0.000692 | 0.9479 | 0.9938 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[always vs never @ t=2]` | the risk by node 2 under treatment at every node minus the risk under no treatment | `cleverly` | -0.000693 to 0.0015 | 0.9613 | 1.0334 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[always vs never @ t=3]` | the risk by node 3 under treatment at every node minus the risk under no treatment | `cleverly` | -0.0021 to 0.0018 | 0.9508 | 0.9998 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[always vs never @ t=4]` | the risk by node 4 under treatment at every node minus the risk under no treatment | `cleverly` | -0.0024 to 0.0033 | 0.9479 | 1.0046 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[policy vs never @ t=1]` | the risk by node 1 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.000157 to 0.000249 | 0.9454 | 0.9967 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[policy vs never @ t=2]` | the risk by node 2 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.000367 to 0.000717 | 0.9554 | 1.0315 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[policy vs never @ t=3]` | the risk by node 3 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.0013 to 0.0010 | 0.9454 | 0.9891 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `ate_regimen[policy vs never @ t=4]` | the risk by node 4 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 minus the risk under no treatment | `cleverly` | -0.000837 to 0.0027 | 0.9467 | 0.9996 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[always @ t=1]` | the risk of the event by node 1 under treatment at every node | `cleverly` | -0.000160 to 0.000586 | 0.9437 | 0.9805 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[always @ t=2]` | the risk of the event by node 2 under treatment at every node | `cleverly` | -0.000383 to 0.0010 | 0.9546 | 1.0254 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[always @ t=3]` | the risk of the event by node 3 under treatment at every node | `cleverly` | -0.0012 to 0.0012 | 0.9487 | 1.0328 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[always @ t=4]` | the risk of the event by node 4 under treatment at every node | `cleverly` | -0.0027 to 0.0012 | 0.9425 | 0.9985 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[never @ t=1]` | the risk of the event by node 1 under no treatment at any node | `cleverly` | -0.000341 to 0.000518 | 0.9454 | 0.9992 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[never @ t=2]` | the risk of the event by node 2 under no treatment at any node | `cleverly` | -0.000947 to 0.000774 | 0.9508 | 1.0209 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[never @ t=3]` | the risk of the event by node 3 under no treatment at any node | `cleverly` | -0.0013 to 0.0017 | 0.9421 | 0.9914 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[never @ t=4]` | the risk of the event by node 4 under no treatment at any node | `cleverly` | -0.0033 to 0.000872 | 0.9404 | 0.9906 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[policy @ t=1]` | the risk of the event by node 1 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.000173 to 0.000442 | 0.9442 | 0.9949 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[policy @ t=2]` | the risk of the event by node 2 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.000401 to 0.000578 | 0.9546 | 1.0182 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[policy @ t=3]` | the risk of the event by node 3 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.000608 to 0.000704 | 0.9563 | 1.0200 | pass |
| four nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; always and a known policy against never treating | `risk_regimen[policy @ t=4]` | the risk of the event by node 4 under the known policy that treats with probability 0.3 + 0.4 A(k-1) + 0.1 W1 | `cleverly` | -0.0011 to 0.000460 | 0.9496 | 0.9884 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; the logit working model (1, d, t, d W1) over the eight static paths, d the cumulative dose through t | `msm_regimen[(intercept)]` | the intercept of the logit working model (1, d, t, d W1) over the risk of each static path by each node | `cleverly` | -0.0057 to 0.0029 | 0.9537 | 1.0187 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; the logit working model (1, d, t, d W1) over the eight static paths, d the cumulative dose through t | `msm_regimen[dose]` | the cumulative-dose coefficient of the logit working model (1, d, t, d W1) over the risk of each static path by each node | `cleverly` | -0.0041 to 0.0034 | 0.9467 | 0.9922 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; the logit working model (1, d, t, d W1) over the eight static paths, d the cumulative dose through t | `msm_regimen[dose_W1]` | the cumulative-dose by W1 coefficient of the logit working model (1, d, t, d W1) over the risk of each static path by each node | `cleverly` | -0.0036 to 0.0041 | 0.9517 | 0.9950 | pass |
| three nodes, two baseline covariates, a binary treatment at every node logistic in W and the previous arm, logistic dropout and event hazard in W and the last two arms, one cause; the logit working model (1, d, t, d W1) over the eight static paths, d the cumulative dose through t | `msm_regimen[t]` | the node coefficient of the logit working model (1, d, t, d W1) over the risk of each static path by each node | `cleverly` | -0.0016 to 0.0027 | 0.9479 | 0.9987 | pass |
<!-- /generated -->

The primary rows draw samples of 2,000 rows for each of `base`, `four` and `msm` and report every
risk and contrast of each regimen fit and every coefficient of the working model. The shared
primary rule (`repeated_sizing.primary_replicates`) sets the count from the committed
calibration pilots of these fit labels, and at the committed pilots it gives 2,400.

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `double_robustness` | `base_contrast__both_correct` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: correct logistic hazard, treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0024 to 0.0024, margin 0.0081, SE ratio 0.9901 | pass |
| `double_robustness` | `base_contrast__both_wrong` | control | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: a hazard learner that reads the node's own arm alone and treatment and censoring learners that predict each node's prior share | bias interval must fall entirely outside the margin, with the reported standard error still on the scale of the empirical spread | bias 0.0973 to 0.1017, margin 0.0075, SE ratio 1.5740 | pass |
| `double_robustness` | `base_contrast__mechanism_correct` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0031 to 0.0016, margin 0.0079, SE ratio 1.0211 | pass |
| `double_robustness` | `base_contrast__outcome_correct` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: a correct hazard learner with treatment and censoring learners that predict each node's prior share | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0022 to 0.0022, margin 0.0074, SE ratio 1.5767 | pass |
| `double_robustness` | `base_policy__mechanism_correct` | positive | the three-node law, n = 2,000, the risk under the known policy by the last node: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.000886 to 0.0010, margin 0.0032, SE ratio 0.9755 | pass |
| `double_robustness` | `base_rule__mechanism_correct` | positive | the three-node law, n = 2,000, the risk under the known rule by the last node: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0017 to 0.000971, margin 0.0045, SE ratio 0.9804 | pass |
| `double_robustness` | `base_stay__mechanism_correct` | positive | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0012 to 0.0014, margin 0.0043, SE ratio 0.9771 | pass |
| `double_robustness` | `four_contrast__mechanism_correct` | positive | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0067 to 0.0016, margin 0.0138, SE ratio 0.9876 | pass |
| `double_robustness` | `four_contrast__node_hazard` | positive | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: a hazard learner that predicts each node's prior share of the event, with no arm and no W, and correct treatment and censoring learners; its plug-in contrast is zero in every sample, so the targeting alone moves the estimate | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0031 to 0.0039, margin 0.0136, SE ratio 1.0053 | pass |
| `double_robustness` | `four_policy__mechanism_correct` | positive | the four-node law, n = 2,000, the risk under the known policy by the last node: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0011 to 0.0010, margin 0.0036, SE ratio 1.0095 | pass |
| `double_robustness` | `msm_dose_W1__mechanism_correct` | positive | the logit working model over the eight static paths of the three-node law, n = 2,000, the cumulative-dose by W1 coefficient: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0059 to 0.0049, margin 0.0180, SE ratio 1.0193 | pass |
| `double_robustness` | `msm_dose__mechanism_correct` | positive | the logit working model over the eight static paths of the three-node law, n = 2,000, the cumulative-dose coefficient: a hazard learner that reads the node's own arm alone, with correct treatment and censoring learners | bias interval inside the equivalence margin, with the reported standard error on the scale of the empirical spread | bias -0.0087 to 0.0015, margin 0.0172, SE ratio 1.0311 | pass |
| `interval_calibration` | `base_contrast__correctly_specified` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9400 to 0.9560, SE ratio 0.9707 to 1.0224, empirical efficiency ratio 0.9726 to 1.0244, reported efficiency ratio 0.9924 to 0.9965 | pass |
| `interval_calibration` | `base_contrast__noise_control` | control | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: one efficiency-bound unit of independent noise is added to each estimate | the empirical efficiency ratio must rise above the band | coverage 0.8197 to 0.8465, SE ratio 0.6823 to 0.7179, empirical efficiency ratio 1.3848 to 1.4577, reported efficiency ratio 0.9924 to 0.9965 | pass |
| `interval_calibration` | `base_contrast__shrunken_se_control` | control | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the reported standard errors are multiplied by a declared factor below one | the SE-ratio interval must fall below the calibration band | coverage 0.8177 to 0.8446, SE ratio 0.6796 to 0.7149, empirical efficiency ratio 0.9737 to 1.0245, reported efficiency ratio 0.6947 to 0.6976 | pass |
| `interval_calibration` | `base_policy__correctly_specified` | positive | the three-node law, n = 2,000, the risk under the known policy by the last node: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9415 to 0.9573, SE ratio 0.9737 to 1.0259, empirical efficiency ratio 0.9740 to 1.0265, reported efficiency ratio 0.9987 to 1.0001 | pass |
| `interval_calibration` | `base_rule__correctly_specified` | positive | the three-node law, n = 2,000, the risk under the known rule by the last node: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9373 to 0.9537, SE ratio 0.9790 to 1.0314, empirical efficiency ratio 0.9684 to 1.0204, reported efficiency ratio 0.9980 to 0.9998 | pass |
| `interval_calibration` | `base_stay__correctly_specified` | positive | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9413 to 0.9571, SE ratio 0.9677 to 1.0203, empirical efficiency ratio 0.9759 to 1.0287, reported efficiency ratio 0.9940 to 0.9971 | pass |
| `interval_calibration` | `four_contrast__correctly_specified` | positive | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9403 to 0.9562, SE ratio 0.9747 to 1.0244, empirical efficiency ratio 0.9758 to 1.0253, reported efficiency ratio 0.9982 to 1.0008 | pass |
| `interval_calibration` | `four_policy__correctly_specified` | positive | the four-node law, n = 2,000, the risk under the known policy by the last node: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9351 to 0.9518, SE ratio 0.9601 to 1.0106, empirical efficiency ratio 0.9915 to 1.0436, reported efficiency ratio 1.0013 to 1.0026 | pass |
| `interval_calibration` | `msm_dose_W1__correctly_specified` | positive | the logit working model over the eight static paths of the three-node law, n = 2,000, the cumulative-dose by W1 coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9411 to 0.9569, SE ratio 0.9738 to 1.0220, empirical efficiency ratio 0.9822 to 1.0309, reported efficiency ratio 1.0030 to 1.0048 | pass |
| `interval_calibration` | `msm_dose__correctly_specified` | positive | the logit working model over the eight static paths of the three-node law, n = 2,000, the cumulative-dose coefficient: both nuisances are correctly specified with an independently computed efficiency bound | SE ratio and coverage intervals both inside their calibration bands, with both efficiency-ratio intervals inside their bands | coverage 0.9390 to 0.9551, SE ratio 0.9612 to 1.0105, empirical efficiency ratio 0.9920 to 1.0429, reported efficiency ratio 1.0011 to 1.0038 | pass |
| `iterated_mean_difference` | `base__iterated_curve_decrease` | diagnostic | the three-node law, n = 2,000, beside the iterated-mean fit of the same regimens on the same draw: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `base_contrast__hazard_fit` | diagnostic | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9178 to 0.9616, SE ratio 0.9988 | reported |
| `iterated_mean_difference` | `base_contrast__isotonic_difference` | diagnostic | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1749, mean difference -0.000338 to 0.000692 | reported |
| `iterated_mean_difference` | `base_contrast__iterated_fit` | diagnostic | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9106 to 0.9564, SE ratio 0.9720 | reported |
| `iterated_mean_difference` | `base_contrast__mechanism_correct_difference` | diagnostic | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard estimate minus the iterated-mean estimate of the same regimens on the same draw, with the hazard learner reading the node's own arm alone and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1326, mean difference -0.000419 to 0.000218 | reported |
| `iterated_mean_difference` | `base_contrast__paired_difference` | diagnostic | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1749, mean difference -0.000338 to 0.000692 | reported |
| `iterated_mean_difference` | `base_policy__hazard_fit` | diagnostic | the three-node law, n = 2,000, the risk under the known policy by the last node: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9310 to 0.9707, SE ratio 1.0219 | reported |
| `iterated_mean_difference` | `base_policy__isotonic_difference` | diagnostic | the three-node law, n = 2,000, the risk under the known policy by the last node: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1643, mean difference -0.000355 to 0.000020 | reported |
| `iterated_mean_difference` | `base_policy__iterated_fit` | diagnostic | the three-node law, n = 2,000, the risk under the known policy by the last node: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9193 to 0.9626, SE ratio 0.9979 | reported |
| `iterated_mean_difference` | `base_policy__mechanism_correct_difference` | diagnostic | the three-node law, n = 2,000, the risk under the known policy by the last node: the hazard estimate minus the iterated-mean estimate of the same regimens on the same draw, with the hazard learner reading the node's own arm alone and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1062, mean difference -0.000010 to 0.000188 | reported |
| `iterated_mean_difference` | `base_policy__paired_difference` | diagnostic | the three-node law, n = 2,000, the risk under the known policy by the last node: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1643, mean difference -0.000355 to 0.000020 | reported |
| `iterated_mean_difference` | `base_rule__hazard_fit` | diagnostic | the three-node law, n = 2,000, the risk under the known rule by the last node: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9222 to 0.9647, SE ratio 1.0033 | reported |
| `iterated_mean_difference` | `base_rule__isotonic_difference` | diagnostic | the three-node law, n = 2,000, the risk under the known rule by the last node: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0813, mean difference -0.000156 to 0.000106 | reported |
| `iterated_mean_difference` | `base_rule__iterated_fit` | diagnostic | the three-node law, n = 2,000, the risk under the known rule by the last node: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9207 to 0.9636, SE ratio 0.9950 | reported |
| `iterated_mean_difference` | `base_rule__mechanism_correct_difference` | diagnostic | the three-node law, n = 2,000, the risk under the known rule by the last node: the hazard estimate minus the iterated-mean estimate of the same regimens on the same draw, with the hazard learner reading the node's own arm alone and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0715, mean difference -0.000162 to 0.000026 | reported |
| `iterated_mean_difference` | `base_rule__paired_difference` | diagnostic | the three-node law, n = 2,000, the risk under the known rule by the last node: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0813, mean difference -0.000156 to 0.000106 | reported |
| `iterated_mean_difference` | `base_stay__hazard_fit` | diagnostic | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9266 to 0.9677, SE ratio 1.0188 | reported |
| `iterated_mean_difference` | `base_stay__isotonic_difference` | diagnostic | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1696, mean difference -0.000501 to 0.000018 | reported |
| `iterated_mean_difference` | `base_stay__iterated_fit` | diagnostic | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9034 to 0.9512, SE ratio 0.9842 | reported |
| `iterated_mean_difference` | `base_stay__mechanism_correct_difference` | diagnostic | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: the hazard estimate minus the iterated-mean estimate of the same regimens on the same draw, with the hazard learner reading the node's own arm alone and the treatment and censoring models correct | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1613, mean difference -0.000015 to 0.000387 | reported |
| `iterated_mean_difference` | `base_stay__paired_difference` | diagnostic | the three-node law, n = 2,000, the risk under a draw at node 1 then kept, by the last node: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.1696, mean difference -0.000501 to 0.000018 | reported |
| `iterated_mean_difference` | `four__iterated_curve_decrease` | diagnostic | the four-node law, n = 2,000, beside the iterated-mean fit of the same regimens on the same draw: the largest decrease of the iterated-mean risk curve of either arm between consecutive visits | no verdict; the share of draws whose curve decreases somewhere is reported | decreasing share 0, 99% interval 0 to 0.0066 | reported |
| `iterated_mean_difference` | `four_contrast__hazard_fit` | diagnostic | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9282 to 0.9688, SE ratio 0.9741 | reported |
| `iterated_mean_difference` | `four_contrast__isotonic_difference` | diagnostic | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.2705, mean difference -0.0023 to 0.000405 | reported |
| `iterated_mean_difference` | `four_contrast__iterated_fit` | diagnostic | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.8963 to 0.9460, SE ratio 0.9153 | reported |
| `iterated_mean_difference` | `four_contrast__paired_difference` | diagnostic | the four-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.2705, mean difference -0.0023 to 0.000405 | reported |
| `iterated_mean_difference` | `four_policy__hazard_fit` | diagnostic | the four-node law, n = 2,000, the risk under the known policy by the last node: the hazard fit of the plans, against the exact value | no verdict; coverage and SE ratio are reported beside the iterated-mean fit's | coverage 0.9252 to 0.9667, SE ratio 1.0065 | reported |
| `iterated_mean_difference` | `four_policy__isotonic_difference` | diagnostic | the four-node law, n = 2,000, the risk under the known policy by the last node: the hazard estimate minus the isotonic iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.2596, mean difference -0.000307 to 0.000373 | reported |
| `iterated_mean_difference` | `four_policy__iterated_fit` | diagnostic | the four-node law, n = 2,000, the risk under the known policy by the last node: the iterated-mean fit of the same plans on the same draw, against the exact value | no verdict; coverage and SE ratio are reported beside the hazard fit's | coverage 0.9179 to 0.9616, SE ratio 0.9532 | reported |
| `iterated_mean_difference` | `four_policy__paired_difference` | diagnostic | the four-node law, n = 2,000, the risk under the known policy by the last node: the hazard estimate minus the iterated-mean estimate of the same plans on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.2596, mean difference -0.000307 to 0.000373 | reported |
| `one_step_walk` | `onestep_contrast__one_step_fit` | diagnostic | the three-node law fitted with the one-step walk, n = 2,000, the risk difference of treating at every node against never by the last node: the one-step fit of the regimens, against the exact value | no verdict; coverage and SE ratio are reported | coverage 0.8894 to 0.9781, SE ratio 0.9873 | reported |
| `one_step_walk` | `onestep_contrast__paired_difference` | diagnostic | the three-node law fitted with the one-step walk, n = 2,000, the risk difference of treating at every node against never by the last node: the one-step estimate minus the per-horizon estimate of the same regimens on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0235, mean difference -0.000006 to 0.000272 | reported |
| `one_step_walk` | `onestep_policy__one_step_fit` | diagnostic | the three-node law fitted with the one-step walk, n = 2,000, the risk under the known policy by the last node: the one-step fit of the regimens, against the exact value | no verdict; coverage and SE ratio are reported | coverage 0.9165 to 0.9897, SE ratio 0.9906 | reported |
| `one_step_walk` | `onestep_policy__paired_difference` | diagnostic | the three-node law fitted with the one-step walk, n = 2,000, the risk under the known policy by the last node: the one-step estimate minus the per-horizon estimate of the same regimens on the same draw | no verdict; the SD of the difference over the efficiency-bound standard error is reported | paired SD over bound 0.0162, mean difference -0.000050 to 0.000025 | reported |
| `power` | `base_contrast__alternative` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the same test applied to a law with a real effect | rejection lower bound clears the minimum power | rejection 1, 0.9990 to 1 | pass |
| `root_n_and_efficiency` | `base_contrast__n_1600` | control | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: bias, coverage and SE calibration at n = 1,600, a rate rung that carries no claim | coverage interval lies below nominal or clears the declared floor | bias -0.000017, coverage 0.9370 to 0.9747, SE ratio 0.9944 | pass |
| `root_n_and_efficiency` | `base_contrast__n_2000` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000228, coverage 0.9208 to 0.9637, SE ratio 0.9821 | pass |
| `root_n_and_efficiency` | `base_contrast__n_8000` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000482, coverage 0.9107 to 0.9565, SE ratio 0.9667 | pass |
| `root_n_and_efficiency` | `base_policy__n_1600` | control | the three-node law, n = 2,000, the risk under the known policy by the last node: bias, coverage and SE calibration at n = 1,600, a rate rung that carries no claim | coverage interval lies below nominal or clears the declared floor | bias -0.000259, coverage 0.9266 to 0.9677, SE ratio 0.9934 | pass |
| `root_n_and_efficiency` | `base_policy__n_2000` | positive | the three-node law, n = 2,000, the risk under the known policy by the last node: bias, coverage and SE calibration at n = 2,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias 0.000027, coverage 0.9179 to 0.9616, SE ratio 0.9918 | pass |
| `root_n_and_efficiency` | `base_policy__n_8000` | positive | the three-node law, n = 2,000, the risk under the known policy by the last node: bias, coverage and SE calibration at n = 8,000 | bias inside the margin, coverage clears the floor, SE ratio inside the sanity band | bias -0.000132, coverage 0.9252 to 0.9667, SE ratio 0.9740 | pass |
| `root_n_rate` | `base_contrast__empirical_sd` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5329 to -0.4288 | pass |
| `root_n_rate` | `base_contrast__reported_se` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.4987 to -0.4922 | pass |
| `root_n_rate` | `base_policy__empirical_sd` | positive | the three-node law, n = 2,000, the risk under the known policy by the last node: log empirical spread of the estimates regressed on log n across three sizes | slope interval inside the root-n band and excluding -1/4 | slope -0.5434 to -0.4342 | pass |
| `root_n_rate` | `base_policy__reported_se` | positive | the three-node law, n = 2,000, the risk under the known policy by the last node: the same regression applied to the mean reported standard error | slope interval inside the root-n band and excluding -1/4 | slope -0.5006 to -0.4985 | pass |
| `simultaneous_coverage` | `base_curve__pointwise_joint_control` | control | the three-node law, n = 2,000, every risk and risk difference of every regimen read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5394 to 0.5751 | pass |
| `simultaneous_coverage` | `base_curve__simultaneous_band` | positive | the three-node law, n = 2,000, every risk and risk difference of every regimen read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9324 to 0.9494 | pass |
| `simultaneous_coverage` | `four_curve__pointwise_joint_control` | control | the four-node law, n = 2,000, every risk and risk difference of every regimen read as one joint set: the pointwise 95% intervals of the same fits, read jointly | joint coverage upper endpoint must fall below the nominal rate | joint coverage 0.5412 to 0.5768 | pass |
| `simultaneous_coverage` | `four_curve__simultaneous_band` | positive | the four-node law, n = 2,000, every risk and risk difference of every regimen read as one joint set: the max-t multiplier band over every estimand the law reports, from the same-row centered influence curves | joint coverage interval inside the calibration coverage band | joint coverage 0.9392 to 0.9553 | pass |
| `targeting_necessity` | `base_contrast__targeted` | positive | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the estimator with a hazard learner that reads the node's own arm alone and correct treatment and censoring learners, so targeting does all the adjusting | bias interval inside the equivalence margin | bias -0.0031 to 0.0016, margin 0.0079 | pass |
| `targeting_necessity` | `base_contrast__untargeted` | control | the three-node law, n = 2,000, the risk difference of treating at every node against never by the last node: the same fit's plug-in of the initial hazards, with every targeting step removed | bias interval must fall entirely outside the margin | bias 0.1120 to 0.1155, margin 0.0057 | pass |
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | on `base` the `always` minus `never` contrast at node 3 and the risks of `rule`, `policy` and `stay`; on `four` the contrast and the policy's risk at node 4; on `msm` the coefficients of $d$ and $d W_1$. The shrunken-SE and noise controls on the `base` contrast | 5,200 at n = 2,000 | an SE ratio or coverage interval outside its band, or an efficiency ratio outside 0.90 to 1.10 |
| `simultaneous_coverage` | the band over every estimate of the `base` and `four` fits, and its pointwise control | as calibration | joint coverage outside the calibration band |
| `double_robustness` | the `base` contrast with both, one or neither nuisance correct; every other calibration arm with the wrong hazard learner and the correct mechanism; the `four` contrast with the node-only hazard learner and the correct mechanism | 1,200 at n = 2,000, and 1,600 for the node-only cell | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the `base` contrast from the wrong hazard learner, targeted and plug-in, read from the `base` `mechanism_correct` fits | as those fits | the targeted bias outside the margin, or a displacement below 0.25 |
| `root_n_and_efficiency`, `root_n_rate` | the `base` contrast and the policy's risk at n = 1,600, 2,000 and 8,000; the n = 1,600 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the `base` contrast against zero, read from the `base` calibration fits | as calibration | a rejection lower bound below the minimum power |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| `one_step_walk` | on `base`, the contrast and the policy's risk from the one-step walk over `always`, `never` and `policy`: coverage, SE ratio and convergence, and the paired difference from the per-horizon fit of the same draw | 200 at n = 2,000 | nothing: the family is reported with no verdict |
| `iterated_mean_difference` | on `base` and `four`: the hazard estimate minus the iterated-mean estimate of the same draw, without and with `isotonic=True`, each fit against the truth, and the largest decrease of an iterated-mean curve. On the `base` `mechanism_correct` draws: the same paired difference with both regressions omitting $W$ | 800 at n = 2,000, and the robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.replicates_from_pilot`
over the eight arms, read at the committed calibration pilot's kept draws (7,996, the `base`
label's 8,000 less its 4 held draws). The pilot's SE ratios run from 0.977 (`base` `stay`) to
0.997 (`base` policy), with mean 0.990. The rule gives 5,200 replicates, with per-cell and
family-wise pass probabilities 0.995 and 0.961. At 5,200 replicates the shrunken-SE control
(standard errors scaled by 0.7) leaves the SE-ratio band, and the noise control leaves the
efficiency band, with probability above 0.999999.

Each `mechanism_correct` fit set and the node-only fit set take the rule of plan X70 rev 2.7
from the wrong-learner pilot of 800 draws per fit label: the larger of 1,200 and the smallest
multiple of 400 that passes with probability 0.99 at the pilot's standardized bias plus one of
its standard errors, with no cap. On `base` the pilot pools the 200 `fits_differ_mechanism_correct`
draws, whose hazard fit is the same. The pilot's standardized biases are at most 0.049 under
the wrong hazard learner, so every `mechanism_correct` fit set takes 1,200. The node-only
learner's `four` contrast reads 0.074, so its fit set takes 1,600. A need above 4,000 would
have stopped the declaration for a diagnosis.

The wrong hazard learner's plug-in `base` contrast has an exact limit 3.54 efficiency standard
errors from the truth at n = 2,000, so the plug-in arm of `targeting_necessity` can fail. With
both nuisances wrong the exact `base` limit lies 3.07 efficiency standard errors from the truth,
so the `both_wrong` control can fail. If it does not clear the margin, the control has failed and
the study is a defect suspect. Without $W$ the wrong learner's plug-in coefficient of $d W_1$ is
zero in every sample, 3.49 standard errors from the truth, so its robustness cell reads the
targeting alone.

Elsewhere the wrong hazard learner's plug-in lies close to the truth: +0.46, -1.01 and -0.96
standard errors for the `base` rule, policy and `stay`, and +0.76 and -0.16 for the `four`
contrast and policy. Those cells pass whether or not targeting moves the estimate, so they are
consistency checks, not evidence of the weighting over the policy paths. That weighting rests
on the exact-law witnesses of `tests/unit/test_hazard_multinode.py` and on the node-only cell.
The node-only learner's plug-in `four` contrast is zero in every sample, 2.20 standard errors
from the truth, so that cell reads the targeting alone; the `four` policy has no such witness.

The power cell's exact-law power at n = 2,000 is above 0.99. The necessity cells read the
`base` `mechanism_correct` fits and the power cell the `base` calibration fits, so neither refits
a configuration another fit set runs; the ladder's 2,000-row rung keeps its own draws so that
every rung has 800. At bias zero every positive bias arm passes with probability above 0.99. That stays true while the true bias is below 0.108
standard deviations at 1,200 replicates and below 0.077 at 800.

The calibration pilots of `base` and `four` fit the simultaneous band, as the declared fits do.
The band covered the whole curve in 94.15% of the kept `base` draws and 94.34% of the `four`
draws, each with a standard error of 0.26 points. The pointwise intervals covered every
estimate at once in 55.3% and 55.7%. At the pilot's joint coverage minus one standard error,
the 99% interval of the declared `simultaneous_coverage` cell at 5,200 replicates lies inside
the band with probability 0.997 on `base` and 0.9995 on `four`, against the 0.90 the design
requires.

The iterated-mean fit regresses each node on the cell means of $W$ and the arms it conditions
on, a saturated regression, so with the correct mechanism both fits are correctly specified and
asymptotically efficient, and they differ by $o_P(n^{-1/2})$. At n = 2,000 the `four` policy's
regression reads up to 128 cells, so the paired difference also carries the noise of sparse
cells. The study reports the ratios and the decreasing share with its interval, with no floor:
X70 reports the agreement of the two fits instead of requiring a gap (the owner's decision of
2026-11-13).

In the committed pilot of 200 draws per fit label, the standard deviation of the paired
difference, over the efficiency-bound standard error, is 0.160 for the `base` contrast, 0.070
for `rule`, 0.154 for the policy and 0.165 for `stay`, and 0.317 for the `four` contrast and
0.245 for the `four` policy. No iterated-mean curve decreased in any of the 400 draws (99%
interval for the share 0 to 0.026 per fit label), so `isotonic=True` changed no estimate and
its ratios equal these. On the `base` `mechanism_correct` draws, with both regressions omitting
$W$, the ratios are 0.131, 0.077, 0.101 and 0.153.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 2400 | primary replications |
| `n` | 2000 | observations per replication of the record's reference size |
| `independent_tests_passed` | 51 | truth tests passing |
| `independent_tests_total` | 51 | truth tests reported |
| `property_cells_passed` | 39 | property cells passing |
| `property_cells_total` | 39 | property cells reported |
| `max_standardized_bias` | 0.0390 | largest primary standardized bias |
| `min_coverage` | 0.9337 | lowest primary coverage |
| `properties[interval_calibration/base_contrast__correctly_specified]:se_ratio` | 0.9957 | SE ratio of the `base` contrast's calibration cell |
| `properties[interval_calibration/base_policy__correctly_specified]:se_ratio` | 0.9989 | SE ratio of the `base` policy's calibration cell |
| `properties[interval_calibration/four_policy__correctly_specified]:se_ratio` | 0.9845 | SE ratio of the `four` policy's calibration cell |
| `properties[interval_calibration/msm_dose_W1__correctly_specified]:se_ratio` | 0.9973 | SE ratio of the working model's $d W_1$ calibration cell |
| `properties[double_robustness/base_contrast__both_wrong]:standardized_bias` | 3.3193 | standardized bias of the both-wrong control |
| `properties[double_robustness/four_contrast__node_hazard]:standardized_bias` | 0.0067 | standardized bias of the node-only hazard cell |
| `properties[targeting_necessity/base_contrast__untargeted]:standardized_bias` | 4.9702 | standardized bias of the plug-in control |
| `properties[simultaneous_coverage/base_curve__simultaneous_band]:coverage` | 0.9413 | joint coverage of the `base` band |
| `properties[simultaneous_coverage/four_curve__simultaneous_band]:coverage` | 0.9477 | joint coverage of the `four` band |
| `properties[iterated_mean_difference/base_contrast__paired_difference]:paired_sd_over_bound` | 0.1749 | paired ratio of the `base` contrast |
| `properties[iterated_mean_difference/four_contrast__paired_difference]:paired_sd_over_bound` | 0.2705 | paired ratio of the `four` contrast |
| `properties[iterated_mean_difference/base__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves, `base` |
| `properties[iterated_mean_difference/four__iterated_curve_decrease]:rejection_rate` | 0 | share of decreasing iterated-mean curves, `four` |
| `properties[one_step_walk/onestep_contrast__one_step_fit]:coverage` | 0.9450 | coverage of the one-step `base` contrast |
| `properties[one_step_walk/onestep_contrast__paired_difference]:paired_sd_over_bound` | 0.0235 | paired ratio of the one-step and per-horizon `base` contrast |
| `margin:confidence_level` | 0.9900 | Monte Carlo confidence level |
| `margin:alpha` | 0.0500 | test size |
| `margin:nominal_coverage` | 0.9500 | nominal interval coverage |
| `margin:bootstrap_replicates` | 10000 | bootstrap replications |
| `margin:standardized_bias` | 0.2500 | standardized-bias margin |
| `margin:coverage_floor` | 0.9000 | primary coverage floor |
| `margin:over_coverage_ceiling` | 0.9900 | descriptive overcoverage threshold |
| `margin:se_ratio_sanity_lower` | 0.8000 | primary SE-ratio lower screen |
| `margin:se_ratio_sanity_upper` | 1.2000 | primary SE-ratio upper screen |
| `margin:calibration_se_ratio_lower` | 0.9300 | calibration SE-ratio lower bound |
| `margin:calibration_se_ratio_upper` | 1.0700 | calibration SE-ratio upper bound |
| `margin:calibration_coverage_lower` | 0.9200 | calibration coverage lower bound |
| `margin:calibration_coverage_upper` | 0.9800 | calibration coverage upper bound |
| `margin:type_i_ceiling` | 0.1000 | type-I upper bound. No cell of this study reads it |
| `margin:paired_difference` | 0.1500 | paired similarity margin in pooled SDs. No cell of this study reads it |
| `margin:rmse_noninferiority` | 1.1000 | RMSE-ratio noninferiority bound. No cell of this study reads it |
| `margin:coverage_noninferiority` | -0.0250 | coverage-difference noninferiority bound. No cell of this study reads it |
| `margin:calibration_noninferiority` | 0.0500 | calibration-excess noninferiority bound. No cell of this study reads it |
| `margin:minimum_power` | 0.8000 | minimum power lower bound |
| `margin:root_n_slope` | -0.5000 | expected root-n slope |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude |
| `margin:union_model_se_lower` | 0.1000 | union-model SE-ratio screen, lower limit |
| `margin:union_model_se_upper` | 10 | union-model SE-ratio screen, upper limit |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 24 | the same bound, in primary replications |
| `margin:efficiency_ratio_lower` | 0.9000 | exact-EIF ratio lower bound |
| `margin:efficiency_ratio_upper` | 1.1000 | exact-EIF ratio upper bound |
| `margin:targeting_displacement` | 0.2500 | minimum targeting displacement |

## Limitations

| limit | what it means for use |
| --- | --- |
| a binary treatment and no covariate after baseline | a treatment with more than two levels is not measured, and a covariate measured after baseline beside a treatment at several nodes is part (b2) of plan X70, refused until then (H2) |
| finite covariates and low-dimensional learners | the evidence covers correct or misspecified logistic learners. A flexible learner is outside it |
| held cells | the laws keep every treatment prefix populated at n = 2,000, and a held draw is a lost replicate. A policy over many nodes on a small sample holds cells and then reports no interval; held cells rest on `tests/unit/test_hazard_multinode.py` |
| the one-step walk on regimen curves | the gated regimen cells use the per-horizon walk. The one-step walk is reported at 200 draws over `always`, `never` and `policy` on `base`, with no verdict, and takes 30 to 160 seconds a fit when a policy is present |
| one cause | the competing-risk path rests on the exact-law tests of `tests/unit/test_hazard_multinode.py` |
| one fold | every fit uses `n_folds=1`; cross-fitted path hazards rest on the unit tests |
| no weights or clusters | sampling weights and the cluster variance rest on the unit tests |
| a node-only witness for one contrast | the `four` policy and the `base` rule, policy and `stay` have no robustness cell whose wrong learner lies far from the truth |
| support rows | the `G62` rows license the in-sample one-cause fits of the three-node and four-node laws under `targeting="iterative"`, with the band, and the working model under `targeting="one_step"`. A one-step regimen fit, a cross-fitted or competing-risk fit, and any other node count or treatment level count match no row and are labelled `experimental` |
| no comparator | no external implementation targets these hazards over treatment paths |

## Reproduction

Run `python tests/canonical/hazard_multinode/regenerate.py --output DIR` with an empty scratch
directory outside the repository to regenerate the artifacts. The pilot runs separately, one
label per run, as `python -m tests.studies.hazard_multinode_pilot --jobs N --labels LABEL`, and
writes `tests/canonical/hazard_multinode/pilot.csv.gz`. The red-cell readings are declared in
`tests/canonical/hazard_multinode/regenerate.py` before any run.
