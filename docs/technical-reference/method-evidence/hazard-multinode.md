---
orphan: true
---

# Treatment at several nodes on the hazard path

This study validates longitudinal TMLE with `survival_model="hazard"` when a binary treatment is
decided again at every node. The slots of the walk are full treatment paths: a regimen
contributes each path it can follow, a policy node enumerates both arms with its own
probabilities, and the copies of one path prefix share one update. The risk of a regimen is the
plug-in sum over its paths of the path mass times the targeted path risk. The section on
[a treatment at several nodes on the hazard path](../longitudinal-tmle.md#a-treatment-at-several-nodes-on-the-hazard-path)
states the estimand, the influence curve and the exact-law evidence.

The study is declared and has not run. The page is not linked from the index until the run
commits its artifacts and the record is registered, so the three result blocks below are empty.

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
| `five` | 5 | close to balanced: 0.47 to 0.55 | 0.04 to 0.09 at node 1, rising to 0.40 to 0.66 at node 5, with smaller arm effects | 0.018 |

`base` is the law of plan X70 section 4.3. `five` has the same form with its own coefficients.
The shipped fit holds the cell of the units at risk with one treatment prefix at zero when none
of them has the event, or at one when all of them do. A regimen that reads a held cell reports
no interval. The policy reads all 32 prefixes at node 5. With the `base` hazard at five nodes,
11 to 14 of 15 scratch draws of 2,000 rows held a cell, against an expected 0.002 held cells per
fit on `five`.

Both laws keep every path's cumulative treatment and retention probability ten times above the
declared bound of $10^{-3}$. The shipped default bound of 0.01 lies within a factor of two of
the smallest, so the study declares its own.

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
| `five` | `five` | `always`, `never`, `policy`; reference `never` | per horizon |
| `msm` | `base` | the eight static paths, under the logit working model $\beta_0 + \beta_1 d + \beta_2 t + \beta_3 d W_1$, with $d$ the path's cumulative dose through $t$ | one step (`targeting="one_step"`) |

| nuisance | correct learner | wrong learner |
| --- | --- | --- |
| hazard | unpenalized logistic regression on `W1`, `W2`, `A1`, ..., `Ak` at each node | logistic regression on `Ak` alone (`LastArmLogistic`) |
| treatment | unpenalized logistic regression on `W1`, `W2`, `A1`, ..., `A(k-1)` at each node | the prior share |
| censoring | unpenalized logistic regression on `W1`, `W2`, `A1`, ..., `Ak` at each node | the prior share |

Each factor of both laws depends on $W$ and on the last one or two arms, so the correct learners
are correctly specified. They hold no interaction, so they are not saturated and targeting
moves every fit. Every `base` and `five` fit raises `SparseInfluenceWarning`: the risk of a static
arm by the last node reads its influence curve from the few units that followed the arm, each
weighted by the inverse of a cumulative probability as small as 0.02. The study declares the
warning as expected on those fit labels, and the calibration rows read the resulting intervals.

The regimen fits use the per-horizon walk, the shipped default, which converges in a few
iterations and about 1.5 seconds a fit at n = 2,000. On one draw of 2,000 rows the one-step walk
took 250 seconds over the five `base` regimens, and 10,776 steps (160 seconds) over `always`,
`never`, `rule` and `policy`, against 166 steps over `always` and `stay`. The rule follows only
the paths of `always` and `never`, so its targets lie close to theirs. The working model needs
the one-step walk, the only algorithm that targets its scores, and that fit converges in about
a second.

No maintained package targets discrete-time hazards over treatment paths, so the study names no
comparator and its agreement block stays empty.

## Accuracy against known truth

<!-- generated: accuracy -->
<!-- /generated -->

The primary rows draw samples of 2,000 rows for each of `base`, `five` and `msm` and report every
risk and contrast of each regimen fit and every coefficient of the working model. The shared
primary rule (`repeated_sizing.primary_replicates`) sets the count from the committed
calibration pilots of these fit labels. Until the pilots are committed the count is the
provisional 2,400.

## Agreement with the canonical implementation

<!-- generated: agreement -->
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
<!-- /generated -->

| family | cells | replicates and n | what fails it |
| --- | --- | --- | --- |
| `interval_calibration` | on `base` the `always` minus `never` contrast at node 3 and the risks of `rule`, `policy` and `stay`; on `five` the contrast and the policy's risk at node 5; on `msm` the coefficients of $d$ and $d W_1$. The shrunken-SE and noise controls on the `base` contrast | 4,000 at n = 2,000 (provisional) | an SE ratio or coverage interval outside its band, or an efficiency ratio outside 0.90 to 1.10 |
| `simultaneous_coverage` | the band over every estimate of the `base` and `five` fits, and its pointwise control | as calibration | joint coverage outside the calibration band |
| `double_robustness` | the `base` contrast with both, one or neither nuisance correct; every other calibration arm with the wrong hazard learner and the correct mechanism | 1,200 at n = 2,000, or the wrong-learner pilot's count | a positive bias interval outside the margin, or a `both_wrong` interval inside it |
| `targeting_necessity` | the `base` contrast from the wrong hazard learner, targeted and plug-in | 1,200 at n = 2,000, or the wrong-learner pilot's count | the targeted bias outside the margin, or a displacement below 0.25 |
| `root_n_and_efficiency`, `root_n_rate` | the `base` contrast and the policy's risk at n = 1,000, 2,000 and 8,000; the n = 1,000 rung is a rate rung that carries no claim | 800 at each n | the shared root-n rules at n = 2,000 and 8,000, and the rate across all three |
| `power` | the `base` contrast against zero | 800 at n = 2,000 | a rejection lower bound below the minimum power |
| every family | `nonconverged_fits`: the fits that stopped short of `tol` with a final relative score at or below $10^{-6}$, beside each cell's count of unexpected warnings | as the cell | a non-converged fit above $10^{-6}$ is a lost replicate under the 0.01 failure bound; more than 1% non-converged fits makes the cell a defect suspect |
| `iterated_mean_difference` | on `base` and `five`: the hazard estimate minus the iterated-mean estimate of the same draw, without and with `isotonic=True`, each fit against the truth, and the largest decrease of an iterated-mean curve. On the `base` `mechanism_correct` draws: the same paired difference with both regressions omitting $W$ | 800 at n = 2,000, and the robustness draws | nothing: the family is reported with no verdict |

The calibration count is the arm-pooled SE-ratio rule of `mtp_study_tools.smallest_replicates`
over the eight arms. At a pilot of 8,000 draws whose SE ratio is one it gives 4,000, and the
committed calibration pilot sets it. At 4,000 replicates the shrunken-SE control leaves the
SE-ratio band and the noise control leaves the efficiency band with probability above 0.999.
Each `mechanism_correct` fit set and the necessity fit set take the rule of plan X70 rev 2.7
from the wrong-learner pilot of 800 draws per fit label: the larger of 1,200 and the smallest
multiple of 400 that passes with probability 0.99 at the pilot's standardized bias plus one of
its standard errors, with no cap. A need above 4,000 stops the declaration for a diagnosis.

The wrong hazard learner's plug-in `base` contrast has an exact limit 3.54 efficiency standard
errors from the truth at n = 2,000, so the plug-in arm of `targeting_necessity` can fail. With
both nuisances wrong the exact `base` limit lies 3.07 efficiency standard errors from the truth,
so the `both_wrong` control can fail. If it does not clear the margin, the control has failed and
the study is a defect suspect. Without $W$ the wrong learner's plug-in coefficient of $d W_1$ is
zero in every sample, 3.49 standard errors from the truth, so its robustness cell reads the
targeting alone.

The power cell's exact-law power at n = 2,000 is above 0.99. At bias zero every positive bias
arm passes with probability above 0.99. That stays true while the true bias is below 0.108
standard deviations at 1,200 replicates and below 0.077 at 800.

With correctly specified parametric learners the hazard fit and the iterated-mean fit are both
efficient, so they differ by $o_P(n^{-1/2})$. The study reports the committed pilot's ratios and
the decreasing share with its interval, with no floor: X70 reports the agreement of the two fits
instead of requiring a gap (the owner's decision of 2026-11-13).

## Limitations

| limit | what it means for use |
| --- | --- |
| a binary treatment and no covariate after baseline | a covariate measured after baseline is part (b) of plan X70 |
| finite covariates and low-dimensional learners | the evidence covers correct or misspecified logistic learners. A flexible learner is outside it |
| held cells | the laws keep every treatment prefix populated at n = 2,000. A policy over many nodes on a small sample holds cells and then reports no interval; held cells rest on `tests/unit/test_hazard_multinode.py` |
| the one-step walk on several regimens | the regimen fits use the per-horizon walk. The one-step walk reaches the same targets, but slowly when regimens share paths |
| no comparator | no external implementation targets these hazards over treatment paths |

## Reproduction

Run `python tests/canonical/hazard_multinode/regenerate.py --output DIR` with an empty scratch
directory outside the repository to regenerate the artifacts. The pilot runs separately, one
label per run, as `python -m tests.studies.hazard_multinode_pilot --jobs N --labels LABEL`, and
writes `tests/canonical/hazard_multinode/pilot.csv.gz`. The red-cell readings are declared in
`tests/canonical/hazard_multinode/regenerate.py` before any run.
