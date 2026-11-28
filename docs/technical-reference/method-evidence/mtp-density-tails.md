# Modified treatment policy on a point dose, corrected density and classifier ratios

This study validates the point TMLE of a modified treatment policy on a continuous dose through
both of its policy-ratio routes. The density route fits a binned conditional density and reads
the ratio from it; its grid refines the outer bins, reads each row's segment through the bin
masses, splits off isolated doses with gap bins and reads a declared support row by row. The
classifier route, which `ratio=None` resolves to for a policy on a continuous dose, learns the
odds that a row is a policy copy, with the policy's step columns in its design. The longitudinal
default route is read on `longitudinal-mtp`'s continuous law.
[Modified treatment policies](../point-treatment-tmle.md#modified-treatment-policies) states the
estimand, the influence curve and both routes.

No canonical implementation is compared. R `lmtp` fits the classifier ratio with its own
learners and has no binned density route, and `txshift` fits a parametric density, so neither
reads the representation this study tests. The independent evidence is the closed-form laws, the
exact influence curve, and the unit witnesses of `tests/unit/test_density_reconstruction.py` and
`tests/unit/test_classifier_policy_steps.py`.

The declared run passed every gated verdict with no failed fit: 6 of 6 primary rows and the 45
property cells with a verdict, 21 positive cells and 24 controls. The 26 reported rows carry no
verdict, and the reported pilot cells carry the declared readings below.

## The estimand and the laws

For a policy $d$ the target is $\psi = E\,Q(d(A, W), W)$, the mean outcome when each unit's dose
is moved by $d$. With $g$ the conditional density of the dose and $g^d$ that of $d(A, W)$, the
ratio is $r = g^d / g$, and the efficient influence curve is
$D = r\,(Y - Q) + Q(d(A, W), W) - \psi$ (Díaz, Williams, Hoffman and Schenck 2023, arXiv v4,
2006.01366v4, Theorem 2). For any estimate the remainder is
$E_0[(\hat r - r_0)(Q_0 - \hat Q)]$ (Lemma 1, eq. (5)). Identification needs no unmeasured
confounding of $A$ given $W$ and positivity of the policy.

Every law shares $W_1$, $W_2 \in \{0, 1\}$, the shift of one and the outcome
$Y = Q_0(A, W) + N(0, 1)$ with $Q_0(a, w) = 0.5a - 0.1a^2 + w_1 + 0.5w_2 + 0.3aw_1$. Laws G, T, C
and S draw $W_1 \sim N(0, 1)$ and centre the dose at $c(W) = 1 + 0.5W_1 + 0.5W_2$.

| law | dose given $W$ | policy | ratio |
| --- | --- | --- | --- |
| G | $N(c, 1.5^2)$ | $A + 1$ | unbounded |
| T | G truncated to $[-2, 4.5]$ | $A + 1$ while $A + 1 \le 4.5$, otherwise $A$ | bounded |
| C | $(1 - 0.0015)\,N(c, 1.5^2) + 0.0015\,N(20, 1.5^2)$: a few doses far beyond a gap | $A + 1$ | unbounded |
| S | $c + 1.5\,t_5$: heavy tails | $A + 1$ | bounded, on an unbounded dose |
| B | $N(u - 1.6 + 0.4W_1, 1.2^2)$ truncated to $[0, u]$, $u = 6 + 2W_2$, $W_1$ a normal truncated to $[-2.5, 2.5]$ | $A + 1$ below $u - 1$, otherwise $A$ | bounded |
| U | as B with $u = 7 + 0.8W_1$ and mean $u - 1.6 + 0.4W_2$, so every bin near the top straddles some row's support edge | as B | bounded |

On B and U the fit declares `Support(0, u(W))`. Each truth $\psi_0$ is a quadrature of the
closed-form density and outcome regression, and the design test checks the module's values to
$10^{-5}$: G 0.762500, T 0.801801, C 0.710994, S 0.612500, B -0.407387 and U -0.270793. The
efficiency bound $SD(D_0)$ is a 4,000,000-draw Monte Carlo value on G, T, C, B and U and a
quadrature on S. Neither calls the estimator. The laws are defined in
`tests/mtp_density_tails_law.py`, which shares no code with the package.

Theorem 3 of Díaz et al. (arXiv v4) gives asymptotic linearity under the product rate, cross-fitting
and a bounded ratio. It covers T, S, B and U, given a bounded estimate of the ratio on B and U. G
and C have unbounded ratios and are outside it. For them the package's own argument keeps the
linearisation, which needs no bound, and conditions on the representation's rate: with a bounded
second derivative of $\log g(\cdot \mid w)$ where $r_0$ has mass, bounded relative error in the
outer bins and $E_0 r_0^4 < \infty$, the representation contributes $O(n^{-1/4})$. C satisfies the
moment condition. `Scale(1.25)` on a Gaussian dose does not, and its `ic_gap` rows are reported.

## The fits and the cells

| route | fit | laws gated | laws reported |
| --- | --- | --- | --- |
| density | the oracle hazard (each row's exact bin probabilities on the default grid, so the grid and the reading are what is left), the correct outcome regression, in sample | calibration on G, T and C at n = 2,000, 3,200 draws; `ic_gap` on G, T, C, B and U | calibration on S, B and U; `ic_gap` on S and on G under `Scale(1.25)` |
| density, double robustness | the oracle density and the wrong outcome regression (OLS on $[1, A, W_1, W_2]$), n = 8,000 | calibration on G and T, 3,200 draws | |
| classifier | `ratio=None` resolved to the classifier: the default stack, five outer folds, the policy's step columns, the correct outcome regression | calibration on G and T at n = 2,000, 3,200 draws | calibration on S; double robustness on G and T, 400 draws |
| longitudinal default | `ratio=None` at both continuous nodes of `longitudinal-mtp`'s continuous law, the default stack, five outer folds | `ate_regimen[up vs natural]` on `LTMLE` and `SDR`, 3,200 draws | the six other estimands of each |
| cluster warning | `isolated_dose_clusters(A, min_rows=3)` on the doses of the `ic_gap` draws, with no fit | at least 26 of 30 C draws at n = 20,000 warn, and at most 6 of the 220 G and T draws | |

Every calibration arm reads the shared rule: the SE ratio's 99% bootstrap interval inside
[0.93, 1.07] and the coverage interval inside [0.92, 0.98]. Coverage reads the interval the fit
ships, the skew-widened t of a point estimate and the MOVER interval of the longitudinal contrast.
The `ic_gap` cell reads $SD(IC - D_0)/SD(D_0)$ against the exact influence curve at n = 2,000 (80
draws) and n = 20,000 (30 draws). Its 2,000 median and each 20,000 gap must stay under the law's
bound, and the 20,000 median must be at most 0.6 times the 2,000 median.

Each claim has a control that must fail it. The shrunken-SE and noise controls of the density
arm on G, the classifier arm on G and the `LTMLE` arm run on their own 1,000 draws and must fail
the SE-ratio clause. The double-robustness arms have a both-wrong control (a density that ignores
$W_1$) and, on G, the pre-fix one-step reading, each of which must fail coverage. The `ic_gap`
controls are the pre-fix reading (equal-mass bins read as a histogram), the declared support
ignored, and no row-wise break at the support edge. The cluster controls are the helper with its
gap rule disabled and an eager helper.

## Accuracy against known truth

<!-- generated: accuracy -->
| law | estimand | what was tested | implementation | bias (99% interval) | coverage | SE ratio | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a truncated normal dose on [0, u(W)] with u = 6 + 2 W2, declared as Support(0, u(W)); the policy adds one below u - 1 and holds the dose above it; outcome Q0(A, W) + N(0, 1), n = 2,000 | `policy_mean` | the mean outcome under the law's policy | `cleverly` | -0.0056 to 0.0050 | 0.9610 | 1.0112 | pass |
| the Gaussian dose with a 0.15% far component near 20, beyond a gap, under a shift of one; outcome Q0(A, W) + N(0, 1), n = 2,000 | `policy_mean` | the mean outcome under the law's policy | `cleverly` | -0.0041 to 0.0054 | 0.9630 | 0.9881 | pass |
| a Gaussian dose, N(c(W), 1.5^2) given W, under a shift of one; the ratio is unbounded; outcome Q0(A, W) + N(0, 1), n = 2,000 | `policy_mean` | the mean outcome under the law's policy | `cleverly` | -0.0026 to 0.0055 | 0.9430 | 0.9807 | pass |
| the dose c(W) + 1.5 t_5, heavy tailed, under a shift of one; outcome Q0(A, W) + N(0, 1), n = 2,000 | `policy_mean` | the mean outcome under the law's policy | `cleverly` | -0.0056 to 0.0031 | 0.9450 | 0.9790 | pass |
| the Gaussian dose truncated to [-2, 4.5] under a shift of one capped at 4.5; the ratio is bounded; outcome Q0(A, W) + N(0, 1), n = 2,000 | `policy_mean` | the mean outcome under the law's policy | `cleverly` | -0.0060 to 0.0016 | 0.9440 | 1.0287 | pass |
| a truncated normal dose on [0, u(W)] with u = 7 + 0.8 W1, declared as Support(0, u(W)); the policy adds one below u - 1 and holds the dose above it; outcome Q0(A, W) + N(0, 1), n = 2,000 | `policy_mean` | the mean outcome under the law's policy | `cleverly` | -0.0018 to 0.0079 | 0.9380 | 0.9699 | pass |
<!-- /generated -->

## Theory properties

<!-- generated: properties -->
| property | cell | role | what was tested | what must hold | measured | result |
| --- | --- | --- | --- | --- | --- | --- |
| `cluster_warning` | `C__gap_disabled_control` | control | law C at n = 20,000, the helper with its gap rule disabled (every bin read as occupied) | must fail the C clause: fewer than 26 of the 30 draws warn | warned on 0 of 30 draws | pass |
| `cluster_warning` | `C__n_20000` | positive | law C at n = 20,000, the doses of the 30 ic_gap draws | the helper warns on at least 26 of the 30 draws | warned on 30 of 30 draws | pass |
| `cluster_warning` | `GT__eager_control` | control | laws G and T, an eager helper (an absolute gap test with min_rows = 1) | must fail the G and T clause: more than 6 of the 220 draws warn | warned on 34 of 220 draws | pass |
| `cluster_warning` | `GT__pooled` | positive | laws G and T at n = 2,000 and 20,000, the doses of the 220 ic_gap draws | the helper warns on at most 6 of the 220 draws | warned on 0 of 220 draws | pass |
| `ic_gap` | `B__n_2000` | positive | law B, the shipped density reading, 80 draws at n = 2,000 | median gap at most 0.04 | median gap 0.0178, largest gap 0.0506 | pass |
| `ic_gap` | `B__n_20000` | positive | law B, the shipped density reading, 30 draws at n = 20,000 | every gap at most 0.03, and the median at most 0.6 times the 2,000 median | median gap 0.0062, largest gap 0.0101, median over the 2,000 median 0.3457 | pass |
| `ic_gap` | `B__no_row_break_n_2000` | control | law B, the reading with no row-wise break at the support edge, n = 2,000 | must fail the 2,000 bound: median gap above 0.04 | median gap 15.2034, largest gap 18137134.4701 | pass |
| `ic_gap` | `B__pre_fix_n_2000` | control | law B, the pre-fix reading (equal-mass grid read as a histogram), n = 2,000 | must fail the 2,000 bound: median gap above 0.04 | median gap 0.1065, largest gap 0.7795 | pass |
| `ic_gap` | `B__pre_fix_n_20000` | control | law B, the pre-fix reading, n = 20,000 | must fail the 20,000 clause: some gap above 0.03 | median gap 0.0480, largest gap 0.4748, median over the 2,000 median 0.4511 | pass |
| `ic_gap` | `B__support_ignored_n_2000` | diagnostic | law B, the shipped reading with the declared support ignored, n = 2,000 | none. The row is reported | median gap 0.0516, largest gap 0.3949 | reported |
| `ic_gap` | `B__support_ignored_n_20000` | control | law B, the shipped reading with the declared support ignored, n = 20,000 | must fail the 20,000 clause: some gap above 0.03 | median gap 0.0207, largest gap 0.1774, median over the 2,000 median 0.4010 | pass |
| `ic_gap` | `C__n_2000` | positive | law C, the shipped density reading, 80 draws at n = 2,000 | median gap at most 0.1 | median gap 0.0376, largest gap 0.0862 | pass |
| `ic_gap` | `C__n_20000` | positive | law C, the shipped density reading, 30 draws at n = 20,000 | every gap at most 0.1, and the median at most 0.6 times the 2,000 median | median gap 0.0156, largest gap 0.0450, median over the 2,000 median 0.4137 | pass |
| `ic_gap` | `C__pre_fix_n_2000` | control | law C, the pre-fix reading (equal-mass grid read as a histogram), n = 2,000 | must fail the 2,000 bound: median gap above 0.1 | median gap 2.5558, largest gap 4.1731 | pass |
| `ic_gap` | `C__pre_fix_n_20000` | control | law C, the pre-fix reading, n = 20,000 | must fail the 20,000 clause: some gap above 0.1 | median gap 2.5947, largest gap 3.1360, median over the 2,000 median 1.0152 | pass |
| `ic_gap` | `G__n_2000` | positive | law G, the shipped density reading, 80 draws at n = 2,000 | median gap at most 0.1 | median gap 0.0457, largest gap 0.1382 | pass |
| `ic_gap` | `G__n_20000` | positive | law G, the shipped density reading, 30 draws at n = 20,000 | every gap at most 0.1, and the median at most 0.6 times the 2,000 median | median gap 0.0192, largest gap 0.0387, median over the 2,000 median 0.4202 | pass |
| `ic_gap` | `G__pre_fix_n_2000` | control | law G, the pre-fix reading (equal-mass grid read as a histogram), n = 2,000 | must fail the 2,000 bound: median gap above 0.1 | median gap 0.3830, largest gap 0.7575 | pass |
| `ic_gap` | `G__pre_fix_n_20000` | control | law G, the pre-fix reading, n = 20,000 | must fail the fall clause: median above 0.6 times the 2,000 median | median gap 0.4680, largest gap 0.6693, median over the 2,000 median 1.2220 | pass |
| `ic_gap` | `Gscale__n_2000` | diagnostic | law G under Scale(1.25), the shipped density reading, 80 draws at n = 2,000 | none. The row is reported | median gap 0.0584, largest gap 0.3148 | reported |
| `ic_gap` | `Gscale__n_20000` | diagnostic | law G under Scale(1.25), the shipped density reading, 30 draws at n = 20,000 | none. The row is reported | median gap 0.0386, largest gap 0.2614, median over the 2,000 median 0.6612 | reported |
| `ic_gap` | `S__n_2000` | diagnostic | law S, the shipped density reading, 80 draws at n = 2,000 | none. The row is reported | median gap 0.0170, largest gap 0.0324 | reported |
| `ic_gap` | `S__n_20000` | diagnostic | law S, the shipped density reading, 30 draws at n = 20,000 | none. The row is reported | median gap 0.0056, largest gap 0.0071, median over the 2,000 median 0.3282 | reported |
| `ic_gap` | `T__n_2000` | positive | law T, the shipped density reading, 80 draws at n = 2,000 | median gap at most 0.05 | median gap 0.0271, largest gap 0.0695 | pass |
| `ic_gap` | `T__n_20000` | positive | law T, the shipped density reading, 30 draws at n = 20,000 | every gap at most 0.04, and the median at most 0.6 times the 2,000 median | median gap 0.0089, largest gap 0.0132, median over the 2,000 median 0.3294 | pass |
| `ic_gap` | `T__pre_fix_n_2000` | control | law T, the pre-fix reading (equal-mass grid read as a histogram), n = 2,000 | must fail the 2,000 bound: median gap above 0.05 | median gap 0.0730, largest gap 0.1161 | pass |
| `ic_gap` | `T__pre_fix_n_20000` | diagnostic | law T, the pre-fix reading, n = 20,000 | none. The row is reported; T's pre-fix control is read at n = 2,000 only | median gap 0.0383, largest gap 0.0416, median over the 2,000 median 0.5248 | reported |
| `ic_gap` | `U__n_2000` | positive | law U, the shipped density reading, 80 draws at n = 2,000 | median gap at most 0.04 | median gap 0.0189, largest gap 0.0450 | pass |
| `ic_gap` | `U__n_20000` | positive | law U, the shipped density reading, 30 draws at n = 20,000 | every gap at most 0.03, and the median at most 0.6 times the 2,000 median | median gap 0.0058, largest gap 0.0128, median over the 2,000 median 0.3055 | pass |
| `ic_gap` | `U__no_row_break_n_2000` | control | law U, the reading with no row-wise break at the support edge, n = 2,000 | must fail the 2,000 bound: median gap above 0.04 | median gap 64149.0269, largest gap 11491785.9342 | pass |
| `ic_gap` | `U__pre_fix_n_2000` | control | law U, the pre-fix reading (equal-mass grid read as a histogram), n = 2,000 | must fail the 2,000 bound: median gap above 0.04 | median gap 0.3042, largest gap 1.1078 | pass |
| `ic_gap` | `U__pre_fix_n_20000` | control | law U, the pre-fix reading, n = 20,000 | must fail the 20,000 clause: some gap above 0.03 | median gap 0.3371, largest gap 1.1865, median over the 2,000 median 1.1083 | pass |
| `ic_gap` | `U__support_ignored_n_2000` | diagnostic | law U, the shipped reading with the declared support ignored, n = 2,000 | none. The row is reported | median gap 0.1038, largest gap 0.3114 | reported |
| `ic_gap` | `U__support_ignored_n_20000` | control | law U, the shipped reading with the declared support ignored, n = 20,000 | must fail the 20,000 clause: some gap above 0.03 | median gap 0.0803, largest gap 0.7048, median over the 2,000 median 0.7735 | pass |
| `interval_calibration` | `B__reported` | diagnostic | law B, the density route, oracle hazard on the default grid, correct outcome regression, in sample, n = 2,000 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9440 to 0.9634, SE ratio 0.9801 to 1.0443 | reported |
| `interval_calibration` | `C__correctly_specified` | positive | law C, the density route, oracle hazard on the default grid, correct outcome regression, in sample, n = 2,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9399 to 0.9600, SE ratio 0.9476 to 1.0100 | pass |
| `interval_calibration` | `G__correctly_specified` | positive | law G, the density route, oracle hazard on the default grid, correct outcome regression, in sample, n = 2,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9311 to 0.9526, SE ratio 0.9475 to 1.0128 | pass |
| `interval_calibration` | `G__noise_control` | control | law G, the density route, oracle hazard on the default grid, correct outcome regression, in sample, on its own 1,000 draws: a declared scale of independent noise added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.7984 to 0.8604, SE ratio 0.6567 to 0.7403 | pass |
| `interval_calibration` | `G__shrunken_se_control` | control | law G, the density route, oracle hazard on the default grid, correct outcome regression, in sample, on its own 1,000 draws: each SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.7941 to 0.8567, SE ratio 0.6614 to 0.7395 | pass |
| `interval_calibration` | `G_classifier__correctly_specified` | positive | law G, the classifier route ratio=None resolves to: the default stack, five outer folds, the policy's step columns, correct outcome regression, n = 2,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9389 to 0.9591, SE ratio 0.9629 to 1.0266 | pass |
| `interval_calibration` | `G_classifier__noise_control` | control | law G, the classifier route ratio=None resolves to: the default stack, five outer folds, the policy's step columns, correct outcome regression, on its own 1,000 draws: a declared scale of independent noise added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8102 to 0.8705, SE ratio 0.6794 to 0.7622 | pass |
| `interval_calibration` | `G_classifier__shrunken_se_control` | control | law G, the classifier route ratio=None resolves to: the default stack, five outer folds, the policy's step columns, correct outcome regression, on its own 1,000 draws: each SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8091 to 0.8696, SE ratio 0.6895 to 0.7682 | pass |
| `interval_calibration` | `G_classifier_dr__reported` | diagnostic | law G, the classifier route with the wrong outcome regression (OLS on [1, A, W1, W2]), n = 2,000 | none. Bias, coverage and the mean remainder proxy are reported against the declared reading | coverage 0.9149 to 0.9738, SE ratio 0.9320 to 1.1258, mean remainder proxy -0.0023 SE | reported |
| `interval_calibration` | `G_classifier_dr_both_wrong__reported` | diagnostic | law G, the wrong outcome regression and a classifier that omits W1, n = 2,000 | none. Bias, coverage and the mean remainder proxy are reported | coverage 0.7438 to 0.8489, SE ratio 0.9767 to 1.1742, mean remainder proxy 1.2546 SE | reported |
| `interval_calibration` | `G_dr__both_wrong_control` | control | law G, the wrong outcome regression and a density that ignores W1, n = 8,000 | the coverage interval must fall below the calibration band | coverage 0.4479 to 0.5302, SE ratio 0.9955 to 1.1123 | pass |
| `interval_calibration` | `G_dr__pre_fix_control` | control | law G, the pre-fix one-step reading with the wrong outcome regression, n = 8,000 | the coverage interval must fall below the calibration band | coverage 0.5927 to 0.6720, SE ratio 0.8422 to 0.9448 | pass |
| `interval_calibration` | `G_dr__wrong_outcome` | positive | law G, the oracle density and the wrong outcome regression (OLS on [1, A, W1, W2]), n = 8,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9284 to 0.9503, SE ratio 0.9488 to 1.0122 | pass |
| `interval_calibration` | `LTMLE_ate_scale_at_2_vs_natural__reported` | diagnostic | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: multiply the second dose by 1.25, capped at 5.5, minus the natural course | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9304 to 0.9520, SE ratio 0.9423 to 1.0047 | reported |
| `interval_calibration` | `LTMLE_ate_up_then_history_vs_natural__reported` | diagnostic | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3, minus the natural course | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9368 to 0.9574, SE ratio 0.9754 to 1.0403 | reported |
| `interval_calibration` | `LTMLE_ate_up_vs_natural__correctly_specified` | positive | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: add 0.5 to the dose at both times, capped at 5.5, minus the natural course | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9403 to 0.9603, SE ratio 0.9902 to 1.0574 | pass |
| `interval_calibration` | `LTMLE_ate_up_vs_natural__noise_control` | control | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: add 0.5 to the dose at both times, capped at 5.5, minus the natural course, on its own 1,000 draws: a declared scale of independent noise added to each estimate | the SE-ratio interval must fall below the calibration band | coverage 0.8230 to 0.8815, SE ratio 0.6996 to 0.7843 | pass |
| `interval_calibration` | `LTMLE_ate_up_vs_natural__shrunken_se_control` | control | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: add 0.5 to the dose at both times, capped at 5.5, minus the natural course, on its own 1,000 draws: each SE multiplied by 0.70 | the SE-ratio interval must fall below the calibration band | coverage 0.8059 to 0.8668, SE ratio 0.7003 to 0.7823 | pass |
| `interval_calibration` | `LTMLE_ey_natural__reported` | diagnostic | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under the natural course | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9454 to 0.9645, SE ratio 0.9879 to 1.0553 | reported |
| `interval_calibration` | `LTMLE_ey_scale_at_2__reported` | diagnostic | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under multiply the second dose by 1.25, capped at 5.5 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9396 to 0.9597, SE ratio 0.9715 to 1.0336 | reported |
| `interval_calibration` | `LTMLE_ey_up__reported` | diagnostic | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under add 0.5 to the dose at both times, capped at 5.5 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9495 to 0.9678, SE ratio 1.0094 to 1.0778 | reported |
| `interval_calibration` | `LTMLE_ey_up_then_history__reported` | diagnostic | longitudinal-mtp's continuous law, LTMLE on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9475 to 0.9662, SE ratio 1.0018 to 1.0704 | reported |
| `interval_calibration` | `SDR_ate_scale_at_2_vs_natural__reported` | diagnostic | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: multiply the second dose by 1.25, capped at 5.5, minus the natural course | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9216 to 0.9446, SE ratio 0.9260 to 0.9874 | reported |
| `interval_calibration` | `SDR_ate_up_then_history_vs_natural__reported` | diagnostic | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3, minus the natural course | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9324 to 0.9537, SE ratio 0.9597 to 1.0258 | reported |
| `interval_calibration` | `SDR_ate_up_vs_natural__correctly_specified` | positive | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: add 0.5 to the dose at both times, capped at 5.5, minus the natural course | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9433 to 0.9628, SE ratio 0.9842 to 1.0513 | pass |
| `interval_calibration` | `SDR_ey_natural__reported` | diagnostic | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under the natural course | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9345 to 0.9554, SE ratio 0.9539 to 1.0157 | reported |
| `interval_calibration` | `SDR_ey_scale_at_2__reported` | diagnostic | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under multiply the second dose by 1.25, capped at 5.5 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9368 to 0.9574, SE ratio 0.9409 to 1.0029 | reported |
| `interval_calibration` | `SDR_ey_up__reported` | diagnostic | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under add 0.5 to the dose at both times, capped at 5.5 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9471 to 0.9659, SE ratio 0.9843 to 1.0492 | reported |
| `interval_calibration` | `SDR_ey_up_then_history__reported` | diagnostic | longitudinal-mtp's continuous law, SDR on the default route at both continuous nodes (the classifier, default stack, five outer folds, per-node steps), n = 2,000: the mean under add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9392 to 0.9594, SE ratio 0.9761 to 1.0412 | reported |
| `interval_calibration` | `S__reported` | diagnostic | law S, the density route, oracle hazard on the default grid, correct outcome regression, in sample, n = 2,000 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9406 to 0.9605, SE ratio 0.9651 to 1.0300 | reported |
| `interval_calibration` | `S_classifier__reported` | diagnostic | law S, the classifier route ratio=None resolves to: the default stack, five outer folds, the policy's step columns, correct outcome regression, n = 2,000; S's influence curve has an infinite fourth moment | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9341 to 0.9552, SE ratio 0.9503 to 1.0230 | reported |
| `interval_calibration` | `T__correctly_specified` | positive | law T, the density route, oracle hazard on the default grid, correct outcome regression, in sample, n = 2,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9416 to 0.9614, SE ratio 0.9716 to 1.0369 | pass |
| `interval_calibration` | `T_classifier__correctly_specified` | positive | law T, the classifier route ratio=None resolves to: the default stack, five outer folds, the policy's step columns, correct outcome regression, n = 2,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9426 to 0.9622, SE ratio 0.9786 to 1.0441 | pass |
| `interval_calibration` | `T_classifier_dr__reported` | diagnostic | law T, the classifier route with the wrong outcome regression (OLS on [1, A, W1, W2]), n = 2,000 | none. Bias, coverage and the mean remainder proxy are reported against the declared reading | coverage 0.9212 to 0.9774, SE ratio 0.9419 to 1.1097, mean remainder proxy -0.2055 SE | reported |
| `interval_calibration` | `T_dr__both_wrong_control` | control | law T, the wrong outcome regression and a density that ignores W1, n = 8,000 | the coverage interval must fall below the calibration band | coverage 0.5249 to 0.6065, SE ratio 0.9771 to 1.0893 | pass |
| `interval_calibration` | `T_dr__wrong_outcome` | positive | law T, the oracle density and the wrong outcome regression (OLS on [1, A, W1, W2]), n = 8,000 | SE ratio and coverage intervals both inside their calibration bands | coverage 0.9457 to 0.9648, SE ratio 0.9779 to 1.0454 | pass |
| `interval_calibration` | `U__reported` | diagnostic | law U, the density route, oracle hazard on the default grid, correct outcome regression, in sample, n = 2,000 | none. The row is reported with its SE ratio and coverage; one outside the band is diagnosed on the page | coverage 0.9447 to 0.9639, SE ratio 0.9909 to 1.0536 | reported |
<!-- /generated -->

## Reported readings

The pilot cells carry no verdict. `reported_readings` in `tests/studies/mtp_density_tails_pilot.py`
computes each declared reading from the committed `pilot.csv.gz` and `properties.csv`, and a fast
test compares the result with the committed `readings.json`, from which every number in this
section comes.

**The default booster on the density route** (cross-fitted, at n = 2,000 on 20 draws per law and at
n = 20,000 on 10 draws on G, T, C and S). On G and C the median ratio error at 20,000 is
0.27 and 0.28 times its 2,000 value, past the declared fall of a third. The 20,000 tail reading
holds on every law. The 2,000 tail reading (at most 2 of 20 draws above 1.0 and none above 3.0)
holds on G, C, S and B and breaches on T (3 draws above 1.0, one of them 5.14) and on U (2 above
1.0, one of them 6.30). B was declared at risk and held; U was not declared at risk and breached.

All 8 draws above 1.0 carry the three-mark misfit signature, so each routes as a learner
limitation: the exact bin probabilities on the same grid read a ratio error of at most 0.15, the
five worst rows carry at least half the squared error, and at the worst row the booster's log
probability of a bin is off by at least 2. Every law's count sits under its frozen ceiling (G 2 of
3, T 3 of 6, B 1 of 8 and U 2 of 4), so no grid choice reopens. The default stacked density reads
no draw above 1.0 on 5 draws per law on G, T, C and S.

**B and U with learned hazards and the declared support.** With the support the median ratio
error is below the pre-fix reading's and no more than 10% above the support-ignored reading's,
for both learners on both laws: boosting 0.290 against 0.425 (pre-fix) and 0.359 (support
ignored) on B, and 0.366 against 0.997 and 0.466 on U; the pooled logistic 0.209 against 0.569
and 0.238 on B, and 0.275 against 0.486 and 0.309 on U. Their boosting cells read no draw above
1.0.

**The classifier's step columns.** On T the steps let the median ratio error fall to 0.588 times
its 2,000 median at 20,000, and without them it stalls at 1.084 times. On the capped `Gcap` and
the two-piece `Bp2` policies the steps cut the 2,000 median to 0.453 and 0.642 times. On G, C and S
at 2,000 they change it by a factor of 0.958, 1.084 and 1.009, inside the declared 10%. Every steps
reading holds.

**The classifier's double-robustness cells** (400 draws, the wrong outcome regression). On G the
mean remainder proxy $\sqrt{n}\,E_n[(\hat r - r_0)(Q_0 - Q_{ols})]/SD(D_0)$ is -0.0023 SE, inside
the declared 0.2 SE, with bias 0.030 SE and coverage 0.950. On T it is negative, -0.206 SE, as
declared, with bias -0.178 SE and coverage 0.955; its size is below the 0.3 to 0.5 SE the probe
suggested. The both-wrong contrast on G, a classifier that omits $W_1$, has bias 1.30 SE and
coverage 0.800. With a learned ratio and a wrong outcome regression the route claims consistency
and not calibrated intervals, because the remainder is then linear in the ratio error.

**The reported calibration rows.** The density route on S, B and U and the classifier route on S
read inside the band. Three of the twelve reported longitudinal estimands read just outside it,
each on one side of one clause, while the gated `ate_regimen[up vs natural]` passes on both
estimators:

- `LTMLE` `ey_regimen[up]`: SE ratio 1.043, 99% interval 1.009 to 1.078, coverage 0.959. The mean
  reported SE is that of `SDR` on the same draws (0.01485 against 0.01486), while the `LTMLE`
  estimates spread less (0.01424 against 0.01462). The interval is conservative, and its upper
  SE-ratio end passes 1.07 by 0.008.
- `LTMLE` `ey_regimen[up then history]`: SE ratio 1.035, interval 1.002 to 1.070, coverage
  0.958. The same pattern: the reported SE matches `SDR`'s (0.01406 against 0.01407), and the
  `LTMLE` spread is smaller (0.01358 against 0.01396).
- `SDR` `ate_regimen[scale at 2 vs natural]`: SE ratio 0.955, interval 0.926 to 0.987, coverage
  0.934. The reported SE matches `LTMLE`'s (0.00858 against 0.00856), and the `SDR` estimates
  spread more (0.00898 against 0.00881), with a bias of 0.12 empirical SD. Both `scale at 2` means
  carry a positive bias on both estimators (0.0010 and 0.0015), and coverage stays inside its
  band.

The two `up` rows are a difference of a few percent between two estimators' empirical spread
under a common standard error. The `SDR` scale row's coverage interval excludes 0.95, and the
`LTMLE` scale contrast reads a smaller shortfall in the same direction (SE ratio 0.972, coverage
0.942). Under the declared rule a reported row outside the band is diagnosed and reported, not a
red row, and none is rerun.

**The scale-policy interval.** Diagnostic runs outside the registered study (400 draws per
configuration) trace the shortfall to the default classifier stack's ratio, not to the
influence-curve formula. Splitting each registered estimate's error into the exact first-order
term and a remainder, on `scale at 2` the remainder carries 13% of the estimates' variance, runs
against the first-order term, and carries the bias. With the correctly specified classifier and
the registered outcome learners at n = 2,000, the remainder falls to 2% of the variance and the
fitted curve's variance slightly exceeds the efficient influence function's (ratio 1.03, SE
0.009; SE ratio 1.05).

With correct outcome regressions and the default stack, the fitted
curve's variance stayed near 0.8 of the efficient influence function's at n = 2,000 and 8,000
(0.80 and 0.81). The SE ratio fell from 0.960 to 0.894, with coverage 0.928 at 8,000, as the
remainder that offset the low curve at 2,000 shrank. At 8,000 the estimates' spread is compatible
with the efficiency bound (1.01, SE 0.07), and a remainder of 7% of the variance remains. In the
dose band just below the scale policy's cap, where its ratio rises steeply, the stack's fitted
ratio is shrunk toward one. Estimates on this route keep the `experimental` label.

## Measured values and declared margins

Names beginning `margin:` are thresholds declared before the run. Everything else is measured
from the committed results and checked at the precision printed.

| quantity | value | source |
| --- | --- | --- |
| `replicates` | 1000 | primary replications |
| `n` | 2000 | observations per replication |
| `independent_tests_passed` | 6 | truth tests passing |
| `independent_tests_total` | 6 | truth tests reported |
| `property_cells_passed` | 45 | property cells passing |
| `property_cells_total` | 45 | property cells reported |
| `max_standardized_bias` | 0.0514 | largest primary standardized bias |
| `min_coverage` | 0.9380 | lowest primary coverage |
| `properties[interval_calibration/G__correctly_specified]:se_ratio` | 0.9785 | SE ratio of the density route on G |
| `properties[interval_calibration/C__correctly_specified]:se_ratio` | 0.9783 | SE ratio of the density route on C |
| `properties[interval_calibration/G_dr__wrong_outcome]:coverage` | 0.9400 | coverage of the density double-robustness arm on G |
| `properties[interval_calibration/G_classifier__correctly_specified]:se_ratio` | 0.9934 | SE ratio of the classifier route on G |
| `properties[interval_calibration/T_classifier__correctly_specified]:se_ratio` | 1.0099 | SE ratio of the classifier route on T |
| `properties[interval_calibration/LTMLE_ate_up_vs_natural__correctly_specified]:se_ratio` | 1.0224 | SE ratio of the gated `LTMLE` contrast |
| `properties[interval_calibration/SDR_ate_up_vs_natural__correctly_specified]:se_ratio` | 1.0160 | SE ratio of the gated `SDR` contrast |
| `properties[interval_calibration/T_classifier_dr__reported]:remainder_proxy` | -0.2055 | mean remainder proxy of the classifier double-robustness cell on T, in SE |
| `properties[ic_gap/G__n_2000]:median_gap` | 0.0457 | median influence-curve gap on G at n = 2,000 |
| `properties[ic_gap/G__n_20000]:gap_fall` | 0.4202 | the 20,000 median over the 2,000 median on G |
| `properties[ic_gap/C__n_20000]:max_gap` | 0.0450 | largest influence-curve gap on C at n = 20,000 |
| `properties[ic_gap/G__pre_fix_n_2000]:median_gap` | 0.3830 | median gap of the pre-fix reading on G at n = 2,000 |
| `properties[cluster_warning/C__n_20000]:warned_draws` | 30 | C draws at n = 20,000 on which the warning fires |
| `properties[cluster_warning/GT__pooled]:warned_draws` | 0 | G and T draws on which the warning fires |
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
| `margin:minimum_power` | 0.8000 | minimum power lower bound. No cell of this study reads it |
| `margin:root_n_slope` | -0.5000 | expected root-n slope. No cell of this study reads it |
| `margin:root_n_slope_lower` | -0.6250 | root-n slope lower bound. No cell of this study reads it |
| `margin:root_n_slope_upper` | -0.3750 | root-n slope upper bound. No cell of this study reads it |
| `margin:excluded_slope` | -0.2500 | slower rate the interval must exclude. No cell of this study reads it |
| `margin:shrunken_se_factor` | 0.7000 | negative-control SE multiplier |
| `margin:subject_failure_bound` | 0.0100 | declared share of replications a cell may lose to a recordable fitting failure |
| `margin:subject_failures_allowed` | 10 | the same bound, in primary replications |

## Limitations

| limit | what it means for use |
| --- | --- |
| oracle hazard on the gated density cells | the gated density arms read the grid and the reading with each row's exact bin probabilities. A learned hazard's error adds to them; the booster and stacked cells report it with no verdict |
| learned ratios with a wrong outcome regression | the classifier route's double-robustness cells are reported. With a learned ratio and a wrong outcome regression the route claims consistency, not calibrated intervals |
| the default booster's tails | at n = 2,000 the booster read a ratio error above 1.0 on 1 to 3 of 20 draws on G, T, B and U, each a few-row misfit of the learner. Pass a learner suited to the dose where its tail matters |
| a contaminated dose on the classifier route | the classifier's calibration is not gated on C. The cluster warning gates the advice to pin `ratio="density"` there |
| heavy tails | the classifier's calibration on S is reported, because S's influence curve has an infinite fourth moment and the SE ratio spreads beyond the normal sizing at any feasible size |
| the conditional rate on unbounded ratios | G and C rest on the package's own argument, which conditions on the representation's rate. No learner's rate is claimed |
| one decision and continuous outcomes | the point laws have one dose and a Gaussian outcome noise; the longitudinal route is read on one law with two nodes |
| no comparator | no implementation reads the binned density this study tests, so the evidence rests on the exact laws and the unit witnesses |

## Reproduction

Run `python tests/canonical/mtp_density_tails/regenerate.py --output DIR` with an empty scratch
directory outside the repository to regenerate the artifacts; the heavy streams run first as
the parts its docstring names. The pilot runs separately as
`python -m tests.studies.mtp_density_tails_pilot`, and `--readings` writes
`tests/canonical/mtp_density_tails/readings.json`. The red-cell readings are declared in
`tests/canonical/mtp_density_tails/regenerate.py` before any run.
