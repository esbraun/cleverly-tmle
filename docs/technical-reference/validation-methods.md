# Sensitivity and validation methods

This page accounts for every instrument `cleverly` gives you to review a method. Each entry answers
three questions in the same order. Why do you use it? What does it tell you? How does it tell you
that?

The instruments fall into four layers, and the layers see different mistakes. Read them in order.
A layer does not replace the layer above it.

| layer | the question it answers | what it cannot answer |
| --- | --- | --- |
| [Diagnostics on the fit you have](#diagnostics-on-the-fit-you-have) | did this fit, on this sample, do what the estimator asked of it? | whether the estimator asked for the right thing |
| [Sensitivity to untestable assumptions](#sensitivity-to-untestable-assumptions) | how wrong would an assumption have to be to change the conclusion? | whether the assumption is in fact wrong |
| [Refutation and simulation you run](#refutation-and-simulation-you-run) | does the fitted workflow behave as it must under a known answer? | anything the law you simulated does not contain |
| [How the library certifies itself](#how-the-library-certifies-itself) | is the derivation this library implements the correct one? | how your own data behaves |

## Diagnostics on the fit you have

These read the artifacts a completed fit already holds. They are cheap, and they run without
refitting a nuisance model.

### Positivity and overlap

**Why.** Every clever covariate for an intervention on treatment divides by an estimated density.
The observed-mean estimand is the exception, and it intervenes on nothing. A small denominator makes
one row dominate the estimate, and targeting does not restore missing support.

**What it tells you.** How much of the estimate rests on how few rows, and how much of the
mechanism the truncation bound replaced.

**How.** `result.diagnostics.support()` returns a `PositivityReport` from
[`sensitivity/positivity.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/positivity.py).
It reports five separate quantities, because they fail in different places.

| quantity | how it is computed | what a bad value means |
| --- | --- | --- |
| effective sample size | Kish's $(\sum \omega)^2 / \sum \omega^2$ over the clever-covariate weights, folded with the observation weights | the interval is that of a much smaller study |
| weight concentration | the share of the estimating equation carried by the top 1% of rows | a handful of rows decide the answer |
| truncation load | the count of clipped propensities, and how far each one moved | the estimate is partly the bound rather than the data |
| per-arm overlap | the mechanism's predicted probability distribution, arm by arm | one arm has a region the other never enters |
| maximum clever covariate | the largest absolute covariate value | the leverage of the single worst row |

The report is per arm. A multi-arm fit reads its arms from the parameter's structured index rather
than assuming two.

### Truncation stability

**Why.** A truncation bound is a finite-sample choice. A conclusion that survives only at one bound
is a conclusion about the bound.

**What it tells you.** How far the estimate moves as the bound moves.

**How.** `result.diagnostics.truncation_curve()` sweeps the `g_bounds` level and **retargets** the
cached nuisances at each level through `TMLE.retarget`. It refits no nuisance model, so it is a
retarget operation rather than a refit operation. `LTMLE` refuses it: `g_bounds` enters the
pseudo-outcome of every earlier node through the backward recursion, so changing it changes what
the earlier regressions were fitted to, and the whole pass has to run again.

### Nuisance model quality

**Why.** A nuisance model can predict well and remain miscalibrated. The clever covariate divides by
the predicted probability itself, so a miscalibrated fit moves every weight.

**What it tells you.** Whether each nuisance fit is calibrated out of fold, and which library
candidates the Super Learner actually used.

**How.** `result.diagnostics.nuisance_models()` returns `NuisanceDiagnostics` from
[`validation/nuisance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/nuisance.py):
out-of-fold propensity AUC, a calibration slope from a logistic recalibration of the out-of-fold
predictions, a calibration table, outcome $R^2$ or Brier score, and the Super Learner candidate
weights.

Read the propensity AUC as a positivity signal and not as a score. A higher AUC means the treatment
is more predictable, which means the arms overlap less. Higher is not better here.

### Score equations

**Why.** TMLE is defined by the equation its fluctuation solves. A fit that stopped early solves it
approximately, and the reported influence curve is then not mean zero.

**What it tells you.** Whether the fluctuation reached the root of the equation the library posed.

**How.** `result.diagnostics.score_equations()` recomputes $P_n \hat{D}^*(O)$ from the fitted
influence curves and compares it against a tolerance scaled to the score's own units
([`validation/score.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/score.py)).
A point-treatment fit compares the score in the outcome's units against
`tolerance * se / sqrt(n)`. A longitudinal fit bounds each node's relative score.

**What it does not tell you.** `score_check` is necessary and not sufficient. A clever covariate
that is wrong in the same way in both the targeting step and the reported curve solves its own
equation exactly. That the equation is the right one is a claim about the library, and
[how the library certifies itself](#how-the-library-certifies-itself) is where that claim is
tested.

### Correction identities

**Why.** `DRTMLE` reports a curve assembled from three separate score equations. The curve is only
mean zero if each correction the curve subtracts is the correction whose equation the fit solved.

**What it tells you.** Two different failures, told apart. An **identity residual** means the
software solved one expression and reported another. A **correction score** means the fit did not
converge.

**How.** `result.diagnostics.corrections()` recomputes each correction's empirical mean from the
exact returned state and compares it with the score the solver recorded
([`validation/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/drtmle.py)).
It reports the two residuals and a clipping bias term separately.

This instrument found a real defect. One clipped row in six hundred left the reported curve
uncentred at `2e-04` while all three fluctuation rows reported `1e-11`.

### Intervention support

**Why.** Arm positivity, regime support, shift support, and incremental support are four different
questions. One propensity histogram answers none of them.

**What it tells you.** Whether the declared intervention puts mass where the data has none.

**How.** Each intervention class exposes its own report through
[`interventions/support.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/support.py).
A shift fit asks whether the density *ratio* stays bounded. A per-arm propensity table has no rows
on a continuous dose, so `diagnostics.support()` dispatches to the question that applies rather
than returning an empty table.

### Design weights

**Why.** Observation weights tilt the population. Their cost to precision is separate from the
clever covariate's cost, and adding the two together hides both.

**What it tells you.** The effective sample size the declared weights leave, before any positivity
cost.

**How.** `WeightReport` in
[`data/weighting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/data/weighting.py)
reports the weighted effective sample size and warns when the weights concentrate.

### The status contract

Every assessment row returns one of six states, and the states are part of the contract rather than a
presentation choice.

| status | what it means |
| --- | --- |
| `passed` | the check ran and its condition holds |
| `failed` | the check ran and its condition does not hold |
| `warning` | the check ran and an explicit diagnostic rule requires qualification |
| `completed` | a descriptive analysis ran and defines no pass or fail rule |
| `not_applicable` | no such analysis exists for this scientific question |
| `unavailable` | the analysis is meaningful, and a derivation or a fitted artifact is missing |

Known omissions carry the capability's reason and remain `unavailable` or `not_applicable`.
An expected refusal after invocation becomes `unavailable` and retains the bound invocation
arguments. The report continues with other accepted operations. Structural errors still propagate.

`ASSESSMENT_CAPABILITIES` in
[`assessment.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/assessment.py)
declares, for each operation and each result family, the answer, the required artifacts, the cost,
and the execution class. The two costly classes are disjoint and are named separately.

A **refit**
operation fits new nuisance models. A **retarget** operation re-solves the fluctuation against
cached ones. `run_all` includes cheap retargets, such as an E-value derivation, by default.
It excludes refits and moderate retargets. Each skipped row names the flag that runs it.
The `arguments` mapping supplies required analyst choices for named operations.

An operation can decline caller-supplied arguments after its capability precheck. The combined
report records that refusal as an omission and continues with later operations. A direct call still
raises `CapabilityError` with the full refusal. Structural exceptions still stop the report.

The top-level `random_state` reaches `refute`, `benchmark`, and `simulated_confounding` only.
A seeded fit supplies its fit seed when this value is absent. An unseeded fit draws and records a
seed. Supplying the seed both places is an error.

## Sensitivity to untestable assumptions

Identification rests on assumptions that no diagnostic can test. These instruments do not test
them either. Some derive a formal scale. The simulated surface reports estimate movement
under a declared perturbation instead. Its `movement_scale` field names the scale of that
movement, which is additive for a difference and logarithmic for a ratio.

| instrument | the assumption it stresses | the number it reports | what it assumes to report it |
| --- | --- | --- | --- |
| omitted-variable bounds | no unmeasured confounding | the largest bias an unmeasured confounder of declared strength can produce | the confounder acts through the outcome regression and the treatment mechanism, with declared partial-$R^2$ strength in each |
| robustness value | no unmeasured confounding | the single strength at which the conclusion flips | that the two strengths are equal |
| benchmark | no unmeasured confounding | the strength of a confounder "as strong as" a named observed covariate | that dropping the covariate and refitting calibrates the scale |
| simulated common cause | no unmeasured confounding | estimate displacement across a declared strength grid, on the additive scale or the log scale | a supported latent perturbation family and plausible declared strengths |
| E-value | no unmeasured confounding | the minimum risk-ratio association with both treatment and outcome that explains away the effect | a risk-ratio scale |
| missingness tilt | outcomes missing at random | how the estimate moves as the unobserved outcomes are tilted away from the observed ones | the tilt is a constant on the logit scale |
| tipping gamma | outcomes missing at random | the tilt at which the conclusion changes | as above |

### Omitted-variable bounds, robustness value, benchmark, and contours

**How.** [`sensitivity/omitted_variable.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/omitted_variable.py)
implements Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022). The bound is

$$
|\text{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1-c_D^2}}\; c_Y \sqrt{\sigma^2 \nu^2},
$$

with $\sigma^2 = E[(Y - \bar{Q})^2]$ and $\nu^2$ the second moment of the Riesz representer. The
two primitives are exposed as `elements()`. A median-combined repeated fit refuses this analysis
because the median bound needs its own influence function.
`robustness_value()` inverts the bound for the single strength that flips the conclusion.
`benchmark()` drops each named observed covariate, refits, and calibrates the strength scale
against what that covariate was worth. `contour()` returns the grid a contour plot needs.

`benchmark()` is the only member of this group that refits.

### Simulated common-cause stress surface

**Why.** An analyst can inspect whether a fitted parameter moves under a plausible latent
common cause. Sharma and Kiciman (2020) name this procedure. Sharma et al. (2021) state its
qualitative limits.

**What it tells you.** `simulated_confounding()` reports the point estimate and its displacement
at every declared strength pair. `movement_scale` states how to read that displacement.
The frame from `to_frame()` carries it as a column on every surface, additive and ratio alike.
Additive parameters use `estimate_difference`. Ratios use `log_ratio` and the signed displacement

$$
\log(\widehat\psi_{\mathrm{refit}})-\log(\widehat\psi_{\mathrm{original}}).
$$

Its exponential is the refitted ratio divided by the original ratio. Each ratio cell keeps its
point estimate on the ratio scale. Gruber and van der Laan (2012), Section 2.1 and Section 2.7,
define the ratios and their log-scale intervals. Appendix A gives their log-scale influence
curves. The pinned `tmle3` `R/delta_functions.R` source uses the same contrasts.

Neither source defines a simulated-confounding displacement. The log difference is this package's
descriptive movement convention, not a paper-derived sensitivity measure.

Each cell also reports `induced_treatment_association`. This value is the realised correlation
between the latent vector and the treatment of that cell. The operation gives no corrected
estimate, bound, p-value, confidence interval, robustness value, threshold, or pass/fail result.

`target_measure` records how the operation reads the analysis rows. It is `unweighted` when the
fit declares no weights. It is `fixed_empirical_tilt` when the fit declares fixed weights.
`weight_report` carries the weight kind, column, scale, effective size, and concentration measures.
The label follows the declared weight column, so read `weight_report.is_weighted` to learn whether
the realized tilt is nonconstant.

**How.** The operation draws one row-level standard-normal latent vector. It reuses that vector
and one refit seed across the complete grid. Each cell starts from the original data.

For binary treatment strength $k_A$, the operation flips treatment when
$U \geq \Phi^{-1}(1-k_A)$. A continuous treatment uses $A' = A+k_AU$. Gaussian outcomes use
$Y' = Y-k_YU$. Binomial outcomes use the same tail-flip construction as binary treatment.
Flip strengths range from zero through 0.5. Continuous treatment strengths are signed finite
coefficients.

**Target populations.** The surface accepts one exact reported alias. Structured parameter metadata
identifies its arms, policy, and baseline stratum. The following contracts distinguish its populations.

| reported field | population contract |
| --- | --- |
| `stratum`, `strata_names` | The selected baseline values and their column names. A marginal parameter has no selected stratum |
| `population="baseline"` | An arm mean, ATE, ratio, or modified-policy parameter uses fixed baseline membership |
| `population="perturbed_treatment_group"` | ATT and ATC use observed-treatment membership after the cell's perturbation |
| `conditioning_arm` | The observed arm defining ATT or ATC. Other targets report `None` |
| `target_population_fraction` in each cell | The conditioning group's weighted share within the baseline population. Other targets report one |
| `association_population` | Correlation uses the selected baseline stratum, or all fitted rows for a marginal parameter. Both arms remain eligible |
| `calibration_population` | Numeric calibration uses the complete original fitted population |
| `refit_population` | Every non-anchor cell fits the complete perturbed dataset |

For baseline strata, the operation holds $S=f(W)$ fixed and changes only treatment and outcome.
The existing perturbation therefore acts on the conditional empirical law within each stratum.
One latent vector spans all rows and all strata. The operation never subsets rows before nuisance
fitting and never draws a new latent vector for each stratum. Two calls with the same seed and
different stratum aliases use the same perturbed datasets.

Ordinary TMLE supports conditional binary arm means, ATE, ratios, ATT, ATC, and modified-policy
means and contrasts. Binary C-TMLE supports conditional arm means, ATE, and ratios. These paths
retain fixed observation weights and estimator-owned repeat aggregation. A DR-TMLE fit refuses
`strata=` when it fits, so no stratified DR-TMLE result exists for the surface to replay.

ATT and ATC condition on the arm that the structured key names. Each cell recomputes that group's
membership from its perturbed treatment. Displacement combines changes in the fitted outcome
contrast and changes in population composition. It does not assess the effect among the original
treated or control rows held fixed. `target_population_fraction` makes the changing group share
visible. `CTMLE` and `DRTMLE` refuse ATT and ATC when they estimate, so this surface receives
ordinary-TMLE results only.

The DoWhy refit at `2116d5c` preserves effect modifiers and the target-unit selector. Its
propensity-weighting estimator recomputes ATT and ATC indicators from the perturbed data. This
supplies a finite-sample population convention, not a new sensitivity theorem. See the
[DoWhy source locators](../references.md#sensitivity-analysis). van der Laan (2010),
[Part I, Section 4](../references.md#point-treatment-and-stochastic-interventions), describes the
observed-treatment conditional target.

The post-fit assessment battery reads `target_population_fraction`. It reports `warning` when
`population` is `"perturbed_treatment_group"` and the smallest successful cell keeps less than half
the anchor cell's fraction. The anchor is the cell whose treatment strength and outcome strength
are both zero, so its fraction is the unperturbed share of the conditioning arm. The rule compares
each surface against its own anchor, so a small treated arm alone raises no warning.

A surface whose `population` is `"baseline"` never warns under this rule, because every cell
reports a fraction of one. The detail line carries the minimum fraction and the anchor fraction on
every surface. `tests/unit/test_post_fit_assessment_battery.py` pins both the detail line and the
warning threshold.

The diagnostic population differs from the ATT or ATC conditioning group. Treatment is constant
inside that group, so its correlation with the latent variable is undefined there. The surface
instead measures association across both arms within the selected baseline population. Numeric
calibration stays global; a conditional surface does not introduce a conditional calibration formula.

`tests/unit/test_simulated_confounding_populations.py` checks these contracts with complete refits
and nonzero population witnesses. Existing target instruments in the
[evidence manifest](evidence.md) establish the reused estimands. This wrapper adds no interval or
repeated-sampling claim and changes no estimator equations.

**The binary treatment axis.** The misclassification analysis below, its closed form, and the
treated-fraction table describe the binary tail flip only. The continuous law follows in its own
block.

The treatment law is non-differential misclassification. It flips a treated row and an untreated
row in the same latent tail. The association it induces between $U$ and the treatment therefore
depends on the treated fraction $\pi$. Write $q = \pi + (1-2\pi)k_A$ for the perturbed treated
fraction. When $A$ is drawn independently of $U$, the induced correlation is

$$
\operatorname{corr}(A', U) = \frac{(1-2\pi)\,\phi(\Phi^{-1}(1-k_A))}{\sqrt{q(1-q)}}.
$$

| treated fraction $\pi$ | $k_A = 0.1$ | $k_A = 0.3$ | $k_A = 0.5$ |
| --- | --- | --- | --- |
| 0.2 | +0.240 | +0.430 | +0.479 |
| 0.35 | +0.108 | +0.210 | +0.239 |
| 0.5 | +0.000 | +0.000 | +0.000 |
| 0.65 | -0.108 | -0.210 | -0.239 |
| 0.8 | -0.240 | -0.430 | -0.479 |

A 500,000-row simulation reproduces every entry to within 0.005. That bound is the Monte Carlo
error at that sample size. On a balanced design the treatment axis induces no association with
$U$. It moves the estimate through misclassification of the treatment alone. Above a treated
fraction of one half the sign reverses.

Sharma and Kiciman (2020) prescribe this construction, so the law does not change. The surface
instead reports what the law achieved on your data. Every cell carries the realised correlation
between the latent vector and its own treatment, in `induced_treatment_association`. Read that
value before you read a movement along the treatment axis as confounding. A cell near the anchor
value moved the estimate by misclassification of the treatment alone. The table above gives the
population value each cell approaches.

**The continuous treatment axis.** Under the continuous law, $U$ changes the dose by construction,
so $\operatorname{corr}(U, A')$ grows with $k_A$. A confounding path also needs $U$ to enter the
outcome. That happens only when the outcome strength is nonzero. A cell in the zero
outcome-strength column therefore carries no confounding path, whatever its association. Its
movement reports dose perturbation alone.

**The zero treatment-strength column.** A cell at $k_A=0$ leaves $U$ out of the treatment, so it
carries no confounding path either. Its movement reports the outcome perturbation alone. The
Gaussian law $Y'=Y-k_YU$ is a level shift, and the surface draws $U$ uncentred. An
`ate_shift[...]` contrast subtracts one policy mean from the other, so it removes most of that
level and the column stays small. An `ey_shift[...]` policy mean keeps it, so read the $k_A=0$
column of a policy-mean surface as an artifact of the outcome law.
`tests/unit/test_simulated_confounding.py::test_continuous_policy_mean_runs_a_real_ordinary_tmle_refit`
measures both columns on one fit.

**The reported association.** The surface measures the correlation on the analysis data. For binary
treatment, the value carries the treated fraction of your own fit. For continuous treatment, the
value reports what $A'=A+k_AU$ achieved on your dose distribution. The `(0, 0)` anchor cell
measures the original treatment, which gives the null level of the same data. A cell with constant
treatment reports no association, because its correlation is undefined. A failed cell keeps the
association of the treatment the surface built for it.

The `(0, 0)` cell returns the original estimate without a refit. A failed replacement or refit
remains visible as a `ReplicationFailure`. Successful cells report their displacement from the
original estimate, on the scale `movement_scale` names.

The surface supports these compositions with one or more cross-fitting draws.
Ratio surfaces require a binomial outcome. The other rows support Gaussian and binomial outcomes.

| treatment | parameter | replayed estimator |
| --- | --- | --- |
| binary | backdoor-identified marginal ATE | ordinary TMLE, collaborative TMLE, or complete-outcome DR-TMLE |
| binary | one explicitly named marginal `ey1`, `ey0`, or `ey[...]` counterfactual mean | ordinary TMLE, collaborative TMLE, or complete-outcome DR-TMLE |
| binary | marginal `rr` risk ratio or `or` odds ratio | ordinary TMLE, collaborative TMLE, or complete-outcome DR-TMLE |
| binary | baseline-stratum arm mean, ATE, risk ratio, or odds ratio | ordinary TMLE or collaborative TMLE |
| binary | marginal or baseline-stratum ATT or ATC | exact ordinary TMLE |
| continuous | one explicitly named marginal or baseline-stratum `ey_shift[...]` policy mean | exact ordinary TMLE |
| continuous | one explicitly named marginal or baseline-stratum `ate_shift[...]` contrast | exact ordinary TMLE |

Every row accepts fixed probability weights under its listed estimators.

For `repeats > 1`, each non-anchor cell calls the replay estimator once. The estimator owns all
repeat draws and reports their coordinatewise median. Additive displacement compares the refitted
median estimate with the original median estimate. Ratio displacement compares their median log
estimates. The result records `n_repeats` and `repeat_aggregation="coordinatewise_median"`.

The root seed gives every non-anchor cell the same repeat seed sequence. It does not preserve
realised folds after the perturbation changes a stratification variable. Treatment-stratified or
outcome-stratified splitting can therefore assign different rows to folds under the same seed.

The root seed reproduces the folds of the original fit only when it equals that fit's seed. The
helper `resolve_assessment_seed` returns an explicit `random_state` first, the fit's own seed
second, and a fresh seed for an unseeded fit. A refit through `TMLE.refit` reuses the estimator's own seed
sequence only for a seed equal to its `random_state`. Under any other seed each non-anchor cell
rebuilds its folds, and the anchor keeps the original ones. Movement near the anchor can therefore
carry a fold artifact.

The binary mean path keeps the named counterfactual arm fixed. Each cell replaces only the
observed treatment and outcome before the complete refit. A sole reported mean needs only the
grid. A fit that reports several means also needs one explicit alias.

The binary ratio path keeps the numerator and denominator arms fixed. It validates their direction
and the stored log estimate before the latent draw. A ratio-only result needs only the grid.

The continuous path keeps the fitted modified treatment policies fixed. Each cell replaces only
the observed dose and outcome before the complete refit.

**Fixed observation weights.** Ordinary-TMLE surfaces accept declared fixed probability weights
for every row in the support table. Binary complete-outcome collaborative-TMLE and DR-TMLE
surfaces also accept them. The weights define $dP_w=w\,dP/E_P[w]$. The operation keeps each
normalized weight on its original row during both replacements and every complete refit.

The latent value stays independent and standard normal. Each weight depends only on its own
observed row, so the joint law factorizes:

$$
d(P_w\times\Phi)(o,u)=\frac{w(o)}{E_P[w]}\,dP(o)\,d\Phi(u).
$$

Benkeser et al. (2017), Theorem 1, supplies the complete-outcome DR-TMLE corrected curve and its
remainder conditions.
`tests/unit/test_remainder_drtmle.py::TestAWeightedFitTransportsToTheTiltedLaw`
transports its conditional expectations, mechanism, marginal means, and scores to $P_w$. It keeps
a wrong-transport control with a nonzero remainder.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_drtmle_witnesses_the_weight_on_the_reduced_regressions`
adds the applied witness. It removes the weight from the reduced regressions alone, and it
requires the cell estimate to move.

Van der Laan and Gruber (2010), Sections 2, 5.1, and 6, define C-TMLE for a generic law and its
empirical distribution. Replacing that law by $P_w$ makes $P_{n,w}$ the empirical measure.
Selector strategies use the same normalized row mass in nuisance fits, targeting, outcome loss,
the influence-curve penalty, cross-validated risk, and the plug-in. The outcome-adaptive strategy
uses it in the outcome fits, categorical mechanism fit, targeting, and plug-in.

The fixed-weight C-TMLE tests cover greedy selection, both data-adaptive ordered preorders, an
explicit ordering, discrete selection, and outcome-adaptive fitting. They reconstruct a selected
path's weighted squared loss, weighted log-likelihood loss, influence-curve penalty, and nested
cross-validated risk term by term. They do not read the value that a production scoring method
returns. The nested reconstruction does call the production path search inside each training fold.
That search runs the production propensity fits, the targeting step, the loss, and the penalty. The
reconstruction discards the risk each candidate carries, and scores that candidate again on the
held-out rows.

The same test runs the production fold splitter on the surface's root seed. It compares the result
against the fold assignment the refit stored. Component mutations strip the weights from the loss,
the influence curve, the targeting step, the fold nuisance fit, the intercept mechanism, or the
candidate mechanism. Each mutation moves the stored nested-risk array.

Three tests ask whether a search decision reads the observation weights. The candidate risk is the
loss plus the penalty. A greedy search on a binomial fit and a logistic preorder on a Gaussian fit
each lose the weights from the loss. Each one moves the risk it is scored on. The
partial-correlation preorder never reaches the loss, so a third test removes the weights from that
ranking and asserts the order it produces. An explicit ordering carries no witness, because the
reader declares that order.

A fourth test pins the selected adjustment set, and not only the order the search visited. A moved
order that leaves the selection alone leaves the fitted mechanism identical. The fixture carries
two candidates that serve opposite weight blocks, in a mass ratio of nine to one. The weighted
search selects one candidate and the unweighted search selects the other, while the visited order
stays equal.

End-to-end mutations drop every selector weight or drop the outcome-adaptive
mechanism weights. Each mutation moves a manual refit that runs on the surface's seeds at strengths
0.2 and 0.3. Neither control reads a surface cell, because its grid holds the anchor alone and the
anchor runs no refit. Each control requires a move above one part in a thousand of the unmutated
refit's estimate.

The tests rebuild the vector-target penalty and the ratio penalty once each. The `ey1`, `ey0`, and
`or` estimands reuse the same arm-curve and delta-method arithmetic, and they carry no separate
dropped-weight mutation.

The pinned R `drtmle` 1.1.2 implementation accepts no observation-weight argument. It supplies no
weighted numerical comparator. This surface therefore claims no weighted R parity and broadens no
interval claim.

The canonical R `ctmle` 0.1.2 source at commit `18de559` also accepts no observation-weight
argument. The `ctmleDiscrete` entry point, which the pinned selector-parity study calls, takes
none, and neither do its `stage2` and `cv` helpers. The `ctmleGeneral`, `stage2_general`, and
`cv_general` paths use unweighted operations. The archived `ctmle3` source supplies
outcome-adaptive control flow but no weighted comparison. The fixed-weight C-TMLE surface
therefore claims parity with neither implementation.

Conditional on the observed rows and fixed weights, the simulation draws each latent value
independently from a standard normal law without using either. After that draw, the realized
weighted empirical distribution is discrete. Its latent marginal need not have weighted mean zero
or variance one, and chance association with the original treatment need not vanish. The anchor's
reported association records that finite-sample imbalance. Each cell reports the same parameter
functional on its perturbed weighted empirical law. This statement does not claim that the
operation reproduces the sampling or selection mechanism.

Hartman and Huang (2024) treat sensitivity to a confounder that the weighting model omits. They
report a bound, a robustness value, and a benchmark. This surface reports none of those, so their
method does not govern it.

The induced treatment association uses weighted means, variances, and covariance. Numeric
calibration also uses weighted scaling, model fitting, prediction-change fractions, and moments.
Constant weights use the exact unweighted calculation path. A common weight scale leaves the cell
estimates, the displacements, the induced associations, and the calibration strengths numerically
unchanged. The agreement is numerical rather than bitwise, because `check_weights` normalizes each
weight vector to mean one.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_surface_is_invariant_to_a_common_weight_scale`
pins ordinary TMLE to 1e-12.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_drtmle_surface_is_invariant_to_a_common_weight_scale`
applies the same tolerance to the DR-TMLE cells. The supplied scale is itself a descriptive
measurement, so
`weight_report.scale` does change with it.
`tests/unit/test_simulated_confounding.py::test_fixed_weight_ctmle_surface_is_invariant_to_a_common_weight_scale`
applies the same tolerance to selector-based and outcome-adaptive C-TMLE cells.

**Refusals.** `_validate_request`, in `src/cleverly/sensitivity/_simulated_confounding_request.py`,
raises before any random draw or refit. The `kind` column uses the
vocabulary of [how to read a refusal](scope-and-refusals.md#how-to-read-a-refusal), plus
`waiting on published theory` from the [roadmap's eligibility rules](../roadmap.md#eligibility).

| refused | kind | why |
| --- | --- | --- |
| a longitudinal result | not written yet | no longitudinal perturbation law is implemented |
| multi-arm treatment | waiting on published theory | no source-backed category-valued perturbation defines the contrast. See the [future investigation](../roadmap.md#f8-multi-arm-simulated-confounding-stress-surface) |
| a missing outcome | not written yet | the surface has no missingness law and no observation refit |
| a controlled direct effect, or any fit that carries an intermediate variable | not written yet | no intermediate-variable perturbation law is written |
| estimated observation weights | not written yet | the result lacks the fitted weight model needed after a perturbation |
| a clustered fit | waiting on published theory | no source chooses a row-level, cluster-level, or mixed latent cause. See the [future investigation](../roadmap.md#f9-clustered-simulated-confounding-stress-surface) |
| identification other than a backdoor mean contrast with explicit adjustment | not written yet | the surface reads registered explicit-adjustment provenance |
| ATT or ATC under C-TMLE or DR-TMLE | not written yet | `CTMLE` and `DRTMLE` refuse these functionals when they estimate, so no such fitted result exists |
| a requested baseline stratum under DR-TMLE | not written yet | a DR-TMLE fit refuses `strata=` when it fits, so no such fitted result exists. The guard keys on the requested stratum, not on `data.has_strata` |
| a regime, and a stochastic, incremental, or MSM parameter | a different question | each names a different intervention with its own influence curve |
| continuous-treatment C-TMLE and DR-TMLE | not written yet | `CTMLE` and `DRTMLE` refuse a modified-treatment-policy functional when they estimate. The surface replays exact ordinary TMLE only |
| the policy mean of a zero-delta shift | wrong by construction | $d_0(a, w) = a$ on both branches, so the policy is the natural course. Its mean is $E[Y]$, and no counterfactual treatment dependence remains for a common cause to move. The `ate_shift[...]` contrast that uses this policy as its reference is still accepted |
| a categorical benchmark covariate | waiting on published theory | no logical-covariate calibration maps categories to these perturbation strengths. See [F10](../roadmap.md#f10-logical-categorical-confounder-calibration) |

Three rows above record an estimator limit rather than a surface limit. The estimator refuses
first, so no fitted result reaches the surface guard. `_replay_refusal` keeps that guard as defence
in depth.

| composition | layer that refuses | when | message |
| --- | --- | --- | --- |
| ATT or ATC under C-TMLE or DR-TMLE | `CTMLE` and `DRTMLE` | `estimate()` | `method 'collaborative_tmle' cannot estimate ATT: no collaborative score is evidenced for this functional`. `DRTMLE` names its reduced-dimension correction instead |
| a modified-treatment policy under C-TMLE or DR-TMLE | `CTMLE` and `DRTMLE` | `estimate()` | `method 'drtmle' cannot estimate ModifiedTreatmentPolicy: no reduced-dimension correction is evidenced for this functional` |
| a baseline stratum under DR-TMLE | `DRTMLE`, in the shared targeting loop of `src/cleverly/estimators/tmle.py` | the fit | `baseline strata are not yet combined with the 'mean' group's alternating targeting equations`. `needs_reduction` holds because a DR-TMLE fit always builds reduced regressions |

The surface also refuses a result with no replay state, a result with no identification metadata,
and a constant benchmark covariate. It refuses a result whose repeat provenance disagrees with
itself. The stored draw count, `config.crossfit.repeats`, and the replay estimator's `repeats`
must state the same number. Each is a statement about the object you hold.

The surface also refuses three observation-weight states. It refuses a weight kind other
than `probability`, because it supports fixed probability weights only. It refuses a result whose
`weights_name` and `WeightSpec` name disagree, which is inconsistent weight provenance. It refuses
nonconstant observation weights with no declared weight column.
`tests/unit/test_simulated_confounding.py::test_weight_refusals_and_provenance_tampering_precede_draws_and_refits`
pins all three before the first draw and refit.

Numeric calibration follows the maintained DoWhy source as secondary implementation provenance.
For a binary variable, it reports the class-prediction change after one standardized column is set
to zero. For a Gaussian outcome or continuous dose, it reports `corr(W_j, V) * sd(V)`.
All terms use $P_w$ when the fit declares fixed probability weights.

The Gaussian calibration is signed. It carries the covariate's own direction of association with
the outcome or with the dose. Each axis converts a calibrated value into a declared strength by its
own rule, because the two perturbation laws carry opposite signs.

| axis | law | conversion |
| --- | --- | --- |
| outcome | $Y'=Y-k_YU$ | the law subtracts, so an outcome strength of $k_Y$ calibrates at $-k_Y$. To match a covariate that calibrates at $+c$, declare an outcome strength of $-c$ |
| treatment, continuous dose | $A'=A+k_AU$ | the law adds, so a treatment strength of $k_A$ calibrates at $+k_A$. To match a covariate that calibrates at $+c$, declare a treatment strength of $+c$ |

A binary treatment or a binomial outcome has no such conversion. Its calibration is the
class-prediction change fraction, which carries no sign.

Calibration does not select or modify the grid. It is not partial R-squared and does not reuse the
omitted-variable `benchmark()` scale.

The maintained source is pinned at revision `2116d5c`. The implementation does not copy its
cumulative cell mutations, schedule-dependent random draws, non-exact zero refit, automatic
ranges, categorical encoded-column deletion, or unstructured failure behavior.

### E-value

**How.** [`sensitivity/evalue.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/evalue.py)
implements VanderWeele and Ding (2017): $E = RR + \sqrt{RR(RR-1)}$. It computes the E-value for the
point estimate and, separately, for the confidence limit, because the second is the one an
adversarial reader asks for.

The selected path depends on the reported contrast and retained artifacts.

| request | E-value path |
| --- | --- |
| reported risk ratio | use it directly and mark the result exact |
| unambiguous default binary marginal ATE or odds ratio from ordinary TMLE | retarget cached nuisances to the matching risk ratio and mark the result exact; combined runs include this cheap retarget by default |
| explicit reported odds ratio, or default odds ratio without exact retarget support | use the common-outcome approximation $\sqrt{OR}$ and mark the result approximate |
| binary ATE without exact retarget support, with a usable reported reference-arm mean | hold the baseline risk fixed and mark the result approximate; includes DR-TMLE, collaborative TMLE, and CV evaluation |
| Gaussian ATE, ATT, or ATC | standardize by the observed outcome standard deviation, weighted on a weighted fit, and mark the result approximate |
| binomial ATT or ATC | refuse because the conditional baseline risk and conditional ratio target are absent |
| level or non-arm parameter | report `not_applicable` because no supported two-arm contrast exists |
| binomial ATE without exact retarget support or a usable reported baseline; controlled direct effect needing derivation | report `unavailable` and name the missing evidence, artifact, or target |

Several eligible contrasts require an explicit alias. Combined runs select availability and cost
from that alias before applying the cost flags.

Raw results compose arm identities forward from fitted treatment metadata.
Explicit structured keys remain authoritative. No routing step parses display aliases.
The fixed-baseline approximation ignores baseline sampling error and requires the contrast's matching reference arm.
Its availability does not depend on persistence or estimator retention.

A usable reference-arm mean is finite, positive, and separated from zero by its own standard error.
The conversion divides by that mean, so a mean at zero leaves the ratio without a stable denominator.
The library also refuses a risk difference at or below the negative of that mean.
Such a difference implies a nonpositive risk in the contrast arm, and no risk ratio describes one.
Both refusals report `unavailable` and name the two reported numbers.

The conversion is affine, so the lower interval bound can leave the parameter space while the point ratio stays inside it.
Only the lower bound can leave it. `normal_ci` gives `high >= psi`, and the refusals above force `baseline.psi > 0` and `baseline.psi + psi > 0`.
The report truncates the lower bound at 0, records the untruncated value in `truncated_bound`, and repeats it in the note.
The `to_dict` mapping and the battery row both carry that value, so no surface presents the 0 as a converted confidence limit.

What the truncation means for the confidence limit depends on the side of the null the point ratio is on.

| point ratio | bound the E-value reads | effect of the truncation |
| --- | --- | --- |
| at or above the null | the truncated lower bound | the interval covers the null, so the confidence-limit E-value is 1 |
| below the null | the untruncated upper bound | none. The interval can still exclude the null, and the confidence-limit E-value can exceed 1 |

For rare outcomes, the odds ratio itself approximates the risk ratio. The square-root
transformation addresses common outcomes and can lie above or below the risk ratio.
See [VanderWeele's analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5617805/).

The Gaussian path first applies Chinn's odds-ratio to standardized-mean-difference relation.
It then applies the common-outcome square-root approximation before calculating the E-value.
The resulting conversion is $RR \approx \exp((1.81 / 2)d) = \exp(0.905d)$.
This retains an approximate analysis, not an exact continuous-outcome risk ratio.

A weighted fit standardizes by the weighted outcome standard deviation.
The estimate targets the population the observation weights describe, and the standardizing scale describes the same population.
The weighted form divides by the reliability-weight correction $\sum w - \sum w^2 / \sum w$.
That correction is $n - 1$ when every weight is one, so an unweighted fit reports the plain sample standard deviation.

### Missingness tilt and tipping gamma

**How.** [`sensitivity/missingness.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/sensitivity/missingness.py)
implements the Scharfstein, Rotnitzky and Robins (1999) tilt. It sets

$$
\bar{Q}^{\text{miss}}_\gamma = \operatorname{expit}\{\operatorname{logit} \bar{Q}^* + \gamma\}
$$

for the unobserved outcomes, and mixes it with $\bar{Q}^*$ by the estimated missingness
probability. At $\gamma = 0$ it reproduces the missing-at-random estimate by construction, which is
the control that says the tilt is wired in. `arm_gamma=` gives per-arm tilt directions and must
name every arm. `tipping_gamma()` inverts the tilt for the value at which the conclusion changes.

This is a retarget operation and not a refit.

### The scope rule

A point-treatment sensitivity formula is not reused on longitudinal data. `LTMLE` reports these
operations `unavailable` with the reason its own capability row declares, rather than borrowing a
derivation. Stagewise support, scores, and nuisance loss are supported longitudinally, because each
has its own derivation.

## Refutation and simulation you run

These fit new models. They cost what a fit costs, multiplied by the number of draws.

### Refutation operations

**Why.** A diagnostic reads the fit you have. A refuter constructs a case whose answer is known and
checks that the workflow returns it.

**What each tells you.** [`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py)
ships seven operations. Six test the implementation. The negative-control outcome tests a design,
and the paragraph below states its boundary.

| refuter | what it does | what must happen | what it tests |
| --- | --- | --- | --- |
| `placebo` | permutes the treatment column and refits | the estimate goes to zero | the pipeline, not the data |
| `random_common_cause` | adds an irrelevant covariate and refits | the estimate does not move | the adjustment set is not sensitive to noise |
| `subset` | refits on random subsamples | the scatter is about one standard error | the reported standard error is the right size |
| `negative_control_outcome` | refits on an outcome the treatment cannot affect | the estimate goes to zero | the design, under the control assumptions the paragraph below states |
| `dummy_outcome` | draws independent Gaussian noise and refits | the empirical draws include zero | outcome replacement and the full estimator pipeline |
| `simulated_outcome` | draws `f(W) + effect * A + epsilon` and refits | the empirical draws include the declared effect | adjustment and treatment terms in the full pipeline |
| `bootstrap_measurement_error` | bootstraps, perturbs declared adjustment variables, and refits | the empirical draws include the original estimate | stability under the declared measurement error |

A refuter refits the nuisance models once for each replication. The three default operations use
five replications each, so `refute()` costs about 15 fits. Empirical refuters use 100 draws by
default because their rule reads a distribution. `run_all(include_refits=True)` runs `refute()`.

`refute()` draws its randomization from the seed of the fit, unless the caller passes
`random_state`. A fit that carries a seed gives the same refutation on every call. A fit that
carries no seed gives a different refutation on every call.

The report records the seed under `random_state`. Pass that value back to `refute()` to obtain
the report again. The seed governs the perturbations and the refits they feed, so it repeats
the report of a fit that carries no seed of its own. The seed applies to a copy of the
estimator, so a refutation never changes the fit it examines.

Each empirical draw derives a child seed from the recorded root seed. The perturbation and its
full refit use that child seed. `report.draws_frame(name)` reports every child seed, estimate,
standard error, family, and failure. `GeneratedOutcomeRecord` remains an alias for the shared
`EmpiricalRefitRecord`.

`EmpiricalInclusionRule` uses a two-sided empirical rank and inclusive half-ties. It passes only
when the empirical probability strictly exceeds alpha, matching the maintained DoWhy convention
that a probability at or below alpha fails. The default rule uses alpha 0.05, requires 40
successful draws, and fails when any refit fails. A failure stays in the report as the shared
`ReplicationFailure` record, so the operation never reports only the conditional distribution of
successful fits.

The rule count and the draw budget are separate defaults. `DEFAULT_OUTCOME_REPLICATES` is 100 and
sets `n_replicates`. `EmpiricalInclusionRule()` requires 40 successful draws and sets no budget.
The operation refuses a budget below the rule's minimum before it refits anything.

Bootstrap measurement error uses the same rule against the original estimate. It draws an iid or
whole-cluster sample through the inference bootstrap design. It perturbs numeric variables with
mean-zero Gaussian noise after sampling. The noise scale is the declared multiplier times the
selected variable's bootstrap-sample standard deviation.

Each sampled cluster occurrence receives a distinct code before perturbation. Repeated draws of
one source cluster therefore remain separate during the refit.

The categorical path recovers original logical levels from `CausalData.encodings`. Each changed
row draws uniformly from the other levels. The operation then rebuilds the complete drop-first
indicator block. Boolean covariates use the same two-level path.

The operation checks every condition in the table below before the first refit. Each row is one
`CapabilityError` branch of `_validate_measurement_error_eligibility` in
[`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py),
and each message names the variable it rejects.

| what the operation refuses | what it requires instead |
| --- | --- |
| a result whose data is not point-treatment `CausalData` | a point-treatment `CausalData` result |
| `resampling="cluster"` on data that carries no cluster ids | declared cluster ids, or iid resampling |
| a selected strata variable | a variable outside the strata, because the operation cannot perturb target metadata coherently |
| a generated indicator name | the original categorical variable that the indicator encodes |
| a name that is not an original adjustment variable | one of the original names the message lists |
| a categorical variable whose encoded block lost an indicator to duplicate-column removal | a retained indicator for every level the encoding generated |
| a selected variable whose values are all 0 or 1 and that carries no `CategoricalEncoding` | the same variable declared to `CausalData.from_frame` as a boolean or categorical column, because relative Gaussian noise makes an undeclared indicator real-valued |
| a selected numeric variable that is constant | a variable with nonzero spread, because a relative noise scale of zero leaves every draw unperturbed |

The report records the declaration, requested draw count, resolved mode, rule, estimates,
standard errors, child seeds, and failures.

**Alpha is a width here, and not a false-alarm rate.** The rule decides one question. Does the
declared effect lie inside the central `1 - alpha` of the refit estimates? At the default alpha
that is the central 95%.

A correct estimator centres the refit distribution on the declared effect, so each draw falls
above it with probability about one half. Under 100 draws the rule then fails only when at most
two draws fall on one side. That probability is `2 * (1 + 100 + 4950) / 2**100`, which is about
`8e-27`. More draws do not buy power. They stabilise the quantile the rule reads.

The default minimum of 40 draws is the smallest budget at alpha 0.05 that can fail on anything
short of a one-sided sweep. With 40 draws, one estimate on the minority side gives `2 / 40`, which
equals alpha and fails. With 39 draws, one estimate on the minority side gives `2 / 39`, which
exceeds alpha and passes. The rule therefore refuses a declaration whose `minimum_draws * alpha`
is below 2, because such a rule fails only when every draw falls on one side.

The first process catalog covers Gaussian outcomes and additive `ate`, `att`, or `atc` contrasts.
`refute()` and its `_validate_generated_eligibility` helper in
[`validation/refute.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/refute.py)
check every condition in the table before the operation refits anything. Each row is one
`CapabilityError` branch, and each message names what is missing.

| what the operation refuses | what it requires instead |
| --- | --- |
| an inclusion rule of any other type | the exact registered `EmpiricalInclusionRule` declaration |
| a process declaration of any other type | the exact registered `GaussianIndependentOutcome` or `GaussianAdjustmentOutcome` |
| a process whose family is not `"gaussian"` | the Gaussian family, which is the one with an implemented effect derivation |
| a legacy fit that carries no identification metadata | identification metadata on the result |
| a functional that is not `BackdoorMeanContrast` | a backdoor-identified additive mean contrast |
| a provider that is not `ExplicitAdjustmentProvider` | registered backdoor provider provenance |
| an original outcome family that is not `"gaussian"` | an original family equal to the process family, so a binomial fit is refused |
| an estimator configured with a family other than `"auto"` or `"gaussian"` | a configured family that accepts the declared process |
| a saved outcome learner with no regression-capable route | a regression-capable outcome learner |
| a longitudinal functional | a point-treatment functional |
| a treatment that is not binary | code one against code zero |
| an intermediate node or a controlled direct effect | no intermediate node |
| a fit with missing outcomes | complete outcomes |
| an MSM functional or the `msm` axis | the `arm` axis |
| an intervention-indexed functional | the `arm` axis with no declared interventions |
| a parameter key that is not a structured `ParameterKey` | a structured key for the selected estimand |
| a ratio estimand, or any other non-additive contrast | `ate`, `att`, or `atc` |
| disagreement between the functional target, the identified estimand, and the parameter key | one estimand named by all three |
| a missing registered identification artifact | the artifact `TARGETS` records for that target |
| an arm contrast that does not resolve to two distinct arms | a resolvable contrast of two distinct arms |
| a draw budget below the rule's minimum | `n_replicates` of at least `minimum_draws` |

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate
from cleverly.validation import GaussianAdjustmentOutcome, GaussianNoise

frame, _ = make_linear_ate(n=200, seed=21)
study = CausalStudy(
    frame,
    design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
)
result = study.estimate(
    ATE(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=21,
)
process = GaussianAdjustmentOutcome(effect=0.5, noise=GaussianNoise())
report = result.diagnostics.refute(
    tests=("simulated_outcome",),
    simulated_outcome=process,
    random_state=21,
)
print(report.summary())
```

Sharma and Kiciman (2020) define the refutation framework. The maintained DoWhy dummy refuter,
pinned at
[`2116d5c`](https://github.com/py-why/dowhy/blob/2116d5cbace5a057937e03b2efba95c13140cc4c/dowhy/causal_refuters/dummy_outcome_refuter.py),
supplies secondary evidence for independent noise and `f(W) + h(A)`. That implementation uses a
normal rule below 100 draws, which `perform_normal_distribution_test` in
`dowhy/causal_refuter.py` applies. It declares no failure policy at all. The pinned file contains
no `try` block, and its refits run under `joblib.Parallel`, so one failed refit aborts the whole
refutation. `cleverly` keeps each failed refit as a `ReplicationFailure` record and fails the
refutation under the rule stated above.

The maintained DoWhy bootstrap refuter at the same revision supplies secondary control-flow
evidence for measurement-error sampling and refitting. `cleverly` does not copy three defects in
that source. It uses logical metadata for numeric and categorical variables. It does not reuse a
Boolean probability array for another categorical variable. It derives a distinct child seed for
each draw.

A negative-control outcome must have no causal path from treatment. It must also share the relevant
confounding structure with the primary outcome. A non-null result flags residual bias or a bad
control, and the refuter cannot tell you which. A null result does not establish that unmeasured
confounding is absent. See [negative controls](../references.md#negative-controls).

### Coverage studies

**Why.** Coverage, bias, and standard-error calibration are claims about repeated sampling. One fit
contains no information about any of them.

**What it tells you.** Three numbers, from
[`validation/simulation.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/validation/simulation.py).

| number | how it is computed | how to read it |
| --- | --- | --- |
| coverage | the share of replications whose interval contains the truth | sustained undercoverage beyond Monte Carlo uncertainty indicates invalid intervals on that law |
| root-n bias | $\sqrt{n}$ times the mean error | bounded values support a negligible first-order bias claim. They do not establish efficiency |
| SE ratio | mean reported standard error over the empirical standard deviation of the estimates | one means the reported uncertainty matches the real spread |

**How.** `CoverageStudy` draws from a generator with a known truth, runs the complete estimator on
each draw, and summarises through `summarize_replications`. A failed draw is retained as a
`ReplicationFailure` record carrying its index, its seed, and its exception. A study that silently
replaced failed draws would report the distribution of the draws that happened to work.

The generators live in
[`datasets/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/datasets) and each
one carries an exact `truth`.

**A simulated law is an instrument, and it can be wrong.** A coverage study is evidence only if the
number it calls the truth is the number an adjusted fit estimates. Two shipped clustered generators
once failed that test, and
[evidence.md](evidence.md#a-simulated-law-is-an-instrument-too-and-it-can-be-wrong-the-same-way)
records what went wrong and what each generator now asserts about itself.

### Variable importance

`variable_importance` gives each candidate covariate the treatment role in its own fit, and
reports the target-relevant change with multiplicity-adjusted p-values
([`variable_importance.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/variable_importance.py)).
It is an assessment of the fitted causal workflow. It is not a predictive feature-importance score,
and it introduces no new influence function.

## How the library certifies itself

The three layers above review *your fit*. This layer reviews *the implementation*. It is the
evidence that the equation `score_check` solved is the right equation.

The instruments go blind in different places, and the differences are the reason there are six of
them. [evidence.md](evidence.md) records which instrument covers which registered estimand, in both
directions, and it is a test rather than a note.

| instrument | why it exists | what it tells you | how it tells you | what it cannot see |
| --- | --- | --- | --- | --- |
| **exact oracle law** | an estimator has to recover a parameter that was computed rather than estimated | the reported number is the parameter, exactly | a finite-support law whose every cell probability is a multiple of $1/N$, so an $N$-row frame **is** the law. Handed oracle nuisances, the fit is exactly right and $\epsilon$ is zero | nothing about a term that is zero at the truth |
| **Gateaux comparison** | the influence curve is what every interval is built from | the reported curve is the pathwise derivative of the parameter | complex-step differentiation of an independently written functional, compared at about `1e-14` absolute with `rtol=0` | a sign on any block that vanishes at correct nuisances, and any counterfactual block, because $\epsilon$ is zero there |
| **second-order remainder** | double robustness *is* the remainder carrying both nuisance errors | one wrong nuisance still leaves the remainder second order | the von Mises expansion evaluated at nuisances that are wrong on purpose, against a longhand form of the exact remainder | a first-order error that cancels inside the remainder |
| **exact identity** | some mistakes are algebraic and cheap to catch | a relation that holds by definition holds bit for bit | relabelling the arms, a null outcome model giving zero, weights scaling out, the one-step and iterative solvers agreeing | anything symmetric in whatever the identity is symmetric in |
| **theorem check** | the anchor the others need | the implementation agrees with the source's own theorem | evaluation at values where the quantity does **not** vanish | nothing the theorem does not state |
| **deliberate mutation** | a passing test proves nothing unless a wrong version fails it | each plausible way of building the thing wrong is shown to fail | the component is broken on purpose and the suite is required to go red | a mistake nobody thought to make |

Three supporting rules make the table mean what it says.

- **The oracle laws share no code with the library.** `tests/unit/test_oracle_independence.py`
  asserts that the oracle modules never import `cleverly`. A shared helper would move both sides of
  the comparison equally.
- **A heading is not enough.** `tests/unit/test_registry.py::TestEvidenceManifest` checks the
  evidence table against the target registry in both directions, checks that every module named
  there exists, and checks that the oracle-law column names the law whose functional really has the
  branch.
- **Cross-fitting is checked without a tolerance.**
  `tests/unit/test_crossfit_leakage.py` rigs a law in which one covariate is constant within a
  cluster and the outcome *is* that covariate with no noise. A nearest-neighbour learner then
  reproduces a held-out row bit for bit if and only if a same-cluster row was in its training set.
  The assertions are array equality, so leakage is not a matter of degree.

### The oracle-law gate

Registering a target whose reported parameters have no branch in an oracle law's `functional` is a
test failure rather than an oversight caught in review. The evidence this package offers that an
influence curve is correct is that it agrees with one obtained by complex-step differentiation of
an independently written functional on an exactly representable law. An estimand without that has
no such evidence.

The gate walks the *parameter* names a target reports rather than the target name, so a per-arm
target needs an oracle for each arm. A target intended for more than two arms needs one on the
three-armed law, because two arms cannot distinguish code that keys by arm from code that has two
columns and calls them 0 and 1. The gate runs in both directions. An oracle branch that no target
reports is dead code, so a law and the registry must cover each other exactly.

### Registered repeated-sampling studies

The instruments above ask whether each parameter is implemented correctly. A registered study asks
the complementary question. Apply a complete estimator to samples from a known law, and does its
bias and its uncertainty behave as its source theory predicts?

The design rules are in
[method benchmarking strategy](../development/method-benchmarking.md). The grid is in
[the technical reference index](method-evidence/validation-grid.md). The test-by-test results
are in [the implementation validation studies](method-evidence/index.md).

Three properties of the harness are worth stating here, because they are what make a green study
mean something.

- **A verdict is bounded by a margin declared before the run.** No rule tests whether a discrepancy
  is exactly zero.
  [The verdict rules](method-evidence/how-to-read.md#the-verdict-rules) give the argument and list
  every rule with its own control.
- **Every positive claim carries a control that must fail.** Double robustness carries a
  both-wrong-nuisance control. A type-I error cell carries a power cell, so an inert test cannot
  pass by never firing. An interval-calibration cell carries deliberately invalid inference.
- **A replication is a fixed sample.** Seeds spawn on the study's own record and on the replication
  index, so replication *k* is the same draw whatever the study's budget. A two-replication probe
  redraws exactly the published first two.

Matching a canonical R implementation is a separate and weaker claim. Two implementations
descended from one source share transcription errors, so agreement localises a discrepancy and does
not certify either one. Every study therefore tests each implementation against known truth first,
and separately.
