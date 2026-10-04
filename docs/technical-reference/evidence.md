# What each estimand's correctness rests on

`tests/unit/test_registry.py`'s `test_every_target_has_an_oracle` answers one question:
*does this target have a branch in an oracle law?* The answer is yes or no. That gate is
what stops an estimand shipping "on the strength of its author's arithmetic alone", and it
is not what this page is for.

**This page is the other half: which instruments a target actually has, and which it does
not.** The distinction matters because the instruments go blind in different places, and
[*How to read a
refusal*](scope-and-refusals.md#how-to-read-a-refusal) is not the only information to record.
How to read a *pass* is also important. The trap that survives a green suite is:

> an exact-law instrument goes blind wherever a quantity vanishes at the truth, which is
> where a parity check is blind too. At correct nuisances `Q_r` and `g_{r,2}` are zero row
> by row, so every `test_influence_gateaux*` module passes against a flipped sign.

A row whose only evidence is a Gateaux comparison is therefore *not* the same claim as a row
that also carries a remainder rate and an exact identity. A reader deciding how far to trust a
number should be able to see which row they are looking at, without opening five test modules.
That is the whole of what this table is.

**It is a gate, not a note.** `tests/unit/test_registry.py::TestEvidenceManifest` checks it
in both directions against `TARGETS`, and checks that every module named here exists. The part
that makes it more than a document is the third check: the *oracle law* column must agree with
the law whose `functional` really has the branch, read through the same `oracle_for` the coverage
gate uses. A row cannot claim an oracle the laws do not provide.

## The instruments

| kind | what it is | what it cannot see |
| --- | --- | --- |
| **oracle law** | the parameter written down longhand on an exactly representable discrete law, sharing no code with `src/` | nothing about a term that is zero at the truth |
| **Gateaux** | the reported influence curve against a complex-step derivative of that functional, to ~1e-14 absolute with `rtol=0` | a sign on any block that vanishes at correct nuisances; a counterfactual block, at `epsilon = 0` |
| **remainder** | the von Mises expansion's second-order term, measured as a rate under one wrong nuisance | a first-order error that cancels inside the remainder |
| **exact identity** | an algebraic relation that holds by definition and so must hold bit-for-bit | anything symmetric in whatever the identity is symmetric in |
| **theorem** | a check against the source's own theorem, *at values where the quantity does not vanish* | this is the anchor the other checks need |
| **bounded implementation witness** | a frozen comparison with an independently maintained implementation, scoped to a named finite-sample choice that the scientific oracles cannot exercise | the estimand's derivation; any behavior outside the deliberately matched nuisance, bound, and targeting settings |

## The table

One row per registered target. `none` in a cell means there is no such instrument for this
target, which is a statement rather than an omission; the **not covered** column says what
follows from it.

| target | oracle law | Gateaux | remainder | exact identity | not covered |
| --- | --- | --- | --- | --- | --- |
| `ate` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_multi.py`, `tests/unit/test_influence_gateaux_multi_collaborative.py` | `tests/unit/test_remainder.py`, `tests/unit/test_remainder_multi.py` | `IC_ate == IC_ey1 - IC_ey0`, and a null outcome model gives zero in every population (`tests/unit/test_invariants.py`) | no theorem anchor; the derivation is checked only where the oracle law can represent it |
| `att` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_multi.py` | `tests/unit/test_remainder.py` | relabelling the arms gives `ATT' == -ATC` (`tests/unit/test_invariants.py`) | swapping the two conditioning populations outright is symmetric in the arms, so the symmetry test cannot see it; the closed-form comparison catches it |
| `atc` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_multi.py` | `tests/unit/test_remainder.py` | relabelling the arms gives `ATC' == -ATT` (`tests/unit/test_invariants.py`) | as `att`, and for the same reason |
| `ey` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_multi.py`, `tests/unit/test_influence_gateaux_multi_collaborative.py` | `tests/unit/test_remainder.py`, `tests/unit/test_remainder_multi.py` | none | no identity of its own; it is the per-arm level the contrasts are built from, so its errors surface in them |
| `ey1` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | `IC_ate == IC_ey1 - IC_ey0` (`tests/unit/test_influence_gateaux.py`) | binary-only by declaration; the multi-arm path reports `ey` instead |
| `ey0` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | `IC_ate == IC_ey1 - IC_ey0` (`tests/unit/test_influence_gateaux.py`) | binary-only by declaration, as `ey1`; and the identity it shares with `ey1` is symmetric in the two arms, so a defect that swaps them survives it |
| `ey_obs` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_natural_course_mar.py`, `tests/unit/test_natural_course_crossfit.py` | `tests/unit/test_remainder_natural_course_mar.py` for ordinary missing outcomes; `tests/unit/test_natural_course_crossfit.py` for the signed fold-specific remainder | the complete-data curve is `w(Y - E_w[Y])`; at `pi == 1` the natural-course construction itself returns that estimate and curve. Respondent-mask, all-row plug-in, response-score, fold-rotation, removed-response, independent leakage, and fail-if-fit treatment-learner controls distinguish the MAR paths | the MAR intervals cover the ordinary-TMLE fit, with fixed weights and clusters, and one binary, one-repeat stacked CV-TMLE configuration on unweighted iid rows. `intermediate=` and bootstrap remain refused. The in-sample fit admits strata (`tests/unit/test_stratified_natural_course_exact.py`). The stacked fit also refuses strata, bounded-continuous outcomes, supplied folds, fold targeting, fold evaluation, and repeated splits. A joint fit's `ey_obs` equals the scalar fit bit for bit (`tests/unit/test_attributable_mar_stack.py`), so the scalar evidence covers it |
| `par` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` (`par` in its functional; this law is outside the registry walk) | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_attributable_mar.py` (all 18 MAR support points, also under two fixed weight functions) | `tests/unit/test_remainder_attributable_mar.py`: the remainder is $R_{\mathrm{nc}}-R_{a_0}$ to $10^{-12}$, including the product-only case | `IC_par == IC_ey_obs - IC_ey0`; nonzero response-score witnesses on both paths; the variance reads the cross-covariance (dropping it moves the SE by 1.951); the cross-covariance witness `test_dropping_the_cross_covariance_moves_the_standard_error`, the response-score witnesses `test_w1_...` and `test_w2_...`, and the path, reference and no-copy witnesses in `tests/unit/test_attributable_mar_stack.py` | controlled intermediates and stacked missing-outcome strata are refused. The pointwise curve check sees each residual term at the truth, but the point and remainder checks at the truth do not, so the mutations run away from the truth |
| `paf` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_attributable_mar.py` | the delta-method identity in `tests/unit/test_remainder_attributable_mar.py`, exact to $10^{-12}$ | its curve is the delta-method transform of `ey_obs` and the reference-arm mean; dropping the cross-covariance moves the SE by 1.715 | defined only for a binary outcome with positive observed risk; small-sample coverage near zero risk is not established |
| `rr` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | the ratio's curve is the delta-method transform of the levels' (`tests/unit/test_influence_gateaux.py`) | the log-scale interval's small-sample coverage is a simulation claim, not an identity |
| `or` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | the odds ratio's curve is the delta-method transform of the levels' (`tests/unit/test_influence_gateaux.py`) | as with `rr`, the log-scale interval's small-sample coverage is a simulation claim rather than an identity; and nothing here pins the odds ratio apart from the risk ratio at values where the two are close |
| `ey_regime` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_regime.py` | `tests/unit/test_remainder_regime.py` | a degenerate regime equals the static arm it puts all its mass on (`tests/unit/test_regimes.py`); `tests/unit/test_stochastic_regime_densities.py` fits the odds tilt of the sample mechanism declared `"known"`. Its curve is the fixed-density curve, and its standard error at `delta = 2` is 0.623 of the estimated-density one. That gap is why `density_kind="estimated"` is refused. `tests/unit/test_rule_and_intervention_declarations.py` repeats that witness on a user-written class declared `"known"`. It also fits a `Rule` with a threshold at the sample mean of a continuous covariate, declared `"known"`. Its standard error is 0.728 of the exact one, which is why `rule_kind="estimated"` is refused | a rule that is not deterministic; the density is evaluated once at fit time and the fit answers for that evaluation. No check detects a density, a rule, or a user-written class that reads the sample and is declared `"known"` |
| `ate_regime` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_regime.py` | `tests/unit/test_remainder_regime.py` | the contrast is the difference of the means (`tests/unit/test_regimes.py`) | a rule that is not deterministic, as for `ey_regime`; and the contrast's identity is a relation between two reported numbers, so a defect common to both means survives it |
| `ey_learned_rule` | `tests/discrete_law.py`, at one fixed rule, conditional on training | `tests/unit/test_influence_gateaux_learned_rule.py`, fold by fold, at each fold's own law and a nonzero pooled fluctuation | none | the rule of each row comes from a fit that never saw the row (`tests/unit/test_learned_rule_fold_locality.py`); the weighted pooled score is zero and the estimate is the `1/V` fold average (`tests/unit/test_learned_rule_targeting.py`); the curve is the fold-centred longhand curve (`tests/unit/test_learned_rule_influence.py`); the variance is the cross-validated form at unequal folds (`tests/unit/test_learned_rule_variance.py`) | the learning step: the oracle and the Gateaux check hold each fold's rule fixed, so they cannot see how the rule was learned. The limiting-rule condition C3, the exceptional laws where it fails, the remainder along the fold rules and the quality of the rule are not covered here. The [gated study](method-evidence/learned-rule-cvtmle.md) measures coverage at two laws that meet C3. The interval under-covers at the exceptional and weak-blip laws that the [boundary study](method-evidence/learned-rule-cvtmle-boundary.md) reads |
| `ey_ipsi` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux_ipsi.py`, `tests/unit/test_influence_gateaux_ipsi_mar.py` | `tests/unit/test_remainder_ipsi.py`, `tests/unit/test_remainder_ipsi_mar.py` | `psi(delta=1)` equals `mean(Y)` row by row whatever the nuisances are (`tests/unit/test_ipsi_fit.py`); this detects an alternation that exits with one equation open | this is the one estimand that is **not** doubly robust: every remainder term carries `(ghat - g0)`, so a consistent `Qbar` cannot substitute for a consistent mechanism. The alternation's linear rate is measured, not bounded |
| `ate_ipsi` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux_ipsi.py`, `tests/unit/test_influence_gateaux_ipsi_mar.py` | `tests/unit/test_remainder_ipsi.py`, `tests/unit/test_remainder_ipsi_mar.py` | `psi(delta=1)` equals `mean(Y)`, and the contrast of two tilts at `delta=1` is zero (`tests/unit/test_ipsi_fit.py`) | not doubly robust, as for `ey_ipsi`; with `delta=` present the `psi(1)` check changes meaning. It is then the MAR-identified `E[Y]` and the complete-case mean is the wrong answer, so the check must not be read across that case |
| `ey_policy` | `tests/discrete_law_shift.py` | `tests/unit/test_influence_gateaux_shift.py`, `tests/unit/test_influence_gateaux_shift_cde.py` | `tests/unit/test_remainder_shift_cde.py` | the negative control: a shift's mean equals the induced stochastic regime's, and its **curve does not** (`tests/unit/test_influence_gateaux_shift.py`) | there is no plain-shift remainder module. The rate is measured only with a third nuisance (`_shift_cde`). A Gateaux check on an exact law cannot see a counterfactual block, so `tests/unit/test_shift_submodel.py` and `tests/unit/test_shift_fit.py` pin those structurally and at `epsilon != 0` instead |
| `ate_policy` | `tests/discrete_law_shift.py` | `tests/unit/test_influence_gateaux_shift.py`, `tests/unit/test_influence_gateaux_shift_cde.py` | `tests/unit/test_remainder_shift_cde.py` | the same negative control as `ey_policy`, taken on the contrast (`tests/unit/test_influence_gateaux_shift.py`) | no plain-shift remainder module, as for `ey_policy`; and the MNAR tilt is refused on this axis by name, so nothing here measures sensitivity to it |
| `msm` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_msm.py` | `tests/unit/test_remainder_msm.py` | a **saturated** working model reproduces the per-arm report exactly, at the covariate and at the estimate (`tests/unit/test_msm_submodel.py`, `tests/e2e/test_msm.py`); continuous-dose quadrature and its nonzero density-ratio score are pinned in `tests/unit/test_msm.py` and `tests/unit/test_continuous_msm.py`; the registered point study adds an unsaturated, nonuniform projection and uniform-weight mutation; `tests/unit/test_msm_projection_weights.py` fits an arm-share weight declared `"known"`. Its curve is the fixed-weight curve, and its `msm[W]` standard error is 0.742 of the estimated-weight one. That gap is why `weights_kind="estimated"` is refused. `tests/unit/test_msm_design_declaration.py` fits a design centred at the sample mean of `W` and declared `"known"`. Its curve is the fixed-centre curve, and its intercept standard error is 0.893 of the estimated-centre one. That gap is why `design_kind="estimated"` is refused | the registered row is identity-link, ordinary, fixed-weight, two-arm evidence. A weight or a design computed from the sample and declared `"known"` is not detected. The saturated identity remains blind to the curvature term, the alternation's restart, and the loss of exact double robustness. The continuous test uses a linear truth; a nonlinear continuous-dose Gateaux oracle remains absent |

## Where the repeated-sampling evidence lives

The table above asks whether each parameter is implemented correctly. The registered studies ask
the complementary question. Apply a complete estimator to samples from a known law, and does its
bias and uncertainty behave as its source theory predicts?

Those studies are summarised in the
[implementation validation grid](method-evidence/validation-grid.md). Their test-by-test
results are the [implementation validation studies](method-evidence/index.md). The two halves are
different instruments and neither one substitutes for the other.

Fixed observation weights are an estimator variant over the same registered point-treatment
targets. Their registered [weighted point-treatment study](method-evidence/weighted-point-treatment-tmle.md)
tests the tilted population target, exact weighted influence-curve efficiency, and the consequence
of omitting the weights on identical selected samples. The separate
[learned-nuisance weighted study](method-evidence/learned-weighted-point-treatment-tmle.md) sends
the weights through both regressions. Its learner-only control moves the untargeted plug-in to the
selected target, while a correct treatment mechanism repairs the targeted estimate.

The registered [ordinary weighted longitudinal
study](method-evidence/ordinary-weighted-end-of-study-longitudinal-tmle.md) and [cross-fitted
weighted longitudinal
study](method-evidence/cross-fitted-weighted-end-of-study-longitudinal-tmle.md) extend that
instrument to two treatment nodes, censoring, static and dynamic plans, and correlated contrasts.
Both use exact-size selected samples and fixed inverse-selection weights. Their independent
property laws include separate controls for omitting weights from the full estimator and from the
nuisance learners alone. These are reporting rows: their committed red cells remain limitations,
not hidden successes.

## Estimator variants over registered targets

`CTMLE` and `DRTMLE` estimate the same registered `ey` and `ate` targets as `TMLE`, so they
do not add target-registry rows. Their multi-arm estimator constructions now have separate
registered repeated-sampling records for [ordinary TMLE](method-evidence/ordinary-multi-arm-tmle.md),
[DR-TMLE](method-evidence/multi-arm-dr-tmle.md),
[outcome-adaptive C-TMLE](method-evidence/outcome-adaptive-multi-arm-c-tmle.md), and
[selector C-TMLE](method-evidence/selector-based-multi-arm-c-tmle.md). Binary compatibility
remains covered by the existing C-TMLE and DR-TMLE rows, which continue down their original
branches.

A `discrete` C-TMLE fit with one declared candidate, equal to the full adjustment set, is the
ordinary TMLE. The `TMLE` studies are its evidence, and
`tests/unit/test_ctmle.py::TestEquivalenceWithPlainTmle` pins the identity to `1e-12`.

Complete-outcome cross-validated DR-TMLE is a construction over the same targets, not a registry
addition. Its source audit maps the pinned R `cvFolds` path to
`cross_fit=True, reduced_crossfit="pooled", targeting_scheme="pooled", cv_evaluation=False`.
Primary and reduced predictions are out of fold. One global alternation follows, then a
whole-sample plug-in mean, then `cov(IC) / n` from the rowwise corrected curve.

A stratum-conditional parameter of `strata=` is the registered marginal parameter of the law
inside one baseline stratum. `stratum_alias` names it, so it adds no registry stem. The table gives
its instruments and what each one cannot see.

| instrument | what it checks | what it cannot see |
| --- | --- | --- |
| `tests/unit/test_stratified_influence_exact.py` | each stratum curve equals $I(V=s) D_s / P_n(V=s)$ at targeted predictions that moved, with within-stratum arm shares for the ATT and the ATC. Two mutations fail it: the marginal arm share, and a stratum mass of 1 | the truth, because it checks an identity at fitted values |
| `tests/unit/test_stratified_targets.py` | the joint score blocks, the weighted stratum masses, and the refusals of a stratum parameter in a sensitivity analysis | coverage |
| the [baseline-strata study](method-evidence/stratified-point-treatment-tmle.md) | bias, coverage, SE calibration and efficiency of each stratum parameter against an exact law, the band across strata, double robustness, and a stratum-targeting control. It pairs the arm means and the ATE with R `tmle3` `tmle_stratified` | ATT, ATC and PAR against a comparator, weights, clusters, missing outcomes, multi-arm treatment, and flexible learners |
| `tests/unit/test_stratified_incremental_exact.py` | each stratum's outcome and mechanism blocks are zero, and each stratum curve is Kennedy's Theorem 2 curve at the targeted pair; the pooled fit equals the subset fits. Four replacement mutations fail it: a marginal mechanism column, a marginal outcome column, no mass in the un-scaling, and no mass in the embedding | the truth; a stratum curve away from the targeted pair |
| `tests/unit/test_stratified_msm_exact.py` | linked stratum coefficients and curves equal the expanded-design MSM, the curve equals a complex-step Gateaux derivative of $\beta_s$, the marginal record equals the unstratified fit bit for bit, and the stratum rank rule refuses before any learner. Two mutations fail it: blocks at the marginal coefficients, and the marginal fluctuation's regression in the stratum estimates | coverage; a nonuniform projection weight with strata |
| `tests/unit/test_stratified_continuous_msm_exact.py` | identity and logit dose coefficients equal the expanded-design dose MSM; a marginal column fails it | the dose truth, because the oracle is the shipped path |
| `tests/unit/test_stratified_natural_course_exact.py` | the in-sample and stacked natural-course blocks are zero, each curve is the analytic one, each stratum equals its subset fit, each stacked standard error is the raw second moment of its curve, and the joint route's PAR is the difference of its stratum curves | the stacked arm-indexed targets, which refuse strata |
| `tests/unit/test_stratified_drtmle_exact.py` | on every route and guard, each stratum estimate and curve equal a subset fit, and the marginal is the mixture. Reductions pooled across strata fail it | the truth; more than one alternation round, whose global stop rule ends the strata together |
| `tests/unit/test_simulated_confounding_strata.py` | the replay of a stratum alias for each new composition | interval validity of the replay |

A cross-fitted clustered `LTMLE` fit estimates the registered longitudinal targets with the
cluster as the unit, so it adds no registry stem.
[Longitudinal clusters](longitudinal-tmle.md#clusters) states the step. The table gives its
instruments and what each one cannot see.

| instrument | what it checks | what it cannot see |
| --- | --- | --- |
| `tests/unit/test_clustered_cross_fitted_ltmle.py` | whole-cluster folds, grouped inner folds, a held-out cluster unseen by a site-memorizing learner, the cluster-summed variance of every reported and derived estimate, the size term, the $t$ reference, the band and the bootstrap kind. Each check has a mutation control that fails it | coverage, and clustering of a competing-risk fit beyond the exact variance identity |
| `tests/unit/test_clustered_longitudinal_laws.py` | the study law keeps the unclustered truths exactly and carries a within-cluster correlation | an informative cluster size, which the law does not carry |
| the `clustered-cross-fitted-ltmle` and `few-cluster-cross-fitted-ltmle` studies | coverage, SE calibration and an IID control at 100 clusters, paired with R `lmtp`, and the $t$ reference at 20 and 30 clusters | an informative cluster size, a competing-risk fit, and 4 to 19 clusters |

Repeated stacked point-treatment CV-TMLE likewise has a separate registered
[reporting-policy study](method-evidence/repeated-cross-fitting.md). It reuses the shared CV-TMLE
laws and structural checks under the median report. The point estimate is the median over complete
fold draws. Its variance is the median of each draw's variance plus squared displacement from that
point. zEpid independently implements the same aggregation formula. Its fold training and targeting
construct a different estimator, so the registered full-method equivalence artifact remains empty.

`tests/unit/test_drtmle_crossfit.py::TestTheCanonicalSourceCVContract` pins the last three choices
on 101 rows over folds of sizes 34, 34, and 33; both the equal-fold plug-in and cross-validated
variance are nonzero mutations there. The same module's training-row and longhand cell-mean tests
pin the first choice for both pooled and nested reduced fits. This is structural implementation
provenance rather than R numerical parity. It does not establish a published cross-fitted theorem,
and it supplies no corrected fold-aggregation result for `targeting_scheme="fold"` or
`cv_evaluation=True`; both refusals remain pinned in `tests/unit/test_drtmle_fit.py`.

The complete-outcome alternation defaults to `update_order="drtmle"`, the canonical R-package
sequence; `"benkeser"` names the published six-step recursion. The canonical round refits only
`gr1`/`gr2` after equation (9) and only `qr` after equation (8). The fit-count mutation in
`tests/unit/test_reduction_alternation.py` requires six reduced-column fits per two-arm univariate
round rather than the former twelve, while the theorem, Gateaux, remainder, score, and correction
gates establish that the returned collection still satisfies the estimator identities. The same
module pins both source-specific solve orders. `tests/unit/test_drtmle_fit.py` separately requires
`validate()` to warn, and to preserve that warning through serialization, when equation (10)'s
historical `ill_conditioned` counter is positive despite a passing final score check.

The bivariate alternative is also a construction over these targets. Its acceptance chain
starts at the bivariate remainder representation of Benkeser, Carone, van der Laan and Gilbert
(2017), Appendix A, for the construction that their Section 3.1 cites as Theorem 3 of van der
Laan (2014), and then separately pins the
pinned R source's two-column reduced probability and `(gr-g)/(g*gr)` outcome direction.
Three modules carry it. `tests/unit/test_reduced_regressions.py` uses a finite-support
joint-conditioning tie that fails if either generated design column is replaced by `W`.
`tests/unit/test_reduced_submodel.py` keeps a nonzero deliberate mutation omitting `1/g`.
`tests/unit/test_oracle_reductions.py` injects the exact bivariate conditional expectations with
both primary nuisances wrong, recovers `ey1`, `ey0`, and `ate`, and requires every score and
correction identity to pass. The production cross-fitted fit
and serialization round trip are pinned in `tests/unit/test_drtmle_fit.py`. `gr2` is `NaN` on this
path by design, so accidental use of the absent univariate-only regression cannot silently return
zero. For multiple treatment levels, the pinned R source applies those same branches once per
requested arm. `tests/unit/test_reduced_regressions.py` checks the three arm-specific joint
conditional probabilities against an exact finite-support law, and
`tests/unit/test_multi_arm_collaborative.py` carries a misspecified, nonzero end-to-end witness:
all armwise corrections are present and the score and correction gates pass. This is evidence for
the source's armwise extension; it does not rewrite van der Laan's binary theorem as a multi-arm
one.

Controlled direct effects now add registered repeated-sampling evidence to their exact-law,
Gateaux, remainder, and mutation chain. The
[controlled direct-effect study](method-evidence/controlled-direct-effect-tmle.md) fits both
intermediate levels, distinguishes the outcome-regression half of robustness from the joint
treatment-intermediate-observation mechanism half, and includes one nonzero control for each
mechanism. Its pinned R comparison recodes each requested level to `tmle`'s first result; a frozen
fixture separately records why the native second result is not a valid exact-nuisance comparator.

The stacked CV-TMLE of arm-indexed means and contrasts with missing outcomes is also an estimator
variant over registered targets. `tests/unit/test_arm_indexed_stacked_mar.py` holds its nonzero
witnesses. Each witness builds its value by hand, and a deliberate mutation makes it fail. The
mutations cover the clever covariate, the respondent mask, both inverse factors, the contrast
pairing, the same-row covariance, the band, and the held-out scale. The registered
[stacked arm-indexed missing-outcome study](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md)
adds repeated-sampling evidence. It also compares R `tmle` 2.1.1 with the same stitched nuisance
predictions.

The randomized missing-outcome DR-TMLE surface is likewise an estimator variant over those
registered targets. Its acceptance evidence is Díaz & van der Laan (2017), §2.1, equation (6),
Theorems 1–2, and equations (11)–(13), plus `tests/unit/test_drtmle_missing.py`. That module keeps
all five reduced regressions and the separate `D_A`, `D_Delta`, and `D_Y` corrections, with a
nonzero finite-array witness that fails if the treatment correction is silently absorbed into
the observation correction. End-to-end fits require all three correction rows to agree with the
scores actually solved, exercise learned and known-randomization paths, refuse partial guards,
and round-trip the five reductions plus the targeted observation mechanism. A rowwise clever-
covariate identity verifies that treatment and observation are bounded separately before their
product is formed.

The registered
[randomized missing-outcome DR-TMLE study](method-evidence/randomized-missing-outcome-dr-tmle.md)
keeps the deliberately misspecified-outcome/correct-observation half of the union model as
repeated-sampling evidence beyond score identities, and directly contrasts the empirical
correction scores before and after the five-reduction cycle. Its R `drtmle` comparison is limited
to the shared both-correct limit because the package exposes one joint treatment-response
mechanism; numeric agreement is not an acceptance gate for the separate reductions.
Above two arms, the same construction runs once per arm indicator, and the arm estimators are
stacked. The instruments are in the table. The column at the right says what each one cannot see.

| instrument | what it checks | what it cannot see |
| --- | --- | --- |
| exact three-arm law, `tests/discrete_law_mar_multi.py` | the oracle estimate equals the truth and the curve equals the EIF, to `1e-12` | the three correction blocks, which vanish at the truth |
| outcome and observation drifts on that law | each block is nonzero at every arm, the estimate stays exact, and the stored scores equal the reported corrections | a mutation on the treatment side, because the oracle `g` solves every `W`-measurable treatment equation |
| mutation controls on a finite-sample fit | a rolled arm mechanism, a swapped response, and rolled observation and outcome-drift columns each fail the correction check | a sign error in the curve, because a solved block has mean zero either way |
| independent one-indicator reference | each arm's estimate and curve equal a test-local implementation of Steps 1 to 5, under live drift | the two-arm route, which takes one shared tilt |
| [multi-arm missing-outcome study](method-evidence/randomized-multi-arm-missing-outcome-dr-tmle.md) | repeated-sampling coverage, size, power and the simultaneous band at three arms | the five-reduction construction in the R pairing, which uses the composite mechanism of Díaz and van der Laan's page 25; and the estimated-mechanism route `randomized=True`, because every study fit passes known probabilities. The unit tests cover that route |

All of these are in `tests/unit/test_drtmle_missing_multi_arm.py` except the study. The R pairing
covers the both-correct limit only.

**The composite indicator.** An observational missing outcome and a declared missing treatment take
the [composite construction](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator),
for the DR-TMLE variants and for the ordinary TMLE. It is an estimator variant over the
registered arm means and contrasts, and adds no target. The instruments are in the table.

| instrument | what it checks | what it cannot see |
| --- | --- | --- |
| two exact laws, `tests/discrete_law_composite.py`, at two and three arms | at the oracle factors the estimate equals the truth and the curve equals the EIF to `1e-12`, at every guard, both reductions, and with a `W`-dependent weight. Each witness limit is computed and checked at import | the corrections, which vanish at the truth |
| outcome and mechanism drifts on those laws | the estimate stays exact, `D*_g` is live under the outcome drift and `D*_Q` under the mechanism drift, and the stored scores equal the reported corrections; both drifts together miss | a defect that is exact on a saturated law |
| nonzero witnesses and mutation controls (E5 to E12, E19) | an omitted treatment or outcome observation factor, a renormalized or rolled composite, the two-arm complement form, a treatment factor fitted on every row, a view that masks by `Delta` alone, and a reader of the raw treatment each move the estimate to its computed limit or fail an identity | a defect the mutation list does not name |
| exact reductions (E13, E14) | on complete data at three arms the composite route is the shipped DR-TMLE bit for bit; with `delta=` alone its covariate and its whole TMLE are the shipped missing-outcome ones bit for bit | the two-arm composite, which is armwise where the complete-data route is not; the per-arm reference covers it |
| independent per-arm reference (E18) | each arm's estimate and curve equal a test-local implementation of Benkeser et al.'s steps on `(W, C_a, C_a Y)` under live drift, and a sign mutation in the curve fails it | each arm under both guards together under the outcome drift, where equation (10) has a near-zero covariate and its fixed point is not numerically identified |
| the [observational missing-data study](method-evidence/observational-missing-data-dr-tmle.md), `composite-missing-drtmle` | repeated-sampling coverage, drift robustness, the rate, calibration, size, power, the simultaneous band, the correction cycle, and a complete-case control, at two and three arms; a regime mean and an arm MSM slope on the composite TMLE; R `drtmle` 1.1.2 runs the same construction. The three-arm `low` contrast's both-wrong control uses a drift of its own | partial guards, the bivariate reduction, weights and clusters, which have no coverage cell; the exact laws and the parent studies cover them |

No instrument can detect a violation of the treatment condition, `Y(a)` independent of `Delta_A`
given `(A, W)`. No observed-data check can, because the data hold no outcome of a row whose
treatment is unrecorded under the other arm.

Cross-validated DR-TMLE missing-data compositions are not
covered, and neither is `treatment_probabilities=` under `n_bootstrap=`, which is refused because
the array cannot be reindexed to a replicate's resampled rows at any `guard=` because the array
is row-aligned however few equations are being solved. An unguarded `delta=` fit with known
probabilities and `cross_fit=False` is a plain TMLE and is accepted as one; `_FailIfFit` is the
witness that the supplied array reaches the fit rather than the refusal merely being gone. With
`cross_fit=True`, the same fit is refused at every `guard=`, including `guard=()`.

**Bounding the two mechanisms separately is what the scope label had to learn.** `contract`
measured its truncation witnesses on the treatment mechanism alone. That is blind in exactly the
regime this construction is for. A randomized trial's `g` is flat by design and cannot clip, so a
fit whose `P(Delta=1|A,W)` was pinned on a fifth of its rows was certified `"theorem"`.
`TestTheContractSeesTheObservationTruncations` is now a pair of fits.

One is well-behaved. The second has its observation mechanism pinched while its treatment
mechanism demonstrably is not. It is asserted to leave every pre-existing column inactive, so a
bound-active verdict there can come only from the two new witnesses. The same fixture carries the positivity
report's derived `P(A=a,Delta=1|W)` row, which counted its truncation against a product of floors
the estimator never applies: 1.1% reported against 20.1% actual, with the old rule kept beside it
as the control.

**The exact law alone is not evidence for either construction, and this is worth writing
down rather than leaving to be rediscovered.** Handed the oracle nuisances,
`tests/discrete_law_multi.py` makes every new term vanish: `max|Qr| = 1.9e-17`, `gr2 = 0`
exactly, the mechanism's `epsilon` is `[0, 0, 0]` and the targeted mechanism equals the
initial one to `2.8e-17`. A fit that recovers all five parameters to `2e-15` there has
therefore said nothing about equation (9), the corrections, or the outcome-adaptive design.
Reversing the columns of the targeted mechanism leaves the exact-law assertions,
`score_check()` and `correction_check()` all passing. So each construction carries its own
nonzero instrument:

| construction | instrument | what fails without it |
| --- | --- | --- |
| armwise equation (9) | `test_armwise_mechanism_matches_an_independent_glm_solve`; `brentq` solves `drtmle`'s own `fluctuateG` score equation, arm by arm, sharing no code with the solver | any change to the response, offset, covariate or arm alignment; agreement is to `1e-13` |
| the reported corrections | `test_drtmle_corrections_are_nonzero_and_solved_under_misspecification`; glm nuisances give `max|Qr| ≈ 4e-2`, and the mechanism leaves the simplex | a targeted mechanism that does not move, or an identity that holds only because both sides are zero |
| arm alignment of the exit state | `test_multi_arm_exit_state_solves_each_arms_equation`; it recomputes equation (9) and asserts that a column permutation does **not** solve it | a per-arm quantity read at the wrong arm, which is invisible to any symmetric check |
| `reduced_mechanism_covariate` at `K` arms | `test_multi_arm_reduced_mechanism_covariate_has_the_r_formula` on a nonzero `Qr` | the binary sign convention carried over, which the exact law cannot see |
| the `oat` design | `test_oat_fits_the_treatment_model_on_the_arm_specific_qbar_matrix` and `test_oat_recovers_a_mechanism_generated_by_qbar`; a saturated learner on a law where `Qbar(·, W)` is a bijection of `W`, so the fitted `g` must equal `g_0` exactly | zeroing, permuting or substituting the design, none of which any exact-law or field-name assertion detects |
| selector joint target | `tests/unit/test_ctmle_multi_arm_selector.py`; categorical paths for every selector, explicit component names, and the trace-plus-vector-bias identity | scoring only the first contrast: the nonzero mutation changes the penalty by more than 100 |

The selector uses one shared categorical path. `ey` contributes all `K` arm curves; `ate`,
`rr`, and `or` contribute all `K - 1` reference contrasts. Its pooled cross-validation array
therefore has shape `(candidate, row, component)`, and the penalty sums every component's
variance and squared mean. The finite-support armwise remainder identities in
`test_remainder_multi.py` cover the underlying mean vector and its reference-contrast map;
the ratio targets use the same independently tested delta-method curves as the final report.

The nonzero scientific instruments are now completed by
`tests/unit/test_remainder_multi.py` and
`tests/unit/test_influence_gateaux_multi_collaborative.py`.

The former evaluates every arm's remainder at nuisances that are wrong on purpose,
and takes both DR-TMLE projections from the shipped `reduced_correction_parts` rather than
rebuilding them, against an exactly saturated `ReducedSet` this finite law admits. Its
longhand derivation is kept beside them as an independent oracle, and
`test_the_library_corrections_are_the_longhand_ones` is the assertion that reaches the
library: flipping the sign of either `d_g` or `d_q`, or zeroing one of them, fails the
module. Its OAT entry is a *design-level* boundary. It uses a coarsened `Qbar` whose generated
mechanism cannot be repaired independently of `W`. It is not a check on the shipped OAT
code, which is pinned by `test_oat_fits_the_treatment_model_on_the_arm_specific_qbar_matrix`
in the table above and by the OAT curve check named next.

The latter checks both multi-arm DR-TMLE union-model cells against the complex-step
derivative through real `DRTMLE` fits, and exercises `CTMLE(strategy="oat")` on the regular
exact law where its generated design identifies `W`.

The four registered rows publish replication accounting, truth and comparison verdicts, and
nuisance-regime properties. They also cover root-n ladders and each selector path against one
forced empty path. The reporting-policy rows keep red cells where the evidence does not
support a stronger claim. Registration does not turn an observed limitation into an inherited
binary theorem.

## Longitudinal estimands outside the target registry

`LTMLE` parameters are indexed by regimen, horizon, and sometimes cause rather than by a
`Target`, so they do not belong in the registry-gated table above. They have their own
bidirectional oracle gates in the named Gateaux modules; this table makes the parallel
evidence structure explicit.

| longitudinal variant | parameter and EIF oracle | nonzero or mutation witness | canonical implementation witness | not covered |
| --- | --- | --- | --- | --- |
| end-of-study, static and dynamic regimens | `tests/discrete_law_longitudinal.py`, `tests/unit/test_influence_gateaux_longitudinal.py` | dynamic-rule arm evaluation and dropped-censoring mutations in the Gateaux module; exact-fold and held-out-prediction checks; the pooled cross-fitted targeting longhand and its four mutations in `tests/unit/test_pooled_longitudinal_targeting.py`; loss/design decomposition in `tests/unit/test_longitudinal_msm_submodel.py`; the node-threshold witness in `tests/unit/test_regimen_rule_declarations.py`, where a threshold at the sample mean of a continuous covariate reports 0.775 of the exact standard error | registered ordinary R `ltmle` 1.3-0 study, and a separate registered cross-fitted study against pinned R `lmtp` 1.5.4 with the mechanism supplied to both. That version fits a training-fold fluctuation, so the cross-fitted paired verdicts compare two constructions | no observation weights or time-respecting splits, and `variance.method="ic"`; cleverly's IC-only standard errors do not implement R's default robust truncation-aware variance; a learned rule, refused |
| categorical treatment nodes, static and dynamic regimens | `tests/discrete_law_longitudinal_multivalue.py`, `tests/unit/test_influence_gateaux_longitudinal_multivalue.py` | nonzero quadratic remainder, third-arm probability and dynamic-rule mutations, exact-fold support checks, and the arm-encoding witness in `tests/unit/test_sequential_design.py`; exact-mechanism, design-column, and comparator-transcription gates in `tests/unit/test_categorical_ltmle_method_study.py`; string-label end outcome, censoring, MSM, survival, and competing-risk fits in `tests/e2e/test_ltmle_multivalue.py`; the pooled-targeting longhand on a categorical dynamic-rule fit in `tests/unit/test_pooled_longitudinal_targeting.py` | registered ordinary and five-fold studies against pinned R `lmtp` 1.5.4; both receive exact assigned-arm probabilities, and the cross-fitted row receives identical rowwise folds | deterministic categorical regimens only; no censoring, missingness, continuous dose, weights, clusters, fold repeats, survival, or competing risks. The default band of each registered fit is red in the [default-band study](method-evidence/default-simultaneous-bands.md) |
| known stochastic categorical policies | `tests/discrete_law_longitudinal_policy.py`, `tests/unit/test_influence_gateaux_longitudinal_policy.py`, on mixed plans, a partial-support policy, and the survival, competing-risk and weighted laws | the policy witnesses and mutations M1 to M9 in the Gateaux module, including the targeting witness under a misspecified initial fit, the one-hot bit identity and the recorded-randomizer projection; the pooled cross-fitted longhand in `tests/unit/test_pooled_longitudinal_policy_targeting.py`, with a per-fold fluctuation, an observed-arm fold carry and moved per-level blocks as its mutations; refusals and replay in `tests/unit/test_longitudinal_policy_plumbing.py` | the registered study `stochastic-categorical-ltmle` pairs the in-sample fit with pinned R `lmtp` 1.5.4 through `tests/canonical/lmtp_policy_adapter.R`, which replicates each unit four times; its run is pending | a false `"known"` declaration, which no code can see; a policy that reads the natural value $A_t$ or depends on $P$; no repeated-sampling cell for weights, clusters, competing risks or a continuous outcome |
| absorbing survival curve | `tests/discrete_law_survival.py`, `tests/unit/test_influence_gateaux_survival.py` | the `t-1` risk-set mutation and end-of-study reduction in `tests/e2e/test_ltmle.py`; exact-fold, held-out-prediction, and registered survivor-only controls; the pooled-targeting longhand on a survival fit in `tests/unit/test_pooled_longitudinal_targeting.py` | registered ordinary R `ltmle` 1.3-0 study across both horizons, and a separate registered cross-fitted study against pinned R `lmtp` 1.5.4 | no time-respecting splits, active truncation, observation weights, or competing events. The one-plan curve band passes, and the band over all ten parameters is red under `band-finite-sample` |
| competing-risks cumulative incidence | `tests/discrete_law_competing.py`, `tests/unit/test_influence_gateaux_competing.py` | mutation from all-cause to cause-specific survival in the Gateaux module; a separate finite-sample all-cause risk-set control; one-cause reduction in `tests/e2e/test_ltmle.py`; the pooled-targeting longhand on a competing-risk fit in `tests/unit/test_pooled_longitudinal_targeting.py` | registered ordinary and five-fold studies against pinned R `lmtp` 1.5.4; both use misspecified outcome regressions, exact mechanisms, and a nonzero targeting witness | two causes, two nodes, and one fixed split; no active truncation, learned-mechanism parity, weights, clustering, or eliminated competing event |
| working model over regimen/horizon cells | `tests/discrete_law_longitudinal.py`, `tests/unit/test_influence_gateaux_longitudinal_msm.py` | non-saturated, nonuniform projection law plus exact pooled-design/loss-weight checks in `tests/unit/test_longitudinal_msm_submodel.py`; a projection over policy cells, the policy-cell carry, and logit and survival reductions at one and five folds in `tests/unit/test_longitudinal_policy_msm.py`; the registered study adds untargeted and uniform-weight controls; at five folds, the saturated reductions, the stacked delta-method curve and eight mutations, including a fold-local fluctuation, in `tests/unit/test_cross_fitted_longitudinal_msm.py` | registered ordinary study against a fixed projection of four correlated R `ltmle` 1.3-0 regimen fits. | survival, competing-risk, weighted and clustered projections have fast-tier exact identities only. R `ltmleMSM` uses a quasibinomial projection, so raw coefficient parity would compare different estimands |

### Descriptive longitudinal truncation replay

The truncation grid adds no new parameter or influence-curve claim, so it has no registry row or
coverage study. `tests/unit/test_longitudinal_truncation_refit.py` is its fast implementation
instrument. It checks exact fitted-bound replay, active-bound movement, complete backward
recursion, the out-of-fold mechanism count, parameter-specific counts, replay omissions, persistence,
parallelism, dataframe backends, and combined-report gates. Separate active witnesses cover
end-of-study, survival, competing-risk, and MSM results. The composition cases include dynamic
categorical plans, weights, clusters, and an audit of the random state inside a nested Super
Learner.

Those tests do not establish interval coverage, a preferred bound, positivity clearance, or
external implementation parity under active truncation. The registered rows above remain evidence
for their fixed-bound point estimates and stated inference regimes only.

## Post-fit functionals of reported estimates

These outputs are computed from the influence curves of a fitted result. None is a `Target`, so
none enters the registry-gated table. Each is an exact delta-method or linear-functional
computation, and each row names the instruments that check it.

| functional | built from | instruments | nonzero or mutation witness | not covered |
| --- | --- | --- | --- | --- |
| ratio contrasts, point treatment: `ratio`, and `contrast(scale="ratio")` | two level estimates of a `TMLEResult` | `tests/unit/test_contrast_conveniences.py`: bit identity with the registered `rr` and `or` on the exact binary and three-arm laws | swapped arms give $1/\psi$ and $-IC$; an odds ratio built with the risk-ratio derivative fails; dropping the $1/v$ factor of the ratio-scale contrast fails | a ratio of two contrasts, and a numeric reference |
| ratio contrasts, longitudinal: `LongitudinalResult.ratio` | two `ey_regimen`, `risk_regimen` or `cif_regimen` levels | `tests/unit/test_contrast_conveniences.py` on the end-of-study, survival and competing exact laws; the ratio cells of the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) | a survival view that forgot the complement; a cause swap; the survival odds ratio as the reciprocal of the risk odds ratio | a ratio as a fit-time estimand, and bands over derived ratios |
| Wald test at a declared null: `wald_test` | one estimate and its standard error | `tests/unit/test_contrast_conveniences.py` against the longhand statistic on every scale; `wald_test().pvalue == pvalue` on every estimate | a ratio-scale test that does not log the null | one-sided tests, and a joint chi-square test |
| transformed contrast: `contrast(transform=)` | a smooth function of estimates and a monotone map | `tests/unit/test_contrast_conveniences.py` against the `drtmle` 1.1.2 list-contrast arithmetic, and bit identity of `Transform.log()` with `scale="ratio"` | a dropped slope; an interval mapped back with the forward map; a decreasing map without the sort; a band without the inverse map | the coverage of a user transform. The user's transform defines that estimand |
| RMST and RMTL: `rmst`, `rmtl` | the risks or cause-specific incidences below the horizon | the enumeration truth of `tests/studies/survival_grid_law.py` on its weighted support, and the two-node survival and competing laws, in `tests/unit/test_contrast_conveniences.py`; the RMST cells of the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) | off-by-one horizon ranges; a dropped cause in the competing RMST | calendar time on an unequal grid |
| C-TMLE logistic plug-in diagnostic: `logistic_plugin` | a selector `CTMLE` fit | `tests/unit/test_ctmle_logistic_plugin.py`: a numpy transcription of `calc_varIC`, and R `ctmle`'s own `calc_varIC` on one fit's inputs in `tests/canonical/ctmle_logistic_plugin` | the term at a constant-outcome fit against the complex-step derivative; a flipped sign, a dropped inverse and a dropped intercept | a coverage claim. The status stays `working_mechanism_plugin` |
| `LTMLE` full-refit bootstrap: `n_bootstrap=` | a full refit of the estimator per resample | `tests/unit/test_ltmle_bootstrap.py`: the subset round trip by field name, a replicate as a plain fit on its resample, a committed `canonical-ltmle` row refitted at `n_bootstrap=0` and 2, the per-kind licence, and the replicate draws of `ratio`, `rmst` and `contrast`; the bootstrap cells of the [full-refit bootstrap study](method-evidence/full-refit-bootstrap-and-derived-contrasts.md) | a refit-count witness that fails when a replicate reuses the fit's mechanism; a shrunken percentile interval that the study's control must fail | a data-adaptive nuisance, and any law or size the study does not run |

## A simulated law is an instrument too, and it can be wrong the same way

A coverage study is only evidence if the number it calls the truth is the number an adjusted
fit is estimating. Two of the generators shipped for clustered inference failed that: the
per-cluster latent drove the treatment mechanism as well as the outcome and was not emitted
as a covariate, so the declared ATE of `1.0` was not identified. The identified value was
`1.83`, and every interval missed by six to ten standard errors while the docstring claimed the
counterfactual means were unchanged. The longitudinal generator failed it twice, since its
shared effect also tilted the outcome on the logit scale, where
`E_S[expit(eta + gamma S)] != expit(eta)` moves the means whatever the mechanism does.

Both now put the sharing where it does not confound, and both assert it. `clustered_dgp` makes
the latent an **effect modifier** independent of treatment. The longitudinal generators share part
of `L2`'s own noise, and preserve its conditional law exactly, so a clustered draw's `truth` is the
*same number* as an unclustered one's. See `tests/unit/test_datasets.py` and
`tests/unit/test_datasets_longitudinal.py`.

The second half is the part that is easy to lose. Removing the confounding *also removes the
clustering*, because a shared additive residual reaches the influence curve only through
`E[H | W]`, which is zero for a well-specified `g`. An additive shared residual measures a design
effect of 1.00. The arm-interacted form `clustered_dgp` ships measures about 1.95 at ten rows per
cluster, which its docstring records. So each generator carries a nonzero within-cluster witness
beside its identification test; without one, a correct-looking fix leaves the study measuring
nothing.

The registered [clustered point-treatment study](method-evidence/clustered-point-treatment-cv-tmle.md)
adds repeated-sampling evidence for that witness. It runs on the **binary** law
`clustered_dgp(family="binomial")`, whose quadrature truths are `ate = 0.10405` and `rr = 1.2513`,
and not on the Gaussian law above. The move is the outcome-scale rule: a cross-fitted continuous
outcome needs a declared support, and this law has none.

The study compares five-fold clustered TMLE against pinned R `lmtp` on one grouped partition that
`cleverly` draws and the R adapter retains. The partition balances nothing. Its IID control reuses
the same rows, estimates, and influence curves. Only cluster aggregation changes.

## What this table says is missing

Read down the **not covered** column and one thing recurs: the **theorem** column is empty
for every row. The tree contains one such instrument: `tests/unit/test_theorem_drtmle.py`,
which checks `DRTMLE` against Benkeser et al.'s Theorem 1 at values where the correction
does not vanish. This instrument exists because that variant's corrections are zero row by row at
correct nuisances, so the exact-law instruments were blind exactly where it mattered.

The arm, regime, shift, tilt and MSM axes are not in that position: their influence curves
do **not** vanish at the truth, so the Gateaux comparison is anchored where the quantity
lives rather than where it disappears. That is the argument for the empty column, and it is
an argument rather than a measurement. That is why it is written here, where the next
person to add an estimand will read it, instead of being left to be re-derived.

**The condition that would fill the column** is a target whose curve contains a block that
is zero at the truth. Registering one means the row above is no longer available, and the
new row needs a check against its source's own theorem, at a value where its block does not
vanish, before the estimand is reported.
