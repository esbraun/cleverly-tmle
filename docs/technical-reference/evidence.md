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
| `ey_obs` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | none | its empirical influence curve is `w(Y - E_w[Y])`; it is zero-mean without a targeting equation | missing outcomes are refused until the MAR natural-course score is derived; no separate remainder exists because the complete-data estimator is empirical |
| `par` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | none | `IC_par == IC_ey_obs - IC_ey0` on the binary oracle | missing outcomes and controlled intermediates are refused; the exact-law identity is blind to a defect shared by both component curves |
| `paf` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | none | its curve is the delta-method transform of `ey_obs` and the reference-arm mean | defined only for a binary outcome with positive observed risk; small-sample coverage near zero risk is not established |
| `rr` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | the ratio's curve is the delta-method transform of the levels' (`tests/unit/test_influence_gateaux.py`) | the log-scale interval's small-sample coverage is a simulation claim, not an identity |
| `or` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | the odds ratio's curve is the delta-method transform of the levels' (`tests/unit/test_influence_gateaux.py`) | as with `rr`, the log-scale interval's small-sample coverage is a simulation claim rather than an identity; and nothing here pins the odds ratio apart from the risk ratio at values where the two are close |
| `ey_regime` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_regime.py` | `tests/unit/test_remainder_regime.py` | a degenerate regime equals the static arm it puts all its mass on (`tests/unit/test_regimes.py`) | a rule that is not deterministic; the density is evaluated once at fit time and the fit answers for that evaluation |
| `ate_regime` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_regime.py` | `tests/unit/test_remainder_regime.py` | the contrast is the difference of the means (`tests/unit/test_regimes.py`) | a rule that is not deterministic, as for `ey_regime`; and the contrast's identity is a relation between two reported numbers, so a defect common to both means survives it |
| `ey_ipsi` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux_ipsi.py`, `tests/unit/test_influence_gateaux_ipsi_mar.py` | `tests/unit/test_remainder_ipsi.py`, `tests/unit/test_remainder_ipsi_mar.py` | `psi(delta=1)` equals `mean(Y)` row by row whatever the nuisances are (`tests/unit/test_ipsi_fit.py`); this detects an alternation that exits with one equation open | this is the one estimand that is **not** doubly robust: every remainder term carries `(ghat - g0)`, so a consistent `Qbar` cannot substitute for a consistent mechanism. The alternation's linear rate is measured, not bounded |
| `ate_ipsi` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux_ipsi.py`, `tests/unit/test_influence_gateaux_ipsi_mar.py` | `tests/unit/test_remainder_ipsi.py`, `tests/unit/test_remainder_ipsi_mar.py` | `psi(delta=1)` equals `mean(Y)`, and the contrast of two tilts at `delta=1` is zero (`tests/unit/test_ipsi_fit.py`) | not doubly robust, as for `ey_ipsi`; with `delta=` present the `psi(1)` check changes meaning. It is then the MAR-identified `E[Y]` and the complete-case mean is the wrong answer, so the check must not be read across that case |
| `ey_shift` | `tests/discrete_law_shift.py` | `tests/unit/test_influence_gateaux_shift.py`, `tests/unit/test_influence_gateaux_shift_cde.py` | `tests/unit/test_remainder_shift_cde.py` | the negative control: a shift's mean equals the induced stochastic regime's, and its **curve does not** (`tests/unit/test_influence_gateaux_shift.py`) | there is no plain-shift remainder module. The rate is measured only with a third nuisance (`_shift_cde`). A Gateaux check on an exact law cannot see a counterfactual block, so `tests/unit/test_shift_submodel.py` and `tests/unit/test_shift_fit.py` pin those structurally and at `epsilon != 0` instead |
| `ate_shift` | `tests/discrete_law_shift.py` | `tests/unit/test_influence_gateaux_shift.py`, `tests/unit/test_influence_gateaux_shift_cde.py` | `tests/unit/test_remainder_shift_cde.py` | the same negative control as `ey_shift`, taken on the contrast (`tests/unit/test_influence_gateaux_shift.py`) | no plain-shift remainder module, as for `ey_shift`; and the MNAR tilt is refused on this axis by name, so nothing here measures sensitivity to it |
| `msm` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_msm.py` | `tests/unit/test_remainder_msm.py` | a **saturated** working model reproduces the per-arm report exactly, at the covariate and at the estimate (`tests/unit/test_msm_submodel.py`, `tests/e2e/test_msm.py`); continuous-dose quadrature and its nonzero density-ratio score are pinned in `tests/unit/test_msm.py` and `tests/unit/test_continuous_msm.py` | the saturated identity is blind to the curvature term, the alternation's restart, and the loss of exact double robustness. A saturated model *fits*, which a projection does not promise. The continuous test uses a linear truth; a nonlinear continuous-dose Gateaux oracle remains absent |

## Where the repeated-sampling evidence lives

The table above asks whether each parameter is implemented correctly. The registered studies ask
the complementary question. Apply a complete estimator to samples from a known law, and does its
bias and uncertainty behave as its source theory predicts?

Those studies are summarised in the
[implementation validation grid](method-evidence/validation-grid.md). Their test-by-test
results are the [implementation validation studies](method-evidence/index.md). The two halves are
different instruments and neither one substitutes for the other.

## Estimator variants over registered targets

`CTMLE` and `DRTMLE` estimate the same registered `ey` and `ate` targets as `TMLE`, so they
do not add registry rows. Their multi-arm constructions have separate evidence in
`tests/unit/test_multi_arm_collaborative.py`. Binary compatibility remains covered by the
existing C-TMLE and DR-TMLE suites, which continue down their original branches.

Complete-outcome cross-validated DR-TMLE is a construction over the same targets, not a registry
addition. Its source audit maps the pinned R `cvFolds` path to
`cross_fit=True, reduced_crossfit="pooled", targeting_scheme="pooled", cv_evaluation=False`.
Primary and reduced predictions are out of fold. One global alternation follows, then a
whole-sample plug-in mean, then `cov(IC) / n` from the rowwise corrected curve.

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
starts at van der Laan (2014), Theorem 3 and the bivariate remainder, then separately pins the
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

The randomized missing-outcome DR-TMLE surface is likewise an estimator variant over those
registered targets. Its acceptance evidence is Díaz & van der Laan (2017), §2.1, equation (6),
Theorems 1–2, and equations (11)–(13), plus `tests/unit/test_drtmle_missing.py`. That module keeps
all five reduced regressions and the separate `D_A`, `D_Delta`, and `D_Y` corrections, with a
nonzero finite-array witness that fails if the treatment correction is silently absorbed into
the observation correction. End-to-end fits require all three correction rows to agree with the
scores actually solved, exercise learned and known-randomization paths, refuse partial guards,
and round-trip the five reductions plus the targeted observation mechanism. A rowwise clever-
covariate identity verifies that treatment and observation are bounded separately before their
product is formed. A `slow` consistency study at `n = 20,000` keeps the deliberately misspecified-
outcome/correct-mechanism half of the union model as statistical evidence beyond score identities.
The canonical R package's missing-data path is provenance only; no numeric parity is an
acceptance gate. Cross-validated, observational, and missing-treatment compositions are not
covered, and neither is `treatment_probabilities=` under `n_bootstrap=`, which is refused because
the array cannot be reindexed to a replicate's resampled rows at any `guard=` because the array
is row-aligned however few equations are being solved. An unguarded `delta=` fit with known
probabilities is a plain TMLE and is accepted as one; `_FailIfFit` is the witness that the
supplied array reaches the fit rather than the refusal merely being gone.

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

`tests/e2e/test_coverage_slow.py::TestMultiArmCollaborativeCoverage` is the repeated-
sampling regression guard: it requires finite, non-dropped replicates, controlled bias and
non-collapsed coverage relative to TMLE. It is deliberately not a nuisance-rate experiment
and therefore does not establish the stronger collaborative-double-robust theorem, price an
adaptive multinomial `g` correction, or establish OAT asymptotics at a tied, nonregular
generated-regressor design.

## Longitudinal estimands outside the target registry

`LTMLE` parameters are indexed by regimen, horizon, and sometimes cause rather than by a
`Target`, so they do not belong in the registry-gated table above. They have their own
bidirectional oracle gates in the named Gateaux modules; this table makes the parallel
evidence structure explicit.

| longitudinal variant | parameter and EIF oracle | nonzero or mutation witness | canonical implementation witness | not covered |
| --- | --- | --- | --- | --- |
| end-of-study, static and dynamic regimens | `tests/discrete_law_longitudinal.py`, `tests/unit/test_influence_gateaux_longitudinal.py` | dynamic-rule arm evaluation and dropped-censoring mutations in the Gateaux module; exact-fold and held-out-prediction checks; loss/design decomposition in `tests/unit/test_longitudinal_msm_submodel.py` | registered ordinary R `ltmle` 1.3-0 study, and a separate registered cross-fitted study against pinned R `lmtp` 1.5.4 with the mechanism supplied to both | no observation weights or time-respecting splits, and `variance.method="ic"`; cleverly's IC-only standard errors do not implement R's default robust truncation-aware variance |
| categorical treatment nodes, static and dynamic regimens | `tests/discrete_law_longitudinal_multivalue.py`, `tests/unit/test_influence_gateaux_longitudinal_multivalue.py` | nonzero quadratic remainder and a third-arm selection mutation that rejects the binary complement; a non-monotone `glm` mechanism witness for the arm *encoding* in `tests/unit/test_sequential_design.py`; string-label end outcome, censoring, MSM, survival, and competing-risk fits in `tests/e2e/test_ltmle_multivalue.py` | source audit of R `ltmle`, `tmle3`, and the Poulos companion repository, with snapshots recorded in `docs/references.md` | deterministic categorical regimens only; no stochastic categorical policies or continuous longitudinal dose. The exact law is blind to how an earlier arm is *coded* into the mechanism's design: its learners are saturated and partition by distinct design row, under which an ordinal code and a drop-first indicator tuple are a bijection. Only the separate non-monotone `glm` witness separates them, and it covers one node, one link and one truth |
| absorbing survival curve | `tests/discrete_law_survival.py`, `tests/unit/test_influence_gateaux_survival.py` | the `t-1` risk-set mutation and end-of-study reduction in `tests/e2e/test_ltmle.py`; exact-fold, held-out-prediction, and registered survivor-only controls | registered ordinary R `ltmle` 1.3-0 study across both horizons, and a separate registered cross-fitted study against pinned R `lmtp` 1.5.4 | no simultaneous curve bands, time-respecting splits, active truncation, observation weights, or competing events |
| competing-risks cumulative incidence | `tests/discrete_law_competing.py`, `tests/unit/test_influence_gateaux_competing.py` | mutation from all-cause to cause-specific survival in the Gateaux module; one-cause reduction in `tests/e2e/test_ltmle.py` | none | no canonical R comparison: the fixture would not add evidence beyond the exact law unless a distinct finite-sample blind spot is first named |
| working model over regimen/horizon cells | `tests/discrete_law_longitudinal.py`, `tests/unit/test_influence_gateaux_longitudinal_msm.py` | non-saturated, nonuniform projection law plus exact pooled-design/loss-weight checks in `tests/unit/test_longitudinal_msm_submodel.py` | none | evidence applies to `n_folds=1`. Cross-fitted coefficient projections are refused. R `ltmleMSM` uses a quasibinomial projection, so raw coefficient parity would compare different estimands |

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
`E[H | W]`, which is zero for a well-specified `g`. The measured design effect fell from 1.87 to
1.00. So each generator carries a nonzero within-cluster witness beside its identification
test; without one, a correct-looking fix leaves the study measuring nothing.

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
