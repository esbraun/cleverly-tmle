# An external review of the DRTMLE plan

**Kept verbatim as received**, including its own truncated final sentence, so that what
[the roadmap](roadmap.md#what-is-still-open) adjudicates can be read against what was
actually said. It is an input to that plan and not itself the plan: where the two differ,
the roadmap says so and gives its reason — three of the claims below were checked against
the code and are narrower than stated (§1's terminology charge, §3's monotonicity charge,
and §7.1's on weights), and the roadmap records each along with what survives of it.

Its numbering is its own and is unrelated to the roadmap's numbered items.

---

Review of the cleverly-tmle / “causly-tmle” DRTMLE Roadmap

Executive assessment

The roadmap is unusually candid and technically serious. Its central judgment is correct: the DRTMLE variant should not be treated as complete merely because the implementation runs and its internal tests pass. The remaining work is primarily statistical validation and theorem-to-code verification, not software polish.

The roadmap’s four-part structure is mostly sound, but the recommended order should be revised:

1. Verify the estimand, limiting influence function, and algorithm against Benkeser et al. (2017) and drtmle.
2. Repair or explicitly delimit weak-overlap behavior.
3. Build a simulation that actually enters the nonregular regime where ordinary TMLE inference fails but DRTMLE inference remains valid.
4. Only then expand scope.

The current roadmap places the weak-overlap investigation before the theorem/source verification. That is defensible operationally, but methodologically the influence-function derivation is the higher-priority gate: until the reported curve and targeting equations are verified against the theorem, neither the weak-overlap diagnosis nor the coverage study has a fully trusted target.

The implementation already shows several strengths:

• The package clearly distinguishes double robustness of the point estimator from double robustness of asymptotic inference.
• It refuses unsupported estimands instead of silently generalizing equations.
• It records targeting failures and score residuals rather than hiding them.
• It recognizes that oracle nuisance functions cannot demonstrate the inferential advantage of DRTMLE.
• It correctly treats coverage as a repeated-sampling property and demands Monte Carlo error and a sample-size trend.

The largest remaining risks are:

1. The variance correction is still source-transcribed rather than theorem-derived.
2. The implementation may conflate an estimator-specific influence function with the ordinary efficient influence function.
3. The reduced-regression cross-fitting construction needs a clearer theoretical justification.
4. The proposed coverage demonstration needs a more precise data-generating design and success criterion.
5. Weak overlap should probably be a declared unsupported region unless a stable targeting strategy can be justified, not merely debugged until scores become small.

────────

1. What DRTMLE is supposed to establish

For a counterfactual mean

[
\psi_a(P)=E_P{\bar Q_P(a,W)},
]

the ordinary efficient influence function is

[ D_a^*(P)(O)

\frac{\mathbb 1(A=a)}{g_P(a\mid W)}
{Y-\bar Q_P(A,W)}
+
\bar Q_P(a,W)-\psi_a(P).
]

An ordinary TMLE or one-step estimator is consistent if either the outcome regression or treatment mechanism is consistently estimated. But asymptotic normality around the true target generally requires more: the second-order remainder must be (o_p(n^{-1/2})). In the standard point-treatment problem this remainder is governed by a product of nuisance errors.

When one nuisance converges to the truth and the other converges to an incorrect limit, ordinary double robustness still protects consistency, but the remainder can be first order in the error of the consistent nuisance. Therefore the usual influence-curve variance need not describe the estimator’s limiting distribution.

Benkeser et al. develop corrected estimators whose first-order expansion remains valid when either nuisance is consistently estimated. The key idea is to estimate the additional drift term using lower-dimensional conditional regressions and then target the relevant score equations.

This distinction is represented correctly in the repository’s conceptual framing.

Important terminology correction

The repository repeatedly calls the reported corrected curve an “efficient influence curve” or suggests that the extra equations alter the efficient influence function. That wording should be tightened.

The nonparametric efficient influence function of the target parameter at the true distribution remains (D^*(P_0)). Under nuisance misspecification, the estimator can instead be asymptotically linear with an estimator-specific influence function evaluated at nuisance probability limits, containing correction terms such as

[
D^* - D_Q^* - D_g^*.
]

That corrected limiting influence function is generally not the canonical gradient at (P_0), and the estimator need not be efficient under misspecification. The code can still call the ordinary component Dstar, but documentation should distinguish:

• canonical gradient / efficient influence function at the true law, and
• corrected asymptotic influence function under one nuisance misspecified.

This is not cosmetic. It prevents users from interpreting the variant as semiparametrically efficient throughout the union model.

────────

The theorem the implementation must satisfy

The release gate should be stated against Theorem 1 of Benkeser, Carone, van der Laan, and Gilbert (2017), not merely against the behavior of the R package. The following is a notation-adapted statement for this repository. It is a faithful mathematical paraphrase rather than a verbatim quotation; the paper’s displayed equations should remain the authoritative source.

Setup

For one treatment level (a), let

[
\psi_{0,a}=E_0{\bar Q_0(a,W)},
\qquad
\bar Q_0(a,w)=E_0(Y\mid A=a,W=w),
\qquad
g_0(a\mid w)=P_0(A=a\mid W=w).
]

Let (\bar Q_n) and (g_n) converge to limits (\bar Q_1) and (g_1), which need not both equal the truth. Define the ordinary counterfactual-mean influence-function term at candidate nuisances by

[ D_a^*(\bar Q,g,\psi)(O)

\frac{\mathbb 1(A=a)}{g(a\mid W)}
{Y-\bar Q(a,W)}
+
\bar Q(a,W)-\psi.
]

For the paper’s univariate correction, define reduced regressions corresponding to the repository’s objects:

[ Q_r(a,w)

E_0!\left[
Y-\bar Q(a,W)
\mid A=a,\ g(a\mid W)=g(a\mid w)
\right],
]

[ g_{r,1}(a\mid w)

P_0!\left[
A=a
\mid \bar Q(a,W)=\bar Q(a,w)
\right],
]

and

[ g_{r,2}(a\mid w)

E_0!\left[
\frac{\mathbb 1(A=a)-g(a\mid W)}
{g(a\mid W)}
;\middle|;
\bar Q(a,W)=\bar Q(a,w)
\right].
]

The two correction terms used by reduced_corrections are

[ D_{g,a}(O)

\frac{Q_r(a,W)}{g(a\mid W)}
{\mathbb 1(A=a)-g(a\mid W)},
]

and

[ D_{Q,a}(O)

\mathbb 1(A=a)
\frac{g_{r,2}(a\mid W)}
{g_{r,1}(a\mid W)}
{Y-\bar Q(a,W)}.
]

Thus the corrected asymptotic influence function represented in the current code is

[ D_{a,\mathrm{DR}}(O)

D_a^*(O)-D_{Q,a}(O)-D_{g,a}(O).
]

This matches the repository’s intended operation

```text
reported_curve = ordinary_curve - outcome_correction - mechanism_correction
```

subject to the paper-to-code verification described below.

Theorem 1, adapted to the repository

Suppose either

[
\bar Q_1=\bar Q_0
]

or

[
g_1=g_0.
]

Assume the estimated nuisance collection—including the primary nuisance functions and the reduced regressions—satisfies the paper’s targeting condition (5). In implementation terms, this requires the empirical means of the three relevant scores to be negligible at the root-(n) scale:

[ P_n D_a^(\hat{\bar Q}^{,},\hat g^{,*},\hat\psi_a)

o_p(n^{-1/2}),
]

[ P_n D_{g,a}

o_p(n^{-1/2}), \qquad P_n D_{Q,a}

o_p(n^{-1/2}).
]

Also assume that the two remaining second-order terms in the paper’s Appendix B are

[
o_p(n^{-1/2}).
]

Then the targeted plug-in estimator obeys

[ \hat\psi_a-\psi_{0,a}

(P_n-P_0)D_{a,\mathrm{DR}}
+
o_p(n^{-1/2}).
]

Consequently,

[
\sqrt n(\hat\psi_a-\psi_{0,a})
\rightsquigarrow
N!\left(0,,
P_0 D_{a,\mathrm{DR}}^2
\right),
]

and its asymptotic variance is consistently estimated by

[ \widehat{\operatorname{Var}}(\hat\psi_a)

\frac{1}{n} P_n!\left[ \left{ \hat D_{a,\mathrm{DR}}

P_n\hat D_{a,\mathrm{DR}}
\right}^2
\right].
]

When the targeting equations have already centered the estimated curve to the required order, this is equivalent asymptotically to the sample second moment divided by (n).

For the ATE,

[
\psi_{\mathrm{ATE}}=\psi_{1}-\psi_{0},
]

the corrected influence function is the armwise difference

[ D_{\mathrm{ATE,DR}}

D_{1,\mathrm{DR}}-D_{0,\mathrm{DR}},
]

and the variance must be computed from that rowwise difference, preserving covariance between the two treatment-specific means.

The theorem therefore gives exactly the claim DRTMLE is meant to support: root-(n) asymptotic normality and consistently estimable variance when either the outcome regression or propensity score is consistently estimated, even if the other converges to an incorrect limit. The paper emphasizes that the guarantee follows naturally for the targeted estimator; it does not generally hold for the analogous corrected one-step estimator. The published article states this result and recommends the targeted construction for the theoretical guarantee. [Benkeser et al., 2017, Biometrika 104(4):863–880, Theorem 1; DOI: 10.1093/biomet/asx053.]

What the theorem does not say automatically

The theorem does not license every feature currently accepted by the surrounding estimator framework. Each of the following requires a separate argument:

• cross-fitting in exactly the pooled form used by this implementation;
• estimated or survey-derived observation weights;
• multivalued treatment;
• ATT or ATC;
• missing outcomes or an intermediate treatment mechanism;
• stochastic interventions, shifts, incremental interventions, or MSM projections;
• composition with C-TMLE;
• valid inference when the score equations remain of order (n^{-1/2});
• rescue of practical positivity violations through truncation.

The theorem also does not say that the corrected influence function is the canonical gradient at the true law when one nuisance is misspecified. It is the estimator’s asymptotic influence function under the union model. If both primary nuisances are consistent, the reduced residual regressions vanish in the relevant directions, the corrections disappear, and the usual efficient influence function is recovered.

Direct code obligations implied by the theorem

The implementation is theorem-conforming only if all of the following hold:

1. ReducedSet.qr, gr1, and gr2 estimate the exact conditional regressions above.
2. gr1 is the conditional treatment probability and gr2 is the conditional normalized treatment residual, regardless of the reversed names in portions of the R source.
3. reduced_corrections uses the final starred (\bar Q^), (g^), and reduced regressions required by the targeting algorithm.
4. Both correction terms are subtracted.
5. Equations corresponding to (D^*), (D_g), and (D_Q) are each solved to (o_p(n^{-1/2})), not merely to a convenient optimizer tolerance.
6. The Appendix B second-order terms are controlled by the convergence rates of the primary and reduced nuisance estimators.
7. The variance is the empirical variance of the complete corrected curve, not the ordinary TMLE curve and not the sum of component variances.
8. ATE inference uses the difference of arm-level corrected curves.
9. If any required score is not negligible, the reported Wald interval is not justified by Theorem 1.
10. Cross-fitting must either reproduce the theorem’s empirical-process conditions or be supported by an additional sample-splitting argument.

The missing Appendix B rate check

Solving the three empirical equations is necessary but not sufficient. Theorem 1 separately assumes that the remaining second-order terms are (o_p(n^{-1/2})). The roadmap should therefore require explicit tests or simulations showing that the residual remainder after correction has the expected product form involving errors in the reduced regressions and primary nuisances.

A release test should estimate, on a known DGP,

[ \hat R_{\mathrm{remaining}}

\hat\psi_a-\psi_{0,a}

(P_n-P_0)\hat D_{a,\mathrm{DR}},
]

and demonstrate across increasing (n) that

[
\sqrt n,\hat R_{\mathrm{remaining}}\to 0
]

in both off-diagonal regimes. Coverage without this check could be accidental; this check directly tests the theorem’s unresolved assumption.

2. Review of roadmap piece A: theorem and cross-language verification

Assessment

This is the most important item and should become the first gate.

The repository currently computes, per treatment arm,

[ D_{\mathrm{reported}}

D^*

D_Q^*

D_g^*,
]

with

[ D_g^*

\frac{Q_r(a,W)}{g^(a\mid W)}
{\mathbb 1(A=a)-g^(a\mid W)},
]

and

[ D_Q^*

\mathbb 1(A=a)
\frac{g_{r,2}(a\mid W)}{g_{r,1}(a\mid W)}
{Y-\bar Q^*(a,W)}.
]

The implementation also notes the easy-to-miss naming reversal between the paper and R package for gr1 and gr2.

That is a strong start, but the roadmap understates how much must be checked. A one-fit comparison of (\hat\psi) and its standard error against R is necessary but insufficient. It can miss compensating differences in targeting, scaling, truncation, fold construction, and influence-curve centering.

Required deliverables

A1. Theorem-to-code notation map

Create a permanent document or test fixture mapping every theoretical object to code:

|Theory object  |Meaning                                  |Python object                                                 |R object                      |
|---------------|-----------------------------------------|--------------------------------------------------------------|------------------------------|
|(\bar Q_n)     |initial outcome regression               |`nuisance.outcome`                                            |corresponding `Qn` object     |
|(g_n)          |initial treatment mechanism              |`nuisance.propensity`                                         |corresponding `gn` object     |
|(Q_{r,n})      |reduced outcome-residual regression      |`ReducedSet.qr`                                               |package-specific name         |
|(g_{r,1,n})    |reduced conditional treatment probability|`ReducedSet.gr1`                                              |confirm swapped R name        |
|(g_{r,2,n})    |reduced treatment-residual regression    |`ReducedSet.gr2`                                              |confirm swapped R name        |
|starred objects|final targeted values                    |`fluctuation.targeted`, targeted propensity, final reduced set|final R fit slots             |
|(D_Q^*,D_g^*)  |correction terms                         |`reduced_corrections`                                         |`eval_Dstar_Q`, `eval_Dstar_g`|

The map must state conditioning variables, signs, denominators, whether values are initial or targeted, and whether each regression is arm-specific.

A2. Verify theorem assumptions, not only formulas

The implementation review should extract the assumptions required for the theorem, including:

• positivity or bounded inverse treatment probabilities;
• convergence of primary nuisances to probability limits;
• convergence rates for reduced regressions;
• empirical-process/Donsker assumptions or the exact form of sample splitting needed to remove them;
• conditions under which the drift representation is valid;
• whether the targeting algorithm must solve equations exactly or only to (o_p(n^{-1/2}));
• whether all equations are evaluated at initial, iterated, or final starred reduced regressions;
• whether the theorem covers treatment-specific means individually, the ATE contrast, or both;
• whether it covers multivalued treatment;
• whether observation weights are covered.

This matters especially because the Python implementation uses cross-fitted primary and reduced regressions, whereas the original theory may be stated without that exact construction.

A3. Cross-language validation at the component level

Commit deterministic fixtures comparing Python and R for at least:

• initial (Q) and (g) predictions;
• each reduced regression prediction;
• each targeting coefficient or final targeted nuisance;
• (D^), (D_Q^), and (D_g^*) separately;
• the full corrected influence curve;
• (\hat\psi), standard error, and confidence interval;
• empirical means of all target scores.

Use simple user-supplied nuisance arrays or deterministic GLMs first. Do not begin with Super Learner, because learner and fold differences can obscure mathematical discrepancies.

Include deliberately misspecified fixtures. At the truth, the reduced correction terms can vanish and a broken implementation may agree with ordinary TMLE.

A4. Add an independent algebraic check

A cross-language comparison can reproduce the same bug. Add at least one independent test based on the paper’s expansion:

• numerically perturb a discrete law and verify the derivative associated with each correction term; or
• construct a finite-support oracle law and check the exact drift decomposition and score identities.

The repository already uses finite-law/Gâteaux tests elsewhere. DRTMLE needs an analogous independent check, even though the corrected curve is estimator-specific rather than simply the parameter’s canonical gradient.

Recommendation

Piece A should be split into:

• A1: theoretical audit, and
• A2: R parity and independent numerical verification.

Both are release blockers.

────────

3. Review of roadmap piece B: weak overlap and targeting convergence

Assessment

The roadmap correctly refuses to solve the problem by loosening numerical tolerances. A score of order se / sqrt(n) is not negligible for an asymptotic linearity argument.

However, the current diagnosis is too narrow. Weak overlap can break more than the main clever covariate:

• (1/g) inflates the ordinary TMLE score;
• (Q_r/g) inflates the mechanism-targeting score;
• (g_{r,2}) itself has a target involving division by (g);
• (g_{r,2}/g_{r,1}) can become unstable if either numerator is noisy or denominator is small;
• truncating (g) changes both the targeting problem and, indirectly, the reduced-regression estimands;
• positivity deterioration can make the union-model asymptotics practically irrelevant at available sample sizes.

Therefore the score failure may be a structural incompatibility between aggressive truncation and the theoretical equations, not merely a convergence defect.

Problems in the current roadmap

The se / sqrt(n) stopping proxy is circular

The internal targeting loop cannot know the final influence-curve standard error before the final curve is constructed. The roadmap recognizes this, but replacing the final scale by a fixed 1/n threshold is not theoretically equivalent.

A cleaner approach is:

1. define an absolute score criterion that is explicitly (o_p(n^{-1/2}));
2. choose a deterministic sequence such as (c_n/\sqrt n), with (c_n\to0) slowly;
3. separately report a post-fit diagnostic normalized by the estimated standard error.

For finite software, a practical criterion might use both:

[
|P_n S_j| \le \tau_{\mathrm{abs}}/\sqrt n
]

and

[
|P_n S_j| / \widehat{\mathrm{sd}}(S_j) \le \tau_{\mathrm{std}}/\sqrt n,
]

but it should be described as a numerical criterion, not as a direct theorem condition.

Convergence of an alternating algorithm is not established by joint log-likelihood monotonicity alone

Refitting the reduced regressions changes the score directions and fitted nuisance functions. Even if each fluctuation step improves a conditional likelihood, the entire map is not obviously coordinate ascent on one fixed finite-dimensional objective. The documentation currently makes a stronger convergence argument than is justified.

The roadmap should require one of the following:

• identify an actual fixed objective whose value is monotone under all updates, including reduced-regression refits; or
• weaken the claim and treat the routine as an estimating-equation iteration with empirical convergence diagnostics only.

The second is safer unless the theorem or R implementation supplies the stronger result.

Needed weak-overlap study

For every failed and successful fit, record:

• minimum and quantiles of raw and truncated (g);
• effective sample size by treatment arm;
• maximum and high quantiles of every clever covariate;
• distributions of (Q_r), (g_{r,1}), (g_{r,2}), and (g_{r,2}/g_{r,1});
• score contribution concentration by row;
• share of score driven by the most influential 1%, 5%, and 10% of observations;
• targeting coefficients and Hessian condition numbers;
• exact change in scores before and after truncation;
• point estimates and standard errors across a truncation grid;
• whether failures persist when reduced regressions are known/oracle.

This will distinguish:

1. optimization failure;
2. noisy reduced regressions;
3. positivity-driven nonregularity;
4. truncation-induced incompatibility;
5. an incorrect score implementation.

Product decision

Unless the study shows a stable region, DRTMLE should explicitly refuse or strongly warn under weak overlap based on a predeclared diagnostic. A robust policy would be:

• fit and return results;
• mark inference invalid when any required score remains larger than the declared statistical tolerance;
• suppress a standard confidence interval or label it non-valid;
• explain that truncation changes the estimating problem and cannot automatically restore the theorem.

A warning alone is easy to miss for a method whose only purpose is inference.

────────

4. Review of roadmap piece C: the coverage demonstration

Assessment

The roadmap is right that the existing pilot cannot demonstrate DRTMLE’s benefit. A correctly specified low-dimensional parametric nuisance converges too quickly, so ordinary TMLE already satisfies the product-rate condition.

The proposed solution—use an adaptive nuisance slower than (n^{-1/4})—is directionally correct, but the experiment needs more precision. Simply choosing a “flexible learner in enough dimensions” does not establish its rate, and an observed coverage gap can arise from finite-sample instability rather than the intended asymptotic drift.

Recommended simulation architecture

Build two deliberately controlled off-diagonal regimes.

Regime Q: outcome consistent but slow; propensity inconsistent

Construct (Q_0(a,w)) in a nonparametric smoothness class where the chosen estimator has a known or empirically verified rate slower than (n^{-1/4}). Keep the treatment mechanism misspecified with a stable nonzero limiting error.

Examples:

• series regression with dimension (K_n) chosen to produce a controlled bias-variance rate;
• histogram or spline regression with a prescribed smoothing sequence;
• high-dimensional sparse regression with a sequence where prediction error rate is analytically bounded;
• a fixed learner whose approximation bias decreases with (n) at a selected rate.

Avoid a black-box Super Learner as the primary demonstration. It is useful for an applied stress test, but its realized convergence rate is hard to identify and reproduce.

Regime g: propensity consistent but slow; outcome inconsistent

Mirror the design for (g), with probabilities bounded away from zero and one so the simulation tests drift correction rather than positivity failure.

Verify that the intended regime was reached

For each sample size and replicate, estimate empirical nuisance errors against known truth:

[
|\hat Q-Q_0|_2,\qquad
|\hat g-g_0|_2,
]

and their errors relative to the probability limits of misspecified learners. Report log-log rate estimates across sample sizes.

The simulation should demonstrate:

• the consistent nuisance error decreases;
• its rate is slower than (n^{-1/4});
• the misspecified nuisance remains bounded away from truth;
• (\sqrt n R_2) does not vanish for ordinary TMLE;
• the targeted reduced-regression remainder does vanish for DRTMLE.

Coverage alone does not prove that the intended mechanism caused the result.

Outcomes to report

For TMLE and DRTMLE:

• bias;
• root-(n) bias;
• empirical standard deviation;
• mean estimated standard error;
• standard-error ratio;
• coverage;
• interval width;
• rejection rate under a null version;
• targeting failure rate;
• fraction of confidence intervals suppressed as invalid;
• Monte Carlo standard errors for all simulation summaries.

Also directly estimate the empirical drift:

[
\sqrt n,E(\hat\psi-\psi_0)
]

and, where possible, the sample analogue of each correction component.

Sample-size trend

Use at least three sizes, not two, if computationally possible. A useful pattern is:

• ordinary TMLE root-(n) bias grows or fails to settle;
• ordinary TMLE coverage deteriorates;
• DRTMLE root-(n) bias stabilizes;
• DRTMLE standard-error ratio approaches one;
• DRTMLE coverage approaches nominal.

Two sizes can be suggestive; three make the rate story substantially more credible.

Replication count and decision rule

Predeclare the release criterion. For example:

• in both off-diagonal regimes, DRTMLE coverage must be statistically compatible with 0.95 at the largest size;
• ordinary TMLE must under-cover by a practically meaningful margin in at least one regime;
• DRTMLE’s standard-error ratio must be near one;
• failure rates must remain acceptably low;
• the conclusion must survive at least one independent seed batch.

The roadmap currently allows “no gap found” as an honest outcome. Keep that. In that event, do not advertise DRTMLE as a production feature; retain it as experimental or remove it from the public API until a useful operating regime is demonstrated.

────────

5. Review of reduced-regression cross-fitting

Assessment

This is the most underdeveloped theoretical issue in the current roadmap.

The implementation reuses the primary nuisance folds to fit reduced regressions whose covariates are out-of-fold nuisance predictions. The code documentation correctly notices a subtle dependence: training-row design values can come from models that saw the validation fold.

The current justification—an independent split does not remove the contamination, while fold-specific full-sample designs create train/test covariate shift—is thoughtful but not enough to establish asymptotic validity.

Recommended next step

Add a specific theory task:

> Determine a cross-fitting construction for the primary nuisances and reduced regressions that satisfies the empirical-process conditions of the DRTMLE expansion, and document whether the current fold reuse does so.

Candidate constructions to evaluate:

1. Nested cross-fitting
Outer folds define evaluation rows. Within each outer training sample, fit all primary and reduced nuisance functions without using the outer validation sample.
2. Double sample splitting
Use separate subsamples for primary nuisance construction, reduced-regression construction, and final score evaluation.
3. Cross-fitted estimating equations with fold-specific reductions
Construct all reductions within each outer training set and evaluate only on its held-out fold.
4. Current pooled construction with a proof that induced dependence is higher order
This may be valid, but it needs an argument rather than an implementation note.

The expensive nested construction can serve as a reference implementation even if a cheaper pooled version is ultimately retained.

Testing implication

The R package’s agreement is not enough here if its implementation predates modern cross-fitting or uses a different dependence structure. Python may intentionally improve on it, but then the new construction needs its own validation.

────────

6. Review of roadmap piece D: bivariate reduction and multivalued treatment

Bivariate reduction

This is a reasonable post-validation extension because van der Laan’s original construction supplies a direct theoretical basis. But it should not be described as mere transcription until the precise score, influence correction, and targeting algorithm are mapped.

The implementation should treat it as a distinct reduction strategy with:

• its own reduced object type;
• its own targeting submodel;
• separate unit tests for score identities;
• separate R or oracle fixtures;
• side-by-side simulation against the univariate reduction.

Do not force bivariate and univariate forms through one array schema if their estimating equations differ structurally.

Multivalued treatment

The roadmap is correct to resist generalizing from software acceptance alone. A multivalued implementation raises real questions:

• Are treatment-specific mean estimators jointly targeted or independently targeted by arm?
• Does the targeted treatment mechanism remain on the multinomial simplex?
• Are armwise logistic tilts variation-independent?
• What is the joint corrected influence function?
• How are covariance and ATE contrasts handled?
• Are positivity conditions arm-specific?
• Does the theorem cover a fixed number of arms only?

A valid implementation likely needs a multinomial fluctuation or another simplex-preserving parameterization, not independent binary mechanism tilts.

This should remain out of scope until theorem coverage is verified.

────────

7. Additional roadmap omissions

7.1 Observation weights require explicit theoretical coverage

The implementation currently states that observation weights “need nothing said about them.” That is too strong.

Known fixed sampling weights may be accommodated by changing the empirical measure and target population, but the DRTMLE theorem and reduced score equations must still be checked under that weighted law. Estimated weights, survey calibration weights, frequency weights, and cluster weights are not interchangeable.

Until verified, narrow the claim to fixed nonnegative analysis weights defining a tilted empirical target, and add weighted parity and coverage tests.

7.2 Repeated cross-fitting needs a DRTMLE-specific argument

Averaging influence curves across repeated split draws is reasonable for ordinary cross-fitted estimators, but DRTMLE includes split-dependent reduced regressions and iterative targeting. The package should test that:

• each repeat independently satisfies its extra scores;
• the averaged corrected influence curve remains centered;
• standard-error calibration is not degraded;
• failed repeats are not silently omitted in a way that changes the target.

7.3 Confidence intervals should be gated by score validity

Because DRTMLE exists specifically to improve inference, returning a conventional Wald interval after failed targeting is misleading. The roadmap should define result states:

• valid;
• numerically unresolved;
• positivity unsupported;
• theory-experimental.

The summary should make invalid inference impossible to overlook.

7.4 Remainder tests should become first-class release evidence

The repository mentions arithmetic tests of the remainder. Expand these into a documented chain:

1. exact first-order expansion on finite-support laws;
2. correction removes the non-negligible drift when either nuisance limit is wrong;
3. remaining term is a product involving reduced-regression error;
4. empirical rates of the remainder match the theory in simulations.

This is more diagnostic than coverage alone.

7.5 Separate computational parity from statistical validity

The roadmap sometimes treats the R package as the reference truth. It should instead distinguish:

• parity: Python implements the same algorithm;
• theoretical validity: the algorithm satisfies the theorem’s conditions;
• empirical usefulness: it improves inference in realistic finite samples.

All three are required, and none implies the others.

────────

8. Revised implementation roadmap

Phase 0 — Rename and status discipline

• Resolve whether the public project name is cleverly, causly-tmle, or another name.
• Keep DRTMLE explicitly experimental.
• Prevent documentation from describing the corrected curve as the efficient influence function under misspecification.
• Gate Wald intervals on successful score diagnostics.

Exit criterion: users cannot mistake the current DRTMLE result for a fully validated production interval.

Phase 1 — Theoretical audit

• Read and annotate Theorem 1 and its appendices/supplement.
• Write a notation concordance across paper, Python, and R.
• List every theorem assumption and whether the implementation satisfies it.
• Verify signs, arm indexing, starred quantities, correction terms, and guard semantics.
• Resolve the cross-fitting requirements.
• Check the scope of weights and multivalued treatment.

Exit criterion: every code formula has a theorem or clearly labeled implementation source, and unsupported assumptions are refused or documented.

Phase 2 — Component-level reference validation

• Create deterministic R fixtures.
• Compare nuisance arrays, reductions, targeting coefficients, scores, correction terms, full influence curves, estimates, and standard errors.
• Add deliberately misspecified cases.
• Add an independent finite-law or drift-decomposition test.

Exit criterion: Python agrees with R where intended and independently satisfies the theoretical identities.

Phase 3 — Targeting and overlap hardening

• Run the expanded weak-overlap diagnostic.
• Replace overclaimed convergence language.
• Define statistical score tolerances separately from numerical solver tolerances.
• Determine whether failures are due to positivity, reduced estimation, or algorithmic iteration.
• Add predeclared invalid-inference states and overlap limits.

Exit criterion: all returned valid intervals satisfy every required empirical score to the declared order; unsupported overlap regimes are unmistakably rejected or labeled invalid.

Phase 4 — Controlled inferential simulation

• Implement rate-controlled slow-nuisance DGPs in both off-diagonal directions.
• Measure nuisance rates and the remainder directly.
• Use at least three sample sizes and sufficient replications.
• Compare TMLE and DRTMLE bias, spread, estimated SE, coverage, and failure rate.
• Run a second seed batch or independent implementation check.

Exit criterion: DRTMLE achieves nominal inference in the intended union-model regime and ordinary TMLE demonstrably fails, or the package honestly concludes that no useful finite-sample regime was demonstrated.

Phase 5 — Applied stress tests

After the controlled proof-of-concept:

• test Super Learner libraries;
• high-dimensional nonlinear DGPs;
• moderate near-positivity stress;
• continuous and binary outcomes;
• fixed analysis weights;
• repeated cross-fitting.

These are generalization checks, not substitutes for Phase 4.

Phase 6 — Scope expansion

• Implement bivariate reduction first.
• Consider multivalued treatment only after theorem confirmation and simplex-preserving targeting design.
• Keep ATT, ATC, stochastic interventions, shifts, longitudinal estimands, missingness extensions, and C-TMLE composition refused unless separately derived.

────────

9. Proposed test matrix

|Layer         |Test                                                    |Required failure control              |
|--------------|--------------------------------------------------------|--------------------------------------|
|Unit          |Definitions of (Q_r,g_{r,1},g_{r,2})                    |swap `gr1`/`gr2`                      |
|Unit          |Signs in corrected influence curve                      |add instead of subtract               |
|Unit          |Targeted vs initial nuisances                           |read initial (g) or reductions        |
|Unit          |Arm-specific indexing                                   |swap arm columns                      |
|Unit          |Score equations 8–10                                    |omit each equation separately         |
|Unit          |Statistical stopping rule                               |remove absolute branch                |
|Oracle        |Exact drift decomposition                               |delete one correction term            |
|Cross-language|R component parity                                      |perturb one component                 |
|Integration   |`guard=()` equals TMLE bit-for-bit                      |route through reduction loop          |
|Integration   |each guard protects the named misspecification direction|cross guard semantics                 |
|Simulation    |slow (Q), wrong (g)                                     |ordinary TMLE undercoverage           |
|Simulation    |slow (g), wrong (Q)                                     |ordinary TMLE undercoverage           |
|Simulation    |both nuisances correct                                  |no material efficiency loss           |
|Simulation    |both nuisances wrong                                    |no false robustness claim             |
|Stress        |weak overlap                                            |inference invalidated when scores fail|
|Stress        |weighted law                                            |compare against known weighted truth  |
|Stress        |repeated splits                                         |averaged curve remains calibrated     |

────────

10. Bottom-line recommendation

The roadmap’s definition of done is correct but incomplete. A convincing DRTMLE release requires a chain of evidence:

1. Theorem fidelity — the implemented expansion and equations are the ones actually derived.
2. Reference fidelity — the Python algorithm agrees with drtmle at the component level.
3. Independent correctness — exact-law or drift tests do not merely copy the R implementation.
4. Numerical validity — required score equations are solved to a statistically negligible order.
5. Inferential usefulness — a controlled simulation demonstrates valid inference in a regime where ordinary TMLE inference fails.

Until phases 1–4 are complete, DRTMLE should remain experimental and should not return an apparently ordinary production confidence interval without a conspicuous validity status.

The implementation is close in software terms, but the remaining work is exactly the part that determines whether the met