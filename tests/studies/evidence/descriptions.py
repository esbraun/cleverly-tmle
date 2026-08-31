"""Plain-English names for everything a study's committed results refer to.

The result files identify a test by key: a scenario, an estimand, a property family, a cell.  Those
keys are precise and unreadable.  ``tests/studies/evidence/document.py`` renders one documentation
row per committed test, and this module supplies the column that says what the test actually
checked.

The descriptions are written by hand.  Everything beside them in a rendered row is read from the
artefacts, so a description is the one part of a published table a person chooses.
``tests/unit/test_method_evidence.py::TestThePublishedTestTables`` requires every key present in a
committed result file to resolve here.  A study cannot publish a row whose meaning is undeclared.

Deliberately not listed in any :class:`~tests.studies.evidence.registry.StudyRecord` ``modules``
tuple.  That tuple is hashed into each ``manifest.json``, and nothing here can change a measured
result -- only how it reads.
"""

from __future__ import annotations

import re

from tests.studies.evidence.property_verdicts import (
    UNION_MODEL_FAMILIES,
    UNION_MODEL_SE_BAND,
)

#: ``estimand`` values carry a regimen in brackets for the longitudinal studies.
_PARAMETERISED = re.compile(r"^(?P<name>[a-z_]+)\[(?P<argument>.+)\]$")

#: What a horizoned estimand key puts between the plan and the time, and what a competing-risk
#: key puts between the plan and the cause.  Mirrored from
#: :data:`cleverly.longitudinal.estimator.HORIZON_INFIX` and :data:`~.CAUSE_INFIX` rather than
#: imported, for the reason the datasets mirror them: a description of a committed result must
#: not depend on the estimator that produced it.
_HORIZON = " @ t="
_CAUSE = ", "

#: Which plan, cause and horizon a longitudinal property cell belongs to.  Cells prefix their
#: arm, so :func:`cell` strips it and reports it beside the family's shared description.
ARMS: dict[str, str] = {
    "ate": "average treatment effect",
    "static": "static plan",
    "dynamic": "dynamic plan",
    "third_arm": "third-arm static plan",
    "static_t1": "static plan at horizon one",
    "static_t2": "static plan at horizon two",
    "dynamic_t2": "dynamic plan at horizon two",
    "five_reduction_cycle": "five-reduction correction cycle",
    "always_t2": "always-treat risk at horizon two",
    "relapse_dynamic_t2": "dynamic relapse contrast at horizon two",
    "death_static_t2": "static death contrast at horizon two",
    "relapse_always_t2": "always-treat relapse incidence at horizon two",
    "death_always_t2": "always-treat death incidence at horizon two",
    "contrast": "odds-x2 contrast",
    "mechanism": "treatment-mechanism targeting",
    "natural": "natural course",
    "never": "never-treated plan",
    "odds_x2": "odds-x2 incremental mean",
    "outcome": "outcome-regression targeting",
    "rule": "covariate-dependent rule",
    "shift": "capped modified treatment policy",
    "tilt": "known stochastic tilt",
    "z0": "controlled direct effect at intermediate level zero",
    "z1": "controlled direct effect at intermediate level one",
}

#: Built from :data:`ARMS` rather than restated.  ``cell`` subscripts ``ARMS`` with whatever
#: this matches, so a hand-maintained second list is a ``KeyError`` waiting for the study that
#: extends one of them and not the other -- and that is the one lookup here which would not
#: raise :class:`Undescribed`.  Longest alternative first, so a longer arm is never shadowed
#: by a shorter one it starts with.
_ARM = re.compile(
    r"^(?P<arm>" + "|".join(sorted(ARMS, key=len, reverse=True)) + r")__(?P<cell>.+)$"
)

#: A contraction ladder's rung, ``"<configuration>_n<size>"``.  Matched only for the family
#: that builds ladders this way, so a cell whose name happens to end in ``_n1500`` elsewhere
#: is still looked up whole.
_SIZE = re.compile(r"^(?P<cell>.+)_n(?P<size>\d+)$")

TERMS: dict[str, str] = {
    "(intercept)": "intercept coefficient",
    "a": "treatment coefficient",
    "W": "baseline-covariate coefficient",
    "duration": "treatment-duration coefficient",
}
#: An MSM coefficient's cell name, ``"<term>__<configuration>"``.  Longest alternative first,
#: for the reason :data:`_ARM` is: a term that another term starts with would otherwise be
#: shadowed, and the shorter match would subscript :data:`TERMS` with the wrong key.
_TERM = re.compile(
    r"^(?P<term>"
    + "|".join(re.escape(term) for term in sorted(TERMS, key=len, reverse=True))
    + r")__(?P<cell>.+)$"
)


IMPLEMENTATIONS: dict[str, str] = {
    "cleverly": "`cleverly`",
    "cleverly-categorical-ltmle": "`cleverly` ordinary categorical LTMLE",
    "cleverly-cde-tmle": "`cleverly` controlled direct-effect TMLE",
    "cleverly-clustered-cvtmle": "`cleverly` clustered point-treatment CV-TMLE",
    "cleverly-cross-fitted-categorical-ltmle": "`cleverly` cross-fitted categorical LTMLE",
    "cleverly-cross-fitted-ltmle": "`cleverly` cross-fitted LTMLE",
    "cleverly-cross-fitted-ltmle-survival": "`cleverly` cross-fitted survival LTMLE",
    "cleverly-cross-fitted-competing-ltmle": "`cleverly` cross-fitted competing-risk LTMLE",
    "cleverly-competing-ltmle": "`cleverly` competing-risk LTMLE",
    "cleverly-ctmle-oat": "`cleverly` outcome-adaptive C-TMLE",
    "cleverly-ctmle-selector": "`cleverly` selector-based C-TMLE",
    "cleverly-fold-evaluated-cvtmle": "`cleverly` fold-evaluated CV-TMLE",
    "cleverly-fold-targeted-cvtmle": "`cleverly` fold-targeted CV-TMLE",
    "cleverly-repeated-cvtmle": "`cleverly` repeated stacked CV-TMLE",
    "cleverly-mar-drtmle": "`cleverly` randomized missing-outcome DR-TMLE",
    "cleverly-mar-tmle": "`cleverly` missing-outcome TMLE",
    "cleverly-multi-arm-ctmle-oat": "`cleverly` multi-arm outcome-adaptive C-TMLE",
    "cleverly-multi-arm-ctmle-selector": "`cleverly` multi-arm selector C-TMLE",
    "cleverly-multi-arm-drtmle": "`cleverly` multi-arm DR-TMLE",
    "cleverly-multi-arm-tmle": "`cleverly` ordinary multi-arm TMLE",
    "cleverly-stacked-cvtmle": "`cleverly` stacked CV-TMLE",
    "cleverly-weighted-tmle": "`cleverly` weighted point-treatment TMLE",
    "cleverly-weighted-ltmle": "`cleverly` ordinary weighted LTMLE",
    "cleverly-cross-fitted-weighted-ltmle": "`cleverly` cross-fitted weighted LTMLE",
    "cleverly-learned-weighted-tmle": (
        "`cleverly` weighted point-treatment TMLE with learned nuisances"
    ),
    "drtmle-r": "R `drtmle`",
    "drtmle-r-mar": "R `drtmle` with a joint treatment-response mechanism",
    "drtmle-r-multi-arm": "R `drtmle` multi-arm extension",
    "ltmle": "R `ltmle`",
    "ltmle-weighted": "R `ltmle` with observation weights",
    "ltmle projected regimen fits": "projected R `ltmle` regimen fits",
    "lmtp": "R `lmtp`",
    "lmtp-weighted": "R `lmtp` with observation weights",
    "npcausal": "R `npcausal`",
    "r-ctmle": "R `ctmle`",
    "tlverse-ctmle3-oat": "R `ctmle3`",
    "ctmle3-multi-arm-oat": "R `ctmle3` multi-arm outcome-adaptive TMLE",
    "tmle3": "R `tmle3`",
    "tmle3-cvtmle": "R `tmle3` CV-TMLE",
    "tmle3-multi-arm": "R `tmle3` multi-arm TMLE",
    "tmle-r": "R `tmle`",
    "tmle-r-cde": "R `tmle` controlled direct-effect path",
    "tmle-r-weighted": "R `tmle` with observation weights",
    "tmle-r-learned-weighted": "R `tmle` with learned weighted nuisances",
    "zepid-single-crossfit-tmle": "Python `zEpid` single-crossfit TMLE",
}


SCENARIOS: dict[str, str] = {
    "binary": "binary-outcome law",
    "binary_discrete": "binary-outcome law, discrete selector",
    "binary_greedy": "binary-outcome law, greedy selector",
    "binary_ordered": "binary-outcome law, ordered selector",
    "censored_end_of_study": "two-time-point law with monotone censoring",
    "selected_censored_end_of_study": (
        "selected two-time-point law with monotone censoring and fixed observation weights"
    ),
    "categorical_end_of_study": "two-time-point law with three treatment levels at both nodes",
    "censored_survival_curve": "two-time-point absorbing-event law with monotone censoring",
    "censored_competing_risk_curve": (
        "two-time-point, two-cause competing-risk law with monotone censoring"
    ),
    "censored_regimen_projection": (
        "two-time-point law with monotone censoring and four projected treatment plans"
    ),
    "continuous": "bounded continuous-outcome law with effect modification",
    "clustered_continuous": (
        "continuous-outcome law with ten rows per cluster and shared effect modification"
    ),
    "bounded_continuous_projection": (
        "bounded continuous-outcome law with an unsaturated working model"
    ),
    "both_correct": "paper binary law, both nuisances correct",
    "outcome_correct": "paper binary law, outcome regression correct",
    "treatment_correct": "paper binary law, treatment mechanism correct",
    "binary_biased_sample": "binary-outcome law sampled with unequal selection probabilities",
    "binary_cde_z0_mar": "binary-outcome MAR observed law, intervention sets the intermediate to zero",
    "binary_cde_z1_mar": "binary-outcome MAR observed law, intervention sets the intermediate to one",
    "binary_dynamic_rule": "binary-outcome law with a covariate-dependent deterministic rule",
    "binary_incremental_odds": "binary-outcome law with three incremental odds multipliers",
    "binary_known_stochastic": "binary-outcome law with a known stochastic treatment density",
    "binary_mar_observational": "binary-outcome observational law with MAR outcomes",
    "binary_mar_randomized": "binary-outcome randomized law with MAR outcomes",
    "continuous_modified_policy": "continuous-dose law with uncapped and capped shifts",
    "continuous_selected_weighted_nuisances": (
        "continuous-outcome law selected by a covariate-dependent density"
    ),
    "multi_arm_binary": "three-arm binary-outcome law",
    "multi_arm_binary_drtmle": "three-arm binary-outcome law with shared cross-fitted nuisances",
    "multi_arm_binary_oat": "three-arm binary-outcome law, outcome-adaptive selector",
    "multi_arm_selector_discrete": "three-arm binary-outcome law, discrete selector",
    "multi_arm_selector_greedy": "three-arm binary-outcome law, greedy selector",
    "multi_arm_selector_ordered": "three-arm binary-outcome law, ordered selector",
}


ESTIMANDS: dict[str, str] = {
    "atc": "average effect on the untreated",
    "ate": "average treatment effect",
    "att": "average effect on the treated",
    "ey0": "counterfactual mean under no treatment",
    "ey1": "counterfactual mean under treatment",
    "ey_obs": "observed outcome mean under the natural course",
    "or": "marginal odds ratio, reported on the log scale",
    "paf": "population attributable fraction",
    "par": "population attributable risk",
    "rr": "marginal risk ratio, reported on the log scale",
}

#: The bracketed half of a longitudinal estimand key.
REGIMENS: dict[str, str] = {
    "always": "treat at both times",
    "continue_if_l2": "treat first, then continue if L2 equals one",
    "high": "assign the high arm at both times",
    "low": "assign the low arm at both times",
    "never": "treat at neither time",
    "respond": "assign standard first, then high if L2 equals one and low otherwise",
    "standard": "assign the standard arm at both times",
    "step_down": "assign high first, then standard",
    "treat then continue if l2 positive": "treat, then continue only if L2 is positive",
    "+0.25": "shift dose by 0.25",
    "+0.5 capped": "shift dose by 0.5 subject to the declared cap",
    "natural course": "leave the observed treatment mechanism unchanged",
    "odds x0.5": "multiply the treatment odds by 0.5",
    "odds x2": "multiply the treatment odds by two",
    "rule": "follow the covariate-dependent rule",
    "tilt": "draw from the known stochastic tilt",
}

#: How a point-treatment study reads a plan label whose ordinary reading counts treatment
#: times.  ``never`` is the only label that differs, and the difference is not cosmetic: a
#: point-treatment study assigns one treatment, so "treat at neither time" describes a design
#: that study does not have.  The bracket alone cannot tell the two apart, because both
#: constructions write ``[never]``; the parameter name is what separates ``ey_regime`` from
#: ``ey_regimen`` and it is passed in rather than guessed.
POINT_REGIMENS: dict[str, str] = {
    "never": "assign no treatment",
}

#: What the parameterised estimand names mean once the regimen is resolved.
PARAMETERISED: dict[str, str] = {
    "cif_regimen": "cumulative incidence under the plan",
    "ey_regimen": "mean outcome under the plan",
    "ate_regimen": "difference in mean outcome between the plans",
    "risk_regimen": "cumulative risk under the plan",
    "ate_ipsi": "difference in means under the incremental interventions",
    "ate_regime": "difference in means under the regimes",
    "ate_shift": "difference in means under the modified treatment policies",
    "ey_ipsi": "mean under the incremental intervention",
    "ey_regime": "mean under the regime",
    "ey_shift": "mean under the modified treatment policy",
}


PROPERTIES: dict[str, str] = {
    "cap_necessity": "the declared cap changes which continuous doses the policy shifts",
    "categorical_probability_necessity": (
        "the assigned categorical arm selects its own mechanism probability"
    ),
    "cde_robustness": (
        "controlled direct-effect TMLE stays consistent when the outcome regression is correct "
        "or when all three mechanisms are correct"
    ),
    "clustered_inference": (
        "cluster-level influence-curve aggregation calibrates inference under within-cluster dependence"
    ),
    "crossfit_overfitting": (
        "cross-fitting removes the optimism a flexible learner puts into an in-sample fit"
    ),
    "corrected_mar_inference": (
        "randomized missing-outcome DR-TMLE retains valid inference when either the outcome "
        "regression or observation mechanism is correct"
    ),
    "correction_necessity": (
        "the five-reduction correction cycle materially reduces the empirical correction scores"
    ),
    "double_robust_contraction": (
        "a bias the equivalence margin rejects at one size contracts as the sample grows, "
        "which is what separates a second-order remainder from an inconsistent estimator"
    ),
    "double_robustness": (
        "the estimator stays consistent when either the outcome regression or the treatment "
        "mechanism is correct"
    ),
    "generated_design": (
        "the outcome-adaptive design costs precision when its regression is estimated rather "
        "than known"
    ),
    "density_necessity": "the declared stochastic intervention density determines the target",
    "interval_calibration": (
        "the reported standard error and the exact coverage both sit inside their declared "
        "two-sided calibration bands"
    ),
    "learner_weight_necessity": (
        "both nuisance learners use the fixed weights that define the target population"
    ),
    "mechanism_requirement": (
        "the treatment mechanism must be correct because it defines the incremental parameter"
    ),
    "mar_robustness": (
        "missing-outcome TMLE stays consistent when the outcome regression is correct or "
        "when both treatment and observation mechanisms are correct"
    ),
    "missingness_necessity": (
        "declaring the observation indicator prevents complete-case selection from changing "
        "the target"
    ),
    "natural_course_identity": "the natural-course intervention reduces exactly to the sample mean",
    "power": "the test detects a real effect, so a passing null result cannot come from an inert test",
    "projection_necessity": (
        "the declared projection measure determines the coefficient rather than an implicit "
        "uniform measure"
    ),
    "ratio_necessity": "the modified-policy density ratio is evaluated in the declared direction",
    "robustness_contract": (
        "the estimator stays consistent under the one nuisance the method's source paper claims"
    ),
    "root_n_and_efficiency": (
        "bias, coverage and standard-error calibration hold across sample sizes"
    ),
    "root_n_rate": "the sampling spread contracts at the root-n rate the theory predicts",
    "rule_necessity": "the covariate-dependent rule, rather than a static substitute, determines the target",
    "selector_necessity": "the collaborative selector is what produces the result, not the fit around it",
    "competing_risk_recursion_necessity": (
        "the cumulative-incidence recursion uses all-cause survival rather than survival from "
        "the target cause alone"
    ),
    "survival_recursion_necessity": (
        "the absorbing-event recursion is what produces cumulative risk, not an analysis "
        "restricted to survivors"
    ),
    "static_reduction": "a static regime reduces exactly to the corresponding treatment-arm target",
    "targeting_necessity": "the targeting step is what produces the result, not the plug-in beneath it",
    "treatment_score_necessity": (
        "incremental inference includes the influence-curve derivative through the treatment mechanism"
    ),
    "type_i_error": "under a confounded sharp null the test rejects no more often than its nominal size",
    "weight_necessity": (
        "fixed inverse-selection weights recover the population target from the selected law"
    ),
}


#: ``(family, cell)`` after any arm prefix is stripped, to ``(what was tested, what must hold)``.
CELLS: dict[tuple[str, str], tuple[str, str]] = {
    ("cde_robustness", "all_correct"): (
        "the outcome regression and all three mechanisms are correct",
        "bias interval inside the equivalence margin",
    ),
    ("cde_robustness", "outcome_correct"): (
        "the outcome regression is correct and all three mechanisms are wrong",
        "bias interval inside the equivalence margin",
    ),
    ("cde_robustness", "mechanisms_correct"): (
        "all three mechanisms are correct and the outcome regression is wrong",
        "bias interval inside the equivalence margin",
    ),
    ("cde_robustness", "treatment_wrong"): (
        "only the treatment mechanism is wrong beside a wrong outcome regression",
        "bias interval must fall entirely outside the margin",
    ),
    ("cde_robustness", "intermediate_wrong"): (
        "only the intermediate mechanism is wrong beside a wrong outcome regression",
        "bias interval must fall entirely outside the margin",
    ),
    ("cde_robustness", "observation_wrong"): (
        "only the observation mechanism is wrong beside a wrong outcome regression",
        "bias interval must fall entirely outside the margin",
    ),
    ("clustered_inference", "cluster_robust"): (
        "five-fold point-treatment TMLE with cluster-robust ATE inference",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control"): (
        "the identical rows, point estimates, and influence curves treated as independent",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("weight_necessity", "weighted"): (
        "the selected sample analyzed with its fixed inverse-selection weights",
        "population-target bias interval inside the equivalence margin",
    ),
    ("weight_necessity", "omitted_control"): (
        "the identical selected rows analyzed without their inverse-selection weights",
        "population-target bias outside its margin and selected-target bias inside its margin",
    ),
    ("weight_necessity", "omitted_weight_control"): (
        "the identical selected rows analyzed without any observation weights",
        "population-target bias outside its margin, selected-target bias inside its margin, "
        "and paired displacement above its threshold",
    ),
    ("learner_weight_necessity", "weighted_targeted"): (
        "the weighted nuisance fits followed by weighted targeting and averaging",
        "target-population bias interval inside the equivalence margin",
    ),
    ("learner_weight_necessity", "unweighted_targeted"): (
        "nuisance fits omit weights, while targeting and averaging retain them",
        "target-population bias interval inside the equivalence margin",
    ),
    ("learner_weight_necessity", "weighted_plugin"): (
        "the untargeted plug-in from both weighted nuisance fits",
        "target-population bias interval inside the equivalence margin",
    ),
    ("learner_weight_necessity", "unweighted_plugin_control"): (
        "the untargeted plug-in from nuisance fits that omit weights",
        "target bias outside its margin, selected-population bias inside its margin, and paired displacement above its threshold",
    ),
    ("learner_weight_necessity", "weighted_learners"): (
        "sampling weights enter nuisance learning, targeting, averaging, and covariance",
        "population-target bias interval inside the equivalence margin",
    ),
    ("learner_weight_necessity", "discarded_learner_weight_control"): (
        "nuisance learners discard sampling weights while later estimator stages retain them",
        "population-target bias outside its margin, learner-selected-target bias inside its "
        "margin, and paired displacement above its threshold",
    ),
    ("corrected_mar_inference", "both_correct"): (
        "the outcome regression and observation mechanism are correctly specified",
        "bias interval inside the margin, coverage clears the floor, SE ratio inside the band",
    ),
    ("corrected_mar_inference", "outcome_drift"): (
        "the outcome regression is misspecified and the observation mechanism is correct",
        "bias interval inside the margin, coverage clears the floor, SE ratio inside the band",
    ),
    ("corrected_mar_inference", "observation_drift"): (
        "the outcome regression is correct and the observation mechanism is misspecified",
        "bias interval inside the margin, coverage clears the floor, SE ratio inside the band",
    ),
    ("corrected_mar_inference", "both_wrong"): (
        "the outcome regression and observation mechanism are both misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("correction_necessity", "closed_score"): (
        "the correction scores after the complete five-reduction cycle",
        "the upper confidence endpoint is below the declared fraction of the initial-score lower endpoint",
    ),
    ("correction_necessity", "initial_score_control"): (
        "the same correction scores before the cycle is run",
        "the lower confidence endpoint clears the declared unresolved-score floor",
    ),
    ("crossfit_overfitting", "cross_fitted_oat"): (
        "outcome-adaptive C-TMLE with cross-fitted nuisances and a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "fold_evaluated_cvtmle"): (
        "fold-evaluated CV-TMLE with a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "fold_targeted_cvtmle"): (
        "fold-targeted CV-TMLE with a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "stacked_cvtmle"): (
        "stacked CV-TMLE with a flexible learner",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "in_sample_control"): (
        "the same flexible learner fitted in sample, with no cross-fitting",
        "SE ratio must fall below the overfitting ceiling",
    ),
    ("crossfit_overfitting", "cross_fitted_ltmle"): (
        "five-fold end-of-study LTMLE with a fully grown outcome tree",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "cross_fitted_categorical_ltmle"): (
        "five-fold categorical LTMLE with a fully grown outcome tree",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "cross_fitted_survival_ltmle"): (
        "five-fold horizon-two survival LTMLE with a fully grown outcome tree",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "cross_fitted_competing_ltmle"): (
        "five-fold horizon-two competing-risk LTMLE with a fully grown outcome tree",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("double_robust_contraction", "outcome_correct"): (
        "only the outcome regression is correctly specified",
        "the exact coverage interval clears the declared floor",
    ),
    ("double_robust_contraction", "treatment_correct"): (
        "only the treatment mechanism is correctly specified",
        "the exact coverage interval clears the declared floor",
    ),
    ("double_robust_contraction", "both_wrong"): (
        "both nuisances are misspecified",
        "the exact coverage interval must fall below the floor",
    ),
    ("double_robust_contraction", "rate_outcome_correct"): (
        "log absolute bias regressed on log n across three sizes, outcome regression correct",
        "slope interval entirely below zero, so the bias contracts",
    ),
    ("double_robust_contraction", "rate_treatment_correct"): (
        "the same regression with only the treatment mechanism correct",
        "slope interval entirely below zero, so the bias contracts",
    ),
    ("double_robust_contraction", "rate_both_wrong"): (
        "the same regression with both nuisances misspecified",
        "slope interval must not establish contraction",
    ),
    ("double_robustness", "both_correct"): (
        "both the outcome regression and the treatment mechanism are correctly specified",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "outcome_correct"): (
        "only the outcome regression is correctly specified",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "treatment_correct"): (
        "only the treatment mechanism is correctly specified",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "density_correct"): (
        "only the continuous-dose density ratio is correctly specified",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "mechanism_correct"): (
        "only the treatment and censoring mechanisms are correctly specified",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "both_wrong"): (
        "both nuisances are misspecified",
        "bias interval must fall entirely outside the margin, with the reported standard "
        "error still on the scale of the empirical spread",
    ),
    ("generated_design", "oracle_design"): (
        "the outcome-adaptive design is supplied rather than estimated",
        "SE ratio interval inside the calibration band",
    ),
    ("generated_design", "estimated"): (
        "the same design is estimated from the data, as a real fit does",
        "the SE-ratio deficit must reach the declared shortfall",
    ),
    ("mechanism_requirement", "both_correct"): (
        "both the outcome regression and treatment mechanism are correctly specified",
        "bias interval inside the equivalence margin",
    ),
    ("mechanism_requirement", "outcome_wrong"): (
        "the treatment mechanism is correct and the outcome regression is misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("mechanism_requirement", "mechanism_wrong"): (
        "the wrong treatment mechanism is held fixed when the incremental target is evaluated",
        "bias interval must fall entirely outside the margin",
    ),
    ("mar_robustness", "both_correct"): (
        "the outcome regression, treatment mechanism and observation mechanism are correct",
        "bias interval inside the equivalence margin",
    ),
    ("mar_robustness", "outcome_correct"): (
        "only the outcome regression is correct",
        "bias interval inside the equivalence margin",
    ),
    ("mar_robustness", "mechanisms_correct"): (
        "the treatment and observation mechanisms are correct and the outcome regression is not",
        "bias interval inside the equivalence margin",
    ),
    ("mar_robustness", "treatment_wrong"): (
        "only the observation mechanism is correct",
        "bias interval must fall entirely outside the margin",
    ),
    ("mar_robustness", "observation_wrong"): (
        "only the treatment mechanism is correct",
        "bias interval must fall entirely outside the margin",
    ),
    ("missingness_necessity", "declared"): (
        "the observation indicator is declared, so correct mechanisms carry a wrong outcome model",
        "bias interval inside the equivalence margin",
    ),
    ("missingness_necessity", "complete_case_control"): (
        "the identical estimator silently discards unobserved outcomes and ignores selection",
        "bias interval must fall entirely outside the margin",
    ),
    ("interval_calibration", "correctly_specified"): (
        "both nuisances are correctly specified",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "treatment_correct"): (
        "the randomized treatment mechanism is correct while the outcome regression omits effect modification",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "shrunken_se_control"): (
        "the reported standard errors are multiplied by a declared factor below one",
        "the SE-ratio interval must fall below the calibration band",
    ),
    ("interval_calibration", "noise_control"): (
        "one efficiency-bound unit of independent noise is added to each estimate",
        "the empirical efficiency ratio must rise above the band",
    ),
    ("power", "alternative"): (
        "the same test applied to a law with a real effect",
        "rejection lower bound clears the minimum power",
    ),
    ("projection_necessity", "declared_weights"): (
        "the working model uses its declared nonuniform projection weights",
        "bias interval inside the equivalence margin",
    ),
    ("projection_necessity", "uniform_weights"): (
        "the identical working model is projected under uniform weights",
        "bias interval must fall entirely outside the margin",
    ),
    ("rule_necessity", "declared"): (
        "the declared covariate-dependent rule assigns both treatment arms",
        "bias interval inside the equivalence margin",
    ),
    ("rule_necessity", "static_control"): (
        "the same fit replaces the rule with an always-treated static plan",
        "bias interval must fall entirely outside the margin",
    ),
    ("rule_necessity", "declared_rule"): (
        "the declared categorical rule selects its second-node arm from the history",
        "bias interval inside the equivalence margin",
    ),
    ("rule_necessity", "reversed_rule"): (
        "the same fit reverses the rule's two history-specific arm assignments",
        "bias interval must fall entirely outside the margin",
    ),
    ("categorical_probability_necessity", "assigned_probability"): (
        "the clever covariate selects the assigned third arm's own probability",
        "bias interval inside the equivalence margin",
    ),
    ("categorical_probability_necessity", "binary_complement"): (
        "the same fit replaces the third arm's probability with a binary complement",
        "bias interval must fall entirely outside the margin",
    ),
    ("density_necessity", "declared"): (
        "the estimator integrates over the declared covariate-dependent treatment density",
        "bias interval inside the equivalence margin",
    ),
    ("density_necessity", "uniform_control"): (
        "the same fit replaces the declared density with a uniform distribution",
        "bias interval must fall entirely outside the margin",
    ),
    ("ratio_necessity", "declared"): (
        "the shifted-to-natural density ratio is used in the declared direction",
        "bias interval inside the equivalence margin",
    ),
    ("ratio_necessity", "reversed_control"): (
        "the density probabilities are deliberately inverted before the pooled-hazard ratio is formed",
        "bias interval must fall entirely outside the margin",
    ),
    ("cap_necessity", "declared_cap"): (
        "the 0.5 shift leaves doses unchanged when the declared cap would be crossed",
        "bias interval inside the equivalence margin",
    ),
    ("cap_necessity", "uncapped_control"): (
        "the same named policy removes the cap and shifts every dose",
        "bias interval must fall entirely outside the margin",
    ),
    ("static_reduction", "regime"): (
        "the never-treated target is requested through the regime axis",
        "the paired estimate must equal the treatment-arm estimate exactly",
    ),
    ("static_reduction", "arm"): (
        "the same target is requested as the ordinary untreated-arm mean",
        "the paired estimate must equal the regime estimate exactly",
    ),
    ("natural_course_identity", "shift"): (
        "the zero-shift policy is evaluated through the continuous-policy axis",
        "the paired estimate must equal the observed sample mean exactly",
    ),
    ("natural_course_identity", "ipsi"): (
        "the odds multiplier one is evaluated through the incremental axis",
        "the paired estimate must equal the observed sample mean exactly",
    ),
    ("natural_course_identity", "mean"): (
        "the observed sample mean is retained as the identity control",
        "the paired intervention estimate must equal it exactly",
    ),
    ("treatment_score_necessity", "full_eif"): (
        "the incremental mean uses the complete efficient influence curve",
        "the reported-to-empirical SE-ratio interval must stay inside the declared band",
    ),
    ("treatment_score_necessity", "regime_curve_control"): (
        "the same point estimates use an influence curve with the treatment-score term removed",
        "the SE-ratio interval must fall below the calibration band",
    ),
    ("robustness_contract", "outcome_correct"): (
        "the outcome regression is correct and the mechanism is not",
        "bias interval inside the equivalence margin",
    ),
    ("robustness_contract", "outcome_wrong"): (
        "the outcome regression is misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("root_n_and_efficiency", "n_500"): (
        "bias, coverage and SE calibration at n = 500",
        "",  # the smallest rung's rule depends on its role; :func:`cell` sets it or raises
    ),
    ("root_n_and_efficiency", "n_1000"): (
        "bias, coverage and SE calibration at n = 1,000",
        "",  # the smallest rung's rule depends on its role; :func:`cell` sets it
    ),
    ("root_n_and_efficiency", "n_1500"): (
        "bias, coverage and SE calibration at n = 1,500",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_2000"): (
        "bias, coverage and SE calibration at n = 2,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_4000"): (
        "bias, coverage and SE calibration at n = 4,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_8000"): (
        "bias, coverage and SE calibration at n = 8,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_32000"): (
        "bias, coverage and SE calibration at n = 32,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_4500"): (
        "bias, coverage and SE calibration at n = 4,500",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_rate", "empirical_sd"): (
        "log empirical spread of the estimates regressed on log n across three sizes",
        "slope interval inside the root-n band and excluding -1/4",
    ),
    ("root_n_rate", "reported_se"): (
        "the same regression applied to the mean reported standard error",
        "slope interval inside the root-n band and excluding -1/4",
    ),
    ("selector_necessity", "collaborative"): (
        "the selector chooses its own mechanism path",
        "bias interval inside the equivalence margin",
    ),
    ("selector_necessity", "greedy"): (
        "the greedy selector chooses its own mechanism path",
        "bias interval inside the equivalence margin, and RMSE below the control's by the "
        "declared ratio",
    ),
    ("selector_necessity", "ordered"): (
        "the ordered selector chooses how far along a fixed covariate order to go",
        "bias interval inside the equivalence margin, and RMSE below the control's by the "
        "declared ratio",
    ),
    ("selector_necessity", "discrete"): (
        "the discrete selector chooses among a declared candidate ladder",
        "bias interval inside the equivalence margin, and RMSE below the control's by the "
        "declared ratio",
    ),
    ("selector_necessity", "empty_control"): (
        "the selector is forced to stop at an empty path",
        "bias interval must fall entirely outside the margin",
    ),
    # "misspecified" rather than "constant", and "fit" rather than "backward recursion":
    # six studies register this family and the description is shared by all of them.  The
    # longitudinal ones fluctuate a sample-prior constant over a two-node recursion; the
    # point-treatment MSM projection fluctuates a deliberately inverted oracle over one node.
    # Naming either study's construction here published a false sentence on the other's page.
    ("targeting_necessity", "targeted"): (
        "the estimator fluctuates a misspecified outcome model, so targeting does all the "
        "adjusting",
        "bias interval inside the equivalence margin",
    ),
    ("targeting_necessity", "untargeted"): (
        "the identical fit with every fluctuation step removed",
        "bias interval must fall entirely outside the margin",
    ),
    ("survival_recursion_necessity", "survival"): (
        "the survival estimator keeps failures in their event node and removes them afterward",
        "bias interval inside the equivalence margin",
    ),
    ("survival_recursion_necessity", "survivor_only"): (
        "the same horizon-two outcome analyzed only among first-node survivors",
        "bias interval must fall entirely outside the margin",
    ),
    ("competing_risk_recursion_necessity", "all_cause"): (
        "the estimator removes every first-node event from the later risk set",
        "bias interval inside the equivalence margin",
    ),
    ("competing_risk_recursion_necessity", "cause_specific_control"): (
        "the same recursion wrongly lets the competing cause remain at risk",
        "bias interval must fall entirely outside the margin",
    ),
    ("type_i_error", "sharp_null"): (
        "a confounded law whose true contrast is exactly zero",
        "one-sided rejection bound stays under the declared type-I ceiling",
    ),
    ("type_i_error", "target_null"): (
        "a selected law whose weighted target contrast is exactly zero",
        "one-sided rejection bound stays under the declared type-I ceiling",
    ),
}


class Undescribed(LookupError):
    """A committed result names a key this module does not describe."""


def implementation(key: str) -> str:
    """The reader-facing name of an implementation column value."""
    try:
        return IMPLEMENTATIONS[key]
    except KeyError:
        raise Undescribed(f"no description for implementation {key!r}") from None


def scenario(key: str) -> str:
    """The law a scenario samples."""
    try:
        return SCENARIOS[key]
    except KeyError:
        raise Undescribed(f"no description for scenario {key!r}") from None


def estimand(key: str) -> str:
    """What an estimand key names, including a bracketed longitudinal regimen."""
    match = _PARAMETERISED.match(key)
    if match is None:
        try:
            return ESTIMANDS[key]
        except KeyError:
            raise Undescribed(f"no description for estimand {key!r}") from None
    name, argument = match.group("name"), match.group("argument")
    if name == "ey":
        return f"counterfactual mean under treatment arm {argument!r}"
    if name in {"ate", "rr", "or"}:
        contrast = argument.replace(" vs ", " versus ")
        if name == "ate":
            return f"difference in counterfactual means, {contrast}"
        ratio = "risk ratio" if name == "rr" else "odds ratio"
        return f"marginal {ratio}, {contrast}, reported on the log scale"
    if name in {"msm", "msm_regimen"}:
        try:
            term = TERMS[argument]
        except KeyError:
            raise Undescribed(f"no description for MSM term {argument!r}") from None
        construction = "point-treatment" if name == "msm" else "longitudinal regimen"
        return f"{construction} MSM projection {term}"
    if name not in PARAMETERISED:
        raise Undescribed(f"no description for parameterised estimand {name!r}")
    parameter = PARAMETERISED[name]
    # ``ey_regime`` is the point-treatment parameter and ``ey_regimen`` the longitudinal one.
    point = name in {"ey_regime", "ate_regime"}
    if _HORIZON not in argument:
        return f"{parameter} {regimen(argument, point=point)}"
    # Two constructions index a regimen mean by horizon, and the bracket is what tells them
    # apart.  A competing-risk key carries a cause beside the plan --
    # ``ate_regimen[always vs never, relapse @ t=2]`` -- and a single-event key does not.
    # Cumulative *incidence* and cumulative *risk* are different parameters, so each gets its
    # own wording rather than inheriting the other's.
    #
    # This reads the notation.  What actually settles it is the study's declared
    # ``outcome_kind``, and that is out of reach here: this function is handed a key, not a
    # record.  The notation is sufficient while these two constructions are the only ones that
    # index by horizon.  A third would need its own infix, or this function would need the
    # record -- it must not be added by widening one of the two branches below.
    if _CAUSE in argument:
        plan_and_cause, horizon = argument.rsplit(_HORIZON, 1)
        plan, cause = plan_and_cause.rsplit(_CAUSE, 1)
        if name == "cif_regimen":
            incidence = f"cumulative incidence of {cause} under the plan"
        elif name == "ate_regimen":
            incidence = f"difference in cumulative incidence of {cause} between the plans"
        else:
            raise Undescribed(
                f"{key!r} carries a cause as well as a horizon, so it is a cumulative "
                f"incidence; {name!r} has no cause-specific description"
            )
        return f"{incidence} {regimen(plan)} at horizon t = {horizon}"
    if name == "ate_regimen":
        parameter = "difference in cumulative risk between the plans"
    return f"{parameter} {regimen(argument, point=point)}"


def regimen(argument: str, *, point: bool = False) -> str:
    """A regimen label, or a contrast written as one label against another.

    Parameters
    ----------
    argument : str
        The bracketed half of an estimand key, which may be one label or ``"a vs b"``.
    point : bool, optional
        Whether the parameter is the point-treatment one. A label listed in
        :data:`POINT_REGIMENS` reads differently there.

    Returns
    -------
    str
        The label in words.
    """
    if _HORIZON in argument:
        label, horizon = argument.rsplit(_HORIZON, 1)
        return f"{regimen(label, point=point)} at horizon t = {horizon}"
    if " vs " in argument:
        treated, reference = argument.split(" vs ", 1)
        treated_words = regimen(treated, point=point)
        reference_words = regimen(reference, point=point)
        return f'"{treated_words}" against "{reference_words}"'
    if point and argument in POINT_REGIMENS:
        return POINT_REGIMENS[argument]
    try:
        return REGIMENS[argument]
    except KeyError:
        raise Undescribed(f"no description for regimen {argument!r}") from None


def claim(family: str) -> str:
    """The property a family tests."""
    try:
        return PROPERTIES[family]
    except KeyError:
        raise Undescribed(f"no description for property family {family!r}") from None


def cell(
    family: str,
    key: str,
    *,
    exact_efficiency: bool = False,
    role: str | None = None,
    nuisance_count: int = 2,
) -> tuple[str, str]:
    """What a property cell configures, and what its verdict requires.

    The arm prefix a longitudinal study puts on a cell name selects the plan rather than the
    configuration, so it is stripped and reported beside the shared description.
    """
    arm = _ARM.match(key)
    base = arm.group("cell") if arm else key
    term = _TERM.match(base)
    if term is not None:
        base = term.group("cell")
    # A contraction ladder names one configuration at three sizes, so the size is a suffix on
    # a shared description rather than three near-identical entries.  Stripped the same way
    # the arm prefix is, and reattached below, so a rung says which size it is.
    size = _SIZE.match(base) if family == "double_robust_contraction" else None
    if size is not None:
        base = size.group("cell")
    try:
        tested, required = CELLS[family, base]
    except KeyError:
        raise Undescribed(f"no description for cell {key!r} of {family!r}") from None
    if size is not None:
        tested = f"{tested}, at n = {int(size.group('size')):,}"
    if family == "interval_calibration" and base == "correctly_specified":
        if nuisance_count == 2:
            tested = "both nuisances are correctly specified"
        else:
            words = {3: "three", 4: "four"}
            count = words.get(nuisance_count, str(nuisance_count))
            tested = f"all {count} required nuisance functions are correctly specified"
    if family == "interval_calibration" and base == "correctly_specified" and exact_efficiency:
        tested += " with an independently computed efficiency bound"
        required += ", with both efficiency-ratio intervals inside their bands"
    if family == "interval_calibration" and base == "noise_control" and not exact_efficiency:
        tested = "a declared scale of independent noise is added to each estimate"
        required = "the SE-ratio interval must fall below the calibration band"
    if family in UNION_MODEL_FAMILIES and family != "double_robustness":
        low, high = UNION_MODEL_SE_BAND
        required += f", SE ratio must remain between {low} and {high}"
    if family == "targeting_necessity" and arm is not None:
        if arm.group("arm") == "mechanism":
            tested = (
                "the treatment mechanism is fluctuated after outcome targeting"
                if base == "targeted"
                else "the identical targeted outcome regression is evaluated at the unfluctuated mechanism"
            )
        elif arm.group("arm") == "outcome":
            tested = (
                "the outcome regression is fluctuated before the incremental target is evaluated"
                if base == "targeted"
                else "the identical targeted mechanism is evaluated with the initial outcome regression"
            )
    if family == "root_n_and_efficiency" and base in {"n_500", "n_1000"}:
        # The smallest rung is a positive cell in some studies and a retained small-sample
        # control in others, and the two are held to opposite rules.  Publishing one rule
        # beside the other role's verdict is the failure this branch exists to prevent, so an
        # unrecognised role raises rather than printing something true of neither.
        if role == "positive":
            required = "bias inside the margin, coverage clears the floor, SE ratio inside the band"
        elif role == "control":
            required = "coverage interval lies below nominal or clears the declared floor"
        else:
            raise Undescribed(
                f"cell {key!r} of {family!r} needs role='positive' or role='control'; "
                f"{role!r} selects neither, and the two rules are not interchangeable"
            )
    if arm is not None:
        tested = f"{ARMS[arm.group('arm')]}: {tested}"
    if term is not None:
        tested = f"{TERMS[term.group('term')]}: {tested}"
    return tested, required
