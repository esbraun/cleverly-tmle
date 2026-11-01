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

#: A stratum-conditional parameter, ``stratum_alias``'s ``<alias>[V=<label>]``.
_STRATUM = re.compile(r"^(?P<base>.+)\[(?P<stratum>[A-Za-z_]\w*=[^\[\]]+)\]$")

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
    "binary_active": "two-arm law with an arm-specific outcome index and an instrument",
    "three_arm_active": "three-arm law with an arm-specific outcome index and an instrument",
    "binary_weighted": "the two-arm law under the declared known weight 0.5 + 1{W1 > 0}",
    "binary_repeats": "the two-arm law, the median estimate over three cross-fitting splits",
    "ate_s0": "average treatment effect in baseline stratum S = 0",
    "ate_s1": "average treatment effect in baseline stratum S = 1",
    "ate_s2": "average treatment effect in baseline stratum S = 2",
    "ey_obs_s0": "observed outcome mean in baseline stratum S = 0",
    "ey_obs_s1": "observed outcome mean in baseline stratum S = 1",
    "ey_obs_s2": "observed outcome mean in baseline stratum S = 2",
    "par_s0": "population attributable risk in baseline stratum S = 0",
    "par_s1": "population attributable risk in baseline stratum S = 1",
    "par_s2": "population attributable risk in baseline stratum S = 2",
    "paf_s0": "population attributable fraction in baseline stratum S = 0",
    "paf_s1": "population attributable fraction in baseline stratum S = 1",
    "paf_s2": "population attributable fraction in baseline stratum S = 2",
    "rr": "marginal risk ratio",
    "or": "marginal odds ratio",
    "att": "average effect on the treated",
    "atc": "average effect on the untreated",
    # ``categorical-incremental`` (X43): the labels of its calibration and property cells.
    "point_contrast_t_a": "three-level law P, T_a against the natural course",
    "point_contrast_t_b": "three-level law P, T_b against the natural course",
    "point_known_t_a": "law P with the treatment mechanism declared, T_a against the natural course",
    "point_stratified_t_a_v1": "law P in baseline stratum V = 1, T_a against the natural course",
    "longitudinal_dropout_contrast_t_a": (
        "two-node law L with dropout and three-level treatment, T_a against the natural course"
    ),
    "longitudinal_contrast_t_a": (
        "two-node law L0 without dropout and three-level treatment, T_a against the natural course"
    ),
    "longitudinal_mean_t_a": "two-node law L with dropout, mean under T_a",
    "ey_ipsi": "incremental mean at odds multiplier 2",
    "ey0": "arm-0 mean",
    "ey1": "arm-1 mean",
    "ey2": "arm-2 mean",
    "ate_1_vs_0": "difference of arm 1 and arm 0",
    "ate_2_vs_0": "difference of arm 2 and arm 0",
    "ate_regimen": "contrast of always against never",
    # The repeated-outcome study's parameters, one label per plan and horizon.
    **{
        f"{plan}_t{horizon}": f"{words} at horizon t = {horizon}, repeated binary outcome"
        for plan, words in (
            ("never", "mean under never"),
            ("always", "mean under always"),
            ("rule", "mean under the rule that treats after an outcome"),
            ("always_vs_never", "contrast of always against never"),
            ("rule_vs_never", "contrast of the rule against never"),
        )
        for horizon in (1, 2, 3)
    },
    # The repeated-outcome IPSI study's parameters, one label per tilt and horizon.
    **{
        f"ipsi_{delta}_t{horizon}": f"{words} at horizon t = {horizon}, repeated binary outcome"
        for delta, words in (
            (0.5, "mean under the odds tilt 0.5"),
            (1.0, "mean under the natural course"),
            (2.0, "mean under the odds tilt 2"),
        )
        for horizon in (1, 2, 3)
    },
    **{
        f"ipsi_{delta}_vs_1.0_t{horizon}": (
            f"contrast of the odds tilt {delta} against the natural course at horizon "
            f"t = {horizon}, repeated binary outcome"
        )
        for delta in (0.5, 2.0)
        for horizon in (1, 2, 3)
    },
    "survival_t5": "difference of arm 1 and arm 0 at visit 5, five-visit held law",
    "survival_t5_per_horizon": (
        "difference of arm 1 and arm 0 at visit 5, five-visit held law, targeted one horizon "
        "at a time"
    ),
    "rmst_5": "RMST difference up to visit 5, five-visit held law",
    "competing_t4": "difference in the death incidence at visit 4, two causes",
    "three_arm_t3": "difference of arm 2 and arm 0 at visit 3, three arms with L_t",
    "continuous_t4": "difference at grid time 3, exponential event time, visit dropout",
    "end_of_study": "held end-of-study difference after two censoring nodes",
    "weighted_t5": "difference at visit 5 under the weight 1 + W1 / 2",
    "policy_t3": (
        "difference of a known baseline policy and arm 0 at visit 3, three arms without L_t"
    ),
    "isotonic_t2": (
        "risk at visit 2 under treatment at both nodes, two-node law, fitted with the isotonic "
        "projection"
    ),
    "clustered_t5": "arm-0 risk at visit 5, 25 clusters of 80 that share W",
    "clustered_t5_per_horizon": (
        "arm-0 risk at visit 5, 25 clusters of 80 that share W, targeted one horizon at a time"
    ),
    "weighted_t5_per_horizon": (
        "difference at visit 5 under the weight 1 + W1 / 2, targeted one horizon at a time"
    ),
    "policy_t5": "baseline policy against the natural course at visit 5",
    "mtp_t5": "baseline minus one against the natural course at visit 5",
    "ate": "average treatment effect",
    "static": "static plan",
    "dynamic": "dynamic plan",
    "ey_obs": "observed outcome mean under the natural course",
    "par": "population attributable risk",
    "paf": "population attributable fraction",
    "attributable_binary": (
        "L1: ey_obs, ey0, par and paf with MAR outcomes, in sample with the law's nuisances"
    ),
    "attributable_three_arm": (
        "L3: ey_obs, the three arm means, par[low] and paf[low] with MAR outcomes, in sample"
    ),
    "attributable_binary_cvtmle": (
        "L1: ey_obs, ey0, par and paf with MAR outcomes, stacked CV-TMLE with depth-five trees"
    ),
    "attributable_weighted": (
        "L1 with fixed weights 0.6, 1.0 and 1.8 by W: ey_obs, ey0, par and paf, in sample"
    ),
    "attributable_clustered": (
        "L1 rows in 100 clusters of 20 that share W: ey_obs, ey0, par and paf, in sample with id="
    ),
    "third_arm": "third-arm static plan",
    "static_t1": "static plan at horizon one",
    "static_t2": "static plan at horizon two",
    "dynamic_t2": "dynamic plan at horizon two",
    "five_reduction_cycle": "five-reduction correction cycle",
    # The one law of the learned-rule property cells.
    "non_exceptional": "the learned-rule law with a blip that is zero with probability zero",
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
    # The six ends of the omitted-variable bounds, each a calibration label of its own.
    "att_lower": "lower omitted-variable bound on the ATT",
    "att_upper": "upper omitted-variable bound on the ATT",
    "atc_lower": "lower omitted-variable bound on the ATC",
    "atc_upper": "upper omitted-variable bound on the ATC",
    "ate_lower": "lower omitted-variable bound on the ATE",
    "ate_upper": "upper omitted-variable bound on the ATE",
    # The seven treatment laws of the calibration-slope study, each with its treatment learner.
    "calibrated_weak": "the known propensity at a weak signal",
    "calibrated_strong": "the known propensity at a strong signal",
    "correct_weak": "a correct logistic model on one weak covariate",
    "correct_nested_weak": "a correct logistic model on four covariates, with a weak signal in one",
    "randomized": "a logistic model on four covariates, for a randomized treatment",
    "overconfident_moderate": "a logistic model with its logit doubled, at a moderate signal",
    "underconfident_strong": "a logistic model with its logit halved, at a strong signal",
    # The four laws of the stacked arm-indexed missing-outcome study, and each estimand's
    # calibration label.  L1 and L2 have two arms; L3 and L4 have three, with ``high`` as
    # the reference arm.  L1 and L3 have a binary outcome; L2 and L4 a bounded continuous one.
    # The joint-coverage labels of the default simultaneous bands.  Each band covers every
    # parameter its fit reports; ``all_reported`` and ``curve_always`` are shared by the
    # survival and the competing-risk studies, so their text names no outcome.
    "arms": "the multi-arm fit's three arm means, two differences, two risk ratios and two odds ratios",
    "regimens": "the three regimen means and their two contrasts",
    "all_reported": "every parameter the calibration fit reports",
    "curve_always": "the always plan alone, at every declared horizon",
    "curve_arm1": "the arm-1 risk curve alone, at the five visits",
    "curve_arm0": "the arm-0 risk curve alone, at the five visits",
    "strata": "the six marginal and eighteen stratum parameters of the in-sample fit",
    "crossfit_strata": "the three marginal and nine stratum parameters of the cross-fitted fit",
    "default_fit": "the shipped TMLE() defaults on the binary law",
    "ordinary_binary": "ordinary in-sample TMLE over ten binary-outcome estimands",
    "fold_evaluated": "fold-evaluated CV-TMLE over five estimands",
    "weighted": "weighted point-treatment TMLE over five estimands",
    "learned_weighted": "weighted point-treatment TMLE with learned nuisances",
    "missing_outcome": "missing-outcome TMLE over the arm means and the ATE",
    "missing_outcome_drtmle": "missing-outcome DR-TMLE over the arm means and the ATE",
    "drtmle": "cross-fitted DR-TMLE over the arm means and the ATE",
    "multi_arm_drtmle": "multi-arm DR-TMLE over nine parameters",
    "cde_z0": "the controlled direct-effect fit at intermediate level zero",
    "cde_z1": "the controlled direct-effect fit at intermediate level one",
    "point_msm": "the three terms of the point-treatment MSM projection",
    "clustered": "grouped cross-fitted TMLE with cluster multipliers",
    "clustered_regimens": (
        "the three regimen means and two contrasts of the cross-fitted clustered LTMLE, with "
        "cluster multipliers"
    ),
    "shift_grid": "three shift policies and their two contrasts",
    "incremental_grid": "three incremental odds multipliers and their two contrasts",
    "stochastic_regimes": "a static and a known stochastic regime and their contrast",
    "deterministic_regimes": "a static regime and a dynamic rule and their contrast",
    "ltmle_crossfit": "cross-fitted end-of-study regimen means and contrasts",
    "survival_crossfit": "the cross-fitted survival curve of three plans at two horizons",
    "competing_crossfit": "cross-fitted cumulative incidence of two causes, three plans, two horizons",
    "weighted_ltmle": "weighted end-of-study regimen means and contrasts",
    "weighted_ltmle_crossfit": "cross-fitted weighted end-of-study regimen means and contrasts",
    "categorical_ltmle": "five categorical regimen means and four contrasts",
    "categorical_ltmle_crossfit": "five cross-fitted categorical regimen means and four contrasts",
    "longitudinal_msm": "the two terms of the longitudinal MSM projection",
    "cross_fitted_msm": "the two terms of the cross-fitted longitudinal MSM projection",
    "duration_logit": "the treatment-duration coefficient of a logit working model",
    "mix": (
        "the known policy mix, as its contrast against low, and as its mean in "
        "randomizer_projection"
    ),
    "up": "the contrast of the shift up at both nodes against the natural course",
    "classifier_route": (
        "the primary policy contrast, its ratio estimated by a classifier on the true log ratio"
    ),
    "x1_25": "the contrast of the dose times 1.25, capped at 5.5, against the natural course",
    "piecewise": "the contrast of the dose lowered by 0.5 above 3 against the natural course",
    "halve": "the contrast of the declared policy halve below 3 against the natural course",
    "categorical_mtp": "the minus-one policy contrast on the six-level law",
    "vector_node": "the vector-node policy contrast on two binary components",
    "randomized_mtp": "the contrast of a randomized node-2 shift against the natural course",
    "survival_mtp_h2": "the cumulative risk at t = 2 of a policy plan on the survival law",
    "msm_mtp": "the dose coefficient of a working model over three continuous plans",
    "history": "the contrast of the history-reading plan against the natural course",
    "point_vector_density": (
        "the flag-and-dose point contrast, its ratio the oracle joint induced density"
    ),
    "point_vector_classifier": (
        "the flag-and-dose point contrast, its ratio a classifier on the true log ratio"
    ),
    "point_vector_ordered_density": (
        "the ordered two-dose contrast read as a point treatment, its ratio the oracle joint "
        "induced density"
    ),
    "longitudinal_vector_density": (
        "the two-node flag-and-dose contrast, its ratios the oracle joint induced density"
    ),
    "longitudinal_vector_classifier": (
        "the two-node flag-and-dose contrast, its ratios a classifier on the true log ratio"
    ),
    "longitudinal_vector_ordered_density": (
        "the ordered two-dose node contrast, its ratio the oracle joint induced density"
    ),
    "longitudinal_vector_ordered_classifier": (
        "the ordered two-dose node contrast, its ratio a classifier on the true log ratio"
    ),
    "flag_scaled": "the two-node contrast of the flag-scaled plan against the natural course",
    "p1": "the two-node contrast of the nonbinding 0.9 scale against the natural course",
    "p2": (
        "the two-node contrast of the shift held at the history-dependent support boundary "
        "against the natural course"
    ),
    "p3": "the two-node contrast of the warped map against the natural course",
    "p4": "the two-node contrast of the two-branch map against the natural course",
    "p5": (
        "the two-node contrast of the map that sends an interval to the declared atom against "
        "the natural course"
    ),
    "p4_sdr": "the two-branch contrast read by the sequentially doubly robust estimator",
    "p5_sdr": "the declared-atom contrast read by the sequentially doubly robust estimator",
    **{
        f"up_oracle_tau{t}": f"the mean purchase under the 12% price rise at {t} nodes, exact ratio"
        for t in (2, 3, 5)
    },
    **{
        f"up_learned_density_tau{t}": (
            f"the mean purchase under the 12% price rise at {t} nodes, learned density"
        )
        for t in (2, 3, 5)
    },
    **{
        f"up_learned_classifier_tau{t}": (
            f"the mean purchase under the 12% price rise at {t} nodes, learned classifier"
        )
        for t in (2, 3, 5)
    },
    "revenue_learned_classifier_tau3": (
        "the mean revenue under the 12% price rise at 3 nodes, learned classifier"
    ),
    "crossfit_ltmle_learned_classifier_tau3": (
        "the mean purchase under the 12% price rise at 3 nodes, LTMLE cross-fitted over five folds"
    ),
    "crossfit_sdr_learned_classifier_tau3": (
        "the mean purchase under the 12% price rise at 3 nodes, SDR cross-fitted over five folds"
    ),
    "up_tau3": "the mean purchase under the 12% price rise at 3 nodes",
    "up_tau5": "the mean purchase under the 12% price rise at 5 nodes",
    "population_dose_quantile_tau2": "the population target of the dose-quantile cap, one fold",
    "population_quantile_regression_tau2": (
        "the population target of the quantile-regression cap, one fold"
    ),
    "population_dose_quantile_crossfit_tau2": (
        "the population target of the dose-quantile cap, five folds, density route"
    ),
    "population_dose_quantile_crossfit_classifier_tau2": (
        "the population target of the dose-quantile cap, five folds, classifier route"
    ),
    "fold_average_dose_quantile_tau2": "the fold-average target of the dose-quantile cap",
    "fold_average_quantile_regression_tau2": (
        "the fold-average target of the quantile-regression cap"
    ),
    "ordered_full": "the ordered-law node contrast of the full plan against the natural course",
    "ordered_crossed": (
        "the ordered-law node contrast of the crossed plan against the natural course"
    ),
    "policy_risk_h2": "the cumulative risk at t = 2 of a known policy on the survival law",
    "msm_policy": "the dose coefficient of a working model over low, mix and taper",
    "primary": "every primary estimand of the study",
    # The stratum labels of the baseline-strata study: ``v<s>_<parameter>``.
    **{
        f"v{stratum}_{key}": f"{words} in stratum V = {stratum}"
        for stratum in (0, 1, 2)
        for key, words in (
            ("ey1", "mean under treatment"),
            ("ey0", "mean under no treatment"),
            ("ate", "average treatment effect"),
            ("att", "average effect on the treated"),
            ("atc", "average effect on the untreated"),
            ("par", "population attributable risk"),
        )
    },
    **{
        f"v{stratum}_ate_crossfit": (
            f"average treatment effect in stratum V = {stratum}, from the cross-fitted fit"
        )
        for stratum in (0, 1, 2)
    },
    # The stratified incremental and MSM study: incremental labels ``v<s>_<parameter>``,
    # linked-MSM labels ``logit_<scope>_<term>``, dose labels ``continuous_<link>_<scope>_<term>``.
    **{
        f"v{stratum}_{key}": f"{words} in stratum V = {stratum}"
        for stratum in (0, 1, 2)
        for key, words in (
            ("ey_natural", "incremental mean at odds multiplier 1"),
            ("ey_x2", "incremental mean at odds multiplier 2"),
            ("ey_x05", "incremental mean at odds multiplier 0.5"),
            ("ate_x2", "incremental contrast of multiplier 2 against 1"),
            ("ate_x05", "incremental contrast of multiplier 0.5 against 1"),
            ("a", "treatment coefficient of the logit MSM (1, a, W)"),
            ("a_arm", "treatment coefficient of the logit MSM (1, a)"),
        )
    },
    **{
        f"natural_{scope}": f"cross-fitted natural-course mean with a missing outcome, {where}"
        for scope, where in (
            ("marginal", "marginal"),
            *((f"v{stratum}", f"in stratum V = {stratum}") for stratum in (0, 1, 2)),
        )
    },
    "marginal_ate": "average treatment effect over every stratum, from the stratified fit",
    **{
        f"logit_{scope}_{key}": f"{words} of the logit MSM (1, a, W), {where}"
        for scope, where in (
            ("marginal", "marginal"),
            *((f"v{stratum}", f"in stratum V = {stratum}") for stratum in (0, 1, 2)),
        )
        for key, words in (
            ("intercept", "intercept coefficient"),
            ("a", "treatment coefficient"),
            ("w", "baseline-covariate coefficient"),
        )
    },
    **{
        f"continuous_{link}_{scope}_{key}": (
            f"{words} of the {link}-link dose MSM (1, a) over the grid, {where}"
        )
        for link in ("identity", "logit")
        for scope, where in (
            ("marginal", "marginal"),
            *((f"v{stratum}", f"in stratum V = {stratum}") for stratum in (0, 1, 2)),
        )
        for key, words in (("intercept", "intercept coefficient"), ("a", "dose coefficient"))
    },
    **{
        f"identity_{scope}_{key}": f"{words} of the identity MSM (1, a, W), {where}"
        for scope, where in (
            ("marginal", "marginal"),
            *((f"v{stratum}", f"in stratum V = {stratum}") for stratum in (0, 1, 2)),
        )
        for key, words in (
            ("intercept", "intercept coefficient"),
            ("a", "treatment coefficient"),
            ("w", "baseline-covariate coefficient"),
        )
    },
    "ipsi_strata": "the five marginal and fifteen stratum incremental parameters of one fit",
    # The composite-indicator missing-data study: the corrected-inference contrasts,
    # the joint-coverage labels and the correction cycle.
    "composite_observational_ate": "the ATE of the two-arm law with an observational MAR outcome",
    "composite_binary_ate": "the ATE of the two-arm law with a MAR outcome and treatment",
    "composite_three_arm_ate_low": (
        "the difference low versus high of the three-arm law with a MAR outcome and treatment"
    ),
    "composite_three_arm_ate_mid": (
        "the difference mid versus high of the three-arm law with a MAR outcome and treatment"
    ),
    "composite_observational": (
        "the two arm means, the difference, the risk ratio and the odds ratio of the two-arm "
        "law with an observational MAR outcome"
    ),
    "composite_binary": (
        "the two arm means, the difference, the risk ratio and the odds ratio of the two-arm "
        "law with a MAR outcome and treatment"
    ),
    "composite_three_arm": (
        "the three arm means, two differences, two risk ratios and two odds ratios of the "
        "three-arm law with a MAR outcome and treatment"
    ),
    "composite_cycle": "composite-indicator correction cycle",
    "l1": "L1, two arms and a binary outcome",
    "l2": "L2, two arms and a bounded continuous outcome",
    "l3": "L3, three arms and a binary outcome",
    "l4": "L4, three arms and a bounded continuous outcome",
    "l1_ey0": "L1 mean under arm 0",
    "l1_ey1": "L1 mean under arm 1",
    "l1_ate": "L1 average treatment effect",
    "l1_rr": "L1 log risk ratio",
    "l1_or": "L1 log odds ratio",
    "l2_ey0": "L2 mean under arm 0",
    "l2_ey1": "L2 mean under arm 1",
    "l2_ate": "L2 average treatment effect",
    "l3_ey_high": "L3 mean under arm high",
    "l3_ey_low": "L3 mean under arm low",
    "l3_ey_mid": "L3 mean under arm mid",
    "l3_ate_low": "L3 difference, low versus high",
    "l3_ate_mid": "L3 difference, mid versus high",
    "l3_rr_low": "L3 log risk ratio, low versus high",
    "l3_rr_mid": "L3 log risk ratio, mid versus high",
    "l3_or_low": "L3 log odds ratio, low versus high",
    "l3_or_mid": "L3 log odds ratio, mid versus high",
    "l4_ey_high": "L4 mean under arm high",
    "l4_ey_low": "L4 mean under arm low",
    "l4_ey_mid": "L4 mean under arm mid",
    "l4_ate_low": "L4 difference, low versus high",
    "l4_ate_mid": "L4 difference, mid versus high",
    # The full-refit bootstrap and derived-contrast study.  Each label names the quantity,
    # the law and the fit; a ``boot_`` label reads the percentile interval of the bootstrap.
    "rr_end_of_study": "log risk ratio of always versus never, end-of-study law",
    "or_end_of_study": "log odds ratio of always versus never, end-of-study law",
    "rr_survival": "log risk ratio of always versus never at t = 4, four-node survival law",
    "survival_rr_survival": (
        "log ratio of the survival probabilities of always versus never at t = 4, four-node law"
    ),
    "rmst_survival": "RMST of always up to t = 5, four-node survival law",
    "rmst_contrast_survival": "RMST contrast of always versus never up to t = 5, four-node law",
    "rmst_crossfit": "RMST of always up to t = 5, four-node law, five folds",
    "rmst_contrast_crossfit": "RMST contrast up to t = 5, four-node law, five folds",
    "boot_ey_end_of_study": "bootstrap percentile interval of the always mean, end-of-study law",
    "boot_ate_end_of_study": "bootstrap percentile interval of the always-never contrast",
    "boot_ey_crossfit": "bootstrap percentile interval of the always mean, five folds",
    "boot_ate_crossfit": "bootstrap percentile interval of the contrast, five folds",
    "boot_risk2_survival": "bootstrap percentile interval of the always risk at t = 2",
    "boot_risk4_survival": "bootstrap percentile interval of the always risk at t = 4",
    "boot_rmst_survival": "bootstrap percentile interval of the always RMST up to t = 5",
    "boot_ey_clustered": "cluster-bootstrap percentile interval of the always mean, 60 clusters",
    "boot_ate_clustered": "cluster-bootstrap percentile interval of the contrast, 60 clusters",
    "boot_ate_point_tmle": "bootstrap percentile interval of the point-treatment ATE",
    "survival_h2": "difference of always and never in the risk at t = 2, survival law",
    "competing_h2": "difference of always and never in the death incidence at t = 2",
    "policy_mix": "contrast of the known policy mix against low, three-level law",
    "mtp_up": "the shift up at both nodes against the natural course, continuous dose",
    "msm_identity": "the duration coefficient of the identity-link working model",
    "msm_logit": "the duration coefficient of the logit-link working model",
    "selection_weighted": "contrast of always against never on the selected law, with weights",
    "clusters_200": "contrast of always against never, 200 clusters of 10 rows",
    "clusters_20": "contrast of always against never, 20 clusters of 100 rows",
    "six_regimens": "the six regimen means and five contrasts of the two-node law",
    # ``longitudinal-incremental`` and ``longitudinal-incremental-dropout`` (X19).
    "mean_x2": "odds-x2 mean at every node, three-node law",
    "contrast_x2": "odds-x2 against the natural course, three-node law",
    "contrast_x2_in_sample": "odds-x2 against the natural course, one fold",
    "dropout_contrast_x2_in_sample": "odds-x2 contrast at the end of study, with dropout, one fold",
    "survival_contrast_x2_h3_in_sample": "odds-x2 contrast in the risk by t = 3, with dropout, one fold",
    "known_x2": "odds-x2 mean with the true treatment probabilities declared",
    "no_positivity": "odds-x2 contrast on a law with propensities of exactly 0 and 1",
    "per_node": "odds multipliers 2, 0.5 and 1.5 at the three nodes",
    "tilt_then_static": "odds-x2 at nodes 1 and 2, then arm 1 at node 3",
    "contrast_x1_5": "odds-x1.5 against the natural course, small arm effects",
    "nine_tilts": "nine odds multipliers from 0.25 to 4, their means and eight contrasts",
    "dropout_contrast_x2": "odds-x2 contrast at the end of study, with dropout",
    "survival_contrast_x2_h3": "odds-x2 contrast in the risk by t = 3, with dropout",
    "competing_contrast_x2_h3": "odds-x2 contrast in the cause-1 incidence by t = 3",
    "msm_log_delta": "slope of the identity-link working model in log odds multiplier",
    "weighted_contrast_x0_5": "odds-x0.5 contrast on the law weighted by 1 + A1 + W/2",
    "survival_grid": "three plans by three horizons, the risks and six contrasts",
    # ``survival-node-bounds`` (X33): the labels of its calibration, witness and necessity cells.
    **{
        f"{model}_{key}": f"{fit}, {words}"
        for model, fit in (
            ("mean_model", "iterated-mean fit with valid node bounds"),
            ("hazard_model", "hazard fit with valid node bounds"),
        )
        for key, words in (
            ("f5_arm1", "risk by visit 5 under arm 1"),
            ("rd5", "risk difference by visit 5"),
            ("f3_arm1", "risk by visit 3 under arm 1"),
            ("f3_arm0", "risk by visit 3 under arm 0"),
            ("f5_arm0", "risk by visit 5 under arm 0"),
        )
    },
    "mean_model": "iterated-mean fit with valid node bounds, five visits",
    "hazard_model": "hazard fit with valid node bounds, five visits",
    "f5_arm1": (
        "risk by visit 5 under arm 1 of the iterated-mean fit at n = 16,000, with a saturated "
        "outcome regression and an intercept-only treatment mechanism"
    ),
    # ``adaptive-iptw`` (X21): the keys its cells read.  ``ey0``, ``ey1``, ``ate``, ``rr``,
    # ``or``, ``ey_obs``, ``par`` and ``paf`` keep the readings above.
    "ey_never": "mean outcome if no one is treated",
    "ey_d": "mean outcome under the deterministic rule that treats where the rule says so",
    "ey_s": "mean outcome under the stochastic rule that treats with the declared probability",
    "ate_d": "mean under the deterministic rule minus the mean if no one is treated",
    "ate_s": "mean under the stochastic rule minus the mean if no one is treated",
    "ate_s_vs_d": "mean under the stochastic rule minus the mean under the deterministic rule",
    "ey_high": "mean outcome under the high arm",
    "ey_medium": "mean outcome under the medium arm",
    "ey_low": "mean outcome under the low arm",
    "ey_mid": "mean outcome under the middle arm",
    "ate_low": "mean under the low arm minus the mean under the high arm",
    "ate_medium": "mean under the medium arm minus the mean under the high arm",
    "ate_mid": "mean under the middle arm minus the mean under the high arm",
    "ate_v0": "average treatment effect in stratum V = 0",
    "ate_v1": "average treatment effect in stratum V = 1",
    "ate_v2": "average treatment effect in stratum V = 2",
    "p_ey1": "arm-1 mean",
    "p_ate": "average treatment effect",
    "pd_d": "mean under the deterministic rule",
    "pd_s": "mean under the stochastic rule",
    "cn_ey_obs": "observed outcome mean under the natural course",
    "rm_d": "mean under the deterministic rule, with a missing outcome",
}

#: The labels of ``sequential-baselines``: an estimator, a law and a target.  Mirrored from
#: ``tests.studies.canonical_sequential_baselines.CALIBRATION_LABELS``, which this module cannot
#: import; ``tests/unit/test_sequential_baselines_design.py`` pins that every cell resolves.
_BASELINE_ESTIMATORS = {
    "ice": "sequential g-computation (ICE)",
    "hajek": "sequential IPW, Hajek",
    "ht": "sequential IPW, Horvitz-Thompson",
}
_BASELINE_TARGETS = {
    "d2_static_12": "law D2, mean under arm 1 then arm 2",
    "d2_static_00": "law D2, mean under arm 0 at both times",
    "d2_rule": "law D2, mean under the rule A1 = L1, A2 = 2 L2",
    "d2_policy": "law D2, mean under arm 1 then the known three-arm policy",
    "d2_contrast": "law D2, arm 1 then arm 2 against arm 0 at both times",
    "d2w_static_12": "law D2 weighted by 1 + 0.5 W, mean under arm 1 then arm 2",
    "d2s_risk_h1": "law D2S, risk by t = 1 under arm 1",
    "d2s_risk_h2": "law D2S, risk by t = 2 under arm 1 then arm 2",
    "d2s_contrast_h2": "law D2S, risk difference by t = 2 against arm 0 at both times",
    "d2c_relapse_h2": "law D2C, relapse incidence by t = 2 under arm 1 then arm 2",
    "g_treat_all": "law G, mean under treatment at all three times",
    "g_treat_none": "law G, mean under no treatment",
    "g_rule": "law G, mean under the rule A_t = V",
    "g_contrast": "law G, treatment at all three times against none",
    "gm_down": "law GM, mean under the dose lowered by 0.5 where it stays in support",
    "gm_contrast": "law GM, the lowered dose against the natural course",
    "x_contrast": "law X, treatment at all three times against none",
}
ARMS.update(
    {
        f"{estimator}_{target}": f"{words}: {target_words}"
        for estimator, words in _BASELINE_ESTIMATORS.items()
        for target, target_words in _BASELINE_TARGETS.items()
    }
)

#: The calibration labels of the plug-in omitted-variable bound study (X26): one per scenario,
#: parameter and row.  Mirrored from ``tests.studies.omitted_variable_bound_plugin``, which this
#: module cannot import; ``tests/unit/test_omitted_variable_plugin_design.py`` pins that every
#: cell resolves.
_PLUGIN_SCENARIOS = {
    "binary_logistic": "in-sample logistic mechanism",
    "binary_logistic_crossfit": "five-fold cross-fitted logistic mechanism",
    "binary_logistic_ridge": "logistic mechanism with a fixed L2 penalty, C = 1",
    "binary_logistic_ridge_crossfit": (
        "five-fold cross-fitted logistic mechanism with a fixed L2 penalty, C = 1"
    ),
    "multinomial_logistic": "three-arm multinomial logistic mechanism",
    "multinomial_logistic_crossfit": (
        "five-fold cross-fitted three-arm multinomial logistic mechanism"
    ),
    "binary_known": "declared known mechanism",
    "binary_unbalanced": "in-sample logistic mechanism, treated share about 0.19",
}
_PLUGIN_PARAMETERS = {
    "binary_logistic": ("ate", "att", "atc"),
    "binary_logistic_crossfit": ("ate", "att", "atc"),
    "binary_logistic_ridge": ("ate", "att", "atc"),
    "binary_logistic_ridge_crossfit": ("ate", "att", "atc"),
    "multinomial_logistic": ("ate_2v0", "att_2v0", "atc_1v0"),
    "multinomial_logistic_crossfit": ("ate_2v0", "att_2v0", "atc_1v0"),
    "binary_known": ("ate", "att", "atc"),
    "binary_unbalanced": ("ate", "att", "atc"),
}
_PLUGIN_TOKENS = {
    "ate": "the ATE",
    "att": "the ATT",
    "atc": "the ATC",
    "ate_2v0": "the ATE of arm 2 versus arm 0",
    "att_2v0": "the ATT of arm 2 versus arm 0",
    "atc_1v0": "the ATC of arm 1 versus arm 0",
}
_PLUGIN_ROWS = {
    "lower": "lower plug-in omitted-variable bound on",
    "upper": "upper plug-in omitted-variable bound on",
}
ARMS.update(
    {
        f"{scenario}_{token}_{row}": f"{row_words} {_PLUGIN_TOKENS[token]}, {words}"
        for scenario, words in _PLUGIN_SCENARIOS.items()
        for token in _PLUGIN_PARAMETERS[scenario]
        for row, row_words in _PLUGIN_ROWS.items()
    }
)

#: The laws of the primary scenarios of the plug-in study (X26): one per scenario and parameter,
#: because a refusal of one parameter must lose that parameter's rows only.  Mirrored from
#: ``tests.studies.omitted_variable_bound_plugin.PRIMARY``.
_PLUGIN_LAWS = {
    "binary_logistic": (
        "`make_linear_ate`, in-sample TMLE with an unpenalized logistic treatment model"
    ),
    "binary_logistic_crossfit": (
        "`make_linear_ate`, five-fold cross-fitted TMLE with an unpenalized logistic treatment "
        "model"
    ),
    "binary_logistic_ridge": (
        "`make_linear_ate`, in-sample TMLE with a logistic treatment model under a fixed L2 "
        "penalty, C = 1"
    ),
    "binary_logistic_ridge_crossfit": (
        "`make_linear_ate`, five-fold cross-fitted TMLE with a logistic treatment model under a "
        "fixed L2 penalty, C = 1"
    ),
    "multinomial_logistic": (
        "three-arm law with a multinomial logistic mechanism, in-sample TMLE with an "
        "unpenalized multinomial logistic treatment model"
    ),
    "multinomial_logistic_crossfit": (
        "three-arm law with a multinomial logistic mechanism, five-fold cross-fitted TMLE with "
        "an unpenalized multinomial logistic treatment model"
    ),
    "binary_known": "`make_linear_ate`, in-sample TMLE with its own mechanism declared known",
    "binary_unbalanced": (
        "`make_linear_ate` with the treatment intercept moved to -1.5 (about 19 percent "
        "treated), in-sample TMLE with an unpenalized logistic treatment model"
    ),
}

#: The labels of ``structural-nested-mean``: a setup and one estimand of it.  Mirrored from
#: ``tests.studies.canonical_structural_nested_mean.CALIBRATION_LABELS``, which this module
#: cannot import; ``tests/unit/test_structural_nested_mean_design.py`` pins that every cell
#: resolves.
_STRUCTURAL_SETUPS = {
    "s2": "law S2, two binary nodes",
    "s3": "law S3, three binary nodes with a path from each treatment to the later ones",
    "k": "law K, two nodes of three levels",
    "c": "law C, two doses with linear conditional means",
    "h": "law H, blip features of the baseline and the past treatment",
    "p": "law P, the second blip reads the first treatment",
    "m": "law M, the second treatment and the outcome read the first covariate",
    "l": "law L, the second blip reads the second covariate, which the feature declares",
    "n": "law N, nonlinear nuisances fitted by a shallow booster",
    "s2w": "law S2 with the known weight 0.5 + 1.5 W",
    "s2c": "law S2 in clusters of ten that share the baseline and an outcome intercept",
    "sp": "law SP, treatment logits multiplied by three, weak positivity",
    "lu": "law L fitted without its second-covariate feature",
    "vb": "law P fitted with a baseline-only second feature",
}
_STRUCTURAL_KEYS = {
    "s2": ("blip_a1_l1", "blip_a2_l1", "ey0", "ey_r11", "ate_r11", "ey_r10"),
    "s3": ("blip_a1_l1", "blip_a2_l1", "blip_a3_l1", "ey0", "ate_r111", "ate_r101"),
    "k": (
        "blip_a1_l1",
        "blip_a1_l2",
        "blip_a2_l1",
        "blip_a2_l2",
        "ey0",
        "ey_r12",
        "ate_r12",
        "ate_r21",
    ),
    "c": ("blip_a1", "blip_a2", "ey0", "ey_dose_mid", "ate_dose_mid", "ate_dose_high"),
    "h": (
        "blip_a1_l1",
        "blip_a1_l1_w",
        "blip_a2_l1",
        "blip_a2_l1_w",
        "blip_a2_l1_a1",
        "ey0",
        "ey_r11",
        "ate_r11",
        "ate_r10",
    ),
    "p": ("blip_a1_l1", "blip_a2_l1", "blip_a2_l1_a1", "ey0", "ate_r11", "ey_r01"),
    "m": ("blip_a1_l1", "blip_a2_l1", "ey0", "ate_r11"),
    "l": ("blip_a1_l1", "blip_a2_l1", "blip_a2_l1_s2", "ey0"),
    "n": ("blip_a1_l1", "blip_a2_l1", "ey0", "ate_r11"),
    "s2w": ("blip_a1_l1", "ey0", "ey_r11", "ate_r11"),
    "s2c": ("blip_a1_l1", "ey0", "ey_r11", "ate_r11"),
    "sp": (
        "blip_a1_l1",
        "blip_a1_l1_w",
        "blip_a2_l1",
        "blip_a2_l1_w",
        "blip_a2_l1_a1",
        "ey0",
        "ey_r11",
        "ate_r11",
        "ate_r10",
        "ate_r01",
    ),
    "lu": ("ey_r11", "ate_r10", "ate_r01"),
    "vb": ("ey_r11", "ate_r11", "ate_r10", "ate_r01"),
}
_STRUCTURAL_ORDINALS = {"1": "first", "2": "second", "3": "third"}
_STRUCTURAL_FEATURES = {"w": "baseline W", "a1": "first treatment", "s2": "second covariate"}
_STRUCTURAL_BLIP = re.compile(r"^blip_a(?P<node>\d)(?:_l(?P<level>\d))?(?:_(?P<feature>w|a1|s2))?$")


def _structural_target(key: str) -> str:
    """The estimand a short label key names, in words."""
    if key == "ey0":
        return "mean outcome with every treatment at its reference"
    blip = _STRUCTURAL_BLIP.match(key)
    if blip is not None:
        words = f"blip of the {_STRUCTURAL_ORDINALS[blip.group('node')]} treatment"
        if blip.group("level"):
            words += f" at level {blip.group('level')}"
        if blip.group("feature"):
            words += f", slope on the {_STRUCTURAL_FEATURES[blip.group('feature')]}"
        return words
    kind, _, plan = key.partition("_")
    if kind == "ey":
        return f"mean outcome under regimen {plan}"
    return f"contrast of regimen {plan} against the reference regimen"


ARMS.update(
    {
        f"{setup}_{key}": f"{_STRUCTURAL_SETUPS[setup]}: {_structural_target(key)}"
        for setup, keys in _STRUCTURAL_KEYS.items()
        for key in keys
    }
)

#: The model-check cells of ``structural-nested-mean``, by their cell prefix.
ARMS.update(
    {
        "s2_f2": "law S2, structural model correct, two outer folds, n = 1,000",
        "s2_f10": "law S2, structural model correct, ten outer folds, n = 1,000",
        "h_f2": "law H, structural model correct, two outer folds, n = 1,000",
        "h_f10": "law H, structural model correct, ten outer folds, n = 1,000",
        "s2_n2000_f2": "law S2, structural model correct, two outer folds, n = 2,000",
        "s2_n2000_f10": "law S2, structural model correct, ten outer folds, n = 2,000",
        "h_n2000_f2": "law H, structural model correct, two outer folds, n = 2,000",
        "h_n2000_f10": "law H, structural model correct, ten outer folds, n = 2,000",
        "sp_f2": "law SP, treatment logits multiplied by three, two outer folds, n = 2,000",
        "sp_f10": "law SP, treatment logits multiplied by three, ten outer folds, n = 2,000",
        "lu_f2": "law L fitted without its second-covariate feature, two outer folds, n = 2,000",
        "lu_f10": "law L fitted without its second-covariate feature, ten outer folds, n = 2,000",
    }
)

#: The labels of ``parametric-gformula``: a method, a law and a target.  Mirrored from
#: ``tests.studies.parametric_gformula_properties``, which this module cannot import;
#: ``tests/unit/test_parametric_gformula_design.py`` pins that every cell resolves.
_PGF_TARGETS = {
    "g1_treat_all": "law G1, mean under treatment at all three times",
    "g1_treat_none": "law G1, mean under no treatment",
    "g1_rule": "law G1, mean under the rule A_t = L_t",
    "g1_odds_x2": "law G1, mean under the odds tilt of two",
    "g1_contrast": "law G1, treatment at all three times against none",
    "g2_treat_all": "law G2, mean under treatment at all three times",
    "g2_treat_none": "law G2, mean under no treatment",
    "g2_policy_b": "law G2, mean under the known policy that treats with probability 0.3 + 0.4 B_t",
    "g2_contrast": "law G2, treatment at all three times against none",
    "g3_relapse_h2": "law G3, relapse incidence by t = 2 under treatment at all four times",
    "g3_relapse_h4": "law G3, relapse incidence by t = 4 under treatment at all four times",
    "g3_death_h4": "law G3, death incidence by t = 4 under treatment at all four times",
    "g3_contrast_h4": "law G3, relapse incidence by t = 4, treatment at all four times against none",
    "g4_shift": "law G4, mean under a dose raised by 0.5 at each of the three times",
    "g4_dose": "law G4, mean under the fixed dose 0.5 at each of the three times",
    "g4_rule_dose": "law G4, mean under the dose 0.5 + 0.5 L_t set by a rule at each time",
    "g4_contrast": "law G4, the raised dose against the natural course",
    "g5_treat_all": "law G5, mean under treatment at all three times",
    "g5_treat_none": "law G5, mean under no treatment",
    "g5_odds_x2": "law G5, mean under the odds tilt of two",
    "g5_contrast": "law G5, treatment at all three times against none",
}
_PGF_PAIRS = {
    "g1_treat_all": _PGF_TARGETS["g1_treat_all"],
    "g1_treat_none": _PGF_TARGETS["g1_treat_none"],
    "g1_contrast": _PGF_TARGETS["g1_contrast"],
    "g2_treat_all": _PGF_TARGETS["g2_treat_all"],
    "g2_treat_none": _PGF_TARGETS["g2_treat_none"],
    "g2_contrast": _PGF_TARGETS["g2_contrast"],
}
_PGF_SHARED = {
    "g1_treat_all": _PGF_TARGETS["g1_treat_all"],
    "g1_contrast": _PGF_TARGETS["g1_contrast"],
    "g1_rule": _PGF_TARGETS["g1_rule"],
    "g1_odds_x2": _PGF_TARGETS["g1_odds_x2"],
    "g2_treat_all": _PGF_TARGETS["g2_treat_all"],
    "g2_contrast": _PGF_TARGETS["g2_contrast"],
}
ARMS.update(
    {
        **{f"nice_{key}": f"parametric g-formula: {words}" for key, words in _PGF_TARGETS.items()},
        **{
            f"pair_{key}": f"parametric g-formula against main-terms ICE: {words}"
            for key, words in _PGF_PAIRS.items()
        },
        **{
            f"target_{key}": f"parametric g-formula beside the targeted estimator: {words}"
            for key, words in _PGF_SHARED.items()
        },
        "g4q_dose": (
            "parametric g-formula with main-terms models: law G4Q, mean under the fixed dose 0.5, "
            "where the outcome carries a quadratic dose term"
        ),
    }
)

#: The rows of the X82 SG limit coverage studies, as arm labels; see the section at the end.
_SG_OVB = "ovb-limit-coverage"
_SG_EVALUE = "evalue-limit-coverage"
_SG_STRENGTH = "at cf_y 0.3, cf_d 0.2, rho 1"
_SG_PARAMETERS = {
    "O1": {
        "ate": "the ATE, outcome missing at random",
        "ey1": "the arm-1 mean, outcome missing at random",
    },
    "O2": {
        "cde_z0": "the controlled direct effect at Z = 0",
        "cde_z1": "the controlled direct effect at Z = 1",
    },
    "O3": {
        "coin": "the mean under the stochastic regime coin",
        "coin_vs_always": "the contrast of the regime coin with always treat",
    },
    "O4": {
        "w2": "the W2 coefficient of the linear MSM with modifier W2",
        "a_w2": "the treatment-by-W2 coefficient of the linear MSM with modifier W2",
    },
    "O5": {
        "plus1": "the mean under the shift +1 on the dose atoms",
        "plus1_vs_natural": "the contrast of the shift +1 with the natural course",
    },
}
_SG_OVB_ROWS = {
    f"{label}_{end}": (cell, f"{end} omitted-variable bound on {words}, {_SG_STRENGTH}")
    for cell, parameters in _SG_PARAMETERS.items()
    for label, words in parameters.items()
    for end in ("lower", "upper")
}
_SG_EVALUE_ROWS = {
    **{
        f"E3_rr[V={stratum}]": f"risk ratio in stratum V={stratum}, on the log scale"
        for stratum in (0, 1, 2)
    },
    "E4_rr_t5": "ratio of the arm-1 and arm-0 risks at t = 5, on the log scale",
}

ARMS.update({f"{cell}_{row}": words for row, (cell, words) in _SG_OVB_ROWS.items()})
ARMS.update(_SG_EVALUE_ROWS)

#: The rows of the X82 SH limit coverage study, as arm labels; see the section at the end.
_SH_ROWS = {
    f"{cell}_ate_{end}": f"{end} end of the ATE bound under the marginal sensitivity model at "
    "Gamma 1.5"
    for cell in ("SH1", "SH2")
    for end in ("lower", "upper")
}
ARMS.update(_SH_ROWS)

#: The settings of ``learned-rule-compositions`` (X11 PR A) that no other study names. Its
#: ``weighted`` and ``clustered`` labels are shared, so :data:`STUDY_ARMS` scopes their words.
ARMS.update(
    {
        "fold_targeted": (
            "the learned-rule law with the blip 0.1 + W1, fitted with fold targeting"
        ),
        "clustered_few": (
            "the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 30 clusters "
            "of 34 rows (n = 1,020), with the t reference at 20 degrees of freedom"
        ),
    }
)

#: The second law of ``learned-rule-split-mean`` (X11 PR B). Its ``non_exceptional`` label is the
#: shared one, and :data:`STUDY_ARMS` scopes that label's words to the study.
ARMS["misspecified_limit"] = (
    "the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner "
    "has no W1^2 term"
)

#: The two arms of ``learned-rule-learners`` (X11 PR C), which :data:`STUDY_ARMS` scopes to the
#: study. They sit here because the cell-name pattern below is built from :data:`ARMS` at import.
ARMS["blip"] = "the learned-rule law with a blip rule learner"
ARMS["weighted_classification"] = "the learned-rule law with a weighted-classification rule learner"

#: The parameter labels of ``fold-evaluated-strata`` (X11 PR E1, X28), one arm per setting,
#: target and stratum, ``<setting>_<target>_<stratum>``. They sit here because the cell-name
#: pattern below is built from :data:`ARMS` at import. Mirrored from
#: ``tests.studies._fold_evaluated_strata_law``, which this module cannot import.
_STRATA_SETTINGS = {
    "unweighted": "the bounded (beta) outcome law with baseline strata, rows only",
    "weighted": "the bounded (beta) outcome law with baseline strata and the weight 1 + 1.5 W2",
    "binary_unweighted": "the binary outcome law with baseline strata, rows only",
    "binary_weighted": "the binary outcome law with baseline strata and the weight 1 + 1.5 W2",
    "binary_sparse": (
        "the binary outcome law with baseline strata and the steeper blip 0.1 + 3.0 W1, rows only"
    ),
}
_STRATA_TARGETS = {"ate": "the ATE", "ey0": "the arm-0 mean", "ey1": "the arm-1 mean"}
_STRATA_PARTS = {
    "s0": "in stratum S = 0",
    "s1": "in stratum S = 1",
    "s2": "in stratum S = 2",
    "all": "marginally",
}
ARMS.update(
    {
        f"{setting}_{target}_{part}": f"{law}, {words} {where}"
        for setting, law in _STRATA_SETTINGS.items()
        for target, words in _STRATA_TARGETS.items()
        for part, where in _STRATA_PARTS.items()
    }
)

#: The parameter labels of ``learned-rule-strata`` (X11 PR E2, X11 (m)), one arm per setting
#: and stratum, ``<setting>_<stratum>``. Mirrored from ``tests.studies._learned_rule_strata_law``,
#: which this module cannot import.
_LEARNED_STRATA_SETTINGS = {
    "unweighted": "the bounded (beta) learned-rule law with baseline strata, rows only",
    "weighted": (
        "the bounded (beta) learned-rule law with baseline strata and the weight 1 + 1.5 W2"
    ),
    "binary_steep": "the binary learned-rule law with baseline strata, rows only",
    "binary_steep_weighted": (
        "the binary learned-rule law with baseline strata and the weight 1 + 1.5 W2"
    ),
    "binary_weak": (
        "the binary learned-rule law with baseline strata and the weak blip 0.1 - 0.3 W1, rows only"
    ),
    "binary_moderate": (
        "the binary learned-rule law with baseline strata and the blip -0.6 + 1.5 W1, rows only"
    ),
}
ARMS.update(
    {
        f"{setting}_{part}": f"{law}, the learned-rule value {where}"
        for setting, law in _LEARNED_STRATA_SETTINGS.items()
        for part, where in _STRATA_PARTS.items()
    }
)

#: The equal-fold setting of ``learned-rule-stacked`` (X11 PR F1, X11 (e)). Its ``weighted``
#: and ``clustered`` labels are shared, so :data:`STUDY_ARMS` scopes their words to the study.
ARMS["rows"] = "the learned-rule law with the blip 0.1 + W1 on rows, in ten equal folds"

#: Built from :data:`ARMS` rather than restated.  ``cell`` subscripts ``ARMS`` with whatever
#: this matches, so a hand-maintained second list is a ``KeyError`` waiting for the study that
#: extends one of them and not the other -- and that is the one lookup here which would not
#: raise :class:`Undescribed`.  Longest alternative first, so a longer arm is never shadowed
#: by a shorter one it starts with.
_ARM = re.compile(
    r"^(?P<arm>"
    + "|".join(re.escape(arm) for arm in sorted(ARMS, key=len, reverse=True))
    + r")__(?P<cell>.+)$"
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
    "cleverly-ctmle-oat-per-arm": "`cleverly` per-arm outcome-adaptive C-TMLE",
    "cleverly-ctmle-oat-per-arm-parametric": (
        "`cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism"
    ),
    "cleverly-ctmle-oat-per-arm-parametric-crossfit": (
        "`cleverly` cross-fitted per-arm outcome-adaptive C-TMLE with a parametric outcome and "
        "mechanism"
    ),
    "cleverly-ctmle-oat-per-arm-parametric-extensions": (
        "`cleverly` per-arm outcome-adaptive C-TMLE with a parametric outcome and mechanism, "
        "at three arms, with known weights and for the population targets"
    ),
    "drtmle-r-oat-per-arm": "R `drtmle` with `adapt_g`, by a binary recode per arm",
    "drtmle-r-oat-per-arm-strata-missing": (
        "R `drtmle` with `adapt_g`, by a binary recode per arm, with `DeltaY`, `DeltaA` and"
        " one fit per stratum"
    ),
    "cleverly-known-mechanism": "`cleverly` TMLE on a declared known mechanism",
    "tmle-r-known-g": "R `tmle` with the known mechanism as `g1W`",
    "cleverly-known-mechanism-drtmle": "`cleverly` DR-TMLE on a declared known mechanism",
    "cleverly-drtmle-one-wrong": (
        "`cleverly` DR-TMLE with guards Q and g and the univariate reduction"
    ),
    "cleverly-drtmle-foldwise": (
        "`cleverly` cross-fitted DR-TMLE with fold-wise reduced regressions"
    ),
    "drtmle-r-known-g": "R `drtmle` with the known mechanism as `gn`",
    "cleverly-known-node-mechanisms": "`cleverly` LTMLE on declared known node mechanisms",
    "ltmle-known-gform": "R `ltmle` with the known factors as a numeric `gform`",
    "cleverly": "`cleverly`",
    "cleverly-cross-fitted-point-survival": "`cleverly` cross-fitted point-treatment survival",
    "cleverly-cross-fitted-hazard-survival": (
        "`cleverly` cross-fitted point-treatment survival with hazard targeting"
    ),
    "cleverly-held-policy-ltmle": "`cleverly` LTMLE with a policy at a held baseline treatment",
    "survtmle": 'R `survtmle`, `method="mean"`',
    "cleverly-categorical-ltmle": "`cleverly` ordinary categorical LTMLE",
    "cleverly-cde-tmle": "`cleverly` controlled direct-effect TMLE",
    "cleverly-ltmle-derived": "`cleverly` LTMLE ratio contrast",
    "cleverly-clustered-cvtmle": "`cleverly` clustered point-treatment CV-TMLE",
    "cleverly-clustered-cross-fitted-ltmle": "`cleverly` clustered cross-fitted LTMLE",
    "cleverly-few-cluster-cross-fitted-ltmle": (
        "`cleverly` clustered cross-fitted LTMLE on a t reference at few clusters"
    ),
    "cleverly-clustered-unequal-cvtmle": (
        "`cleverly` clustered point-treatment CV-TMLE at unequal cluster sizes"
    ),
    "cleverly-cross-fitted-categorical-ltmle": "`cleverly` cross-fitted categorical LTMLE",
    "cleverly-cross-fitted-ltmle-msm": "`cleverly` cross-fitted longitudinal MSM projection",
    "cleverly-mtp-ltmle": "`cleverly` LTMLE under modified treatment policies",
    "cleverly-policy-tmle": "`cleverly` point-treatment TMLE under modified treatment policies",
    "cleverly-vector-mtp": (
        "`cleverly` point TMLE and LTMLE under modified treatment policies on a vector treatment"
    ),
    "cleverly-mtp-boundaries": (
        "`cleverly` LTMLE and SDR under modified treatment policies with declared supports and "
        "atoms"
    ),
    "cleverly-mtp-pricing": (
        "`cleverly` LTMLE and SDR under price policies that bind at a declared support"
    ),
    "cleverly-mtp-tuning": "`cleverly` LTMLE and SDR under a shift with a tuned cap",
    "cleverly-stochastic-categorical-ltmle": (
        "`cleverly` ordinary LTMLE under known stochastic categorical policies"
    ),
    "cleverly-cross-fitted-ltmle": "`cleverly` cross-fitted LTMLE",
    "cleverly-cross-fitted-ltmle-survival": "`cleverly` cross-fitted survival LTMLE",
    "cleverly-cross-fitted-competing-ltmle": "`cleverly` cross-fitted competing-risk LTMLE",
    "cleverly-competing-ltmle": "`cleverly` competing-risk LTMLE",
    "cleverly-ctmle-derivative": "`cleverly` C-TMLE0 derivative covariate",
    "cleverly-ctmle-oat": "`cleverly` outcome-adaptive C-TMLE",
    "cleverly-ctmle-selector": "`cleverly` selector-based C-TMLE",
    "cleverly-ctmle-sequences": "`cleverly` C-TMLE lasso and declared-learner sequences",
    "cleverly-fold-evaluated-cvtmle": "`cleverly` fold-evaluated CV-TMLE",
    "cleverly-fold-targeted-cvtmle": "`cleverly` fold-targeted CV-TMLE",
    "cleverly-repeated-cvtmle": "`cleverly` repeated stacked CV-TMLE",
    "cleverly-omitted-variable-bound": "`cleverly` omitted-variable bound",
    "cleverly-omitted-variable-bound-plugin": (
        "`cleverly` omitted-variable bound with the plug-in nu^2"
    ),
    "cleverly-calibration-slope-rule": "`cleverly` calibration-slope rule",
    "cleverly-default-bands": "`cleverly` shipped TMLE() default fit",
    "cleverly-stratified-incremental": "`cleverly` incremental TMLE with baseline strata",
    "npcausal-stratified": "R `npcausal` `ipsi`, once per stratum subset and on every row",
    "cleverly-stratified-msm": "`cleverly` identity-link MSM TMLE with baseline strata",
    "tmle3-msm-stratified": "R `tmle3` `Param_MSM`, once per stratum subset",
    "cleverly-stratified-drtmle": "`cleverly` DR-TMLE with baseline strata",
    "drtmle-r-stratified": "R `drtmle`, once per stratum subset",
    "cleverly-stratified-tmle": "`cleverly` ordinary TMLE with baseline strata",
    "cleverly-mar-drtmle": "`cleverly` randomized missing-outcome DR-TMLE",
    "cleverly-composite-drtmle": "`cleverly` composite-indicator missing-data DR-TMLE",
    "cleverly-mar-natural-course-tmle": "`cleverly` missing-outcome natural-course TMLE",
    "cleverly-mar-attributable-tmle": "`cleverly` missing-outcome attributable-effect TMLE",
    "cleverly-stacked-mar-natural-course-cvtmle": (
        "`cleverly` stacked missing-outcome natural-course CV-TMLE"
    ),
    "cleverly-mar-tmle": "`cleverly` missing-outcome TMLE",
    "cleverly-multi-arm-ctmle-oat": "`cleverly` multi-arm outcome-adaptive C-TMLE",
    "cleverly-multi-arm-ctmle-selector": "`cleverly` multi-arm selector C-TMLE",
    "cleverly-multi-arm-drtmle": "`cleverly` multi-arm DR-TMLE",
    "cleverly-multi-arm-mar-drtmle": "`cleverly` randomized multi-arm missing-outcome DR-TMLE",
    "cleverly-multi-arm-tmle": "`cleverly` ordinary multi-arm TMLE",
    "cleverly-stacked-cvtmle": "`cleverly` stacked CV-TMLE",
    "cleverly-stacked-mar-arm-indexed-cvtmle": (
        "`cleverly` stacked arm-indexed missing-outcome CV-TMLE"
    ),
    "cleverly-weighted-tmle": "`cleverly` weighted point-treatment TMLE",
    "cleverly-weighted-ltmle": "`cleverly` ordinary weighted LTMLE",
    "cleverly-cross-fitted-weighted-ltmle": "`cleverly` cross-fitted weighted LTMLE",
    "cleverly-learned-weighted-tmle": (
        "`cleverly` weighted point-treatment TMLE with learned nuisances"
    ),
    "cleverly-learned-rule-cvtmle": "`cleverly` fold-evaluated CV-TMLE of the learned-rule value",
    "drtmle-r": "R `drtmle`",
    "drtmle-r-mar": "R `drtmle` with a joint treatment-response mechanism",
    "drtmle-r-composite": "R `drtmle` with an NA treatment and a missing outcome",
    "drtmle-r-multi-arm": "R `drtmle` multi-arm extension",
    "drtmle-r-multi-arm-mar": (
        "R `drtmle` at three arms with a joint treatment-response mechanism"
    ),
    "ltmle": "R `ltmle`",
    "ltmle-weighted": "R `ltmle` with observation weights",
    "ltmle projected regimen fits": "projected R `ltmle` regimen fits",
    "lmtp projected regimen fits": "projected R `lmtp` cross-fitted regimen fits",
    "lmtp": "R `lmtp`",
    "lmtp-weighted": "R `lmtp` with observation weights",
    "npcausal": "R `npcausal`",
    "cleverly-ipsi": "`cleverly` `IPSI`, one-step over time",
    "cleverly-ipsi-dropout": "`cleverly` `IPSI` with a censoring mechanism",
    "cleverly-categorical-incremental": ("`cleverly` TMLE and `IPSI` with a categorical odds tilt"),
    "r-ctmle": "R `ctmle`",
    "tlverse-ctmle3-oat": "R `ctmle3`",
    "ctmle3-multi-arm-oat": "R `ctmle3` multi-arm outcome-adaptive TMLE",
    "tmle3": "R `tmle3`",
    "tmle3-cvtmle": "R `tmle3` CV-TMLE",
    "tmle3-multi-arm": "R `tmle3` multi-arm TMLE",
    "tmle3-stratified": "R `tmle3` stratified TMLE, `tmle_stratified`",
    "tmle-r": "R `tmle`",
    "tmle-r-population-mean": "R `tmle` population-mean path",
    "tmle-r-composed-attributable": (
        "R `tmle` population-mean fits for the natural course and each arm, composed by the "
        "delta method from R's own influence curves"
    ),
    "tmle-r-stitched-arm-indexed": (
        "R `tmle` two-arm path, or its population-mean path once per arm"
    ),
    "tmle-r-cde": "R `tmle` controlled direct-effect path",
    "tmle-r-weighted": "R `tmle` with observation weights",
    "tmle-r-learned-weighted": "R `tmle` with learned weighted nuisances",
    "cleverly-sdr": "`cleverly` sequentially doubly robust (SDR) longitudinal estimator",
    "zepid-single-crossfit-tmle": "Python `zEpid` single-crossfit TMLE",
}


SCENARIOS: dict[str, str] = {
    "strata": (
        "two arms, three baseline strata, an instrument, complete data, correct outcome regression"
    ),
    "missing_outcome": (
        "two arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression"
    ),
    "three_arm_missing": (
        "three arms, outcomes missing at random given W1, W3 and the arm, correct outcome regression"
    ),
    "missing_treatment": (
        "two arms, treatment unrecorded at random given W3 and the arm, outcomes missing at random"
    ),
    "point_in_sample": "three-level treatment, four baseline values, nuisances fitted in sample",
    "point_crossfit": "three-level treatment, four baseline values, five folds",
    "longitudinal_dropout_folds_5": (
        "two treatment nodes with three levels each, dropout after each node, five folds"
    ),
    "longitudinal_folds_5": ("two treatment nodes with three levels each, no dropout, five folds"),
    "ipsi_folds_2": "three-node law with a continuous outcome, two folds",
    "ipsi_folds_5": "three-node law with a continuous outcome, five folds",
    "repeated_folds_5": (
        "three-time-point law with a repeated binary outcome and monotone censoring, five folds"
    ),
    "repeated_folds_1": (
        "three-time-point law with a repeated binary outcome and monotone censoring, in sample"
    ),
    "dropout_folds_1": "three-node law with dropout after each arm, one fold",
    "dropout_folds_5": "three-node law with dropout after each arm, five folds",
    "survival_folds_1": "three-node survival law with dropout, horizons 1 to 3, one fold",
    "survival_folds_5": "three-node survival law with dropout, horizons 1 to 3, five folds",
    "binary_active": (
        "two arms, an arm-specific outcome index, an instrument, correct outcome regression"
    ),
    "three_arm_active": (
        "three arms, an arm-specific outcome index, an instrument, correct outcome regression"
    ),
    "binary_q_correct": ("randomized by W2 with a known mechanism, correct outcome regression"),
    "binary_q_wrong": ("randomized by W2 with a known mechanism, outcome regression without W2"),
    "guard_none": "known mechanism, outcome regression without W2, no DR-TMLE guard",
    "guard_q": "known mechanism, outcome regression without W2, guard Q",
    "guard_g": "known mechanism, outcome regression without W2, guard g",
    "guard_qg": "known mechanism, outcome regression without W2, guards Q and g",
    "q_wrong_g_known": (
        "law L, declared mechanism, outcome regression on A and W1 (wrong), n = 4,000"
    ),
    "three_arm_guard_none": (
        "three-arm law with a known mechanism, outcome regression without W2, no DR-TMLE guard"
    ),
    "smart_q_correct": "two-node SMART with known factors, correct outcome regressions",
    "smart_q_wrong": "two-node SMART with known factors, outcome regressions without L1",
    "binary": "binary-outcome law",
    "binary_lasso": "binary-outcome law, lasso candidate sequence",
    "binary_learners": "binary-outcome law, four declared lasso learners",
    "binary_declared_pair": "binary-outcome law, both sides at one declared pair of penalties",
    "binary_pipeline": "binary-outcome law, each side's own penalty cross-validation",
    "binary_constant_q": (
        "binary-outcome law, one declared pair of penalties, a constant outcome regression"
    ),
    "survival": "five visits, a binary baseline treatment held over every node, visit dropout",
    "competing": "four visits, two competing causes, a held binary treatment, visit dropout",
    "three_arm": "three visits, three arms held over every node, a binary L_t at each later visit",
    "policy": "five visits, a held binary treatment, a known stochastic policy at baseline",
    "mtp": "five visits, a held dose on 0 to 5, lmtp's minus-one policy at baseline",
    "clustered_end_of_study": (
        "100 clusters of 40 rows, a mean-zero cluster component on the final outcome, censored "
        "two-node panel"
    ),
    "few_equal_j20": (
        "20 clusters of 40 rows, a scaled mean-zero cluster component on the final outcome"
    ),
    "unequal_noninformative": (
        "200 clusters of size uniform on 2 to 18, the size absent from the outcome"
    ),
    "unequal_informative": (
        "200 clusters of size uniform on 2 to 18, the size in the outcome; truth the "
        "row-weighted mean"
    ),
    "unequal_informative_cluster_average": (
        "the informative samples weighted by one over the cluster size; truth the "
        "cluster-average mean"
    ),
    "few_unequal_informative_j20": (
        "20 clusters of size uniform on 2 to 18, the size in the outcome, one treatment node"
    ),
    "stratified_msm_identity": (
        "baseline-strata law with a bounded outcome Y ~ Beta(24 Q, 24 (1 - Q))"
    ),
    "stratified_both_correct": (
        "Benkeser et al. (2017) binary law with three baseline strata, correct GLM nuisances"
    ),
    "stratified_incremental": (
        "binary law with three unequal baseline strata, three incremental odds multipliers"
    ),
    "end_of_study_ratio": "two-node end-of-study law, saturated cell-mean learners",
    "linear": "linear Gaussian-outcome law with a constant effect, `make_linear_ate`",
    **{
        f"{scenario}_{token}": law
        for scenario, law in _PLUGIN_LAWS.items()
        for token in _PLUGIN_PARAMETERS[scenario]
    },
    "calibrated_weak": "known propensity, logit g0 = 0.15 W1",
    # The four learned-rule laws of RM30, by their blip b(W) on the logit scale.
    "non_exceptional": "learned-rule law with the blip 0.1 + W1, correctly specified learners",
    "misspecified_limit": (
        "learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner"
    ),
    "exceptional": "learned-rule law with the blip 0, no treatment effect for any unit",
    "weak_blip": "learned-rule law with the weak blip 0.15 W1",
    "calibrated_strong": "known propensity, logit g0 = W1 - 0.5 W2",
    "binary_discrete": "binary-outcome law, discrete selector",
    "binary_greedy": "binary-outcome law, greedy selector",
    "binary_ordered": "binary-outcome law, ordered selector",
    "censored_end_of_study": "two-time-point law with monotone censoring",
    "selected_censored_end_of_study": (
        "selected two-time-point law with monotone censoring and fixed observation weights"
    ),
    "categorical_end_of_study": "two-time-point law with three treatment levels at both nodes",
    "mtp_continuous": "two-node law with a truncated-normal dose at both nodes",
    "policy_point": "one truncated-normal dose on [0, 6] with an outcome quadratic in the dose",
    "point_vector_mixed": "a point treatment of one binary flag and one truncated-normal dose",
    "point_vector_categorical": "a point treatment of two binary components",
    "longitudinal_vector_mixed": "two nodes, each a binary flag and a truncated-normal dose",
    "longitudinal_vector_ordered_classifier": (
        "one node of a binary flag and two ordered doses, its ratio a fitted stacked classifier"
    ),
    "mtp_continuous_crossfit": ("the two-node continuous-dose law, cross-fitted over five folds"),
    "boundaries_ltmle": (
        "two-node law with a truncated-normal dose on a history-dependent support, LTMLE"
    ),
    "boundaries_sdr": (
        "two-node law with a truncated-normal dose on a history-dependent support, SDR"
    ),
    "atom_ltmle": "the same law with a declared atom at the second dose, LTMLE",
    "atom_sdr": "the same law with a declared atom at the second dose, SDR",
    **{
        f"pricing_{estimator}_tau{t}": (
            f"{t}-node pricing law with engagement feedback on the declared price support, "
            f"{estimator.upper()}"
        )
        for t in (2, 3, 5)
        for estimator in ("ltmle", "sdr")
    },
    "population_quantile_ltmle": "two-node law Q with a dose-quantile cap, LTMLE",
    "population_quantile_sdr": "two-node law Q with a dose-quantile cap, SDR",
    "mtp_categorical": "two-node law with six integer treatment levels at both nodes",
    "rr_tilt": "two-node law with a binary treatment at both nodes",
    "categorical_policy_end_of_study": (
        "the three-level law with known stochastic policies at both nodes"
    ),
    "censored_survival_curve": "two-time-point absorbing-event law with monotone censoring",
    "censored_repeated_outcome": (
        "three-time-point law with a three-valued baseline covariate, a repeated binary outcome "
        "that reads the two previous nodes, and monotone censoring"
    ),
    "censored_competing_risk_curve": (
        "two-time-point, two-cause competing-risk law with monotone censoring"
    ),
    "cross_fitted_regimen_projection": (
        "two-time-point law with monotone censoring and four projected treatment plans, "
        "fitted with five outer folds"
    ),
    "censored_regimen_projection": (
        "two-time-point law with monotone censoring and four projected treatment plans"
    ),
    "continuous": "bounded continuous-outcome law with effect modification",
    "clustered_binary": (
        "binary-outcome law with ten rows per cluster and shared effect modification"
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
    "binary_mar_natural_course": "binary-outcome observational natural-course law with MAR outcomes",
    "binary_mar_natural_course_flexible": (
        "binary-outcome observational natural-course law with learned MAR nuisances"
    ),
    "binary_mar_randomized": "binary-outcome randomized law with MAR outcomes",
    "binary_observational_mar": ("L1: two-arm binary-outcome observational law with MAR outcomes"),
    "binary_mar_outcome_and_treatment": (
        "L1: two-arm binary-outcome observational law with a MAR outcome and a MAR treatment "
        "whose recording depends on the treatment"
    ),
    "three_arm_mar_outcome_and_treatment": (
        "L3: three-arm binary-outcome observational law with a MAR outcome and a MAR "
        "treatment whose recording depends on the treatment"
    ),
    "three_arm_mar_randomized": (
        "L3: three-arm binary-outcome law with MAR outcomes and known W-stratified assignment"
    ),
    "binary_mar_two_arm_stacked": "L1: two-arm binary-outcome law with MAR outcomes",
    "binary_mar_three_arm_stacked": "L3: three-arm binary-outcome law with MAR outcomes",
    "continuous_mar_two_arm_stacked": (
        "L2: two-arm bounded continuous-outcome law with MAR outcomes"
    ),
    "continuous_mar_three_arm_stacked": (
        "L4: three-arm bounded continuous-outcome law with MAR outcomes"
    ),
    "continuous_mar_natural_course": (
        "bounded continuous-outcome observational natural-course law with MAR outcomes"
    ),
    "binary_mar_attributable": "L1: two-arm binary-outcome law with MAR outcomes, in sample",
    "continuous_mar_attributable": (
        "L1 tables with a bounded continuous Beta outcome and MAR outcomes, in sample"
    ),
    "three_arm_mar_attributable": (
        "L3: three-arm binary-outcome law with MAR outcomes, in sample, reference arm low"
    ),
    "binary_mar_attributable_cvtmle": (
        "L1: two-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five trees"
    ),
    "three_arm_mar_attributable_cvtmle": (
        "L3: three-arm binary-outcome law with MAR outcomes, stacked CV-TMLE with depth-five "
        "trees, reference arm low"
    ),
    "binary_mar_attributable_weighted": (
        "L1 with fixed analysis weights 0.6, 1.0 and 1.8 by level of W, in sample"
    ),
    "binary_mar_attributable_clustered": (
        "L1 rows in 100 clusters of 20 that share W, in sample with id="
    ),
    "continuous_modified_policy": "continuous-dose law with uncapped and capped shifts",
    "continuous_selected_weighted_nuisances": (
        "continuous-outcome law selected by a covariate-dependent density"
    ),
    "multi_arm_binary": "three-arm binary-outcome law",
    "stratified_binary": "binary-outcome law with three unequal baseline strata",
    "default_fit": "binary-outcome law, fitted with the shipped TMLE() defaults",
    "multi_arm_binary_drtmle": "three-arm binary-outcome law with shared cross-fitted nuisances",
    "multi_arm_binary_oat": "three-arm binary-outcome law, outcome-adaptive selector",
    "multi_arm_selector_discrete": "three-arm binary-outcome law, discrete selector",
    "multi_arm_selector_greedy": "three-arm binary-outcome law, greedy selector",
    "multi_arm_selector_ordered": "three-arm binary-outcome law, ordered selector",
    "sdr_end_of_study": "two-node law with monotone censoring, three plans, one fold",
    "sdr_end_of_study_crossfit": "the same law and plans, five folds",
    "sdr_survival": "censored survival law, risks at horizons 1 and 2, one fold",
    "sdr_mtp": "two-node law with a truncated-normal dose at both nodes and four plans",
    "sdr_policy": "the three-level law with the plans low, mix and taper",
}


ESTIMANDS: dict[str, str] = {
    "propensity_calibration_slope": (
        "calibration slope of the propensity report, one intercept per fold; truth 1"
    ),
    "atc": "average effect on the untreated",
    "atc_lower": "lower omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.3, rho 1",
    "atc_upper": "upper omitted-variable bound on the ATC, at cf_y 0.5, cf_d 0.3, rho 1",
    "ate": "average treatment effect",
    "ate_lower": "lower omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.3, rho 1",
    "ate_upper": "upper omitted-variable bound on the ATE, at cf_y 0.5, cf_d 0.3, rho 1",
    "att": "average effect on the treated",
    "att_lower": "lower omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.3, rho 1",
    "att_upper": "upper omitted-variable bound on the ATT, at cf_y 0.5, cf_d 0.3, rho 1",
    **{
        f"plugin_{token}_{row}": f"{row_words} {_PLUGIN_TOKENS[token]}, at cf_y 0.5, cf_d 0.5, rho 1"
        for token in _PLUGIN_TOKENS
        for row, row_words in _PLUGIN_ROWS.items()
    },
    "ey0": "counterfactual mean under no treatment",
    "tmle_ate": "average treatment effect of the composite TMLE, paired with R `drtmle`'s out$tmle",
    "tmle_ate_mid": (
        "difference in counterfactual means, mid versus high, of the composite TMLE, paired "
        "with R `drtmle`'s out$tmle"
    ),
    "ey1": "counterfactual mean under treatment",
    "ey_obs": "observed outcome mean under the natural course",
    "or": "marginal odds ratio, reported on the log scale",
    "paf": "population attributable fraction",
    "par": "population attributable risk",
    "rr": "marginal risk ratio, reported on the log scale",
}

#: The plans of the repeated-outcome study, which has three treatment times rather than two.
REPEATED_REGIMENS: dict[str, str] = {
    "never": "treat at no time",
    "always": "treat at every time",
    "treat then treat after an outcome": (
        "treat at the first time, then treat at each later time after an outcome of one"
    ),
    "odds x0.5": "multiply the treatment odds by 0.5",
    "odds x2": "multiply the treatment odds by two",
    "natural course": "leave the observed treatment mechanism unchanged",
}

#: The bracketed half of a longitudinal estimand key.
REGIMENS: dict[str, str] = {
    "arm0": "assign arm 0 at baseline and hold it",
    "arm1": "assign arm 1 at baseline and hold it",
    "arm2": "assign arm 2 at baseline and hold it",
    "baseline policy": "draw the baseline arm from a known policy, then hold it",
    "baseline minus one": (
        "lower the baseline dose by one wherever it stays at least one, then hold it"
    ),
    "always": "treat at both times",
    "mix": "draw each arm from a known policy at both times",
    "natural": "leave the observed dose unchanged at both times",
    "up": "add 0.5 to the dose at both times, capped at 5.5",
    "scale at 2": "multiply the second dose by 1.25, capped at 5.5",
    "up then history": (
        "add 0.5 to the first dose, then add 0.5 to the second where the first exceeds 3"
    ),
    "minus one": "lower the level by one at both times wherever it stays at least one",
    "gated": "lower the second level by one where L2 equals one",
    "rr 0.25": "keep each treated unit treated with probability 0.25 at both times",
    "random at 2": "add 0.5 to the second dose with probability 0.5",
    "vector": "set the second component to zero where both components are one",
    "mtp": "risk-ratio tilt 0.5 first, then treat where L2 equals one",
    "taper": "assign the high arm first, then draw from a known policy",
    "continue_if_l2": "treat first, then continue if L2 equals one",
    "high": "assign the high arm at both times",
    "low": "assign the low arm at both times",
    "never": "treat at neither time",
    "respond": "assign standard first, then high if L2 equals one and low otherwise",
    "standard": "assign the standard arm at both times",
    "step_down": "assign high first, then standard",
    "treat then continue if l2 positive": "treat, then continue only if L2 is positive",
    "treat then treat after an outcome": (
        "treat first, then treat at each later time only if the previous outcome was one"
    ),
    "+0.25": "shift dose by 0.25",
    "+0.5 capped": "shift dose by 0.5 subject to the declared cap",
    "natural course": "leave the observed treatment mechanism unchanged",
    "x1.25": "multiply the dose by 1.25, holding doses whose product exceeds 5.5",
    "piecewise": "lower the dose by 0.5 above 3, and leave it unchanged at or below 3",
    "halve below 3": "move a dose at or below 3 halfway toward 3",
    "odds x0.5": "multiply the treatment odds by 0.5",
    "odds x2": "multiply the treatment odds by two",
    "T_a": "multiply the odds of the second level by 2 and of the third by 0.5, against the first",
    "T_b": "multiply the odds of the second level by 0.5 and of the third by 2, against the first",
    "rule": "follow the covariate-dependent rule",
    "learned rule": "learned on each fold's training rows from its outcome regression",
    "tilt": "draw from the known stochastic tilt",
    "dose up": "add 0.5 to each dose, capped at 5.5",
    "flag scaled": (
        "set each flag to one, and multiply the dose by 1.25 where the natural flag is one"
    ),
    "ordered": (
        "multiply the first dose by 1.25, then add 0.5 to the second where the first is at most 3"
    ),
    "full": (
        "set the flag to one where the first dose is at least 3, scale the first dose where the "
        "flag is one, and move the second dose as in ordered"
    ),
    "crossed": (
        "set the flag to one where the first dose is at least 3, and multiply the first dose by "
        "1.25, capped at 5.5, where the flag is zero"
    ),
    "a2 off": "set the second component to zero where the first is one",
    "a1 on": "set the first component to one",
    "p1": "multiply the dose by 0.9 at both times",
    "p2": "add 1 to the dose at both times, holding doses within 1 of the support's upper end",
    "p3": (
        "map the dose a to u - D(u - a) at both times, D a smooth warp whose slope runs from "
        "0.4 to 1.6, u the support's upper end"
    ),
    "p4": "add 1 to a dose below 3 and subtract 1.5 from a dose at or above 3, at both times",
    "p5": "set the second dose to the larger of 3 and 0.65 times the dose",
    "natural_prices": "leave every observed price unchanged",
    "price_up": (
        "multiply each price by 1.12, holding a price within 12% of the support's upper end"
    ),
    "price_down": (
        "multiply each price by 0.85, holding a price within 15% of the support's lower end"
    ),
    "tuned_cap": "add 1 to the dose at both times where the result stays below the tuned cap",
    "tuned_cap, fold average": (
        "add 1 to the dose at both times below each fold's tuned cap, averaged over the folds"
    ),
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
    "rr_regimen": "risk ratio, with inference on the log scale, between the plans",
    "or_regimen": "odds ratio, with inference on the log scale, between the plans",
    "ate_ipsi": "difference in means under the incremental interventions",
    "ate_regime": "difference in means under the regimes",
    "ate_policy": "difference in means under the modified treatment policies",
    "ey_ipsi": "mean under the incremental intervention",
    "ey_learned_rule": "fold average of the value of the rule",
    "ey_regime": "mean under the regime",
    "ey_policy": "mean under the modified treatment policy",
}


PROPERTIES: dict[str, str] = {
    "one_wrong_inference": (
        "with one nuisance inconsistent, the DR-TMLE interval is valid for a declared "
        "mechanism and conservative for a correct parametric mechanism, and a constructed "
        "level-set-breaking error produces a bias that grows with n"
    ),
    "one_wrong_reported": (
        "the arms of the first-order controls and the law-M contrast, reported with no verdict"
    ),
    "known_mechanism_accuracy": (
        "a fit that divides by the declared mechanism is accurate and its interval covers, "
        "whatever the outcome regression"
    ),
    "bootstrap_coverage": (
        "the percentile interval of the full-refit bootstrap covers, each replicate reading "
        "the declared values of its own rows"
    ),
    "variance_direction": (
        "the spread of the ATE when the mechanism is estimated by a correct parametric model, "
        "against its spread when it is declared"
    ),
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
    "in_sample_agreement": (
        "how far the cross-fitted coefficient sits from the in-sample coefficient on the same "
        "draw, in cross-fitted standard errors, reported rather than gated"
    ),
    "crossfit_overfitting": (
        "cross-fitting removes the optimism a flexible learner puts into an in-sample fit"
    ),
    "cluster_aggregation_rule": (
        "the same curve aggregated by cluster sums and by cluster means, reported rather than "
        "gated, because only the cluster sum is the variance of the row-weighted mean at "
        "unequal sizes"
    ),
    "estimand_weighting": (
        "the unweighted fit targets the row-weighted mean, and weights of one over the cluster "
        "size target the cluster-average mean"
    ),
    "few_cluster_reference": (
        "measures the coverage of the t reference that clustered fits take below 40 clusters, "
        "with parametric nuisance learners, reported rather than gated"
    ),
    "corrected_mar_inference": (
        "randomized missing-outcome DR-TMLE keeps bias and coverage within the margins when "
        "one nuisance is wrong and the other is the exact law table"
    ),
    "correction_necessity": (
        "the correction cycle materially reduces the empirical correction scores"
    ),
    "ordinary_targeting": (
        "the composite TMLE, with no extra equation, is unbiased and its interval calibrated "
        "when every nuisance is correct"
    ),
    "treatment_complete_case": (
        "dropping the rows with an unrecorded treatment targets another parameter when the "
        "recording depends on the treatment"
    ),
    "double_robust_contraction": (
        "a bias the equivalence margin rejects at one size contracts as the sample grows, "
        "which is what separates a second-order remainder from an inconsistent estimator; "
        "each rung's coverage claim is read at the rung's declared replication budget, and "
        "any draws beyond that budget serve the fitted slope alone"
    ),
    "double_robustness": (
        "the estimator stays consistent when either the outcome regression or the treatment "
        "mechanism is correct"
    ),
    "fold_locality": (
        "each fold's learned functions are fitted without that fold's rows, directly or through "
        "another fitted model, so no row that evaluates a learned function helped to fit it"
    ),
    "fold_policy": (
        "each split the fit draws is drawn under a named policy, on one law and one set of "
        "draws, and the coverage of each policy is reported rather than gated, because a split "
        "that reads the treatment or the outcome is refused and no cell here establishes that "
        "one is valid"
    ),
    "generated_design": (
        "the outcome-adaptive interval calibrates to its own sampling spread when its regression "
        "is estimated and when the design is known"
    ),
    "density_necessity": "the declared stochastic intervention density determines the target",
    "policy_necessity": "the declared longitudinal policy densities determine the target",
    "inverse_necessity": (
        "the declared inverse of a policy piece enters the density ratio of Equation (3)"
    ),
    "preimage_necessity": (
        "every categorical preimage of a vector value enters the joint induced density"
    ),
    "branch_necessity": (
        "every inverse branch of a policy enters the density ratio of Equation (3)"
    ),
    "classifier_split": (
        "the outer folds of the stacked classifier hold out units, not stacked rows"
    ),
    "atom_necessity": (
        "the mass of every interval a policy sends to a declared atom enters the ratio at the atom"
    ),
    "jacobian_necessity": (
        "the inverse derivative of every non-unit dose map, the D1 scale, enters the joint "
        "induced density"
    ),
    "image_necessity": (
        "a categorical map in the joint induced density reads the natural doses of each preimage"
    ),
    "randomizer_projection": (
        "the integrated policy estimator is the projection of the estimator that records the "
        "randomizer, so its spread is no larger"
    ),
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
    "multi_arm": (
        "the candidate sequence targets a contrast of a three-arm treatment without bias"
    ),
    "power": "the test detects a real effect, so a passing null result cannot come from an inert test",
    "projection_necessity": (
        "the declared projection measure determines the coefficient rather than an implicit "
        "uniform measure"
    ),
    "repeat_stability": (
        "three fold draws reduce the ATE's sensitivity to the realized fold assignment"
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
    "isotonic_projection": (
        "the isotonic projection leaves a regular curve unchanged, and the study records how "
        "often it moves a level"
    ),
    "simultaneous_coverage": (
        "the multiplier band over a law's reported estimands covers the whole truth vector at "
        "the nominal joint rate"
    ),
    "selector_necessity": "the collaborative selector is what produces the result, not the fit around it",
    "stratum_targeting_necessity": (
        "one score block per stratum, rather than one marginal fluctuation, removes the bias "
        "inside each stratum"
    ),
    "competing_risk_recursion_necessity": (
        "the cumulative-incidence recursion uses all-cause survival rather than survival from "
        "the target cause alone"
    ),
    "small_sample_calibration": (
        "the SE ratio and coverage of the in-sample fits at n = 2,000, where few units follow "
        "the rule to the last node, reported rather than gated"
    ),
    "weak_margin_diagnostic": (
        "the coverage and SE ratio of the stratified learned-rule fit and of the unstratified "
        "learned-rule fit on the same draws, on binary laws with a weak margin at the rule "
        "boundary, reported rather than gated"
    ),
    "fold_evaluated_reporting": (
        "the bias of the fold-evaluated report of the learned-rule value beside the stacked "
        "report on the same draws and folds, at 200 clusters whose size rises with the "
        "outcome, reported rather than gated"
    ),
    "target_distinction": (
        "the in-sample learned-rule fit read against the fold average of the rules that ten "
        "training folds of the same draw learn, the target of the cross-fitted fit, with the "
        "mean gap between the two targets, reported rather than gated"
    ),
    "weak_limit_necessity": (
        "the targeted and untargeted in-sample learned-rule fits of an outcome learner whose "
        "limit benefit is near zero, so that its rule flips between samples, with the flip "
        "share and the bias on settled and on flipped draws, reported rather than gated"
    ),
    "sparse_event_diagnostic": (
        "the coverage and SE ratio of the fold-evaluated and the stacked stratified fits on a "
        "binary law with few outcome events in one stratum-arm, reported rather than gated"
    ),
    "small_sample_diagnostic": (
        "the SE ratio, coverage and bias of the declared rule-learner fit at n = 1,000, below "
        "the calibration size, reported rather than gated"
    ),
    "repeated_outcome_necessity": (
        "reading the repeated outcome as an absorbing event targets the cumulative maximum "
        "rather than the mean at the horizon, both arms at n = 8,000"
    ),
    "outcome_history_necessity": (
        "leaving the earlier outcomes out of every history targets a different g-formula, "
        "because the earlier outcomes confound the later nodes"
    ),
    "booster_calibration": (
        "the SE ratio and coverage of the cross-fitted fits with a shallow regularized booster "
        "for both outcome learners, reported rather than gated"
    ),
    "full_history_necessity": (
        "an outcome regression that reads the full history, rather than the previous node "
        "alone, determines the target when the outcome depends on two nodes back"
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
    "warning_rate": (
        "a warning fires on a law where it is false no more often than the type-I ceiling, and "
        "a control rule on the same fits fires more often than that"
    ),
    "weight_necessity": (
        "fixed inverse-selection weights recover the population target from the selected law"
    ),
    "mechanism_term_necessity": (
        "the influence curve includes the term from the estimated treatment mechanism"
    ),
    "single_robustness": (
        "the estimate stays consistent when the treatment mechanism is correct and the "
        "outcome regressions are misspecified linear fits"
    ),
    "censoring_complete_case": (
        "how far a fit to the rows that never drop out sits from the target, reported rather "
        "than gated"
    ),
    "in_sample_fit": (
        "how far a fit with one fold sits from the target on the same histories, reported "
        "rather than gated"
    ),
    "mechanism_targeting": (
        "the softmax update of the treatment mechanism removes the first-order error of a "
        "tilted mechanism, and outcome targeting alone does not"
    ),
    "sequential_double_robustness": (
        "the estimate stays consistent when at each node either the outcome regression or "
        "the treatment and censoring mechanisms are correct"
    ),
    "weighted_calibration": (
        "the interval of a fit with fixed observation weights is calibrated, with no "
        "efficiency claim"
    ),
    "clustered_calibration": (
        "grouped cross-fitting with cluster-level influence curves calibrates inference under "
        "within-cluster dependence"
    ),
    "construction_diagnostic": (
        "how two other cross-fit constructions behave on the overfitting draws, reported "
        "rather than gated"
    ),
    "sdr_tmle_agreement": (
        "how far the SDR estimate sits from the LTMLE estimate on the same draw, in SDR "
        "standard errors, reported rather than gated"
    ),
}


#: ``(family, cell)`` after any arm prefix is stripped, to ``(what was tested, what must hold)``.
CELLS: dict[tuple[str, str], tuple[str, str]] = {
    ("known_mechanism_accuracy", "known_g__q_correct"): (
        "declared mechanism, correct outcome regression",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "known_g__q_wrong"): (
        "declared mechanism, outcome regression without W2",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "estimated_g_wrong__q_wrong"): (
        "intercept-only estimated mechanism, outcome regression without W2",
        "bias interval lies outside the margin",
    ),
    ("known_mechanism_accuracy", "known_g__q_wrong__cv"): (
        "declared mechanism, outcome regression without W2, stacked CV-TMLE over five folds",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "known_g__incremental"): (
        "declared mechanism, outcome regression without W2, the known stochastic regime of the tilt",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "known_g__multi_arm__q_wrong"): (
        "declared three-arm mechanism, outcome regression without W2",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "drtmle_known__none"): (
        "DR-TMLE with no guard on the declared mechanism, outcome regression without W2",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "drtmle_known__Q"): (
        "DR-TMLE with guard Q on the declared mechanism, outcome regression without W2",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "drtmle_known__g"): (
        "DR-TMLE with guard g on the declared mechanism, outcome regression without W2",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "drtmle_known__Qg"): (
        "DR-TMLE with guards Q and g on the declared mechanism, outcome regression without W2",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "ltmle_known__q_wrong"): (
        "LTMLE on a two-node SMART with every factor declared, outcome regressions without L1",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("known_mechanism_accuracy", "ltmle_known_treatment__censoring_estimated"): (
        "LTMLE on a two-node SMART with the treatment declared and the censoring "
        "estimated, outcome regressions without L1",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("bootstrap_coverage", "known_g__q_wrong__percentile"): (
        "declared mechanism, outcome regression without W2, 200-replicate percentile interval",
        "coverage interval clears the floor",
    ),
    ("interval_calibration", "known_g_q_wrong"): (
        "declared mechanism, outcome regression without W2",
        "SE-ratio and coverage intervals inside the calibration bands",
    ),
    ("variance_direction", "known_mechanism"): (
        "declared mechanism, outcome regression without W2",
        "reported, no verdict",
    ),
    ("variance_direction", "parametric_mechanism"): (
        "mechanism estimated by a logistic regression on W2, outcome regression without W2",
        "reported, no verdict",
    ),
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
    ("clustered_inference", "cluster_robust_fold_evaluated"): (
        "fold-evaluated CV-TMLE at unequal cluster sizes, centered cluster variance per fold",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_fold_evaluated"): (
        "the fold-evaluated fit's curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("clustered_inference", "cluster_robust_drtmle"): (
        "cross-fitted DR-TMLE at unequal cluster sizes with cluster-robust inference",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_drtmle"): (
        "the DR-TMLE curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("clustered_inference", "cluster_robust_static"): (
        "cross-fitted clustered LTMLE, ate_regimen[always vs never], 100 clusters of 40 rows",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_static"): (
        "the static LTMLE contrast's curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("clustered_inference", "cluster_robust_dynamic"): (
        "cross-fitted clustered LTMLE, the dynamic rule against never, on the primary fits",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_dynamic"): (
        "the dynamic contrast's curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("clustered_inference", "cluster_robust_unequal"): (
        "cross-fitted clustered LTMLE at cluster sizes uniform on 10 to 70",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_unequal"): (
        "the unequal-size contrast's curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("clustered_inference", "cluster_robust_survival"): (
        "cross-fitted clustered survival LTMLE, the risk contrast at t=2",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_survival"): (
        "the survival contrast's curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("clustered_inference", "cluster_robust_fold_evaluated_cluster_covariate"): (
        "fold-evaluated CV-TMLE of the treated mean, 40 clusters in 10 folds, a covariate "
        "shared within a cluster",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    ),
    ("clustered_inference", "iid_control_fold_evaluated_cluster_covariate"): (
        "the same fold-evaluated curve treated as independent rows",
        "the SE-ratio upper endpoint must not exceed the declared IID-control ceiling",
    ),
    ("estimand_weighting", "individual_average"): (
        "the unweighted interval against the row-weighted mean",
        "exact coverage lower bound clears the floor",
    ),
    ("estimand_weighting", "cluster_average_truth"): (
        "the unweighted interval against the cluster-average mean",
        "exact coverage upper bound falls below 0.50",
    ),
    ("estimand_weighting", "cluster_average_weights"): (
        "the interval with weights of one over the cluster size against the cluster-average mean",
        "exact coverage lower bound clears the floor",
    ),
    ("estimand_weighting", "individual_average_truth"): (
        "the weighted interval against the row-weighted mean",
        "exact coverage upper bound falls below 0.50",
    ),
    ("cluster_aggregation_rule", "cluster_sum"): (
        "the cluster sum of the curve, as this package and R ltmle aggregate it",
        "reported only",
    ),
    ("cluster_aggregation_rule", "cluster_mean"): (
        "the cluster mean of the curve, as R tmle 2.1.1 and tmle3 aggregate it",
        "reported only",
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
    ("corrected_mar_inference", "mechanism_drift"): (
        "the outcome regression is correct and the composite mechanism is misspecified",
        "bias interval inside the margin, coverage clears the floor, SE ratio inside the band",
    ),
    ("ordinary_targeting", "binary"): (
        "the composite TMLE on the two-arm law with a MAR outcome and treatment",
        "bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside "
        "(0.80, 1.20)",
    ),
    ("ordinary_targeting", "three_arm"): (
        "the composite TMLE on the three-arm law, mid versus high",
        "bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside "
        "(0.80, 1.20)",
    ),
    ("ordinary_targeting", "binary_regime"): (
        "the composite TMLE of a known W-dependent regime mean on the two-arm law with a MAR "
        "outcome and treatment",
        "bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside "
        "(0.80, 1.20)",
    ),
    ("ordinary_targeting", "binary_msm"): (
        "the composite TMLE of the linear arm MSM slope on the two-arm law with a MAR outcome "
        "and treatment",
        "bias interval inside the margin, coverage lower bound at least 0.90, SE ratio inside "
        "(0.80, 1.20)",
    ),
    ("treatment_complete_case", "drop_unrecorded__control"): (
        "the rows with an unrecorded treatment dropped, and the rest fitted with delta= alone",
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
    # The fold policies. Each "what must hold" cell says the same thing on purpose:
    # nothing has to hold. A reported row that borrowed a gated row's language would read
    # as a claim, which is the one reading these cells must not carry.
    #
    # The map is keyed globally by (property, cell), so one entry describes the arm in
    # every study that runs it. The wording therefore names the split rather than a layer
    # count: CV-TMLE draws one split, DR-TMLE draws one, and selector-based C-TMLE draws
    # three that all read the same policy.
    ("fold_policy", "unstratified"): (
        "each split is drawn without reading the treatment or the outcome",
        "reported, not gated: the coverage interval is published and no margin is applied",
    ),
    ("fold_policy", "treatment_stratified"): (
        "each split is balanced on the treatment, which the package refuses",
        "reported, not gated: the coverage interval and its paired difference from the "
        "unstratified arm are published and no margin is applied",
    ),
    ("fold_policy", "treatment_outcome_stratified"): (
        "each split is balanced on the treatment crossed with the outcome, which the "
        "package refuses",
        "reported, not gated: the coverage interval and its paired difference from the "
        "unstratified arm are published and no margin is applied",
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
    ("crossfit_overfitting", "stacked_mar_natural_course_cvtmle"): (
        "stacked MAR natural-course CV-TMLE with a fully grown outcome tree",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "stacked_arm_indexed_cvtmle"): (
        "stacked arm-indexed MAR CV-TMLE with a fully grown outcome tree",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("crossfit_overfitting", "in_sample_control"): (
        "the same flexible learner fitted in sample, with no cross-fitting",
        "SE ratio must fall below the overfitting ceiling",
    ),
    ("crossfit_overfitting", "cross_fitted_msm"): (
        "five-fold longitudinal MSM projection with fully grown outcome trees",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("in_sample_agreement", "in_sample_agreement"): (
        "the absolute difference from the in-sample coefficient on the same draw, over the "
        "cross-fitted standard error; the summary publishes its mean",
        "none; the row reports the statistic",
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
        "the exact coverage interval clears the declared floor, over the replications this "
        "rung declares its own verdict at",
    ),
    ("double_robust_contraction", "treatment_correct"): (
        "only the treatment mechanism is correctly specified",
        "the exact coverage interval clears the declared floor, over the replications this "
        "rung declares its own verdict at",
    ),
    ("double_robust_contraction", "both_wrong"): (
        "both nuisances are misspecified",
        "the exact coverage interval must fall below the floor, over the replications this "
        "rung declares its own verdict at",
    ),
    ("double_robust_contraction", "rate_outcome_correct"): (
        "log absolute bias regressed on log n across three sizes, outcome regression correct, "
        "over every replication each rung ran",
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
    ("double_robustness", "response_correct"): (
        "only the outcome-observation mechanism is correctly specified",
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
    ("double_robustness", "outcome_wrong"): (
        "the outcome regression is misspecified and the joint density is the oracle, read from "
        "320 oracle bins per dose",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("preimage_necessity", "declared_preimage"): (
        "the joint density sums over every categorical preimage, with the outcome misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("preimage_necessity", "preimage_cut_control"): (
        "the same fit keeps only the preimages in the value's own categorical cell",
        "bias interval must fall entirely outside the margin",
    ),
    ("jacobian_necessity", "declared_jacobian"): (
        "the joint density multiplies each preimage by the inverse derivative of the D1 scale, "
        "the only non-unit Jacobian, with the outcome misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("jacobian_necessity", "jacobian_drop_control"): (
        "the same fit drops the inverse derivative of the D1 scale",
        "bias interval must fall entirely outside the margin",
    ),
    ("image_necessity", "declared_image"): (
        "the joint density evaluates the flag map at each preimage's natural first dose, with "
        "the outcome misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("image_necessity", "shifted_read_control"): (
        "the same fit evaluates the flag map at the value's own first dose",
        "bias interval must fall entirely outside the margin",
    ),
    ("double_robustness", "multi_arm_treatment_correct"): (
        "three arms with a multinomial lasso and a constant outcome regression, on the "
        "contrast of arm 1.0 against arm 0.0",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "missing_outcome_treatment_correct"): (
        "outcomes missing at random with a correct response learner, and an outcome GLM "
        "that omits a confounder",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "cross_fitted_treatment_correct"): (
        "five cross-fitting folds, a consistent lasso mechanism, and an outcome GLM that "
        "omits a confounder",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "both_wrong_lasso"): (
        "the lasso sequence with a constant outcome regression, so both nuisances are misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("double_robustness", "both_wrong_learners"): (
        "the declared learners with a constant outcome regression, so both nuisances are "
        "misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("double_robustness", "lasso_outcome_correct"): (
        "only the outcome regression is correct, and no lasso candidate is a correct mechanism",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "lasso_treatment_correct"): (
        "only the treatment mechanism is correct, and the lasso path contains it",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "learners_outcome_correct"): (
        "only the outcome regression is correct, and no declared learner is a correct mechanism",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "learners_treatment_correct"): (
        "only the treatment mechanism is correct, and the terminal learner specifies it",
        "bias interval inside the equivalence margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("generated_design", "oracle_design"): (
        "the outcome-adaptive design is supplied rather than estimated",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("generated_design", "estimated"): (
        "the same design is estimated from the data, as a real fit does",
        "SE ratio and coverage intervals both inside their calibration bands; the paired "
        "deficit is reported only",
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
    ("mar_robustness", "only_outcome_wrong"): (
        "only the outcome regression is wrong",
        "bias interval inside the equivalence margin",
    ),
    ("mar_robustness", "only_treatment_wrong"): (
        "only the treatment mechanism is wrong",
        "bias interval inside the equivalence margin",
    ),
    ("mar_robustness", "only_response_wrong"): (
        "only the observation mechanism is wrong",
        "bias interval inside the equivalence margin",
    ),
    ("mar_robustness", "product_only"): (
        "the outcome regression and observation mechanism are wrong, and the product of the "
        "treatment and observation mechanisms is correct at the reference arm",
        "bias interval must fall entirely outside the margin",
    ),
    ("mar_robustness", "outcome_and_response_wrong"): (
        "the outcome regression and the observation mechanism are both wrong",
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
    ("interval_calibration", "in_sample"): (
        "the internal lasso is consistent for the main-terms treatment law, and the outcome "
        "regression is a correct GLM",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "cross_fitted"): (
        "the same law and nuisances, cross-fitted over five folds",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "multi_arm"): (
        "three arms with a multinomial lasso, on the contrast of arm 1.0 against arm 0.0",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "missing_outcome"): (
        "outcomes are missing at random, with a correct response learner",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "flexible_learning"): (
        "separate depth-five trees fit the outcome and response nuisances out of fold; on this "
        "six-cell law each tree fits the saturated, correct model",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "treatment_correct"): (
        "the randomized treatment mechanism is correct while the outcome regression omits effect modification",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "learned_nuisances"): (
        "separate depth-five trees fit the outcome, treatment and observation nuisances out of "
        "fold; each law has nine or fewer covariate cells, so each tree can fit the saturated model",
        "SE ratio and coverage intervals both inside their calibration bands",
    ),
    ("interval_calibration", "shrunken_se_control"): (
        "the reported standard errors are multiplied by a declared factor below one",
        "the SE-ratio interval must fall below the calibration band",
    ),
    ("interval_calibration", "inflated_se_control"): (
        "the reported standard errors come from a curve that is too wide on this law",
        "the SE-ratio interval must fall above the calibration band",
    ),
    ("interval_calibration", "noise_control"): (
        "one efficiency-bound unit of independent noise is added to each estimate",
        "the empirical efficiency ratio must rise above the band",
    ),
    ("warning_rate", "rule"): (
        "the calibration-slope rule, which warns when the Bonferroni interval over the tested "
        "models lies above 0 and excludes 1",
        "rejection upper bound at or below the type-I ceiling",
    ),
    ("warning_rate", "fixed_band"): (
        "the band before RM15 on the same fits, which warns when the pooled one-intercept slope "
        "of the propensity or the outcome lies outside 0.7 to 1.4",
        "rejection lower bound above the type-I ceiling",
    ),
    ("power", "rule"): (
        "the calibration-slope rule, on a learner whose limit slope is not 1",
        "rejection lower bound clears the minimum power",
    ),
    ("power", "randomized_alternative"): (
        "the same test applied to the randomized law's own nonzero contrast",
        "rejection lower bound clears the minimum power",
    ),
    ("power", "ate"): (
        "the same test applied to the binary missing-treatment law's ATE of 0.21, at n = 1,000",
        "rejection lower bound clears the minimum power",
    ),
    ("power", "alternative"): (
        "the same test applied to a law with a real effect",
        "rejection lower bound clears the minimum power",
    ),
    ("isotonic_projection", "moved"): (
        "per replication, the distance the projection moved the level, against a truth of zero",
        "none. The row is a diagnostic and states no verdict",
    ),
    ("simultaneous_coverage", "simultaneous_band"): (
        "the max-t multiplier band over every estimand the law reports, from the same-row "
        "centered influence curves",
        "joint coverage interval inside the calibration coverage band",
    ),
    ("simultaneous_coverage", "pointwise_joint_control"): (
        "the pointwise 95% intervals of the same fits, read jointly",
        "joint coverage upper endpoint must fall below the nominal rate",
    ),
    ("repeat_stability", "three_repeats"): (
        "the median ATE over three fold draws across labelled base seeds",
        "the paired spread-ratio upper endpoint must fall below the declared boundary",
    ),
    ("repeat_stability", "one_repeat"): (
        "the ATE from each repeated fit's first actual fold draw, alone on the same fixed "
        "sample; the base of both spread ratios",
        "the pair's verdict: the spread of the median over this spread stays under the "
        "declared boundary",
    ),
    ("repeat_stability", "second_draw_control"): (
        "the ATE from each repeated fit's second actual fold draw, alone",
        "the spread of an independent single draw over the first draw's spread must not fall "
        "below the declared boundary (no reduction exists to read)",
    ),
    ("projection_necessity", "declared_weights"): (
        "the working model uses its declared nonuniform projection weights",
        "bias interval inside the equivalence margin",
    ),
    ("projection_necessity", "uniform_weights"): (
        "the identical working model is projected under uniform weights",
        "bias interval must fall entirely outside the margin",
    ),
    ("projection_necessity", "stratified"): (
        "the stratified logit-MSM fit with an outcome regression that omits the stratum, so "
        "the per-stratum score blocks do all the adjusting",
        "bias interval inside the equivalence margin",
    ),
    ("projection_necessity", "marginal_fluctuation"): (
        "the same outcome regression with the marginal MSM fluctuation, its targeted "
        "regression projected on the stratum's rows",
        "bias interval must fall entirely outside the margin",
    ),
    ("interval_calibration", "estimated_mechanisms"): (
        "n = 8,000, in sample, main-terms logistic mechanisms estimated, quasibinomial GLM outcome "
        "regressions, every horizon",
        "SE-ratio and coverage intervals inside the calibration bands",
    ),
    ("interval_calibration", "crossfit_glm"): (
        "n = 8,000, five folds, main-terms logistic mechanisms estimated, quasibinomial GLM outcome "
        "regressions",
        "SE-ratio and coverage intervals inside the calibration bands",
    ),
    ("interval_calibration", "crossfit_boosted"): (
        "n = 8,000, five folds, main-terms logistic mechanisms estimated, a depth-2 booster of 50 rounds "
        "with an L2 penalty for both outcome learners",
        "SE-ratio and coverage intervals inside the calibration bands",
    ),
    ("interval_calibration", "continuous_gaussian"): (
        "n = 8,000, the three-value outcome law, least-squares outcome regressions, main-terms logistic "
        "mechanisms estimated",
        "SE-ratio and coverage intervals inside the calibration bands",
    ),
    ("small_sample_calibration", "estimated_mechanisms"): (
        "the in-sample binary calibration arm at n = 2,000",
        "reported, no verdict",
    ),
    ("small_sample_calibration", "continuous_gaussian"): (
        "the in-sample continuous calibration arm at n = 2,000",
        "reported, no verdict",
    ),
    ("repeated_outcome_necessity", "repeated"): (
        "the repeated-outcome fit, mechanisms estimated, GLM outcome regressions",
        "bias interval inside the equivalence margin",
    ),
    ("repeated_outcome_necessity", "absorbing_reading"): (
        "an end-of-study fit of the maximum of the outcomes through the horizon, cell means "
        "with a GLM on an unseen cell",
        "bias interval outside the margin, and inside it around the maximum's own truth",
    ),
    ("outcome_history_necessity", "repeated"): (
        "the repeated-outcome fit, mechanisms estimated, GLM outcome regressions",
        "bias interval inside the equivalence margin",
    ),
    ("outcome_history_necessity", "history_free"): (
        "an end-of-study fit with no earlier outcome in any history, cell-mean outcome "
        "regressions: the earlier outcomes are confounders, and the fit leaves them out of "
        "the outcome regression and both mechanisms",
        "bias interval outside the margin, and inside it around the coarse g-formula",
    ),
    ("full_history_necessity", "full_history"): (
        "an outcome regression of cell means on the full history, with intercept-only "
        "mechanisms, so the regression alone carries the estimate",
        "bias interval inside the equivalence margin",
    ),
    ("full_history_necessity", "markov_history"): (
        "the same fit with the outcome regression on the previous node only, "
        "[W, Y_{t-1}, L_t], and the same intercept-only mechanisms",
        "bias interval outside the margin, and inside it around the exact limit of the "
        "previous-node fit",
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
    ("inverse_necessity", "declared_inverse"): (
        "the density ratio reads each piece at its declared inverse",
        "bias interval inside the equivalence margin",
    ),
    ("inverse_necessity", "inverse_dropped_control"): (
        "the same fit reads each moving piece at the dose itself",
        "bias interval must fall entirely outside the margin",
    ),
    ("crossfit_overfitting", "cross_fitted_mtp_ltmle"): (
        "five-fold policy LTMLE with interpolating extra-trees outcome learners",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
    ),
    ("policy_necessity", "declared_policy"): (
        "the recursion integrates over the declared policy at each node",
        "bias interval inside the equivalence margin",
    ),
    ("policy_necessity", "uniform_control"): (
        "the same fit replaces each declared policy with a uniform distribution",
        "bias interval must fall entirely outside the margin",
    ),
    ("randomizer_projection", "integrated"): (
        "the integrated estimator, with the policy-weighted recursion and the ratio q / g",
        "bias interval inside the equivalence margin; the sd ratio bound below one",
    ),
    ("randomizer_projection", "recorded_randomizer"): (
        "the same sample with a recorded uniform randomizer per node and the rule that reads it",
        "bias interval inside the equivalence margin",
    ),
    ("crossfit_overfitting", "cross_fitted_policy_ltmle"): (
        "five-fold policy LTMLE with fully grown outcome trees",
        "SE ratio clears the overfitting floor and stays inside the sanity band",
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
    ("ratio_necessity", "unit_ratio_control"): (
        "the same fit replaces the density ratio with one, with the outcome misspecified",
        "bias interval must fall entirely outside the margin",
    ),
    ("ratio_necessity", "learned_classifier"): (
        "the declared classifier estimates the density ratio, with the outcome misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("double_robustness", "outcome_wrong_learned_classifier"): (
        "the outcome is misspecified and the declared classifier estimates the density ratio",
        "bias interval inside the equivalence margin, with the reported standard error on the "
        "scale of the empirical spread",
    ),
    ("double_robustness", "both_wrong_control"): (
        "the outcome is misspecified and the density ratio is one",
        "bias interval must fall entirely outside the margin, with the reported standard "
        "error still on the scale of the empirical spread",
    ),
    ("classifier_split", "unit_folds"): (
        "each unit's natural and assigned rows share an outer fold, with the outcome misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("classifier_split", "row_folds_control"): (
        "the outer folds are drawn over stacked rows, so a unit's copy trains its own classifier",
        "bias interval must fall entirely outside the margin",
    ),
    ("interval_calibration", "term_dropped_control"): (
        "the reported standard error omits the tuning term of the curve",
        "the SE-ratio interval must fall below the calibration band",
    ),
    ("interval_calibration", "population_truth_control"): (
        "the fold-average rows are scored on the population target",
        "the SE-ratio interval must fall below the calibration band",
    ),
    ("jacobian_necessity", "declared"): (
        "the density ratio reads each piece's inverse slope, with an outcome model blind to "
        "the second dose",
        "bias interval inside the equivalence margin",
    ),
    ("jacobian_necessity", "jacobian_dropped_control"): (
        "the same fit replaces each inverse slope with one",
        "bias interval must fall entirely outside the margin",
    ),
    ("branch_necessity", "declared"): (
        "the density ratio sums over both inverse branches, with the outcome misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("branch_necessity", "one_branch_control"): (
        "the same fit sums over the first branch only",
        "bias interval must fall entirely outside the margin",
    ),
    ("atom_necessity", "declared"): (
        "the ratio at the atom adds the mass of the interval sent to it, with the outcome "
        "misspecified",
        "bias interval inside the equivalence margin",
    ),
    ("atom_necessity", "mass_dropped_control"): (
        "the same fit drops that mass, so the ratio at the atom is one",
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
    ("robustness_contract", "natural_course_insample"): (
        "missing outcomes, in-sample per-arm design, n = 2,250, natural-course estimands",
        "bias interval inside the equivalence margin, SE ratio inside (0.1, 10)",
    ),
    ("robustness_contract", "missing_outcome_crossfit"): (
        "missing outcomes, five-fold cross-fitted per-arm design, n = 2,000",
        "bias interval inside the equivalence margin, SE ratio inside (0.1, 10)",
    ),
    ("robustness_contract", "strata_missing"): (
        "three strata with missing outcomes, cross-fitted per-arm design per stratum, n = 3,000",
        "bias interval inside the equivalence margin, SE ratio inside (0.1, 10)",
    ),
    ("robustness_contract", "weighted_strata_missing"): (
        "the strata law under the declared known weight 0.5 + 1{W1 > 0}, n = 3,000",
        "bias interval inside the equivalence margin, SE ratio inside (0.1, 10)",
    ),
    ("robustness_contract", "strata_missing_outcome_wrong"): (
        "the strata law with the outcome regression without W1, n = 3,000",
        "bias interval must fall entirely outside the margin, SE ratio inside (0.1, 10)",
    ),
    ("root_n_and_efficiency", "n_6000"): (
        "bias, coverage and SE calibration at n = 6,000",
        "",  # the smallest rung's rule depends on its role; :func:`cell` sets it
    ),
    ("root_n_and_efficiency", "n_24000"): (
        "bias, coverage and SE calibration at n = 24,000",
        "",  # the smallest rung's rule depends on its role; :func:`cell` sets it
    ),
    ("root_n_and_efficiency", "n_64000"): (
        "bias, coverage and SE calibration at n = 64,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    ),
    ("root_n_and_efficiency", "n_96000"): (
        "bias, coverage and SE calibration at n = 96,000",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
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
    ("root_n_and_efficiency", "n_16000"): (
        "bias, coverage and SE calibration at n = 16,000",
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
    ("selector_necessity", "lasso_collaborative"): (
        "the selector chooses its own penalty along the lasso sequence",
        "bias interval inside the equivalence margin, and RMSE at most the declared ratio of "
        "the paired control's",
    ),
    ("selector_necessity", "learners_collaborative"): (
        "the selector chooses its own learner along the declared nested sequence",
        "bias interval inside the equivalence margin, a declared expected red, and RMSE at most "
        "the declared ratio of the paired control's",
    ),
    ("multi_arm", "lasso_treatment_correct"): (
        "the lasso sequence on the three-arm law, with a constant outcome regression",
        "bias interval inside the equivalence margin",
    ),
    ("multi_arm", "empty_control"): (
        "the selector is forced to stop at an empty path on the three-arm law",
        "bias interval must fall entirely outside the margin",
    ),
    # "misspecified" rather than "constant", and "fit" rather than "backward recursion":
    # six studies register this family and the description is shared by all of them.  The
    # longitudinal ones fluctuate a sample-prior constant over a two-node recursion; the
    # point-treatment MSM projection fluctuates a deliberately inverted oracle over one node.
    # Naming either study's construction here published a false sentence on the other's page.
    ("fold_locality", "fold_local"): (
        "a random forest learns each fold's rule on the fold's training rows",
        "bias interval inside the equivalence margin",
    ),
    ("fold_locality", "validation_rows"): (
        "the same fit, with each fold's rule learned on the fold's own validation rows",
        "bias interval must fall entirely outside the margin",
    ),
    ("stratum_targeting_necessity", "stratified"): (
        "the stratified fit with an outcome regression that omits the stratum, so the "
        "per-stratum score blocks do all the adjusting",
        "bias interval inside the equivalence margin",
    ),
    ("stratum_targeting_necessity", "marginal_fluctuation"): (
        "the same outcome regression with one marginal fluctuation per arm, averaged inside "
        "each stratum",
        "bias interval must fall entirely outside the margin",
    ),
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
    ("type_i_error", "randomized_sharp_null"): (
        "a randomized law whose two compared arms share one outcome regression, so the "
        "contrast is exactly zero",
        "one-sided rejection bound stays under the declared type-I ceiling, and coverage clears "
        "the floor",
    ),
    ("type_i_error", "target_null"): (
        "a selected law whose weighted target contrast is exactly zero",
        "one-sided rejection bound stays under the declared type-I ceiling",
    ),
}


#: The settings of the DR-TMLE one-wrong witness study, by the stem its cells start with.
#: Mirrored from ``tests.studies.drtmle_one_wrong.SETTINGS``, which this module cannot import;
#: ``tests/unit/test_drtmle_one_wrong_design.py`` checks every published cell is described.
_ONE_WRONG_SETTINGS = {
    "q_wrong_g_glm": "law L, outcome regression on A and W1 (wrong), mechanism a correct "
    "logistic GLM, in sample",
    "q_wrong_g_known": "law L, outcome regression on A and W1 (wrong), declared mechanism, "
    "the draws of q_wrong_g_glm",
    "q_wrong_g_glm_cf": "law L, outcome regression on A and W1 (wrong), mechanism a correct "
    "logistic GLM, cross-fitted over the default folds",
    "q_wrong_g_slow": "law L, outcome regression on A and W1 (wrong), mechanism a correct "
    "GLM plus the constructed logit offset n^(-1/3) W2",
    "q_slow_g_wrong": "law L, outcome regression a correct GLM plus the constructed logit "
    "offset 3 n^(-1/3) (2 W1 + W2) / sqrt(5), mechanism a GLM on W2 (wrong)",
    "mar_q_wrong_pi_slow": "law M, randomized missing outcome with the declared treatment "
    "probability 1/2, outcome regression on A and W1 (wrong), response mechanism a correct "
    "GLM plus the constructed logit offset 2 n^(-1/3) W2",
}
_ONE_WRONG_RULES = {
    "conservative": "SE-ratio interval wholly above the calibration band, coverage clears "
    "the floor, bias inside the margin",
    "valid": "coverage clears the floor, bias inside the margin, SE ratio inside the sanity band",
    "first_order": "bias interval wholly beyond the margin on the predicted negative side, SE "
    "ratio between 0.1 and 10",
    "growth": "the lower bound of the 99% interval of the signed growth of bias/SD from "
    "n = 1,000 to n = 16,000 lies above zero",
}
_ONE_WRONG_ESTIMANDS = {"ate": "ATE", "ey1": "arm-1 mean", "ey0": "arm-0 mean"}
_ONE_WRONG_REPORTED = "reported: bias, SE ratio and coverage, with no verdict"


def _one_wrong(family: str, stem: str, estimand: str, kind: str, size: int = 0) -> None:
    setting = _ONE_WRONG_SETTINGS[stem]
    tested = f"{_ONE_WRONG_ESTIMANDS[estimand]}: {setting}"
    if size:
        stem, tested = f"{stem}_n{size}", f"{tested}, at n = {size:,}"
    rule = _ONE_WRONG_REPORTED if kind == "reported" else _ONE_WRONG_RULES[kind]
    CELLS[family, f"{stem}__{estimand}__{kind}"] = (tested, rule)


for _stem in ("q_wrong_g_glm", "q_wrong_g_known", "q_wrong_g_glm_cf"):
    _one_wrong(
        "one_wrong_inference",
        _stem,
        "ate",
        "valid" if _stem == "q_wrong_g_known" else "conservative",
    )
    for _arm in ("ey1", "ey0"):
        _one_wrong("one_wrong_inference", _stem, _arm, "valid")
for _stem in ("q_wrong_g_slow", "q_slow_g_wrong"):
    _one_wrong("one_wrong_inference", _stem, "ate", "growth")
    for _size in (1_000, 4_000, 16_000):
        _one_wrong("one_wrong_inference", _stem, "ate", "first_order", _size)
        for _arm in ("ey1", "ey0"):
            _one_wrong("one_wrong_reported", _stem, _arm, "reported", _size)
for _arm in ("ey1", "ey0"):
    _one_wrong("one_wrong_inference", "mar_q_wrong_pi_slow", _arm, "first_order")
_one_wrong("one_wrong_reported", "mar_q_wrong_pi_slow", "ate", "reported")


class Undescribed(LookupError):
    """A committed result names a key this module does not describe."""


#: A study-specific reading of a generic cell kind, keyed by family, arm prefix and kind.
#: The generic entry in :data:`CELLS` says what the kind does; this says how one study built it.
ARM_CELLS: dict[tuple[str, str, str], tuple[str, str]] = {
    (family, arm, "inflated_se_control"): (
        "the standard errors come from the curve of nu^2 without the conditioning-share term, "
        "as before RM22",
        "the SE-ratio interval must fall above the calibration band",
    )
    for family in ("interval_calibration",)
    for arm in ("att_lower", "att_upper")
}
ARM_CELLS.update(
    {
        ("stratum_targeting_necessity", f"v{stratum}_ey_x2", "marginal_fluctuation"): (
            "the same outcome regression in the unstratified incremental fit, its targeted "
            "mixture averaged inside the stratum",
            "bias interval must fall entirely outside the margin",
        )
        for stratum in (0, 2)
    }
)
ARM_CELLS.update(
    {
        ("interval_calibration", arm, "inflated_se_control"): (
            "the cross-covariance of the two parent curves is dropped, so the standard error "
            "adds the natural-course and reference-arm variances as if independent",
            "the SE-ratio interval must fall above the calibration band",
        )
        for arm in ("par", "paf")
    }
)
ARM_CELLS.update(
    {
        ("rule_necessity", arm, "reversed_rule"): (
            "the same fit treats after a zero outcome instead of after an outcome of one",
            "bias interval must fall entirely outside the margin, and inside it around the "
            "reversed rule's own truth",
        )
        for arm in ("rule_t2", "rule_t3")
    }
)
ARM_CELLS.update(
    {
        ("double_robustness", label, "both_wrong"): (
            "outcome regression of cell means on the baseline covariate alone, and "
            "intercept-only treatment and censoring mechanisms",
            "bias interval must fall entirely outside the margin, and inside it around the "
            "exact limit of that fit, with the reported standard error still on the scale of "
            "the empirical spread",
        )
        for label in (
            "never_t3",
            "always_t3",
            "rule_t3",
            "always_vs_never_t3",
            "rule_vs_never_t3",
        )
    }
)
ARM_CELLS.update(
    {
        ("targeting_necessity", label, "untargeted"): (
            "the iterated cell means on the baseline covariate among the units that followed "
            "the plan, with no fluctuation step",
            "bias interval must fall entirely outside the margin, and inside it around the "
            "baseline-only plug-in limit",
        )
        for label in (
            "never_t3",
            "always_t3",
            "rule_t3",
            "always_vs_never_t3",
            "rule_vs_never_t3",
        )
    }
)
#: A reading of a label or a cell kind that holds for one study only, keyed by study slug.  A
#: label such as ``rule_t3`` names different estimands in different studies, and the shared
#: tables above cannot say which.
STUDY_ARMS: dict[tuple[str, str], str] = {
    ("point-survival-curve", "rule_t3"): (
        "difference of the rule that treats where W1 is one and arm 0 at visit 3, three arms "
        "without L_t"
    ),
}
STUDY_ARM_CELLS: dict[tuple[str, str, str, str], tuple[str, str]] = {
    ("point-survival-curve", "clustered_inference", label, "cluster_robust"): (
        "the cluster-robust interval from the cluster-level influence curve of a single-fold "
        "hazard fit",
        "SE-ratio and coverage intervals both stay inside their calibration bands",
    )
    for label in ("clustered_t5", "clustered_t5_per_horizon")
}
STUDY_ARM_CELLS.update(
    {
        ("omitted-variable-bound-plugin-se", "interval_calibration", label, kind): words
        for scenario in _PLUGIN_SCENARIOS
        for token in _PLUGIN_PARAMETERS[scenario]
        if not token.startswith("ate")
        for row in _PLUGIN_ROWS
        for label in (f"{scenario}_{token}_{row}",)
        for kind, words in (
            (
                "inflated_se_control",
                (
                    "the standard errors come from the package curve of nu^2 minus the "
                    "mechanism term the study rebuilt, which is too wide at a correct model",
                    "the SE-ratio interval must fall above the calibration band",
                ),
            ),
            (
                "shrunken_se_control",
                (
                    "the standard errors come from the known-mechanism curve plus the "
                    "mechanism term of a study-side logistic fit, which is too narrow for a "
                    "declared mechanism",
                    "the SE-ratio interval must fall below the calibration band",
                ),
            ),
        )
        if (kind == "shrunken_se_control") == (scenario == "binary_known")
    }
)


#: The ``longitudinal-repeated-ipsi`` cells, keyed by label, which only this study publishes.
#: The labels mirror
#: ``tests.studies.canonical_ipsi_repeated.LABELS``, which this module cannot import (descriptions
#: stay out of every study's hashed module list).  ``tests/unit/test_ipsi_repeated_method_study.py``
#: pins the two equal.
_IPSI_LABELS = (
    *(f"ipsi_{delta}_t{horizon}" for delta in (0.5, 1.0, 2.0) for horizon in (1, 2, 3)),
    *(f"ipsi_{delta}_vs_1.0_t{horizon}" for delta in (0.5, 2.0) for horizon in (1, 2, 3)),
)
_IPSI_BAND = "SE-ratio and coverage intervals inside the calibration bands"
_IPSI_FITS: dict[tuple[str, str], tuple[str, str]] = {
    ("interval_calibration", "in_sample"): (
        "in sample, cell-mean outcome regressions with a backoff, main-terms quasibinomial "
        "treatment and censoring mechanisms, n = 8,000",
        _IPSI_BAND,
    ),
    ("interval_calibration", "crossfit"): (
        "five folds, cell-mean outcome regressions with a backoff, main-terms quasibinomial "
        "treatment and censoring mechanisms, n = 8,000",
        _IPSI_BAND,
    ),
    ("interval_calibration", "continuous"): (
        "the three-value outcome law, in sample, cell-mean outcome regressions with a backoff, "
        "main-terms quasibinomial mechanisms, outcome bounds (0, 1), n = 8,000",
        _IPSI_BAND,
    ),
    ("booster_calibration", "crossfit_boosted"): (
        "five folds, a shallow regularized booster for both outcome learners, main-terms "
        "quasibinomial mechanisms, n = 8,000",
        "reported, no verdict",
    ),
    ("small_sample_calibration", "in_sample"): (
        "the in-sample binary calibration arm at n = 2,000",
        "reported, no verdict",
    ),
    ("small_sample_calibration", "continuous"): (
        "the continuous calibration arm at n = 2,000",
        "reported, no verdict",
    ),
    ("single_robustness", "w_only_m"): (
        "the outcome regression reads the baseline covariate and the current arm alone, which is "
        "wrong for the history, with the correctly specified mechanisms, n = 2,000",
        "bias interval inside the equivalence margin",
    ),
    ("single_robustness", "one_step_necessity"): (
        "the tilt plug-in with the same wrong regression and the same fitted treatment "
        "mechanism, written without `cleverly`",
        "bias interval outside the margin, and inside it around the plug-in's exact limit",
    ),
    ("repeated_outcome_necessity", "repeated"): (
        "the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions",
        "bias interval inside the equivalence margin",
    ),
    ("repeated_outcome_necessity", "absorbing_reading"): (
        "an end-of-study fit of the maximum of the outcomes through the horizon, with the earlier "
        "outcomes in the history",
        "bias interval outside the margin, and inside it around the absorbing tilt functional",
    ),
    ("outcome_history_necessity", "repeated"): (
        "the repeated-outcome fit, correctly specified mechanisms, cell-mean outcome regressions",
        "bias interval inside the equivalence margin",
    ),
    ("outcome_history_necessity", "history_free"): (
        "an end-of-study fit with no earlier outcome in any history, cell-mean outcome "
        "regressions and mechanisms on the coarse key",
        "bias interval outside the margin, and inside it around the coarse tilt functional",
    ),
}
for _label in _IPSI_LABELS:
    for (_family, _base), _words in _IPSI_FITS.items():
        ARM_CELLS[(_family, _label, _base)] = _words

#: The ``longitudinal-repeated-sdr`` cells.  The labels are those of the X32a repeated-outcome
#: study, so the shared tables describe a cell by another study's fit.  The entries below say what
#: this study fits.  Mirrored from ``tests.studies.sdr_repeated_properties``, which this module
#: cannot import (descriptions stay out of every study's hashed module list).
#: ``tests/unit/test_sdr_repeated_method_study.py`` pins the labels equal.
_SDR_SLUG = "longitudinal-repeated-sdr"
_SDR_GATED = ("never_t3", "always_t3", "rule_t3", "always_vs_never_t3", "rule_vs_never_t3")
_SDR_BAND = "SE-ratio and coverage intervals inside the calibration bands"
_SDR_MECHANISMS = "main-terms quasibinomial treatment and censoring mechanisms"
_SDR_FITS: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, str]]] = {
    ("interval_calibration", "in_sample"): (
        (
            *_SDR_GATED,
            "never_t1",
            "never_t2",
            "always_t1",
            "always_t2",
            "rule_t2",
            "always_vs_never_t1",
            "always_vs_never_t2",
            "rule_vs_never_t2",
        ),
        (
            f"in sample, cell-mean outcome regressions with a backoff, {_SDR_MECHANISMS}, "
            "n = 32,000",
            _SDR_BAND,
        ),
    ),
    ("interval_calibration", "crossfit"): (
        _SDR_GATED,
        (
            f"five folds, cell-mean outcome regressions with a backoff, {_SDR_MECHANISMS}, "
            "n = 8,000",
            _SDR_BAND,
        ),
    ),
    ("interval_calibration", "continuous"): (
        _SDR_GATED,
        (
            "the three-value outcome law, in sample, cell-mean outcome regressions with a "
            f"backoff, {_SDR_MECHANISMS}, outcome bounds (0, 1), n = 32,000",
            _SDR_BAND,
        ),
    ),
    ("interval_calibration", "crossfit_boosted"): (
        _SDR_GATED,
        (
            "five folds, a shallow regularized booster for both outcome learners (depth 2, 50 "
            f"rounds, an L2 penalty), {_SDR_MECHANISMS}, n = 8,000",
            "the lower ends of the SE-ratio and coverage intervals at or above the lower ends "
            "of the calibration bands",
        ),
    ),
    ("small_sample_calibration", "in_sample"): (
        _SDR_GATED,
        ("the in-sample binary calibration arm at n = 2,000", "reported, no verdict"),
    ),
    ("small_sample_calibration", "continuous"): (
        _SDR_GATED,
        ("the continuous calibration arm at n = 2,000", "reported, no verdict"),
    ),
    ("correction_necessity", "corrected"): (
        _SDR_GATED,
        (
            "the sequential regression on the baseline covariate alone, with the correction "
            "and correctly specified mechanisms",
            "bias interval inside the equivalence margin",
        ),
    ),
    ("correction_necessity", "uncorrected"): (
        _SDR_GATED,
        (
            "the same sequential regression carried back with no correction, written without "
            "`cleverly`",
            "bias interval outside the margin, and inside it around the plug-in's exact limit",
        ),
    ),
    ("repeated_outcome_necessity", "repeated"): (
        ("never_t2", "never_t3", "always_t2", "always_t3", "rule_t2", "rule_t3"),
        (
            f"the repeated-outcome fit, cell-mean outcome regressions with a backoff, "
            f"{_SDR_MECHANISMS}",
            "bias interval inside the equivalence margin",
        ),
    ),
    ("repeated_outcome_necessity", "absorbing_reading"): (
        ("never_t2", "never_t3", "always_t2", "always_t3", "rule_t2", "rule_t3"),
        (
            "an end-of-study fit of the maximum of the outcomes through the horizon, with the "
            "earlier outcomes in the history",
            "bias interval outside the margin, and inside it around the absorbing g-formula",
        ),
    ),
    ("outcome_history_necessity", "repeated"): (
        ("never_t3", "always_t3", "always_vs_never_t3"),
        (
            f"the repeated-outcome fit, cell-mean outcome regressions with a backoff, "
            f"{_SDR_MECHANISMS}",
            "bias interval inside the equivalence margin",
        ),
    ),
}
for (_family, _base), (_cell_labels, _words) in _SDR_FITS.items():
    for _label in _cell_labels:
        STUDY_ARM_CELLS[(_SDR_SLUG, _family, _label, _base)] = _words
STUDY_ARMS[(_SDR_SLUG, "always_t2")] = "mean under always at horizon t = 2, repeated binary outcome"
ARM_CELLS[("interval_calibration", "par", "correctly_specified")] = (
    "the outcome regression, treatment mechanism and observation mechanism are correct",
    "SE ratio and coverage intervals both inside their calibration bands",
)
ARM_CELLS[("interval_calibration", "paf", "correctly_specified")] = ARM_CELLS[
    ("interval_calibration", "par", "correctly_specified")
]
ARM_CELLS[("missingness_necessity", "par", "declared")] = (
    "the observation indicator is declared and every nuisance is correct",
    "bias interval inside the equivalence margin",
)
ARM_CELLS[("targeting_necessity", "par", "targeted")] = (
    "both fluctuations are solved; the outcome regression is wrong and both mechanisms are correct",
    "bias interval inside the equivalence margin",
)
ARM_CELLS[("targeting_necessity", "par", "untargeted")] = (
    "the identical wrong outcome regression is read without either fluctuation",
    "bias interval must fall entirely outside the margin",
)
ARM_CELLS.update(
    {
        ("corrected_mar_inference", arm, "both_wrong"): (
            "the outcome regression and observation mechanism are both misspecified",
            "bias interval must fall entirely outside the margin, and its distance from zero "
            "must clear the declared floor",
        )
        for arm in ("l3_ate_low", "l3_ate_mid")
    }
)
ARM_CELLS.update(
    {
        ("corrected_mar_inference", arm, configuration): (
            {
                "both_correct": "the outcome regression and the composite mechanism are correct",
                "outcome_drift": (
                    "the outcome regression is misspecified and the composite mechanism is correct"
                ),
                "mechanism_drift": (
                    "the outcome regression is correct and the treatment factor and treatment "
                    "observation factor of the composite mechanism are misspecified"
                ),
                "both_wrong": (
                    "the outcome regression and the composite mechanism are both misspecified"
                ),
            }[configuration],
            (
                "bias interval must fall entirely outside the margin, and its distance from "
                "zero must clear the declared floor"
                if configuration == "both_wrong"
                else "bias interval inside the margin, coverage clears the floor, SE ratio "
                "inside the band"
            ),
        )
        for arm in (
            "composite_observational_ate",
            "composite_binary_ate",
            "composite_three_arm_ate_low",
            "composite_three_arm_ate_mid",
        )
        for configuration in ("both_correct", "outcome_drift", "mechanism_drift", "both_wrong")
    }
)
ARM_CELLS[("corrected_mar_inference", "composite_three_arm_ate_low", "both_wrong")] = (
    "the outcome table's arm columns rolled, a uniform treatment factor, and a wrong treatment "
    "observation factor: a drift of this contrast's own",
    "bias interval must fall entirely outside the margin, and its distance from zero must "
    "clear the declared floor",
)
ARM_CELLS.update(
    {
        ("correction_necessity", "composite_cycle", "closed_score"): (
            "the largest equation-(9) and equation-(10) scores over the arms after the cycle, "
            "under the outcome drift",
            "the upper confidence endpoint is below the declared fraction of the initial-score "
            "lower endpoint",
        ),
        ("correction_necessity", "composite_cycle", "initial_score_control"): (
            "the largest equation-(10) score over the arms before the cycle is run",
            "the lower confidence endpoint clears the declared unresolved-score floor",
        ),
    }
)


#: The ``ctmle-oat-per-arm-parametric`` cells (X58a).  A calibration cell reads one estimand of
#: one fit, so the entries compose the label's description with the estimand's.  The label
#: ``binary_active`` is an :data:`ARMS` key, so its cells resolve through :data:`ARM_CELLS`.
_PARAMETRIC_ESTIMANDS = {
    "ey0": "arm-0 mean",
    "ey1": "arm-1 mean",
    "ate": "ATE",
    "log_rr": "log risk ratio",
    "log_or": "log odds ratio",
}
_PARAMETRIC_CALIBRATION = ("SE ratio and coverage intervals both inside their calibration bands",)
_PARAMETRIC_FITS: dict[str, tuple[str, tuple[str, ...]]] = {
    "auto_bounds": (
        'the stacked curve with the default g_bounds of "auto"',
        ("ey0", "ey1", "ate"),
    ),
    "bounded_outcome": (
        "the stacked curve for an outcome bounded in 2 to 10, fitted with q_bounds on the "
        "scaled outcome",
        ("ey0", "ey1", "ate"),
    ),
    "bounds_active": (
        "the stacked curve with g_bounds of (0.05, 0.95), which clip a share of the rows at the "
        "limit",
        ("ey0", "ey1", "ate"),
    ),
    "kappa_cubic": (
        "the stacked curve on a U-shaped mechanism, with the cubic basis in logit q",
        ("ey0", "ey1", "ate"),
    ),
    "kappa_linear": (
        "the stacked curve on a U-shaped mechanism, with the linear basis in logit q",
        ("ey0", "ey1", "ate"),
    ),
    "n_500": ("the stacked curve at n = 500", ("ey0", "ey1", "ate")),
    "n_2000": ("the stacked curve at n = 2,000", ("ey0", "ey1", "ate")),
    "n_8000": ("the stacked curve at n = 8,000", ("ey0", "ey1", "ate")),
    "pseudo_true": (
        "the stacked curve with W1 dropped from the outcome features, so the interval covers "
        "the limit psi*",
        ("ey0", "ey1", "ate"),
    ),
    "shared_prognosis": (
        "the stacked curve on a law whose two arm means share W1 and W2 in their outcome index",
        ("ate", "log_rr", "log_or"),
    ),
}
for _label, (_fit, _estimands) in _PARAMETRIC_FITS.items():
    for _estimand in _estimands:
        CELLS[("interval_calibration", f"{_label}__{_estimand}__corrected")] = (
            f"{_fit}: {_PARAMETRIC_ESTIMANDS[_estimand]}",
            *_PARAMETRIC_CALIBRATION,
        )
for _estimand in ("ey0", "ey1", "ate"):
    ARM_CELLS[("interval_calibration", "binary_active", f"{_estimand}__corrected")] = (
        f"the stacked curve at n = 1,500: {_PARAMETRIC_ESTIMANDS[_estimand]}",
        *_PARAMETRIC_CALIBRATION,
    )
ARM_CELLS.update(
    {
        ("interval_calibration", "binary_active", "ate__shrunken_se_control"): (
            "the ATE standard errors are multiplied by a declared factor below one",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("interval_calibration", "binary_active", "ate__noise_control"): (
            "a declared scale of independent noise is added to each ATE estimate",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("interval_calibration", "binary_active", "ate__shipped_curve__shrunken_se_control"): (
            "the ATE standard errors come from the shipped adaptive-propensity curve of the "
            "same fits, which omits the generated-design term",
            "the SE-ratio interval must fall below the calibration band",
        ),
    }
)
CELLS[
    ("interval_calibration", "shared_prognosis__ate__cross_term_dropped__inflated_se_control")
] = (
    "the ATE standard errors are built from the two arm variances without the cross-arm "
    "covariance, on a law whose arm estimates correlate",
    "the SE-ratio interval must fall above the calibration band",
)

#: The ``ctmle-oat-per-arm-parametric-crossfit`` cells (X58b1).  Each calibration label names a
#: cross-fitted fit of the exact V-fold stack at five folds; a cell reads one estimand of it.
_CROSSFIT_FITS: dict[str, tuple[str, tuple[str, ...]]] = {
    "crossfit_1500": (
        "the cross-fitted stacked curve at n = 1,500, five folds",
        ("ey0", "ey1", "ate", "log_rr", "log_or"),
    ),
    "crossfit_500": (
        "the cross-fitted stacked curve at n = 500, five folds",
        ("ey0", "ey1", "ate"),
    ),
    "crossfit_8000": (
        "the cross-fitted stacked curve at n = 8,000, five folds",
        ("ey0", "ey1", "ate"),
    ),
    "crossfit_pseudo_true": (
        "the cross-fitted stacked curve with W1 dropped from the outcome features, so the "
        "interval covers the limit psi*",
        ("ey0", "ey1", "ate"),
    ),
    "repeats": (
        "the median over three cross-fitting draws, with the split-adjusted variance of the "
        "stacked curve",
        ("ey0", "ey1", "ate"),
    ),
}
for _label, (_fit, _estimands) in _CROSSFIT_FITS.items():
    for _estimand in _estimands:
        CELLS[("interval_calibration", f"{_label}__{_estimand}__corrected")] = (
            f"{_fit}: {_PARAMETRIC_ESTIMANDS[_estimand]}",
            *_PARAMETRIC_CALIBRATION,
        )
    CELLS[("interval_calibration", f"{_label}__ate__shrunken_se_control")] = (
        "the ATE standard errors are multiplied by a declared factor below one",
        "the SE-ratio interval must fall below the calibration band",
    )
    CELLS[("interval_calibration", f"{_label}__ate__noise_control")] = (
        "a declared scale of independent noise is added to each ATE estimate",
        "the SE-ratio interval must fall below the calibration band",
    )
CELLS[("interval_calibration", "crossfit_1500__ate__shipped_curve__shrunken_se_control")] = (
    "the ATE standard errors come from the shipped adaptive-propensity curve of the same "
    "cross-fitted fits, which omits the generated-design term",
    "the SE-ratio interval must fall below the calibration band",
)
CELLS[("interval_calibration", "repeats__ate__shipped_curve__shrunken_se_control")] = (
    "the ATE standard errors are the median over the three draws of the shipped "
    "adaptive-propensity standard error, with the split-adjusted variance",
    "the SE-ratio interval must fall below the calibration band",
)

#: The ``ctmle-oat-per-arm-parametric-extensions`` cells (X58b2).  Each calibration label names one
#: fit of the stacked curve; a cell reads one estimate of it.  The names follow
#: ``tests.studies.canonical_ctmle_oat_per_arm_parametric_extensions.slug``: ``ate[0 vs 1]`` is
#: ``ate_0_vs_1`` and ``par[1]`` is ``par_1``.
_EXTENSION_SLUG = "ctmle-oat-per-arm-parametric-extensions"
#: The label ``weighted`` is an :data:`ARMS` key of other studies, so the cell lookup strips it
#: as an arm prefix.  Its cells are therefore keyed by study in :data:`STUDY_ARM_CELLS`, and the
#: fit description is the study's reading of the arm.
_EXTENSION_ARM_LABELS = ("weighted",)


def _extension_put(family: str, name: str, tested: str, required: str) -> None:
    """Register one cell of the extensions study under :data:`CELLS` or its study-keyed arm."""
    label, _, rest = name.partition("__")
    if label in _EXTENSION_ARM_LABELS:
        STUDY_ARM_CELLS[(_EXTENSION_SLUG, family, label, rest)] = (tested, required)
    else:
        CELLS[(family, name)] = (tested, required)


_EXTENSION_FITS: dict[str, tuple[str, tuple[str, ...]]] = {
    "three_arm": (
        "the stacked curve on the three-arm law, in sample, reference arm 1",
        (
            "ey_0",
            "ey_1",
            "ey_2",
            "ate_0_vs_1",
            "ate_2_vs_1",
            "log_rr_0_vs_1",
            "log_rr_2_vs_1",
            "log_or_0_vs_1",
            "log_or_2_vs_1",
            "par_1",
            "paf_1",
            "ey_obs",
        ),
    ),
    "three_arm_shared": (
        "the stacked curve on a three-arm law whose arms share W1 and W2 in their outcome index, "
        "in sample, reference arm 1",
        ("ey_0", "ey_1", "ey_2", "ate_0_vs_1", "ate_2_vs_1"),
    ),
    "binary_population": (
        "the stacked curve of the population targets on the two-arm law, in sample",
        ("par", "paf", "ey_obs"),
    ),
    "three_arm_crossfit": (
        "the cross-fitted stacked curve on the three-arm law, five folds, reference arm 0",
        ("ey_0", "ey_1", "ey_2", "ate_1_vs_0", "ate_2_vs_0"),
    ),
    "weighted": (
        "the stacked curve on the two-arm law under the known weight 0.5 + 1{W1 > 0}, in sample",
        ("ey0", "ey1", "ate", "log_rr", "log_or", "par", "paf", "ey_obs"),
    ),
    "weighted_crossfit": (
        "the cross-fitted stacked curve on the two-arm law under the known weight, five folds",
        ("ey0", "ey1", "ate"),
    ),
    "composition": (
        "the cross-fitted stacked curve on the three-arm law under the known weight, five folds, "
        "reference arm 1",
        ("ey_0", "ey_1", "ey_2", "ate_0_vs_1", "ate_2_vs_1"),
    ),
}
_EXTENSION_ESTIMATES = {
    "ey0": "arm-0 mean",
    "ey1": "arm-1 mean",
    "ate": "ATE",
    "ey_obs": "observed mean of the outcome",
    "log_rr": "log risk ratio",
    "log_or": "log odds ratio",
    "par": "population attributable risk against reference arm 0",
    "paf": "population attributable fraction against reference arm 0",
}


def _extension_estimate(token: str) -> str:
    """The reader-facing name of an estimate slug of the extensions study."""
    if token in _EXTENSION_ESTIMATES:
        return _EXTENSION_ESTIMATES[token]
    for prefix, words in (
        ("ey_", "mean of arm {0}"),
        ("par_", "population attributable risk against reference arm {0}"),
        ("paf_", "population attributable fraction against reference arm {0}"),
    ):
        if token.startswith(prefix):
            return words.format(token[len(prefix) :])
    for prefix, words in (
        ("ate_", "ATE, arm {0} versus arm {1}"),
        ("log_rr_", "log risk ratio, arm {0} versus arm {1}"),
        ("log_or_", "log odds ratio, arm {0} versus arm {1}"),
    ):
        if token.startswith(prefix):
            first, second = token[len(prefix) :].split("_vs_")
            return words.format(first, second)
    raise KeyError(token)


for _label, (_fit, _tokens) in _EXTENSION_FITS.items():
    for _token in _tokens:
        _extension_put(
            "interval_calibration",
            f"{_label}__{_token}__corrected",
            _extension_estimate(_token)
            if _label in _EXTENSION_ARM_LABELS
            else f"{_fit}: {_extension_estimate(_token)}",
            *_PARAMETRIC_CALIBRATION,
        )
for _label, _contrast in (
    ("three_arm", "ate_0_vs_1"),
    ("three_arm_shared", "ate_0_vs_1"),
    ("binary_population", "par"),
    ("three_arm_crossfit", "ate_1_vs_0"),
    ("weighted", "ate"),
    ("weighted_crossfit", "ate"),
    ("composition", "ate_0_vs_1"),
):
    _extension_put(
        "interval_calibration",
        f"{_label}__{_contrast}__shrunken_se_control",
        "the standard errors of this contrast are multiplied by a declared factor below one",
        "the SE-ratio interval must fall below the calibration band",
    )
    _extension_put(
        "interval_calibration",
        f"{_label}__{_contrast}__noise_control",
        "a declared scale of independent noise is added to each estimate of this contrast",
        "the SE-ratio interval must fall below the calibration band",
    )
for _label, _token in (
    ("three_arm", "ey_1"),
    ("three_arm_crossfit", "ey_1"),
    ("binary_population", "par"),
    ("weighted", "ey0"),
):
    _extension_put(
        "interval_calibration",
        f"{_label}__{_token}__shipped_curve__shrunken_se_control",
        f"the standard errors of {_extension_estimate(_token)} come from the shipped "
        "adaptive-propensity curve of the same fits, which omits the generated-design term",
        "the SE-ratio interval must fall below the calibration band",
    )
CELLS[
    (
        "interval_calibration",
        "three_arm_shared__ate_0_vs_1__cross_term_dropped__inflated_se_control",
    )
] = (
    "the standard errors of the contrast are built from the two arm variances without the "
    "cross-arm covariance, on a law whose arm estimates correlate",
    "the SE-ratio interval must fall above the calibration band",
)
for _label in ("three_arm", "three_arm_crossfit", "binary_population"):
    CELLS[("simultaneous_coverage", f"{_label}__simultaneous_band")] = (
        "the max-t multiplier band over the declared set of stacked estimates, from the "
        "same-row stacked influence curves",
        "joint coverage interval inside the calibration coverage band",
    )
for _label in ("three_arm", "binary_population"):
    CELLS[("simultaneous_coverage", f"{_label}__pointwise_joint_control")] = (
        "the pointwise 95% intervals of the same fits, read jointly",
        "joint coverage upper endpoint must fall below the nominal rate",
    )
    CELLS[("simultaneous_coverage", f"{_label}__shipped_curve_band_control")] = (
        "the max-t band built from the shipped adaptive-propensity curves of the same estimates",
        "joint coverage upper endpoint must fall below the nominal rate",
    )
for _contrast in ("ate_1_vs_0", "ate_2_vs_0"):
    ARM_CELLS[("generated_design", "three_arm_active", f"oracle_design__{_contrast}")] = (
        "the outcome-adaptive design is supplied rather than estimated, cross-fitted, "
        f"{_extension_estimate(_contrast)}",
        "SE ratio and coverage intervals both inside their calibration bands",
    )
    ARM_CELLS[("generated_design", "three_arm_active", f"estimated__{_contrast}")] = (
        "the same design is estimated from the data, as a real fit does, cross-fitted, "
        f"{_extension_estimate(_contrast)}",
        "SE ratio and coverage intervals both inside their calibration bands; the paired "
        "deficit is reported only",
    )
for _contrast in ("ate_0_vs_1", "ate_2_vs_1"):
    CELLS[("type_i_error", f"three_arm_null__{_contrast}")] = (
        "the three-arm law whose arms share one outcome index, so every contrast is zero: "
        f"{_extension_estimate(_contrast)}",
        "one-sided rejection bound stays under the declared type-I ceiling",
    )

STUDY_ARMS[(_EXTENSION_SLUG, "weighted")] = _EXTENSION_FITS["weighted"][0]

# ---------------------------------------------------------------- repeated longitudinal fits
# The repeated cross-fitted longitudinal study (X46).  Its keys are declared in
# ``tests/studies/longitudinal_repeated_crossfit.py`` and in its properties module.

IMPLEMENTATIONS["cleverly-repeated-longitudinal"] = (
    "`cleverly` repeated cross-fitted longitudinal fit, three outer splits"
)
SCENARIOS.update(
    {
        "law_d_ltmle": (
            "two-node censored law, `LTMLE` with five folds, three fold draws and saturated "
            "cell means"
        ),
        "law_d_sdr": (
            "the same two-node censored law, `SDR` with five folds, three fold draws and "
            "saturated cell means"
        ),
        "law_s": (
            "three-node survival law with an absorbing event, `LTMLE` with five folds and "
            "three fold draws"
        ),
        "law_c": (
            "clusters of ten units on a two-node law, `LTMLE` with `id=`, five folds and "
            "three fold draws"
        ),
        "law_m": (
            "the two-node law with an identity working model on treatment duration, `LTMLE`, "
            "five folds and three fold draws"
        ),
    }
)
REGIMENS["treat_if_l2"] = "treat first, then treat at the second time only where L2 equals one"
PROPERTIES.update(
    {
        "repeat_increment": (
            "the median over three draws moves the SE ratio of the survival risk difference at "
            "horizon 3 by less than the declared ceiling, against the first draw alone"
        ),
        "ipsi_primary": (
            "the repeated `IPSI` means and contrasts at n = 80,000 sit inside the primary "
            "bias, coverage and SE-ratio rules"
        ),
        "single_draw_reading": (
            "the first draw's SE ratio and coverage, reported beside the median for reading"
        ),
        "split_displacement": (
            "the share of the reported variance that comes from the displacement of the draws, "
            "reported for reading"
        ),
    }
)

_REPEATED_LABELS = {
    "d_ltmle_static": "`LTMLE` risk difference of always against never on the two-node law",
    "d_ltmle_dynamic": "`LTMLE` difference of treat-if-L2 against never on the two-node law",
    "d_sdr_static": "`SDR` difference of always against never on the two-node law",
    "d_sdr_dynamic": "`SDR` difference of treat-if-L2 against never on the two-node law",
    "i_contrast_x2": "`IPSI` contrast of odds multiplier 2 against 1 on the incremental law",
    "i_contrast_x0_5": "`IPSI` contrast of odds multiplier 0.5 against 1 on the incremental law",
    "s_contrast_h2": "`LTMLE` risk difference of always against never at horizon 2 on the survival law",
    "c_static": "`LTMLE` difference of always against never on the clustered law",
    "m_duration": "`LTMLE` duration coefficient of the working model",
}
_REPEATED_BANDS = {
    "d_ltmle": "every mean and contrast of `LTMLE` on the two-node law",
    "d_sdr": "every mean and contrast of `SDR` on the two-node law",
    "i_ipsi": "every mean and contrast of `IPSI` on the incremental law",
    "s_ltmle": "every risk and risk difference of `LTMLE` at horizons 1 to 3 on the survival law",
    "c_ltmle": "the means and the contrast of `LTMLE` on the clustered law",
    "m_ltmle": "both coefficients of the working model",
}
_REPEATED_DRAW = "the median over three fold draws"
_REPEATED_SLUG = "longitudinal-repeated-crossfit"
_REPEATED_ARM_WORDS = {
    "contrast_x0_5": "contrast of odds multiplier 0.5 against 1",
    "contrast_x2": "contrast of odds multiplier 2 against 1",
    "mean_x0_5": "mean under odds multiplier 0.5",
    "mean_x1": "mean under the natural course",
    "mean_x2": "mean under odds multiplier 2",
}


class _RepeatedCells:
    """File a cell of the repeated study, under its study arm when the key opens with one."""

    def __setitem__(self, index: tuple[str, str], value: tuple[str, str]) -> None:
        family, key = index
        arm = _ARM.match(key)
        if arm is None:
            CELLS[(family, key)] = value
            return
        name = arm.group("arm")
        STUDY_ARMS[(_REPEATED_SLUG, name)] = _REPEATED_ARM_WORDS.get(name, name)
        STUDY_ARM_CELLS[(_REPEATED_SLUG, family, name, arm.group("cell"))] = value


_REPEATED_CELLS = _RepeatedCells()
for _label, _words in _REPEATED_LABELS.items():
    _REPEATED_CELLS[("interval_calibration", f"{_label}__correctly_specified")] = (
        f"{_words}; {_REPEATED_DRAW} and its split-adjusted variance",
        "SE ratio and coverage intervals both inside their calibration bands",
    )
    _REPEATED_CELLS[("interval_calibration", f"{_label}__shrunken_se_control")] = (
        f"{_words}, with each reported SE multiplied by 0.70",
        "the SE-ratio interval must fall below the calibration band",
    )
    _REPEATED_CELLS[("single_draw_reading", f"{_label}__draw_0")] = (
        f"{_words}; the first of the three draws alone",
        "reported only; no verdict",
    )
    _REPEATED_CELLS[("split_displacement", f"{_label}__share")] = (
        f"{_words}; the share of the reported variance that is the squared displacement of "
        "the draws",
        "reported only; no verdict",
    )
for _label, _words in _REPEATED_BANDS.items():
    _REPEATED_CELLS[("simultaneous_coverage", f"{_label}__simultaneous_band")] = (
        f"the shipped band over {_words}",
        "joint coverage interval inside the calibration coverage band",
    )
    _REPEATED_CELLS[("simultaneous_coverage", f"{_label}__pointwise_joint_control")] = (
        f"the pointwise 95% intervals over {_words}, read jointly",
        "joint coverage upper endpoint must fall below the nominal rate",
    )
for _label in (
    "d_ltmle_static",
    "d_ltmle_dynamic",
    "d_sdr_static",
    "d_sdr_dynamic",
    "i_contrast_x2",
):
    _REPEATED_CELLS[("repeat_stability", f"{_label}__three_repeats")] = (
        f"{_REPEATED_LABELS[_label]}; {_REPEATED_DRAW} on one fixed sample, over many fold seeds",
        "the spread of the median over the spread of its first draw stays under the declared "
        "boundary, above the spread floor",
    )
    _REPEATED_CELLS[("repeat_stability", f"{_label}__one_repeat")] = (
        f"{_REPEATED_LABELS[_label]}; the first draw alone on the same fixed sample; the base "
        "of the pair's spread ratio",
        "the pair's verdict: the spread of the median over this spread stays under the "
        "declared boundary, above the spread floor",
    )
    _REPEATED_CELLS[("repeat_stability", f"{_label}__second_draw_control")] = (
        f"{_REPEATED_LABELS[_label]}; the second draw of the same three-draw fit, alone",
        "the spread of an independent single draw over the first draw's spread must not fall "
        "below the declared boundary (no reduction exists to read), above the spread floor",
    )
for _horizon in (1, 2, 3):
    _reading = f"survival risk difference at horizon {_horizon}, the SE ratio of "
    _role = "gated" if _horizon == 3 else "reported only"
    _REPEATED_CELLS[("repeat_increment", f"h{_horizon}__repeat_median")] = (
        f"{_reading}the median over three draws",
        "the increment over the first draw stays under its ceiling"
        if _horizon == 3
        else "reported only; no verdict",
    )
    _REPEATED_CELLS[("repeat_increment", f"h{_horizon}__single_draw")] = (
        f"{_reading}the first draw alone",
        "the first-draw ratio, the base of the increment"
        if _horizon == 3
        else "reported only; no verdict",
    )
_REPEATED_CELLS[("repeat_increment", "h3__inflated_se_control")] = (
    "the horizon-3 median with its reported SE multiplied by 1.05",
    "the increment must reach or exceed the ceiling",
)
for _label, _words in {
    "mean_x0_5": "the mean under odds multiplier 0.5",
    "mean_x1": "the mean under the natural course",
    "mean_x2": "the mean under odds multiplier 2",
    "contrast_x0_5": "the contrast of odds multiplier 0.5 against 1",
    "contrast_x2": "the contrast of odds multiplier 2 against 1",
}.items():
    _REPEATED_CELLS[("ipsi_primary", f"{_label}__repeated")] = (
        f"`IPSI` at n = 80,000, {_words}; {_REPEATED_DRAW}",
        "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
    )
for _fit, _name in (("ltmle_static", "`LTMLE`"), ("sdr_static", "`SDR`")):
    for _n in (2000, 8000, 32000):
        _REPEATED_CELLS[("root_n_and_efficiency", f"{_fit}__n_{_n}")] = (
            f"{_name} static contrast on the two-node law at n = {_n:,}; {_REPEATED_DRAW}",
            "bias inside the margin, coverage clears the floor, SE ratio inside the band",
        )
    _REPEATED_CELLS[("root_n_rate", f"{_fit}__empirical_sd")] = (
        f"{_name} static contrast; the log-log slope of the empirical SD over the three sizes",
        "the slope interval lies inside the root-n band",
    )
    _REPEATED_CELLS[("root_n_rate", f"{_fit}__reported_se")] = (
        f"{_name} static contrast; the log-log slope of the mean reported SE over the three sizes",
        "the slope interval lies inside the root-n band",
    )


#: The ``drtmle-foldwise`` cells (X83).  Each cell names its nuisance regime: the leak pair and
#: four variants fit ``treatment_correct``, ``bivariate_oc`` and the stratified cells fit
#: ``outcome_correct``, and the control fits ``both_wrong``.  The fits are cross-fitted with
#: fold-wise reduced regressions unless a cell says otherwise.
_FOLDWISE_SLUG = "drtmle-foldwise"
_FOLDWISE_POSITIVE = (
    "bias interval inside the equivalence margin, with the reported standard error on the "
    "scale of the empirical spread"
)
STUDY_ARMS[(_FOLDWISE_SLUG, "multi_arm_drtmle")] = "medium-versus-high ATE of the three-arm law"
for (_family, _arm, _base), _words in {
    ("fold_locality", "ate", "foldwise"): (
        "only the treatment mechanism is correct, n = 1,000, ten folds; each fold's reduced "
        "regressions train on leave-two-out designs",
        "bias interval inside the equivalence margin",
    ),
    ("fold_locality", "ate", "pooled"): (
        "the same draw with pooled reduced regressions, R drtmle's cvFolds construction, which "
        "lets each fold's data reach its own reduced regressions",
        "bias interval must fall entirely outside the margin",
    ),
    ("double_robustness", "ate", "benkeser_order"): (
        "only the treatment mechanism is correct, n = 3,000, the published six-step round order",
        _FOLDWISE_POSITIVE,
    ),
    ("double_robustness", "ate", "bivariate"): (
        "only the treatment mechanism is correct, n = 3,000, the single bivariate reduced "
        "mechanism",
        _FOLDWISE_POSITIVE,
    ),
    ("double_robustness", "ate", "bivariate_oc"): (
        "only the outcome regression is correct, n = 3,000, the single bivariate reduced mechanism",
        _FOLDWISE_POSITIVE,
    ),
    ("double_robustness", "ate", "k5"): (
        "only the treatment mechanism is correct, n = 3,000, five folds",
        _FOLDWISE_POSITIVE,
    ),
    ("double_robustness", "ate", "spline_qr"): (
        "only the treatment mechanism is correct, n = 3,000, a cubic-spline reduced outcome "
        "regression Q_r with the canonical GLMs for g_r1 and g_r2; "
        "piecewise-constant reduced learners (capped boosting, k-NN) reached max_outer = 100 "
        "under both constructions on the pilot, so the study covers smooth reduced learners only",
        _FOLDWISE_POSITIVE,
    ),
    ("double_robustness", "ate", "both_wrong"): (
        "both nuisances are misspecified, n = 3,000, ten folds",
        "bias interval must fall entirely outside the margin, with the reported standard error "
        "on the scale of the empirical spread",
    ),
    ("double_robustness", "multi_arm_drtmle", "treatment_correct"): (
        "only the treatment mechanism is correct, n = 2,000, five folds, on the draws of the "
        "registered multi-arm cell",
        _FOLDWISE_POSITIVE,
    ),
    **{
        ("double_robustness", _label, "stratified"): (
            "the stratified paper law; only the outcome regression is correct, n = 2,000, ten "
            "folds",
            _FOLDWISE_POSITIVE,
        )
        for _label in ("marginal_ate", "v0_ate", "v1_ate", "v2_ate")
    },
}.items():
    STUDY_ARM_CELLS[(_FOLDWISE_SLUG, _family, _arm, _base)] = _words


def implementation(key: str) -> str:
    """The reader-facing name of an implementation column value."""
    try:
        return IMPLEMENTATIONS[key]
    except KeyError:
        raise Undescribed(f"no description for implementation {key!r}") from None


#: Scenario descriptions that one study scopes, by ``(slug, scenario)``. A study whose
#: recorded scenario key another study already describes names its own law here.
STUDY_SCENARIOS: dict[tuple[str, str], str] = {}


def scenario(key: str, *, slug: str | None = None) -> str:
    """The law a scenario samples, read from the study's own entry first."""
    if slug is not None and (slug, key) in STUDY_SCENARIOS:
        return STUDY_SCENARIOS[slug, key]
    try:
        return SCENARIOS[key]
    except KeyError:
        raise Undescribed(f"no description for scenario {key!r}") from None


#: Estimand descriptions that one study scopes, by ``(slug, estimand)``.  A study whose row key
#: another study already describes, such as ``ate_lower`` at another strength, names its own
#: reading here.
STUDY_ESTIMANDS: dict[tuple[str, str], str] = {}


def estimand(key: str, *, outcome_kind: str = "absorbing", slug: str | None = None) -> str:
    """What an estimand key names, including a bracketed longitudinal regimen.

    ``outcome_kind`` is the study record's declaration.  It decides the wording of a
    horizon-indexed contrast, which reads the same on a survival and a repeated outcome.
    ``slug`` selects the study's own entry in :data:`STUDY_ESTIMANDS` first.
    """
    if slug is not None and (slug, key) in STUDY_ESTIMANDS:
        return STUDY_ESTIMANDS[slug, key]
    if slug in LONG_HORIZON_SLUGS:
        return _long_horizon_estimand(key)
    stratum = _STRATUM.match(key)
    if stratum is not None:
        base = estimand(stratum.group("base"), outcome_kind=outcome_kind)
        return f"{base}, in stratum {stratum.group('stratum')}"
    match = _PARAMETERISED.match(key)
    if match is None:
        try:
            return ESTIMANDS[key]
        except KeyError:
            raise Undescribed(f"no description for estimand {key!r}") from None
    name, argument = match.group("name"), match.group("argument")
    if name == "blip":
        treatment, level, feature = argument.split(":")
        words = f"blip of treatment {treatment}"
        words += " per unit dose" if level == "dose" else f" at level {float(level):g}"
        return words if feature == "intercept" else f"{words}, coefficient of {feature}"
    if name == "ey":
        return f"counterfactual mean under treatment arm {argument!r}"
    if name in {"par", "paf"}:
        return f"{ESTIMANDS[name]}, natural course versus reference arm {argument!r}"
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
    # ``ey_regime`` and ``ey_policy`` are point-treatment parameters, ``ey_regimen`` the
    # longitudinal one.
    point = name in {"ey_regime", "ate_regime", "ey_policy", "ate_policy"}
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
    # record -- it must not be added by widening one of the two branches below.  The repeated
    # outcome is that third construction, and the record's ``outcome_kind`` settles it.
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
    if outcome_kind == "repeated":
        if name not in {"ey_regimen", "ate_regimen"}:
            raise Undescribed(f"{key!r} is not a parameter of a repeated outcome")
        plan, horizon = argument.rsplit(_HORIZON, 1)
        mean = (
            "mean of the repeated outcome under the plan"
            if name == "ey_regimen"
            else "difference in the mean of the repeated outcome between the plans"
        )
        return f"{mean} {_repeated_plans(plan)} at horizon t = {horizon}"
    if name == "ate_regimen":
        parameter = "difference in cumulative risk between the plans"
    return f"{parameter} {regimen(argument, point=point)}"


def _repeated_plans(argument: str) -> str:
    """A plan or a contrast of two plans of the repeated-outcome study, in words."""
    labels = argument.split(" vs ", 1)
    try:
        words = [REPEATED_REGIMENS[label] for label in labels]
    except KeyError:
        raise Undescribed(f"no description for repeated-outcome plan {argument!r}") from None
    if len(words) == 1:
        return words[0]
    return f'"{words[0]}" against "{words[1]}"'


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
    response_nuisance: bool = False,
    slug: str | None = None,
) -> tuple[str, str]:
    """What a property cell configures, and what its verdict requires.

    The arm prefix a longitudinal study puts on a cell name selects the plan rather than the
    configuration, so it is stripped and reported beside the shared description.
    """
    if slug in LONG_HORIZON_SLUGS:
        return _long_horizon_cell(family, key, role, slug)
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
    scoped = STUDY_ARM_CELLS.get((slug, family, arm.group("arm"), base)) if arm and slug else None
    try:
        tested, required = (
            scoped
            or (ARM_CELLS.get((family, arm.group("arm"), base)) if arm else None)
            or CELLS[family, base]
        )
    except KeyError:
        raise Undescribed(f"no description for cell {key!r} of {family!r}") from None
    if size is not None:
        tested = f"{tested}, at n = {int(size.group('size')):,}"
    if family == "interval_calibration" and base == "correctly_specified" and scoped is None:
        if nuisance_count == 2:
            tested = "both nuisances are correctly specified"
        else:
            words = {3: "three", 4: "four"}
            count = words.get(nuisance_count, str(nuisance_count))
            tested = f"all {count} required nuisance functions are correctly specified"
    if family == "double_robustness" and base == "both_correct" and response_nuisance:
        tested = "both the outcome regression and the outcome-response mechanism are correct"
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
    if family == "root_n_and_efficiency" and base in {"n_500", "n_1000", "n_24000"}:
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
        words = STUDY_ARMS.get((slug, arm.group("arm"))) if slug else None
        tested = f"{words or ARMS[arm.group('arm')]}: {tested}"
    if term is not None:
        tested = f"{TERMS[term.group('term')]}: {tested}"
    return tested, required


#: The ``few_cluster_reference`` cells of ``clustered-few-cluster-tmle``: one entry per fit,
#: size law, cluster count and arm, composed forward from the four parts.
#: Mirrored from ``tests.studies.clustered_few_cluster_tmle.CLUSTER_COUNTS``, which this module
#: cannot import (descriptions stay out of every study's hashed module list).
#: ``tests/unit/test_clustered_unequal_study.py`` pins the two equal.
_FEW_CLUSTER_COUNTS = (10, 20, 30)
_FEW_CLUSTER_FITS = {
    "tmle_crossfit": "stacked CV-TMLE",
    "tmle_cv_evaluation": "fold-evaluated CV-TMLE",
    "tmle_in_sample": "in-sample TMLE",
    "drtmle_crossfit": "cross-fitted DR-TMLE",
    "ltmle_in_sample": "in-sample LTMLE on one node",
}
_FEW_CLUSTER_SIZES = {
    "equal10": "clusters of 10 rows",
    "unequal_informative": "clusters of size uniform on 2 to 18, the size in the outcome",
}
_FEW_CLUSTER_ARMS = {
    "t_reference": (
        "the reported interval, t with J - 2 degrees of freedom, or min(J - 2, J - V) for the "
        "fold-evaluated fit",
        "exact coverage lower bound clears the floor, bias inside the margin",
    ),
    "iid_t_control": (
        "the same point and t quantile with the IID row standard error",
        "the SE-ratio upper endpoint falls below 0.80",
    ),
    "normal_reference": (
        "the cluster-robust standard error with the normal quantile",
        "reported only",
    ),
    "t_j_minus_2_reference": (
        "the fold-evaluated standard error with t at J - 2 degrees of freedom",
        "reported only",
    ),
}
CELLS.update(
    {
        (
            "few_cluster_reference",
            f"{fit}__{sizes}__j{clusters}__{arm}",
        ): (
            f"{fit_text}, {clusters} {size_text}: {arm_text}",
            required,
        )
        for fit, fit_text in _FEW_CLUSTER_FITS.items()
        for sizes, size_text in _FEW_CLUSTER_SIZES.items()
        for clusters in _FEW_CLUSTER_COUNTS
        for arm, (arm_text, required) in _FEW_CLUSTER_ARMS.items()
        if arm != "t_j_minus_2_reference" or fit == "tmle_cv_evaluation"
        if fit != "ltmle_in_sample" or clusters >= 20
    }
)


#: The ``few_cluster_reference`` cells of ``few-cluster-cross-fitted-ltmle``. Mirrored from
#: ``tests.studies.few_cluster_crossfit_ltmle``; ``tests/unit/test_clustered_crossfit_ltmle_design.py``
#: pins the two equal.
_CROSSFIT_FEW_CLUSTER_COUNTS = (20, 30)
_CROSSFIT_FEW_CLUSTER_SIZES = {
    "equal40": "clusters of 40 rows",
    "unequal40": "clusters of size uniform on 10 to 70",
}
# ``ltmle_crossfit`` is an arm prefix of :data:`ARMS`, so ``cell`` strips it and reads these by
# ``(family, arm, rest)``; the arm text is printed before each description.
ARM_CELLS.update(
    {
        ("few_cluster_reference", "ltmle_crossfit", f"{sizes}__j{clusters}__{arm}"): (
            f"with id= and five whole-cluster folds, {clusters} {size_text}: {arm_text}",
            required,
        )
        for sizes, size_text in _CROSSFIT_FEW_CLUSTER_SIZES.items()
        for clusters in _CROSSFIT_FEW_CLUSTER_COUNTS
        for arm, (arm_text, required) in _FEW_CLUSTER_ARMS.items()
        if arm != "t_j_minus_2_reference"
    }
)

#: ``longitudinal-mtp`` reads its mechanism-only arms at 320 oracle bins.  With the outcome
#: regressions wrong, the binned density's error enters the bias at first order: 0.54 to 0.60
#: standardized at 80 bins, 0.21 to 0.24 at 160 (100 draws of 8,000 rows).  These cells
#: therefore witness the targeting with a nearly exact ratio, not the shipped default density.
ARM_CELLS[("double_robustness", "up", "mechanism_correct")] = (
    "only the mechanisms are correctly specified, read from 320 oracle bins so the binned "
    "density's own first-order error stays below the margin",
    "bias interval inside the equivalence margin, with the reported standard error "
    "on the scale of the empirical spread",
)
ARM_CELLS[("targeting_necessity", "up", "targeted")] = (
    "the targeted fit with only the mechanisms correct, at 320 oracle bins",
    "bias interval inside the equivalence margin",
)
ARM_CELLS[("inverse_necessity", "history", "declared_inverse")] = (
    "the density ratio reads each piece at its declared inverse, at 320 oracle bins",
    "bias interval inside the equivalence margin",
)
ARM_CELLS[("inverse_necessity", "history", "inverse_dropped_control")] = (
    "the same fit reads each moving piece at the dose itself, at 320 oracle bins",
    "bias interval must fall entirely outside the margin",
)
ARM_CELLS[("interval_calibration", "classifier_route", "correctly_specified")] = (
    "the ratio by stacked classification, with a logistic model on the true log ratio, so the "
    "classifier is correctly specified",
    "SE ratio and coverage intervals both inside their calibration bands",
)


#: The cells of ``longitudinal-incremental`` and ``longitudinal-incremental-dropout`` (X19)
#: that no other study shares.  The arm prefix names the plan (:data:`ARMS`).
CELLS.update(
    {
        ("mechanism_term_necessity", "full_eif"): (
            "the reported standard error from the full influence curve",
            "the SE-ratio interval inside the efficiency-ratio band",
        ),
        ("mechanism_term_necessity", "mechanism_term_dropped"): (
            "the same estimates with the treatment-mechanism term, on the outcome's own scale, "
            "removed from the curve",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("single_robustness", "mechanism_correct__n_48000"): (
            "cell-mean treatment mechanism and linear outcome regressions, at n = 48,000",
            "bias inside the margin at n = 48,000 and at n = 96,000",
        ),
        ("single_robustness", "mechanism_correct__n_96000"): (
            "cell-mean treatment mechanism and linear outcome regressions, at n = 96,000",
            "bias inside the margin at n = 48,000 and at n = 96,000",
        ),
        ("single_robustness", "mechanism_constant"): (
            "constant treatment mechanism and cell-mean outcome regressions, beside the "
            "exact-law limit",
            "reported only",
        ),
        ("targeting_necessity", "one_step"): (
            "the one-step estimate with linear outcome regressions",
            "bias inside the margin",
        ),
        ("targeting_necessity", "plug_in"): (
            "the mean of the first-node regression before the correction, on the same fits",
            "bias interval outside the margin",
        ),
        ("mechanism_term_necessity", "binary_blip"): (
            "the same estimates with the mechanism term read from the first two levels only",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("single_robustness", "mechanism_correct__n_8000"): (
            "cell-frequency treatment mechanism and a constant outcome regression, at n = 8,000",
            "bias inside the margin at n = 8,000 and at n = 16,000",
        ),
        ("single_robustness", "mechanism_correct__n_16000"): (
            "cell-frequency treatment mechanism and a constant outcome regression, at n = 16,000",
            "bias inside the margin at n = 8,000 and at n = 16,000",
        ),
        ("in_sample_fit", "one_fold"): (
            "the same histories fitted with one fold, beside the exact-law truth",
            "reported only",
        ),
        ("mechanism_targeting", "mechanism_targeted"): (
            "the treatment mechanism is tilted along the T_a mechanism score by 2 / sqrt(n), "
            "then the full targeting runs",
            "bias inside the margin and the SE-ratio interval inside the efficiency-ratio band",
        ),
        ("mechanism_targeting", "outcome_targeted_only"): (
            "the same tilted mechanism with the outcome fluctuation alone",
            "bias interval must fall above zero and entirely outside the margin",
        ),
        ("censoring_complete_case", "complete_case"): (
            "a fit to the rows that never drop out, without a censoring model, beside the "
            "exact-law limit",
            "reported only",
        ),
    }
)


#: The ``sequential_double_robustness`` cells of ``longitudinal-sdr``: one pattern of wrong
#: nuisances per node, at two sizes.  Mirrored from
#: ``tests.studies.longitudinal_sdr_properties.PATTERN_NUISANCES``.
_SDR_PATTERNS = {
    "both_correct": (
        "every outcome regression and mechanism is correct",
        "bias inside the margin, SE ratio inside the union-model band, coverage clears the floor",
    ),
    "P_C": (
        "both mechanisms are correct and both outcome regressions are wrong",
        "bias inside the margin, SE ratio inside the union-model band, coverage clears the floor",
    ),
    "P_A": (
        "both outcome regressions are correct and both mechanisms are wrong",
        "absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio "
        "inside the union-model band",
    ),
    "P_B": (
        "node 1 outcome regression wrong, node 2 mechanism wrong",
        "absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio "
        "inside the union-model band",
    ),
    "P_D": (
        "node 1 mechanism wrong, node 2 outcome regression wrong, the pattern LTMLE misses",
        "absolute bias + 2 MCSE at n = 8,000 below 0.01 and below its value at n = 2,000, SE ratio "
        "inside the union-model band",
    ),
}
CELLS.update(
    {
        ("sequential_double_robustness", f"{pattern}__n_{size}"): (
            f"{tested}, at n = {size:,}",
            required,
        )
        for pattern, (tested, required) in _SDR_PATTERNS.items()
        for size in (2_000, 8_000)
    }
)
CELLS.update(
    {
        ("sequential_double_robustness", "node2_both_wrong_control"): (
            "SDR with the outcome regression and the mechanism both wrong at node 2",
            "bias interval at least 3 empirical standard deviations from zero, SE ratio inside "
            "the union-model band",
        ),
        ("sequential_double_robustness", "ltmle_P_D_control"): (
            "LTMLE in pattern P_D, where its recursion needs the node 2 outcome regression",
            "bias interval at least 3 empirical standard deviations from zero, SE ratio inside "
            "the union-model band",
        ),
        ("crossfit_overfitting", "cross_fitted_sdr"): (
            "five-fold SDR with fully grown outcome trees and the known mechanism",
            "SE ratio clears the overfitting floor and stays inside the sanity band",
        ),
        ("construction_diagnostic", "stitched_cross_fit"): (
            "the stitched construction: each row's node 2 transform comes from the fold that "
            "holds it out, and node 1 regresses the stitched transforms",
            "reported only",
        ),
        ("construction_diagnostic", "estimated_flexible_mechanism"): (
            "the shipped construction with a tree mechanism estimated in each fold, under the "
            "default bounds (0.01, 1)",
            "reported only",
        ),
        ("sdr_tmle_agreement", "static"): (
            "the absolute difference of SDR and LTMLE over the SE of SDR, for always against never, "
            "on the calibration draws",
            "reported only",
        ),
        ("clustered_calibration", "known_ratio"): (
            "grouped folds, the known ratio and GLM outcome regressions",
            "SE ratio and coverage intervals both inside their calibration bands",
        ),
        ("weighted_calibration", "correctly_specified"): (
            "every nuisance is correct; no efficiency ratio is claimed (RM18-fixed-weights)",
            "SE ratio and coverage intervals both inside their calibration bands",
        ),
        ("weighted_calibration", "shrunken_se_control"): (
            "the reported standard errors are multiplied by a declared factor below one",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("weighted_calibration", "noise_control"): (
            "one efficiency-bound unit of independent noise is added to each estimate",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("clustered_calibration", "shrunken_se_control"): (
            "the reported standard errors are multiplied by a declared factor below one",
            "the SE-ratio interval must fall below the calibration band",
        ),
        ("clustered_calibration", "noise_control"): (
            "a declared scale of independent noise is added to each estimate",
            "the SE-ratio interval must fall below the calibration band",
        ),
    }
)


# ------------------------------------------------------------------ sequential-baselines
IMPLEMENTATIONS["cleverly-sequential-baselines"] = (
    "`cleverly` sequential g-computation (ICE) and sequential IPW, Hajek and Horvitz-Thompson, "
    "with GLM nuisances"
)
SCENARIOS.update(
    {
        **{
            f"d2_{estimator}": f"law D2: two nodes, a three-arm second node, censoring; {words}"
            for estimator, words in _BASELINE_ESTIMATORS.items()
        },
        **{
            f"g_{estimator}": f"law G: three nodes, a continuous outcome, censoring; {words}"
            for estimator, words in _BASELINE_ESTIMATORS.items()
        },
    }
)
REGIMENS.update(
    {
        "static_12": "assign arm 1 first, then arm 2",
        "static_00": "assign arm 0 at both times",
        "policy": "assign arm 1 first, then draw the arm from the known policy (0.2, 0.3, 0.5)",
        "treat_all": "treat at each of the three times",
        "treat_none": "treat at none of the three times",
        "down": "lower each dose by 0.5 where the result stays in its conditional support",
    }
)
PROPERTIES.update(
    {
        "misspecification_control": (
            "with every main-terms GLM wrong, the interval covers the misspecified limit, so the "
            "calibration rule fails"
        ),
        "fixed_mechanism_se": (
            "the IPW standard error that treats the fitted mechanism as known, against the "
            "empirical spread"
        ),
        "first_node_se": (
            "the ICE standard error of the first-node term alone, against the empirical spread"
        ),
        "adaptive_learners": (
            "points with the default learner library, reported without an interval"
        ),
    }
)
CELLS.update(
    {
        ("misspecification_control", "main_terms"): (
            "main-terms GLM outcome regressions and mechanisms on law X, where both are wrong",
            "declared red: the calibration rule must fail",
        ),
        ("fixed_mechanism_se", "fixed_mechanism"): (
            "the mechanism is fitted, and the standard error omits its estimation term",
            "where gated, the SE-ratio interval must rise above the calibration band; "
            "otherwise reported",
        ),
        ("first_node_se", "first_node"): (
            "the spread of the first-node term alone, without the later chain terms",
            "the SE-ratio interval must fall below the calibration band",
        ),
        **{
            ("adaptive_learners", f"n_{size}"): (
                f"the default learner library on law X, at n = {size:,}",
                "reported only",
            )
            for size in (1_000,)
        },
    }
)


# ------------------------------------------------------------- structural-nested-mean
IMPLEMENTATIONS["cleverly-structural-nested-mean"] = (
    "`cleverly` structural nested mean model, g-estimation with GLM nuisances (a shallow "
    "booster on law N)"
)
IMPLEMENTATIONS["econml"] = (
    "EconML 0.17.0 `DynamicDML`, full-history state, the subject's folds and `GLM` design"
)
ESTIMANDS["ey_reference"] = "mean outcome with every treatment at its reference level"
SCENARIOS.update(
    {
        "s2": "law S2: two binary nodes, a constant blip at each node",
        "s3": "law S3: three binary nodes with a path from each treatment to the later ones",
        "k": "law K: two nodes of three levels with unequal level effects",
        "c": "law C: two doses on -1, 0, 1, every conditional mean linear in the history",
        "h": "law H: blip features of the baseline at node 1 and of the baseline and the past "
        "treatment at node 2",
        "p": "law P: the second blip depends on the first treatment",
        "m": "law M: the second treatment and the outcome read the first covariate",
        "l": "law L: the second blip depends on the second covariate, which the feature declares",
        "n": "law N: nonlinear nuisances on a nine-point baseline, fitted by a shallow booster",
        "s2w": "law S2 with the known weight 0.5 + 1.5 W, target the reweighted population",
        "s2c": "law S2 in clusters of ten that share the baseline and an outcome intercept",
    }
)
# -------------------------------------------------------------------- parametric-gformula
IMPLEMENTATIONS["cleverly-parametric-gformula"] = (
    "`cleverly` parametric g-formula (NICE), a forward simulation of node-specific GLMs"
)
SCENARIOS.update(
    {
        "g1_nice": (
            "law G1: three nodes, a binary covariate, censoring and a binary end-of-study outcome"
        ),
        "g2_nice": (
            "law G2: three nodes, a Gaussian then a binary covariate per node and a continuous "
            "outcome"
        ),
        "g3_nice": ("law G3: four nodes, a binary covariate, censoring and two competing causes"),
        "g4_nice": (
            "law G4: three nodes, a Gaussian covariate and a continuous dose per node, and a "
            "continuous outcome"
        ),
        "g5_nice": (
            "law G5: three nodes, a binary and a three-level categorical covariate, a binary "
            "treatment and a continuous outcome"
        ),
    }
)
REGIMENS.update(
    {
        "r00": "assign no treatment at both nodes",
        "r11": "assign treatment at both nodes",
        "r10": "assign treatment at the first node only",
        "r01": "assign treatment at the second node only",
        "r000": "assign no treatment at the three nodes",
        "r111": "assign treatment at the three nodes",
        "r101": "assign treatment at the first and third nodes only",
        "r12": "assign level 1 at the first node and level 2 at the second",
        "r21": "assign level 2 at the first node and level 1 at the second",
        "dose0": "assign dose 0 at both nodes",
        "dose_mid": "assign dose 0.5 at the first node and 0.25 at the second",
        "dose_high": "assign dose 1 at the first node and -0.5 at the second",
        "treat_all_4": "treat at each of the four times",
        "treat_none_4": "treat at none of the four times",
        "policy_b": "treat with probability 0.3 + 0.4 B_t at each of the three times",
        "shift_half": "add 0.5 to the dose at each of the three times",
        "dose_half": "assign the fixed dose 0.5 at each of the three times",
        "rule_dose": "assign the dose 0.5 + 0.5 L_t at each of the three times",
    }
)
PROPERTIES["blip_feature_control"] = (
    "with a blip feature missing, the regimen estimates move off the g-formula truth, and the "
    "bias interval shows it"
)
PROPERTIES["positivity_reporting"] = (
    "the bias, coverage and SE ratio of the structural estimator when some treatment "
    "probabilities reach 0.05 and 0.94, reported rather than gated"
)
PROPERTIES["unclustered_se"] = (
    "the standard error that ignores the cluster column, against the empirical spread"
)
CELLS.update(
    {
        ("blip_feature_control", "too_few_features"): (
            "the structural fit with the second-node blip feature the law needs left out",
            "the bias interval must lie beyond the margin on the side of the computed gap",
        ),
        ("warning_rate", "model_check"): (
            "the structural model check against the nonparametric contrasts, with the "
            "structural model correct; the rejection rate is published beside the verdict, and on "
            "the sparse-positivity law the covariance has few degrees of freedom, so the check "
            "can reject above its nominal level there",
            "rejection upper bound at or below the type-I ceiling",
        ),
        ("power", "model_check"): (
            "the same check on a law whose second blip reads a covariate the structural "
            "model omits",
            "rejection lower bound clears the minimum power",
        ),
        ("positivity_reporting", "weak_positivity"): (
            "the structural fit on a law with weak positivity, correct blip features",
            "reported with no verdict",
        ),
        ("unclustered_se", "unclustered"): (
            "the fit without the cluster column, on the clustered draws",
            "the gated cells `s2c_ey0` and `s2c_ey_r11` need the SE-ratio interval below the "
            "calibration band, and the result column reads reported; the other cells state no "
            "verdict",
        ),
    }
)
PROPERTIES.update(
    {
        "monte_carlo_accounting": (
            "with one simulated history per unit, the interval that includes the simulation "
            "error is calibrated, and a curve without that term is too narrow"
        ),
        "ice_necessity": (
            "the main-terms ICE fit is biased where the parametric g-formula is not, on the "
            "same draws"
        ),
        "ice_reported": (
            "the main-terms ICE fit beside the parametric g-formula on the same draws, reported "
            "without a verdict"
        ),
        "shared_target": (
            "the targeted estimator beside the parametric g-formula on the same draws, reported "
            "without a verdict"
        ),
        "extrapolation_report": (
            "a main-terms fit extrapolates to a fixed dose where the outcome has a quadratic "
            "dose term, reported without a verdict"
        ),
    }
)
CELLS.update(
    {
        ("monte_carlo_accounting", "n_draws_1"): (
            "one simulated history per unit behind the estimate",
            "bias inside the margin, SE-ratio and coverage intervals inside their calibration "
            "bands",
        ),
        ("monte_carlo_accounting", "mc_removed"): (
            "the estimate of the one-history fit, with the standard error of an independent fit "
            "of 500 histories per unit, so the simulation term is absent from the curve",
            "reported only: the SE-ratio interval, with no verdict",
        ),
        ("ice_necessity", "nice"): (
            "the parametric g-formula with main-terms models",
            "bias inside the margin",
        ),
        ("ice_necessity", "ice"): (
            "the main-terms ICE fit on the same draws",
            "bias outside the margin, and displaced from the parametric g-formula by the "
            "declared threshold",
        ),
        ("ice_reported", "nice"): (
            "the parametric g-formula with main-terms models",
            "reported only",
        ),
        ("ice_reported", "ice"): ("the main-terms ICE fit on the same draws", "reported only"),
        ("shared_target", "nice"): (
            "the parametric g-formula with main-terms models",
            "reported only",
        ),
        ("shared_target", "ltmle"): (
            "`LTMLE` with GLM nuisances on the same draws, which targets the same parameter",
            "reported only",
        ),
        ("shared_target", "ipsi"): (
            "`IPSI` with GLM nuisances on the same draws, which targets the same parameter",
            "reported only",
        ),
        ("extrapolation_report", "main_terms"): (
            "a main-terms model of a law with a quadratic dose term, fitted and extrapolated to "
            "a fixed dose",
            "reported only",
        ),
    }
)


# ``survival-node-bounds`` (X33): a point-treatment survival law with five visits and node bounds.
SCENARIOS.update(
    {
        "mean_model": (
            "point-treatment survival law with five visits, binary arm and two binary baseline "
            "covariates; the iterated-mean fit, every node bounded"
        ),
        "hazard_model": (
            "the same law; the pooled-hazard fit with per-horizon targeting, every node's hazard "
            "bounded"
        ),
    }
)
PROPERTIES.update(
    {
        "reported_calibration": (
            "the bounded fits' intervals for the parameters the bounds do not gate, reported "
            "without a verdict"
        ),
        "binding_witness": (
            "the declared bounds bind: the unbounded fit leaves them on at least half of the "
            "draws, and the bounded estimates differ from the unbounded ones on at least half"
        ),
        "unbounded_fit": (
            "the same fits without node bounds on the same draws, with the RMSE of the bounded "
            "and unbounded fits side by side, reported without a verdict"
        ),
        "bound_necessity": (
            "bounds that exclude the truth bias the bounded estimate beyond the margin, and valid "
            "bounds do not"
        ),
    }
)
CELLS.update(
    {
        ("reported_calibration", "reported"): (
            "the bounded fit on the draws of the calibration cells, for a parameter the bounds "
            "do not gate",
            "reported with no verdict",
        ),
        ("binding_witness", "binding"): (
            "the share of draws on which the unbounded fit leaves the valid bounds on a node "
            "follower, and the share on which the bounded and unbounded estimates of the risk "
            "by visit 5 under arm 1 or of the risk difference differ by more than 1e-4",
            "both shares at least one half for the iterated-mean fit; the hazard fit reports "
            "its shares without a verdict",
        ),
        ("unbounded_fit", "unbounded"): (
            "the identical fit with no node bounds, on the draws of the calibration cells",
            "reported with no verdict",
        ),
        ("bound_necessity", "valid_bounds"): (
            "the valid bounds, a saturated outcome regression and an intercept-only treatment "
            "mechanism",
            "bias interval must fall inside the margin",
        ),
        ("bound_necessity", "excluding_bound"): (
            "the upper bound at every node lowered to half the way from the arm-1 follower mean "
            "of the true iterated mean to its largest value, so it excludes the truth in the "
            "high-risk covariate cells",
            "the displacement from the valid-bounds fit must reach one quarter of the "
            "valid-bounds standard deviation, and the bias interval must fall beyond the margin",
        ),
    }
)


# ------------------------------------------------------------------------- adaptive-iptw (X21)
#: ``p_ey0`` is an arm of another study with another reading.
STUDY_ARMS[("adaptive-iptw", "p_ey0")] = "arm-0 mean"

IMPLEMENTATIONS["adaptive_iptw"] = (
    "`cleverly` `AdaptiveIPTW`, the Horvitz-Thompson mean with a targeted propensity"
)
SCENARIOS["known_arms"] = (
    "effect-modified two-arm law with a continuous and a binary covariate; the cross-fitted "
    "targeted fit of the arm means and the ATE with the treatment mechanism declared known"
)

PROPERTIES.update(
    {
        "bias": (
            "the point estimate of the fit is unbiased in the unit of its own spread: the "
            "standardized bias lies inside the margin where the learner is consistent for the "
            "mechanism, and outside it where the learner is not"
        ),
        "root_n_points": (
            "the estimate under a mechanism that converges at the cube-root rate keeps a "
            "standardized bias inside the margin at each of three sizes"
        ),
        "spread": (
            "the plug-in standard error divided by the empirical spread of the estimates, "
            "reported without a verdict"
        ),
    }
)


def _adaptive_cells() -> None:
    """The cells of ``adaptive-iptw`` that no other study shares.

    A cell name is ``<key label>__<configuration>``.  The key label is an arm of :data:`ARMS`,
    so :func:`cell` reads the configuration here, and prints the key beside it.  A
    configuration that another study already describes would be overwritten silently, so this
    raises instead.
    """
    positive = "bias interval inside the equivalence margin"
    control = "bias interval must fall entirely outside the margin"
    band = "SE-ratio and coverage intervals both inside their calibration bands"
    flexible = (
        "bias interval inside the equivalence margin; pre-routed to reporting if it is red, "
        "because the learner's rate is not shown"
    )
    table: dict[tuple[str, str], tuple[str, str]] = {}
    for fit, words in (
        ("known_arms", "arm fit"),
        ("known_regimes", "regime fit"),
    ):
        for form, form_words in (("", "targeted"), ("one_step_", "one-step")):
            table[("interval_calibration", f"{form}{fit}__correctly_specified")] = (
                f"the {form_words} {words} with the declared mechanism of the effect-modified "
                f"two-arm law, so the validity term is zero and the interval is published",
                band,
            )
    for cell_name, law in (
        ("known_weighted", "the weighted law"),
        ("known_clustered", "the clustered law with unequal cluster sizes"),
        ("known_stratified", "the stratified law, one target per stratum"),
    ):
        table[("interval_calibration", f"{cell_name}__correctly_specified")] = (
            f"the targeted fit on {law} with its declared mechanism",
            band,
        )
    table[("simultaneous_coverage", "known_band__simultaneous_band")] = (
        "the simultaneous band over the arm means, the effect, the risk ratio and the odds "
        "ratio, with the declared mechanism",
        "the band covers the whole truth vector at the nominal joint rate",
    )
    for bias_cell, words, required in (
        (
            "correct",
            "the correct logistic mechanism, cross-fitted, on the law with a correct reduced regression",
            positive,
        ),
        (
            "rate",
            "a mechanism whose logit is off by 2 n^(-1/3) (1 + W2), cross-fitted, on the law with a correct reduced regression",
            positive,
        ),
        ("insample_correct", "the correct logistic mechanism, fitted in sample", positive),
        (
            "wrong_mechanism",
            "a main-terms logistic mechanism that omits the W1 W2 interaction of the true mechanism",
            control,
        ),
        ("flexible", "a gradient-boosting mechanism, cross-fitted", flexible),
        ("three_arm", "the correct multinomial mechanism on a three-arm law, five folds", positive),
        (
            "rate_reduced",
            "a multinomial mechanism shrunk toward uniform by 2 n^(-1/3), five folds, on a law where the reduced regression is not exact; each arm probability stays a function of the true one, so the level sets survive",
            positive,
        ),
        (
            "composite_l1",
            "the composite mechanism of treatment and response on a finite two-arm law, cross-fitted",
            positive,
        ),
        (
            "composite_l3",
            "the composite mechanism on a finite three-arm law, cross-fitted",
            positive,
        ),
        (
            "insample_composite_l1",
            "the composite mechanism on a finite two-arm law, fitted in sample",
            positive,
        ),
        (
            "insample_composite_l3",
            "the composite mechanism on a finite three-arm law, fitted in sample",
            positive,
        ),
        (
            "weighted",
            "an estimated mechanism on the weighted law, on the draws of the known-mechanism cell",
            positive,
        ),
        (
            "clustered",
            "an estimated mechanism on the clustered law, on the draws of the known-mechanism cell",
            positive,
        ),
        (
            "stratified",
            "an estimated mechanism on the stratified law, on the draws of the known-mechanism cell",
            positive,
        ),
        (
            "regime",
            "the regime fit with the mechanism logit scaled by 1 + 2 n^(-1/3), on the effect-modified law, where the scaling keeps the level sets",
            positive,
        ),
        (
            "natural_course",
            "the natural-course fit with a response mechanism off by a multiple of n^(-1/3)",
            positive,
        ),
        (
            "insample_regime_missing",
            "the regime fit with a missing outcome, fitted in sample",
            positive,
        ),
        (
            "insample_missing_treatment",
            "the arm fit with a missing treatment and a missing outcome, fitted in sample",
            positive,
        ),
        (
            "insample_regime_missing_treatment",
            "the regime fit with a missing treatment and a missing outcome, fitted in sample",
            positive,
        ),
        (
            "missing_treatment_crossfit",
            "the arm fit with a missing treatment and a missing outcome, cross-fitted",
            positive,
        ),
        (
            "level_set_breaking",
            "a mechanism whose error moves across the level sets of the true mechanism (it adds n^(-1/3) W2 to the logit), on a law where the reduced regression is not exact",
            control,
        ),
        (
            "level_set_breaking_h1",
            "the same mechanism error on a law where the reduced regression is exact",
            positive,
        ),
        (
            "parametric_q_ne_qr",
            "the correct logistic mechanism on a law where the reduced regression is not exact",
            positive,
        ),
    ):
        table[("bias", bias_cell)] = (words, required)
    for rung in (500, 1_500, 4_500):
        table[("root_n_points", f"n_{rung}")] = (
            f"the average treatment effect at n = {rung:,} under a mechanism off by a multiple "
            f"of n^(-1/3)",
            positive,
        )
    table[("spread", "parametric_q_ne_qr")] = (
        "the plug-in standard error over the empirical spread of the estimates, for the correct "
        "logistic mechanism on a law where the reduced regression is not exact",
        "reported only: the ratio and its interval, with no verdict",
    )
    for key, value in table.items():
        if key in CELLS:
            raise ValueError(f"{key} is already described")
        CELLS[key] = value
    for label in ("p_ey0", "p_ey1", "p_ate", "pd_d", "pd_s", "cn_ey_obs", "rm_d"):
        STUDY_ARM_CELLS[("adaptive-iptw", "targeting_necessity", label, "targeted")] = (
            "the targeted estimate of the fit",
            positive,
        )
        STUDY_ARM_CELLS[("adaptive-iptw", "targeting_necessity", label, "untargeted")] = (
            "the Horvitz-Thompson mean of the same fit at the estimated mechanism, before the "
            "targeting step, on the same draws",
            "the displacement from the targeted estimate must reach one standard deviation of "
            "the targeted estimate, and the bias interval must fall entirely outside the margin",
        )


_adaptive_cells()


def _refuter_calibration_cells() -> None:
    """The longitudinal refutation study (X82 SL): its laws, families and 40 cells."""
    slug = "refuter-calibration-longitudinal"
    STUDY_SCENARIOS[(slug, "end_of_study")] = (
        "law E, two nodes, a Gaussian outcome and a time-varying confounder; `SDR` at five folds"
    )
    STUDY_SCENARIOS[(slug, "survival")] = (
        "law S, the registered censored survival law, risk at t = 2; `LTMLE` at five folds"
    )
    IMPLEMENTATIONS["cleverly-refute-longitudinal"] = (
        "`cleverly` fit that the longitudinal refutation rules read"
    )
    PROPERTIES["refuter_size"] = (
        "under its null a refutation rule rejects in no more than 17 of 200 trials"
    )
    PROPERTIES["refuter_power"] = (
        "a refutation rule detects a declared defect in at least 38 of 50 trials"
    )
    fits = {
        "L1": "LTMLE, law S",
        "S1": "SDR, law E",
        "I1": "IPSI at odds x2, law E",
        "B1": "SequentialGComputation, law E",
        "B2": "SequentialIPW, law E",
        "B3": "ParametricGFormula, law E",
        "SN1": "StructuralNestedMean, law E",
    }
    size = "no more than 17 rejections in 200 trials"
    power = "at least 38 detections in 50 trials"
    readings = {
        "placebo": "the placebo t rule against the exact Markov null",
        "subset": "the subset rho rule, both tails",
        "negative_control": "the negative-control z rule on a valid control",
    }
    defects = {
        "placebo_stale_rows": "the placebo t rule, with a declared share of stale rows",
        "subset_half_se": "the subset rho rule, at the reported SE halved",
        "subset_double_se": "the subset rho rule, at the reported SE doubled",
        "negative_control_confounded": (
            "the negative-control z rule, on a control that reads an unmeasured cause of A1"
        ),
    }
    table: dict[tuple[str, str], tuple[str, str]] = {}
    for cell, fit in fits.items():
        for reading, words in readings.items():
            if cell == "L1" and reading == "negative_control":
                continue
            table[("refuter_size", f"{cell}__{reading}")] = (f"{fit}: {words}", size)
        for defect, words in defects.items():
            if cell == "L1" and defect.startswith("negative_control"):
                continue
            table[("refuter_power", f"{cell}__{defect}")] = (f"{fit}: {words}", power)
    for key, value in table.items():
        if key in CELLS:
            raise ValueError(f"{key} is already described")
        CELLS[key] = value


_refuter_calibration_cells()


# ------------------------------------------------------------------ X82 SG limit coverage

#: The rows of ``ovb-limit-coverage`` and ``evalue-limit-coverage``.  Mirrored from
#: ``tests.studies.ovb_limit_coverage.CELLS`` and ``tests.studies.evalue_limit_coverage``, which
#: this module cannot import; ``tests/unit/test_sensitivity_limit_coverage_registry.py`` pins
#: that every published key resolves.
IMPLEMENTATIONS["cleverly-ovb-limit"] = "`cleverly` omitted-variable bound limits (X82 part G)"
IMPLEMENTATIONS["cleverly-evalue-limit"] = "`cleverly` risk-ratio interval behind the E-value limit"
STUDY_SCENARIOS.update(
    {
        (_SG_OVB, "O1"): "part G law, outcome missing at random given A and W; saturated GLMs",
        (_SG_OVB, "O2"): "part G law with a binary intermediate Z; saturated GLMs",
        (_SG_OVB, "O3"): "part G law, complete data, three regimes; saturated GLMs",
        (_SG_OVB, "O4"): "part G law, complete data, linear MSM with modifier W2; saturated GLMs",
        (_SG_OVB, "O5"): "`tests.discrete_law_shift` on its four dose atoms; saturated GLMs",
        (_SG_EVALUE, "E3"): "`stratified_law`, three strata, as `canonical-stratified-tmle` fits",
        (_SG_EVALUE, "E4"): "the `survival` law of `point-treatment-survival`, hazard LTMLE",
    }
)
STUDY_ESTIMANDS.update({(_SG_OVB, row): words for row, (_, words) in _SG_OVB_ROWS.items()})
STUDY_ESTIMANDS[_SG_EVALUE, "rr_t5"] = _SG_EVALUE_ROWS["E4_rr_t5"]
STUDY_ARM_CELLS.update(
    {
        (slug, "interval_calibration", label, kind): words
        for slug, labels in (
            (_SG_OVB, [f"{cell}_{row}" for row, (cell, _) in _SG_OVB_ROWS.items()]),
            (_SG_EVALUE, list(_SG_EVALUE_ROWS)),
        )
        for label in labels
        for kind, words in (
            (
                "correctly_specified",
                (
                    "every nuisance is correctly specified, and the row reads its reported limit",
                    "SE ratio and coverage intervals both inside their calibration bands",
                ),
            ),
            (
                "inflated_se_control",
                (
                    "the reported standard error is multiplied by 1.5",
                    "the SE-ratio interval must fall above the calibration band",
                ),
            ),
        )
    }
)
PROPERTIES["response_mask_necessity"] = (
    "the response mask in sigma^2 is what puts the response-mechanism bound at its truth"
)
CELLS[("response_mask_necessity", "masked")] = (
    "the package bound, whose sigma^2 is the masked mean over all rows",
    "bias interval inside the equivalence margin",
)
CELLS[("response_mask_necessity", "unmasked_sigma2")] = (
    "sigma^2 read as the mean over the observed rows only",
    "bias interval must fall entirely outside the margin",
)


def _refuter_point_cells() -> None:
    """The point-treatment refutation study (X82 SP): its laws and 27 cells."""
    slug = "refuter-calibration-point"
    STUDY_SCENARIOS[(slug, "gaussian")] = (
        "the point law with 20 covariates and a Gaussian outcome; TMLE at five folds"
    )
    STUDY_SCENARIOS[(slug, "binomial")] = (
        "the point law with 20 covariates and a binary outcome of prevalence 0.100; "
        "TMLE at five folds"
    )
    IMPLEMENTATIONS["cleverly-refute-point"] = (
        "`cleverly` fit that the point-treatment refutation rules read"
    )
    fits = {
        "T1": "TMLE with GLM nuisances, Gaussian outcome",
        "T2": "TMLE with GLM nuisances, binary outcome",
        "T3": "TMLE with a depth-2 booster, binary outcome",
        "C1": "greedy C-TMLE, binary outcome",
        "D1": "DR-TMLE with a declared mechanism, binary outcome",
        "D2": "DR-TMLE with an estimated mechanism, binary outcome",
    }
    readings = {
        "T1": ("placebo", "subset", "negative_control"),
        "T2": ("placebo", "subset", "negative_control"),
        "T3": ("placebo",),
        "C1": ("placebo",),
        "D1": ("placebo", "subset", "negative_control"),
        "D2": ("placebo",),
    }
    size = "no more than 17 rejections in 200 trials"
    power = "at least 38 detections in 50 trials"
    words = {
        "placebo": "the placebo t rule against the exact permutation null",
        "subset": "the subset rho rule, both tails",
        "negative_control": "the negative-control z rule on a valid control",
    }
    defects = {
        "placebo": {
            "placebo_stale_rows": "the placebo t rule, with a declared share of stale rows"
        },
        "subset": {
            "subset_half_se": "the subset rho rule, at the reported SE halved",
            "subset_double_se": "the subset rho rule, at the reported SE doubled",
        },
        "negative_control": {
            "negative_control_confounded": (
                "the negative-control z rule, on a control that reads an unmeasured cause of A"
            )
        },
    }
    table: dict[tuple[str, str], tuple[str, str]] = {}
    for cell, fit in fits.items():
        for reading in readings[cell]:
            table[("refuter_size", f"{cell}__{reading}")] = (f"{fit}: {words[reading]}", size)
            for defect, text in defects[reading].items():
                table[("refuter_power", f"{cell}__{defect}")] = (f"{fit}: {text}", power)
    for key, value in table.items():
        if key in CELLS:
            raise ValueError(f"{key} is already described")
        CELLS[key] = value


_refuter_point_cells()


def _refuter_band_cells() -> None:
    """The band-rank placebo study (X82 SJ): its point law and four cells."""
    slug = "refuter-calibration-band"
    STUDY_SCENARIOS[(slug, "point")] = (
        "law P: a continuous dose in a band [L(W), U(W)] whose location and width read W; "
        "TMLE of a capped +0.5 shift at five folds"
    )
    IMPLEMENTATIONS["cleverly-refute-band"] = "`cleverly` fit that the band-rank placebo rule reads"
    fits = {
        "PJ1": "TMLE of a capped dose shift on law P",
        "LJ1": "LTMLE of a dose shift at three nodes on the pricing-shaped law L",
    }
    size = "no more than 20 rejections in 200 trials"
    power = "at least 38 detections in 50 trials"
    for cell, fit in fits.items():
        for key, value in {
            ("refuter_size", f"{cell}__placebo"): (
                f"{fit}: the placebo t rule against the exact band-rank null",
                size,
            ),
            ("refuter_power", f"{cell}__placebo_stale_rows"): (
                f"{fit}: the placebo t rule, with a declared share of stale rows",
                power,
            ),
        }.items():
            if key in CELLS:
                raise ValueError(f"{key} is already described")
            CELLS[key] = value


_refuter_band_cells()


# ------------------------------------------------------------------ X82 SH limit coverage


def _marginal_bound_cells() -> None:
    """The marginal sensitivity bound limit study (X82 SH): two laws, two ends each.

    Mirrored from ``tests.studies.marginal_bound_coverage``, which this module cannot
    import; ``tests/unit/test_marginal_bound_limit_registry.py`` pins that every published
    key resolves.
    """
    slug = "marginal-bound-limit-coverage"
    IMPLEMENTATIONS["cleverly-msm-bound-limit"] = (
        "`cleverly` marginal sensitivity bound limits (X82 part H)"
    )
    STUDY_SCENARIOS.update(
        {
            (slug, "SH1"): "Gaussian linear outcome, 20 covariates, logistic propensity; GLMs",
            (slug, "SH2"): "logistic outcome, prevalence near 0.33, 20 covariates; GLMs",
        }
    )
    ends = {
        end: f"{end} end of the ATE bound under the marginal sensitivity model at Gamma 1.5"
        for end in ("lower", "upper")
    }
    rows = {f"ate_{end}": words for end, words in ends.items()}
    STUDY_ESTIMANDS.update({(slug, row): words for row, words in rows.items()})
    labels = {f"{cell}_{row}" for cell in ("SH1", "SH2") for row in rows}
    kinds = {
        "correctly_specified": (
            "every nuisance is correctly specified, and the row reads its reported standard error",
            "SE ratio and coverage intervals both inside their calibration bands",
        ),
        "inflated_se_control": (
            "the reported standard error is multiplied by 1.5",
            "the SE-ratio interval must fall above the calibration band",
        ),
        "wrong_gamma_control": (
            "the end at Gamma 1.25, read against the Gamma 1.5 truth",
            "the coverage interval must fall below the calibration band",
        ),
    }
    STUDY_ARM_CELLS.update(
        {
            (slug, "interval_calibration", label, kind): words
            for label in labels
            for kind, words in kinds.items()
        }
    )


_marginal_bound_cells()


# ------------------------------------------------------------------------------ long horizon
#: The three long-horizon studies (X39).  Their keys follow one grammar, so the description of
#: an estimand and of a cell is composed from its parts instead of listed row by row.
LONG_HORIZON_SLUGS = frozenset(
    {"long-horizon-ltmle", "long-horizon-categorical-ltmle", "long-horizon-sdr"}
)
_LH_PLANS = {
    "never": "never treat",
    "always": "treat at every decision",
    "low": "assign the lowest arm at every decision",
    "high": "assign the highest arm at every decision",
    "respond": (
        "assign the highest arm where the covariate rule is on (W of at least 2 at the first "
        "decision, L_t = 1 afterwards) and the lowest arm otherwise"
    ),
    "switch": "assign the highest arm at every decision but the last, and the lowest arm at the last",
    "coin": "draw the arm at each decision from a known policy that reads the covariate",
    "coinlast": (
        "draw the arm from the known policy `coin` at every decision but the last, and assign "
        "the lowest arm at the last"
    ),
    "mix": (
        "draw the arm at each decision from a known policy that reads the covariate and the "
        "previous arm"
    ),
    "mixlast": (
        "draw the arm from the known policy `mix` at every decision but the last, and assign "
        "the lowest arm at the last"
    ),
    "taper": (
        "assign the highest arm at the first decision, then draw from the known policy `mix`"
    ),
}
_LH_ESTIMAND = re.compile(
    r"^(?P<name>ey_regimen|ate_regimen|msm_regimen)\[(?P<argument>.+?)\](?P<bca>__bca)?$"
)
_LH_BCA = ", read from the BCa interval of the refit bootstrap at B = 199"


def _long_horizon_plan(label: str) -> str:
    try:
        return _LH_PLANS[label]
    except KeyError:
        raise Undescribed(f"no description for regimen {label!r}") from None


def _long_horizon_estimand(key: str) -> str:
    """The parameter of a long-horizon row, with the interval kind of a ``__bca`` row."""
    match = _LH_ESTIMAND.match(key)
    if match is None:
        raise Undescribed(f"no description for estimand {key!r}")
    name, argument = match.group("name"), match.group("argument")
    if name == "msm_regimen":
        try:
            term = TERMS[argument]
        except KeyError:
            raise Undescribed(f"no description for MSM term {argument!r}") from None
        words = f"working-model coefficient of treatment duration over the horizon: the {term}"
    elif name == "ey_regimen":
        words = f"counterfactual mean outcome under the plan: {_long_horizon_plan(argument)}"
    else:
        first, _, second = argument.partition(" vs ")
        words = (
            "difference in counterfactual mean outcome between the plans "
            f'"{_long_horizon_plan(first)}" against "{_long_horizon_plan(second)}"'
        )
    return words + (_LH_BCA if match.group("bca") else "")


_LH_TARGETS = {
    "respond_vs_ref": (
        "the contrast of the covariate rule `respond` against the reference plan of the lowest arm"
    ),
    "policy": "the mean under the known stochastic policy",
    "policy_vs_ref": "the contrast of the known stochastic policy against the lowest-arm plan",
    "policy_vs_policylast": (
        "the paired contrast of the known stochastic policy against the same policy that "
        "assigns the lowest arm at the last decision"
    ),
    "static_contrast": "the contrast of the highest-arm plan against the lowest-arm plan",
    "msm_duration": "the duration coefficient of the working model (identity link, unit weights)",
    "respond": "the mean under the covariate rule `respond`",
    "static": "the mean under the static plan",
}
_LH_ARMS = {"binary": "two arms", "categorical": "three arms"}
_LH_BIAS_POSITIVE = "bias interval inside the equivalence margin"
_LH_BIAS_CONTROL = "bias interval must fall entirely outside the margin"
_LH_REPORTED = "reported: bias, SE ratio and coverage, with no verdict"
_LH_CALIBRATION = "SE ratio and coverage intervals both inside their calibration bands"
_LH_FITS = re.compile(
    r"^(?:(?P<arms>binary|categorical)_)?(?P<what>[a-z_]+?)__k(?P<K>\d)(?P<xf>_crossfit)?(?:__(?P<rest>.+))?$"
)


def _long_horizon_target(what: str) -> str:
    try:
        return _LH_TARGETS[what]
    except KeyError:
        raise Undescribed(f"no description for long-horizon target {what!r}") from None


def _long_horizon_cell(family: str, key: str, role: str | None, slug: str) -> tuple[str, str]:
    """What a long-horizon property cell configures, and what its verdict requires."""
    undescribed = Undescribed(f"no description for cell {key!r} of {family!r}")
    if family == "sequential_double_robustness":
        patterns = {
            "P_odd": "the mechanisms are correct at decisions 1, 3 and 5 and the outcome regressions at 2 and 4",
            "P_even": "the mechanisms are correct at decisions 2 and 4 and the outcome regressions at 1, 3 and 5",
        }
        stem, _, size = key.partition("__")
        if stem in patterns:
            tested = f"five decisions, three arms, the mean under `respond`: {patterns[stem]}"
            if size == "n4000":
                return f"{tested}, at n = 4,000", _LH_REPORTED
            if size == "":
                return tested, _LH_BIAS_POSITIVE
        if key == "odd_nodes_both_wrong_control":
            return (
                "pattern P_odd with the mechanism also wrong at decisions 1, 3 and 5, so no "
                "decision has a correct nuisance",
                _LH_BIAS_CONTROL,
            )
        if key == "ltmle_P_odd_control":
            return (
                "`LTMLE` on pattern P_odd, which is not sequentially doubly robust",
                _LH_BIAS_CONTROL,
            )
        raise undescribed
    if family == "simultaneous_coverage":
        band = (
            "the studentized bootstrap sup-t band at B = 199"
            if slug == "long-horizon-categorical-ltmle"
            else "the Wald-curve band of the influence curves"
        )
        if key == "k5__simultaneous_band":
            return (
                f"{band} over the six plans of the five-decision fit, read as one joint set",
                "joint coverage interval inside the calibration coverage band",
            )
        if key == "k5__ic_band":
            return (
                "the Wald-curve band of the influence curves, from the same fit as the bootstrap band",
                _LH_REPORTED,
            )
        if key == "k5__pointwise_joint_control":
            return (
                "the pointwise 95% intervals of the same fits, read as one joint set",
                "joint coverage upper endpoint must fall below the nominal rate",
            )
        raise undescribed
    match = _LH_FITS.match(key)
    if match is None:
        raise undescribed
    what, K, rest = match.group("what"), int(match.group("K")), match.group("rest") or ""
    crossfit = ", cross-fitted over five folds" if match.group("xf") else ""
    arms = _LH_ARMS.get(match.group("arms") or "", "")
    arms = f" with {arms}" if arms else ""
    where = f"{K} decisions{arms}{crossfit}"
    if family == "interval_calibration":
        target = _long_horizon_target(what)
        controls = {
            "known_mechanism": (
                f"{target} at {where}, with the declared mechanism passed as known probabilities",
                _LH_CALIBRATION,
            ),
            "shrunken_se_control": (
                f"{target} at {where}: the reported standard errors are multiplied by a "
                "declared factor below one",
                "the SE-ratio interval must fall below the calibration band",
            ),
            "noise_control": (
                f"{target} at {where}: a declared scale of independent noise is added to each "
                "estimate",
                "the SE-ratio interval must fall below the calibration band",
            ),
            "inflated_se_control": (
                f"{target} at {where}: the standard error comes from the two plan variances "
                "alone, which is too wide for a paired contrast",
                "the SE-ratio interval must fall above the calibration band",
            ),
        }
        if rest not in controls:
            raise undescribed
        return controls[rest]
    if family == "root_n_and_efficiency":
        size = re.fullmatch(r"n_(\d+)", rest)
        if size is None:
            raise undescribed
        tested = (
            f"{_long_horizon_target(what)} at {where}: bias, coverage and SE calibration at "
            f"n = {int(size.group(1)):,}"
        )
        if role == "control":
            return tested, "coverage interval lies below nominal or clears the declared floor"
        return (
            tested,
            "bias inside the margin, coverage clears the floor, SE ratio inside the sanity band",
        )
    if family == "root_n_rate":
        rates = {
            "empirical_sd": "log empirical spread of the estimates regressed on log n across three sizes",
            "reported_se": "the same regression applied to the mean reported standard error",
        }
        if rest not in rates:
            raise undescribed
        return (
            f"{_long_horizon_target(what)} at {where}: {rates[rest]}",
            "slope interval inside the root-n band and excluding -1/4",
        )
    if family == "double_robustness":
        arms_words = {
            "both_correct": "both the outcome regression and the treatment mechanism are correctly specified",
            "outcome_correct": "only the outcome regression is correctly specified",
            "mechanism_correct": "only the treatment and censoring mechanisms are correctly specified",
            "both_wrong": "both nuisances are misspecified",
        }
        if rest not in arms_words:
            raise undescribed
        required = _LH_BIAS_CONTROL if rest == "both_wrong" else _LH_BIAS_POSITIVE
        return f"{_long_horizon_target(what)} at {where}: {arms_words[rest]}", required
    if family == "targeting_necessity":
        words = {
            "targeted": "the estimator fluctuates a misspecified outcome regression, so targeting does all the adjusting",
            "untargeted": "the identical fit with every fluctuation step removed",
        }
        if rest not in words:
            raise undescribed
        required = _LH_BIAS_CONTROL if rest == "untargeted" else _LH_BIAS_POSITIVE
        return f"{_long_horizon_target(what)} at {where}: {words[rest]}", required
    if family == "censoring_necessity":
        words = {
            "declared": "the censoring mechanism is the declared one",
            "ignored": "the same fit sets the retention probability to one, so it ignores censoring",
        }
        if rest not in words:
            raise undescribed
        required = _LH_BIAS_CONTROL if rest == "ignored" else _LH_BIAS_POSITIVE
        return f"{_long_horizon_target(what)} at {where}: {words[rest]}", required
    if family == "categorical_probability_necessity":
        words = {
            "assigned_probability": "the clever covariate selects the assigned third arm's own probability",
            "binary_complement": "the same fit replaces the third arm's probability with a binary complement",
        }
        if what != "third_arm" or rest not in words:
            raise undescribed
        required = _LH_BIAS_CONTROL if rest == "binary_complement" else _LH_BIAS_POSITIVE
        return f"{words[rest]}, at {where}", required
    if family == "rule_necessity":
        words = {
            "declared_rule": "the declared covariate rule assigns the highest arm where it is on",
            "reversed_rule": "the same fit reverses the rule's two assignments",
        }
        if rest not in words:
            raise undescribed
        required = _LH_BIAS_CONTROL if rest == "reversed_rule" else _LH_BIAS_POSITIVE
        return f"{_long_horizon_target(what)} at {where}: {words[rest]}", required
    if family == "policy_necessity":
        words = {
            "declared_policy": "the recursion integrates over the declared policy at each decision",
            "mix_reversed_control": "the same fit reverses the arm probabilities of the policy at every decision",
            "mix_node1_control": "the same fit uses the policy of the first decision at every decision",
        }
        if rest not in words:
            raise undescribed
        required = _LH_BIAS_POSITIVE if rest == "declared_policy" else _LH_BIAS_CONTROL
        return f"{_long_horizon_target(what)} at {where}: {words[rest]}", required
    if family == "type_i_error":
        if rest != "sharp_null":
            raise undescribed
        return (
            f"{_long_horizon_target(what)} at {where}, on a law whose outcome does not read "
            "any arm: the Wald test of the contrast",
            "rejection interval below the type-I ceiling, and coverage above the floor",
        )
    if family == "power":
        if rest != "alternative":
            raise undescribed
        return (
            f"{_long_horizon_target(what)} at {where}: the same test on a law with a real effect",
            "rejection lower bound clears the minimum power",
        )
    if family == "crossfit_overfitting":
        words = {
            "cross_fitted": (
                "five-fold fit with fully grown outcome trees",
                "SE ratio clears the overfitting floor and stays inside the sanity band",
            ),
            "in_sample_control": (
                "the same flexible learner fitted in sample, with no cross-fitting",
                "SE ratio must fall below the overfitting ceiling",
            ),
        }
        if rest not in words:
            raise undescribed
        tested, required = words[rest]
        return f"{_long_horizon_target(what)} at {where}: {tested}", required
    if family == "support_loss":
        size = re.fullmatch(r"n(\d+)__(default_bounds|no_bounds)", rest)
        if size is None:
            raise undescribed
        bounds = (
            "the default mechanism bounds of 0.01"
            if size.group(2) == "default_bounds"
            else "no mechanism bounds"
        )
        return (
            f"{_long_horizon_target(what)} at {where}, n = {int(size.group(1)):,}, with {bounds}",
            _LH_REPORTED,
        )
    if family == "learned_mechanism":
        if rest != "learned":
            raise undescribed
        return (
            f"{_long_horizon_target(what)} at {where}: the treatment mechanism is learned from "
            "the data instead of declared",
            _LH_REPORTED,
        )
    raise undescribed


def _long_horizon_study_text() -> None:
    """The implementation, scenario and property-family descriptions of the three studies."""
    for slug, name in (
        ("long-horizon-ltmle", "cleverly-long-horizon-ltmle"),
        ("long-horizon-categorical-ltmle", "cleverly-long-horizon-categorical-ltmle"),
        ("long-horizon-sdr", "cleverly-long-horizon-sdr"),
    ):
        estimator = "SDR" if slug.endswith("sdr") else "LTMLE"
        IMPLEMENTATIONS[name] = (
            f"`cleverly` `{estimator}` at three to five decisions, with the declared mechanism"
        )
    base = "law L_R (B = 1, s = 0.1, c0 = 3.5)"
    scenarios = {
        "long-horizon-ltmle": {
            "binary_k3": f"three binary decisions on {base}",
            "binary_k4": f"four binary decisions on {base}",
            "binary_k5": f"five binary decisions on {base}",
            "binary_k5_msm": (
                f"five binary decisions on {base}, with the working model of treatment duration"
            ),
        },
        "long-horizon-categorical-ltmle": {
            "categorical_k3": f"three decisions with three arms on {base}",
            "categorical_k4": f"four decisions with three arms on {base}",
            "categorical_k4_high": "four decisions with three arms on law (B, s, c0) = (0.75, 0.1, 3.5)",
            "categorical_k4_static": "four decisions with three arms on law (B, s, c0) = (1, 0, 4.5)",
            "categorical_k5": f"five decisions with three arms on {base}",
            "categorical_k5_crossfit": (
                f"five decisions with three arms on {base}, cross-fitted over five folds"
            ),
            "categorical_k5_static": "five decisions with three arms on law (B, s, c0) = (0.5, 0.05, 3.5)",
            "categorical_k5_switch": "five decisions with three arms on law (B, s, c0) = (0.5, 0.1, 3.5)",
            "categorical_k5_high": "five decisions with three arms on law (B, s, c0) = (0.25, 0, 3.5)",
            "categorical_k5_mix": "five decisions with three arms on law (B, s, c0) = (1, 0.05, 3.5)",
        },
        "long-horizon-sdr": {
            "sdr_binary_k3": f"three binary decisions on {base}",
            "sdr_binary_k4": f"four binary decisions on {base}",
            "sdr_binary_k5": f"five binary decisions on {base}",
            "sdr_categorical_k5": f"five decisions with three arms on {base}",
            "sdr_categorical_k5_crossfit": (
                f"five decisions with three arms on {base}, cross-fitted over five folds"
            ),
            "sdr_categorical_k5_static": "five decisions with three arms on law (B, s, c0) = (0.5, 0.05, 3.5)",
            "sdr_categorical_k5_switch": "five decisions with three arms on law (B, s, c0) = (0.5, 0.1, 3.5)",
            "sdr_categorical_k5_high": "five decisions with three arms on law (B, s, c0) = (0.25, 0, 3.5)",
            "sdr_categorical_k5_mix": "five decisions with three arms on law (B, s, c0) = (1, 0.05, 3.5)",
            "sdr_policy_k5": f"five decisions with three arms on {base}, with known stochastic policies",
            "sdr_policy_k5_mix": "five decisions with three arms on law (B, s, c0) = (1, 0.05, 3.5), with known stochastic policies",
            "sdr_policy_k5_taper": "five decisions with three arms on law (B, s, c0) = (1, 0, 3.5), with known stochastic policies",
        },
    }
    for slug, table in scenarios.items():
        STUDY_SCENARIOS.update({(slug, name): words for name, words in table.items()})
    PROPERTIES.update(
        {
            "censoring_necessity": (
                "the declared censoring mechanism, rather than a mechanism that ignores "
                "censoring, determines the target"
            ),
            "support_loss": (
                "what the default mechanism bounds do to the estimate when the plan has "
                "little support, reported rather than gated"
            ),
            "learned_mechanism": (
                "the same contrast with a learned treatment mechanism, reported rather than gated"
            ),
        }
    )


_long_horizon_study_text()


#: ``learned-rule-compositions`` (X11 PR A): the learned-rule value with fold targeting, known
#: weights and clusters. The laws are those of ``tests/studies/_learned_rule_compositions_law.py``.
_COMPOSITIONS_SLUG = "learned-rule-compositions"
IMPLEMENTATIONS["cleverly-learned-rule-compositions"] = (
    "`cleverly` fold-evaluated CV-TMLE of the learned-rule value, with fold targeting, known "
    "weights or clusters"
)
STUDY_SCENARIOS.update(
    {
        (_COMPOSITIONS_SLUG, "fold_targeted"): (
            "learned-rule law with the blip 0.1 + W1, fold targeting"
        ),
        (_COMPOSITIONS_SLUG, "weighted"): (
            "learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight "
            "1 + 1.5 * 1{W1 > 0}, a misspecified outcome learner"
        ),
        (_COMPOSITIONS_SLUG, "clustered"): (
            "learned-rule law with the blip 0.1 + W1 and a cluster effect of SD 1.5 on the logit "
            "scale, 200 clusters of 10 rows"
        ),
    }
)
STUDY_ARMS.update(
    {
        (_COMPOSITIONS_SLUG, "weighted"): (
            "the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight "
            "1 + 1.5 * 1{W1 > 0}"
        ),
        (_COMPOSITIONS_SLUG, "clustered"): (
            "the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters "
            "of 10 rows"
        ),
    }
)
STUDY_ARM_CELLS.update(
    {
        (_COMPOSITIONS_SLUG, "interval_calibration", label, "declared"): (
            "the declared fit with the study's logistic learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in ("fold_targeted", "weighted", "clustered", "clustered_few")
    }
)
STUDY_ARM_CELLS[
    (_COMPOSITIONS_SLUG, "interval_calibration", "clustered", "row_variance_control")
] = (
    "the same estimate with the row-level standard error that ignores the clusters",
    "the SE-ratio interval must fall below the calibration band",
)


#: ``learned-rule-split-mean`` (X11 PR B): the learned-rule value as the mean over five splits.
#: It reads the shared RM30 laws, so only its implementation, its estimand and its
#: independent-splits control need words.
_SPLIT_MEAN_SLUG = "learned-rule-split-mean"
STUDY_ARMS[_SPLIT_MEAN_SLUG, "non_exceptional"] = (
    "the learned-rule law with the blip 0.1 + W1, whose logistic outcome learner is correct"
)
IMPLEMENTATIONS["cleverly-learned-rule-split-mean"] = (
    "`cleverly` fold-evaluated CV-TMLE of the learned-rule value, as the mean over five "
    "independent ten-fold splits"
)
STUDY_ESTIMANDS[_SPLIT_MEAN_SLUG, "ey_learned_rule[learned rule]"] = (
    "mean over five splits of the fold average of the value of the rule learned on each "
    "fold's training rows"
)
STUDY_ARM_CELLS.update(
    {
        (_SPLIT_MEAN_SLUG, "interval_calibration", label, "declared"): (
            "the declared fit with five splits and the study's logistic learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in ("non_exceptional", "misspecified_limit")
    }
)
STUDY_ARM_CELLS.update(
    {
        (_SPLIT_MEAN_SLUG, "interval_calibration", label, "independent_splits_control"): (
            "the same estimate with the mean single-split standard error over the square root "
            "of five, which treats the five splits as independent samples",
            "the SE-ratio interval must fall below the calibration band",
        )
        for label in ("non_exceptional", "misspecified_limit")
    }
)


#: ``learned-rule-learners`` (X11 PR C): the learned-rule value when a blip or a
#: weighted-classification rule learner sets each fold's rule.
_LEARNERS_SLUG = "learned-rule-learners"
IMPLEMENTATIONS["cleverly-learned-rule-learners"] = (
    "`cleverly` fold-evaluated CV-TMLE of the learned-rule value, with a rule learner fitted to "
    "the blip pseudo-outcome on each fold's training rows"
)
_LEARNERS_LAW = "the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose logistic outcome learner has no W1^2 term"
STUDY_ARMS.update(
    {
        (_LEARNERS_SLUG, "blip"): (
            f"{_LEARNERS_LAW}, with `LearnedRule(loss='blip')` on a quadratic basis and linear "
            "regression"
        ),
        (_LEARNERS_SLUG, "weighted_classification"): (
            f"{_LEARNERS_LAW}, with `LearnedRule(loss='weighted_classification')` on a "
            "quadratic basis and logistic regression"
        ),
    }
)
STUDY_SCENARIOS.update(
    {
        (_LEARNERS_SLUG, "blip"): (
            "learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner, "
            "and a blip rule learner"
        ),
        (_LEARNERS_SLUG, "weighted_classification"): (
            "learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, a misspecified outcome learner, "
            "and a weighted-classification rule learner"
        ),
    }
)
STUDY_ESTIMANDS[_LEARNERS_SLUG, "ey_learned_rule[learned rule]"] = (
    "fold average of the value of the rule that a rule learner fits on each fold's training rows"
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNERS_SLUG, "interval_calibration", label, "declared"): (
            "the declared fit with the study's rule learner and logistic nuisance learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in ("blip", "weighted_classification")
    }
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNERS_SLUG, "interval_calibration", label, "validation_rule_control"): (
            "the same sample and folds with each fold's rule learner fitted on that fold's own "
            "validation rows, so the rule has seen the rows that evaluate it",
            "the coverage interval must fall below the calibration band",
        )
        for label in ("blip", "weighted_classification")
    }
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNERS_SLUG, "small_sample_diagnostic", label, "n1000"): (
            "the declared fit at n = 1,000, reported without a verdict",
            "none. The cell reports the SE ratio, coverage and bias at the smaller size",
        )
        for label in ("blip", "weighted_classification")
    }
)


#: ``fold-evaluated-strata`` (X11 PR E1, X28): fold-evaluated CV-TMLE with baseline strata.
_STRATA_SLUG = "fold-evaluated-strata"
IMPLEMENTATIONS["cleverly-fold-evaluated-strata"] = (
    "`cleverly` fold-evaluated CV-TMLE with baseline strata: each stratum estimate is the fold "
    "average of the stratum plug-ins, and the marginal is their share mixture"
)
_STRATA_LAW = (
    "W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5), logit g0 = 0.3 W1 - 0.2 W2, and three strata cut "
    "from W1 at shares 0.2, 0.35 and 0.45"
)
STUDY_SCENARIOS.update(
    {
        (_STRATA_SLUG, "unweighted"): (
            f"{_STRATA_LAW}; a beta outcome with the blip 0.1 + 1.5 W1, rows only"
        ),
        (_STRATA_SLUG, "weighted"): (
            f"{_STRATA_LAW}; a beta outcome with the blip 0.1 + 1.5 W1 and the known weight "
            "1 + 1.5 W2"
        ),
        (_STRATA_SLUG, "binary_unweighted"): (
            f"{_STRATA_LAW}; a binary outcome with the blip 0.1 - 0.3 W1, rows only"
        ),
        (_STRATA_SLUG, "binary_weighted"): (
            f"{_STRATA_LAW}; a binary outcome with the blip 0.1 - 0.3 W1 and the known weight "
            "1 + 1.5 W2"
        ),
    }
)
_STRATA_CALIBRATION = [
    f"{setting}_{target}_{part}"
    for setting in ("unweighted", "weighted", "binary_unweighted", "binary_weighted")
    for target in _STRATA_TARGETS
    for part in _STRATA_PARTS
]
_STRATA_BELOW = "the SE-ratio interval must fall below the calibration band"
STUDY_ARM_CELLS.update(
    {
        (_STRATA_SLUG, "interval_calibration", label, "declared"): (
            "the declared fold-evaluated stratified fit with correctly specified learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in _STRATA_CALIBRATION
    }
)
STUDY_ARM_CELLS.update(
    {
        (_STRATA_SLUG, "interval_calibration", label, "unnormalised_curve_control"): (
            "the same fit with each fold's stratum curve not divided by the fold's own stratum "
            "share, so its SE is about the stratum share times the reported one",
            _STRATA_BELOW,
        )
        for label in _STRATA_CALIBRATION
        if not label.endswith("_all")
    }
)
STUDY_ARM_CELLS.update(
    {
        (_STRATA_SLUG, "interval_calibration", label, "share_dropped_control"): (
            "the same fit with the marginal curve missing its share term, the spread of the "
            "stratum estimates over the strata",
            _STRATA_BELOW,
        )
        for label in (
            "unweighted_ate_all",
            "unweighted_ey1_all",
            "weighted_ate_all",
            "weighted_ey1_all",
        )
    }
)
_STRATA_NECESSITY = [
    f"unweighted_{target}_{part}" for target in _STRATA_TARGETS for part in ("s0", "s1", "s2")
]
STUDY_ARM_CELLS.update(
    {
        (_STRATA_SLUG, "targeting_necessity", label, "targeted"): (
            "the fit with an outcome learner that sees neither W1 nor the stratum, so its "
            "initial plug-in is nearly the same in every stratum and targeting does the "
            "stratum adjusting",
            "bias interval inside the equivalence margin",
        )
        for label in _STRATA_NECESSITY
    }
)
STUDY_ARM_CELLS.update(
    {
        (_STRATA_SLUG, "targeting_necessity", label, "untargeted"): (
            "the same fit without its fluctuation: the fold average of the stratum-blind "
            "initial stratum plug-ins",
            "bias interval must fall entirely outside the margin",
        )
        for label in _STRATA_NECESSITY
    }
)
STUDY_ARM_CELLS.update(
    {
        (_STRATA_SLUG, "sparse_event_diagnostic", f"binary_sparse_{target}_{part}", "diagnostic"): (
            "the declared fit and the stacked stratified fit (cv_evaluation=False) on the same "
            "samples and folds at n = 1,000, reported without a verdict",
            "none. The cell reports both fits' coverage and SE ratio and the stratum-arm event "
            "count",
        )
        for target in _STRATA_TARGETS
        for part in _STRATA_PARTS
    }
)


#: ``learned-rule-strata`` (X11 PR E2, X11 (m)): the learned-rule value inside baseline strata.
_LEARNED_STRATA_SLUG = "learned-rule-strata"
IMPLEMENTATIONS["cleverly-learned-rule-strata"] = (
    "`cleverly` fold-evaluated learned-rule value with baseline strata: each stratum estimate "
    "is the fold average of the stratum plug-ins at the fold's learned rule, and the marginal "
    "is their share mixture"
)
STUDY_SCENARIOS.update(
    {
        (_LEARNED_STRATA_SLUG, "unweighted"): (
            f"{_STRATA_LAW}; a beta outcome with the blip 0.1 + 1.5 W1, rows only"
        ),
        (_LEARNED_STRATA_SLUG, "weighted"): (
            f"{_STRATA_LAW}; a beta outcome with the blip 0.1 + 1.5 W1 and the known weight "
            "1 + 1.5 W2"
        ),
        (_LEARNED_STRATA_SLUG, "binary_steep"): (
            f"{_STRATA_LAW}; a binary outcome with the blip -1.2 + 3.0 W1, rows only"
        ),
        (_LEARNED_STRATA_SLUG, "binary_steep_weighted"): (
            f"{_STRATA_LAW}; a binary outcome with the blip -1.2 + 3.0 W1 and the known weight "
            "1 + 1.5 W2"
        ),
    }
)
_LEARNED_STRATA_CALIBRATION = [
    f"{setting}_{part}"
    for setting in ("unweighted", "weighted", "binary_steep", "binary_steep_weighted")
    for part in _STRATA_PARTS
]
STUDY_ARM_CELLS.update(
    {
        (_LEARNED_STRATA_SLUG, "interval_calibration", label, "declared"): (
            "the declared stratified learned-rule fit with correctly specified learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in _LEARNED_STRATA_CALIBRATION
    }
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNED_STRATA_SLUG, "interval_calibration", label, "unnormalised_curve_control"): (
            "the same fit with each fold's stratum curve not divided by the fold's own stratum "
            "share, so its SE is about the stratum share times the reported one",
            _STRATA_BELOW,
        )
        for label in _LEARNED_STRATA_CALIBRATION
        if not label.endswith("_all")
    }
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNED_STRATA_SLUG, "interval_calibration", label, "share_dropped_control"): (
            "the same fit with the marginal curve missing its share term, the spread of the "
            "stratum estimates over the strata",
            _STRATA_BELOW,
        )
        for label in ("unweighted_all", "weighted_all")
    }
)
_LEARNED_STRATA_NECESSITY = [f"unweighted_{part}" for part in ("s0", "s1", "s2")]
STUDY_ARM_CELLS.update(
    {
        (_LEARNED_STRATA_SLUG, "targeting_necessity", label, "targeted"): (
            "the fit with an outcome learner that sees neither W1 nor the stratum, so its "
            "initial plug-in is nearly the same in every stratum and its rule reads W2 alone; "
            "the truth is the harness refit with that learner",
            "bias interval inside the equivalence margin",
        )
        for label in _LEARNED_STRATA_NECESSITY
    }
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNED_STRATA_SLUG, "targeting_necessity", label, "untargeted"): (
            "the same fit without its fluctuation: the fold average of the stratum-blind "
            "initial stratum plug-ins at the fit's rule",
            "bias interval must fall entirely outside the margin",
        )
        for label in _LEARNED_STRATA_NECESSITY
    }
)
STUDY_ARM_CELLS.update(
    {
        (_LEARNED_STRATA_SLUG, "weak_margin_diagnostic", f"{setting}_{part}", "diagnostic"): (
            "the declared fit and the same fit without strata on the same samples and folds at "
            "n = 1,000, reported without a verdict",
            "none. The cell reports both fits' coverage and SE ratio and the boundary "
            "stratum's fold-rule disagreement",
        )
        for setting in ("binary_weak", "binary_moderate")
        for part in _STRATA_PARTS
    }
)


#: ``learned-rule-stacked`` (X11 PR F1, X11 (e)): the stacked report of the learned-rule value.
#: The laws are those of ``tests/studies/_learned_rule_stacked_law.py``.
_STACKED_SLUG = "learned-rule-stacked"
IMPLEMENTATIONS["cleverly-learned-rule-stacked"] = (
    "`cleverly` stacked report of the learned-rule value: one pooled fluctuation over every "
    "validation row, each fold weighted by its share of the weight mass, and the curve centred "
    "at the pooled estimate"
)
STUDY_ESTIMANDS[_STACKED_SLUG, "ey_learned_rule[learned rule]"] = (
    "fold-mass-weighted average of the value of the rule learned on each fold's training rows"
)
STUDY_SCENARIOS.update(
    {
        (_STACKED_SLUG, "rows"): (
            "learned-rule law with the blip 0.1 + W1, rows only, ten equal folds"
        ),
        (_STACKED_SLUG, "weighted"): (
            "learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight "
            "1 + 1.5 * 1{W1 > 0}, a misspecified outcome learner"
        ),
        (_STACKED_SLUG, "clustered"): (
            "learned-rule law with the blip 0.1 + W1 and a cluster effect of SD 1.0 on the logit "
            "scale, 200 clusters of size 5 + Binomial(10, expit(1.5 z)) that rises with the "
            "effect"
        ),
    }
)
STUDY_ARMS.update(
    {
        (_STACKED_SLUG, "weighted"): (
            "the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3 and the known weight "
            "1 + 1.5 * 1{W1 > 0}"
        ),
        (_STACKED_SLUG, "clustered"): (
            "the learned-rule law with the blip 0.1 + W1 and a cluster effect, in 200 clusters "
            "whose size rises with the effect"
        ),
    }
)
STUDY_ARM_CELLS.update(
    {
        (_STACKED_SLUG, "interval_calibration", label, "declared"): (
            "the declared stacked fit with the study's logistic learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in ("rows", "weighted", "clustered")
    }
)
STUDY_ARM_CELLS[(_STACKED_SLUG, "interval_calibration", "clustered", "row_variance_control")] = (
    "the same estimate with the row-level centred standard error that ignores the clusters",
    "the SE-ratio interval must fall below the calibration band",
)
STUDY_ARM_CELLS[(_STACKED_SLUG, "fold_evaluated_reporting", "clustered", "fold_evaluated")] = (
    "the fold-evaluated fit (cv_evaluation=True) on every clustered calibration draw, with the "
    "same folds and rules, read against the 1/V average of the same fold values",
    "none. The cell reports the fold-evaluated bias beside the stacked bias and their paired "
    "error difference",
)


#: ``learned-rule-in-sample`` (X11 PR F2, X11 (a)): one rule learned on every row and its value
#: estimated on the same rows (``cross_fit=False``). The laws are RM30's; the learners are those
#: of ``tests/studies/_learned_rule_in_sample_law.py``.
_IN_SAMPLE_SLUG = "learned-rule-in-sample"
IMPLEMENTATIONS["cleverly-learned-rule-in-sample"] = (
    "`cleverly` in-sample TMLE of the learned-rule value: one plug-in rule learned from the "
    "outcome regression fitted on every row, and one regime fluctuation at that rule on the "
    "same rows"
)
STUDY_ESTIMANDS[_IN_SAMPLE_SLUG, "ey_learned_rule[learned rule]"] = (
    "value of the one rule learned on every row"
)
STUDY_SCENARIOS.update(
    {
        (_IN_SAMPLE_SLUG, "non_exceptional"): (
            "learned-rule law with the blip 0.1 + W1, a correct interaction logistic outcome "
            "learner"
        ),
        (_IN_SAMPLE_SLUG, "misspecified_limit"): (
            "learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, an interaction logistic "
            "outcome learner with no W1^2 term"
        ),
    }
)
STUDY_ARMS.update(
    {
        (_IN_SAMPLE_SLUG, "non_exceptional"): (
            "the learned-rule law with the blip 0.1 + W1, whose interaction logistic outcome "
            "learner is correct"
        ),
        (_IN_SAMPLE_SLUG, "misspecified_limit"): (
            "the learned-rule law with the blip 0.8 W1 + W1^2 - 0.3, whose interaction "
            "logistic outcome learner has no W1^2 term"
        ),
    }
)
STUDY_ARM_CELLS.update(
    {
        (_IN_SAMPLE_SLUG, "interval_calibration", label, "declared"): (
            "the declared in-sample fit with the study's parametric logistic learners",
            "SE ratio and coverage intervals both inside their calibration bands",
        )
        for label in ("non_exceptional", "misspecified_limit")
    }
)
STUDY_ARM_CELLS.update(
    {
        (_IN_SAMPLE_SLUG, "target_distinction", label, "fold_average"): (
            "every declared calibration fit read against the fold average (1/10) sum_v "
            "Psi_{d_v}(P0) of the rules the declared learner learns on ten training folds of "
            "the same draw",
            "none. The cell reports the mean gap between the two targets, its SD over the "
            "error SD, and the share of the grid on which the fold rules differ from the "
            "full-sample rule",
        )
        for label in ("non_exceptional", "misspecified_limit")
    }
)
STUDY_ARM_CELLS[
    (_IN_SAMPLE_SLUG, "interval_calibration", "non_exceptional", "interpolating_control")
] = (
    "an interpolating tree (min_samples_leaf=1, probabilities clipped to [0.01, 0.99]), an "
    "adaptive outcome learner fitted in sample on the same draw and read against the value of "
    "its own rule",
    "the SE-ratio interval must fall below the calibration band",
)
STUDY_ARM_CELLS.update(
    {
        (_IN_SAMPLE_SLUG, "targeting_necessity", "non_exceptional", "targeted"): (
            "the in-sample fit with WithoutW1MainLogistic, a parametric outcome learner with "
            "no W1 main term, so targeting does the adjusting",
            "bias interval inside the equivalence margin",
        ),
        (_IN_SAMPLE_SLUG, "targeting_necessity", "non_exceptional", "untargeted"): (
            "the same fit's initial plug-in mean at its own rule, with no fluctuation",
            "bias interval must fall entirely outside the margin",
        ),
    }
)
#: ``non_exceptional_drop_w1`` is no arm of :data:`ARMS`, so these cells are read whole.
CELLS.update(
    {
        ("weak_limit_necessity", "non_exceptional_drop_w1__targeted"): (
            "the learned-rule law with the blip 0.1 + W1: the in-sample fit with RM30's "
            "drop_w1 outcome learner, whose limit benefit (0.037 and 0.049) is too weak at "
            "n = 2,000 for its rule to settle",
            "none. The cell reports the bias, the share of draws whose rule flipped, and the "
            "bias on settled and on flipped draws",
        ),
        ("weak_limit_necessity", "non_exceptional_drop_w1__untargeted"): (
            "the learned-rule law with the blip 0.1 + W1: the same drop_w1 fit's initial "
            "plug-in mean at its own rule, with no fluctuation",
            "none. The cell reports the bias and the displacement from the targeted arm",
        ),
    }
)
