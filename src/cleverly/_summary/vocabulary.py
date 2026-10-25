"""Every word a result summary shares with another summary.

A summary is printed by several modules, and two modules that spell one fact two ways
teach a reader that they are two facts. Each label, column name, method name, learner
name, note code and evidence phrase that more than one summary can print lives here once.
A seam that needs a new shared word adds it here, with its test, in its own change.

Nothing here is public. Users meet these strings in printed summaries only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

# ----------------------------------------------------------------- header labels

QUESTION: Final = "Question"
OUTCOME: Final = "Outcome"
ADJUSTED_FOR: Final = "Adjusted for"
METHOD: Final = "Method"
ROWS: Final = "Rows"
#: The unit count of a fit read from a long panel, which has one row per unit and time.
UNITS: Final = "Units"
TREATED_SHARE: Final = "Treated share"
ARM_SHARES: Final = "Arm shares"
DOSE: Final = "Dose"
OUT_OF_FOLD: Final = "Out-of-fold predictions"
OUTCOME_MODEL: Final = "Outcome model"
TREATMENT_MODEL: Final = "Treatment model"
PROBABILITY_LIMITS: Final = "Probability limits"
MISSINGNESS_MODEL: Final = "Missingness model"
#: Appended to a share label on weighted data, whose shares the weights tilt.
WEIGHTED: Final = " (weighted)"
#: The model of whether a row's treatment was recorded.
RECORDING_MODEL: Final = "Recording model"
#: The model of the intermediate variable a controlled direct effect holds fixed.
INTERMEDIATE_MODEL: Final = "Intermediate model"
CLUSTERS: Final = "Clusters"
FEWEST_IN_STRATUM: Final = "Fewest in a stratum"
WEIGHTS: Final = "Weights"
#: Foreseen for the longitudinal and sensitivity summaries.
TIME_POINTS: Final = "Time points"
HORIZON: Final = "Horizon"
CENSORING_MODEL: Final = "Censoring model"
#: The censoring indicators of a longitudinal fit, and what every plan does with them.
CENSORING: Final = "Censoring"
ANALYSIS: Final = "Analysis"
#: Shared with the study-setup summaries (X89 T5).
DESIGN: Final = "Design"
TREATMENT: Final = "Treatment"
STRATA: Final = "Strata"
#: The longitudinal and structural summaries (X89 T2).
COVARIATE_MODELS: Final = "Covariate models"
CONTRASTS: Final = "Contrasts"
FOLD_SPLITS: Final = "Fold splits"
TEST: Final = "Test"
DRAWS: Final = "Draws per unit"
DERIVATIVE_DRAWS: Final = "Draws per unit (derivatives)"
MODELS: Final = "Models"
ALL_PARAMETRIC: Final = "All models parametric"
REGIMES: Final = "Regimes"
LEAST_SUPPORTED: Final = "Least supported"
POLICY: Final = "Policy"
MOVED: Final = "Rows moved"
CAPPED: Final = "Rows held by the cap"
SMALLEST_DENSITY: Final = "Smallest density"
FACTORS: Final = "Factors"
WEIGHT_BOUND: Final = "Weight bound"
SMALLEST_PROBABILITY: Final = "Smallest probability"

#: Every header label above, so a test can check a summary uses only shared labels.
HEADER_LABELS: frozenset[str] = frozenset(
    {
        QUESTION,
        UNITS,
        CENSORING,
        OUTCOME,
        ADJUSTED_FOR,
        METHOD,
        ROWS,
        TREATED_SHARE,
        ARM_SHARES,
        DOSE,
        OUT_OF_FOLD,
        OUTCOME_MODEL,
        TREATMENT_MODEL,
        PROBABILITY_LIMITS,
        MISSINGNESS_MODEL,
        CLUSTERS,
        FEWEST_IN_STRATUM,
        WEIGHTS,
        TIME_POINTS,
        HORIZON,
        CENSORING_MODEL,
        ANALYSIS,
        DESIGN,
        TREATMENT,
        STRATA,
        COVARIATE_MODELS,
        CONTRASTS,
        FOLD_SPLITS,
        TEST,
        DRAWS,
        DERIVATIVE_DRAWS,
        MODELS,
        ALL_PARAMETRIC,
        REGIMES,
        LEAST_SUPPORTED,
        POLICY,
        MOVED,
        CAPPED,
        SMALLEST_DENSITY,
        FACTORS,
        WEIGHT_BOUND,
        SMALLEST_PROBABILITY,
    }
)

# ----------------------------------------------------------------- column labels

ESTIMATE: Final = "estimate"
P_VALUE: Final = "p-value"
DF: Final = "df"
SUPPORT: Final = "support"
EXPERIMENTAL: Final = "experimental"
MEANING: Final = "meaning"
BAND: Final = "band"
#: The suffix a row label takes when its estimate supplies no inference in a table whose
#: other rows do (docs/architecture-invariants.md, inference status).
NO_INFERENCE_MARK: Final = "*"
#: The cell a table prints where a value does not apply.
MISSING: Final = "-"


def interval_label(level: float) -> str:
    """The interval column label for a confidence level such as ``0.95``.

    Parameters
    ----------
    level : float
        The confidence level, between 0 and 1.

    Returns
    -------
    str
        For example ``"95% CI"``.
    """
    return f"{level * 100:g}% CI"


#: Plain column labels for the spread names :func:`~cleverly.inference.influence.spread_name`,
#: ``spread_columns()`` and :func:`~cleverly.inference.influence.bootstrap_column` publish.
#: A diagnostic name says so in its label, so no diagnostic reads as a standard error.
SPREAD_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "std_err": "std. error",
        "plugin_std_err": "plug-in SE (diagnostic)",
        "ci_lower": "CI lower",
        "ci_upper": "CI upper",
        "plugin_interval_lower": "plug-in range lower (diagnostic)",
        "plugin_interval_upper": "plug-in range upper (diagnostic)",
        "plugin_interval": "plug-in range (diagnostic)",
        "p_value": "p-value",
        "plugin_p_value": "plug-in p-value (diagnostic)",
        "bootstrap se": "bootstrap SE",
        "bootstrap sd": "bootstrap SD (diagnostic)",
        "bootstrap_se": "bootstrap SE",
        "bootstrap_sd": "bootstrap SD (diagnostic)",
        "bootstrap_ci_lower": "bootstrap CI lower",
        "bootstrap_ci_upper": "bootstrap CI upper",
        "bootstrap_range_lower": "bootstrap range lower (diagnostic)",
        "bootstrap_range_upper": "bootstrap range upper (diagnostic)",
        "robustness_value_ci": "limit robustness value",
        "rva": "limit robustness value",
        "robustness_value_plugin_interval": "plug-in limit robustness value (diagnostic)",
        "rv_plugin_interval": "plug-in limit robustness value (diagnostic)",
    }
)

# ----------------------------------------------------------------- methods

#: The plain name of each fitted method, keyed by ``fitted_method``.
METHOD_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "targeted maximum likelihood (TMLE)",
        "collaborative_tmle": "collaborative TMLE (C-TMLE)",
        "drtmle": "doubly robust TMLE (DR-TMLE)",
        "adaptive_iptw": "adaptive-propensity inverse probability weighting",
        "sdr": "sequentially doubly robust regression (SDR)",
        "ipsi": "incremental propensity score intervention (IPSI)",
        "gcomp": "sequential g-computation",
        "ipw": "sequential inverse probability weighting",
        "parametric_gformula": "parametric g-formula",
        "riesz_tmle": "TMLE with a learned Riesz representer",
        "ep": "efficient plug-in learner",
    }
)

#: The short name a summary title carries, keyed by ``fitted_method``.
METHOD_SHORT_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "TMLE",
        "collaborative_tmle": "C-TMLE",
        "drtmle": "DR-TMLE",
        "adaptive_iptw": "adaptive IPTW",
    }
)


def method_name(fitted_method: str) -> str:
    """The plain method name, with TMLE for a method this table does not list yet."""
    return METHOD_NAMES.get(fitted_method, METHOD_NAMES["tmle"])


def method_short_name(fitted_method: str) -> str:
    """The short method name, with TMLE for a method this table does not list yet."""
    return METHOD_SHORT_NAMES.get(fitted_method, METHOD_SHORT_NAMES["tmle"])


#: The title of a point-treatment estimate summary, filled with the short method name.
ESTIMATE_TITLE: Final = "Effect estimate ({method})"

# ----------------------------------------------------------------- learners

#: Plain learner names, keyed by class name.
LEARNER_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "LinearRegression": "linear regression",
        "LogisticRegression": "logistic regression",
        "LogisticRegressionCV": "logistic regression (tuned penalty)",
        "Ridge": "ridge regression",
        "RidgeCV": "ridge regression (tuned penalty)",
        "Lasso": "lasso",
        "LassoCV": "lasso",
        "ElasticNet": "elastic net",
        "ElasticNetCV": "elastic net",
        "PoissonRegressor": "Poisson regression",
        "GammaRegressor": "gamma regression",
        "TweedieRegressor": "Tweedie regression",
        "GLM": "generalized linear model",
        "ParametricOutcome": "parametric logistic model",
        "ParametricMechanism": "parametric logistic model",
        "L1Logistic": "lasso logistic regression",
        "RandomForestRegressor": "random forest",
        "RandomForestClassifier": "random forest",
        "ExtraTreesRegressor": "extra trees",
        "ExtraTreesClassifier": "extra trees",
        "HistGradientBoostingRegressor": "gradient boosting",
        "HistGradientBoostingClassifier": "gradient boosting",
        "GradientBoostingRegressor": "gradient boosting",
        "GradientBoostingClassifier": "gradient boosting",
        "KNeighborsRegressor": "nearest neighbours",
        "KNeighborsClassifier": "nearest neighbours",
        "DecisionTreeRegressor": "decision tree",
        "DecisionTreeClassifier": "decision tree",
        "StandardScaler": "scaling",
        "PolynomialFeatures": "polynomial terms",
        "SplineTransformer": "splines",
        "OneHotEncoder": "one-hot coding",
        "MLPRegressor": "neural network",
        "MLPClassifier": "neural network",
    }
)

#: The name of the learner a fit uses when the caller passes none.
DEFAULT_LEARNER: Final = "super learner (default library)"
#: The name of an ensemble, filled with its candidate count.
SUPER_LEARNER: Final = "super learner ({k} candidates)"
#: The name of a model the data declares known rather than one the fit estimated.
KNOWN_MECHANISM: Final = "known (declared)"

#: The learner classes whose in-sample fit cannot make an interval too narrow through
#: overfitting: fixed-dimension parametric models, matched by exact class name. Seeded from
#: ``PER_ARM_PARAMETRIC_OPTIONS`` (``ParametricOutcome``, ``ParametricMechanism``), the only
#: classification the package had. Lasso-type learners select terms from the data and are
#: left out; so is every class not named here, which counts as flexible.
PARAMETRIC_LEARNERS: Final[frozenset[str]] = frozenset(
    {
        "ParametricOutcome",
        "ParametricMechanism",
        "GLM",
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "PoissonRegressor",
        "GammaRegressor",
        "TweedieRegressor",
    }
)

#: Pipeline steps that leave a parametric final step parametric.
PARAMETRIC_STEPS: Final[frozenset[str]] = frozenset({"StandardScaler"})

# ----------------------------------------------------------------- estimands

#: What one row of a point-treatment table estimates, keyed by target name. ``{Y}`` is the
#: outcome, ``{A}`` the treatment, ``{v}`` the row's arm and ``{r}`` the reference arm.
ESTIMAND_MEANINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ate": (
            "average effect: mean {Y} if every unit had {A} = {v}, minus the mean if every "
            "unit had {A} = {r}"
        ),
        "att": (
            "average effect among units that had {A} = {v}: their mean {Y} with {A} = {v}, "
            "minus their mean if they had had {A} = {r}"
        ),
        "atc": (
            "average effect among units that had {A} = {r}: their mean {Y} if they had had "
            "{A} = {v}, minus their mean with {A} = {r}"
        ),
        "ey": "mean {Y} if every unit had {A} = {v}",
        "ey1": "mean {Y} if every unit had {A} = {v}",
        "ey0": "mean {Y} if every unit had {A} = {v}",
        "ey_obs": "mean {Y} with each unit's treatment as received",
        "rr": (
            "risk ratio: mean {Y} if every unit had {A} = {v}, divided by the mean if every "
            "unit had {A} = {r}"
        ),
        "or": (
            "odds ratio: odds of {Y} if every unit had {A} = {v}, divided by the odds if "
            "every unit had {A} = {r}"
        ),
        "par": (
            "mean {Y} with each unit's treatment as received, minus the mean if every unit "
            "had {A} = {r}"
        ),
        "paf": (
            "share of the mean {Y} with treatment as received that would go away if every "
            "unit had {A} = {r}"
        ),
    }
)

#: The Question fact of a single-row table: the meaning's opening words, keyed by target.
QUESTION_PHRASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ate": "average effect of {A} on {Y} ({v} compared with {r})",
        "att": "average effect of {A} on {Y} among units with {A} = {v} ({v} compared with {r})",
        "atc": "average effect of {A} on {Y} among units with {A} = {r} ({v} compared with {r})",
        "ey": "mean {Y} if every unit had {A} = {v}",
        "ey1": "mean {Y} if every unit had {A} = {v}",
        "ey0": "mean {Y} if every unit had {A} = {v}",
        "ey_obs": "mean {Y} with treatment as received",
        "rr": "risk ratio of {Y} for {A} ({v} compared with {r})",
        "or": "odds ratio of {Y} for {A} ({v} compared with {r})",
        "par": "mean {Y} as treated, minus the mean if every unit had {A} = {r}",
        "paf": "share of mean {Y} that would go if every unit had {A} = {r}",
    }
)

#: The Question fact when a table reports several rows; the key table says what each is.
SEVERAL_ROWS_QUESTION: Final = "effect of {A} on {Y}; each row is defined below"
#: The Question fact of a controlled direct effect.
DIRECT_EFFECT_QUESTION: Final = "effect of {A} on {Y} with {Z} held at {z} for every unit"
#: The Question fact of a learned-rule fit, whose target the data chose.
LEARNED_RULE_QUESTION: Final = "value of a treatment rule learned from the data"
#: The title of the key table.
KEY_TITLE: Final = "What each row estimates"
#: A per-stratum row: the row's meaning restricted to the units of one baseline stratum.
STRATUM_MEANING: Final = "{meaning}, among units with {stratum}"

# ----------------------------------------------------------------- notes

NoteKind = Literal["warning", "note"]

#: Every note code, with its kind. A ``warning`` changes how to read a number; a ``note``
#: explains one. A summary may only print a code listed here.
NOTE_CODES: Mapping[str, NoteKind] = MappingProxyType(
    {
        "no_inference": "warning",
        "experimental": "note",
        "in_sample_flexible": "warning",
        "weights_capped": "note",
        "weighted_population": "note",
        "t_reference": "note",
        "spread_scale": "note",
        "repeated_splits": "note",
        "simultaneous_bands": "note",
        "bootstrap_diagnostic": "note",
        "bootstrap_experimental": "note",
        "score_unsolved": "warning",
        "stopping_rule": "note",
        "folds_reduced": "note",
        "fit_warning_positivity": "warning",
        "fit_warning_convergence": "warning",
        "fit_warning_weighting": "warning",
        "fit_warning_sparse_influence": "warning",
        "fit_warning_monte_carlo": "warning",
        "fit_warning_parameter_space": "warning",
        "fit_warning_data": "warning",
        "fit_warning_other": "warning",
        # The longitudinal, structural and support summaries (X89 T2).
        "bootstrap_bands": "note",
        "model_based_interval": "note",
        "working_model": "note",
        "isotonic_moves": "note",
        "model_reliance": "note",
    }
)

#: How a cross-fitted fit built its estimate, from its targeting scheme and evaluation.
CROSS_FIT_CONSTRUCTION: Final = (
    "The models predict each row from the folds that leave it out; {targeting}, and {evaluation}."
)
#: Keyed by whether one targeting step serves every fold (``targeting_scheme="pooled"``).
CROSS_FIT_TARGETING: Final[Mapping[bool, str]] = MappingProxyType(
    {
        True: "one final adjustment of the outcome predictions serves every fold",
        False: "each fold gets its own final adjustment of the outcome predictions",
    }
)
#: Keyed by ``cv_evaluation``.
CROSS_FIT_EVALUATION: Final[Mapping[bool, str]] = MappingProxyType(
    {
        False: "each estimate is computed once on all rows",
        True: "each estimate is the average of its estimates in the held-out folds",
    }
)

#: What a score-check row that belongs to no reported row covers.
SCORE_MODEL_ROW: Final = "the fitted models"

#: Plain names of the axes a support reason says the nearest validated fit differs on, keyed
#: by the names in ``cleverly._support.KEY_AXES``.
SUPPORT_AXES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "kind": "the kind of estimator",
        "engine": "the estimation method",
        "family": "the estimands",
        "outcome": "the outcome type or its declared range",
        "missing": "missing data",
        "treatment": "the treatment type",
        "weighting": "observation weights",
        "splitting": "how the sample is split for the models",
        "dependence": "clustering or the number of clusters",
        "qualifiers": "strata or a known treatment mechanism",
        "options": "estimator options",
    }
)

#: Fit-time warnings a note already states, by that note's code: when the note prints, a
#: recorded warning whose message holds one of these phrases is left out of the Warnings
#: block, and the note prints among the warnings in its place. The phrases are filled with
#: the fit's folds (``declared``, ``used``) and mean treatment bounds (``low``, ``high``),
#: so a warning about other folds or other bounds stays.
RESTATED_WARNINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "folds_reduced": ("reducing n_folds from {declared} to", "to {used}: only"),
        "weights_capped": (
            "treatment probability outside the truncation bounds [{low}, {high}]",
            "values fall below the nuisance bound",
            "Cumulative mechanism truncation exceeded",
        ),
    }
)

#: The note code of a recorded fit-time warning, keyed by the warning class name.
WARNING_NOTE_CODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PositivityWarning": "fit_warning_positivity",
        "ConvergenceWarning": "fit_warning_convergence",
        "WeightingWarning": "fit_warning_weighting",
        "SparseInfluenceWarning": "fit_warning_sparse_influence",
        "MonteCarloWarning": "fit_warning_monte_carlo",
        "ParameterSpaceWarning": "fit_warning_parameter_space",
        "DataWarning": "fit_warning_data",
    }
)

#: Why a row is experimental, by the kind of gap its support reason names
#: (``_estimates.reason_kind``). The full summary prints the reason itself under Settings.
EXPERIMENTAL_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "no_row": (
            "No validation study covers this kind of fit, so cleverly marks {names} experimental."
        ),
        "family": (
            "The validation study of this kind of fit covers other outcome types or another "
            "kind of interval, so cleverly marks {names} experimental."
        ),
        "condition": (
            "This fit has a feature the validation study of its kind does not cover, so "
            "cleverly marks {names} experimental."
        ),
    }
)

#: Why a bootstrap range is a diagnostic, by its source (``bootstrap_is_diagnostic``).
BOOTSTRAP_DIAGNOSTIC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "withheld": (
            "The bootstrap range of {names} is a diagnostic, not a confidence interval, as "
            "the fit withholds that interval."
        ),
        "below_floor": (
            "The bootstrap range of {names} is a diagnostic, not a confidence interval, "
            "because the fit reads fewer than {clusters} clusters, and no validation study "
            "measures a bootstrap interval at so few."
        ),
        "declared": (
            "The bootstrap range of {names} is a diagnostic, not a confidence interval: the "
            "collaborative fit builds its treatment model from the data again on each "
            "resample, and no result shows that this range has its stated coverage."
        ),
    }
)

#: Appended to a quoted warning the fit raised more than once.
REPEATS: Final = " ({count} times)"

NOTE_TEXTS: Mapping[str, str] = MappingProxyType(
    {
        "no_inference": (
            "No confidence interval or p-value for {names}. {reason} The {label} column is "
            "a diagnostic, not a standard error."
        ),
        "no_inference_row": (
            "No confidence interval or p-value for {names}, marked *. {reason} Their "
            "{label}: {spreads}."
        ),
        "experimental": (
            "The numbers are computed as for a validated fit; read {what} with more caution. "
            "The full summary gives the reason."
        ),
        "in_sample_flexible": (
            "The models were fitted and evaluated on the same rows. With flexible learners "
            "this can make the interval too narrow; cross-fitting (cross_fit=True, or "
            "CrossFitting() in a study method) gives out-of-fold predictions."
        ),
        "weights_capped": (
            "cleverly limited the estimated treatment probabilities to [{low}, {high}] for "
            "{count} of {n} rows{split} ({reads})."
        ),
        "weights_capped_missingness": (
            "cleverly raised {share} of the estimated probabilities {what}, counted over "
            "every arm, to {bound}."
        ),
        "weights_capped_effect": (
            "Each limit caps the weight of the rows it moves. That steadies the estimate, "
            "and it can move the estimate if the true probabilities lie outside the limits. "
            "result.diagnostics.truncation_curve() shows the estimate under other limits."
        ),
        "weighted_population": (
            "The estimate describes the population that the observation weights represent, "
            "not the sample as drawn. result.data.weight_report() describes the weights."
        ),
        "t_reference": (
            "A row based on fewer than {threshold} clusters with positive weight mass, in "
            "the fit or in its stratum, takes its interval and p-value from a t distribution "
            "with the degrees of freedom in the df column, which widens them; a row marked "
            "normal uses the normal distribution. The full summary gives the rule and its "
            "sources."
        ),
        "spread_scale": (
            "The std. error of {names} is on the log scale, where the interval is computed "
            "before it is turned back to the scale of the estimate."
        ),
        "spread_scale_transform": (
            "The std. error of {names} is on the transformed scale, where the interval is "
            "computed before it is turned back to the scale of the estimate."
        ),
        "repeated_splits": (
            "Each row is the median over {repeats} random splits into folds, and its "
            "{spread} includes the spread between splits. Rows are combined one at a time, "
            "so a difference of two rows need not equal a third."
        ),
        "simultaneous_bands": (
            "The bands cover all {count} rows at once with {level} probability; use them "
            "when you read several rows together."
        ),
        "score_unsolved": (
            "The fit did not solve its estimating equation for {names} within tolerance, so "
            "the {spread} may be wrong. result.diagnostics.score_equations() gives the "
            "numbers."
        ),
        "score_rule_unmet": (
            "The fit did not meet its declared stopping rule for {names}, so the {spread} "
            "may be off by more than the rule allows. result.diagnostics.score_equations() "
            "gives the numbers."
        ),
        "stopping_rule_plain": (
            "The fit stopped at its declared stopping rule, which leaves the estimating "
            "equation solved up to the remainder the rule allows."
        ),
        "stopping_rule": (
            "Under the declared stopping rule the estimating equation is solved up to a "
            "remainder worth {b} standard errors of bias (on {name}). If the standard error "
            "is right, a {level} interval with that bias still covers the truth at least "
            "{bound} of the time."
        ),
        "bootstrap_experimental": (
            "No validation study covers the bootstrap interval of {names} for this kind of "
            "fit, so cleverly marks it experimental."
        ),
        "bands_experimental": (
            "No validation study covers the bands for this kind of fit, so cleverly marks "
            "them experimental."
        ),
        "bands_leave_out": "The bands leave out {names}, which have no interval.",
        "folds_reduced": (
            "The fit asked for {declared} folds and used {used}, because the rarest group "
            "or the cluster count allowed no more."
        ),
        "fit_warning": "During the fit cleverly warned: {messages}",
        "fit_warning_more": (
            "It also gave {count} more warnings of this kind; result.fit_warnings lists them."
        ),
        # The longitudinal, structural and support summaries (X89 T2).
        "in_sample_flexible_longitudinal": (
            "The models were fitted and evaluated on the same rows. With flexible learners "
            "this can make the interval too narrow; cross-fitting (n_folds of two or more, or "
            "cross_fitting=CrossFitting() on a study) gives out-of-fold predictions."
        ),
        "weights_capped_cumulative": (
            "cleverly limited the estimated probability of following plan {plan} through "
            "time {time} to [{low}, {high}] for {share} of the units still following it "
            "then{split}."
        ),
        "weights_capped_paths": (
            "The probability limits [{low}, {high}] moved {share} of the followed paths of "
            "plan {plan} that start at time {time}{split}."
        ),
        "weights_capped_cumulative_effect": (
            "A limited probability caps the weight of the unit it moves. That steadies the "
            "estimate, and it can move the estimate if the true probabilities lie outside the "
            "limits. result.diagnostics.truncation_curve(bounds) refits under other limits."
        ),
        "score_unsolved_longitudinal": (
            "At some time point the fit did not solve its estimating equation within "
            "tolerance{draw}, so the estimate and its {what} may be off. "
            "result.diagnostics.score_equations() gives each time point."
        ),
        "score_rule_unmet_longitudinal": (
            "The fit did not meet its declared stopping rule{draw}: {misses}. The estimate and "
            "its {what} may be off by more than the rule allows; "
            "result.diagnostics.score_equations() gives each time point."
        ),
        "bootstrap_bands": (
            "The bootstrap bands cover all {count} rows at once with {level} probability. They "
            "come from {replicates} resamples."
        ),
        "model_based_interval_gcomp": (
            "The interval holds only if every outcome model is correctly specified. Under a "
            "wrong model it covers the limit of that model, not the plan mean. LTMLE and SDR "
            "stay consistent when either all outcome models or all treatment and censoring "
            "models are right."
        ),
        "model_based_interval_ipw": (
            "The interval holds only if every treatment and censoring model is correctly "
            "specified. Under a wrong model it covers the limit of that model, not the plan "
            "mean. LTMLE and SDR stay consistent when either all outcome models or all "
            "treatment and censoring models are right."
        ),
        "model_based_interval_gformula": (
            "The interval holds only if every covariate model, outcome (or event-hazard) model "
            "and natural-treatment model the simulation reads is correctly specified. It "
            "includes the simulation error."
        ),
        "known_factors": (
            "The factors declared known ({factors}) use the declared probabilities and need no "
            "model."
        ),
        "working_model": (
            "Each coefficient comes from projecting the plan means onto the working model "
            "{terms} ({link} link). The coefficients are defined whether or not that model "
            "holds, and they are its causal coefficients only when it does."
        ),
        "isotonic_moves": (
            "cleverly made each curve non-decreasing, which moved {count} levels ({names}). "
            "The standard errors are unchanged; an interval at a moved level is valid only "
            "where the true curve strictly increases."
        ),
        "model_reliance": "{statement}",
    }
)

# ----------------------------------------------------------------- full detail

SECTION_ASSUMPTIONS: Final = "Assumptions"
SECTION_SETTINGS: Final = "Settings"
SECTION_METHOD: Final = "Method details"
SECTION_STUDY: Final = "Study description"
ASSUMPTIONS_LEAD: Final = "The estimate equals {functional} when these hold:"
ASSUMPTIONS_LEAD_GENERIC: Final = "The estimate has its causal meaning when these hold:"
#: The conditions a fit with no study adds for a partly recorded outcome.
MISSING_OUTCOME_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "outcome missing at random: whether {Y} is recorded is independent of {Y} given {A} and W",
    "recording positivity: P({Y} recorded | {A}, W) > 0 almost surely",
)
#: The conditions a fit with no study adds for a partly recorded treatment.
MISSING_TREATMENT_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "treatment missing at random: the potential outcome is independent of whether {A} is "
    "recorded given {A} and W",
    "recording positivity: P({A} = a and {A} recorded | W) > 0 almost surely for every arm a",
)
#: The conditions a fit with no study adds when it holds an intermediate at one value.
INTERMEDIATE_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "no unmeasured intermediate confounding: {Y}(a, {z}) is independent of {Z} given {A} and W",
    "intermediate positivity: P({Z} = {z} | {A} = a, W) > 0 almost surely for every arm a "
    "(for a continuous {Z}, a positive density at {z})",
)
SENSITIVITY_POINTER: Final = (
    "result.sensitivity.omitted_confounding() shows how strong an unmeasured cause would "
    "need to be to change the conclusion."
)
NO_PROTOCOL: Final = "No study protocol was recorded."

# ----------------------------------------------------------------- internals

#: Words that name the package's internals rather than the analysis. No summary prints
#: them, at either detail (Eric, 2026-10-24: "dont include library arcana like hashes etc
#: - just the parts relevant to the analysis"). The contract test reads this list.
INTERNAL_TERMS: Final[tuple[str, ...]] = (
    "fluctuation",
    "iterative",
    "pooled",
    "fingerprint",
    "schema",
    "nuisance",
    "registry",
    "rung",
    "cache",
    "targeting",
    "submodel",
    "Levy",
    "run_id",
    "docs/roadmap.md",
    "clever covariate",
    "Kish",
)

# ----------------------------------------------------------------- checks (T3)

#: Header labels of the check and refutation summaries.
RESULT: Final = "Result"
ESTIMATE_TESTED: Final = "Estimate tested"
TESTS_RUN: Final = "Tests run"
REPEATS_PER_TEST: Final = "Repeats per test"
HEADER_LABELS = HEADER_LABELS | {RESULT, ESTIMATE_TESTED, TESTS_RUN, REPEATS_PER_TEST}

#: The notes of a refutation report: one per test, a warning when that test failed.
T3_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "test_placebo": "note",
        "test_random_common_cause": "note",
        "test_subset": "note",
        "test_negative_control_outcome": "note",
        "test_dummy_outcome": "note",
        "test_simulated_outcome": "note",
        "test_bootstrap_measurement_error": "note",
        "test_bootstrap_stability": "note",
        "test_plasmode": "note",
        "test_future_treatment": "note",
        "refit_failures": "warning",
        "refit_spread_diagnostic": "warning",
        "ratio_scale": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_NOTE_CODES})

T3_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "refutation_test": (
            "{name}: {purpose}{conditions} For this kind of fit the test {evidence}."
        ),
        "refit_failures": (
            "{failed} of {total} repeats of the {name} test failed and are left out of its "
            'row; report.draws_frame("{test}") lists why.'
        ),
        "refit_spread_diagnostic": (
            "The refits of {names} have no valid standard error, only a plug-in diagnostic, "
            "so their statistics read the {label}."
        ),
        "ratio_scale": "For a ratio, {parts}.",
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_NOTE_TEXTS})

#: What a check found, in the words every check summary prints for it: in the ``Result``
#: header fact and in a table's ``result`` column. The keys are what the code decides.
CHECK_RESULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": "passed",
        "failed": "failed",
        "descriptive": "no pass or fail",
        "warning": "needs attention",
    }
)

#: Plain evidence words for each refutation evidence level (X89 plan 6.6), as a verb phrase
#: that completes "For this kind of fit the test ...". A pass or fail needs the third or
#: fourth level (``validation/refute.py``).
EVIDENCE_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "L0": "gives a description only, with no pass or fail",
        "L1": (
            "compares the refits with a stated reference, but how often it fails a correct "
            "fit is not measured, so it gives no pass or fail"
        ),
        "L2": (
            "gives a pass or fail backed by an exact reference or by a simulation study of "
            "this kind of fit"
        ),
        "L3": (
            "gives a pass or fail backed the same way, with an external package's result beside it"
        ),
        "withheld": "is studied, but gives no pass or fail here because {reason}",
        "studied": "is studied, but gives no pass or fail on this report",
        "not_run": "gives no pass or fail, because no refit ran",
    }
)

#: Each refutation test by what it does, for the ``test`` column.
REFUTATION_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "placebo": "shuffled treatment",
        "random_common_cause": "added noise column",
        "subset": "random subsamples",
        "negative_control_outcome": "negative-control outcome",
        "dummy_outcome": "noise outcome",
        "simulated_outcome": "simulated outcome",
        "bootstrap_measurement_error": "measurement error",
        "bootstrap_stability": "bootstrap stability",
        "plasmode": "plasmode outcome",
        "future_treatment": "future treatment",
    }
)

#: What cleverly does in each refutation test, the lead of that test's note. ``{null}`` is
#: the value the test expects with no effect, read from the test itself.
REFUTATION_PURPOSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "placebo": (
            "cleverly replaces the treatment with a shuffled or independently drawn copy, so "
            "the treatment has no effect and the estimate should be about {null}."
        ),
        "random_common_cause": (
            "cleverly adds a column of pure noise to the adjustment set, so the estimate "
            "should barely move."
        ),
        "subset": (
            "cleverly refits on random subsamples of the rows. If the standard error is "
            "right, the refits spread about as much as the expected SD."
        ),
        "negative_control_outcome": (
            "cleverly refits with the control outcome you gave, which you assume the "
            "treatment cannot affect, so the estimate should be about {null}."
        ),
        "dummy_outcome": (
            "cleverly replaces the outcome with random noise unrelated to the treatment and "
            "the covariates, so the true effect is known, and checks {readings} against it."
        ),
        "simulated_outcome": (
            "cleverly replaces the outcome with one simulated from the process you declared, "
            "with a known effect, and checks {readings} against it."
        ),
        "plasmode": (
            "cleverly resamples your covariates, draws treatment from a frozen treatment "
            "model and the outcome from the process you declared, with a known effect, and "
            "checks {readings} against it."
        ),
        "bootstrap_measurement_error": (
            "cleverly resamples the rows, adds the measurement error you declared to "
            "{variables}, and refits."
        ),
        "bootstrap_stability": (
            "cleverly resamples units, keeping their folds, and refits, to show how far the "
            "estimate moves."
        ),
        "placebo_not_run": (
            "cleverly would replace the treatment with a shuffled copy, but runs no refit here."
        ),
        "future_treatment": (
            "cleverly estimates the effect of a later treatment on a control you measured "
            "before it, which the later treatment cannot cause, so the estimate should be "
            "about {null}."
        ),
    }
)

#: Plain names of a refutation statistic, keyed by the test's ``statistic_name`` or the
#: label of one of its extra numbers.
STATISTIC_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "t": "t of the refits against {null}",
        "z": "z of the estimate against {null}",
        "scatter ratio rho": "spread ratio (refit SD / expected SD)",
        "paired shift in standard errors": "shift, in std. errors",
        "paired shift in plug-in standard errors": "shift, in plug-in SEs (diagnostic)",
        "mean reported SE over SD of the draws": "mean std. error / refit SD",
        "mean plug-in SE over SD of the draws": "mean plug-in SE (diagnostic) / refit SD",
        "spread of the paired differences (SE units)": "SD of the shifts, in std. errors",
        "spread across seeds with no noise column (SE units)": (
            "SD across seeds without noise, in std. errors"
        ),
        "shift against the original fit": "shift from the original fit",
        "standardized bias": "standardized bias",
        "bias": "std. bias",
        "se_ratio": "SE ratio",
        "coverage": "coverage",
    }
)

#: The reading sentence of a refutation report, by its result.
REFUTATION_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": "Every test that gives a pass or fail passed.",
        "failed": (
            "{names} failed. A failure points to the kind of defect the test is built to "
            "find, but a correct fit also fails by chance, at about the rate its evidence "
            "measured or its exact reference sets; the note for each test says what it "
            "can show."
        ),
        "descriptive_all": (
            "No test failed, and none of the {n} tests gives a pass or fail for this kind of fit, "
            "so compare each mean with the expected column."
        ),
        "descriptive_one": (
            "The test gives no pass or fail for this kind of fit, so compare its mean with "
            "the expected column."
        ),
        "descriptive_both": (
            "No test failed, and neither test gives a pass or fail for this kind of fit, so "
            "compare each mean with the expected column."
        ),
        "descriptive_some": (
            "No test failed. {k} of the {n} tests {verb} no pass or fail for this kind of "
            "fit, so compare their means with the expected column."
        ),
    }
)

#: The outcome processes a generated-outcome test can declare, in plain words, keyed by the
#: process's ``name``.
PROCESS_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "gaussian_independent_noise": "Gaussian noise unrelated to the data",
        "gaussian_adjustment_dependent": "a Gaussian outcome built from your covariates",
        "bernoulli_adjustment_dependent": "a binary outcome built from your covariates",
        "bounded_beta_adjustment_dependent": "a bounded outcome built from your covariates",
        "logistic_hazard": "event times from a logistic hazard",
        "multinomial_hazard": "competing event times from a multinomial hazard",
    }
)

#: What a generated-outcome test reads, by reading name, for its note.
READING_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"bias": "the bias", "se_ratio": "the std. error", "coverage": "the coverage"}
)

#: The two parts of the ``ratio_scale`` note, each naming the tests it applies to.
RATIO_GEOMETRIC: Final = "the mean of the {names} is the geometric mean of the refits"
RATIO_LOG: Final = "the SD of the {names} is on the log scale"

#: How a resampling test drew its rows, by the report's ``resampling``.
REFIT_RESAMPLING_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"iid": "rows resampled one at a time", "cluster": "whole clusters resampled"}
)

#: Section titles of a refutation report's full detail.
SECTION_COMPARED: Final = "What each test was compared with"
SECTION_DRAWS: Final = "Repeats"
REFUTATION_TITLE: Final = "Refutation tests"
STATISTICS_TITLE: Final = "Test statistics"
READINGS_TITLE: Final = "Readings against their allowed bands"

# ------------------------------------------- checks (T3): equations solved, DR-TMLE corrections

#: Header labels of the equation and correction checks.
EQUATIONS: Final = "Equations"
ALLOWED_ERROR: Final = "Allowed error"
PREDICTIONS: Final = "Predictions"
HEADER_LABELS = HEADER_LABELS | {EQUATIONS, ALLOWED_ERROR, PREDICTIONS}

#: The notes of the equation check (``ScoreCheck``) and the correction check
#: (``CorrectionCheck``).
T3_EQUATION_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "identity_failure": "warning",
        "row_not_converged": "warning",
        "equations_only": "note",
        "dr_curve": "note",
        "held_to_nothing": "note",
        "correction_rows": "note",
        "default_bar": "note",
        "correction_scale": "note",
        "theorem_in_sample": "note",
        "theorem_cross_fitted": "note",
        "cap_active": "note",
        "unsolved_guard": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_EQUATION_NOTE_CODES})

T3_EQUATION_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "identity_failure": (
            "{count} of the failed rows {verb} two computations of the same expression, "
            "and they disagree. That is a defect in cleverly, not in your data; please "
            "report it. Running the correction step longer will not fix it."
        ),
        "row_not_converged": "{sentences}",
        "equations_only": (
            "This fit reports no interval, so the check verifies the equations only. {reason}"
        ),
        "dr_curve": (
            "The standard error comes from the doubly robust estimator's own curve. Under "
            "the method's conditions its interval is valid; with one of the two models wrong "
            "it holds only in the cases cleverly.estimators.drtmle lists."
        ),
        "held_to_nothing": "Rows marked described are held to no allowed error: {reasons}",
        "correction_rows": (
            "In the corrections table, a size row holds the mean of a DR-TMLE correction to "
            "its allowed error, and an identity row checks that the value the correction "
            "step solved equals the value the standard error uses."
        ),
        "default_bar": (
            "The declared rule judges only the equations it stopped; {names} {verb} held to "
            "the default allowed error, {allowed}."
        ),
        "correction_scale": (
            "{scale}A correction is allowed an error of {threshold}, and a solved value and "
            "the value the standard error uses must agree to within {identity}."
        ),
        "theorem_in_sample": (
            "No cap binds, so this in-sample fit is the estimator the DR-TMLE theorem "
            "covers, under the theorem's conditions."
        ),
        "theorem_cross_fitted": (
            "No cap binds, so this fit uses the theorem's construction, but the theorem does "
            "not cover cross-fitted (out-of-fold) predictions."
        ),
        "cap_active": (
            "A cap is active ({which}). Simulation studies support such fits, but the "
            "theorem does not cover them. This is not a failure, and the result above does "
            "not read it."
        ),
        "unsolved_guard": (
            "The {equations} {verb} not among the equations this fit's guard= solves, so the "
            "reported standard error does not subtract {pronoun}; {rows} a description, "
            "held to nothing."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_EQUATION_NOTE_TEXTS})

#: Plain names of the equations a fit solves, keyed by the group or estimand name the
#: equation check's rows carry. An estimand alias the caller chose prints as written.
EQUATION_GROUPS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "mean": "outcome means",
        "regime": "rule means",
        "msm": "projection",
        "mtp": "shifted-treatment mean",
    }
)

#: Plain names of the parts of an equation, keyed by the qualifier in a row name.
EQUATION_PARTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "mechanism": "treatment model",
        "reduced": "reduced regression",
        "strata": "strata",
        "D*_Q": "outcome correction",
        "D*_Y": "outcome correction",
        "D*_g": "treatment correction",
        "D*_A": "treatment correction",
        "D*_M": "recording correction",
    }
)

#: What each row of the equation check's corrections table checks, for its ``check``
#: column, keyed by the row's ``kind``.
CORRECTION_ROW_KINDS: Final[Mapping[str, str]] = MappingProxyType(
    {"correction": "size", "identity": "identity", "diagnostic": "described"}
)

#: The ``Allowed error`` fact of the equation check, by the rule the fit declared.
ALLOWED_RELATIVE: Final = "{tolerance:g} x SE / sqrt(n)"
ALLOWED_DECLARED: Final = "{rule} rule"

#: Why a step of the correction stopped short, keyed by the code the step records.
STEP_FAILURES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max_iter_reached": "it reached its iteration limit",
        "nonfinite_step": "a step gave a value that is not finite",
        "line_search_exhausted": "no step size reduced the error",
        "singular_hessian": "the step's curvature matrix was singular",
        "bounds_pinned": "most predictions sat at their bounds",
        "separation_suspected": (
            "its step size grew without limit, the sign of treatment groups that the "
            "covariates separate almost perfectly"
        ),
        "bound_excludes_data": "the outcome bound excludes some of the data",
    }
)

#: The sentences of the ``row_not_converged`` note, one per finding.
ROW_STOPPED: Final = "{name}: the correction step stopped because {reason}."
ROW_FOLDS: Final = "{name}: {failed} of {total} folds did not converge."
ROW_CONDITION: Final = (
    "{name}: the step's curvature matrix has condition number {condition}, so the step "
    "size is barely determined and a solved equation may still be fragile."
)

#: Why a diagnostic row of the equation check is held to nothing.
HELD_UNSOLVED: Final = "{names} {verb} not among the equations this fit's guard= solves"
HELD_BY_RULE: Final = (
    "{names} restate{s} outcome equations that the reduced-regression row judges by the "
    "declared stopping rule"
)

#: The reading sentence of the equation check, by its result.
EQUATION_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": (
            "The correction step solved every estimating equation to within its allowed "
            "error. That is needed for the {spread} to describe this estimate, but it does "
            "not show that it does."
        ),
        "passed_declared": (
            "The correction step met its declared stopping rule on every equation it judges "
            "by that rule; those equations are solved to the rule's bound, not to machine "
            "precision."
        ),
        "failed": (
            "{count} of {total} equations {verb} not solved within {its} allowed error "
            "({names}), so the {spread} may be wrong."
        ),
        "failed_rule": (
            "The correction step did not meet its declared stopping rule on {names}, so the "
            "{spread} may be off by more than the rule allows."
        ),
        "none": "This fit reports no estimating equations.",
        "failed_identity": (
            "Two computations of the same expression disagree ({names}), so the {spread} "
            "does not describe this estimate."
        ),
    }
)

#: Plain names of the caps the DR-TMLE contract reads, keyed by the names
#: ``CorrectionCheck.truncations_active`` returns.
CAP_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "g-hat at the initial fit": "the initial treatment model",
        "pi-hat at the initial fit": "the initial recording model",
        "g* at the exit": "the treatment model at the end",
        "pi* at the exit": "the recording model at the end",
        "g_r1": "the reduced regression",
        "gamma_a / gamma_m": "the reduced regressions",
    }
)

#: The reading sentence of the correction check, by its result.
CORRECTION_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": (
            "Every correction is a negligible mean of the same expression the correction "
            "step solved."
        ),
        "passed_partial": (
            "Every correction this fit solves is a negligible mean of the same expression "
            "the correction step solved; the others are described only."
        ),
        "none": "This fit reports no corrections.",
        "identity": (
            "{count} identity {checks} failed: a solved value and the value the standard "
            "error uses differ. That is a defect in cleverly, not in your data, and the "
            "standard error does not describe this estimate."
        ),
        "correction": (
            "{count} {corrections} above the allowed error at the fit's final state, so the "
            "standard error may be wrong."
        ),
    }
)

#: The first sentence of the ``correction_scale`` note, when the fit rescaled its outcome.
RESCALED: Final = (
    "Corrections are on the outcome's own scale, {scale} times the scale they were solved on. "
)

#: The ``Predictions`` fact of the correction check.
PREDICTION_WORDS: Final[Mapping[bool, str]] = MappingProxyType(
    {True: "cross-fitted", False: "in-sample"}
)

#: A line of the correction check's full detail, one per split.
CORRECTION_MARGINS: Final = (
    "{split}the cap binds on {clipped} of {n} rows at the end; the treatment model comes "
    "within {margin} of its limits{observation}, and the reduced {regression} within "
    "{reduced}."
)

#: Titles of the equation and correction checks and their full-detail sections.
EQUATION_TITLE: Final = "Check of the estimating equations"
STEP_EQUATIONS_TITLE: Final = "Equations the correction step solves"
CORRECTIONS_TITLE: Final = "DR-TMLE corrections"
ESTIMATE_EQUATIONS_TITLE: Final = "Each estimate's equation"
CORRECTION_TITLE: Final = "Check of the DR-TMLE corrections"
SECTION_DETAILS: Final = "Details"


# -------------------------------------- checks (T3): influence measures, observation weights

#: Header labels of the influence report and the weight report.
ESTIMATES_MEASURED: Final = "Estimates"
WEIGHT_COLUMN: Final = "Weight column"
WEIGHT_KIND: Final = "Weight type"
WEIGHTS_TREATED: Final = "Treated as"
EFFECTIVE_ROWS: Final = "Effective rows"
HEADER_LABELS = HEADER_LABELS | {
    ESTIMATES_MEASURED,
    WEIGHT_COLUMN,
    WEIGHT_KIND,
    WEIGHTS_TREATED,
    EFFECTIVE_ROWS,
}

#: The notes of the influence report (``InfluenceReport``) and the weight report
#: (``WeightReport``).
T3_SPREAD_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "influence_terms": "note",
        "cluster_unit": "note",
        "sparse_warning_pointer": "note",
        "weights_population": "note",
        "weights_fixed": "note",
        "weights_estimated": "note",
        "zero_weights": "note",
        "outcome_adaptive_weighted": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_SPREAD_NOTE_CODES})

#: The sentence on survey designs that both weight notes end with.
_DESIGN_SENTENCE: Final = (
    "Stratification and finite-population corrections are ignored, which is conservative; "
    "a multi-stage design must give its sampling units with id=."
)

T3_SPREAD_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "influence_terms": (
            "Each estimate is an average of one contribution per {unit}. Effective {units} is "
            "how many equal contributions would give the same spread; it is the {units} over "
            "the contributions' kurtosis. Skewness measures their asymmetry, which a "
            "symmetric shape lacks. SE uncertainty "
            "is the relative standard deviation of the variance estimate. Coverage loss is "
            "predicted from the shape for the {level}% interval: positive means the interval "
            "covers less than its level, negative more."
        ),
        "cluster_unit": (
            "The fit has clusters, so each measure reads the sum of each cluster's "
            "contributions and counts clusters, not rows."
        ),
        "sparse_warning_pointer": (
            "The fit's warning names the interval to prefer for this kind of fit."
        ),
        "weights_population": (
            "An estimate on these weights describes the population they represent: each row "
            "counts in proportion to its weight, not once."
        ),
        "weights_fixed": (
            "The standard error treats the weights as fixed known numbers. " + _DESIGN_SENTENCE
        ),
        "weights_estimated": (
            "The weights were declared estimated, so an interval a fit reports treats the "
            "fitted weights as known. For weights fitted by maximum likelihood in a correct "
            "selection model that is conservative for the whole population; for calibrated, "
            "raked or trimmed weights there is no general guarantee. The bootstrap "
            "(n_bootstrap=) does not close the gap: it resamples rows and keeps the weights "
            "it was given. A DRTMLE fit with a non-empty guard, and an AdaptiveIPTW fit with "
            "a known treatment mechanism, show an interval on estimated weights under the "
            "experimental label. " + _DESIGN_SENTENCE
        ),
        "zero_weights": (
            "{count} rows carry zero weight: they do not enter the estimate but count "
            "toward the rows."
        ),
        "outcome_adaptive_weighted": (
            "An outcome-adaptive C-TMLE fit reports no standard error on weighted data. "
            "Under the default per-arm design, its diagnostic standard error reads the curve "
            "of {source}, which is not the efficient one."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_SPREAD_NOTE_TEXTS})

#: The source the ``outcome_adaptive_weighted`` note cites.
CTMLE_CURVE_SOURCE: Final = "Benkeser, Cai and van der Laan (2020), Theorem 1"

#: Titles of the influence report, its table and its full-detail sections.
INFLUENCE_TITLE: Final = "How single {units} move the estimate"
INFLUENCE_DEFINITIONS: Final = "How these are measured"
INFLUENCE_BOOTSTRAP: Final = "Bootstrap shape"
INFLUENCE_KURTOSIS: Final = "Kurtosis of each estimate"

#: A line of the kurtosis section, one per estimate. The table leaves kurtosis out: it is
#: the units over the effective units.
INFLUENCE_KURTOSIS_LINE: Final = "{name}: {kurtosis}"

#: The definitions the influence report's full detail prints, one line each.
INFLUENCE_DEFINITION_LINES: Final[tuple[str, ...]] = (
    "effective {units}: (sum of squared contributions)^2 / sum of fourth powers, which is "
    "the {units} divided by the kurtosis",
    "skewness: third central moment over the second to the power 1.5",
    "kurtosis: fourth central moment over the second squared (3 for a normal shape)",
    "SE uncertainty: sqrt((kurtosis - 1) / {units}), the relative standard deviation of the "
    "variance estimate; the fit warns above {threshold}",
    "coverage loss: the second-order Edgeworth term of the studentized statistic "
    "(Hall 1992), positive for a predicted shortfall",
)

#: A line of the bootstrap section, one per estimate with an attached bootstrap.
INFLUENCE_BOOTSTRAP_LINE: Final = "{name}: bias correction {z0}, acceleration {acceleration}"

#: The flag column of the influence report.
INFLUENCE_FLAG: Final = "noisy"

#: The reading of the influence report.
INFLUENCE_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "none": (
            "No estimate's standard error is itself noisy: every SE uncertainty is at most "
            "{threshold}, the value that triggers the warning."
        ),
        "one": (
            "{names}: the variance estimate is itself noisy (relative SD {value}, above the "
            "{threshold} that triggers the warning), so a Wald interval can miss its level."
        ),
        "several": (
            "{names}: each variance estimate is itself noisy (relative SD up to {value}, "
            "above the {threshold} that triggers the warning), so a Wald interval can miss "
            "its level."
        ),
        "empty": "This fit reports no estimate with a curve, so there is nothing to measure.",
    }
)

#: Titles of the weight report and its full-detail section.
WEIGHT_TITLE: Final = "Observation weights"
WEIGHT_ESTIMAND: Final = "Estimand"

#: The ``Treated as`` fact of the weight report.
WEIGHT_TREATMENT: Final[Mapping[bool, str]] = MappingProxyType({True: "estimated", False: "fixed"})

#: The rows of the weight report's table, in printed order.
WEIGHT_MEASURES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "design_effect": "design effect",
        "min_ratio": "smallest weight / mean",
        "max_ratio": "largest weight / mean",
        "cv": "variation (CV)",
        "top_1pct": "share held by the top 1%",
        "top_5pct": "share held by the top 5%",
        "n_zero": "rows with zero weight",
    }
)

#: The reading of the weight report.
WEIGHT_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "constant": "The weights are constant, so the fit is the unweighted one.",
        "dominant": (
            "The weights dominate: the effective rows are under 1/{limit:g} of the rows, so "
            "estimates rest on a few heavily weighted rows and the interval's large-sample "
            "argument works from that smaller number."
        ),
    }
)

#: The full-detail estimand lines of the weight report.
WEIGHT_ESTIMAND_LINES: Final[tuple[str, ...]] = (
    "Estimand: the requested causal parameter in the weight-tilted population "
    "dP_w = w dP / E[w], estimated on the weighted sample (weighted model fits, a weighted "
    "correction step and a weighted plug-in).",
    "Standard errors: the influence curve (w / E[w]) times the efficient influence function "
    "of that parameter, which is its efficient influence function when the weights are "
    "observed data.",
)


# ------------------------------------------- checks (T3): outcome and treatment models

#: Header labels of the model check.
SPLIT_SHOWN: Final = "Split shown"
HEADER_LABELS = HEADER_LABELS | {SPLIT_SHOWN}

#: The notes of the model check (``NuisanceDiagnostics``). Each finding the code makes has
#: its own warning; the ``findings`` property keeps the sentences as data.
T3_MODEL_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "separation": "warning",
        "recording_separation": "warning",
        "calibration_finding": "warning",
        "mean_only": "warning",
        "weak_outcome_model": "warning",
        "weak_treatment_model": "note",
        "calibration": "note",
        "collaborative_model": "note",
        "declared_treatment": "note",
        "fit_status": "note",
        "selection": "note",
        "split_ratio": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_MODEL_NOTE_CODES})

T3_MODEL_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "separation": (
            "The treatment model separates the arms almost perfectly (AUC {auc}). That "
            "signals an overlap problem, not a good fit."
        ),
        "recording_separation": (
            "The recording model predicts almost perfectly (AUC {auc}): some units had almost "
            "no chance of a recorded outcome, so their weights are extreme. See "
            "result.diagnostics.support()."
        ),
        "calibration_finding": (
            "The {model}'s out-of-fold predictions are {direction} than the observed rates "
            "(calibration slope {slope}, {level} interval {low} to {high}{extra}).{noise} "
            "Review the learner before you change the model."
        ),
        "mean_only": (
            "The {model} put {share} of its weight on the overall mean: no candidate beat "
            "predicting the average."
        ),
        "weak_outcome_model": (
            "The outcome model explains little of the outcome (R squared {r2}), so the estimate "
            "behaves like inverse-probability weighting and is correspondingly noisy."
        ),
        "weak_treatment_model": (
            "The treatment model predicts treatment barely better than chance (AUC {auc}). If "
            "the model captures how treatment was assigned, treated and untreated rows look "
            "alike on these covariates, which favours overlap."
        ),
        "calibration": "{sentences}",
        "collaborative_model": (
            "The treatment model here is the one the collaborative search chose, so its "
            "metrics describe that working model, not treatment given all covariates."
        ),
        "declared_treatment": (
            "The treatment probabilities were declared, not fitted, so the treatment rows "
            "compare the declaration with the treatment each row received."
        ),
        "fit_status": "{sentences}",
        "selection": "{sentence}",
        "split_ratio": (
            "Each split fits its own models. The SD between splits and the reported {label} "
            "are on the estimate's inference scale (the log scale for a ratio); their ratio "
            "is a description, and no threshold applies."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_MODEL_NOTE_TEXTS})

#: Sentences of the ``calibration`` note.
CALIBRATION_IN_SAMPLE: Final = (
    "Calibration slopes are in sample and carry no test: an unpenalized logistic model with "
    "an intercept has an in-sample slope of {one} by construction."
)
CALIBRATION_OUT_OF_FOLD: Final = (
    "The calibration slope refits the out-of-fold predictions by logistic regression with "
    "one intercept per fold; {one} means the predictions are as spread as the outcomes."
)
CALIBRATION_RULE: Final = (
    "A finding needs the {level} interval to lie above {zero} and exclude {one}, a "
    "Bonferroni level over {tested} tested {models}."
)
CALIBRATION_MISSING: Final = "The {model} has no calibration slope: {reason}."
CALIBRATION_NOISE: Final = (
    " Estimation noise can cause this even in a correctly specified model when the signal "
    "is weak or the model has many covariates."
)
CALIBRATION_DIRECTIONS: Final[Mapping[bool, str]] = MappingProxyType(
    {True: "more extreme", False: "more moderate"}
)

#: The first sentence of the ``fit_status`` note on a fit that reports no interval.
NO_INTERVAL: Final = "This fit reports no confidence interval or p-value."

#: The ``selection`` note of a collaborative selector fit.
SELECTION_SENTENCE: Final = (
    "The collaborative search chose candidate {selected} of {count} for {estimand}{more}."
)
SELECTION_STOPPED: Final = "; it stopped after candidate {stopped}"
SELECTION_FINAL: Final = "; the final step used candidate {final}"

#: Plain names of the fitted models, by report name (an arm label follows in brackets).
MODEL_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "propensity": "treatment model",
        "outcome": "outcome model",
        "missingness": "recording model",
        "intermediate": "intermediate model",
        "treatment_observation": "treatment-recording model",
    }
)

#: Table titles and columns of the model check.
MODEL_TITLE: Final = "Outcome and treatment model check"
PROBABILITY_TABLE: Final = "Models that predict a probability"
REGRESSION_TABLE: Final = "Models that predict a mean"
PROBABILITY_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "auc": "AUC",
        "brier": "Brier score",
        "log_loss": "log loss",
        "calibration_slope": "calib. slope",
        "calibration_slope_se": "slope SE",
    }
)
REGRESSION_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {"r2": "R2", "mse": "MSE", "regression_slope": "regression slope"}
)
LEARNER_TABLE: Final = "Super learner weights"
SPREAD_TABLE: Final = "Spread between splits"

#: The ``Predictions`` fact of the model check, by ``evaluation``.
EVALUATION_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"out_of_fold": "out of fold", "in_sample": "in sample"}
)

#: The ``Split shown`` fact of a repeated fit.
SPLIT_OF: Final = "split {split} of {total}"

#: The reading of the model check.
MODEL_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "clear": "The models show no problem these metrics can detect.",
        "findings": "{count} {findings} about the models; see the warnings.",
    }
)

#: Lines of the model check's full detail, for what the fit could not report.
MODEL_OMISSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "selection": "The collaborative selection is not available: {reason}.",
        "outcome": "No outcome model was fitted: {reason}.",
        "spread": "No spread between splits is available: {reason}.",
    }
)
MODEL_NOTHING_OMITTED: Final = "The fit reported every model this check reads."


# ------------------------------------------------------- checks (T3): overlap of the weights

#: Header labels of the overlap check.
UNITS_CAPPED: Final = "Units capped"
HEADER_LABELS = HEADER_LABELS | {UNITS_CAPPED}

#: The notes of the overlap check. None is a warning: the reading carries the graded tier.
T3_OVERLAP_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "effective_share": "note",
        "equation_concentration": "note",
        "simplex": "note",
        "composed_excluded": "note",
        "group_omitted": "note",
        "first_split": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_OVERLAP_NOTE_CODES})

T3_OVERLAP_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "effective_share": (
            "The smallest arm keeps an effective {share} of its rows after weighting. That "
            "share describes how concentrated the weights are; it is not a test, and no "
            "threshold applies. Effective rows count how many equally weighted rows would "
            "carry the same weight, and the top shares are the part of the total weight "
            "that the largest {top_small} and {top_large} of the weights hold."
        ),
        "equation_concentration": (
            "In the estimating equation the weights concentrate most for {group}: {effective} "
            "effective rows of {rows} ({share}). This describes how unevenly rows are "
            "weighted; it is not the estimator's sample size, its precision or a test of "
            "overlap."
        ),
        "simplex": (
            "With more than two arms each arm's probability is capped on its own and not "
            "rescaled, so a row's capped probabilities can sum to other than one, here by up "
            "to {deviation}. A large value means the limits bind hard."
        ),
        "composed_excluded": (
            "{lead} {groups}. Read the single-probability rows for those estimates.{pointer}"
        ),
        "group_omitted": "{sentences}",
        "first_split": (
            "Overlap is shown for split {split} of {total}; each split fits its own "
            "treatment model."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_OVERLAP_NOTE_TEXTS})

#: The sentences of the ``group_omitted`` note: a group with no equation row, and a group
#: whose row lacks its capped count (an omission keyed ``"{group}.clipping"``).
GROUP_OMISSION_ROW: Final = "No equation row for {group}: {reason}."
GROUP_OMISSION_CAP: Final = "The capped count for {group} is not available: {reason}."

#: Pieces of the ``composed_excluded`` note.
COMPOSED_LEAD: Final = "The {name} row does not describe"
COMPOSED_LEAD_NONE: Final = "No joint-probability row is reported for"
COMPOSED_GROUP: Final = "{group}, which divides by {what}"
COMPOSED_POINTER: Final = " Their weights in the estimating equation are in the equation table."

#: What each estimand the joint-probability row leaves out divides by instead.
COMPOSED_EXCLUSION_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "att": (
            "the share treated rather than each row's treatment probability, and reweights "
            "the comparison arm by the odds of treatment"
        ),
        "atc": (
            "the share untreated rather than each row's treatment probability, and "
            "reweights the other arm by the odds of treatment"
        ),
        "ipsi": "a tilted treatment probability that drops the fitted one",
        "mtp": "a density ratio rather than a treatment probability",
    }
)
COMPOSED_EXCLUSION_OTHER: Final = "a quantity this row does not form"

#: Plain names of the other probabilities the weights divide by. A name not listed prints
#: as written.
MECHANISM_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "P(Delta=1|A,W)": "probability that the outcome is recorded",
        "P(Z=z|A,W)": "probability of the held intermediate value",
        "P(Delta_A=1|W)": "probability that the treatment is recorded",
        "P(A=a,Delta=1|W)": "joint probability of the treatment and a recorded outcome",
        "P(A=a,Z=z|W)": "joint probability of the treatment and the held intermediate value",
        "P(A=a,Delta=1,Z=z|W)": (
            "joint probability of the treatment, a recorded outcome and the held intermediate value"
        ),
        "P(A=a,Delta_A=1|W)": "joint probability of the treatment and its recording",
        "P(A=a,Delta_A=1,Delta=1|W)": (
            "joint probability of the treatment, its recording and a recorded outcome"
        ),
    }
)
#: Short row labels of the same probabilities, for the table, completing "probability of".
MECHANISM_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "P(Delta=1|A,W)": "outcome recorded",
        "P(Z=z|A,W)": "intermediate value",
        "P(Delta_A=1|W)": "treatment recorded",
        "P(A=a,Delta=1|W)": "arm and outcome recorded",
        "P(A=a,Z=z|W)": "arm and intermediate value",
        "P(A=a,Delta=1,Z=z|W)": "arm, outcome recorded, intermediate value",
        "P(A=a,Delta_A=1|W)": "arm and treatment recorded",
        "P(A=a,Delta_A=1,Delta=1|W)": "arm, treatment and outcome recorded",
    }
)
#: The condition a per-arm outcome-adaptive design puts in place of the covariates.
MECHANISM_DESIGN: Final = "{name}, given the fitted outcome means"
MECHANISM_DESIGN_STRATA: Final = "{name}, given the fitted outcome means and the strata"

#: The ``Result`` fact of the overlap check, worded from the branch that fired. The shares
#: come from the constants the tiers read. On a branch another probability fired, the
#: reading names it; the fact stays short enough for its half of the header.
OVERLAP_RESULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "adequate": "at most {strain} of units capped",
        "strain": "{strain} to {serious} of units capped",
        "serious": "over {serious} of units capped",
        "mechanism_capped": "another probability capped",
        "mechanism_concentrated": "weights concentrated",
    }
)

#: The reading of the overlap check, by branch.
OVERLAP_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "adequate": (
            "{share} of units were capped at {limits}, so the estimate does not lean on "
            "the cap. This check grades the cap only; the tables show the overlap itself."
        ),
        "strain": (
            "{share} of units were capped at {limits}. Report {curve} beside the estimate "
            "so readers see how much the answer depends on the cap."
        ),
        "serious": (
            "{share} of units were capped at {limits}, so the estimate leans on the cap "
            "rather than the fitted model for those units. The cap does not change the "
            "question, but treat the estimate as sensitive to it and check {curve} before "
            "you draw conclusions."
        ),
        "mechanism_capped": (
            "The {name} was capped on {share} of the rows at {limits}. It divides the "
            "weights as the treatment probability does, so those rows carry outsized weight "
            "whatever the treatment overlap looks like. Check {curve}."
        ),
        "mechanism_concentrated": (
            "The {name} falls to {low} and leaves an effective {effective} of the rows it "
            "weights, below the {threshold} share at which this check reports it, so a few "
            "rows carry much of the weight. Check {curve}."
        ),
        "mechanism_treatment": " {share} of units were capped at the treatment limits {limits}.",
    }
)
#: The limits a tier read, when they are one group's rather than the fit's.
GROUP_LIMITS: Final = "{limits}, the limits of {group}"
#: The curves a reading points to.
CURVE_TREATMENT: Final = "result.diagnostics.truncation_curve()"
CURVE_MECHANISM: Final = "result.diagnostics.truncation_curve(mechanism=True)"
CURVE_BOTH: Final = (
    "result.diagnostics.truncation_curve() and "
    "result.diagnostics.truncation_curve(mechanism=True), since either cap moves this row"
)

#: Table titles and columns of the overlap check.
OVERLAP_TITLE: Final = "Overlap check"
QUANTILE_TABLE: Final = "Treatment probabilities by group"
ARM_TABLE: Final = "Weights by arm"
TAIL_TABLE: Final = "Rows near 0 or 1"
EQUATION_TABLE: Final = "Weight concentration in the estimating equation"
MECHANISM_TABLE: Final = "Other probabilities in the weights"
WEIGHT_SHARE_COLUMNS: Final[tuple[str, ...]] = (
    "effective rows",
    "eff. share",
    "top 1%",
    "top 5%",
)
TAIL_COLUMNS: Final[tuple[str, ...]] = ("threshold t", "share below t", "share above 1 - t")
#: The ``Treatment model`` fact when the data declares the treatment probabilities.
KNOWN_TREATMENT: Final = "known (declared)"
#: A group's limits or cap count when the fit does not report them.
NOT_APPLICABLE: Final = "does not apply"

#: Titles and lines of the overlap check's full detail.
OVERLAP_DETAILS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "extreme": "Most extreme uncapped treatment probability: {value}.",
        "largest": "Largest weight in the correction step for {group}: {value}.",
        "equation": (
            "{group}: equation {equation}, {zero} of its rows carry zero weight, limits {limits}."
        ),
        "mechanism_limits": "The {name} is capped at {limits}.",
        "factor_by_factor": "{limits}, factor by factor",
        "composite": (
            "{name}: smallest value {low}; {share} of unit-arm cells fall below the "
            "treatment floor {floor}. The joint probability is adjusted inside [{lower}, "
            "{upper}], the product of the factor floors."
        ),
    }
)
OVERLAP_NOTHING_MORE: Final = "The tables above report everything this check measured."


# ----------------------------------------------------- checks (T3): longitudinal diagnostics

#: Header labels of the longitudinal checks.
REGIMENS: Final = "Regimens"
HEADER_LABELS = HEADER_LABELS | {REGIMENS}

#: The notes of the three longitudinal checks (overlap, equations and models over time).
T3_LONGITUDINAL_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "cumulative_weights": "note",
        "assigned": "note",
        "policy_nodes": "note",
        "tilt_nodes": "note",
        "trimmed": "note",
        "capped_nodes": "warning",
        "not_converged": "warning",
        "relative_error": "note",
        "every_split": "note",
        "working_model_rows": "note",
        "iterated_outcome": "note",
        "model_omitted": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_LONGITUDINAL_NOTE_CODES})

T3_LONGITUDINAL_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cumulative_weights": (
            "rows counts the units that followed the regimen and stayed observed through the "
            "time point. max weight and eff. rows describe the cumulative weight: one over the "
            "product of the treatment and censoring probabilities up to that time point, times "
            "the observation weight on a weighted fit. A product can be extreme while every "
            "time point looks fine. capped is the share of rows whose cumulative probability "
            "the lower limit replaced."
        ),
        "assigned": "{sentence}",
        "policy_nodes": (
            "At {nodes} the regimen draws the arm from a probability rather than fixing it. "
            "There assigned is the mean probability of each arm over the units at risk, rows "
            "counts the units whose observed arm has a positive probability, and the weight "
            "is the ratio of the regimen's probability to the fitted one. capped counts the "
            "rows whose denominator the limit replaced, which need not carry a large ratio."
        ),
        "tilt_nodes": (
            "Each time point of an incremental regimen tilts the odds of treatment, so "
            "assigned is the mean tilted probability of treatment over the units at risk."
        ),
        "trimmed": (
            "trimmed is the share of rows whose weight ratio at that time point the ratio trim "
            "moved."
        ),
        "capped_nodes": "{sentences}",
        "not_converged": "{sentences}",
        "relative_error": (
            "The remaining error is the largest part of a time point's estimating equation, as "
            "a share of the largest value that part could take. An equation passes when that "
            "share is at most the allowed error and its correction step converged."
        ),
        "every_split": "Rows cover every split; each split fits and corrects its own models.",
        "working_model_rows": (
            "A row with a component solves one equation of the working model, which combines "
            "every regimen, so it names no regimen."
        ),
        "iterated_outcome": (
            "An iterated outcome model regresses the prediction carried back from the next "
            "time point on the history up to its own; the last time point's outcome model "
            "regresses the outcome itself."
        ),
        "model_omitted": "{sentences}",
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_LONGITUDINAL_NOTE_TEXTS})

#: Sentences of the ``assigned`` note, by the kind of treatment.
ASSIGNED_SHARE: Final = (
    "assigned is the share of the units at risk at that time point whom the regimen treats; "
    "a regimen that names one arm assigns exactly {zero} or {one}."
)
ASSIGNED_SHARES: Final = (
    "assigned lists the share of the units at risk that the regimen sends to each arm."
)

#: Sentences of the ``capped_nodes`` and ``not_converged`` warnings.
CAPPED_NODES: Final = (
    "At {nodes} the lower limit replaced the cumulative probability of every row, so the "
    "estimate there rests on the limit, not on the fitted models."
)
CAPPED_OTHER_SPLITS: Final = "The limit replaced every row of some time point in {splits}."
NOT_CONVERGED: Final = (
    "The correction step did not converge at {nodes}. result.diagnostics.score_equations() "
    "shows how far each equation is from solved."
)
NOT_CONVERGED_OTHER_SPLITS: Final = "A correction step also failed to converge in {splits}."
NODE_NOT_SETTLED: Final = "{name}: the correction step did not converge."

#: Sentences of the ``model_omitted`` note.
MODEL_OMITTED_AT: Final = "No {model} model at time {time}{folds}: {reason}."
MODEL_OMITTED_ALL: Final = "No {model} model{folds}: {reason}."
MODEL_OMITTED_FOLDS: Final = " in fold {folds}"

#: How the longitudinal checks name a row: a regimen, its qualifiers and a time point.
NODE_AT: Final = "{name} at time {time}"
NODE_CAUSE: Final = "cause {cause}"
NODE_HORIZON: Final = "horizon {horizon}"
NODE_COMPONENT: Final = "working-model term {component}"
NODE_SPLIT: Final = "split {split}"
ALL_REGIMENS: Final = "all"
EVERY_REGIMEN: Final = "every regimen"

#: Plain names of the longitudinal model roles.
LONGITUDINAL_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "treatment": "treatment",
        "censoring": "censoring",
        "outcome": "outcome",
        "pseudo_outcome": "iterated outcome",
        "hazard": "event hazard",
    }
)

#: Short names of the losses the model check prints.
LOSS_SHORT: Final[Mapping[str, str]] = MappingProxyType(
    {"log_loss": "log loss", "brier": "Brier score", "mse": "MSE"}
)

#: Titles, columns, results and readings of the longitudinal checks.
STAGE_TITLE: Final = "Overlap over time"
STAGE_COLUMNS: Final[tuple[str, ...]] = ("rows", "max weight", "eff. rows", "capped")
STAGE_TRIMMED: Final = "trimmed"
STAGE_CONVERGED: Final = "converged"
STAGE_ASSIGNED: Final = "assigned"
CONVERGED_WORDS: Final[Mapping[bool, str]] = MappingProxyType({True: "yes", False: "no"})
STAGE_READING: Final = (
    "Overlap is thinnest for {node}: {effective} effective rows of {rows} followed."
)
STAGE_READING_NONE: Final = "This fit reports no time point with weights."
#: The full-detail lines of an SDR or IPSI fit's overlap over time, one per row.
SDR_LINE: Final = (
    "{node}: {outside} of the regression targets fall outside [{zero}, {one}] (from {low} to "
    "{high}), {predictions} of the predictions fall outside it, and the limits moved "
    "{bound} of the partial paths."
)
IPSI_LINE: Final = (
    "{node}: the tilt reweights a unit by a factor from {low} to {high}, which the limits "
    "never cap; {share} of the estimate's variance comes from this time point's treatment "
    "model; {degenerate} of the units at risk have a fitted treatment probability of "
    "exactly {zero} or {one}, which a tilt tolerates."
)

LONGITUDINAL_EQUATIONS_TITLE: Final = "Equations solved over time"
ALLOWED_SHARE: Final = "{tolerance:g} of max possible"
EQUATION_COLUMNS: Final[tuple[str, ...]] = ("remaining error", "result")
LONGITUDINAL_EQUATION_FAILED: Final = (
    "{count} of {total} equations {verb} not solved ({names}), so the standard error may be wrong."
)
#: The full-detail line of each equation: the steps taken and the equation's raw value.
STEPS_LINE: Final = "{node}: {steps} {word}, equation value {score}."

EQUATION_SCALE: Final = (
    "The allowed error is a share of the largest value each equation could take, the scale "
    "the correction loop itself stops on. The point-treatment check compares an equation "
    "with tolerance x SE / sqrt(n) instead."
)

LONGITUDINAL_MODELS_TITLE: Final = "Models over time"
MODEL_COLUMNS: Final[tuple[str, ...]] = ("model", "rows", "loss", "measure")
#: The full-detail line of each model: the metrics its kind reports.
METRIC_LINE: Final = "{node}, {model} model: {metrics}."
METRIC_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {"auc": "AUC", "calibration_slope": "calibration slope", "r2": "R2"}
)
METRIC_NONE: Final = "no metric reported"

LONGITUDINAL_MODELS_READING: Final = (
    "Each row describes how well one model fits; no pass or fail applies. A loss compares "
    "only with rows of the same model and measure."
)
LONGITUDINAL_SPREAD_LABEL: Final = "standard error"


# --------------------------------------------- checks (T3): combined reports of the checks

#: Header labels of the combined reports (``validate()``, ``run_all()``, ``assess()``).
CHECKS_RUN: Final = "Checks run"
NEED_ATTENTION: Final = "Need attention"
NOT_RUN: Final = "Not run"
REFITS_ALLOWED: Final = "Refits allowed"
RETARGETS_ALLOWED: Final = "Re-estimates allowed"
HEADER_LABELS = HEADER_LABELS | {
    CHECKS_RUN,
    NEED_ATTENTION,
    NOT_RUN,
    REFITS_ALLOWED,
    RETARGETS_ALLOWED,
}

#: The notes of the combined reports.
T3_BATTERY_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "next_steps": "note",
        "report_access": "note",
        "inference_status": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_BATTERY_NOTE_CODES})

T3_BATTERY_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "next_steps": "Suggested next steps: {steps}.",
        "inference_status": "About the interval: {reason}",
        "report_access": (
            "report.to_frame() lists every row with its detail, report.next_steps() the "
            "suggested calls, and report.report(name) the full report of one check."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_BATTERY_NOTE_TEXTS})

#: The name a reader recognises for each operation, the battery's row label (X89 T3 2.3).
CHECK_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "support": "overlap",
        "score_equations": "equations solved",
        "nuisance_models": "outcome and treatment models",
        "corrections": "DR-TMLE corrections",
        "truncation_curve": "cap on extreme weights",
        "refute": "refutation tests",
        "influence": "influential rows",
        "omitted_confounding": "unmeasured confounding",
        "robustness_value": "robustness value",
        "elements": "confounding bound inputs",
        "contour": "confounding grid",
        "benchmark": "covariate benchmark",
        "evalue": "E-value",
        "simulated_confounding": "simulated confounder",
        "marginal_bounds": "worst-case bounds",
        "missingness": "missing not at random",
        "tipping_gamma": "missing-data tipping point",
    }
)

#: The status of a battery row in words.
STATUS_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "passed": "passed",
        "failed": "failed",
        "warning": "needs attention",
        "completed": "done",
        "deferred": "on request",
        "not_applicable": "does not apply",
        "unavailable": "not available",
    }
)

#: Titles, tables, results and readings of the combined reports.
BATTERY_TITLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "validation": "Quick checks of the fit",
        "diagnostics": "Diagnostics of the fit",
        "sensitivity": "Sensitivity analyses of the fit",
        "assessment": "Assessment of the fit",
    }
)
BATTERY_COLUMNS: Final[tuple[str, ...]] = ("check", "result", "finding")
BATTERY_CHECKS: Final = "Checks"
BATTERY_RESULTS: Final = "Results"
BATTERY_RESULT_COLUMNS: Final[tuple[str, ...]] = ("analysis", "finding")
BATTERY_NOT_RUN: Final = "Not run"
BATTERY_NOT_RUN_COLUMNS: Final[tuple[str, ...]] = ("check", "why")
BATTERY_SURFACE: Final = "{name} ({surface})"
BATTERY_ROWS_IN_FULL: Final = "Rows in full"
BATTERY_ROW_LINE: Final = "{name} ({status}): {text}"
BATTERY_NO_ROWS: Final = "This report holds no rows."
YES_NO: Final[Mapping[bool, str]] = MappingProxyType({True: "yes", False: "no"})

#: The finding of a row whose report has no reading.
FINDING_HELD: Final = "see report.report('{name}')"

#: The ``Result`` of a combined report, by what its rows hold.
BATTERY_RESULTS_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "failed": "failed",
        "attention": "needs attention",
        "incomplete": "incomplete",
        "passed": "passed",
        "clear": "nothing needs attention",
    }
)
BATTERY_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "failed": "{names} failed.",
        "attention": "{names} {verb} attention; each row says why.",
        "incomplete": "{names} did not run, so validation is incomplete.",
        "passed": "Every quick check passed or gave a description.",
        "diagnostics_clear": "Nothing failed or needs attention.",
        "assessment": "{count} {checks} {verb} attention: {names}.",
        "assessment_clear": "No check needs attention.",
    }
)

#: What a check costs to run, by the capability's execution class and cost.
CHECK_COSTS: Final[Mapping[str, str]] = MappingProxyType(
    {"cheap": "cheap", "retarget": "re-estimates", "refit": "refits the models"}
)

#: The facades' list of the checks a fit offers (``result.diagnostics``, ``.sensitivity``).
FACADE_TITLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "diagnostics": "Diagnostics available for this fit",
        "sensitivity": "Sensitivity analyses available for this fit",
    }
)
FACADE_COLUMNS: Final[tuple[str, ...]] = ("check", "what it answers", "available", "cost")
CHECKS_AVAILABLE: Final = "Available"
CHECKS_LISTED: Final = "Checks"
HEADER_LABELS = HEADER_LABELS | {CHECKS_AVAILABLE, CHECKS_LISTED}
FACADE_READING: Final = (
    "Call result.{surface}.<check>() to run one, or result.{surface}.run_all() to run every "
    "check that needs no argument."
)

#: What a saved result can redo (``Replayability``).
REPLAY_TITLE: Final = "What this saved result can redo"
REPLAY_COLUMNS: Final[tuple[str, ...]] = ("action", "possible")
REPLAY_POSSIBLE: Final = "Possible"
HEADER_LABELS = HEADER_LABELS | {REPLAY_POSSIBLE}
#: Each slot of ``Replayability``: its action in words, its short repr name, and what it needs.
REPLAY_ACTIONS: Final[Mapping[str, tuple[str, str, str]]] = MappingProxyType(
    {
        "summarize_existing_artifacts": (
            "show the stored checks",
            "summarize",
            "the checks stored with the fit.",
        ),
        "retarget_cached_nuisances": (
            "re-estimate from the stored models without refitting them",
            "re-estimate",
            "the stored model predictions and the estimator's settings.",
        ),
        "evaluate_stored_representer": (
            "evaluate other quantities from the stored weighting function",
            "evaluate",
            "a stored weighting function, which no estimator keeps yet.",
        ),
        "refit_nuisances": (
            "refit the models",
            "refit",
            "the estimator's settings, learners that can be copied with their seeds, and every "
            "declaration the fit checked.",
        ),
        "evaluate_new_data": (
            "score new data",
            "score",
            "fitted models kept for prediction on new rows, which no estimator keeps yet.",
        ),
        "rerun_configuration": (
            "rerun the same settings on changed data",
            "rerun",
            "the estimator's settings and the declarations a fit checks; a rerun does not "
            "promise the same numbers.",
        ),
    }
)
REPLAY_NEEDS: Final = "What each action needs"
REPLAY_NEED_LINE: Final = "{action}: {needs}"
#: The parts ``Replayability.unreconstructible`` names, in words.
REPLAY_PARTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "estimator configuration": "the estimator's settings",
        "point_replay_function_declaration": "a declaration that a stored function is known",
        "point_replay_refit_configuration": "settings that a refit accepts",
        "longitudinal_replay_learner_unclonable": "learners that can be copied",
        "longitudinal_replay_random_state_unseeded": "a seed for each learner",
        "longitudinal_replay_random_state_non_integer": "an integer seed for each learner",
        "longitudinal_replay_fitted_bound_mismatch": "probability limits that match the fit",
        "longitudinal_replay_regimen_declaration": "a declaration that a regimen rule is known",
        "longitudinal_replay_msm_declaration": (
            "a declaration that the working-model design and weights are known"
        ),
    }
)
REPLAY_READINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "all": "This result can redo every action listed.",
        "some": "This result cannot {cannot}.",
        "none": "This result can redo none of the actions listed.",
    }
)
T3_ACCESS_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {"missing_parts": "note", "why_not": "note"}
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T3_ACCESS_NOTE_CODES})
T3_ACCESS_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "missing_parts": "The result lacks {parts}.",
        "why_not": ("Cannot run here: {names}. result.{surface}.capability(name).reason says why."),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T3_ACCESS_NOTE_TEXTS})

#: Code that a summary may print whole, though it holds a word the house style keeps out of
#: prose: a public call such as ``result.diagnostics.nuisance_models()``, or the attribute
#: path that counts the DR-TMLE reduction's hard steps, which no check prints. The contract
#: test takes each match out before it reads the text for internal terms.
PUBLIC_CALL: Final = re.compile(
    r"result\.(?:diagnostics|sensitivity)\.(?:"
    + "|".join(sorted(CHECK_NAMES, key=len, reverse=True))
    + r")\([^)]*\)"
    r"|report\.(?:to_frame|next_steps|reports|report|draws_frame|summary)\([^)]*\)"
    r"|result\.repeats\[\*\]\.fluctuations\['mean'\]\.reduction\.ill_conditioned"
)


# ----------------------------------------------------------------- study setup (T5)

#: The header labels of the study-setup and simulation summaries.
BASELINE: Final = "Baseline"
FOLLOWED: Final = "Observed to the end"
EFFECTIVE_SAMPLE_SIZE: Final = "Effective sample size"
CENSORED_BEFORE_END: Final = "Censored before the end"
HELD_FIXED: Final = "Held fixed"
PROTOCOL: Final = "Study protocol"
ANSWERED_BY: Final = "Answered by"
METHODS_FIT: Final = "Methods that fit"
POPULATION: Final = "Population"
TIME_ZERO: Final = "Time zero"
INTERFERENCE_UNIT: Final = "Interference unit"
REPLICATIONS: Final = "Replications"
ROWS_PER_REPLICATION: Final = "Rows per replication"
INTERVAL_LEVEL: Final = "Interval level"
TOTAL_TIME: Final = "Total time"
ACCOUNTED: Final = "Accounted for"
HEADER_LABELS = HEADER_LABELS | {
    BASELINE,
    FOLLOWED,
    EFFECTIVE_SAMPLE_SIZE,
    CENSORED_BEFORE_END,
    HELD_FIXED,
    PROTOCOL,
    ANSWERED_BY,
    METHODS_FIT,
    POPULATION,
    TIME_ZERO,
    INTERFERENCE_UNIT,
    REPLICATIONS,
    ROWS_PER_REPLICATION,
    INTERVAL_LEVEL,
    TOTAL_TIME,
    ACCOUNTED,
}

#: The study-setup and simulation summaries' note codes and texts.
T5_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "further_conditions": "note",
        "checkable": "note",
        "model_restriction": "note",
        "several_readings": "note",
        "unbound_effect": "warning",
        "no_method": "warning",
        "other_methods": "note",
        "failed_replications": "warning",
        "measures_diagnostic": "warning",
        "mixed_statuses": "note",
        "worker_time": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T5_NOTE_CODES})

T5_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "other_methods": (
            "The methods table lists the methods that fit. {count} other {methods} "
            "cannot estimate this question; the full summary says why."
        ),
        "further_conditions": (
            "The full summary states {count} further {conditions} the answer rests on."
        ),
        "checkable": (
            "The data cannot confirm these conditions. Overlap (marked partly) can be probed: "
            "after a fit, result.diagnostics.support() shows where the fitted treatment model "
            "gives units little chance of a compared treatment. That check reads a fitted "
            "model, so a clean result does not prove overlap."
        ),
        "model_restriction": (
            "The design restricts the model of the data; the full summary states how. The "
            "question is unchanged, and the interval of a fit is claimed only under the "
            "restriction."
        ),
        "several_readings": (
            "The answer has {count} causal readings, each resting on its own condition; the "
            "full summary states each."
        ),
        "unbound_effect": (
            "This question is not bound to data, so it cannot be estimated, and the methods "
            "table is worked out from the question without the study's design."
        ),
        "no_method": "No method fits this question; the full summary says why each does not.",
        "data_warning": "When cleverly read the data it warned: {messages}",
        "failed_replications": (
            "Replications that raised an error: {count} of {total}. Every column leaves "
            "them out, and the full summary lists the errors."
        ),
        "measures_diagnostic": (
            "One or more estimates supply no interval{some}, so mean SE, SE ratio, coverage "
            "and rejected describe the {noun} and not a confidence interval."
        ),
        "mixed_statuses": (
            "The replicates took more than one inference status, named as in the inference "
            "column of to_frame(): {clauses}."
        ),
        "worker_time": (
            "The parallel workers' times are each worker's own elapsed time, added across "
            "workers, so they can pass the time the fit waited and are not added to the "
            "table above."
        ),
        "spread_scale_simulation": (
            "For {names}, mean SE and SE ratio are on the log scale; truth, bias and MC SD "
            "are on the ratio scale."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T5_NOTE_TEXTS})

#: The Design fact, keyed by the design class's name. ``{T}`` is the number of time points.
DESIGN_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "PointTreatment": "one treatment decision",
        "LongitudinalTreatment": "treatment at {T} times",
        "TimeToEvent": "time to an event, {T} time points",
        "LongPanel": "{T} times, long rows read as one row per unit",
    }
)

#: The Answered by fact, keyed by whether the design is longitudinal.
ANSWERED_BY_WORDS: Final[Mapping[bool, str]] = MappingProxyType(
    {
        False: "adjusting for covariates",
        True: "adjusting for history",
    }
)
#: The Outcome fact of a time-to-event design.
EVENT_OUTCOME: Final = "{event} (event code), timed by {time}"
#: The Answered by fact of a randomized design that adjusts for nothing.
ANSWERED_BY_RANDOMIZED: Final = "randomized assignment"
#: The Adjusted for fact of a randomized design that adjusts for nothing.
RANDOMIZED_NO_ADJUSTMENT: Final = "none (randomized, as declared)"
#: The title of each setup summary.
STUDY_TITLE: Final = "Study design"
QUESTION_TITLE: Final = "Causal question ({name})"
METHODS_TITLE: Final = "Estimation methods"
PROTOCOL_TITLE: Final = "Study protocol"
SIMULATION_STUDY_TITLE: Final = "Simulation study: {label}"
PHASES_TITLE: Final = "Time by phase"
#: Table titles.
ROLES_TITLE: Final = "Columns by role"
CONDITIONS_TITLE: Final = "Conditions the answer rests on"
ARMS_TITLE: Final = "Share of rows by treatment level"
#: The share column of that table, marked weighted on weighted data.
SHARE_COLUMN: Final = "share"
HISTORY_TITLE: Final = "What each treatment time adjusts for"
METHODS_TABLE_TITLE: Final = "Methods"
STRATEGIES_TITLE: Final = "Treatment strategies"
ELIGIBILITY_TITLE: Final = "Eligibility"
WORKERS_TITLE: Final = "In parallel workers"
#: Full-detail sections of the setup summaries.
SECTION_UNAVAILABLE: Final = "Why a method does not fit"
SECTION_INTERCURRENT: Final = "Intercurrent events"
SECTION_RATIONALE: Final = "Rationale for the assumptions"
SECTION_PER_ESTIMAND: Final = "Per estimand"
SECTION_FAILURES: Final = "Failed replications"
SECTION_COLUMNS: Final = "Reading the columns"
#: The lead of a question's Assumptions section, with the terms of art the short table's
#: plain names stand for.
CONDITIONS_LEAD: Final = (
    "The answer equals {functional} when the statements below hold. The short table calls "
    "consistency well-defined treatment, exchangeability no hidden confounding, and "
    "positivity overlap."
)

#: One condition's plain words: the name the short table prints, its sentence, and
#: whether the data can say anything about it.
Checkable = Literal["no", "partly"]

#: The plain words for the identification statements that recur, matched on the start of
#: the statement in this order, most specific first. Each value is ``(name, sentence,
#: checkable)``; ``{A}``, ``{Y}`` and ``{Z}`` are the treatment, outcome and held variable.
#: A statement no entry matches prints only in the full summary, verbatim, such as the
#: definedness of a ratio, a known policy or a model restriction. ``test_setup_summaries``
#: pins which entry each statement of a broad set of questions reaches.
CONDITION_WORDS: Final[tuple[tuple[str, tuple[str, str, Checkable]], ...]] = (
    (
        "consistency: each observed history",
        (
            "well-defined treatment",
            "Each unit's observed outcomes are the ones it would have had under the "
            "treatments it actually received over time. This needs each treatment to be "
            "well defined: every way of getting the same treatment has the same effect.",
            "no",
        ),
    ),
    (
        "consistency:",
        (
            "well-defined treatment",
            "Each unit's observed outcome is the outcome it would have had under the "
            "treatment it actually received{held}. This needs the treatment to be well "
            "defined: every way of getting the same level has the same effect.",
            "no",
        ),
    ),
    (
        "no interference:",
        ("no interference", "One unit's treatment does not change another unit's outcome.", "no"),
    ),
    (
        "no unmeasured intermediate confounding:",
        (
            "no hidden confounding of {Z}",
            "Among units with the same treatment and adjusted-for values, the value of {Z} "
            "is unrelated to the outcomes they would have had.",
            "no",
        ),
    ),
    (
        "no unmeasured treatment confounding:",
        (
            "no hidden confounding",
            "Among units with the same adjusted-for values, which treatment a unit got is "
            "unrelated to the outcomes it would have had. This holds when the adjusted-for "
            "columns account for the common causes of {A} and {Y} and none of them is "
            "affected by {A}.",
            "no",
        ),
    ),
    (
        "no unmeasured confounding:",
        (
            "no hidden confounding",
            "Among units with the same adjusted-for values, which treatment a unit got is "
            "unrelated to the outcomes it would have had. This holds when the adjusted-for "
            "columns account for the common causes of {A} and {Y} and none of them is "
            "affected by {A}.",
            "no",
        ),
    ),
    (
        "sequential exchangeability given the recorded history",
        (
            "no hidden confounding at each time",
            "At each time, among units with the same recorded history, which treatment a "
            "unit got and whether it stays in follow-up are unrelated to the outcomes it "
            "would have had.",
            "no",
        ),
    ),
    (
        "sequential exchangeability in the form the chosen interpretation names",
        (
            "no hidden confounding at each time",
            "At each time, among units with the same recorded history, the dose a unit got "
            "is unrelated to the outcomes it would have had, in the strong or the standard "
            "form that each causal reading names.",
            "no",
        ),
    ),
    (
        "dropout is missing at random given the recorded history",
        (
            "follow-up loss at random",
            "At each time, among units with the same recorded history, whether a unit stays "
            "in follow-up is unrelated to its later outcomes, and each unit still at risk had "
            "some chance of staying.",
            "no",
        ),
    ),
    (
        "sequential positivity for treatment and remaining under observation",
        (
            "overlap at each time",
            "At each time, units with every history that following the compared regimens "
            "can produce had some chance of the treatment a regimen assigns and of staying "
            "in follow-up.",
            "partly",
        ),
    ),
    (
        "positivity: the dose a policy assigns",
        (
            "overlap at each time",
            "At each time, units with every recorded history had some chance of the dose "
            "the policy assigns wherever they had a chance of their observed dose, and of "
            "staying in follow-up.",
            "partly",
        ),
    ),
    (
        "positivity: ",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of "
            "each compared level of {A}.",
            "partly",
        ),
    ),
    (
        "positivity for the evaluated arm:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "evaluated level of {A}.",
            "partly",
        ),
    ),
    (
        "positivity for the reference arm within each comparison population:",
        (
            "overlap",
            "Among units that got a compared level of {A}, every group that shares the "
            "adjusted-for values had some chance of the reference level.",
            "partly",
        ),
    ),
    (
        "positivity for each comparison arm within the reference population:",
        (
            "overlap",
            "Among units that got the reference level of {A}, every group that shares the "
            "adjusted-for values had some chance of each compared level.",
            "partly",
        ),
    ),
    (
        "positivity for the reference intervention:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "reference level of {A}.",
            "partly",
        ),
    ),
    (
        "positivity for the evaluated doses:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of each "
            "dose of {A} the question evaluates.",
            "partly",
        ),
    ),
    (
        "positivity for the evaluated treatment values:",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of each "
            "treatment value the question evaluates.",
            "partly",
        ),
    ),
    (
        "positivity *for the regime*",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "level of {A} the rule assigns them.",
            "partly",
        ),
    ),
    (
        "positivity *for the assigned dose*",
        (
            "overlap",
            "Every group of units that share the adjusted-for values had some chance of the "
            "dose the policy assigns, wherever it had some chance of its observed dose.",
            "partly",
        ),
    ),
    (
        "positivity for the treated arm wherever the tilt keeps it",
        (
            "overlap",
            "Every group of units the tilt moves towards a level of {A} had some chance of "
            "that level.",
            "partly",
        ),
    ),
    (
        "intermediate positivity at",
        (
            "overlap for {Z}",
            "Every group of units that share the treatment and adjusted-for values had some "
            "chance of the value {Z} is held at.",
            "partly",
        ),
    ),
    (
        "missingness at random for",
        (
            "outcome missing at random",
            "Among units with the same treatment and adjusted-for values, whether {Y} was "
            "recorded is unrelated to its value.",
            "no",
        ),
    ),
    (
        "response positivity for",
        (
            "overlap for recorded outcomes",
            "Every group of units that share the treatment and adjusted-for values that the "
            "question reads had some chance of a recorded {Y}.",
            "partly",
        ),
    ),
    (
        "product positivity at the reference arm",
        (
            "overlap for recorded outcomes",
            "Every group of units that share the adjusted-for values had some chance of "
            "both the reference level of {A} and a recorded {Y}.",
            "partly",
        ),
    ),
)

#: The plain words that replace an entry of :data:`CONDITION_WORDS` when the design has a
#: feature: ``"held"`` a held intermediate, ``"randomized"`` a randomized assignment with no
#: adjustment, ``"uncensored"`` a longitudinal design without censoring.
CONDITION_VARIANTS: Final[Mapping[str, tuple[tuple[str, tuple[str, str, Checkable]], ...]]] = (
    MappingProxyType(
        {
            "held": (
                (
                    "missingness at random for",
                    (
                        "outcome missing at random",
                        "Among units with the same treatment, {Z} and adjusted-for values, "
                        "whether {Y} was recorded is unrelated to its value.",
                        "no",
                    ),
                ),
            ),
            "randomized": (
                (
                    "no unmeasured confounding:",
                    (
                        "no hidden confounding",
                        "Which treatment a unit got is unrelated to the outcomes it would "
                        "have had, as in a randomized trial. The design declares this, and "
                        "the data cannot show it.",
                        "no",
                    ),
                ),
                (
                    "positivity: ",
                    (
                        "overlap",
                        "Every unit had some chance of each compared level of {A}, as a "
                        "randomized assignment gives.",
                        "partly",
                    ),
                ),
            ),
            "uncensored": (
                (
                    "sequential exchangeability given the recorded history",
                    (
                        "no hidden confounding at each time",
                        "At each time, among units with the same recorded history, which "
                        "treatment a unit got is unrelated to the outcomes it would have had.",
                        "no",
                    ),
                ),
                (
                    "sequential positivity for treatment and remaining under observation",
                    (
                        "overlap at each time",
                        "At each time, units with every history that following the compared "
                        "regimens can produce had some chance of the treatment a regimen "
                        "assigns.",
                        "partly",
                    ),
                ),
                (
                    "positivity: the dose a policy assigns",
                    (
                        "overlap at each time",
                        "At each time, units with every recorded history had some chance of "
                        "the dose the policy assigns wherever they had a chance of their "
                        "observed dose.",
                        "partly",
                    ),
                ),
            ),
        }
    )
)

#: The words a question's held intermediate adds to the well-defined-treatment sentence.
HELD_CLAUSE: Final = " and the value of {Z} it was held at"


#: The reading of a simulation study.
SIMULATION_PASS: Final = (
    "Coverage is not below {level} by more than Monte Carlo error, and the bias is within "
    "Monte Carlo error of zero."
)
SIMULATION_DIAGNOSTIC_PASS: Final = (
    "The bias is within Monte Carlo error of zero. The coverage column measures a {noun}, so "
    "it certifies no confidence interval."
)
SIMULATION_UNDERCOVERAGE: Final = (
    "{name}: coverage {coverage} is below the nominal {level} by more than Monte Carlo error"
)
SIMULATION_SMALL_UNDERCOVERAGE: Final = (
    "{name}: coverage {coverage} is below the nominal {level} by more than Monte Carlo error, "
    "though by no more than {floor} percentage points"
)
#: One estimand's clause of the mixed-statuses note.
MIXED_STATUS_CLAUSE: Final = "{name} took {statuses}, and each of its columns reads {what}"
SIMULATION_SE_SMALL: Final = "; the {spread} is {factor}x too small"
SIMULATION_SE_ZERO: Final = "; every reported standard error is zero"
SIMULATION_SE_RIGHT: Final = (
    "; the standard error is not too small, so the shortfall points to bias or a skewed "
    "sampling distribution"
)
SIMULATION_BIAS: Final = "{name}: bias {bias} is {count} Monte Carlo standard errors from zero"
SIMULATION_CONSTANT_BIAS: Final = (
    "{name}: every replicate returned the same estimate, so its bias {bias} has no Monte Carlo "
    "error"
)
SIMULATION_SMALL_BIAS: Final = (
    "{name}: bias {bias} is {count} Monte Carlo standard errors from zero, though no more "
    "than {floor} of the larger of 1 and the absolute truth"
)
SIMULATION_EXPERIMENTAL: Final = (
    "At least one replicate's estimate of {names} has no validation study behind its kind of "
    "fit, so cleverly marks it experimental."
)
#: How to read the columns of a phase timing profile.
PHASE_COLUMNS: Final[tuple[str, ...]] = (
    "own s: the seconds spent in the phase itself, outside any phase nested in it; this "
    "column adds up to the accounted time.",
    "with nested s: the seconds from entering the phase to leaving it, nested phases "
    "included, so this column overlaps itself and does not add up.",
    "share: own seconds as a share of the total time.",
)

# ----------------------------------------------------------------- longitudinal (T2)

#: The plain name of each longitudinal estimator, keyed by ``LongitudinalConfig.estimator``.
LONGITUDINAL_METHOD_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "longitudinal targeted maximum likelihood (LTMLE)",
        "msm": "longitudinal TMLE of a working model (MSM)",
        "sdr": "sequentially doubly robust estimator (SDR)",
        "ipsi": "incremental propensity score estimator (IPSI)",
        "gcomp": "sequential g-computation",
        "ipw": "inverse probability weighting (IPW)",
        "parametric_gformula": "parametric g-formula (forward simulation)",
    }
)

#: The short name a longitudinal summary title carries.
LONGITUDINAL_SHORT_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": "LTMLE",
        "msm": "LTMLE, working model",
        "sdr": "SDR",
        "ipsi": "IPSI",
        "gcomp": "g-computation",
        "ipw": "IPW",
        "parametric_gformula": "parametric g-formula",
    }
)

#: The IPW normalization in words, keyed by ``LongitudinalConfig.normalization``.
IPW_FORMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "hajek": "Hajek form, normalized by the weight sum",
        "horvitz_thompson": "Horvitz-Thompson form, normalized by n",
    }
)

#: The title of a longitudinal summary, filled with the short method name.
LONGITUDINAL_TITLE: Final = "Longitudinal estimate ({method})"
STRUCTURAL_TITLE: Final = "Structural blip estimates (SNMM)"
STRUCTURAL_METHOD: Final = "linear structural nested mean model (SNMM)"
CHECK_TITLE: Final = "Structural model check"
SIMULATION_TITLE: Final = "Simulation report (parametric g-formula)"
REGIME_SUPPORT_TITLE: Final = "Overlap of each regime"
POLICY_SUPPORT_TITLE: Final = "Overlap of policy {name}"
TILT_SUPPORT_TITLE: Final = "Overlap of tilt {name}"

#: The structural nested mean summary (X89 T2): the Question, the row meanings and the
#: blip-feature table.
STRUCTURAL_QUESTION: Final = "effect of each treatment time on {Y} under the blip model"
STRUCTURAL_QUESTION_PLANS: Final = (
    "effect of each treatment time on {Y} under the blip model, and the plan means it implies"
)
STRUCTURAL_LEVEL: Final = "{A} = {level} instead of {ref}"
STRUCTURAL_DOSE: Final = "one more unit of {A}"
STRUCTURAL_MEANINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "blip": (
            "coefficient of {feature} in the effect on {Y} of {change} at time {t}, with later "
            "treatments at their reference"
        ),
        "intercept": (
            "effect on {Y} of {change} at time {t}, with later treatments at their reference, "
            "for a unit whose other blip features are zero"
        ),
        "reference": "mean {Y} if every unit had the reference level of every treatment",
        "level": "mean {Y} if every unit followed plan {d}, read from the blip model",
        "contrast": (
            "mean {Y} under plan {d} minus that mean under plan {r}, read from the blip model"
        ),
    }
)
BLIP_TITLE: Final = "Blip features at each time point"
TIME: Final = "time"
BLIP_TREATMENT: Final = "treatment"
BLIP_LEVELS: Final = "levels"
BLIP_FEATURES: Final = "features"
STRUCTURAL_FAMILY: Final = "Outcome regressions: {family}."
FAMILY_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"binomial": "logistic, for a 0-or-1 outcome", "gaussian": "linear, for a continuous outcome"}
)
STRUCTURAL_EIGENVALUE: Final = (
    "Smallest standardized eigenvalue of the estimating equation's derivative: {value}. Near "
    "zero, the treatment varies little given the history, or the blip features are nearly "
    "collinear given it, so the coefficients are weakly identified."
)
STRUCTURAL_EXTREME: Final = (
    "Largest share of rows with an estimated treatment probability within {margin} of 0 or 1, "
    "over the time points: {share}."
)
STRUCTURAL_RESTRICTION: Final = (
    "the effect of each treatment time, with later treatments at their reference, is linear "
    "in the declared blip features"
)
STRUCTURAL_CONSTRUCTION: Final = (
    "G-estimation: at each time point, from the last to the first, the blip coefficients solve "
    "an estimating equation that pairs the outcome, with later blips removed, against the "
    "treatment minus its predicted value."
)

#: The overlap reports of regimes, policies and tilts.
REGIME_ROW_TITLE: Final = "Overlap of regime {name}"
SUPPORT_REGIMES: Final = "Regimes"
SUPPORT_NO_REGIMES: Final = "The fit declares no regimes, so there is no overlap to report."
SUPPORT_WORST: Final = "Least supported"
SUPPORT_REGIME_TABLE: Final = "How much the data support each regime"
SUPPORT_REGIME_ONE_TABLE: Final = "How much the data support the regime"
SUPPORT_POLICY_TABLE: Final = "How much the data support the policy"
SUPPORT_TILT_TABLE: Final = "How much the data support the tilt"
SUPPORT_SPLIT: Final = " (score load from split {split} of {repeats})"
SUPPORT_REGIME: Final = "regime"
SUPPORT_MIN_G: Final = "min probability"
SUPPORT_MAX_RATIO: Final = "max ratio"
SUPPORT_RATIO_ESS: Final = "effective n"
SUPPORT_LOAD_EFFECTIVE: Final = "score load"
SUPPORT_UNSUPPORTED: Final = "unsupported"
SUPPORT_TWO_MECHANISMS: Final = (
    "min probability is the smallest estimated probability of an arm the regime assigns, max "
    "ratio the largest weight ratio, and effective n the effective sample size of those ratios. "
    "All three read the estimated treatment model before any limit, which is what the data "
    "support. unsupported counts the rows given an arm whose estimated probability is exactly zero."
)
SUPPORT_REGIME_DETAIL: Final = (
    "{name}: weight ratio quantiles {quantiles}; probability of the assigned arm {tails}."
)
SUPPORT_MOVED: Final = "Rows moved"
SUPPORT_UNSUPPORTED_ROWS: Final = "Rows sent to zero density"
SUPPORT_NOT_MEASURED: Final = "not measured"
SUPPORT_OUTSIDE: Final = "Rows sent outside the declared support"
SUPPORT_MECHANISM_NOTE: Final = (
    "The weight is the density ratio divided by the probability of staying under observation "
    "after its bound, and by the intermediate's density when the fit holds one; the smallest "
    "other factor is their smallest product."
)
SUPPORT_CAPPED: Final = "Rows held by the cap"
SUPPORT_MIN_DENSITY: Final = "Smallest density at an observed dose"
SUPPORT_NO_DENSITY: Final = "not measured: the classifier fits no density"
SUPPORT_MIN_MECHANISM: Final = "Smallest other factor dividing the weight"
SUPPORT_RATIO: Final = "density ratio"
SUPPORT_WEIGHT: Final = "weight"
SUPPORT_QUANTILE: Final = "{q} quantile"
SUPPORT_LARGEST: Final = "largest"
SUPPORT_ESS_ROW: Final = "effective size"
SUPPORT_SHARE_OF_ROWS: Final = "effective share"
SUPPORT_MEAN_RATIO: Final = "mean ratio"
SUPPORT_FOLD_MEAN: Final = "mean ratio, fold {fold}"
SUPPORT_MEAN_RATIO_READING: Final = (
    "The mean ratio's expected value under the true density equals the probability that the "
    "policy's dose stays inside the support of the observed doses given the covariates, which is "
    "one when the policy keeps that support; an upper cap alone does not ensure this for a "
    "negative shift or a gap in the support. In a sample the mean also moves with sampling and "
    "with error in the density model, so a mean near one does not prove that the support holds "
    "or that the effect is identified."
)
SUPPORT_ATOMS: Final = "Mean fitted probability of each declared atom: {masses}."
SUPPORT_NO_ZERO_CHECK: Final = (
    "The classifier fits no density, so the report cannot count rows sent to a dose of zero "
    "density."
)
SUPPORT_FACTORS: Final = "Odds factors"
SUPPORT_BOUND: Final = "Tilt weight bound"
SUPPORT_NO_FACTOR: Final = "none (the natural course)"
SUPPORT_TILT_READING: Final = (
    "An odds tilt keeps the tilt part of each weight inside the bound for any treatment "
    "model, so it needs no treatment probability away from 0 or 1. When outcomes are missing, "
    "the probability of staying under observation still divides the weight and needs "
    "positivity; the largest weight and the effective size do not include it. min "
    "probability is shown for comparison with a fit that sets each unit's treatment."
)

#: The parametric g-formula simulation report.
YES: Final = "yes"
NO: Final = "no"
GFORMULA_DRAWS: Final = "Draws per unit (estimate)"
GFORMULA_DERIVATIVE_DRAWS: Final = "Draws per unit (derivatives)"
GFORMULA_MODELS: Final = "Models"
GFORMULA_PARAMETRIC: Final = "All models parametric"
GFORMULA_MODELS_TITLE: Final = "The models the simulation draws from"
GFORMULA_MODEL: Final = "model"
GFORMULA_ROLE: Final = "role"
GFORMULA_RESPONSE: Final = "response"
GFORMULA_LEARNER: Final = "learner"
GFORMULA_READ: Final = "in estimate"
GFORMULA_CONSTANT: Final = "constant"
#: Short learner names for the models table, which has six columns.
GFORMULA_LEARNERS: Final[Mapping[str, str]] = MappingProxyType(
    {"GLM": "GLM", "LogisticRegression": "logistic", "LinearRegression": "linear"}
)
GFORMULA_ROLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "covariate": "covariate",
        "outcome": "outcome",
        "hazard": "event hazard",
        "treatment": "natural treatment",
    }
)
GFORMULA_RESPONSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "binary": "0 or 1",
        "multi": "categories",
        "normal": "normal",
        "mean": "mean",
    }
)
GFORMULA_SHARE_TITLE: Final = "Share of each row's variance from the simulation"
GFORMULA_SHARE: Final = "share"
GFORMULA_SUPPORT_TITLE: Final = "Support of each set dose"
GFORMULA_COLUMN: Final = "treatment"
GFORMULA_DOSE: Final = "dose"
GFORMULA_RULE_DOSE: Final = "set by a rule"
GFORMULA_OFF: Final = "histories > {z} SD off"
GFORMULA_WINDOW: Final = "observed within {z} SD"
GFORMULA_READ_MODELS: Final = (
    "The simulation draws each unit's covariates and outcome forward in time from these "
    "models, under each plan. in estimate says whether the estimate draws from the model: a "
    "set dose replaces the treatment model, which is then fitted but not used."
)
GFORMULA_READ_SHARE: Final = (
    "share is the part of each row's variance that comes from simulating a finite number of "
    "draws per unit rather than from the data; a larger n_draws lowers it."
)
GFORMULA_READ_SUPPORT: Final = (
    "For each set dose, the first share counts simulated histories whose fitted mean dose lies "
    "more than {z} fitted standard deviations from the dose, and the second the observed doses "
    "within {z} standard deviations of it. A large first share or a small second one says the "
    "dose lies outside what the data show, so the estimate rests on the models' extrapolation."
)

#: The structural model check summary.
CHECK_QUESTION: Final = "do the model's plan contrasts agree with LTMLE on the same rows?"
CHECK_CONTRASTS: Final = "Contrasts"
CHECK_SPLITS_LABEL: Final = "Fold splits"
CHECK_TEST: Final = "Test"
CHECK_TEST_F: Final = "scaled joint Wald statistic against F({df}, {df2})"
CHECK_TEST_CHI2: Final = "joint Wald statistic against chi-squared({df})"
CHECK_TABLE_TITLE: Final = "The model's contrasts against LTMLE"
CHECK_STRUCTURAL: Final = "structural"
CHECK_LTMLE: Final = "LTMLE"
CHECK_DIFFERENCE: Final = "difference"
CHECK_SPREAD: Final = "std. error"
CHECK_JOINT_TITLE: Final = "Joint test"
CHECK_STATISTIC: Final = "statistic"
CHECK_CONTRASTS_COLUMN: Final = "contrasts"
CHECK_COVARIANCE_DF: Final = "covariance df"
CHECK_PVALUE: Final = "p-value"
CHECK_READING: Final = (
    "A small p-value says the model fails for these contrasts; a large one does not confirm "
    "the model."
)
CHECK_NO_TEST: Final = "No test: {reason}"
CHECK_HOW_TO_READ: Final[tuple[str, ...]] = (
    "The check fits LTMLE on the same rows, weights, clusters and folds and compares each plan "
    "contrast with the one the structural model implies. When the model holds, each "
    "difference has mean zero. The two estimates share most of their error, so the test reads "
    "the standard error of the difference, not that of either estimate.",
    "covariance df is the effective degrees of freedom of the estimated covariance. The test "
    "scales the Wald statistic and reads it against an F reference whose second degrees of "
    "freedom come from it, which allows for a covariance estimated with error. A small value "
    "says a few rows or clusters carry most of the variance, or few fold splits were averaged.",
)

#: The small summaries of an MSM fit's working-model record and an SNMM's positivity record.
MSM_FIT_TITLE: Final = "Working-model coefficients"
MSM_TABLE: Final = "Coefficient of each term"
MSM_TERM: Final = "term"
MSM_COEFFICIENT: Final = "coefficient"
MSM_LINK: Final = "Link"
MSM_LINK_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"identity": "the mean", "log": "the log of the mean", "logit": "the log odds"}
)
MSM_CELLS: Final = "Plan and time cells"
MSM_CONVERGED: Final = "Converged"
MSM_CAUSE_LABEL: Final = "Cause"
MSM_FIT_READING: Final = "The result's summary() gives each coefficient with its interval."
MSM_HOW_TO_READ: Final = (
    "The coefficients project the plan means at every plan and time onto the working model, "
    "which is linear in {scale}. They summarize those means, so they keep their meaning when "
    "the working model is wrong."
)
DIAGNOSTICS_TITLE: Final = "Positivity of the structural fit"
DIAGNOSTICS_TABLE: Final = "What each time point measured"
DIAGNOSTICS_NODE: Final = "treatment time"
DIAGNOSTICS_TIME: Final = "time {time}"
DIAGNOSTICS_EIGENVALUE: Final = "smallest eigenvalue"
DIAGNOSTICS_EXTREME: Final = "near 0 or 1"
DIAGNOSTICS_CONSTANT: Final = "Fits with a one-class fold"
DIAGNOSTICS_READING: Final = (
    "smallest eigenvalue is the smallest standardized eigenvalue of the estimating "
    "equation's derivative at that time: near zero, the treatment varies little given the "
    "history, or the blip features are nearly collinear given it, so the coefficients are "
    "weakly identified. near 0 or 1 is the share of rows whose estimated treatment "
    "probability lies within the positivity margin of 0 or 1; a "
    "dose has none."
)

#: What one row of a longitudinal table estimates, keyed by the name's prefix and the
#: outcome kind. ``{Y}`` is the outcome (with ``{at}`` its time, empty at the end of study),
#: ``{d}`` the plan, ``{r}`` the reference plan, ``{t}`` the time and ``{cause}`` the cause.
LONGITUDINAL_MEANINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ey": "mean {Y}{at} if every unit followed plan {d}",
        "ate": "mean {Y}{at} if every unit followed plan {d}, minus that mean under plan {r}",
        "risk": "probability of the event by time {t} if every unit followed plan {d}",
        "risk_ate": (
            "probability of the event by time {t} if every unit followed plan {d}, minus "
            "that probability under plan {r}"
        ),
        "cif": "probability of {cause} by time {t} if every unit followed plan {d}",
        "cif_ate": (
            "probability of {cause} by time {t} if every unit followed plan {d}, minus "
            "that probability under plan {r}"
        ),
        "msm": "coefficient of {term} in the working model of the plan means",
    }
)

#: A competing-risks working-model coefficient names its cause.
MSM_CAUSE: Final = "{text}, for {cause}"

#: The Censoring header fact: the indicators, and that every plan keeps each unit observed,
#: so a row is its value had no unit been censored.
CENSORING_FACT: Final = "{names}; each plan keeps every unit observed"
#: The Outcome fact of a competing-risks fit.
COMPETING_FACT: Final = "{causes} (competing causes)"

#: The Question fact of a single-row longitudinal table, keyed as the meanings are.
LONGITUDINAL_QUESTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ey": "mean {Y}{at} under plan {d}",
        "ate": "average effect on {Y}{at} of plan {d} instead of {r}",
        "risk": "risk of the event by time {t} under plan {d}",
        "risk_ate": "change in the risk of the event by time {t}, plan {d} instead of {r}",
        "cif": "risk of {cause} by time {t} under plan {d}",
        "cif_ate": "change in the risk of {cause} by time {t}, plan {d} instead of {r}",
        "msm": "working-model coefficient of {term}",
    }
)

#: The Question fact when a longitudinal table reports several rows.
LONGITUDINAL_SEVERAL_ROWS: Final = "{Y} under each plan; each row is defined below"
#: The outcome a several-row Question names on a fit with an outcome at every time point.
REPEATED_OUTCOME: Final = "the outcome at each time point"
#: The Question fact of a working-model fit.
LONGITUDINAL_MSM_QUESTION: Final = (
    "working-model projection of the plan means of {Y}; each row is defined below"
)

#: The start of the fit-time warning a longitudinal ``weights_capped`` note restates.
CUMULATIVE_TRUNCATION_WARNING: Final = "Cumulative mechanism truncation exceeded"
#: Where a longitudinal note's largest share came from, on a fit with repeated splits.
IN_ONE_SPLIT: Final = ", in at least one split"

#: The titles and column labels of the longitudinal tables.
PLANS_TITLE: Final = "What each plan assigns and who followed it to the end"
PLANS_SPLIT_TITLE: Final = (
    "What each plan assigns and who followed it to the end (split one of {repeats})"
)
PLAN: Final = "plan"
ASSIGNS: Final = "assigns"
IN_PLAN: Final = "followed"
#: The count column of a tilt plan, which keeps every unit at risk: the units observed.
OBSERVED_TO_END: Final = "observed"
FOLLOWED_IN_DATA: Final = "followed in data"
LARGEST_WEIGHT: Final = "max weight"
EFFECTIVE_N: Final = "effective size"
BOOTSTRAP_BANDS_TITLE: Final = "Bootstrap sup-t {level} bands"

#: Plain words for one node of a plan.
PLAN_RULE: Final = "a declared rule"
PLAN_NAMED_RULE: Final = "rule {name}"
PLAN_POLICY: Final = "a known random draw"
PLAN_NAMED_POLICY: Final = "random draw {name}"
PLAN_TILT: Final = "odds of treatment times {delta}"
PLAN_TILT_LEVELS: Final = "odds of each level times {factors}"
PLAN_NATURAL: Final = "the data's own treatment process"
PLAN_SHIFT: Final = "the observed treatment {delta}{cap}"
PLAN_SCALE: Final = "the observed treatment times {factor}{cap}"
PLAN_NAMED_MTP: Final = "policy {name}"
PLAN_EVERY_TIME: Final = "{words} at every time point"
PLAN_AT_TIME: Final = "{words} at time {t}"
PLAN_HELD: Final = "{words} from baseline, held"

#: The conditions a longitudinal fit with no study rests on, in plain words.
LONGITUDINAL_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "consistency: each observed history equals the potential history under the plan it followed",
    "no interference: one unit's outcome does not depend on other units' plans",
    "sequential exchangeability: at every time point, treatment and remaining under "
    "observation are independent of the potential outcomes given the recorded history",
    "sequential positivity: at every time point, every unit on a plan has a positive "
    "probability of the plan's treatment and of remaining under observation",
)
#: What an odds tilt needs in place of treatment positivity (Kennedy 2019; Kim, Kennedy and
#: Naimi 2021).
TILT_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "an odds tilt needs no treatment positivity at a tilt time point, because it keeps the "
    "support of each arm (Kennedy 2019, Theorem 1); a fixed, rule or random-draw time point "
    "still needs positivity wherever its plan puts mass",
    "dropout is missing at random given the recorded history, and each unit at risk remains "
    "under observation with positive probability (Kim, Kennedy and Naimi 2021, Assumptions "
    "A2-M and A3)",
)
#: The construction of each estimator, with its source, for the Method details section.
LONGITUDINAL_CONSTRUCTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "tmle": (
            "A backward sequence of outcome regressions, each updated so that its estimating "
            "equation holds, from the last time point to the first."
        ),
        "sdr": (
            "Sequentially doubly robust estimation: a backward sequence of regressions of a "
            "transformed outcome, with no separate update step (Diaz, Williams, Hoffman and "
            "Schenck 2023, Section 5.3)."
        ),
        "ipsi": (
            "The mean of the influence function at the fitted models (Kennedy 2019, Algorithm "
            "1; Kim, Kennedy and Naimi 2021, Algorithm 1, under censoring). An odds tilt moves "
            "each unit's treatment probability pi to delta pi / (delta pi + 1 - pi)."
        ),
        "gcomp": (
            "The iterated conditional expectation plug-in (Bang and Robins 2005, Section 4), "
            "with no update step."
        ),
        "ipw": "Each unit's outcome weighted by the inverse probability of staying on the plan.",
        "parametric_gformula": (
            "Forward simulation of the fitted covariate, outcome and hazard models under each "
            "plan (Robins 1986; McGrath et al. 2020), with no update step."
        ),
    }
)
#: The caveat on a rule's follower counts, for the Settings section.
POLICY_FOLLOWERS: Final = (
    "A plan that draws each unit's treatment at random counts as followed by the units whose "
    "observed treatment the draw can give."
)
RULE_FOLLOWERS: Final = (
    "A rule's followers depend on their covariates, so the counts in the plan table describe "
    "this sample."
)
#: The ratio-diagnostic column labels of the support tables.
SUPPORT_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "regime": "regime",
        "min_propensity": "smallest probability",
        "max_ratio": "largest ratio",
        "effective_n": "effective sample size",
        "score_load": "score load",
        "unsupported": "rows without support",
    }
)

EVERY_ROW: Final = "every row"
SECTION_HOW_TO_READ: Final = "How to read"

#: The notes of the structural, g-formula and overlap summaries (X89 T2).
T2_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "structural_model": "note",
        "check_splits": "note",
        "draw_order": "note",
        "simulation_share": "note",
        "unsupported_rows": "warning",
        "score_load": "note",
        "score_load_rows": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T2_NOTE_CODES})

T2_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "check_experimental": "cleverly marks this check experimental. {reason}",
        "check_splits": (
            "Each number is the mean over {k} fold splits, the first of them the fit's own."
        ),
        "draw_order": (
            "The order in which the covariates of one time point are drawn defines the "
            "models, so at most one order is correctly specified in general."
        ),
        "score_load": (
            "Score load reads the limited treatment model the fit used, while the ratios and "
            "their effective size read the estimated model before any limit. It counts the "
            "rows that effectively carry the estimating equation, of the rows the equation "
            "reads, so it is not the effective size of the ratios."
        ),
        "score_load_rows": (
            "Score load counts the rows that effectively carry the estimating equation, of the "
            "rows the equation reads, so it is not the effective size of the ratios."
        ),
        "simulation_share": (
            "The simulation contributes up to {share} of the variance ({name}); a larger "
            "n_draws lowers it."
        ),
        "structural_check": "result.check_model() compares the plan contrasts with LTMLE.",
        "structural_covariate_feature": (
            "A blip feature reads a covariate measured after baseline, so the fit reports the "
            "blips and the reference mean but no plan means or contrasts."
        ),
        "structural_model": (
            "The intervals hold under the linear structural nested mean model with the blip "
            "features declared at each time point. A blip that depends on an undeclared "
            "earlier treatment or covariate makes the coefficients, and the plan means read "
            "from them, biased."
        ),
        "unsupported_density": (
            "{name} moves {count} rows to a dose where the estimated density is exactly zero. "
            "This flags the density model; it does not prove that the effect is unidentified."
        ),
        "unsupported_rows": (
            "Rows given an arm whose estimated probability is exactly zero: {counts}. The data "
            "give no support there. If the true probability is zero, the effect is not "
            "identified for those rows."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T2_NOTE_TEXTS})


# ----------------------------------------------------------------- point-estimate objects (T1)

#: Header labels of the summaries of a single estimate, a test, a bootstrap, a cross-fitted
#: report and the method records a point fit keeps.
ESTIMAND: Final = "Estimand"
NULL_VALUE: Final = "Null value"
SCALE: Final = "Scale"
REFERENCE: Final = "Reference"
RESAMPLING: Final = "Resampling"
REPLICATES: Final = "Replicates asked for"
ESTIMANDS_COVERED: Final = "Estimands"
FAILED: Final = "Failed"
INTERVAL: Final = "Interval"
FOLDS: Final = "Folds"
SPLITS: Final = "Splits"
CANDIDATES: Final = "Candidates"
CHOSEN: Final = "Chosen"
CRITERION: Final = "Criterion"
SEARCH: Final = "Search"
COVARIATES_SEARCHED: Final = "Covariates searched"
ADJUSTMENT: Final = "Multiple testing"
LOSS: Final = "Loss"
BEST: Final = "Best candidate"
ENSEMBLE_RISK: Final = "Ensemble CV risk"
GUARD: Final = "Extra equations for"
REDUCTION: Final = "Extra regressions"
FORM: Final = "Form"
ROUNDS: Final = "Rounds"
STOPPED: Final = "Stopped"
SOURCE: Final = "Source"
TRAINING_SETS: Final = "Training sets"
INTERMEDIATE: Final = "Intermediate"
CRITICAL_VALUE: Final = "Critical value"
MULTIPLIER_DRAWS: Final = "Multiplier draws"
MULTIPLIER_KIND: Final = "Multiplier weights"
STATISTIC: Final = "statistic"

HEADER_LABELS = HEADER_LABELS | {
    ESTIMAND,
    NULL_VALUE,
    SCALE,
    REFERENCE,
    RESAMPLING,
    REPLICATES,
    ESTIMANDS_COVERED,
    FAILED,
    INTERVAL,
    FOLDS,
    SPLITS,
    CANDIDATES,
    CHOSEN,
    CRITERION,
    SEARCH,
    COVARIATES_SEARCHED,
    ADJUSTMENT,
    LOSS,
    BEST,
    ENSEMBLE_RISK,
    GUARD,
    REDUCTION,
    FORM,
    ROUNDS,
    STOPPED,
    DESIGN,
    STRATA,
    SOURCE,
    TRAINING_SETS,
    INTERMEDIATE,
    CRITICAL_VALUE,
    MULTIPLIER_DRAWS,
    MULTIPLIER_KIND,
}

#: Titles of the T1 summaries.
ESTIMATE_ONE_TITLE: Final = "Estimate: {name}"
WALD_TITLE: Final = "Wald test: {name}"
BOOTSTRAP_TITLE: Final = "Bootstrap replicates"
BOOTSTRAP_INTERVAL_TITLE: Final = "Bootstrap interval"
BANDS_TITLE: Final = "Simultaneous {level} bands"
SPREAD_TITLE: Final = "Spread across cross-fitting splits"
SEARCH_TITLE: Final = "Collaborative treatment-model search"
ENSEMBLE_TITLE: Final = "Super learner"
IMPORTANCE_TITLE: Final = "Variable importance"
OAT_TITLE: Final = "Outcome-adaptive treatment model"
DERIVATIVE_TITLE: Final = "Collaborative TMLE, derivative form"
REDUCED_TITLE: Final = "DR-TMLE extra regressions"
IPTW_TITLE: Final = "Adaptive IPTW fit"
CV_TITLE: Final = "Cross-validated estimates"
RESULT_SET_TITLE: Final = "Controlled direct effects ({method})"
#: The Question of a result set over several levels of the intermediate variable.
RESULT_SET_QUESTION: Final = "effect of {A} on {Y} with {Z} held at each level below for every unit"

#: The parts of an outcome-adaptive per-arm treatment model, keyed by the start of the
#: factor label ``CTMLEOutcomeAdaptiveFit.factors`` records.
OAT_FACTORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "P(A=a": "treatment",
        "P(Delta=1": "outcome recorded",
        "P(Delta_A=1": "treatment recorded",
    }
)
#: Where a derivative-form fit took its two treatment models, keyed by ``source``.
DERIVATIVE_SOURCES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lasso": "lasso path",
        "learners": "two declared learners",
        "arrays": "two declared probability arrays",
    }
)
#: The two forms of an Adaptive IPTW estimate, keyed by ``form``.
IPTW_FORMS: Final[Mapping[str, str]] = MappingProxyType(
    {"targeted": "targeted", "one_step": "one-step"}
)
#: Why the Adaptive IPTW weight adjustment stopped, keyed by ``stop``.
IPTW_STOPS: Final[Mapping[str, str]] = MappingProxyType(
    {"tolerance": "converged", "cap": "round limit"}
)
#: The spread columns of a cross-validated report, keyed by its frame column.
CV_SPREAD_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cv_std_err": "CV std. error",
        "cv_plugin_std_err": "CV SE (diagnostic)",
        "pooled_std_err": "std. error",
        "pooled_plugin_std_err": "plug-in SE (diagnostic)",
    }
)
#: The row label of a one-row bootstrap table.
REPLICATES_ROW: Final = "replicates"
#: The kind of a bootstrap interval, keyed by ``BootstrapSummary.interval``.
BOOTSTRAP_INTERVAL_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"percentile": "percentile", "bca": "BCa"}
)
#: What a bootstrap resamples, keyed by ``BootstrapResult.resampling``.
RESAMPLING_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {"iid": "rows", "cluster": "clusters"}
)

#: The row label of a result set's estimate at one level of the intermediate variable.
LEVEL_ROW: Final = "{name} at {Z} = {z}"

#: Plain words for the scale an estimate's interval is built on, keyed by ``scale``.
SCALE_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "difference": "difference",
        "level": "level",
        "fraction": "fraction",
        "ratio": "ratio (interval on the log scale)",
    }
)

#: Plain words for each collaborative search, keyed by the ``strategy=`` of ``CTMLE``.
CTMLE_SEARCHES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "greedy": "adding covariates one at a time",
        "ordered": "adding covariates in a fixed order",
        "discrete": "over declared covariate sets",
        "lasso": "along a lasso penalty path",
        "learners": "over declared learners",
        "oat": "outcome-adaptive",
        "derivative": "derivative form",
    }
)

#: The order an ``ordered`` search adds covariates in, keyed by ``preorder``.
CTMLE_PREORDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "logistic": "ranked by a logistic fit of the treatment",
        "partial_correlation": "ranked by partial correlation with the outcome residual",
        "custom": "as declared",
    }
)

#: The losses a collaborative search and a super learner score candidates by.
LOSS_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "loglik": "negative log-likelihood",
        "squared": "squared error",
        "mse": "squared error",
        "squared_error": "squared error",
        "nll": "negative log-likelihood",
        "neg_log_likelihood": "negative log-likelihood",
        "log_loss": "negative log-likelihood",
        "negative_log_likelihood": "negative log-likelihood",
    }
)

#: What each DR-TMLE guard protects against, keyed by the guard name.
GUARD_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Q": "a wrong outcome model",
        "g": "a wrong treatment model",
    }
)

#: What the extra equations of each DR-TMLE guard are shown to do, keyed by the guard
#: name. ``drtmle.py``'s module docstring states the conditions: the one-wrong interval is
#: valid for a known mechanism, conservative for a correctly specified parametric one, not
#: established for a data-adaptive learner, and not established at all with the mechanism
#: wrong.
GUARD_SENTENCES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Q": (
            "The fit solved extra equations meant to keep the interval valid when the "
            "outcome model is wrong. With the outcome model wrong, the interval is valid when "
            "the treatment probabilities are known, conservative (wider than needed) when "
            "they come from a correctly specified parametric model, and not established for "
            "a flexible learner."
        ),
        "g": (
            "The fit solved the extra equations meant for a wrong treatment model. When the "
            "treatment model is wrong, no result shows that the interval keeps its coverage."
        ),
    }
)

#: The Method details line of a DR-TMLE fit whose design restricts the treatment model to
#: ``{names}``: when the restriction fails, the treatment model is wrong, and the
#: mechanism-wrong rows of the one-wrong table of the DR-TMLE theorem page decide the interval.
DRTMLE_RESTRICTED: Final = (
    "The treatment model reads only {names}, as the design declares. If treatment depends "
    "on other covariates too, the treatment model is wrong, and no result shows that the "
    "interval keeps its coverage."
)
#: The Method details lines of an Adaptive IPTW fit whose design restricts the treatment
#: model. It fits no outcome model, so it is consistent only when the omitted covariates
#: confound nothing beyond ``{names}``.
IPTW_RESTRICTED: Final = "The treatment model reads only {names}, as the design declares."
IPTW_RESTRICTED_OMITTED: Final = (
    " The estimate is consistent only when {omitted} add no confounding beyond those."
)
#: The Method details line of a collaborative search drawn from a declared pool.
SEARCH_POOL: Final = "The search drew only from {names}, as the design declares."

#: The extra regressions of each DR-TMLE construction, keyed by ``reduction``.
REDUCTION_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "univariate": "three one-dimensional",
        "bivariate": "two, one of them two-dimensional",
        "missing_outcome": "five, for missing outcomes",
    }
)

#: Plain names of the DR-TMLE extra regressions, keyed by ``ReducedFit.diagnostics`` key.
REDUCED_REGRESSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "qr": "outcome error on treatment probability",
        "gr1": "treatment on outcome prediction",
        "gr1_bivariate": "treatment on outcome and treatment predictions",
        "gr2": "weighted treatment error on outcome prediction",
        "gamma_a": "treatment on outcome prediction",
        "gamma_m": "outcome recorded on outcome prediction",
        "r_a": "weighted treatment error on outcome prediction",
        "r_m": "weighted recording error on outcome prediction",
        "e": "outcome error on P(treated and recorded)",
    }
)

#: The column labels of the T1 tables.
REPLICATE_COLUMNS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "usable": "usable",
        "mean": "mean",
        "sd": "SD of replicates",
        "smallest": "smallest",
        "largest": "largest",
    }
)
SPREAD_DIAGNOSTIC: Final = "spread (diagnostic)"
SPREAD_RATIO: Final = "spread / SE"
SPREAD_RATIO_PLUGIN: Final = "spread / plug-in SE"
SPREAD_PLUGIN_SE: Final = "plug-in SE"
FOLD_AVERAGED: Final = "fold-averaged"
WHOLE_SAMPLE: Final = "whole-sample"
FOLD_SD: Final = "fold SD"
FOLD_RANGE: Final = "fold range"
ADJUSTED_P_VALUE: Final = "adjusted p-value"
CV_RISK: Final = "CV risk"
IN_SAMPLE_RISK: Final = "in-sample risk"
WEIGHT: Final = "weight"
TREATMENT_MODEL_COLUMN: Final = "treatment model"
LOWER: Final = "lower"
UPPER: Final = "upper"
#: The marks a search table puts beside the candidate the fit reports.
CHOSEN_MARK: Final = "chosen"
FINAL_MARK: Final = "final step"

#: More note codes, used by the point-estimate objects (thread T1).
T1_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        "rounds_capped": "warning",
        "levels_separate": "note",
        "null_scale": "note",
        "bootstrap_failed": "warning",
        "bootstrap_describes": "note",
        "split_spread": "note",
        "spread_missing": "note",
        "cv_two_reports": "note",
        "cv_first_split": "note",
        "left_out": "note",
        "search_stopped": "note",
        "risk_order": "warning",
        "stacked_interval": "note",
        "ordinary_tmle": "note",
        "multiple_testing": "note",
        "separate_fits": "note",
        "learners_failed": "warning",
        "ensemble_risk": "note",
        "fit_cautions": "warning",
        "stacked_refused": "note",
        "rows_shown": "note",
        "guard_scope": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T1_NOTE_CODES})

T1_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "rounds_capped": (
            "The weight adjustment stopped at its round limit ({rounds}) before the "
            "weights converged for every estimate, so an estimate may keep part "
            "of the bias the adjustment is meant to remove."
        ),
        "levels_separate": (
            "Each level of {Z} is a different parameter. The levels do not split the total "
            "effect into direct and indirect parts, and their difference is an interaction "
            "contrast, not a mediated effect."
        ),
        "null_scale": (
            "The test compares the estimate with the null value on the {scale} scale, where "
            "the interval is built."
        ),
        "bootstrap_failed": (
            "{failed} of {requested} replicates gave no usable estimate and are left out. If "
            "those were the resamples where a treatment group was rarest, the remaining "
            "spread, and any interval from it, can be too narrow."
        ),
        "bootstrap_describes": (
            "These numbers describe the replicate estimates. The fitted result's summary "
            "says whether they give a confidence interval for this fit."
        ),
        "bootstrap_describes_one": (
            "These numbers describe the replicate estimates of one row. The fitted result's "
            "summary says whether the interval is a confidence interval for this fit."
        ),
        "split_spread": (
            "Each spread is the standard deviation of the estimate across {repeats} random "
            "splits into folds. It shows how much the split moved the estimate; it is a "
            "diagnostic, not a standard error. {tail}"
        ),
        "spread_missing": (
            "{names} {verb} no spread: a split gave a value that is not finite on the scale "
            "of the interval."
        ),
        "cv_two_reports": (
            "The fold-averaged estimate evaluates the parameter inside each fold and "
            "averages the folds, with {spread}. The whole-sample estimate puts the held-out "
            "predictions together and evaluates them once. They agree for the means and the "
            "average effect when every fold carries the same total weight, and they can "
            "differ for ratios and for effects among the treated or untreated."
        ),
        "cv_first_split": (
            "Both estimates combine {repeats} splits by their median; the fold columns "
            "describe the first split only."
        ),
        "left_out": (
            "The chosen treatment model leaves out {names}. The search chose by the "
            "cross-validated criterion of the adjusted outcome predictions; it does not show "
            "why a covariate was left out."
        ),
        "search_stopped": (
            "The search stopped after candidate {stopped}: {patience} since the best one so "
            "far had a higher cross-validated risk, which is the stopping rule. Later "
            "candidates were {later}."
        ),
        "search_never_stopped": (
            "The stopping rule (patience {patience}) never triggered, so every candidate "
            "was scored."
        ),
        "risk_order": (
            "The declared learners are not ordered from least to most adaptive: the "
            "treatment-model risk rises at candidate {names}. The search assumes that order."
        ),
        "ordinary_tmle": (
            "No extra regressions were fitted (an empty guard), so the estimate is the "
            "ordinary TMLE."
        ),
        "multiple_testing": (
            "The adjusted p-values control the false discovery rate over the {count} tests "
            "({method}) if the tests are independent or positively dependent, which these "
            "fits do not guarantee. The intervals and the unadjusted p-values are not "
            "adjusted."
        ),
        "separate_fits": (
            "Each row comes from its own fit with that candidate as the treatment; the "
            "full summary lists what each fit adjusts for. Each row needs its own no-unmeasured-"
            "confounding and positivity conditions, and a row is not the candidate's share "
            "of one effect."
        ),
        "learners_failed": "These candidates failed to fit and were left out: {names}.",
        "ensemble_risk": (
            "Each candidate's CV risk is scored on held-out folds. The ensemble's weights "
            "were chosen on those same held-out predictions, so its CV risk is optimistic "
            "beside the candidates' risks."
        ),
        "fit_cautions": (
            "The fits of {names} carry warnings that change how to read their rows; print "
            "each with result.fits[name].summary()."
        ),
        "stacked_interval": (
            "Each interval is pointwise and comes from one stacked estimating equation of "
            "the declared parametric outcome and treatment models{weights}. It covers the "
            "arm mean when the outcome model is correct; otherwise it covers the value the "
            "estimate settles at, which can move with the sample size while a limit on the "
            "treatment probabilities binds."
        ),
        "stacked_weights": (
            ", with the data weights in every equation, so the interval treats the weights as fixed"
        ),
        "stacked_weights_estimated": (
            ", with the data weights in every equation, so the interval treats the weights "
            "as fixed although they are declared estimated"
        ),
        "stacked_refused": "No stacked interval: {reason}",
        "rows_shown": (
            "The table shows {shown} of the {total} candidates: the first, the last, the "
            "chosen one and each one whose covariates change. selection.to_frame() lists "
            "them all."
        ),
        "spread_withheld": (
            "No interval for {names}, so the std. error column is empty there; the spread "
            "is still a diagnostic of the split."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T1_NOTE_TEXTS})

#: The end of the ``split_spread`` note, by whether the fit supplies an interval.
SPLIT_SPREAD_TAILS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "inference": "The reported std. error already includes it.",
        "withheld": ("The fit gives no interval, so the plug-in SE beside it is a diagnostic too."),
    }
)
#: What the fold-averaged estimate of a cross-validated report carries, by the same rule.
CV_SPREAD_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "inference": "the cross-validated standard error",
        "withheld": (
            "a cross-validated spread that is a diagnostic here, as the fit gives no interval"
        ),
    }
)
#: The Adaptive IPTW cap warning ``rounds_capped`` restates.
RESTATED_WARNINGS = MappingProxyType(
    {
        **RESTATED_WARNINGS,
        "rounds_capped": ("before the relative score reached tol=",),
        "risk_order": ("strategy='learners' needs treatment_candidates= ordered",),
    }
)

# ----------------------------------------------------------------- sensitivity analyses (T4)

#: Header labels of the sensitivity summaries. ``POPULATION`` and ``SCALE`` above serve too.
ESTIMATE_ANALYSED: Final = "Estimate"
OUTCOME_STRENGTH: Final = "Outcome-side strength"
TREATMENT_STRENGTH: Final = "Treatment-side strength"
ALIGNMENT: Final = "Alignment"
LARGEST_SHIFT: Final = "Largest shift"
CONCLUSION: Final = "Conclusion"
BENCHMARK_COVARIATES: Final = "Benchmark covariates"
IMPLIED_ALIGNMENT: Final = "Implied alignment"
GRID: Final = "Grid"
MISSING_OUTCOMES: Final = "Missing outcomes"
DIRECTION: Final = "Direction"
MOVEMENT: Final = "Movement"
CELLS: Final = "Cells"
REPEATED_SPLITS: Final = "Repeated splits"

HEADER_LABELS = HEADER_LABELS | {
    ESTIMATE_ANALYSED,
    OUTCOME_STRENGTH,
    TREATMENT_STRENGTH,
    ALIGNMENT,
    LARGEST_SHIFT,
    CONCLUSION,
    BENCHMARK_COVARIATES,
    IMPLIED_ALIGNMENT,
    GRID,
    MISSING_OUTCOMES,
    DIRECTION,
    MOVEMENT,
    CELLS,
    REPEATED_SPLITS,
}

T4_NOTE_CODES: Final[Mapping[str, NoteKind]] = MappingProxyType(
    {
        # The sensitivity analyses (X89 T4).
        "sensitivity_assumption": "note",
        "no_inference_sensitivity": "warning",
        "limits_withheld": "warning",
        "limits_descriptive": "warning",
        "rv_limit": "note",
        "alignment_stated": "note",
        "joint_strength": "note",
        "consistent_mechanism": "note",
        "strength_clipped": "note",
        "grid_thinned": "note",
        "approximate_scale": "note",
        "censoring_conditions": "note",
        "per_arm_conditions": "note",
        "bound_truncated": "note",
        "tilt_diagnostic": "note",
        "association_reading": "note",
        "population_moves": "note",
        "incremental_reading": "note",
        "learned_rule_reading": "note",
        "observation_reads": "note",
        "bounded_rows": "note",
        "redrawn_folds": "note",
        "cells_failed": "warning",
        "calibration_nan": "note",
    }
)
NOTE_CODES = MappingProxyType({**NOTE_CODES, **T4_NOTE_CODES})

T4_NOTE_TEXTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        # The sensitivity analyses (X89 T4). ``assumption_<analysis>`` texts are the
        # ``sensitivity_assumption`` note of each analysis; ``verdict_*`` texts are its
        # one reading sentence.
        "no_inference_sensitivity": (
            "No confidence statement for {names}. {reason} The {label} is a diagnostic, "
            "and it claims no coverage."
        ),
        "assumption_bounds": (
            "The range is the estimate moved down and up by the largest bias that variables "
            "the adjustment left out could, at worst, cause at the stated strengths. cf_y is "
            "the share of the outcome variation left after adjustment that they would "
            "explain, cf_d the share of the average squared weight that only they would add, "
            "and rho how far their two effects line up (one is the worst case)."
        ),
        "assumption_bounds_limits": (
            "Each one-sided limit is a {level} confidence limit for its own end of the range: "
            "the lower limit for the lower end, the upper limit for the upper end."
        ),
        "verdict_rv_reached": (
            "Left-out variables with both strengths at {rv} could, at worst, move the "
            "estimate to {null}{alignment}; weaker ones, with both strengths below {rv}, "
            "could not."
        ),
        "verdict_rv_at_null": (
            "The estimate already equals {null}, so the robustness value is zero."
        ),
        "verdict_rv_unreached": (
            "No left-out variables with equal strengths below one could move the estimate to "
            "{null}{alignment}."
        ),
        "rv_limit": (
            "The {limit} robustness value (RVa) is the same strength for the {level} "
            "one-sided limit nearest {null}: below it, that limit stays on its side of "
            "{null}."
        ),
        "rv_limit_plugin": (
            "The plug-in limit robustness value is the same strength for the plug-in limit "
            "nearest {null}, a diagnostic that claims no coverage: below it, that limit "
            "stays on its side of {null}."
        ),
        "assumption_robustness": (
            "The robustness value is the strength that left-out variables would need, with "
            "the same value for cf_y (the share of the outcome variation left after "
            "adjustment that they would explain) and cf_d (the share of the average squared "
            "weight that only they would add), to move the estimate to {null}."
        ),
        "alignment_stated": (
            "rho = {rho} assumes the two effects of the left-out variables line up less "
            "than in the worst case, where rho is one."
        ),
        "assumption_elements": (
            "The bias scale is the square root of the product of the first two rows. A bound "
            "multiplies it by a factor that grows with the strengths of the left-out "
            "variables, so these rows give the range at any strengths."
        ),
        "assumption_benchmark": (
            "A common calibration (Chernozhukov et al.) takes these as the strengths that "
            "left-out variables would have if leaving them out hurt the fit as much as "
            "leaving out {covariates} does. Leaving those out moved the estimate by {delta}."
        ),
        "benchmark_pointer": (
            "Pass this record as benchmark= to result.sensitivity.omitted_confounding() for "
            "the range at these strengths."
        ),
        "assumption_contour": (
            "Each cell is the {bound} end of the range for {estimand} when the variables the "
            "adjustment left out have the outcome-side strength of its row (cf_y) and the "
            "treatment-side strength of its column (cf_d), at alignment {rho}."
        ),
        "verdict_evalue": (
            "An unmeasured confounder would need a risk ratio of {point} with both the "
            "treatment and the outcome, beyond the measured covariates, to explain away the "
            "estimate; {limit_sentence}"
        ),
        "verdict_evalue_per_arm": (
            "An unmeasured confounder would need a risk ratio of {point} in each of four "
            "associations, with the treatment and with the outcome in each of the two arms, "
            "beyond the measured covariates, to explain away the estimate; {limit_sentence}"
        ),
        "evalue_limit_reached": "{limit} to move the interval to the null.",
        "evalue_limit_covers": "the interval already includes the null.",
        "assumption_evalue": (
            "The E-value is the strength at which the confounder's associations with the "
            "treatment and with the outcome, taken as equal, could explain the estimate "
            "away. Unequal strengths can trade off: one association may be below the "
            "E-value if the other is above it."
        ),
        "assumption_evalue_per_arm": (
            "The estimate holds the intermediate fixed, so its bias is a product of one "
            "bounding factor per arm, and the E-value is the equal strength of all four "
            "risk-ratio associations, two per arm. Unequal strengths can trade off."
        ),
        "evalue_conversion_or": (
            "The odds ratio was converted to a risk ratio by its square root, an "
            "approximation for common outcomes that can land above or below the true risk "
            "ratio."
        ),
        "evalue_conversion_rd": (
            "The risk difference was converted to a risk ratio with the estimated risk of "
            "the reference arm, held fixed, so the sampling error of that risk is ignored."
        ),
        "evalue_conversion_smd": (
            "The mean difference was divided by the outcome's standard deviation and "
            "converted to a risk ratio through an odds ratio; each step is an approximation."
        ),
        "evalue_derived": (
            "The risk ratio {name} was derived from the models fitted for the reported "
            "{source}, with no model refit."
        ),
        "censoring_conditions": (
            "The risk is the chance of the event by the horizon. The E-value also assumes "
            "that censoring is random given the treatment and covariates, and that the "
            "unmeasured confounder does not affect censoring."
        ),
        "per_arm_conditions": (
            "It also assumes the confounder is a baseline variable the treatment does not "
            "affect, that every unit could receive each treatment and intermediate level, "
            "and that no common cause of the intermediate and the outcome is affected by "
            "the treatment."
        ),
        "assumption_marginal": (
            "gamma is the largest factor by which an unmeasured confounder may change a "
            "unit's odds of treatment, given the covariates, in either direction, with no "
            "limit on its link to the outcome. The range holds every value of {estimand} "
            "that allows, if this confounder is all the adjustment missed. At a gamma of one, no "
            "unmeasured confounding, the range shrinks to one value, the bound's own doubly "
            "robust estimate, close to the fit's."
        ),
        "marginal_interval": (
            "In large samples the interval contains the true range with probability at "
            "least {level}, if either the treatment model or the outcome-quantile model is "
            "right."
        ),
        "assumption_tilt": (
            "gamma is how far the average outcome of units whose outcome is missing differs "
            "from that of recorded units with the same treatment and covariates, on the "
            "log-odds scale of the outcome rescaled to run from zero to one{arm_clause}. A "
            "gamma of zero is the main analysis, which assumes outcomes are missing at "
            "random. {direction}"
        ),
        "tilt_direction": "A positive gamma means the missing outcomes were higher.",
        "tilt_direction_arms": (
            "Where gamma times an arm's factor is positive, that arm's missing outcomes were "
            "higher; a factor of zero leaves the arm at missing at random."
        ),
        "tilt_interval": (
            "Where an interval is printed off a gamma of zero, it reuses the main analysis's "
            "standard error and treats gamma as known, so it describes sampling error at "
            "that gamma, not doubt about gamma."
        ),
        "assumption_surface": (
            "Each cell adds a simulated common cause of {moved} and the outcome, refits the "
            "analysis, and reports how far the estimate moved. The surface is qualitative: "
            "it is not a bound and carries no interval. {laws}"
        ),
        "law_binary": (
            "The treatment strength is the share of units, those highest on the simulated "
            "cause, whose treatment is flipped."
        ),
        "law_categorical": (
            "The treatment strength is the share of units in arms {a} and {b}, those highest "
            "on the simulated cause, whose arms are swapped."
        ),
        "law_continuous": (
            "The treatment strength is how far the dose moves per standard deviation of the "
            "simulated cause."
        ),
        "law_vector": (
            "Each binary component flips for a share of units, those highest on the "
            "simulated cause, set by its strength, and each dose moves by its strength per "
            "standard deviation of the cause."
        ),
        "law_longitudinal": (
            "Each treatment node flips for a share of units, those highest on the simulated "
            "cause, set by its strength; one cause value per unit is shared by every node "
            "and the outcome."
        ),
        "law_intermediate": (
            "Every cell holds the treatment and flips the intermediate {name} for a share of "
            "units, those highest on the simulated cause, so the cause confounds the "
            "intermediate and the outcome."
        ),
        "law_outcome_gaussian": (
            "The outcome moves by its strength, with the sign reversed, per standard "
            "deviation of the simulated cause."
        ),
        "law_outcome_binomial": (
            "The outcome flips for a share of units, those highest on the simulated cause, "
            "set by the outcome strength."
        ),
        "law_outcome_logit": (
            "The outcome moves on the log-odds scale of its declared bounds, so it stays "
            "inside them, and the outcome strength is not in standard-deviation units: "
            "compare cells across fits by their outcome association."
        ),
        "law_outcome_normal_score": (
            "The outcome moves on a normal-score scale and is mapped back inside its "
            "declared bounds, so the law depends on the sample, and the outcome strength is "
            "not in standard-deviation units: compare cells across fits by their outcome "
            "association."
        ),
        "association_defined": (
            "An association column is the correlation, on these data, between the simulated "
            "cause and the treatment{intermediate} the cell produced{stratum}."
        ),
        "incremental_reading": (
            "Each cell refits the treatment model, and the shift in treatment odds reads "
            "that refitted model, so a cell moves the intervention along with the "
            "confounding."
        ),
        "learned_rule_reading": (
            "Each cell keeps every unit's learned rule fixed, so it reads confounding of "
            "those fixed rules, not of the procedure that learned them."
        ),
        "observation_reads": (
            "The cells that move the treatment are valid only if whether a value is recorded "
            "depends on {reads} alone, as declared."
        ),
        "bounded_rows": (
            "A share {share} of recorded outcomes sits at a bound and never moves in any cell."
        ),
        "redrawn_folds": (
            "Each refit redraws the cross-fitting folds, so every movement includes fold "
            "noise; the anchor is the original fit."
        ),
        "limits_withheld_learner": (
            "No one-sided limits or limit robustness value: the plug-in estimate of the "
            "average squared weight has a standard error only for a "
            "declared known treatment model or a qualifying scikit-learn LogisticRegression, "
            "and this fit's treatment model does not qualify. {alternative}"
        ),
        "limits_withheld_configuration": (
            "No one-sided limits or limit robustness value: this bound's weight also reads "
            "a second model or a stratum share, and the standard error of the plug-in "
            "estimate covers the treatment model alone. "
            "{alternative}"
        ),
        "limits_withheld_correction": (
            "No one-sided limits or limit robustness value: the plug-in estimate of the "
            "average squared weight needs a correction before it "
            "has a standard error, and the correction does not hold on this fit. "
            "{alternative}"
        ),
        "limits_withheld_fit": (
            "No one-sided limits or limit robustness value: the standard error of the "
            "plug-in estimate of the average squared weight cannot be built from this "
            "fit's stored treatment model. {alternative}"
        ),
        "limits_withheld_consistent": (
            "No one-sided limits or limit robustness value: with "
            "assume_consistent_mechanism=True the bound is a point bound, and no cited "
            "source gives its standard error. {alternative}"
        ),
        "limits_withheld_zero": (
            "No one-sided limits or limit robustness value: the bias scale is zero on this "
            "fit, and a zero scale has no standard error."
        ),
        "limits_descriptive": (
            "No validation study licenses the {what} for this kind of fit, so read it as "
            "descriptive."
        ),
        "joint_strength": (
            "Here the treatment-side strength measures how much the left-out variables add "
            "to the weights of the treatment and of {mechanisms} together."
        ),
        "consistent_mechanism": (
            "This point bound also assumes that the treatment model behind its weights is "
            "correct (assume_consistent_mechanism=True); the package does not check it."
        ),
        "strength_clipped_y": (
            "Leaving out {covariates} raised the outcome variation left by a factor above "
            "two (gain {gain}), so cf_y is one: left-out variables this strong would explain "
            "all of the outcome variation left after adjustment."
        ),
        "strength_clipped_d": (
            "The treatment-side gain is {gain}, at least one, so cf_d is one, where the bound "
            "has no finite value."
        ),
        "grid_thinned": (
            "The table shows {rows} of the {y_size} cf_y values and {columns} of the "
            "{d_size} cf_d values; to_frame() holds every cell."
        ),
        "bound_truncated": (
            "The converted lower limit, {value}, lies below zero, outside the risk-ratio "
            "scale, so the table shows zero."
        ),
        "bound_truncated_upper": (
            "The converted lower limit, {value}, lies below zero, outside the risk-ratio "
            "scale. The limit nearest the null is the upper one, which this does not touch."
        ),
        "tilt_diagnostic": (
            "For {names} the spread off a gamma of zero is the main analysis's spread around the "
            "moved estimate{scale}, a diagnostic: no interval is claimed there."
        ),
        "population_moves": (
            "Each cell rebuilds the population of units with {arm_phrase} from its own "
            "treatment, so part of a movement can be a change of population; the population "
            "share column shows the share of units in that group."
        ),
        "cells_failed": (
            "{failed} of {total} cells failed, so their movement is missing. "
            "surface.failures gives each error."
        ),
    }
)
NOTE_TEXTS = MappingProxyType({**NOTE_TEXTS, **T4_NOTE_TEXTS})

#: Symbols a short sensitivity summary leaves to ``detail="full"`` (X89 plan 8.1 row T4).
SHORT_DETAIL_SYMBOLS: Final[tuple[str, ...]] = ("sigma^2", "nu^2", "Riesz")
