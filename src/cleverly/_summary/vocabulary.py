"""Every word a result summary shares with another summary.

A summary is printed by several modules, and two modules that spell one fact two ways
teach a reader that they are two facts. Each label, column name, method name, learner
name, note code and evidence phrase that more than one summary can print lives here once.
A seam that needs a new shared word adds it here, with its test, in its own change.

Nothing here is public. Users meet these strings in printed summaries only.
"""

from __future__ import annotations

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
#: The study-setup and simulation summaries (X89 T5).
DESIGN: Final = "Design"
TREATMENT: Final = "Treatment"
BASELINE: Final = "Baseline"
FOLLOWED: Final = "Observed to the end"
EFFECTIVE_SAMPLE_SIZE: Final = "Effective sample size"
CENSORED_BEFORE_END: Final = "Censored before the end"
STRATA: Final = "Strata"
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
        BASELINE,
        FOLLOWED,
        EFFECTIVE_SAMPLE_SIZE,
        CENSORED_BEFORE_END,
        STRATA,
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
        # The study-setup and simulation summaries (X89 T5).
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
        # The study-setup and simulation summaries (X89 T5).
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
)

# ----------------------------------------------------------------- checks (T3)

#: Plain evidence words for each refutation rung (X89 plan 6.6). A verdict needs L2 or
#: above (``validation/refute.py``).
EVIDENCE_WORDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "L0": "description only, no pass or fail",
        "L1": (
            "compared with a stated reference; how often it fails a correct fit is not "
            "measured, so no pass or fail"
        ),
        "L2": "pass or fail from an exact reference or a simulation study of this kind of fit",
        "L3": "as L2, with an external package's result beside it",
    }
)

# ----------------------------------------------------------------- study setup (T5)

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
            "not a confidence limit."
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
            "nearest {null}, a diagnostic and not a confidence limit: below it, that limit "
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
