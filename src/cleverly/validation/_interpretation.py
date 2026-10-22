r"""The registry of evidence that licenses a refutation verdict.

A refutation test computes a statistic. Whether the report may print ``passed`` or ``failed``
beside it depends on evidence about the rule, not on the availability of the test. This module
holds that evidence as data: :data:`VERDICT_EVIDENCE` is a tuple of :class:`VerdictEvidence`
rows, one per rule, and :func:`resolve` answers one question at report time. Does a row cover
this fit? A rule with no covering row prints its numbers and no verdict (``passed is None``).

The ladder
----------

======  ====================================================================================
rung    what backs the reading
======  ====================================================================================
``L0``  a description of what moved; no reference law
``L1``  a stated reference law whose size or power has no measurement
``L2``  an exact reference law, or a measured size and power on the fit's configuration
``L3``  an L2 reading with a comparator or an external truth beside it
======  ====================================================================================

A row at ``L2`` or ``L3`` names a registered study and its cells unless its reference is exact.
A reference is exact when it follows from the construction (a generated law with a known truth)
and not from an asymptotic argument. :class:`CalibrationScope` records the configurations the
cells covered. A fit outside the scope gets the descriptive form, and the report prints the
covered configuration in one line.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = [
    "DESCRIPTIVE_SENTENCE",
    "RUNGS",
    "VERDICT_EVIDENCE",
    "CalibrationCell",
    "CalibrationScope",
    "FitConfiguration",
    "Resolution",
    "Rung",
    "VerdictEvidence",
    "check_scope_containment",
    "fit_configuration",
    "learner_fingerprint",
    "resolve",
]

Rung = Literal["L0", "L1", "L2", "L3"]

#: The rungs in order. A rule at ``L2`` or above may print a verdict.
RUNGS: Final[tuple[Rung, ...]] = ("L0", "L1", "L2", "L3")

#: ``HistGradientBoosting*`` switches early stopping on above this many rows when
#: ``early_stopping="auto"`` (scikit-learn ``gradient_boosting.py``). A calibration at fewer
#: rows therefore does not cover a fit above it.
_BOOSTER_EARLY_STOPPING_ROWS: Final[int] = 10_000

_BOOSTER_CLASSES: Final[tuple[str, ...]] = (
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
)


def learner_fingerprint(learner: Any, n: int) -> str:
    """Name a nuisance learner by its class and its resolved parameters.

    The declared parameters are not enough where a parameter resolves with the row count.
    ``early_stopping="auto"`` on a ``HistGradientBoosting*`` learner stops early above 10,000
    rows, so the fingerprint writes the resolved value. The resolution reads the full row
    count of the fit, although each learner trains on (k - 1)/k of it under k-fold
    cross-fitting. That reads early stopping as on slightly more often than the fit ran it,
    which keeps a row from covering a fit that it did not calibrate.

    Parameters
    ----------
    learner : object or None
        A scikit-learn compatible estimator, or ``None`` for the default library.
    n : int
        Number of rows the learner fits.

    Returns
    -------
    str
        ``"default-library"`` for ``None``, else the class name and the sorted parameters.
    """
    if learner is None:
        return "default-library"
    name = type(learner).__name__
    getter = getattr(learner, "get_params", None)
    if getter is None:
        return name
    try:
        params = dict(getter(deep=False))
    except Exception:  # a learner whose parameters cannot be read is named by its class
        return name
    if name in _BOOSTER_CLASSES and params.get("early_stopping") == "auto":
        params["early_stopping"] = bool(n > _BOOSTER_EARLY_STOPPING_ROWS)
    else:
        # Name only the parameters that differ from the class default, so the line stays
        # short. A class that cannot be built without arguments names all of them.
        try:
            defaults = dict(type(learner)().get_params(deep=False))
        except Exception:
            defaults = {}
        params = {
            key: value for key, value in params.items() if defaults.get(key, object()) != value
        }
    fields = ",".join(f"{key}={params[key]!r}" for key in sorted(params))
    return f"{name}({fields})"


@dataclass(frozen=True)
class FitConfiguration:
    """The configuration of a fit, read off the fitted result.

    Every field is a scope axis of :class:`CalibrationScope`. The values come from the fitted
    result and its data, not from the declaration that produced them.

    Parameters
    ----------
    method : str
        The fitted method, for example ``"TMLE"`` or ``"CTMLE"``.
    estimand_kind : str
        The parameter axis and estimand kind, for example ``"arm:ate"``.
    outcome_family : str
        The outcome family of the fit.
    treatment_kind : str
        ``"discrete"``, ``"continuous"`` or ``"vector"``.
    learners : tuple of str
        One :func:`learner_fingerprint` per nuisance learner.
    cross_fitting : str
        ``"off"``, or ``"folds=<k>"``.
    inference : str
        The inference status of the estimate.
    known_mechanism : bool
        Whether the treatment mechanism is declared known.
    n : int
        Number of rows.
    covariates_per_row : float
        Adjustment columns divided by ``n``.
    min_event_count : int or None
        The smallest count of events, or of non-events, in any arm. ``None`` for a
        continuous outcome.
    clusters : int or None
        Number of clusters, or ``None`` on an unclustered fit.
    arms : int
        Number of treatment levels.
    replicates : int or None
        The replicate count R of the test that asks. ``None`` where the question is not about
        a replicate count. :func:`~cleverly.validation.refute.refute` sets it per test.
    nodes : int or None
        The number of treatment nodes ``T`` of a longitudinal fit, and ``None`` on a
        point-treatment fit.
    """

    method: str
    estimand_kind: str
    outcome_family: str
    treatment_kind: str
    learners: tuple[str, ...]
    cross_fitting: str
    inference: str
    known_mechanism: bool
    n: int
    covariates_per_row: float
    min_event_count: int | None
    clusters: int | None
    arms: int
    replicates: int | None = None
    nodes: int | None = None


def event_count(data: Any) -> int | None:
    """Return the smallest count of events or non-events in any arm.

    Parameters
    ----------
    data : CausalData
        The analysis rows.

    Returns
    -------
    int or None
        ``None`` unless the outcome family is binomial.
    """
    arm_codes = tuple(getattr(data, "arm_codes", ()) or ())
    if str(data.family) != "binomial" or not arm_codes:
        return None
    counts: list[int] = []
    outcome = np.asarray(data.outcome, dtype=float)
    treatment = np.asarray(data.treatment, dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    for code in arm_codes:
        in_arm = observed & (treatment == code)
        events = int(np.count_nonzero(outcome[in_arm] > 0.5))
        counts.extend((events, int(in_arm.sum()) - events))
    return min(counts)


def fit_configuration(result: TMLEResult, estimand: str) -> FitConfiguration:
    """Read the configuration of ``result`` for one estimand.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        The alias the test runs for.

    Returns
    -------
    FitConfiguration
        The scope axes as the fitted result records them.
    """
    if getattr(result, "assessment_family", None) == "longitudinal":
        return _longitudinal_configuration(result, estimand)
    data = result.data
    estimator = result.estimator
    estimate = result[estimand]
    n = int(data.n)
    # Only a learner that the fit runs enters the fingerprint: a declared known mechanism
    # fits no treatment learner, and a fully observed outcome fits no missingness learner.
    used = ["outcome_learner"]
    if data.known_treatment is None:
        used.append("treatment_learner")
    if not bool(np.all(data.observed)):
        used.append("missingness_learner")
    learners = tuple(learner_fingerprint(getattr(estimator, name, None), n) for name in used)
    if getattr(estimator, "cross_fit", True):
        cross_fitting = f"folds={int(getattr(estimator, 'n_folds', 0))}"
    else:
        cross_fitting = "off"
    arm_codes = tuple(getattr(data, "arm_codes", ()) or ())
    family = str(data.family)
    min_events = event_count(data)
    clusters = None if data.cluster is None else int(np.unique(data.cluster).size)
    return FitConfiguration(
        method=type(estimator).__name__,
        estimand_kind=str(getattr(estimate, "name", estimand)),
        outcome_family=family,
        treatment_kind=str(data.treatment_kind),
        learners=learners,
        cross_fitting=cross_fitting,
        inference=str(estimate.inference),
        known_mechanism=data.known_treatment is not None,
        n=n,
        covariates_per_row=float(data.covariates.shape[1]) / max(1, n),
        min_event_count=min_events,
        clusters=clusters,
        arms=max(1, len(arm_codes)),
    )


def _longitudinal_event_count(data: Any) -> int | None:
    """Return the smaller of the event and non-event counts at the last node, or ``None``.

    A survival panel counts the units with an event by ``T`` among those uncensored
    through ``T``. A binomial end-of-study outcome counts the uncensored outcome values.
    A Gaussian or repeated outcome has no event count.
    """
    if data.is_repeated:
        return None
    observed = np.asarray(data.uncensored_through(data.n_times), dtype=bool)
    if data.is_survival:
        events = np.asarray(data.event_by(data.n_times), dtype=float)[observed]
    elif str(data.family) == "binomial":
        events = np.asarray(data.outcome, dtype=float)[observed]
    else:
        return None
    count = int(np.count_nonzero(events > 0.5))
    return min(count, int(events.size) - count)


def _longitudinal_learners(estimator: Any, data: Any) -> tuple[Any, ...]:
    """The learners whose fits move a longitudinal estimate, in a fixed order.

    The targeted estimators fit the outcome, pseudo-outcome, treatment and censoring
    regressions. An unset pseudo learner defaults to the outcome learner. The baselines fit
    fewer. Sequential g-computation fits only its outcome
    regressions, and its pseudo learner defaults to the outcome learner. Its mechanisms feed
    the positivity diagnostics only. Sequential IPW fits only the mechanisms. The parametric
    g-formula fits a covariate model per time-varying column, an outcome or hazard model,
    and the natural-treatment models.
    """
    method = type(estimator).__name__
    known = data.known_mechanisms
    fits_treatment = known is None or not all(
        known.treatment_at(time) is not None
        for time in range(1, data.n_times + 1)
        if not data.is_held_node(time)
    )
    mechanisms = []
    if fits_treatment:
        mechanisms.append(estimator.treatment_learner)
    if data.has_censoring:
        mechanisms.append(estimator.censoring_learner)
    # An unset pseudo learner reads the outcome learner: g-computation uses it as it is, and
    # the targeted estimators read its library as a regression.
    pseudo = getattr(estimator, "pseudo_learner", None)
    regressions = (
        estimator.outcome_learner,
        estimator.outcome_learner if pseudo is None else pseudo,
    )
    if method == "SequentialGComputation":
        return regressions
    if method == "SequentialIPW":
        return tuple(mechanisms)
    if method == "ParametricGFormula":
        from ..learners import GLM

        # ``None`` is the documented main-terms GLM default of each of these models.
        def resolved(learner: Any) -> Any:
            return GLM() if learner is None else learner

        has_covariates = any(block.shape[1] for block in data.time_varying)
        models = [resolved(estimator.covariate_learner)] if has_covariates else []
        outcome = estimator.hazard_learner if data.is_survival else estimator.outcome_learner
        models.append(resolved(outcome))
        return (*models, *mechanisms)
    return (*regressions, *mechanisms)


def _longitudinal_configuration(result: Any, estimand: str) -> FitConfiguration:
    """Read the scope axes of a longitudinal fit for one estimand.

    The estimand kind is the parameter stem, such as ``"ate_regimen"``, because a
    regimen's label is the user's own text. The outcome family is the outcome kind of a
    survival, competing-risk or repeated panel, and the outcome family otherwise.
    """
    from ..targets.base import parameter_stem

    data = result.data
    estimator = result.estimator
    estimate = result[estimand]
    n = int(data.n)
    known = data.known_mechanisms
    learners = tuple(
        learner_fingerprint(learner, n) for learner in _longitudinal_learners(estimator, data)
    )
    folds = int(getattr(estimator, "n_folds", 1))
    kind = str(result.config.outcome_kind)
    if data.has_continuous_node:
        treatment_kind = "continuous"
    elif any(len(columns) > 1 for columns in data.treatment_components):
        treatment_kind = "vector"
    else:
        treatment_kind = "discrete"
    width = data.baseline.shape[1] + sum(block.shape[1] for block in data.time_varying)
    arms = max((len(levels) for levels in data.treatment_levels), default=1)
    return FitConfiguration(
        method=type(estimator).__name__,
        estimand_kind=parameter_stem(estimand),
        outcome_family=kind if kind != "end_of_study" else str(data.family),
        treatment_kind=treatment_kind,
        learners=learners,
        cross_fitting=f"folds={folds}" if folds > 1 else "off",
        inference=str(estimate.inference),
        known_mechanism=bool(known is not None and known.declares_treatment),
        n=n,
        covariates_per_row=float(width) / max(1, n),
        min_event_count=_longitudinal_event_count(data),
        clusters=None if data.cluster is None else int(np.unique(data.cluster).size),
        arms=max(1, arms),
        nodes=int(data.n_times),
    )


def _frozen(values: Iterable[Any] | None) -> frozenset[Any] | None:
    return None if values is None else frozenset(values)


@dataclass(frozen=True)
class CalibrationScope:
    """The configurations a rule's calibration cells covered.

    A categorical axis holds the set of covered values. ``None`` leaves an axis unrestricted,
    which only a row with an exact reference may do. A fit matches a row only when every axis
    matches. The numeric axes follow the plan's match rules: a lower bound on ``n``, the
    event count and the cluster count, and an upper bound on the covariate count over ``n``
    and, where a learner's behaviour switches with ``n``, on ``n``.

    Parameters
    ----------
    methods, estimand_kinds, outcome_families, treatment_kinds : frozenset of str or None
        The covered values of the first four axes of :class:`FitConfiguration`.
    learners : frozenset of str or None
        Every nuisance fingerprint of a covered fit lies in this set.
    cross_fitting, inference : frozenset of str or None
        The covered cross-fitting settings and inference statuses.
    known_mechanism : frozenset of bool or None
        The covered values of the known-mechanism declaration.
    arms : frozenset of int or None
        The covered treatment level counts.
    replicates : frozenset of int or None
        The covered replicate counts R. A row calibrated at one R covers no other.
    nodes : frozenset of int or None, or None
        The covered treatment node counts ``T``. ``None`` inside the set stands for a
        point-treatment fit, which has no node count, so a point row holds
        ``frozenset({None})``.
    n_min : int
        The smallest covered ``n``.
    n_max : int or None
        The largest covered ``n``, or ``None`` where no learner switches behaviour with ``n``.
    covariates_per_row_max : float or None
        The largest covered adjustment columns over ``n``.
    min_event_count_min : int or None
        The smallest covered event count, for a binary or event outcome.
    clusters_min : int or None
        The smallest covered cluster count. ``None`` leaves a clustered fit uncovered.
    """

    methods: frozenset[str] | None = None
    estimand_kinds: frozenset[str] | None = None
    outcome_families: frozenset[str] | None = None
    treatment_kinds: frozenset[str] | None = None
    learners: frozenset[str] | None = None
    cross_fitting: frozenset[str] | None = None
    inference: frozenset[str] | None = None
    known_mechanism: frozenset[bool] | None = None
    arms: frozenset[int] | None = None
    replicates: frozenset[int] | None = None
    nodes: frozenset[int | None] | None = None
    n_min: int = 0
    n_max: int | None = None
    covariates_per_row_max: float | None = None
    min_event_count_min: int | None = None
    clusters_min: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "methods",
            "estimand_kinds",
            "outcome_families",
            "treatment_kinds",
            "learners",
            "cross_fitting",
            "inference",
            "known_mechanism",
            "arms",
            "replicates",
            "nodes",
        ):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    @classmethod
    def unrestricted(cls) -> CalibrationScope:
        """Return the scope that covers every fit.

        Returns
        -------
        CalibrationScope
            Only a row with an exact reference may use it.
        """
        return cls()

    def mismatch(self, fit: FitConfiguration | None) -> str | None:
        """Name the first axis on which ``fit`` lies outside this scope.

        Parameters
        ----------
        fit : FitConfiguration or None
            The configuration to test. ``None`` is a fit whose configuration could not be
            read, which only an unrestricted scope covers.

        Returns
        -------
        str or None
            The axis name and the fit's value, or ``None`` when the fit is covered.
        """
        if fit is None:
            return None if self == CalibrationScope.unrestricted() else "an unreadable fit"
        pairs: tuple[tuple[str, frozenset[Any] | None, Any], ...] = (
            ("method", self.methods, fit.method),
            ("estimand kind", self.estimand_kinds, fit.estimand_kind),
            ("outcome family", self.outcome_families, fit.outcome_family),
            ("treatment kind", self.treatment_kinds, fit.treatment_kind),
            ("cross-fitting", self.cross_fitting, fit.cross_fitting),
            ("inference status", self.inference, fit.inference),
            ("known mechanism", self.known_mechanism, fit.known_mechanism),
            ("arm count", self.arms, fit.arms),
            ("replicate count", self.replicates, fit.replicates),
            ("node count", self.nodes, fit.nodes),
        )
        for axis, covered, value in pairs:
            if covered is not None and value not in covered:
                return f"{axis} {value!r}"
        if self.learners is not None:
            for fingerprint in fit.learners:
                if fingerprint not in self.learners:
                    return f"learner {fingerprint}"
        if fit.n < self.n_min:
            return f"n = {fit.n} (below {self.n_min})"
        if self.n_max is not None and fit.n > self.n_max:
            return f"n = {fit.n} (above {self.n_max})"
        if (
            self.covariates_per_row_max is not None
            and fit.covariates_per_row > self.covariates_per_row_max
        ):
            return f"covariates per row {fit.covariates_per_row:.4g}"
        if self.min_event_count_min is not None and (
            fit.min_event_count is None or fit.min_event_count < self.min_event_count_min
        ):
            return f"minimum event count {fit.min_event_count}"
        if fit.clusters is not None and (
            self.clusters_min is None or fit.clusters < self.clusters_min
        ):
            return f"cluster count {fit.clusters}"
        return None

    def covers(self, fit: FitConfiguration | None) -> bool:
        """Return whether ``fit`` lies inside this scope.

        Parameters
        ----------
        fit : FitConfiguration or None
            The configuration to test.

        Returns
        -------
        bool
            ``True`` when every axis matches.
        """
        return self.mismatch(fit) is None

    def describe(self) -> str:
        """Return the covered configuration in one line.

        Returns
        -------
        str
            The restricted axes, or ``"every configuration"``.
        """
        parts: list[str] = []
        for label, values in (
            ("methods", self.methods),
            ("estimands", self.estimand_kinds),
            ("outcome families", self.outcome_families),
            ("treatment kinds", self.treatment_kinds),
            ("learners", self.learners),
            ("cross-fitting", self.cross_fitting),
            ("inference", self.inference),
        ):
            if values is not None:
                parts.append(f"{label} {sorted(map(str, values))}")
        if self.n_min:
            parts.append(f"n >= {self.n_min}")
        if self.n_max is not None:
            parts.append(f"n <= {self.n_max}")
        return "; ".join(parts) if parts else "every configuration"


@dataclass(frozen=True)
class VerdictEvidence:
    """One row of evidence that licenses a verdict.

    Parameters
    ----------
    instrument : str
        The refutation test name, for example ``"placebo"``.
    rule : str
        The reading the row licenses, for example ``"t"`` or ``"bias_band"``.
    rung : {"L0", "L1", "L2", "L3"}
        The evidence rung. A verdict needs ``"L2"`` or ``"L3"``.
    theory : str
        A locator for the theory of the rule, and a statement of what is measured, not proved.
    scope : CalibrationScope
        The configurations the evidence covers.
    sentence : str
        The only interpretation text the test may print when the row covers the fit.
    study : tuple of str
        The registered study id, then the size and power cell ids. Required for ``"L2"`` and
        ``"L3"`` unless ``exact`` is true.
    exact : bool
        Whether the reference follows from the construction, so no study is needed.
    """

    instrument: str
    rule: str
    rung: Rung
    theory: str
    scope: CalibrationScope
    sentence: str
    study: tuple[str, ...] = ()
    exact: bool = False

    def __post_init__(self) -> None:
        if self.rung not in RUNGS:
            raise ValueError(f"unknown rung {self.rung!r}; choose from {list(RUNGS)}")
        if self.rung in ("L2", "L3") and not (self.exact or self.study):
            raise ValueError(
                f"{self.instrument}/{self.rule} claims {self.rung} without a registered study "
                "and without an exact reference"
            )
        if self.scope == CalibrationScope.unrestricted() and not self.exact:
            raise ValueError(
                f"{self.instrument}/{self.rule} declares an unrestricted scope without an "
                "exact reference"
            )


@dataclass(frozen=True)
class Resolution:
    """The answer of :func:`resolve` for one reading.

    Parameters
    ----------
    rung : {"L0", "L1", "L2", "L3"}
        The rung the reading prints at.
    licensed : bool
        Whether the rung licenses a verdict.
    sentence : str
        The interpretation sentence the test may print.
    evidence : VerdictEvidence or None
        The covering row, or ``None`` when the reading is descriptive.
    """

    rung: Rung
    licensed: bool
    sentence: str
    evidence: VerdictEvidence | None = None


DESCRIPTIVE_SENTENCE = (
    "No calibration covers this configuration, so the numbers are descriptive and the "
    "report states no verdict."
)

#: Rows that license a verdict. A row with ``exact=False`` is added by the registered study
#: that measured its size and power, in the pull request that registers the study.
VERDICT_EVIDENCE: tuple[VerdictEvidence, ...] = (
    VerdictEvidence(
        instrument="generated_outcome",
        rule="bias_band",
        rung="L2",
        theory=(
            "X82 plan 4.2.1. The draws are i.i.d. given the held design, so the bias and its "
            "Monte Carlo standard error are exact for every target."
        ),
        scope=CalibrationScope(nodes=frozenset({None})),
        sentence=(
            "On the generated law named here, the bias lies inside its equivalence band. "
            "Your own outcome law is not this one."
        ),
        exact=True,
    ),
    VerdictEvidence(
        instrument="generated_outcome",
        rule="se_ratio_band",
        rung="L2",
        theory=(
            "X82 plan 4.2.1 and 5.5.1. Where the covariate part of the influence curve is "
            "identically zero on the generated law, the held-design standard error "
            "reference is the unconditional one."
        ),
        scope=CalibrationScope(inference=frozenset({"influence_curve"}), nodes=frozenset({None})),
        sentence=(
            "On the generated law named here, the standard-error ratio lies inside its "
            "equivalence band. Your own outcome law is not this one."
        ),
        exact=True,
    ),
    VerdictEvidence(
        instrument="generated_outcome",
        rule="coverage_band",
        rung="L2",
        theory=(
            "X82 plan 4.2.1 and 5.5.1. The Clopper-Pearson interval of the coverage is "
            "exact given i.i.d. draws, on a law with a zero covariate part."
        ),
        scope=CalibrationScope(inference=frozenset({"influence_curve"}), nodes=frozenset({None})),
        sentence=(
            "On the generated law named here, the 95% interval coverage lies inside its "
            "equivalence band. Your own outcome law is not this one."
        ),
        exact=True,
    ),
    VerdictEvidence(
        instrument="plasmode",
        rule="bias_band",
        rung="L2",
        theory=(
            "X82 plan 4.2.1 and 5.6. The plasmode draws are i.i.d. units from a law whose "
            "covariates are the empirical support, so the truth is an exact finite sum and "
            "the bias and its Monte Carlo standard error are exact."
        ),
        scope=CalibrationScope.unrestricted(),
        sentence=(
            "On the plasmode law named here, the bias lies inside its equivalence band. "
            "The plasmode resembles your data and does not estimate its truth."
        ),
        exact=True,
    ),
    VerdictEvidence(
        instrument="plasmode",
        rule="se_ratio_band",
        rung="L2",
        theory=(
            "X82 plan 4.2.1 and 5.6. The plasmode resamples the covariates, so the fit's "
            "standard error targets the unconditional variance across the i.i.d. draws."
        ),
        scope=CalibrationScope(inference=frozenset({"influence_curve"})),
        sentence=(
            "On the plasmode law named here, the standard-error ratio lies inside its "
            "equivalence band. The plasmode resembles your data and does not estimate its truth."
        ),
        exact=True,
    ),
    VerdictEvidence(
        instrument="plasmode",
        rule="coverage_band",
        rung="L2",
        theory=(
            "X82 plan 4.2.1 and 5.6. The Clopper-Pearson interval of the coverage is exact "
            "given i.i.d. plasmode draws."
        ),
        scope=CalibrationScope(inference=frozenset({"influence_curve"})),
        sentence=(
            "On the plasmode law named here, the 95% interval coverage lies inside its "
            "equivalence band. The plasmode resembles your data and does not estimate its truth."
        ),
        exact=True,
    ),
)


def resolve(
    instrument: str,
    rule: str,
    fit: FitConfiguration | None,
    *,
    evidence: Iterable[VerdictEvidence] | None = None,
) -> Resolution:
    """Ask the registry what a reading may print for ``fit``.

    Parameters
    ----------
    instrument : str
        The refutation test name, or ``"generated_outcome"`` for the two held-design
        generated tests.
    rule : str
        The reading.
    fit : FitConfiguration or None
        The configuration of the fit under test, or ``None`` when it cannot be read.
    evidence : iterable of VerdictEvidence or None
        The rows to search. ``None`` reads :data:`VERDICT_EVIDENCE` at call time, so a test
        may patch the module attribute.

    Returns
    -------
    Resolution
        A licensed resolution when a row of rung ``L2`` or above covers the fit. Otherwise a
        descriptive resolution whose sentence names the nearest row's covered configuration.
    """
    rows = VERDICT_EVIDENCE if evidence is None else tuple(evidence)
    candidates = [row for row in rows if row.instrument == instrument and row.rule == rule]
    nearest: VerdictEvidence | None = None
    for row in candidates:
        if row.scope.covers(fit):
            if row.rung in ("L2", "L3"):
                return Resolution(rung=row.rung, licensed=True, sentence=row.sentence, evidence=row)
            nearest = nearest or row
    if nearest is not None:
        return Resolution(
            rung=nearest.rung,
            licensed=False,
            sentence=DESCRIPTIVE_SENTENCE,
            evidence=nearest,
        )
    if candidates:
        row = candidates[0]
        mismatch = row.scope.mismatch(fit)
        return Resolution(
            rung="L0",
            licensed=False,
            sentence=(
                f"{DESCRIPTIVE_SENTENCE} Covered: {row.scope.describe()}. "
                f"This fit differs on {mismatch}."
            ),
        )
    return Resolution(rung="L0", licensed=False, sentence=DESCRIPTIVE_SENTENCE)


@dataclass(frozen=True)
class CalibrationCell:
    """One calibration cell as a study recorded it, for :func:`check_scope_containment`.

    Parameters
    ----------
    kind : {"size", "power"}
        Whether the cell measures the rule's size under the null or its power.
    passed : bool
        Whether the cell passed its declared threshold.
    configuration : FitConfiguration
        The configuration the fit recorded, read off the fitted result.
    """

    kind: Literal["size", "power"]
    passed: bool
    configuration: FitConfiguration = field(repr=False)


def check_scope_containment(
    row: VerdictEvidence,
    cells: Mapping[str, CalibrationCell] | Iterable[tuple[str, CalibrationCell]],
) -> list[str]:
    """List the ways a non-exact row's scope is wider than its named cells.

    Parameters
    ----------
    row : VerdictEvidence
        The row to check.
    cells : mapping of str to cell
        The recorded cells, by id. Each holds the cell kind, whether it passed, and the
        configuration the fit recorded.

    Returns
    -------
    list of str
        One message per violation. An empty list means the scope lies inside the cells.
        An exact row has nothing to contain and returns an empty list.
    """
    if row.exact:
        return []
    table = dict(cells)
    named = [table[cell_id] for cell_id in row.study[1:] if cell_id in table]
    problems: list[str] = []
    for kind in ("size", "power"):
        passing = [cell for cell in named if cell.kind == kind and cell.passed]
        if not passing:
            problems.append(f"{row.instrument}/{row.rule}: no passing {kind} cell")
            continue
        scope = row.scope
        axes: tuple[tuple[str, frozenset[Any] | None, Any], ...] = (
            ("methods", scope.methods, lambda c: c.method),
            ("estimand_kinds", scope.estimand_kinds, lambda c: c.estimand_kind),
            ("outcome_families", scope.outcome_families, lambda c: c.outcome_family),
            ("treatment_kinds", scope.treatment_kinds, lambda c: c.treatment_kind),
            ("cross_fitting", scope.cross_fitting, lambda c: c.cross_fitting),
            ("inference", scope.inference, lambda c: c.inference),
            ("known_mechanism", scope.known_mechanism, lambda c: c.known_mechanism),
            ("arms", scope.arms, lambda c: c.arms),
            ("replicates", scope.replicates, lambda c: c.replicates),
            ("nodes", scope.nodes, lambda c: c.nodes),
        )
        unrestricted = False
        for axis, covered, _read in axes:
            if covered is None:
                problems.append(f"{row.instrument}/{row.rule}: axis {axis} is unrestricted")
                unrestricted = True
        if scope.learners is None:
            problems.append(f"{row.instrument}/{row.rule}: axis learners is unrestricted")
            unrestricted = True
        if not unrestricted:
            # A fit is covered when its whole combination of categorical values is, so every
            # tuple in the product of the row's axes must be one passing cell's configuration.
            # Each axis passing alone is not enough: cells TMLE-gaussian and CTMLE-binomial
            # leave TMLE-binomial uncovered.
            recorded = {
                (*(read(c.configuration) for _, _, read in axes), c.configuration.learners)
                for c in passing
            }
            assert scope.learners is not None
            ran = {fp for c in passing for fp in c.configuration.learners}
            if not scope.learners <= ran:
                problems.append(
                    f"{row.instrument}/{row.rule}: a learner has no passing {kind} cell"
                )
            by_length: dict[int, list[tuple[str, ...]]] = {}
            for c in passing:
                by_length.setdefault(len(c.configuration.learners), []).append(
                    c.configuration.learners
                )
            for values in product(*(sorted(covered, key=str) for _, covered, _ in axes if covered)):
                for length, tuples in sorted(by_length.items()):
                    # Each position (outcome, treatment, missingness) takes the fingerprints
                    # that the row names and a cell ran in that position.
                    positions = [
                        sorted({t[i] for t in tuples} & scope.learners) for i in range(length)
                    ]
                    for fingerprints in product(*positions):
                        if (*values, fingerprints) not in recorded:
                            problems.append(
                                f"{row.instrument}/{row.rule}: the combination "
                                f"{[str(v) for v in values]} with learners {list(fingerprints)} "
                                f"has no passing {kind} cell"
                            )
        smallest = min(c.configuration.n for c in passing)
        if scope.n_min < smallest:
            problems.append(f"{row.instrument}/{row.rule}: n_min {scope.n_min} < cell n {smallest}")
        if scope.n_max is not None and scope.n_max > max(c.configuration.n for c in passing):
            problems.append(f"{row.instrument}/{row.rule}: n_max above every {kind} cell")
        ratio = max(c.configuration.covariates_per_row for c in passing)
        if scope.covariates_per_row_max is None or scope.covariates_per_row_max > ratio:
            problems.append(
                f"{row.instrument}/{row.rule}: covariates per row bound above the {kind} cells"
            )
        events = [
            c.configuration.min_event_count
            for c in passing
            if c.configuration.min_event_count is not None
        ]
        if events and (
            scope.min_event_count_min is None or scope.min_event_count_min < min(events)
        ):
            problems.append(
                f"{row.instrument}/{row.rule}: event-count bound below the {kind} cells"
            )
        clusters = [c.configuration.clusters for c in passing if c.configuration.clusters]
        if scope.clusters_min is not None and (not clusters or scope.clusters_min < min(clusters)):
            problems.append(f"{row.instrument}/{row.rule}: cluster bound below the {kind} cells")
    return problems
