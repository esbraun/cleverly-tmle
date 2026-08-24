"""Capability-aware post-fit diagnostics, validation, and replay metadata.

Assessment lives on a fitted result because its questions depend on the artifacts the
method actually produced.  The facades in this module do not infer support from a result
class and hope for the best: every public operation has a declaration for every public
scalar result family, including deliberate refusals.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Any, Literal

import numpy as np

from ._typing import FloatArray
from .data.weighting import effective_sample_size
from .exceptions import CapabilityError
from .utils.frames import emit_frame
from .utils.text import format_table
from .validation.drtmle import IDENTITY_TOLERANCE
from .validation.score import DEFAULT_TOLERANCE

__all__ = [
    "ASSESSMENT_CAPABILITIES",
    "SENSITIVITY_ROUTES",
    "AssessmentCapability",
    "AssessmentItem",
    "AssessmentStatus",
    "DiagnosticReport",
    "DiagnosticsFacade",
    "LongitudinalDiagnostics",
    "LongitudinalNuisanceDiagnostics",
    "LongitudinalScoreDiagnostics",
    "Replayability",
    "SensitivityFacade",
    "SensitivityRoute",
    "ValidationReport",
    "assessment_capabilities",
    "replayability",
    "validate_result",
]


class AssessmentStatus(StrEnum):
    """Status returned by a diagnostic or validation operation.

    ``NOT_APPLICABLE`` means that the operation does not apply to the fitted
    estimand. ``UNAVAILABLE`` means that the operation applies, but the result
    does not contain the artifacts needed to run it.

    Parameters
    ----------
    *values
        Present because :class:`enum.StrEnum` gives every member a synthetic
        constructor. Reach a status by name, as ``AssessmentStatus.PASSED``.
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AssessmentCapability:
    """Describe whether and how a result supports one assessment operation.

    Parameters
    ----------
    operation : str
        Public operation name on a diagnostics or sensitivity facade.
    result_family : str
        Result family for which the declaration applies.
    methods : tuple of str
        Estimation methods covered by the declaration.
    available : bool
        Whether the operation can run on the result family.
    status : AssessmentStatus
        Status to report when the operation cannot run.
    required_artifacts : tuple of str
        Fitted artifacts that the operation reads.
    execution : {"summarize", "retarget", "refit"}
        Most expensive work the operation performs.
    deterministic_from_saved : bool
        Whether saved artifacts determine the result without a refit.
    interpretation : str
        Statistical question that the operation answers.
    cost : {"cheap", "moderate", "expensive"}
        Relative cost category used by combined reports.
    reason : str or None
        Explanation when the operation is not available.
    requires_arguments : tuple of str
        Caller-supplied arguments that have no default.
    """

    operation: str
    result_family: str
    methods: tuple[str, ...]
    available: bool
    status: AssessmentStatus
    required_artifacts: tuple[str, ...]
    execution: Literal["summarize", "retarget", "refit"]
    deterministic_from_saved: bool
    interpretation: str
    cost: Literal["cheap", "moderate", "expensive"]
    reason: str | None = None
    #: Arguments the caller must supply for which this operation has no default, so a
    #: combined report cannot run it. Declared here rather than special-cased by name in
    #: ``run_all``, which knows nothing about any particular operation.
    requires_arguments: tuple[str, ...] = ()


def _capability(
    operation: str,
    family: str,
    *,
    artifacts: Sequence[str],
    interpretation: str,
    execution: Literal["summarize", "retarget", "refit"] = "summarize",
    deterministic: bool = True,
    cost: Literal["cheap", "moderate", "expensive"] = "cheap",
    available: bool = True,
    status: AssessmentStatus = AssessmentStatus.PASSED,
    reason: str | None = None,
    requires_arguments: Sequence[str] = (),
) -> AssessmentCapability:
    return AssessmentCapability(
        operation=operation,
        result_family=family,
        methods=("tmle", "collaborative_tmle", "drtmle") if family == "point" else ("tmle",),
        available=available,
        status=status,
        required_artifacts=tuple(artifacts),
        execution=execution,
        deterministic_from_saved=deterministic,
        interpretation=interpretation,
        cost=cost,
        reason=reason,
        requires_arguments=tuple(requires_arguments),
    )


# This is intentionally data rather than branches hidden inside the facade.  Contract tests
# check both directions: every operation covers every family, and no declaration names a
# family or operation that the public result surface no longer exposes.
ASSESSMENT_CAPABILITIES: tuple[AssessmentCapability, ...] = (
    _capability(
        "support",
        "point",
        artifacts=("fitted analytic mechanism", "targeting weights"),
        interpretation="arm, regimen, shift, or incremental support for the identified functional",
    ),
    _capability(
        "nuisance_models",
        "point",
        artifacts=("out-of-fold nuisance predictions",),
        interpretation="held-out fit quality and calibration of each fitted nuisance",
    ),
    _capability(
        "score_equations",
        "point",
        artifacts=("targeting state", "influence curves"),
        interpretation="the score equations the selected method actually solved",
    ),
    _capability(
        "corrections",
        "point",
        artifacts=("doubly-robust correction state",),
        interpretation="the correction identities solved by guarded doubly-robust targeting",
    ),
    _capability(
        "truncation_curve",
        "point",
        artifacts=("fitted nuisance predictions", "targeting state"),
        execution="retarget",
        cost="moderate",
        interpretation="estimate stability across declared mechanism bounds",
    ),
    _capability(
        "refute",
        "point",
        artifacts=("fitted estimator configuration", "analysis data"),
        execution="refit",
        deterministic=False,
        cost="expensive",
        interpretation="behavior under placebo, noise, and subsampling perturbations",
    ),
    _capability(
        "stagewise",
        "point",
        artifacts=(),
        available=False,
        status=AssessmentStatus.NOT_APPLICABLE,
        reason="a point-treatment fit has no sequential nodes",
        interpretation="node-specific longitudinal recursion diagnostics",
    ),
    _capability(
        "support",
        "longitudinal",
        artifacts=("cumulative mechanism products", "regimen histories"),
        interpretation="history-specific support and cumulative leverage at every node",
    ),
    _capability(
        "nuisance_models",
        "longitudinal",
        artifacts=("node pseudo-outcomes", "initial node predictions"),
        interpretation="stagewise held-out loss for the sequential outcome regressions",
    ),
    _capability(
        "score_equations",
        "longitudinal",
        artifacts=("stagewise targeting state",),
        interpretation="one targeting score and convergence record per regimen and node",
    ),
    _capability(
        "corrections",
        "longitudinal",
        artifacts=(),
        available=False,
        status=AssessmentStatus.NOT_APPLICABLE,
        reason="longitudinal targeting does not use the point-treatment correction system",
        interpretation="the correction identities solved by guarded doubly-robust targeting",
    ),
    _capability(
        "truncation_curve",
        "longitudinal",
        artifacts=("sequential nuisance predictions",),
        execution="refit",
        cost="expensive",
        available=False,
        status=AssessmentStatus.UNAVAILABLE,
        reason=(
            "changing a sequential bound changes every earlier pseudo-outcome and requires "
            "a full refit"
        ),
        interpretation="estimate stability across declared mechanism bounds",
    ),
    _capability(
        "refute",
        "longitudinal",
        artifacts=("fitted sequential estimator configuration",),
        execution="refit",
        deterministic=False,
        cost="expensive",
        available=False,
        status=AssessmentStatus.UNAVAILABLE,
        reason="no evidence-backed longitudinal perturbation/refit adapter is implemented",
        interpretation="behavior under longitudinal data perturbations",
    ),
    _capability(
        "stagewise",
        "longitudinal",
        artifacts=("sequential steps", "cumulative mechanism products"),
        interpretation="risk sets, assignment, leverage, truncation, and convergence by node",
    ),
)


@dataclass(frozen=True)
class SensitivityRoute:
    """Where one sensitivity operation is implemented, and how its estimand is supplied.

    ``needs_estimand`` is a property of the target's *signature*, not of the operation's
    name: the four omitted-variable analyses and :func:`tipping_gamma` all take
    ``estimand`` as their second positional argument, so the facade can fill it in for a
    fit that reports no bare ``"ate"``.  ``benchmark`` and ``missingness`` take
    ``covariates`` and ``gamma`` there, and ``evalue`` selects for itself from a ``None``
    sentinel -- injecting a name into any of those would silently pass it as something
    else.
    """

    module: str
    function: str
    needs_estimand: bool = False


#: Data for the same reason ``ASSESSMENT_CAPABILITIES`` is, and paired with it by a contract
#: test in both directions: every declared sensitivity capability has a route, and every
#: route is declared.  The alternative -- a ``module``/``function`` pair passed by each
#: method plus an ``operation in {...}`` set inside the dispatcher -- is a second registry
#: that no test can see, and it is how ``tipping_gamma`` came to be the one operation of its
#: signature shape that did not get a default estimand.
SENSITIVITY_ROUTES: dict[str, SensitivityRoute] = {
    "omitted_confounding": SensitivityRoute(
        "omitted_variable", "omitted_variable_bounds", needs_estimand=True
    ),
    "robustness_value": SensitivityRoute(
        "omitted_variable", "robustness_value", needs_estimand=True
    ),
    "elements": SensitivityRoute("omitted_variable", "sensitivity_elements", needs_estimand=True),
    "benchmark": SensitivityRoute("omitted_variable", "benchmark"),
    "contour": SensitivityRoute("omitted_variable", "contour_data", needs_estimand=True),
    "evalue": SensitivityRoute("evalue", "evalue"),
    "missingness": SensitivityRoute("missingness", "missingness_tilt"),
    "tipping_gamma": SensitivityRoute("missingness", "tipping_gamma", needs_estimand=True),
}


def _family(result: Any) -> str:
    name = type(result).__name__
    if name == "TMLEResult":
        return "point"
    if name == "LongitudinalResult":
        return "longitudinal"
    raise TypeError(f"assessment has no declared result family for {name}")


def assessment_capabilities(result: Any) -> tuple[AssessmentCapability, ...]:
    """All operation declarations for the result's family."""

    family = _family(result)
    return tuple(item for item in ASSESSMENT_CAPABILITIES if item.result_family == family)


@dataclass(frozen=True)
class AssessmentItem:
    """One immutable result in a combined diagnostic or validation report."""

    name: str
    status: AssessmentStatus
    detail: str
    next_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticReport:
    """Collect statuses from a combined diagnostic or sensitivity run.

    Parameters
    ----------
    items : tuple of AssessmentItem
        Results for the requested operations.
    include_refits : bool
        Whether the run allowed operations that refit nuisance models.
    include_retargets : bool
        Whether the run allowed operations that retarget cached nuisances.
    backend : str or None
        Dataframe backend used by :meth:`to_frame` when ``data`` is omitted.

    See Also
    --------
    ValidationReport : The battery that reads stored artifacts only.
    cleverly.assessment.DiagnosticsFacade : What produces this report.
    cleverly.AssessmentCapability : The declaration behind one item.

    Notes
    -----
    An unavailable item remains in the report. This makes a skipped or refused
    operation visible to the caller.

    Examples
    --------
    >>> from cleverly import AssessmentStatus, DiagnosticReport
    >>> from cleverly.assessment import AssessmentItem
    >>> item = AssessmentItem("support", AssessmentStatus.PASSED, "no material warning")
    >>> report = DiagnosticReport(items=(item,))
    >>> report["support"].status == AssessmentStatus.PASSED
    True
    """

    items: tuple[AssessmentItem, ...]
    #: The two cost classes a caller can opt into, declared separately because they are
    #: disjoint: ``refute`` and ``benchmark`` refit nuisances without retargeting, and
    #: ``truncation_curve``, ``missingness`` and ``tipping_gamma`` retarget cached
    #: nuisances without refitting any.  Folding them into one flag made whichever class
    #: it did not name a silent rider on the other.
    include_refits: bool = False
    include_retargets: bool = False
    backend: str | None = None

    def __getitem__(self, name: str) -> AssessmentItem:
        """Return the report item named ``name``.

        Parameters
        ----------
        name
            Operation name to retrieve.

        Returns
        -------
        AssessmentItem
            Matching report item.

        Raises
        ------
        KeyError
            If the report does not contain ``name``.
        """
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(f"no diagnostic named {name!r}; have {[item.name for item in self.items]}")

    def to_frame(self, data: Any = None) -> Any:
        """Return report items as a dataframe.

        Parameters
        ----------
        data : Any
            Optional dataframe whose backend selects the output type.

        Returns
        -------
        Any
            A pandas or Polars dataframe with one row per operation.
        """
        return emit_frame(
            {
                "check": [item.name for item in self.items],
                "status": [item.status.value for item in self.items],
                "detail": [item.detail for item in self.items],
                "next_steps": ["; ".join(item.next_steps) for item in self.items],
            },
            data,
            backend=self.backend,
        )

    def summary(self) -> str:
        """Return a printable table of operation statuses.

        Returns
        -------
        str
            A printable table, one line per requested operation.
        """
        return format_table(
            ["diagnostic", "status", "detail"],
            [[item.name, item.status.value, item.detail] for item in self.items],
        )


@dataclass(frozen=True)
class ValidationReport:
    """Collect results from the inexpensive validation battery.

    Parameters
    ----------
    items : tuple of AssessmentItem
        Validation checks and their statuses.
    backend : str or None
        Dataframe backend used by :meth:`to_frame` when ``data`` is omitted.

    See Also
    --------
    DiagnosticReport : The combined run, which may retarget or refit.
    cleverly.estimators.TMLEResult : Carries the artifacts this battery reads.
    cleverly.validation.score_check : One of the checks the battery runs.

    Notes
    -----
    The default battery reads stored artifacts. It does not refit nuisance
    models.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(
    ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
    ...     ),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     n_folds=2,
    ...     random_state=0,
    ... )
    >>> report = result.validate()
    >>> len(report.items) > 0
    True
    """

    items: tuple[AssessmentItem, ...]
    backend: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether every required check passed."""
        return all(
            item.status not in {AssessmentStatus.FAILED, AssessmentStatus.UNAVAILABLE}
            for item in self.items
        )

    def __bool__(self) -> bool:
        return self.passed

    def __getitem__(self, name: str) -> AssessmentItem:
        """Return the validation item named ``name``.

        Parameters
        ----------
        name
            Check name to retrieve.

        Returns
        -------
        AssessmentItem
            Matching validation item.

        Raises
        ------
        KeyError
            If the report does not contain ``name``.
        """
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(f"no validation check named {name!r}")

    def to_frame(self, data: Any = None) -> Any:
        """Return validation items as a dataframe.

        Parameters
        ----------
        data : Any
            Optional dataframe whose backend selects the output type.

        Returns
        -------
        Any
            A pandas or Polars dataframe with one row per check.
        """
        return emit_frame(
            {
                "check": [item.name for item in self.items],
                "status": [item.status.value for item in self.items],
                "detail": [item.detail for item in self.items],
                "next_steps": ["; ".join(item.next_steps) for item in self.items],
            },
            data,
            backend=self.backend,
        )

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable table, one line per validation check.
        """
        heading = "Validation: PASS" if self.passed else "Validation: ATTENTION REQUIRED"
        return "\n".join(
            [
                heading,
                "-" * len(heading),
                format_table(
                    ["check", "status", "detail"],
                    [[item.name, item.status.value, item.detail] for item in self.items],
                ),
            ]
        )


@dataclass(frozen=True)
class Replayability:
    """Describe which post-fit actions a saved result can reproduce.

    Parameters
    ----------
    summarize_existing_artifacts : bool
        Whether stored diagnostics can be summarized.
    retarget_cached_nuisances : bool
        Whether targeting can run again without fitting nuisance models.
    evaluate_stored_representer : bool
        Whether the stored representer can evaluate another parameter.
    refit_nuisances : bool
        Whether the result retains enough configuration for nuisance refits.
    evaluate_new_data : bool
        Whether the fitted result can score new observations.
    unreconstructible : tuple of str
        Missing components that prevent reconstruction.
    """

    summarize_existing_artifacts: bool
    retarget_cached_nuisances: bool
    evaluate_stored_representer: bool
    refit_nuisances: bool
    evaluate_new_data: bool
    unreconstructible: tuple[str, ...] = ()


def replayability(result: Any) -> Replayability:
    """Derive replay capabilities from stored artifacts and the normalized method."""

    if _family(result) == "longitudinal":
        # Whole-result persistence retains the method and its unfitted learner templates.
        # Changing a bound still requires the entire recursion rather than cached retargeting.
        method = getattr(result, "method", None)
        missing = () if method is not None else ("method configuration",)
        return Replayability(True, False, False, method is not None, False, missing)

    estimator = getattr(result, "estimator", None)
    if estimator is None:
        return Replayability(True, True, False, False, False, ("estimator configuration",))
    return Replayability(True, True, False, True, False)


@dataclass(frozen=True)
class _CachedFrame:
    columns: tuple[str, ...]
    values: tuple[tuple[Any, ...], ...]
    backend: str | None

    @classmethod
    def from_frame(cls, frame: Any, backend: str | None) -> _CachedFrame:
        if type(frame).__module__.startswith("polars"):
            payload = frame.to_dict(as_series=False)
        else:
            payload = frame.to_dict(orient="list")
        columns = tuple(str(column) for column in payload)
        values = tuple(
            tuple(_python_scalar(value) for value in payload[column]) for column in columns
        )
        return cls(columns, values, backend)

    def materialize(self) -> Any:
        return emit_frame(dict(zip(self.columns, self.values, strict=True)), backend=self.backend)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _pack_cached(value: Any, backend: str | None) -> Any:
    module = type(value).__module__
    if module.startswith("pandas") or module.startswith("polars"):
        return _CachedFrame.from_frame(value, backend)
    return value


def _unpack_cached(value: Any) -> Any:
    return value.materialize() if isinstance(value, _CachedFrame) else value


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return {"array": [list(value.shape), str(value.dtype), digest]}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    return {"object": repr(value), "type": type(value).__qualname__}


def _cache_key(operation: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    normalized = {"args": _normalize(tuple(args)), "kwargs": _normalize(kwargs)}
    return f"{operation}:{json.dumps(normalized, sort_keys=True, separators=(',', ':'))}"


def _cached(
    result: Any,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    compute: Callable[[], Any],
) -> Any:
    cache = result.assessment_cache
    key = _cache_key(operation, args, kwargs)
    if key in cache:
        return _unpack_cached(cache[key])
    value = compute()
    cache[key] = _pack_cached(value, getattr(result.data, "backend", None))
    return value


@dataclass(frozen=True)
class LongitudinalStageRow:
    regimen: str
    cause: str | None
    horizon: int | None
    time: int
    n_followed: int
    assignment: float | str
    max_weight: float
    effective_n: float
    share_truncated: float
    epsilon: tuple[float, ...]
    converged: bool


@dataclass(frozen=True)
class LongitudinalDiagnostics:
    """A row per regimen and node: how much data it had, and how hard it leaned on it.

    ``n_followed`` is the number of units that followed the regimen and stayed under
    observation through the node -- the sample the regression there was fitted on.
    ``max_weight`` and ``effective_n`` describe the cumulative clever covariate, which is
    where sequential positivity shows up: they are properties of the *product* of the
    node-by-node mechanisms and can be alarming while every node looks fine.  On a weighted
    fit they describe ``w / prod g`` rather than ``1 / prod g``, because the two
    reweightings multiply -- see
    :attr:`~cleverly.longitudinal.sequential.RegimenFit.leverage`.  For the weighting's own
    cost, and the estimand statement that goes with it, see ``result.data.weight_report()``.

    ``share_assigned_1`` is the fraction of the units at risk at that node whom the regimen
    would treat.  For a static regimen it is exactly ``0`` or ``1``, so the column doubles
    as a check on the plan the fit actually ran; for a dynamic rule it is the number a
    reader needs, since what a rule assigns is a property of the data rather than of the
    declaration and appears nowhere in the settings report.

    **When any treatment node is categorical the column is** ``assigned_shares``
    **instead**, holding ``"active=0.62, none=0.38"`` in that node's label order -- the
    presentation :func:`~cleverly.estimators.base._arm_shares` uses for the same question
    about a point treatment.  A single share cannot answer it at three arms: "the fraction
    assigned arm 1" is the fraction assigned whichever label happens to sort second, which
    is not a quantity anybody asked for, and a static plan on a third arm would report ``0``
    exactly as a plan on the first arm does.  A wholly two-level panel keeps
    ``share_assigned_1`` and its values unchanged, so the switch is visible in the columns
    rather than hidden in them.

    ``share_truncated`` compares the raw and bounded cumulative probabilities on the same
    ``trained_on`` rows as the node's score.  Unlike ``max_weight``, it reveals when the
    configured cap replaced every contributing row.
    """

    rows: tuple[LongitudinalStageRow, ...]
    epsilon_names: tuple[str, ...]
    categorical: bool
    survival: bool
    competing: bool
    backend: str | None = None

    def to_frame(self, data: Any = None) -> Any:
        assignment = "assigned_shares" if self.categorical else "share_assigned_1"
        payload: dict[str, Any] = {
            "regimen": [row.regimen for row in self.rows],
            **({"cause": [row.cause for row in self.rows]} if self.competing else {}),
            **({"horizon": [row.horizon for row in self.rows]} if self.survival else {}),
            "time": [row.time for row in self.rows],
            "n_followed": [row.n_followed for row in self.rows],
            assignment: [row.assignment for row in self.rows],
            "max_weight": [row.max_weight for row in self.rows],
            "effective_n": [row.effective_n for row in self.rows],
            "share_truncated": [row.share_truncated for row in self.rows],
        }
        for position, name in enumerate(self.epsilon_names):
            payload[name] = [row.epsilon[position] for row in self.rows]
        payload["converged"] = [row.converged for row in self.rows]
        return emit_frame(payload, data, backend=self.backend)


@dataclass(frozen=True)
class LongitudinalScoreRow:
    """One node's targeting score, and the verdict two separate gates reach about it.

    ``converged`` is the fit's own flag: whether that node's Newton step settled against
    the targeting tolerance it was configured with.  ``passed`` additionally holds the
    node's ``relative_score`` to the tolerance the *caller* asked for.  They are kept apart
    because they can disagree, and because only their conjunction is safe -- a caller
    tolerance may tighten the verdict and may never license a fluctuation whose step
    failed.
    """

    regimen: str
    cause: str | None
    horizon: int | None
    time: int
    score: float
    relative_score: float
    converged: bool
    passed: bool
    n_iter: int
    failure: str | None


@dataclass(frozen=True)
class LongitudinalScoreDiagnostics:
    """Stagewise targeting scores, gated at the tolerance they were asked for.

    ``tolerance`` bounds each node's *relative* score -- the largest score component as a
    fraction of its maximum possible magnitude, which is the quantity the sequential
    targeting loop itself gates on.  The point-treatment report answers the same question
    on a different scale, comparing the score in the outcome's own units against
    ``tolerance * se / sqrt(n)``; see :data:`~cleverly.validation.score.DEFAULT_TOLERANCE`.
    The number is carried here so a report says which gate produced its verdict.
    """

    rows: tuple[LongitudinalScoreRow, ...]
    tolerance: float
    backend: str | None = None

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "regimen": [row.regimen for row in self.rows],
                "cause": [row.cause for row in self.rows],
                "horizon": [row.horizon for row in self.rows],
                "time": [row.time for row in self.rows],
                "score": [row.score for row in self.rows],
                "relative_score": [row.relative_score for row in self.rows],
                "converged": [row.converged for row in self.rows],
                "passed": [row.passed for row in self.rows],
                "n_iter": [row.n_iter for row in self.rows],
                "failure": [row.failure for row in self.rows],
            },
            data,
            backend=self.backend,
        )


@dataclass(frozen=True)
class LongitudinalNuisanceRow:
    regimen: str
    cause: str | None
    horizon: int | None
    time: int
    n: int
    mse: float


@dataclass(frozen=True)
class LongitudinalNuisanceDiagnostics:
    """Stagewise loss of each sequential outcome/pseudo-outcome regression."""

    rows: tuple[LongitudinalNuisanceRow, ...]
    backend: str | None = None

    def to_frame(self, data: Any = None) -> Any:
        return emit_frame(
            {
                "regimen": [row.regimen for row in self.rows],
                "cause": [row.cause for row in self.rows],
                "horizon": [row.horizon for row in self.rows],
                "time": [row.time for row in self.rows],
                "n": [row.n for row in self.rows],
                "mse": [row.mse for row in self.rows],
            },
            data,
            backend=self.backend,
        )


def _assigned_shares(assigned: FloatArray, levels: Sequence[object]) -> str:
    """What a regimen assigns at one node, as ``"active=0.62, none=0.38"``.

    The categorical counterpart of ``share_assigned_1``, and written in the *labels* rather
    than the dense codes for the reason every user-facing string in this package is: a
    reader asked to translate ``2.0`` back to ``"none"`` has been handed the encoding rather
    than the answer.  Every level appears, including one the regimen never assigns, so the
    shares in a row sum to one and a zero is legible as "not this arm" rather than as a
    level the fit forgot about.

    Deliberately a string and not a column per level: the level sets are per node, so
    numeric columns would be ragged across a frame whose rows are ``(regimen, time)`` pairs,
    and most of them empty.
    """
    if not assigned.size:
        return ""
    return ", ".join(
        f"{level}={float(np.mean(assigned == float(code))):.3g}"
        for code, level in enumerate(levels)
    )


def _longitudinal_stagewise(result: Any) -> LongitudinalDiagnostics:
    terms = () if result.msm is None else result.msm.terms
    epsilon_names = ("epsilon",) if result.msm is None else tuple(f"epsilon[{t}]" for t in terms)
    # One column shape for the whole frame rather than one per row: the level sets are a
    # property of the data, so whether a share is answerable by a single number is settled
    # before any node is read.
    categorical = any(len(levels) > 2 for levels in result.data.treatment_levels)
    rows = []
    # Read off the fit's own fields rather than the key it is filed under: on a survival fit
    # that key is the regimen *and* the horizon, and a ``regimen`` column carrying both would
    # be the one column here nobody could group by.
    for fit in result.fits.values():
        for step in fit.steps:
            weights = (fit.obs_weights * step.clever)[step.trained_on]
            assigned = fit.assignment[step.at_risk, step.time - 1]
            assignment: float | str = (
                _assigned_shares(assigned, result.data.treatment_levels[step.time - 1])
                if categorical
                else (float(np.mean(assigned == 1.0)) if assigned.size else float("nan"))
            )
            raw = fit.cumulative_unbounded[:, step.time - 1][step.trained_on]
            bounded = fit.cumulative[:, step.time - 1][step.trained_on]
            rows.append(
                LongitudinalStageRow(
                    regimen=fit.regimen.label,
                    cause=fit.cause,
                    horizon=fit.horizon if result.data.is_survival else None,
                    time=step.time,
                    n_followed=step.n_trained,
                    assignment=assignment,
                    max_weight=float(np.max(weights)) if weights.size else float("nan"),
                    effective_n=effective_sample_size(weights, on_degenerate=0.0),
                    share_truncated=float(np.mean(raw != bounded)) if raw.size else float("nan"),
                    epsilon=tuple(float(value) for value in step.fluctuation.epsilon),
                    converged=bool(step.fluctuation.converged),
                )
            )
    return LongitudinalDiagnostics(
        tuple(rows),
        epsilon_names,
        categorical,
        result.data.is_survival,
        result.data.is_competing,
        result.data.backend,
    )


def _longitudinal_scores(result: Any, *, tolerance: float) -> LongitudinalScoreDiagnostics:
    """Every node's targeting score, gated at ``tolerance`` on the relative scale.

    The gate is a *conjunction*, and deliberately so.  Sequential targeting settles against
    its own ``tol`` -- ``1e-10``, far tighter than the default asked for here -- so requiring
    ``converged`` as well leaves the default verdict exactly what it was while letting a
    caller tighten it.  Gating on the relative score alone would do the opposite: a node
    whose Newton step failed but whose residual score happens to sit under a loose tolerance
    would be reported as passing, which is the one answer this diagnostic must never give.
    """
    rows = []
    for fit in result.fits.values():
        for step in fit.steps:
            fluctuation = step.fluctuation
            relative = float(fluctuation.relative_score_norm)
            converged = bool(fluctuation.converged)
            rows.append(
                LongitudinalScoreRow(
                    fit.regimen.label,
                    fit.cause,
                    fit.horizon if result.data.is_survival else None,
                    step.time,
                    float(result.scaler.range * fluctuation.score_norm),
                    relative,
                    converged,
                    converged and relative <= tolerance,
                    int(fluctuation.n_iter),
                    fluctuation.failure,
                )
            )
    return LongitudinalScoreDiagnostics(tuple(rows), tolerance, result.data.backend)


def _longitudinal_nuisances(result: Any) -> LongitudinalNuisanceDiagnostics:
    rows = []
    for fit in result.fits.values():
        for step in fit.steps:
            mask = step.trained_on
            residual = np.asarray(step.pseudo_outcome[mask] - step.initial[mask], dtype=float)
            rows.append(
                LongitudinalNuisanceRow(
                    fit.regimen.label,
                    fit.cause,
                    fit.horizon if result.data.is_survival else None,
                    step.time,
                    int(mask.sum()),
                    float(np.mean(np.square(residual))) if residual.size else float("nan"),
                )
            )
    return LongitudinalNuisanceDiagnostics(tuple(rows), result.data.backend)


class _CapabilityFacade:
    """Lookup, refusal, and combined-report machinery shared by both public facades.

    The two facades answer different questions from different declarations, but the way
    they *route* a question is one algorithm: find the operation's row, refuse by that row
    when it is unavailable, and in a combined report skip what the caller has not paid for.
    Written twice, it drifted five ways -- the availability and cost checks in opposite
    orders, two spellings of the cost gate, one refusal that re-derived a reason the record
    already carried, different caught-exception sets, and one side that discarded what its
    operations returned and so could only ever report ``passed``.  Subclasses supply
    :attr:`capabilities` and the two labels below; everything else is settled here.
    """

    #: What an operation of this kind is called in a refusal: ``diagnostic 'refute' is ...``.
    _kind: str
    #: The attribute a caller reaches it through, for the ``next_steps`` of a skipped row.
    _attribute: str

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        raise NotImplementedError  # pragma: no cover - subclasses declare their own

    def capability(self, operation: str) -> AssessmentCapability:
        for item in self.capabilities:
            if item.operation == operation:
                return item
        raise KeyError(f"unknown {self._kind} {operation!r}")

    def _require(self, operation: str) -> AssessmentCapability:
        item = self.capability(operation)
        if not item.available:
            # ``reason`` first, and ``interpretation`` only as a fallback: the record knows
            # why *this* operation is refused, and re-deriving one from the result family
            # gave an E-value on a longitudinal fit a rationale about pseudo-outcome
            # recursion, which is a true sentence about a different operation.
            raise CapabilityError(
                f"{self._kind} {operation!r} is {item.status.value}: "
                f"{item.reason or item.interpretation}"
            )
        return item

    def _run_all(self, *, include_refits: bool, include_retargets: bool) -> DiagnosticReport:
        def compute() -> DiagnosticReport:
            items = []
            for capability in self.capabilities:
                # Availability first. A row that is unavailable *and* expensive is refused
                # for the reason it declares, not for a cost the caller could have paid --
                # "pass include_refits=True" is a false instruction when no flag can make
                # the operation exist.
                if not capability.available:
                    items.append(_item_from_capability(capability))
                    continue
                skipped = _cost_refusal(
                    capability, self._attribute, include_refits, include_retargets
                )
                if skipped is not None:
                    items.append(skipped)
                    continue
                if capability.requires_arguments:
                    # A combined report runs every operation argument-free, so one with a
                    # required argument and no default cannot appear in it.  Choosing a
                    # value here -- which covariates to benchmark against -- would be a
                    # scientific choice made silently on the caller's behalf.
                    needed = ", ".join(capability.requires_arguments)
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            f"needs an explicit {needed} argument, which a combined report "
                            f"has no basis to choose",
                            (
                                f"call result.{self._attribute}.{capability.operation}() "
                                f"directly with {needed}",
                            ),
                        )
                    )
                    continue
                try:
                    report = getattr(self, capability.operation)()
                except CapabilityError as error:
                    # Only the refusal type, deliberately.  Merging the two facades' handlers
                    # took the *union* of what each caught, which handed the diagnostics side
                    # ``KeyError`` and ``TypeError`` -- and no routed operation raises either
                    # as a refusal.  Every ``raise KeyError`` in the package is a lookup on an
                    # already-computed report and every ``raise TypeError`` is structural, so
                    # catching them turned a signature or state bug into a scientific-sounding
                    # ``unavailable``.  ``tests/e2e/test_ltmle.py`` records what that costs: a
                    # missing keyword inside a loop was reported as "too unstable to
                    # bootstrap", a statistical diagnosis of an engineering fault.
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            f"refused on inspection: {error}",
                            (
                                f"call result.{self._attribute}.{capability.operation}() "
                                f"directly for the refusal in full",
                            ),
                        )
                    )
                else:
                    items.append(_diagnostic_item(capability.operation, report))
            return DiagnosticReport(
                tuple(items),
                include_refits=include_refits,
                include_retargets=include_retargets,
                backend=self._result.data.backend,
            )

        return _cached(
            self._result,
            f"{self._attribute}.run_all",
            (),
            {"include_refits": include_refits, "include_retargets": include_retargets},
            compute,
        )


def _cost_refusal(
    capability: AssessmentCapability,
    attribute: str,
    include_refits: bool,
    include_retargets: bool,
) -> AssessmentItem | None:
    """The skip a combined report owes an operation the caller has not paid for.

    Two flags rather than one because the two costs are disjoint: refutation and
    benchmarking refit nuisances without retargeting, and the truncation curve and the
    missingness tilt retarget cached nuisances without refitting any.  One flag made
    whichever class it did not name run silently under the other's permission.
    """
    allowed = {"summarize": True, "refit": include_refits, "retarget": include_retargets}
    if allowed[capability.execution]:
        return None
    work = "refits nuisance models" if capability.execution == "refit" else "retargets the fit"
    flag = "include_refits" if capability.execution == "refit" else "include_retargets"
    return AssessmentItem(
        capability.operation,
        AssessmentStatus.UNAVAILABLE,
        f"not run by default because it {work}; pass {flag}=True",
        (f"call result.{attribute}.{capability.operation}() directly, or pass {flag}=True",),
    )


class DiagnosticsFacade(_CapabilityFacade):
    """Access diagnostics supported by a fitted causal result.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        Fitted point-treatment or longitudinal result.

    See Also
    --------
    SensitivityFacade : The analyses that ask what would overturn the estimate.
    cleverly.AssessmentCapability : One row of :attr:`capabilities`.
    cleverly.DiagnosticReport : What :meth:`run_all` returns.

    Notes
    -----
    Access this facade through ``result.diagnostics``. Inspect
    :attr:`capabilities` before optional or potentially expensive operations.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(
    ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
    ...     ),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     n_folds=2,
    ...     random_state=0,
    ... )
    >>> support = result.diagnostics.support()
    >>> support.n
    200
    """

    _kind = "diagnostic"
    _attribute = "diagnostics"

    @property
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        """Return declared operations and their availability."""
        return assessment_capabilities(self._result)

    def stagewise(self) -> LongitudinalDiagnostics:
        """Return support and targeting diagnostics by longitudinal stage.

        Returns
        -------
        LongitudinalDiagnostics
            Diagnostics for each regimen and time point.

        Raises
        ------
        CapabilityError
            If the fitted result is not longitudinal.
        """
        self._require("stagewise")
        return _cached(
            self._result,
            "diagnostics.stagewise",
            (),
            {},
            lambda: _longitudinal_stagewise(self._result),
        )

    def support(self) -> Any:
        """Return the support diagnostic for the fitted intervention.

        Returns
        -------
        PositivityReport or LongitudinalDiagnostics or SupportReport or dict
            Support diagnostic specialized for arms, regimens, shifts, or incremental
            interventions.
        """
        self._require("support")
        if _family(self._result) == "longitudinal":
            return self.stagewise()

        def compute() -> Any:
            nuisance = self._result.nuisance
            from .interventions import (
                check_incremental_support,
                check_shift_support,
                check_support,
            )
            from .sensitivity.positivity import positivity_report

            if nuisance.regimes is not None:
                return check_support(
                    nuisance.regimes,
                    self._result.data.treatment,
                    nuisance.propensity.values,
                    backend=self._result.data.backend,
                )
            # ``shifts`` alone, not ``shifts and density``: a shift fit without a fitted
            # density is a broken shift fit, and the density-ratio report says so by name.
            # Adding the second condition sent it to the arm-level report instead, which
            # answers a different question or refuses for an unrelated reason.
            if nuisance.shifts is not None:
                bound = self._result.config.missingness_bound
                level = self._result.intermediate_value
                mechanisms = [
                    values
                    for values in (
                        nuisance.bounded_missingness(bound),
                        None if level is None else nuisance.intermediate_density(level, bound),
                    )
                    if values is not None
                ]
                return check_shift_support(
                    nuisance.shifts,
                    nuisance.density,
                    self._result.data.treatment,
                    mechanisms=mechanisms,
                )
            if nuisance.incremental is not None:
                return check_incremental_support(nuisance.incremental, self._result.data.treatment)
            return positivity_report(self._result)

        return _cached(self._result, "diagnostics.support", (), {}, compute)

    def nuisance_models(self) -> Any:
        """Return held-out fit diagnostics for nuisance models.

        Returns
        -------
        NuisanceDiagnostics or LongitudinalNuisanceDiagnostics
            Held-out diagnostics for point or sequential nuisance models.
        """
        self._require("nuisance_models")

        def compute() -> Any:
            if _family(self._result) == "longitudinal":
                return _longitudinal_nuisances(self._result)
            from .validation.nuisance import nuisance_diagnostics

            return nuisance_diagnostics(self._result)

        return _cached(self._result, "diagnostics.nuisance_models", (), {}, compute)

    def score_equations(self, *, tolerance: float = DEFAULT_TOLERANCE) -> Any:
        """Whether targeting solved the score equations this fit relies on.

        Parameters
        ----------
        tolerance : float
            Relative tolerance used to evaluate the fitted score equations.

        Returns
        -------
        ScoreCheck or LongitudinalScoreDiagnostics
            Score-equation checks for the fitted result family.

        Notes
        -----
        ``tolerance`` gates both families but on the scale each one's score lives on, and
        the two are not interchangeable.  A point-treatment fit compares the score in the
        outcome's own units against ``tolerance * se / sqrt(n)``
        (:data:`~cleverly.validation.score.DEFAULT_TOLERANCE` says why that shape).  A
        longitudinal fit bounds each node's *relative* score -- the largest component as a
        fraction of its maximum possible magnitude -- which is the quantity the sequential
        targeting loop gates on, and it can only tighten a node's verdict beyond the fit's
        own convergence flag.
        """
        self._require("score_equations")

        def compute() -> Any:
            if _family(self._result) == "longitudinal":
                return _longitudinal_scores(self._result, tolerance=tolerance)
            from .validation.score import score_check

            return score_check(self._result, tolerance=tolerance)

        return _cached(
            self._result,
            "diagnostics.score_equations",
            (),
            {"tolerance": tolerance},
            compute,
        )

    def corrections(
        self,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        identity_tolerance: float = IDENTITY_TOLERANCE,
    ) -> Any:
        """Return correction-equation checks for a DRTMLE fit.

        Parameters
        ----------
        tolerance : float
            Tolerance for the reduced correction score.
        identity_tolerance : float
            Tolerance for the correction identity.

        Returns
        -------
        CorrectionCheck
            Correction diagnostics for the fitted result.

        Raises
        ------
        CapabilityError
            If the fitted method does not use the correction system.
        """
        self._require("corrections")
        from .validation.drtmle import correction_check

        return _cached(
            self._result,
            "diagnostics.corrections",
            (),
            {"tolerance": tolerance, "identity_tolerance": identity_tolerance},
            lambda: correction_check(
                self._result,
                tolerance=tolerance,
                identity_tolerance=identity_tolerance,
            ),
        )

    def truncation_curve(
        self,
        bounds: Sequence[float] | None = None,
        *,
        estimands: Sequence[str] | None = None,
        mechanism: bool = False,
    ) -> Any:
        """Retarget estimates across a sequence of mechanism bounds.

        Parameters
        ----------
        bounds : sequence of float or None
            Mechanism bounds to evaluate. Default bounds are used when omitted.
        estimands : sequence of str or None
            Reported estimands to include. All compatible estimands are the default.
        mechanism : bool
            For an incremental intervention, vary a separate observation mechanism.

        Returns
        -------
        dataframe
            Estimates and uncertainty at each requested bound.

        Raises
        ------
        CapabilityError
            If the requested curve changes the estimand or cannot be replayed.
        """
        self._require("truncation_curve")
        from .sensitivity.positivity import truncation_curve

        if self._result.nuisance.incremental is not None and not mechanism:
            raise CapabilityError(
                "the propensity g is *inside* the estimand for an incremental intervention, "
                "so a propensity-bound curve would compare different parameters; use "
                "diagnostics.support(), or pass mechanism=True when a separate observation "
                "mechanism was fitted"
            )
        return _cached(
            self._result,
            "diagnostics.truncation_curve",
            (bounds,),
            {"estimands": estimands, "mechanism": mechanism},
            lambda: truncation_curve(
                self._result, bounds, estimands=estimands, mechanism=mechanism
            ),
        )

    def refute(self, **kwargs: Any) -> Any:
        """Refit the analysis under the requested data perturbations.

        Parameters
        ----------
        **kwargs
            Options forwarded to :func:`cleverly.validation.refute`.

        Returns
        -------
        RefutationResult
            Refutation results for each requested perturbation.

        Raises
        ------
        CapabilityError
            If the fitted estimator cannot be reconstructed.
        """
        self._require("refute")
        if not replayability(self._result).refit_nuisances:
            missing = replayability(self._result).unreconstructible
            raise CapabilityError(
                "refutation requires nuisance refits, but this estimator cannot be "
                f"reconstructed; unavailable slots: {list(missing)}"
            )
        from .validation.refute import refute

        return _cached(
            self._result,
            "diagnostics.refute",
            (),
            kwargs,
            lambda: refute(self._result, **kwargs),
        )

    def run_all(
        self, *, include_refits: bool = False, include_retargets: bool = False
    ) -> DiagnosticReport:
        """Run available diagnostics that need no new arguments.

        Parameters
        ----------
        include_refits : bool
            Include operations that refit nuisance models.
        include_retargets : bool
            Include operations that retarget cached nuisance predictions.

        Returns
        -------
        DiagnosticReport
            One item for every declared diagnostic operation.

        See Also
        --------
        cleverly.DiagnosticReport : The combined operation statuses.
        DiagnosticsFacade.capabilities : Declare availability and cost before execution.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_linear_ate
        >>> frame, _ = make_linear_ate(n=80, seed=1)
        >>> study = CausalStudy(
        ...     frame,
        ...     design=PointTreatment(
        ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
        ...     ),
        ... )
        >>> result = study.identify(ATE()).estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     n_folds=2,
        ...     random_state=0,
        ... )
        >>> report = result.diagnostics.run_all()
        >>> report.include_refits, report.include_retargets
        (False, False)
        """
        return self._run_all(include_refits=include_refits, include_retargets=include_retargets)


def _item_from_capability(capability: AssessmentCapability) -> AssessmentItem:
    return AssessmentItem(
        capability.operation,
        capability.status,
        capability.reason or capability.interpretation,
    )


def _diagnostic_item(name: str, report: Any) -> AssessmentItem:
    passed = getattr(report, "passed", None)
    if passed is False:
        return AssessmentItem(name, AssessmentStatus.FAILED, "the diagnostic reported a failure")
    if name == "support":
        warning = _support_warning(report)
        return AssessmentItem(
            name,
            AssessmentStatus.WARNING if warning else AssessmentStatus.PASSED,
            warning or "no material support warning in the stored diagnostic",
        )
    return AssessmentItem(name, AssessmentStatus.PASSED, "completed from stored fitted artifacts")


def _support_warning(report: Any) -> str | None:
    if hasattr(report, "truncated"):
        fraction = float(report.truncated.get("fraction", 0.0))
        if fraction > 0.05:
            return f"{fraction:.1%} of propensity cells were truncated"
        ratios = [
            float(item.get("ess_ratio", 1.0)) for item in getattr(report, "mechanisms", {}).values()
        ]
        if ratios and min(ratios) < 0.2:
            return "a fitted mechanism leaves less than 20% effective sample size"
    if isinstance(report, LongitudinalDiagnostics):
        if any(row.share_truncated > 0.05 for row in report.rows):
            return "cumulative mechanism truncation exceeds 5% at one or more nodes"
        if any(row.n_followed and row.effective_n / row.n_followed < 0.2 for row in report.rows):
            return "cumulative weights leave less than 20% effective sample size at a node"
    if isinstance(report, Mapping):
        # A shift or IPSI fit reports one support record per declared intervention rather
        # than a single object, so none of the attribute probes above sees it.  These
        # dataclasses carry the same quantities the other branches threshold on, at the
        # same tiers; read them by field so the two classes need no separate branches --
        # ``IncrementalSupport`` has an ``ess_ratio`` but no ``unsupported`` or
        # ``capped_fraction``, and a missing field must not read as a breach.
        for name, item in report.items():
            if int(getattr(item, "unsupported", 0)) > 0:
                return f"intervention {name!r} has units with estimated zero support"
            if float(getattr(item, "ess_ratio", 1.0)) < 0.2:
                return f"intervention {name!r} leaves less than 20% effective sample size"
            if float(getattr(item, "capped_fraction", 0.0)) > 0.05:
                return f"intervention {name!r} had more than 5% of its weights capped"
        return None
    regimes = getattr(report, "regimes", None)
    if regimes and any(item.unsupported for item in regimes.values()):
        return "one or more declared regimes has estimated zero support"
    return None


def validate_result(result: Any) -> ValidationReport:
    """Run only cheap, cache-only checks appropriate to this fitted method."""

    def compute() -> ValidationReport:
        diagnostics = result.diagnostics
        score = diagnostics.score_equations()
        score_passed = bool(getattr(score, "passed", False))
        score_rows = tuple(getattr(score, "rows", ()))
        conditioning = _reduction_conditioning_warning(result) if score_passed else None
        items = [
            AssessmentItem(
                "score_equations",
                (
                    AssessmentStatus.WARNING
                    if conditioning
                    else AssessmentStatus.PASSED
                    if score_passed
                    else AssessmentStatus.FAILED
                ),
                (
                    conditioning or f"all {len(score_rows)} stored score checks converged"
                    if score_passed
                    else "one or more stored score equations failed its convergence check"
                ),
                (("inspect result.repeats[*].fluctuations['mean'].reduction.ill_conditioned"),)
                if conditioning
                else ()
                if score_passed
                else ("inspect result.diagnostics.score_equations()",),
            )
        ]
        support = diagnostics.support()
        warning = _support_warning(support)
        items.append(
            AssessmentItem(
                "support",
                AssessmentStatus.WARNING if warning else AssessmentStatus.PASSED,
                warning or "stored support diagnostics show no material warning",
                () if not warning else ("inspect result.diagnostics.support()",),
            )
        )
        nuisance = diagnostics.nuisance_models()
        finite = []
        if isinstance(nuisance, LongitudinalNuisanceDiagnostics):
            finite = [row.mse for row in nuisance.rows if np.isfinite(row.mse)]
        nuisance_status = (
            AssessmentStatus.WARNING
            if isinstance(nuisance, LongitudinalNuisanceDiagnostics) and not finite
            else AssessmentStatus.PASSED
        )
        items.append(
            AssessmentItem(
                "nuisance_models",
                nuisance_status,
                "stagewise held-out losses are available"
                if isinstance(nuisance, LongitudinalNuisanceDiagnostics)
                else "out-of-fold nuisance fit and calibration diagnostics are available",
                ("inspect result.diagnostics.nuisance_models()",),
            )
        )
        return ValidationReport(tuple(items), result.data.backend)

    return _cached(result, "validate", (), {}, compute)


def _reduction_conditioning_warning(result: Any) -> str | None:
    """Summarize successful equation-(10) solves that needed an ill-conditioned step."""
    ill_conditioned = 0
    rounds = 0
    affected = 0
    repeats = tuple(getattr(result, "repeats", ()))
    for repeat in repeats:
        fluctuation = getattr(repeat, "fluctuations", {}).get("mean")
        reduction = getattr(fluctuation, "reduction", None)
        count = int(getattr(reduction, "ill_conditioned", 0))
        rounds += int(getattr(reduction, "rounds", 0))
        if count <= 0:
            continue
        ill_conditioned += count
        affected += 1
    if ill_conditioned == 0:
        return None
    fraction = ill_conditioned / rounds if rounds else float("nan")
    draws = f" across {affected} of {len(repeats)} repeat(s)" if len(repeats) > 1 else ""
    return (
        f"all stored score checks converged, but equation (10) reported numerical difficulty in "
        f"{ill_conditioned} of {rounds} refitting round(s) ({fraction:.1%}){draws}; "
        "one or more inner solves were ill-conditioned or stopped at their numerical "
        "tolerance even though the returned score equations passed"
    )


class SensitivityFacade(_CapabilityFacade):
    """Access sensitivity analyses supported by a fitted causal result.

    Parameters
    ----------
    result : TMLEResult or LongitudinalResult
        Fitted point-treatment or longitudinal result.

    See Also
    --------
    DiagnosticsFacade : The checks that ask whether the fit itself is sound.
    cleverly.AssessmentCapability : One row of :attr:`capabilities`.
    cleverly.sensitivity.evalue.evalue : The same analysis as a free function.

    Notes
    -----
    Access this facade through ``result.sensitivity``. Methods use the fitted
    estimate and nuisance artifacts as their first input. Their remaining
    arguments match the corresponding functions in :mod:`cleverly.sensitivity`.
    Inspect :attr:`capabilities` to distinguish an unsupported analysis from one
    that needs additional arguments or expensive work.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from cleverly import ATE, CausalStudy, PointTreatment
    >>> from cleverly.datasets import make_linear_ate
    >>> frame, _ = make_linear_ate(n=200, seed=0)
    >>> study = CausalStudy(
    ...     frame,
    ...     design=PointTreatment(
    ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
    ...     ),
    ... )
    >>> result = study.identify(ATE()).estimate(
    ...     outcome_learner=LinearRegression(),
    ...     treatment_learner=LogisticRegression(max_iter=1000),
    ...     n_folds=2,
    ...     random_state=0,
    ... )
    >>> capability = result.sensitivity.capability("omitted_confounding")
    >>> capability.available, capability.cost
    (True, 'cheap')
    """

    _kind = "sensitivity"
    _attribute = "sensitivity"

    @cached_property
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        """Return sensitivity operations, their availability, cost, and requirements."""
        family = _family(self._result)
        longitudinal = family == "longitudinal"
        missing = (
            False
            if longitudinal
            else getattr(self._result.nuisance, "missingness", None) is not None
        )
        benchmarkable = not longitudinal and replayability(self._result).refit_nuisances
        available = not longitudinal
        status = AssessmentStatus.PASSED if available else AssessmentStatus.UNAVAILABLE
        reason = "no longitudinal sensitivity derivation is registered" if longitudinal else None

        def standard(
            operation: str,
            *,
            artifacts: Sequence[str],
            interpretation: str,
            cost: Literal["cheap", "moderate", "expensive"] = "cheap",
        ) -> AssessmentCapability:
            return _capability(
                operation,
                family,
                artifacts=artifacts,
                interpretation=interpretation,
                cost=cost,
                available=available,
                status=status,
                reason=reason,
            )

        def tilt(operation: str, *, interpretation: str) -> AssessmentCapability:
            """The two MNAR analyses, which share every field but their interpretation.

            Three cases, not two.  A point fit that *has* a missingness mechanism can run
            the tilt; one that has none is answering a question about a functional with no
            observation mechanism in it, which is ``not_applicable``; and a longitudinal
            fit is refused because no adapter has been derived, which is ``unavailable``.
            """
            return _capability(
                operation,
                family,
                artifacts=("observation mechanism", "published tilt identification"),
                execution="retarget",
                cost="moderate",
                interpretation=interpretation,
                available=missing,
                status=(
                    AssessmentStatus.PASSED
                    if missing
                    else AssessmentStatus.UNAVAILABLE
                    if longitudinal
                    else AssessmentStatus.NOT_APPLICABLE
                ),
                reason=(
                    None
                    if missing
                    else "no longitudinal missingness-tilt adapter is implemented"
                    if longitudinal
                    else "the identified functional has no observation mechanism"
                ),
            )

        return (
            standard(
                "omitted_confounding",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="omitted-confounder bias for the fitted orthogonal score",
            ),
            standard(
                "robustness_value",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="confounding strength needed to move the estimate to its null",
            ),
            standard(
                "elements",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="raw components of the omitted-confounder bias bound",
            ),
            _capability(
                "benchmark",
                artifacts=("fitted estimator configuration",),
                execution="refit",
                deterministic=False,
                cost="expensive",
                interpretation="calibration against named observed covariates",
                available=benchmarkable,
                status=AssessmentStatus.PASSED if benchmarkable else AssessmentStatus.UNAVAILABLE,
                reason=(
                    None
                    if benchmarkable
                    else "benchmarking requires a replayable point-treatment estimator"
                ),
                requires_arguments=("covariates",),
                family=family,
            ),
            standard(
                "contour",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="bias bounds over a grid of confounding strengths",
                cost="moderate",
            ),
            standard(
                "evalue",
                artifacts=("ratio-scale estimate",),
                interpretation="minimum risk-ratio association needed to explain away an effect",
            ),
            tilt(
                "missingness",
                interpretation="departure from missing-at-random identification",
            ),
            tilt(
                "tipping_gamma",
                interpretation="missingness departure at which the conclusion reaches its null",
            ),
        )

    def omitted_confounding(self, *args: Any, **kwargs: Any) -> Any:
        """Bound omitted-confounder bias for a reported estimand.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.omitted_variable_bounds`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        SensitivityBounds
            Bias bounds for each requested estimand.

        See Also
        --------
        cleverly.sensitivity.omitted_variable_bounds : The same bound as a free function.

        Examples
        --------
        >>> from sklearn.linear_model import LinearRegression, LogisticRegression
        >>> from cleverly import ATE, CausalStudy, PointTreatment
        >>> from cleverly.datasets import make_linear_ate
        >>> frame, _ = make_linear_ate(n=80, seed=1)
        >>> study = CausalStudy(
        ...     frame,
        ...     design=PointTreatment(
        ...         outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
        ...     ),
        ... )
        >>> result = study.identify(ATE()).estimate(
        ...     outcome_learner=LinearRegression(),
        ...     treatment_learner=LogisticRegression(max_iter=1000),
        ...     n_folds=2,
        ...     random_state=0,
        ... )
        >>> bounds = result.sensitivity.omitted_confounding(cf_y=0.05, cf_d=0.05)
        >>> bounds.lower < result["ate"].psi < bounds.upper
        True
        """
        return self._dispatch("omitted_confounding", args, kwargs)

    def robustness_value(self, *args: Any, **kwargs: Any) -> Any:
        """Return the confounding strength needed to cross the null.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.robustness_value`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        dict of str to float
            Null-crossing strengths for the estimate and confidence limit, plus the
            maximum-bias scale.

        See Also
        --------
        cleverly.sensitivity.robustness_value : The same value as a free function.
        """
        return self._dispatch("robustness_value", args, kwargs)

    def elements(self, *args: Any, **kwargs: Any) -> Any:
        """Return the variance elements used by omitted-variable bounds.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.sensitivity_elements`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        SensitivityElements
            The variance elements the omitted-variable bounds are built from.

        See Also
        --------
        cleverly.sensitivity.sensitivity_elements : The same elements as a free function.
        """
        return self._dispatch("elements", args, kwargs)

    def benchmark(self, *args: Any, **kwargs: Any) -> Any:
        """Benchmark confounding strength against observed covariates.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.benchmark`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        BenchmarkResult
            Confounding strength expressed in units of the named observed covariates.

        See Also
        --------
        cleverly.sensitivity.benchmark : The same benchmark as a free function.
        """
        return self._dispatch("benchmark", args, kwargs)

    def contour(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate bias bounds over a confounding-strength grid.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.contour_data`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        dataframe
            Bias bounds over the requested grid of confounding strengths.

        See Also
        --------
        cleverly.sensitivity.contour_data : The same grid as a free function.
        """
        return self._dispatch("contour", args, kwargs)

    def evalue(self, *args: Any, **kwargs: Any) -> Any:
        """Return an E-value on the risk-ratio scale.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.evalue`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        EValue
            The minimum risk-ratio association that would explain the effect away.

        See Also
        --------
        cleverly.sensitivity.evalue : The same E-value as a free function.
        """
        return self._dispatch("evalue", args, kwargs)

    def missingness(self, *args: Any, **kwargs: Any) -> Any:
        """Vary unobserved-outcome odds under a missingness tilt.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.missingness_tilt`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        dataframe
            One row per tilt value and estimand.

        See Also
        --------
        cleverly.sensitivity.missingness_tilt : The same tilt as a free function.
        """
        return self._dispatch("missingness", args, kwargs)

    def tipping_gamma(self, *args: Any, **kwargs: Any) -> Any:
        """Find the missingness tilt at which the estimate crosses its null.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.tipping_gamma`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        float or None
            The tilt at which the requested estimand reaches its null.

        See Also
        --------
        cleverly.sensitivity.tipping_gamma : The same search as a free function.
        """
        return self._dispatch("tipping_gamma", args, kwargs)

    def _dispatch(self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Refuse by the declared capability, then call the declared implementation."""
        self._require(operation)
        route = SENSITIVITY_ROUTES[operation]
        if route.needs_estimand:
            # Before the cache key, so an implicit call and the explicit call it resolves
            # to share one entry rather than computing the same bound twice.
            args = self._with_default_parameter(args, kwargs)
        module = importlib.import_module(f".sensitivity.{route.module}", __package__)
        function = getattr(module, route.function)
        return _cached(
            self._result,
            f"sensitivity.{operation}",
            args,
            kwargs,
            lambda: function(self._result, *args, **kwargs),
        )

    def _with_default_parameter(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Supply the estimand only when the fit leaves no choice about which one.

        These analyses default to ``"ate"``, which a multi-arm fit never reports under that
        bare name -- it reports ``"ate[medium vs low]"``.  Filling the gap is worth doing
        when exactly one reported parameter is one the analysis applies to, and is a
        scientific choice made on the caller's behalf as soon as there are two: picking the
        first would answer about ``ey1`` on an ``ey1``/``ey0`` fit, silently returning a
        statement about a counterfactual mean to someone who asked about an effect.

        When it is ambiguous this returns the arguments untouched and the analysis refuses
        for itself -- :func:`~cleverly.sensitivity.omitted_variable.resolve_parameter` and
        :func:`~cleverly.sensitivity.missingness.missingness_tilt` both already name every
        estimand they could have answered for.
        """
        if args or "estimand" in kwargs or "ate" in self._result.estimates:
            return args
        from .sensitivity._parameters import arm_parameters

        known = arm_parameters(self._result)
        candidates = [name for name in self._result.estimates if name in known]
        return (candidates[0],) if len(candidates) == 1 else args

    def run_all(
        self, *, include_refits: bool = False, include_retargets: bool = False
    ) -> DiagnosticReport:
        """Run available sensitivity analyses that need no new arguments.

        Parameters
        ----------
        include_refits : bool
            Include operations that refit nuisance models.
        include_retargets : bool
            Include operations that retarget cached nuisance predictions.

        Returns
        -------
        DiagnosticReport
            One item for every declared sensitivity operation.
        """
        return self._run_all(include_refits=include_refits, include_retargets=include_retargets)
