"""Capability-aware post-fit diagnostics, validation, and replay metadata.

Assessment lives on a fitted result because its questions depend on the artifacts the
method actually produced.  The facades in this module do not infer support from a result
class and hope for the best: every public operation has a declaration for every public
scalar result family, including deliberate refusals.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._assessment_cache import (
    _RETAIN_PACKED,
    _cache_key,
    _cached,
    _frame_payload,
    _normalize,
    _pack_cached,
    _unpack_cached,
)
from .exceptions import CapabilityError
from .utils.frames import emit_frame
from .utils.text import format_table
from .validation.drtmle import IDENTITY_TOLERANCE
from .validation.longitudinal import (
    STITCHED_SCORE_Z_TOLERANCE,
    LongitudinalDiagnostics,
    LongitudinalNuisanceDiagnostics,
    LongitudinalNuisanceRow,
    LongitudinalScoreDiagnostics,
    LongitudinalScoreRow,
    LongitudinalStageRow,
    _longitudinal_nuisances,
    _longitudinal_scores,
    _longitudinal_stagewise,
)
from .validation.score import DEFAULT_TOLERANCE

__all__ = [
    "ASSESSMENT_CAPABILITIES",
    "SENSITIVITY_ROUTES",
    "STITCHED_SCORE_Z_TOLERANCE",
    "VALIDATION_OPERATIONS",
    "AssessmentCapability",
    "AssessmentItem",
    "AssessmentReport",
    "AssessmentStatus",
    "DiagnosticReport",
    "DiagnosticsFacade",
    "LongitudinalDiagnostics",
    "LongitudinalNuisanceDiagnostics",
    "LongitudinalNuisanceRow",
    "LongitudinalScoreDiagnostics",
    "LongitudinalScoreRow",
    "LongitudinalStageRow",
    "Replayability",
    "SensitivityFacade",
    "SensitivityRoute",
    "ValidationReport",
    "assessment_capabilities",
    "replayability",
    "validate_result",
]


#: The operations the validation battery owns, in the order it runs them.
#:
#: :func:`validate_result` runs each one argument-free, and
#: :meth:`AssessmentReport._presented` shows the validation row rather than the
#: diagnostics row of the same name.  A caller argument for one of these names would
#: therefore be answered on the diagnostics surface and then discarded, so
#: :func:`assess_result` refuses it instead.  That refusal needs a name here *and* a
#: parameter the caller could fill: ``support`` and ``nuisance_models`` take none, so an
#: argument for either is a ``TypeError`` from the signature rather than a composition
#: this module has to rule out.
VALIDATION_OPERATIONS: tuple[str, ...] = ("score_equations", "support", "nuisance_models")


class AssessmentStatus(StrEnum):  # numpydoc ignore=PR01,PR02
    """Status returned by a diagnostic or validation operation.

    ``NOT_APPLICABLE`` means that the operation does not apply to the fitted
    estimand. ``UNAVAILABLE`` means that the operation applies, but the result
    does not contain the artifacts needed to run it.

    Reach a status by name, as ``AssessmentStatus.PASSED``.

    The synthetic constructor that :class:`enum.StrEnum` gives every member is not a
    caller argument, and its parameters *differ by interpreter*: 3.11 exposes ``value``,
    ``names``, ``module``, ``qualname``, ``type``, ``start`` and ``boundary``, while 3.12
    collapses several into ``*values``.  Documenting either one satisfies numpydoc on that
    version and fails it on the other, which is how a ``Parameters`` block written for 3.12
    left `PR01` and `PR02` firing on the 3.11 build.  Ignoring both is the version-independent
    answer, and ``docs/development/contributing.md`` tells a contributor to build the
    documentation on whichever interpreter they have.
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    COMPLETED = "completed"
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
        Arguments the caller must pass. An entry either has no default, or has a
        default that the operation refuses on this result family.
    accepts_random_state : bool
        Whether a combined run can forward its top-level seed.
    requires_replay : str or None
        Name of the :class:`Replayability` field the operation needs. ``None`` means the
        operation reads stored artifacts only.
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
    #: Arguments the caller must supply, so a combined report cannot run this operation.
    #: An entry covers two cases: the operation has no default for it, or the operation
    #: refuses its own default on this fit. ``simulated_confounding`` is the second case,
    #: because a continuous fit refuses the bare ``estimand="ate"`` default. Declared
    #: here rather than special-cased by name in ``run_all``, which knows nothing about
    #: any particular operation.
    requires_arguments: tuple[str, ...] = ()
    accepts_random_state: bool = False
    #: The :class:`Replayability` field this operation needs from the stored result.
    #: One declaration per row, applied once in :meth:`_CapabilityFacade._capability_map`,
    #: because two facades each patching their own rows left ``refute`` claiming
    #: ``available=True`` on a result with no estimator while ``truncation_curve``,
    #: ``benchmark`` and ``simulated_confounding`` beside it reported the truth.
    requires_replay: str | None = None


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
    methods: Sequence[str] | None = None,
    accepts_random_state: bool = False,
    requires_replay: str | None = None,
) -> AssessmentCapability:
    return AssessmentCapability(
        operation=operation,
        result_family=family,
        methods=tuple(
            methods
            if methods is not None
            else (("tmle", "collaborative_tmle", "drtmle") if family == "point" else ("tmle",))
        ),
        available=available,
        status=status,
        required_artifacts=tuple(artifacts),
        execution=execution,
        deterministic_from_saved=deterministic,
        interpretation=interpretation,
        cost=cost,
        reason=reason,
        requires_arguments=tuple(requires_arguments),
        accepts_random_state=accepts_random_state,
        requires_replay=requires_replay,
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
        methods=("drtmle",),
    ),
    _capability(
        "truncation_curve",
        "point",
        artifacts=("fitted nuisance predictions", "targeting state"),
        execution="retarget",
        cost="moderate",
        interpretation="estimate stability across declared mechanism bounds",
        requires_replay="retarget_cached_nuisances",
    ),
    _capability(
        "refute",
        "point",
        artifacts=("fitted estimator configuration", "analysis data"),
        execution="refit",
        deterministic=False,
        cost="expensive",
        accepts_random_state=True,
        interpretation="behavior under placebo, noise, and subsampling perturbations",
        requires_replay="refit_nuisances",
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
    "simulated_confounding": SensitivityRoute(
        "simulated_confounding", "simulated_confounding", needs_estimand=True
    ),
}


def _family(result: Any) -> str:
    family = getattr(result, "assessment_family", None)
    if not isinstance(family, str) or not family:
        raise TypeError(
            "assessment requires the fitted artifact to declare a non-empty assessment_family"
        )
    return family


def _method(result: Any) -> str:
    declared = getattr(result, "fitted_method", None)
    if isinstance(declared, str) and declared:
        return declared
    raise TypeError("assessment requires the fitted artifact to declare its fitted method")


def assessment_capabilities(result: Any) -> tuple[AssessmentCapability, ...]:
    """All operation declarations for the result's family."""

    family = _family(result)
    rows = tuple(item for item in ASSESSMENT_CAPABILITIES if item.result_family == family)
    if (
        family == "point"
        and result.data.is_continuous_treatment
        and result.config.parameter_axis == "msm"
    ):
        # The arm positivity report cannot interpret a continuous density. No
        # dose-grid support diagnostic is implemented for this projection axis.
        rows = tuple(
            replace(
                row,
                available=False,
                status=AssessmentStatus.UNAVAILABLE,
                reason="a continuous MSM has no implemented dose-grid support diagnostic",
            )
            if row.operation == "support"
            else row
            for row in rows
        )
    return tuple(
        replace(
            row,
            available=False,
            status=AssessmentStatus.NOT_APPLICABLE,
            reason=(
                "the fitted DR-TMLE guard subtracts no correction term"
                if _method(result) == "drtmle"
                else "the fitted method does not use the correction system"
            ),
        )
        if row.operation == "corrections"
        and row.available
        and not getattr(result, "solved_corrections", False)
        else row
        for row in rows
    )


class _AbsentReport(Enum):
    TOKEN = "absent report"


#: What an interpreter reads when nobody supplied arguments for its operation.
_NO_ARGUMENTS: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class AssessmentItem:
    """One immutable result in a combined diagnostic or validation report.

    Parameters
    ----------
    name : str
        Operation name.
    status : AssessmentStatus
        Outcome or omission status.
    detail : str
        Interpreted findings or the omission reason.
    next_steps : tuple of str
        Suggested follow-up actions.
    _report : Any
        Retained payload, including a legitimate None, or the private absence sentinel.
        Excluded from equality. Dataframes use immutable cached storage.
    arguments : mapping of str to Any
        Effective invocation arguments, including resolved defaults and seeds.
        Excluded from equality because argument values can contain arrays.
    """

    name: str
    status: AssessmentStatus
    detail: str
    next_steps: tuple[str, ...] = ()
    _report: Any = field(default=_AbsentReport.TOKEN, compare=False, repr=False)
    arguments: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def report(self) -> Any:
        """Return the retained payload, or None when this operation did not run."""
        return None if self._report is _AbsentReport.TOKEN else _unpack_cached(self._report)


def _item_columns(items: Sequence[AssessmentItem], **extra: Any) -> dict[str, Any]:
    """The frame payload every assessment surface emits, after any leading columns."""
    return {
        **extra,
        "check": [item.name for item in items],
        "status": [item.status.value for item in items],
        "detail": [item.detail for item in items],
        "next_steps": ["; ".join(item.next_steps) for item in items],
    }


def _item_rows(items: Sequence[AssessmentItem], *, next_steps: bool = True) -> list[list[str]]:
    """The printable rows of a report table, with or without the next-step column."""
    return [
        [item.name, item.status.value, item.detail]
        + (["; ".join(item.next_steps)] if next_steps else [])
        for item in items
    ]


def _retained(item: AssessmentItem, what: str) -> Any:
    """The payload an operation kept, or a ``KeyError`` naming what did not run."""
    if item._report is _AbsentReport.TOKEN:
        raise KeyError(f"{what} {item.name!r} did not run")
    return item.report


def _distinct_steps(items: Sequence[AssessmentItem]) -> tuple[str, ...]:
    """Every suggested next step across ``items``, in its first-seen order."""
    return tuple(dict.fromkeys(step for item in items for step in item.next_steps))


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
        Whether the run allowed moderate retargets, beyond the default cheap retargets.
    backend : str or None
        Dataframe backend used by :meth:`to_frame` when ``data`` is omitted.

    See Also
    --------
    ValidationReport : The battery that reads stored artifacts only.
    cleverly.assessment.DiagnosticsFacade : What produces this report.
    cleverly.AssessmentCapability : The declaration behind one item.

    Notes
    -----
    A skipped or refused item remains in the report. A capability known to be unsupported
    is an omission. A refusal raised during an aggregate run remains unavailable,
    and later operations still run.

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
        return emit_frame(_item_columns(self.items), data, backend=self.backend)

    def report(self, name: str) -> Any:
        """Return the retained object for an operation that ran.

        Parameters
        ----------
        name : str
            Operation name.

        Returns
        -------
        Any
            The operation's retained report or dataframe.

        Raises
        ------
        KeyError
            If the operation did not run.
        """
        return _retained(self[name], "diagnostic")

    def reports(self) -> dict[str, Any]:
        """Return retained objects for operations that ran.

        Returns
        -------
        dict of str to Any
            Operation names mapped to their retained reports.
        """
        return {
            item.name: item.report for item in self.items if item._report is not _AbsentReport.TOKEN
        }

    def next_steps(self) -> tuple[str, ...]:
        """Return de-duplicated next steps in report order.

        Returns
        -------
        tuple of str
            Suggested follow-up actions in their first-seen order.
        """
        return _distinct_steps(self.items)

    def summary(self) -> str:
        """Return a printable table of operation statuses.

        Returns
        -------
        str
            A printable table, one line per requested operation.
        """
        return format_table(["diagnostic", "status", "detail", "next step"], _item_rows(self.items))


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
        return emit_frame(_item_columns(self.items), data, backend=self.backend)

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
                    _item_rows(self.items, next_steps=False),
                ),
            ]
        )


@dataclass(frozen=True)
class AssessmentReport:
    """Collect validation, diagnostics, and sensitivity in one post-fit battery.

    Parameters
    ----------
    validation : ValidationReport
        Checks that read stored fitted artifacts.
    diagnostics : DiagnosticReport
        Method and support diagnostics.
    sensitivity : DiagnosticReport
        Sensitivity analyses and explicit omissions.
    """

    validation: ValidationReport
    diagnostics: DiagnosticReport
    sensitivity: DiagnosticReport

    @property
    def backend(self) -> str | None:
        """Return the backend inherited from the validation surface."""
        return self.validation.backend

    def _presented(self) -> tuple[tuple[str, AssessmentItem], ...]:
        owned = {item.name for item in self.validation.items}
        return (
            *(("validation", item) for item in self.validation.items),
            *(("diagnostics", item) for item in self.diagnostics.items if item.name not in owned),
            *(("sensitivity", item) for item in self.sensitivity.items),
        )

    @property
    def attention(self) -> tuple[AssessmentItem, ...]:
        """Return rows with an explicit failure or warning."""
        statuses = {AssessmentStatus.FAILED, AssessmentStatus.WARNING}
        return tuple(item for _, item in self._presented() if item.status in statuses)

    @property
    def omissions(self) -> tuple[AssessmentItem, ...]:
        """Return rows that were not applicable or unavailable."""
        statuses = {AssessmentStatus.NOT_APPLICABLE, AssessmentStatus.UNAVAILABLE}
        return tuple(item for _, item in self._presented() if item.status in statuses)

    def next_steps(self) -> tuple[str, ...]:
        """Return de-duplicated next steps in presentation order.

        Returns
        -------
        tuple of str
            Suggested follow-up actions in their first-seen order.
        """
        return _distinct_steps([item for _, item in self._presented()])

    def report(self, name: str, *, surface: str | None = None) -> Any:
        """Return one retained report by operation and optional surface.

        Parameters
        ----------
        name : str
            Operation name.
        surface : {"validation", "diagnostics", "sensitivity"} or None
            Surface to select when names overlap.

        Returns
        -------
        Any
            Retained operation report.
        """
        if surface is not None:
            if surface not in {"validation", "diagnostics", "sensitivity"}:
                raise KeyError(f"unknown assessment surface {surface!r}")
            candidates = tuple((surface, item) for item in getattr(self, surface).items)
        else:
            candidates = self._presented()
        matches = [(owner, item) for owner, item in candidates if item.name == name]
        if not matches:
            raise KeyError(f"no presented assessment report named {name!r}")
        if len(matches) > 1:
            owners = [owner for owner, _ in matches]
            raise KeyError(
                f"assessment report {name!r} is ambiguous across {owners}; pass surface="
            )
        return _retained(matches[0][1], "assessment operation")

    def to_frame(self, data: Any = None) -> Any:
        """Return one row per presented surface and operation.

        Parameters
        ----------
        data : Any
            Optional dataframe whose backend selects the output type.

        Returns
        -------
        Any
            A pandas or Polars dataframe with a ``surface`` column.
        """
        rows = self._presented()
        return emit_frame(
            _item_columns([item for _, item in rows], surface=[surface for surface, _ in rows]),
            data,
            backend=self.validation.backend,
        )

    def summary(self) -> str:
        """Return the three report sections and their attention lists.

        Returns
        -------
        str
            Printable validation, diagnostics, sensitivity, attention, and omission sections.
        """
        sections = []
        for surface in ("validation", "diagnostics", "sensitivity"):
            rows = [item for owner, item in self._presented() if owner == surface]
            sections.extend(
                [
                    surface.capitalize(),
                    "-" * len(surface),
                    format_table(["operation", "status", "detail", "next step"], _item_rows(rows)),
                    "",
                ]
            )
        sections.append("Attention: " + (", ".join(item.name for item in self.attention) or "none"))
        sections.append("Omissions: " + (", ".join(item.name for item in self.omissions) or "none"))
        return "\n".join(sections)


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
        return Replayability(True, False, False, False, False, ("estimator configuration",))
    return Replayability(True, True, False, True, False)


#: What each :class:`Replayability` slot lets an operation do, in the words a refusal uses.
_REPLAY_WORK: dict[str, str] = {
    "summarize_existing_artifacts": "summarizing the stored artifacts",
    "retarget_cached_nuisances": "retargeting the cached nuisances",
    "evaluate_stored_representer": "evaluating the stored representer",
    "refit_nuisances": "refitting the nuisance models",
    "evaluate_new_data": "scoring new observations",
}


def _require_argument_mapping(arguments: Any) -> None:
    """Refuse a per-operation argument block that is not a mapping.

    Owned here rather than restated per caller: ``assess_result`` has to split the block
    across two facades before either can validate it, so it reaches the same wrong type
    first and must refuse it in the same words.
    """
    if arguments is not None and not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping from operation names to mappings")


def _accepts_arguments(facade: _CapabilityFacade, operation: str) -> bool:
    """Whether ``operation`` declares a parameter a caller could fill.

    Read from the routed callable rather than from a list of names, so an operation that
    gains or loses a parameter needs no second edit here.  The first parameter is the
    result or the facade itself, which no caller supplies.
    """
    function, _ = facade._routed_callable(operation)
    return len(inspect.signature(function).parameters) > 1


def _method_gated(item: AssessmentCapability, method: str) -> AssessmentCapability:
    """Refuse a row the fitted method does not declare."""
    if (
        not item.available
        or method in item.methods
        or (method == "unknown" and item.execution == "summarize")
    ):
        return item
    reason = (
        # An artifact records a method or it does not. Reading "the fitted method
        # 'unknown' does not support this operation" sent a reader to look up a method
        # named ``unknown``, when the cause is that the artifact names no method at all.
        "this artifact records no fitted method, so its support for this operation "
        "cannot be established"
        if method == "unknown"
        else f"the fitted method {method!r} does not support this operation"
    )
    return replace(item, available=False, status=AssessmentStatus.NOT_APPLICABLE, reason=reason)


def _replay_gated(item: AssessmentCapability, replay: Replayability) -> AssessmentCapability:
    """Refuse a row whose declared replay slot this stored result cannot supply."""
    slot = item.requires_replay
    if not item.available or slot is None or getattr(replay, slot):
        return item
    missing = list(replay.unreconstructible)
    return replace(
        item,
        available=False,
        status=AssessmentStatus.UNAVAILABLE,
        reason=(
            f"{_REPLAY_WORK[slot]} needs artifacts this stored result no longer carries; "
            f"unavailable slots: {missing}"
        ),
    )


class _CapabilityFacade:
    """Lookup, refusal, and combined-report machinery shared by both public facades.

    The two facades answer different questions from different declarations, but the way
    they *route* a question is one algorithm: find the operation's row, refuse by that row
    when it is unavailable, and in a combined report skip what the caller has not paid for.
    Written twice, it drifted five ways -- the availability and cost checks in opposite
    orders, two spellings of the cost gate, one refusal that re-derived a reason the record
    already carried, different caught-exception sets, and one side that discarded what its
    operations returned and so could only ever report ``passed``.  Subclasses supply
    :attr:`_declared` and the two labels below; everything else is settled here.

    A subclass declares what its operations *are*.  It does not decide whether this stored
    result can run them: the method gate and the replay gate live here and run over every
    row.  Both were once written per facade, and ``refute`` reached the shipped surface
    claiming ``available=True`` on a result with no estimator because the diagnostics side
    patched one row by name and forgot the other.
    """

    #: What an operation of this kind is called in a refusal: ``diagnostic 'refute' is ...``.
    _kind: str
    #: The attribute a caller reaches it through, for the ``next_steps`` of a skipped row.
    _attribute: str

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def _declared(self) -> tuple[AssessmentCapability, ...]:
        raise NotImplementedError  # pragma: no cover - subclasses declare their own

    @property
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        """Return declared operations, gated by the fitted method and by replayability."""
        return tuple(self._capability_map.values())

    @cached_property
    def _capability_map(self) -> dict[str, AssessmentCapability]:
        method = _method(self._result)
        replay = replayability(self._result)
        return {
            item.operation: _replay_gated(_method_gated(item, method), replay)
            for item in self._declared
        }

    def capability(self, operation: str) -> AssessmentCapability:
        try:
            return self._capability_map[operation]
        except KeyError:
            raise KeyError(f"unknown {self._kind} {operation!r}") from None

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

    def _capability_for_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        """Resolve request-specific availability and cost before aggregate execution."""
        return self.capability(operation)

    def _skipped(
        self,
        capability: AssessmentCapability,
        arguments: Mapping[str, Any],
        *,
        include_refits: bool,
        include_retargets: bool,
    ) -> AssessmentItem | None:
        """The row a combined report owes an operation, or ``None`` to run it.

        Availability first, then the missing argument, and the cost last. Every gate above
        the cost gate refuses for a reason no flag can pay off, and a report that named the
        cost first sent the caller to ``include_refits=True`` and then, on the very next
        call, to the argument it never mentioned. A refusal has to name the first thing
        that is wrong.

        Parameters
        ----------
        capability : AssessmentCapability
            The row resolved for this request, not the bare declaration.
        arguments : mapping of str to Any
            Arguments the caller supplied for this operation.
        include_refits : bool
            Whether the caller paid for operations that refit nuisance models.
        include_retargets : bool
            Whether the caller paid for moderate retargets.

        Returns
        -------
        AssessmentItem or None
            The omission to report, or ``None`` when the operation may run.
        """
        if not capability.available:
            return _item_from_capability(capability)
        missing = tuple(name for name in capability.requires_arguments if name not in arguments)
        if missing:
            return _missing_argument_item(capability, self._attribute, missing)
        return _cost_refusal(capability, self._attribute, include_refits, include_retargets)

    def _run_all(
        self,
        *,
        include_refits: bool,
        include_retargets: bool,
        supplied: Mapping[str, Mapping[str, Any]],
        random_state: int | None,
    ) -> DiagnosticReport:
        """Execute the combined report from arguments :meth:`_validated_arguments` cleared.

        Validation is the caller's job because binding every supplied operation's signature
        is not free, and the battery reaches this method through ``assess_result``, which
        has already had to validate both facades to fail before either one runs.  Doing it
        again here bound each signature a second and a third time per ``assess()`` call.
        """

        def compute() -> DiagnosticReport:
            items = []
            for declared in self.capabilities:
                operation_arguments = dict(supplied.get(declared.operation, {}))
                capability = self._capability_for_arguments(declared.operation, operation_arguments)
                skipped = self._skipped(
                    capability,
                    operation_arguments,
                    include_refits=include_refits,
                    include_retargets=include_retargets,
                )
                if skipped is not None:
                    items.append(skipped)
                    continue
                try:
                    if capability.accepts_random_state and random_state is not None:
                        operation_arguments["random_state"] = random_state
                    token = _RETAIN_PACKED.set(True)
                    try:
                        report = getattr(self, capability.operation)(**operation_arguments)
                    finally:
                        _RETAIN_PACKED.reset(token)
                except CapabilityError as error:
                    # A capability refusal is an expected result of asking a broad battery to
                    # inspect one fitted object.  Keep it visible as an omission and continue to
                    # later rows.  Catch only the refusal type: ``KeyError`` and ``TypeError``
                    # are structural defects, and turning either into a scientific-sounding
                    # report row would hide an implementation error.
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            f"the operation declined this request: {error}",
                            (
                                f"call result.{self._attribute}.{capability.operation}() "
                                f"directly for the refusal in full",
                            ),
                            arguments=self._effective_arguments(
                                capability.operation, operation_arguments, None
                            ),
                        )
                    )
                else:
                    effective = self._effective_arguments(
                        capability.operation, operation_arguments, report
                    )
                    interpreted = INTERPRETERS[capability.operation](
                        report, self._result, operation_arguments
                    )
                    items.append(
                        replace(
                            interpreted,
                            _report=_pack_cached(report, self._result.data.backend),
                            arguments=effective,
                        )
                    )
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
            {
                "include_refits": include_refits,
                "include_retargets": include_retargets,
                "arguments": supplied,
                "random_state": random_state,
            },
            compute,
        )

    def _validated_arguments(
        self,
        arguments: Mapping[str, Mapping[str, Any]] | None,
        random_state: int | None,
    ) -> dict[str, dict[str, Any]]:
        _require_argument_mapping(arguments)
        if arguments is None:
            return {}
        declared = {row.operation for row in self.capabilities}
        unknown = sorted(set(arguments) - declared)
        if unknown:
            raise KeyError(f"unknown {self._kind} operation(s): {unknown}")
        validated: dict[str, dict[str, Any]] = {}
        for operation, values in arguments.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"arguments[{operation!r}] must be a mapping")
            kwargs = dict(values)
            capability = self.capability(operation)
            if random_state is not None and "random_state" in kwargs:
                raise ValueError(
                    f"random_state was supplied both to run_all and arguments[{operation!r}]"
                )
            if "random_state" in kwargs and not capability.accepts_random_state:
                raise TypeError(f"{operation!r} does not accept random_state")
            self._bind_arguments(operation, kwargs, partial=True, resolve=capability.available)
            validated[operation] = kwargs
        return validated

    def _routed_callable(self, operation: str) -> tuple[Callable[..., Any], bool]:
        if self._attribute == "sensitivity":
            route = SENSITIVITY_ROUTES[operation]
            module = importlib.import_module(f".sensitivity.{route.module}", __package__)
            return getattr(module, route.function), True
        if operation == "refute":
            from .validation.refute import refute

            return refute, True
        return getattr(type(self), operation), False

    def _with_default_parameter(
        self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Resolve a sensitivity parameter in the subclass that owns those routes."""
        raise NotImplementedError  # pragma: no cover - diagnostics never need this route

    def _bind_arguments(
        self,
        operation: str,
        kwargs: Mapping[str, Any],
        *,
        partial: bool,
        args: tuple[Any, ...] = (),
        resolve: bool = True,
    ) -> inspect.BoundArguments:
        function, result_first = self._routed_callable(operation)
        if (
            resolve
            and self._attribute == "sensitivity"
            and SENSITIVITY_ROUTES[operation].needs_estimand
        ):
            args = self._with_default_parameter(operation, args, dict(kwargs))
        first = self._result if result_first else self
        signature = inspect.signature(function)
        binder = signature.bind_partial if partial else signature.bind
        return binder(first, *args, **kwargs)

    def _effective_arguments(
        self,
        operation: str,
        kwargs: Mapping[str, Any],
        report: Any,
        *,
        args: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        bound = self._bind_arguments(operation, kwargs, partial=False, args=args)
        effective = _bound_arguments(bound, report)
        if operation == "evalue" and effective.get("estimand") is None:
            # Explicit OR requests choose the approximation; keep None replayable there.
            source = getattr(report, "source_estimand", None)
            if source is not None:
                from .sensitivity.evalue import _DERIVED_RR

                selection = self._evalue_selection(None)
                if (
                    selection.branch != _DERIVED_RR
                    or _arm_source_target(self._result, source) != "or"
                ):
                    effective["estimand"] = source
        return effective

    def _evalue_selection(self, estimand: str | None) -> Any:
        raise NotImplementedError

    def _invoke(self, operation: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        function, _ = self._routed_callable(operation)
        bound = self._bind_arguments(operation, kwargs, partial=False, args=args)
        effective = _bound_arguments(bound)

        def compute() -> Any:
            return function(*bound.args, **bound.kwargs)

        if operation == "evalue":
            from .sensitivity.evalue import _evalue_from_selection

            selection = self._evalue_selection(effective["estimand"])

            def compute() -> Any:
                return _evalue_from_selection(self._result, selection)

        report = _cached(self._result, f"{self._attribute}.{operation}", (), effective, compute)
        resolved = _bound_arguments(bound, report)
        if _normalize(resolved) != _normalize(effective):
            self._result.assessment_cache[
                _cache_key(f"{self._attribute}.{operation}", (), resolved)
            ] = _pack_cached(report, self._result.data.backend)
        return report


def _arm_source_target(result: Any, source: str) -> str:
    from .sensitivity._parameters import arm_parameter_keys

    return str(arm_parameter_keys(result)[source].estimand)


def _bound_arguments(bound: inspect.BoundArguments, report: Any = None) -> dict[str, Any]:
    """Canonicalize one public invocation for execution, caching, and replay."""
    bound.apply_defaults()
    effective = dict(bound.arguments)
    effective.pop(next(iter(bound.signature.parameters)), None)
    for name, parameter in bound.signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            effective.update(effective.pop(name, {}))
    if effective.get("random_state") is None:
        resolved = getattr(report, "random_state", getattr(report, "root_seed", None))
        if resolved is not None:
            effective["random_state"] = resolved
    return effective


def _missing_argument_item(
    capability: AssessmentCapability, attribute: str, missing: tuple[str, ...]
) -> AssessmentItem:
    """The skip a combined report owes an operation whose argument it cannot choose.

    A combined report runs every operation argument-free, so one with a required argument
    and no default cannot appear in it.  Choosing a value here -- which covariates to
    benchmark against -- would be a scientific choice made silently on the caller's behalf.

    A row may declare more than one argument, so the sentence has to agree in number.
    ``", ".join`` alone rendered "an explicit grid, estimand argument", which reads as one
    argument named "grid, estimand".
    """
    needed = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} and {missing[-1]}"
    phrase = (
        f"an explicit {needed} argument" if len(missing) == 1 else f"explicit {needed} arguments"
    )
    return AssessmentItem(
        capability.operation,
        AssessmentStatus.UNAVAILABLE,
        f"needs {phrase}, which a combined report has no basis to choose",
        (f"call result.{attribute}.{capability.operation}() directly with {needed}",),
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
    allowed = {
        "summarize": True,
        "refit": include_refits,
        "retarget": include_retargets or capability.cost == "cheap",
    }
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

    @cached_property
    def _declared(self) -> tuple[AssessmentCapability, ...]:
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

        **A cross-fitted longitudinal fit gets two rows per node**, because it poses two
        questions with different right answers.  Its ``"solver"`` row asks whether every
        outer fold reached the root of its own equation, which ``tolerance`` gates as
        above.  Its ``"stitching"`` row asks whether the score of the *stitched* fit sits
        where sampling would leave it -- which is not zero, because each fold fits its
        coefficient on rows it does not report -- and is gated in standard errors by
        :data:`STITCHED_SCORE_Z_TOLERANCE` instead.  Holding that row to ``tolerance``
        would fail every cross-fitted fit for doing what the construction does.
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
        return self._invoke("refute", (), kwargs)

    def run_all(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> DiagnosticReport:
        """Run available diagnostics that need no new arguments.

        Parameters
        ----------
        include_refits : bool
            Include operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets; cheap E-value retargets run by default.
        arguments : mapping or None
            Per-operation keyword arguments.
        random_state : int or None
            Common seed for stochastic refit operations.

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
        return self._run_all(
            include_refits=include_refits,
            include_retargets=include_retargets,
            supplied=self._validated_arguments(arguments, random_state),
            random_state=random_state,
        )


def _item_from_capability(capability: AssessmentCapability) -> AssessmentItem:
    return AssessmentItem(
        capability.operation,
        capability.status,
        capability.reason or capability.interpretation,
    )


def _score_item(
    report: Any, result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    ratios = []
    for row in getattr(report, "rows", ()):
        if hasattr(row, "ratio"):
            ratios.append(float(row.ratio))
        elif getattr(row, "kind", None) == "stitching":
            ratios.append(abs(float(row.z)) / float(report.z_tolerance))
        else:
            ratios.append(float(row.relative_score) / float(report.tolerance))
    passed = bool(getattr(report, "passed", False))
    conditioning = _reduction_conditioning_warning(result) if passed else None
    status = (
        AssessmentStatus.WARNING
        if conditioning
        else AssessmentStatus.PASSED
        if passed
        else AssessmentStatus.FAILED
    )
    worst = max((value for value in ratios if np.isfinite(value)), default=float("nan"))
    detail = f"{len(ratios)} score row(s); worst abs(score) / threshold = {worst:.3g}"
    if conditioning:
        detail += f"; {conditioning}"
    steps = (
        ("inspect result.repeats[*].fluctuations['mean'].reduction.ill_conditioned",)
        if conditioning
        else ()
        if passed
        else ("inspect result.diagnostics.score_equations()",)
    )
    return AssessmentItem("score_equations", status, detail, steps)


def _support_metrics(report: Any) -> tuple[float | None, float | None]:
    truncated: list[float] = []
    ess: list[float] = []
    if hasattr(report, "truncated"):
        truncated.append(float(report.truncated.get("fraction", 0.0)))
        for values in getattr(report, "effective_sample_size", {}).values():
            if "ratio" in values:
                ess.append(float(values["ratio"]))
        for values in getattr(report, "mechanisms", {}).values():
            if "ess_ratio" in values:
                ess.append(float(values["ess_ratio"]))
    if isinstance(report, LongitudinalDiagnostics):
        truncated.extend(float(row.share_truncated) for row in report.rows)
        ess.extend(float(row.effective_n / row.n_followed) for row in report.rows if row.n_followed)
    if isinstance(report, Mapping):
        truncated.extend(float(getattr(row, "capped_fraction", 0.0)) for row in report.values())
        ess.extend(float(getattr(row, "ess_ratio", np.nan)) for row in report.values())
    regimes = getattr(report, "regimes", None)
    if regimes:
        ess.extend(float(getattr(row, "ess_ratio", np.nan)) for row in regimes.values())
    clean_ess = [value for value in ess if np.isfinite(value)]
    return (max(truncated) if truncated else None, min(clean_ess) if clean_ess else None)


def _support_facts(truncated: float | None, ess: float | None) -> list[str]:
    """Present the two support metrics, on one scale, for every row that reports them."""
    facts = []
    if truncated is not None:
        facts.append(f"maximum truncated fraction {truncated:.1%}")
    if ess is not None:
        facts.append(f"minimum effective-sample-size ratio {ess:.1%}")
    return facts


def _support_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    warning = _support_warning(report)
    facts = _support_facts(*_support_metrics(report))
    if warning:
        facts.append(warning)
    detail = "; ".join(facts) if facts else "stored support report completed"
    return AssessmentItem(
        "support",
        AssessmentStatus.WARNING if warning else AssessmentStatus.PASSED,
        detail,
        () if warning is None else ("inspect result.diagnostics.support()",),
    )


def _nuisance_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    findings = tuple(getattr(report, "findings", ()))
    if findings:
        return AssessmentItem(
            "nuisance_models",
            AssessmentStatus.WARNING,
            "; ".join(findings),
            ("inspect result.diagnostics.nuisance_models()",),
        )
    if isinstance(report, LongitudinalNuisanceDiagnostics):
        finite = [row.mse for row in report.rows if np.isfinite(row.mse)]
        if not finite:
            return AssessmentItem(
                "nuisance_models",
                AssessmentStatus.WARNING,
                "no finite stagewise held-out loss is available",
                ("inspect result.diagnostics.nuisance_models()",),
            )
        detail = f"{len(finite)} stagewise held-out loss value(s) are available"
    else:
        detail = f"{len(getattr(report, 'models', ()))} nuisance model report(s) are available"
    return AssessmentItem("nuisance_models", AssessmentStatus.COMPLETED, detail)


def _correction_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    identity = [abs(float(row.residual)) for row in report.rows if np.isfinite(row.residual)]
    magnitude = [abs(float(row.reported)) for row in report.rows]
    detail = (
        f"contract={report.contract}; maximum identity residual "
        f"{max(identity, default=float('nan')):.3g}; maximum reported correction magnitude "
        f"{max(magnitude, default=float('nan')):.3g}"
    )
    return AssessmentItem(
        "corrections",
        AssessmentStatus.PASSED if report.passed else AssessmentStatus.FAILED,
        detail,
        () if report.passed else ("inspect result.diagnostics.corrections()",),
    )


def _range(values: Sequence[Any]) -> tuple[float, float] | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return (min(finite), max(finite)) if finite else None


def _format_range(values: tuple[float, float] | None) -> str:
    return "no finite values" if values is None else f"[{values[0]:.4g}, {values[1]:.4g}]"


def _truncation_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    payload = _frame_payload(report)
    bounds = _range(payload.get("bound", payload.get("g_bound", ())))
    estimates = _range(payload.get("psi", payload.get("estimate", ())))
    return AssessmentItem(
        "truncation_curve",
        AssessmentStatus.COMPLETED,
        f"evaluated bound range {_format_range(bounds)}; estimate range {_format_range(estimates)}",
    )


def _refute_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    failed = [test.name for test in report.tests if not test.passed]
    return AssessmentItem(
        "refute",
        AssessmentStatus.FAILED if failed else AssessmentStatus.PASSED,
        "all refutation tests passed"
        if not failed
        else f"failed tests {failed}; inspect their retained draws",
        ()
        if not failed
        else tuple(f"inspect result.diagnostics.refute().draws_frame({name!r})" for name in failed),
    )


def _omitted_item(
    report: Any, _result: Any, arguments: Mapping[str, Any] | None = None
) -> AssessmentItem:
    """Interpret one omitted-confounding bound, and say which strengths produced it.

    The bound is a statement about an assumed confounder, so a reader who did not choose
    ``cf_y`` and ``cf_d`` has to be told that the library did.  The prefix is written here
    rather than by the combined report because it is a fact about *this* operation, and a
    name-matching branch in generic routing machinery is what the declaration tables exist
    to remove.

    ``arguments=None`` means the caller's choices are unknown rather than absent, so the
    row claims no provenance.  ``run_all`` always passes the mapping it cleared, so only a
    direct call reaches that case.
    """
    spans = report.lower <= report.null_hypothesis <= report.upper
    defaults = [] if arguments is None else [n for n in ("cf_y", "cf_d") if n not in arguments]
    provenance = ""
    if defaults:
        provenance = (
            "at the default strengths; "
            if len(defaults) == 2
            else f"at the default {defaults[0]} strength; "
        )
    return AssessmentItem(
        "omitted_confounding",
        AssessmentStatus.WARNING if spans else AssessmentStatus.COMPLETED,
        f"{provenance}"
        f"cf_y={report.cf_y:.3g}, cf_d={report.cf_d:.3g}, rho={report.rho:.3g}; "
        f"bias-adjusted interval [{report.lower:.4g}, {report.upper:.4g}]",
        () if not spans else ("inspect the retained omitted-confounding bounds",),
    )


def _robustness_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    return AssessmentItem(
        "robustness_value",
        AssessmentStatus.COMPLETED,
        f"point robustness value {report['rv']:.4g}; confidence-limit value {report['rva']:.4g}",
    )


def _elements_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    return AssessmentItem(
        "elements",
        AssessmentStatus.COMPLETED,
        f"sigma2={report.sigma2:.4g}, nu2={report.nu2:.4g}, max_bias={report.max_bias:.4g}",
    )


def _contour_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    payload = _frame_payload(report)
    return AssessmentItem(
        "contour",
        AssessmentStatus.COMPLETED,
        f"grid {len(set(payload['cf_d']))} x {len(set(payload['cf_y']))}; "
        f"cf_d range {_format_range(_range(payload['cf_d']))}; "
        f"cf_y range {_format_range(_range(payload['cf_y']))}; "
        f"value range {_format_range(_range(payload['value']))}; inspect the retained frame",
    )


def _benchmark_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    return AssessmentItem(
        "benchmark",
        AssessmentStatus.COMPLETED,
        f"covariates={report.covariates}; cf_y={report.cf_y:.3g}, cf_d={report.cf_d:.3g}, "
        f"rho={report.rho:.3g}, delta_psi={report.delta_psi:.4g}",
    )


def _conditioning_population(report: Any) -> tuple[float | None, float | None, bool]:
    """The conditioning arm's smallest and unperturbed share, and whether it collapsed.

    A surface whose target population is ``"perturbed_treatment_group"`` rebuilds its ATT
    or ATC population in every cell, so a cell can move because its population changed and
    not because a confounding path opened.  A one-line report that gives only the movement
    hides that third channel.

    **The collapse rule.**  The anchor cell perturbs nothing, so its
    ``target_population_fraction`` is the unperturbed share of the conditioning arm.  A
    surface has collapsed when its smallest cell keeps **less than half** of that anchor
    share.  The rule is relative to the fit's own anchor rather than an absolute cut, so a
    study whose treated arm is a tenth of the sample is not warned about for that alone;
    the constant is the half, which names "most of the group is gone" and is not tuned to
    any dataset.  A surface that averages over its baseline population never collapses,
    because every cell reports a fraction of one.

    **Every cell, not every successful cell.**  The cell that collapses hardest is the one
    whose conditioning arm emptied, and that cell fails: its refit raises
    :exc:`~cleverly.exceptions.DataError` because the arm keeps no positive-weight row.
    ``simulated_confounding`` records ``target_population_fraction`` from the perturbed
    treatment *before* it refits, so a failed cell still carries the fraction that explains
    the failure.  A minimum over ``successful_cells`` therefore drops exactly the collapse
    this rule exists to name, and where only the anchor survives it reports the anchor
    against itself.  The minimum below runs over every cell that recorded a fraction.  A
    cell that failed before the surface built its treatment records ``None`` and is
    skipped, because it measured no population.

    Parameters
    ----------
    report : Any
        A :class:`~cleverly.sensitivity.SimulatedConfoundingResult`.

    Returns
    -------
    tuple of (float or None, float or None, bool)
        The smallest recorded cell fraction, the anchor fraction, and whether the
        conditioning population collapsed.
    """
    fractions = [
        cell.target_population_fraction
        for cell in report.cells
        if cell.target_population_fraction is not None
    ]
    anchor = next(
        (
            cell.target_population_fraction
            for cell in report.cells
            if cell.treatment_strength == 0.0 and cell.outcome_strength == 0.0
        ),
        None,
    )
    minimum = min(fractions, default=None)
    collapsed = (
        report.population == "perturbed_treatment_group"
        and minimum is not None
        and anchor is not None
        and minimum < 0.5 * anchor
    )
    return minimum, anchor, collapsed


def _simulated_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    movements = [
        abs(float(cell.displacement))
        for cell in report.successful_cells
        if cell.displacement is not None
    ]
    corner = report.cells[-1].induced_treatment_association if report.cells else None
    minimum, anchor, collapsed = _conditioning_population(report)
    detail = (
        f"maximum successful displacement {max(movements, default=float('nan')):.4g}; "
        f"movement scale {report.movement_scale}; "
        f"failed cells {len(report.failures)}; corner association {corner}; "
        + "; ".join(report.population_lines())
        + f"; minimum target population fraction {_format_fraction(minimum)}"
        + f" against anchor {_format_fraction(anchor)}"
    )
    advice = []
    if report.failures:
        advice.append("inspect the retained cell failures")
    if collapsed:
        advice.append(
            "read target_population_fraction beside the movement; the conditioning group "
            "keeps under half its unperturbed share, so part of the movement is a change "
            "of population"
        )
    return AssessmentItem(
        "simulated_confounding",
        AssessmentStatus.WARNING if report.failures or collapsed else AssessmentStatus.COMPLETED,
        detail,
        tuple(advice),
    )


def _format_fraction(value: float | None) -> str:
    """Render one population fraction, or name its absence.

    Parameters
    ----------
    value : float or None
        A conditioning-arm share, or ``None`` when no cell recorded one.

    Returns
    -------
    str
        The share to four significant figures, or ``"n/a"``.
    """
    return "n/a" if value is None else f"{value:.4g}"


def _evalue_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    detail = (
        f"point={report.point:.4g}, limit={report.limit:.4g}, source scale={report.scale}; "
        + ("approximate conversion" if report.approximate else "exact risk-ratio branch")
    )
    if report.limit == 1.0:
        detail += "; the interval already includes the null"
    truncated = getattr(report, "truncated_bound", None)
    if truncated is not None:
        # The reported lower bound is the boundary of the parameter space, not a converted
        # confidence limit.  A row that shows only ``point`` and ``limit`` would hide that.
        detail += f"; the lower risk-ratio bound is truncated at 0 from {float(truncated):.4g}"
    return AssessmentItem("evalue", AssessmentStatus.COMPLETED, detail)


def _missingness_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    payload = _frame_payload(report)
    return AssessmentItem(
        "missingness",
        AssessmentStatus.COMPLETED,
        f"gamma range {_format_range(_range(payload['gamma']))}; "
        f"estimate range {_format_range(_range(payload['psi']))}",
    )


def _tipping_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    detail = (
        "no tipping value occurred in the searched interval"
        if report is None
        else f"tipping gamma {float(report):.4g}"
    )
    return AssessmentItem("tipping_gamma", AssessmentStatus.COMPLETED, detail)


def _stagewise_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    truncated, ess = _support_metrics(report)
    # The same two numbers ``_support_item`` reports, so they carry the same presentation.
    # Interpolated raw they printed "0.8888888888888887" beside a sibling row reading
    # "88.9%", and "None" where the sibling says nothing at all.
    return AssessmentItem(
        "stagewise",
        AssessmentStatus.COMPLETED,
        "; ".join(
            [
                f"{len(report.rows)} stage row(s)",
                *_support_facts(truncated, ess),
            ]
        ),
    )


#: How each operation's own report becomes one report row.
#:
#: The third argument is what the caller supplied for that operation.  An interpreter that
#: has to say how its answer was parameterized reads it here, so that the combined report
#: keeps no branch on an operation name.
#:
#: Internal routing rather than a caller surface: the values take report objects this module
#: builds, so it is deliberately absent from ``__all__``.
INTERPRETERS: dict[str, Callable[[Any, Any, Mapping[str, Any]], AssessmentItem]] = {
    "score_equations": _score_item,
    "support": _support_item,
    "nuisance_models": _nuisance_item,
    "corrections": _correction_item,
    "truncation_curve": _truncation_item,
    "refute": _refute_item,
    "stagewise": _stagewise_item,
    "omitted_confounding": _omitted_item,
    "robustness_value": _robustness_item,
    "elements": _elements_item,
    "contour": _contour_item,
    "benchmark": _benchmark_item,
    "simulated_confounding": _simulated_item,
    "evalue": _evalue_item,
    "missingness": _missingness_item,
    "tipping_gamma": _tipping_item,
}


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


def validate_result(result: Any, diagnostics: DiagnosticsFacade | None = None) -> ValidationReport:
    """Run only cheap, cache-only checks appropriate to this fitted method."""

    def compute() -> ValidationReport:
        facade = result.diagnostics if diagnostics is None else diagnostics
        items = []
        for name in VALIDATION_OPERATIONS:
            capability = facade.capability(name)
            if not capability.available:
                items.append(_item_from_capability(capability))
                continue
            report = getattr(facade, name)()
            items.append(
                replace(
                    INTERPRETERS[name](report, result, _NO_ARGUMENTS),
                    _report=_pack_cached(report, result.data.backend),
                )
            )
        return ValidationReport(tuple(items), result.data.backend)

    return _cached(result, "validate", (), {}, compute)


def assess_result(
    result: Any,
    *,
    include_refits: bool = False,
    include_retargets: bool = False,
    arguments: Mapping[str, Mapping[str, Any]] | None = None,
    random_state: int | None = None,
) -> AssessmentReport:
    """Compose a battery from one pair of operation facades."""
    _require_argument_mapping(arguments)
    supplied = {} if arguments is None else dict(arguments)
    diagnostics, sensitivity = result.diagnostics, result.sensitivity
    diagnostic_names = set(diagnostics._capability_map)
    sensitivity_names = set(sensitivity._capability_map)
    unknown = sorted(set(supplied) - diagnostic_names - sensitivity_names)
    if unknown:
        raise KeyError(f"unknown assessment operation(s): {unknown}")
    # Only an operation that would really hide an answer is refused.  An empty mapping
    # applies nothing, so nothing can be hidden, and an operation with no parameters can
    # take no argument at all.  For those two, ``_validated_arguments`` below reports the
    # precise ``TypeError`` from binding the caller's own keywords, which names the
    # signature rather than substituting a vaguer refusal.
    owned = [
        name
        for name in VALIDATION_OPERATIONS
        if supplied.get(name) and _accepts_arguments(diagnostics, name)
    ]
    if owned:
        # The battery presents the validation row for these names and hides the
        # diagnostics row of the same name.  Running the caller's arguments would answer
        # the caller's question on the hidden row and then show the argument-free answer,
        # so a check that failed at the requested tolerance would never reach
        # ``attention``.  Refuse the composition and name the call that does answer it.
        calls = ", ".join(f"result.diagnostics.{name}(...)" for name in owned)
        raise CapabilityError(
            f"the validation battery owns {owned} and runs each one argument-free, so "
            f"assess() cannot apply arguments to them; call {calls} for one answer, or "
            f"result.diagnostics.run_all(arguments=...) for the diagnostics surface"
        )
    # Validate both surfaces before either one runs, and carry the cleared arguments
    # forward.  A block that is wrong for the sensitivity facade must not first make the
    # diagnostics facade refit anything.
    diagnostic_arguments = diagnostics._validated_arguments(
        {k: v for k, v in supplied.items() if k in diagnostic_names}, random_state
    )
    sensitivity_arguments = sensitivity._validated_arguments(
        {k: v for k, v in supplied.items() if k in sensitivity_names}, random_state
    )
    return AssessmentReport(
        validation=validate_result(result, diagnostics),
        diagnostics=diagnostics._run_all(
            include_refits=include_refits,
            include_retargets=include_retargets,
            supplied=diagnostic_arguments,
            random_state=random_state,
        ),
        sensitivity=sensitivity._run_all(
            include_refits=include_refits,
            include_retargets=include_retargets,
            supplied=sensitivity_arguments,
            random_state=random_state,
        ),
    )


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
    def _declared(self) -> tuple[AssessmentCapability, ...]:
        family = _family(self._result)
        longitudinal = family == "longitudinal"
        missing = (
            False
            if longitudinal
            else getattr(self._result.nuisance, "missingness", None) is not None
        )
        # Whether a *point* fit is replayable is settled by ``requires_replay`` below.
        # This row only says whether the analysis exists for the family at all.
        benchmarkable = not longitudinal
        # ``simulated_confounding`` refuses the bare ``ate`` default on a continuous fit.
        # A binary arm, fixed-regime, incremental, or MSM fit can use the facade's sole-
        # parameter substitution. Several eligible aliases require an explicit choice.
        continuous = not longitudinal and bool(
            getattr(self._result.data, "is_continuous_treatment", False)
        )
        if longitudinal or continuous:
            binary_needs_estimand = False
        else:
            from .sensitivity._simulated_confounding_request import (
                _eligible_binary_parameter_names,
            )

            binary_parameters = _eligible_binary_parameter_names(self._result)
            binary_needs_estimand = "ate" not in binary_parameters and len(binary_parameters) > 1
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
                    else "no longitudinal benchmarking derivation is registered"
                ),
                requires_arguments=("covariates",),
                accepts_random_state=True,
                requires_replay="refit_nuisances",
                family=family,
            ),
            _capability(
                "simulated_confounding",
                artifacts=("fitted estimator configuration", "analysis data"),
                execution="refit",
                deterministic=False,
                cost="expensive",
                interpretation="estimate movement under a simulated common cause",
                available=benchmarkable,
                status=AssessmentStatus.PASSED if benchmarkable else AssessmentStatus.UNAVAILABLE,
                reason=(
                    None
                    if benchmarkable
                    else "no longitudinal simulated-confounder perturbation law is implemented"
                ),
                requires_arguments=("grid", "estimand")
                if continuous or binary_needs_estimand
                else ("grid",),
                accepts_random_state=True,
                requires_replay="refit_nuisances",
                family=family,
            ),
            standard(
                "contour",
                artifacts=("fitted representer", "outcome residuals"),
                interpretation="bias bounds over a grid of confounding strengths",
                cost="moderate",
            ),
            self._evalue_row(),
            tilt(
                "missingness",
                interpretation="departure from missing-at-random identification",
            ),
            tilt(
                "tipping_gamma",
                interpretation="missingness departure at which the conclusion reaches its null",
            ),
        )

    @cached_property
    def _evalue_selections(self) -> dict[str | None, Any]:
        return {}

    def _evalue_selection(self, estimand: str | None) -> Any:
        from .sensitivity.evalue import _EValueRefusal, _select_evalue

        if estimand not in self._evalue_selections:
            try:
                selected: Any = _select_evalue(self._result, estimand)
            except _EValueRefusal as error:
                selected = (error.status, str(error))
            self._evalue_selections[estimand] = selected
        selected = self._evalue_selections[estimand]
        if isinstance(selected, tuple):
            raise _EValueRefusal(*selected)
        return selected

    def _evalue_row(self, estimand: str | None = None) -> AssessmentCapability:
        from .sensitivity.evalue import _DERIVED_RR, _EValueRefusal

        status: str | None
        reason: str | None
        execution: Literal["summarize", "retarget"]
        try:
            selected = self._evalue_selection(estimand)
        except _EValueRefusal as error:
            available, status, reason, execution = False, error.status, str(error), "summarize"
        else:
            available, status, reason = True, None, None
            execution = "retarget" if selected.branch == _DERIVED_RR else "summarize"
        return _capability(
            "evalue",
            _family(self._result),
            artifacts=("structured arm contrast", "ratio or derivation artifacts"),
            interpretation="minimum risk-ratio association needed to explain away an effect",
            available=available,
            status=AssessmentStatus.PASSED if status is None else AssessmentStatus(status),
            reason=reason,
            execution=execution,
            cost="cheap",
        )

    def _capability_for_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        if operation == "evalue":
            return self._evalue_row(arguments.get("estimand"))
        return super()._capability_for_arguments(operation, arguments)

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

    def simulated_confounding(self, *args: Any, **kwargs: Any) -> Any:
        """Refit across a simulated common-cause strength grid.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`cleverly.sensitivity.simulated_confounding`, without its
            first argument, which this facade supplies from the fitted result.

        Returns
        -------
        SimulatedConfoundingResult
            Qualitative estimate movements on the scale of the surface, and retained cell
            failures.

        See Also
        --------
        cleverly.sensitivity.simulated_confounding : The same surface as a free function.
        """
        return self._dispatch("simulated_confounding", args, kwargs)

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
        explicit_evalue = operation == "evalue" and (bool(args) or "estimand" in kwargs)
        if not explicit_evalue:
            self._require(operation)
        return self._invoke(operation, args, kwargs)

    def _with_default_parameter(
        self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        """Supply the estimand only when the fit leaves no choice about which one.

        These analyses default to ``"ate"``, which a multi-arm fit never reports under that
        bare name -- it reports ``"ate[medium vs low]"``.  Filling the gap is worth doing
        when exactly one reported parameter is one the analysis applies to, and is a
        scientific choice made on the caller's behalf as soon as there are two: picking the
        first would answer about ``ey1`` on an ``ey1``/``ey0`` fit, silently returning a
        statement about a counterfactual mean to someone who asked about an effect.

        ``simulated_confounding`` also answers for ratio and population attributable
        contrasts. It consults its own eligible set first, including ordinary-TMLE ATT
        and ATC targets. It falls back to the linear set when its eligible set does not name
        exactly one parameter. Unsupported variants then receive the selected alias and
        explain their source boundary.

        When the choice stays ambiguous this returns the arguments untouched and the
        analysis refuses for itself.
        :func:`~cleverly.sensitivity.omitted_variable.resolve_parameter` and
        :func:`~cleverly.sensitivity.missingness.missingness_tilt` both name every estimand
        they could have answered for.
        :func:`~cleverly.sensitivity.simulated_confounding.simulated_confounding` refuses on
        its own ``"ate"`` default instead. Its binary selection message lists only the
        arm, fixed-regime, and identity-MSM aliases the stored estimator can replay.
        Every alias that estimator's own
        boundary refuses is dropped, not natural-course means alone. Its continuous
        selection message omits a zero-delta policy mean, which is the natural course and
        which the surface cannot assess.

        Parameters
        ----------
        operation : str
            Routed sensitivity operation, which selects the eligible parameter set.
        args : tuple
            Positional arguments the caller gave the facade method.
        kwargs : dict
            Keyword arguments the caller gave the facade method.

        Returns
        -------
        tuple
            The original positional arguments, or a one-element tuple naming the estimand.
        """
        if args or "estimand" in kwargs or "ate" in self._result.estimates:
            return args
        if operation == "simulated_confounding":
            from .sensitivity._simulated_confounding_request import (
                _eligible_binary_parameter_names,
            )

            binary_candidates = _eligible_binary_parameter_names(self._result)
            if len(binary_candidates) == 1:
                return (binary_candidates[0],)
        from .sensitivity._parameters import arm_parameters

        known = arm_parameters(self._result)
        candidates = [name for name in self._result.estimates if name in known]
        return (candidates[0],) if len(candidates) == 1 else args

    def run_all(
        self,
        *,
        include_refits: bool = False,
        include_retargets: bool = False,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        random_state: int | None = None,
    ) -> DiagnosticReport:
        """Run available sensitivity analyses that need no new arguments.

        Parameters
        ----------
        include_refits : bool
            Include operations that refit nuisance models.
        include_retargets : bool
            Include moderate retargets; cheap E-value retargets run by default.
        arguments : mapping or None
            Per-operation keyword arguments.
        random_state : int or None
            Common seed for stochastic refit operations.

        Returns
        -------
        DiagnosticReport
            One item for every declared sensitivity operation.
        """
        return self._run_all(
            include_refits=include_refits,
            include_retargets=include_retargets,
            supplied=self._validated_arguments(arguments, random_state),
            random_state=random_state,
        )
