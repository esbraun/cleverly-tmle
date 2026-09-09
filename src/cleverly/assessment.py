"""Capability-aware post-fit diagnostics, validation, and replay metadata.

Assessment lives on a fitted result because its questions depend on the artifacts the
method actually produced.  The facades in this module do not infer support from a result
class and hope for the best: every public operation has a declaration for every public
scalar result family, including deliberate refusals.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Container, Mapping, Sequence
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
from .data.weighting import REPORTED_DRAW, format_score_load
from .exceptions import CapabilityError
from .utils.frames import emit_frame
from .utils.memos import without_memos
from .utils.text import format_draw, format_table
from .validation.drtmle import IDENTITY_TOLERANCE
from .validation.longitudinal import (
    LONGITUDINAL_CENSORING_NOT_FITTED,
    LONGITUDINAL_MECHANISM_PREDICTIONS_MISSING,
    STITCHED_SCORE_Z_TOLERANCE,
    LongitudinalDiagnostics,
    LongitudinalNuisanceDiagnostics,
    LongitudinalNuisanceOmission,
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
    "LONGITUDINAL_CENSORING_NOT_FITTED",
    "LONGITUDINAL_MECHANISM_PREDICTIONS_MISSING",
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
    "LongitudinalNuisanceOmission",
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

    ``DEFERRED`` means that the operation can run after the caller supplies a
    required choice or cost opt-in. ``NOT_APPLICABLE`` means that the operation
    does not apply to the fitted estimand. ``UNAVAILABLE`` means that the operation
    applies, but this fit cannot run it. Four causes reach that status: a missing
    method, a missing derivation, a missing replay artifact, and a requested variant
    that has no support. An operation that raises a capability refusal after the
    combined report invoked it is unavailable for the same reason.

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
    DEFERRED = "deferred"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


# How a combined report presents one status.  The three sets below partition
# ``AssessmentStatus``, and ``tests/unit/test_assessment_contract.py`` asserts that
# partition, so a new member lands in exactly one of them rather than in none of them.
# The three groupings were spelled inline at three call sites with nothing tying them
# together, and adding ``DEFERRED`` reached two of the three by hand.
#
# Presentation policy belongs beside the enum rather than on it: which bucket a status
# falls into is a decision ``AssessmentReport`` makes, not a property of the status.  A
# member on a ``StrEnum``, beside a class attribute of the same shape, also reads as one
# more status at every call site.

#: A row the reader has to act on.  Both statuses come from a check that ran and reached a
#: verdict, which is why ``WARNING`` is here and no omission is.
_ATTENTION: frozenset[AssessmentStatus] = frozenset(
    {AssessmentStatus.FAILED, AssessmentStatus.WARNING}
)

#: A row that produced no verdict, whether the caller, the fitted estimand, or the fit's
#: artifacts are the reason.  These are omissions the report keeps visible.
_OMISSIONS: frozenset[AssessmentStatus] = frozenset(
    {
        AssessmentStatus.DEFERRED,
        AssessmentStatus.NOT_APPLICABLE,
        AssessmentStatus.UNAVAILABLE,
    }
)

#: A row that ran and asks nothing of the reader.  ``COMPLETED`` is a descriptive analysis
#: with no pass criterion, and ``PASSED`` is a check whose condition holds.
_SETTLED: frozenset[AssessmentStatus] = frozenset(
    {AssessmentStatus.PASSED, AssessmentStatus.COMPLETED}
)

#: What stops :attr:`ValidationReport.passed`.  This one cuts across the three above rather
#: than refining one of them: a ``WARNING`` is actionable and still passes, a
#: ``NOT_APPLICABLE`` is an omission and still passes, and a ``DEFERRED`` required check is
#: an unanswered question rather than evidence that validation succeeded.
_BLOCKING: frozenset[AssessmentStatus] = frozenset(
    {
        AssessmentStatus.FAILED,
        AssessmentStatus.DEFERRED,
        AssessmentStatus.UNAVAILABLE,
    }
)


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
        Status to report when the operation cannot run. ``DEFERRED`` means that the caller
        lifts the refusal by passing every name in ``requires_arguments``.
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
    include_in_combined : bool
        Whether :meth:`DiagnosticsFacade.run_all` includes the capability as a report row.
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
    #: Whether the combined report presents this operation. Compatibility aliases remain
    #: explicit capabilities even when their canonical operation is the only combined row.
    include_in_combined: bool = True


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
    include_in_combined: bool = True,
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
        include_in_combined=include_in_combined,
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
        artifacts=(
            "out-of-fold nuisance predictions",
            "method selection state",
            "repeat-specific point estimates",
            "reported standard errors",
        ),
        interpretation=(
            "held-out nuisance fit, collaborative selection, and descriptive split spread"
        ),
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
        artifacts=(
            "observed-law treatment predictions",
            "observed-law censoring predictions",
            "node pseudo-outcomes",
            "initial node predictions",
            "nuisance learner diagnostics",
        ),
        interpretation=(
            "weighted treatment, censoring, outcome, and pseudo-outcome fit by node and "
            "fitted recursion"
        ),
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
        include_in_combined=False,
    ),
)


@dataclass(frozen=True)
class SensitivityRoute:
    """Where one sensitivity operation is implemented, and how its estimand is supplied.

    ``needs_estimand`` is a property of the target's *signature*, not of the operation's
    name: four omitted-variable analyses and :func:`tipping_gamma` all take ``estimand``
    as their second positional argument, so the facade can fill that position in for a fit
    that reports no bare ``"ate"``.  ``benchmark`` and ``missingness`` take ``covariates``
    and ``gamma`` there, and ``evalue`` selects for itself from a ``None`` sentinel --
    injecting a name into any of those positions would silently pass it as something else.

    It does not say whether the operation takes an estimand at all.  ``benchmark`` takes
    one as a keyword-only argument with the same ambiguous ``"ate"`` default, and the
    facade fills that in by keyword.  :func:`_defaults_to_ambiguous_estimand` answers that
    second question from the signature, and it is the one the deferral reads.
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
    rows = tuple(
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
    if _method(result) == "drtmle" and getattr(result, "solved_corrections", False):
        # A guarded DR-TMLE truncation curve is not a retarget.  `truncation_curve` calls
        # `estimator.retarget` once per bound, that reaches `_solve_reduction`, and
        # `DRTMLE._reduction` hands the alternation a closure that refits the reduced
        # regressions.  The ordinary closure refits them at the *fitted* reduced bounds;
        # only the missing-outcome construction receives the swept ones, because they
        # define two of its regression targets.  Either way a bound costs a refit.
        #
        # Declared here rather than as a second registry row because a family may declare
        # each operation once, and the cost is a fact about the fitted result rather than
        # about the family.  `reason` stays unset: it is documented, and contract-tested,
        # as the explanation an *unavailable* row owes its caller, and this row runs.
        rows = tuple(
            replace(
                row,
                execution="refit",
                cost="expensive",
                requires_replay="refit_nuisances",
            )
            if row.operation == "truncation_curve" and row.available
            else row
            for row in rows
        )
    return rows


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
        The invocation this row describes, and empty when there is none. A row whose
        operation ran, and a row whose operation refused after it was invoked, carry the
        effective arguments, which include the signature defaults and the resolved seed.
        A deferred row carries the request as supplied, plus the combined run's seed when
        the operation accepts one. It binds no signature default, because the default is
        what such a row refuses: an ambiguous ``estimand="ate"`` recorded as effective
        would name an argument the operation declines. A row this fit refuses outright
        carries nothing, because no request was considered and none can make it run.
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
    A skipped or refused item remains in the report. Caller-deferred work has its own
    status. A capability known to be unsupported is an omission. A refusal raised during
    an aggregate run remains unavailable, and later operations still run.

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
        return all(item.status not in _BLOCKING for item in self.items)

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
        return tuple(item for _, item in self._presented() if item.status in _ATTENTION)

    @property
    def omissions(self) -> tuple[AssessmentItem, ...]:
        """Return rows that were deferred, not applicable, or unavailable."""
        return tuple(item for _, item in self._presented() if item.status in _OMISSIONS)

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


#: The estimand default that a fit reporting no bare ``"ate"`` leaves ambiguous.
_AMBIGUOUS_ESTIMAND = "ate"


def _defaults_to_ambiguous_estimand(function: Callable[..., Any]) -> bool:
    """Whether calling ``function`` without an estimand asks for the bare ``"ate"``.

    Read from the signature rather than from a list of operation names, so the next
    operation that ships this default is gated the day it is written.  It is a different
    question from :attr:`SensitivityRoute.needs_estimand`, which says *where* the estimand
    goes: ``benchmark`` takes ``covariates`` positionally and ``estimand="ate"`` by
    keyword, so the positional flag is ``False`` while the default is as ambiguous as any
    other.  Gating on that flag left ``benchmark`` reporting ``unavailable`` beside six
    rows that reported ``deferred`` for the identical missing choice.

    Parameters
    ----------
    function : callable
        The routed implementation an operation dispatches to.

    Returns
    -------
    bool
        True when the target declares an ``estimand`` parameter defaulting to ``"ate"``.
    """
    parameter = inspect.signature(function).parameters.get("estimand")
    return parameter is not None and parameter.default == _AMBIGUOUS_ESTIMAND


def _default_estimand_candidates(result: Any, eligible: Container[str]) -> tuple[str, ...]:
    """Reported parameters an operation defaulting to ``"ate"`` is left to choose between.

    Eight operations across the two facades declare ``estimand="ate"`` and answer for one
    parameter: the five omitted-variable analyses, ``tipping_gamma``,
    ``simulated_confounding`` and ``refute``.  A fit that reports a bare ``"ate"`` settles
    the choice, and this returns nothing.  A fit that reports none has to be asked which
    one, and the answer separates three cases a combined report has to tell apart:

    ============ =============================================================
    length       what the caller is owed
    ============ =============================================================
    ``0``        nothing the operation applies to, so no argument makes it run
    ``1``        one facade fills this name in, another defers on it
    ``2`` upward the ambiguity only the caller can settle, which is a deferral
    ============ =============================================================

    Length one is therefore not one answer.  ``_CapabilityFacade._substitutes_estimand``
    says which facade this fit is asking, because a facade that supplies nothing leaves
    the operation to run on its ambiguous default and refuse.

    The length-zero and the length-two answers are the two the report used to give one
    status.  A multi-arm fit reports ``ate[high vs low]`` and ``ate[medium vs low]``, and
    the bound refuses the bare default with "estimand 'ate' was not requested in this
    fit", which is also what an incremental fit hears when the bound applies to nothing it
    reports.  The first is a choice; the second is a missing derivation.

    Parameters
    ----------
    result : Any
        Fitted result whose reported parameters settle the choice.
    eligible : container of str
        Names the operation can answer for. Reported parameters outside it are not
        choices, so they are not counted towards the ambiguity.

    Returns
    -------
    tuple of str
        The eligible reported names, in the order the fit reports them.
    """
    if "ate" in result.estimates:
        return ()
    return tuple(name for name in result.estimates if name in eligible)


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
    #: Whether this facade supplies the sole eligible estimand on the caller's behalf.
    #: It settles what one candidate means. A facade that substitutes runs the row under
    #: that name and defers only at two, and a facade that substitutes nothing has to
    #: defer at one: nothing else fills the gap, so the operation would be invoked on its
    #: ambiguous default and refuse. ``refute`` did exactly that and reported
    #: ``unavailable`` on a fit where naming its one reported alias runs it.
    _substitutes_estimand: bool = False

    def __init__(self, result: Any) -> None:
        self._result = result

    def __getstate__(self) -> dict[str, Any]:
        """Persist the facade without the verdicts it memoized.

        A result carries its facades in ``__dict__``, and ``save`` pickles the whole result,
        so every :func:`~functools.cached_property` on a facade the caller has touched is
        written into the artifact.  Those memos are derived state: ``_declared``,
        ``_capability_map`` and ``_evalue_selections`` all restate what this version of the
        package concludes from the stored artifacts.  Persisting one pins the conclusion of
        the version that saved it.  A multi-arm result saved before ``DEFERRED`` existed
        carried ``{None: ("unavailable", ...)}``, and the loaded result kept reporting an
        unavailable E-value with no next step, past the cache generation that exists to
        force exactly that recompute.

        Dropping every memo rather than the three by name, because the next one added is
        stale in an artifact the moment it is written, and recomputing costs one pass over
        the declarations.

        Returns
        -------
        dict of str to Any
            The instance state, without the entries a ``cached_property`` owns.
        """
        return without_memos(type(self), self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a facade, and discard any memo the artifact already carries.

        :meth:`__getstate__` keeps a memo out of every artifact this version writes. It
        cannot reach one an older version wrote, and that artifact is the migration case:
        the stale verdict is inside the file. Filtering on the way in heals it.

        Parameters
        ----------
        state : dict of str to Any
            The pickled instance state.
        """
        self.__dict__.update(without_memos(type(self), state))

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

    def _estimand_candidates(self, operation: str) -> tuple[str, ...]:
        """Reported parameters this operation is left to choose between, or ``()``.

        The one place either facade decides whether a fit leaves an operation's estimand
        ambiguous.  :meth:`_estimand_gated` reads it to defer the row, and
        :meth:`SensitivityFacade._with_default_parameter` reads it to substitute the sole
        eligible name.  Written twice, the two disagreed: the substitution declined to
        guess between two contrasts and the row still advertised itself as runnable, so a
        combined report invoked the operation and published the refusal as ``unavailable``
        with a next step that named no argument.

        An operation that takes no ``estimand``, or whose default this fit settles,
        returns ``()`` and is never gated.  :func:`_default_estimand_candidates` states
        what each length means.

        Parameters
        ----------
        operation : str
            Declared operation on this facade.

        Returns
        -------
        tuple of str
            Eligible reported names, empty for an operation that faces no choice.
        """
        return ()

    def _estimand_gated(
        self, capability: AssessmentCapability, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        """Defer a row whose ``estimand`` default this fit leaves ambiguous.

        The deferral is resolved per request, exactly as the E-value's is.  A caller who
        names a supported estimand gets the row back untouched and the operation runs, so
        the row cannot stay deferred once the argument that lifts it is supplied. That
        matters because :meth:`_skipped` defers unconditionally on the status.

        Only the ambiguity moves.  An operation the fit refuses outright keeps its own
        ``unavailable`` answer, because no argument makes a missing derivation, a missing
        replay artifact or an unsupported contrast run.

        Parameters
        ----------
        capability : AssessmentCapability
            The declared row, already gated by method and by replayability.
        arguments : mapping of str to Any
            Arguments the caller supplied for this operation.

        Returns
        -------
        AssessmentCapability
            The row, deferred on the estimand when this request leaves it ambiguous.
        """
        if not capability.available:
            # An availability refusal outranks a choice: the caller cannot name their way
            # past a derivation this family does not have.
            return capability
        if "estimand" in capability.requires_arguments:
            # ``simulated_confounding`` already declares the argument, so the
            # missing-argument gate reports it and names ``grid`` beside it.
            return capability
        if arguments.get("estimand") is not None:
            # Read the value, not the key. ``estimand=None`` is the documented public
            # default, so that spelling has to defer exactly as the bare request does.
            return capability
        candidates = self._estimand_candidates(capability.operation)
        if len(candidates) < (2 if self._substitutes_estimand else 1):
            # One candidate is a deferral or a substitution, and which one it is belongs
            # to the facade rather than to the count. See ``_substitutes_estimand``.
            return capability
        return replace(
            capability,
            available=False,
            status=AssessmentStatus.DEFERRED,
            reason=(
                f"{capability.operation} answers for one estimand; choose an explicit "
                f"estimand from {list(candidates)}"
            ),
            requires_arguments=(*capability.requires_arguments, "estimand"),
        )

    def _capability_for_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        """Resolve request-specific availability and cost before aggregate execution."""
        return self._estimand_gated(self.capability(operation), arguments)

    def _skipped(
        self,
        capability: AssessmentCapability,
        arguments: Mapping[str, Any],
        *,
        include_refits: bool,
        include_retargets: bool,
    ) -> AssessmentItem | None:
        """The row a combined report owes an operation, or ``None`` to run it.

        A deferred row first, then availability, then the missing argument, and the cost
        last. Every gate above the cost gate refuses for a reason no flag can pay off, and a
        report that named the cost first sent the caller to ``include_refits=True`` and
        then, on the very next call, to the argument it never mentioned. A refusal has to
        name the first thing that is wrong.

        Each gate decides the sentence alone, and this method attaches the request once on
        the way out. Two of the three branches used to copy ``arguments`` themselves and
        the third dropped them, so a combined report published two omissions that named the
        caller's request and one that did not.

        The availability branch is the deliberate exception, and the only one. A row
        refused fit-wide describes no invocation, so it carries no arguments to replay.

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
        item: AssessmentItem | None
        fit_wide = False
        missing = tuple(name for name in capability.requires_arguments if name not in arguments)
        if capability.status is AssessmentStatus.DEFERRED and capability.requires_arguments:
            # A deferred row is refused by this request, not by the fit: the operation runs
            # once the caller names the argument the row declares.  Routed by status rather
            # than by key membership, because the E-value defers on the *value* -- an
            # explicit ``estimand=None`` is the documented public default and stays
            # ambiguous.  Testing membership alone let that spelling past both gates, into
            # the invocation, and out through the generic refusal handler as ``unavailable``
            # with no argument to act on, while the bare call reported ``deferred``.
            item = _missing_argument_item(
                capability,
                self._attribute,
                capability.requires_arguments,
                reason=capability.reason,
            )
        elif not capability.available:
            # The one branch that describes no invocation. This row is refused fit-wide,
            # before any request was considered: the operation did not run, and no
            # argument makes it run. Stamping the caller's request onto it, seed included,
            # would report "no longitudinal benchmarking derivation is registered" beside
            # a ``random_state`` that nothing ever drew from. The asymmetry with the three
            # branches below is the point: each of those answers a request the caller
            # made, and carries it.
            fit_wide = True
            item = _item_from_capability(capability)
        elif missing:
            item = _missing_argument_item(capability, self._attribute, missing)
        else:
            item = _cost_refusal(capability, self._attribute, include_refits, include_retargets)
        if item is None or fit_wide:
            return item
        return replace(item, arguments=dict(arguments))

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
                if not declared.include_in_combined:
                    continue
                operation_arguments = dict(supplied.get(declared.operation, {}))
                capability = self._capability_for_arguments(declared.operation, operation_arguments)
                if capability.accepts_random_state and random_state is not None:
                    # Before the gates, not after them. The seed is part of the request for
                    # every row that accepts one, and injecting it inside the invocation
                    # branch left a deferred ``refute`` row reporting ``arguments == {}``
                    # after ``run_all(random_state=7)``. That row is not deterministic, so a
                    # caller replaying it from ``item.arguments`` reproduced a different
                    # draw from the one the report would have taken.
                    operation_arguments["random_state"] = random_state
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
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
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
        if resolve and self._attribute == "sensitivity":
            # Every routed operation, not the positional ones alone: the substitution
            # decides for itself whether this target takes an estimand and where it goes.
            args, kwargs = self._with_default_parameter(operation, args, dict(kwargs))
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
    capability: AssessmentCapability,
    attribute: str,
    missing: tuple[str, ...],
    *,
    reason: str | None = None,
) -> AssessmentItem:
    """The skip a combined report owes an operation whose argument it cannot choose.

    A combined report cannot run an operation until every required argument is present.
    Choosing a value here -- which covariates to benchmark against -- would be a scientific
    choice made silently on the caller's behalf.

    A row may declare more than one argument, so the sentence has to agree in number.
    ``", ".join`` alone rendered "an explicit grid, estimand argument", which reads as one
    argument named "grid, estimand".

    ``reason`` is the sentence *this* deferral prints, and the gate passes it in rather than
    this builder reading ``capability.reason``, which means something else: the explanation
    a refused row owes its caller.  The two coincide on the one row that carries both, and
    the printed sentence must not depend on that coincidence.  ``None`` keeps the generic
    sentence, which is what a row with a declared but unsupplied argument -- ``benchmark``
    without ``covariates`` -- has to say.
    """
    needed = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} and {missing[-1]}"
    phrase = (
        f"an explicit {needed} argument" if len(missing) == 1 else f"explicit {needed} arguments"
    )
    return AssessmentItem(
        capability.operation,
        AssessmentStatus.DEFERRED,
        reason or f"needs {phrase}, which a combined report has no basis to choose",
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

    The class is read off the row rather than off the operation name, because a guarded
    DR-TMLE fit resolves ``truncation_curve`` into the ``refit`` class -- its targeting
    alternation refits the reduced regressions at every bound.  See
    :func:`assessment_capabilities`.
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
        AssessmentStatus.DEFERRED,
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

    def _estimand_candidates(self, operation: str) -> tuple[str, ...]:
        """The reported aliases ``refute`` may be asked to choose between.

        ``refute`` refits under one alias and refuses every name the fit did not report,
        so its eligible set is the reported set itself.  Which diagnostics face the choice
        is read from the routed signature, exactly as the sensitivity side reads it:
        ``refute`` is the one diagnostic today whose ``estimand`` defaults to the bare
        ``"ate"``, and a second one is gated the day it is written rather than the day
        somebody remembers this method.  :data:`SENSITIVITY_ROUTES` answers a different
        question, which is where in :mod:`cleverly.sensitivity` an analysis is
        implemented, and this facade routes to :func:`cleverly.validation.refute`.

        No substitution follows from a single candidate here, which is why
        ``_substitutes_estimand`` stays false and one candidate defers the row.  The
        sensitivity facade may fill in a sole eligible parameter because its analyses
        answer for whichever contrast they are given and it has always done so;
        ``refute`` refits, and refitting under a name the caller never wrote is a
        scientific choice made on their behalf.

        Parameters
        ----------
        operation : str
            Declared diagnostic operation.

        Returns
        -------
        tuple of str
            Reported aliases for ``refute``, and ``()`` for every other diagnostic.
        """
        function, _ = self._routed_callable(operation)
        if not _defaults_to_ambiguous_estimand(function):
            return ()
        return _default_estimand_candidates(self._result, self._result.estimates)

    def stagewise(self) -> LongitudinalDiagnostics:
        """Return support and targeting diagnostics by longitudinal stage.

        This is a compatibility alias. :meth:`support` is the canonical name for the same
        report, and the combined report presents it under that name alone. The alias keeps
        its own capability, so it still refuses a point-treatment fit by name.

        Returns
        -------
        LongitudinalDiagnostics
            Diagnostics for each regimen and time point.

        Raises
        ------
        CapabilityError
            If the fitted result is not longitudinal.

        See Also
        --------
        support : The canonical name for this report.
        """
        self._require("stagewise")
        return self.support()

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
            return _cached(
                self._result,
                "diagnostics.support",
                (),
                {},
                lambda: _longitudinal_stagewise(self._result),
            )

        def compute() -> Any:
            nuisance = self._result.nuisance
            from .interventions import (
                check_incremental_support,
                check_shift_support,
                check_support,
            )
            from .sensitivity.positivity import positivity_report

            def score_arguments(group: str) -> dict[str, Any]:
                """The fitted score artifact for one group, as ``check_*`` keywords.

                A group with no fluctuation at all reports no equations, which is a
                different state from a fluctuation whose artifact predates the retained
                weights, and the two get different reasons in the report.
                """
                fluctuation = self._result.fluctuations.get(group)
                return {
                    "absolute_score_weights": getattr(fluctuation, "absolute_score_weights", None),
                    "equations": () if fluctuation is None else tuple(fluctuation.names),
                    "n_repeats": self._result.n_repeats,
                }

            if nuisance.regimes is not None:
                return check_support(
                    nuisance.regimes,
                    self._result.data.treatment,
                    nuisance.propensity.values,
                    backend=self._result.data.backend,
                    **score_arguments("regime"),
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
                    **score_arguments("mtp"),
                )
            if nuisance.incremental is not None:
                return check_incremental_support(
                    nuisance.incremental,
                    self._result.data.treatment,
                    **score_arguments("ipsi"),
                )
            return positivity_report(self._result)

        return _cached(self._result, "diagnostics.support", (), {}, compute)

    def nuisance_models(self) -> Any:
        """Return method-aware held-out nuisance and split diagnostics.

        Returns
        -------
        NuisanceDiagnostics or LongitudinalNuisanceDiagnostics
            Held-out diagnostics for point or sequential nuisance models. Point-treatment
            reports also retain C-TMLE selection state and pair repeated-split spread with
            each reported standard error when those artifacts exist.
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
            Lower mechanism bounds to evaluate. Treatment-mechanism values are symmetric
            shorthand for ``(bound, 1 - bound)``; observation- and intermediate-mechanism
            values use ``(bound, 1)``. Default bounds, including each requested parameter's
            exact fitted pair, are used when omitted. An explicit sequence is not expanded.
        estimands : sequence of str or None
            Reported estimands to include. All compatible estimands are the default.
        mechanism : bool
            For an incremental intervention, vary a separate observation mechanism.

        Returns
        -------
        dataframe
            Estimates and uncertainty at each requested bound, with the evaluated and
            parameter-specific fitted pairs and movement from the fitted estimate.

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
            One item for every diagnostic operation included in the combined report.

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


def _intervention_rows(report: Any) -> tuple[tuple[str, Any], ...]:
    """Every per-intervention record of a support report, whichever shape it arrived in.

    A shift or incremental report *is* the mapping of its rows; a regime report keeps them
    under ``regimes``. Both spellings reach the same records, and two readers that
    enumerate them separately can come to disagree about which rows exist.
    """
    interventions = report.items() if isinstance(report, Mapping) else ()
    regimes = getattr(report, "regimes", {}).items()
    return (*interventions, *regimes)


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
        # Group rows carry two facts that must not be conflated. Their clipping fraction is
        # the action of an estimand-specific propensity bound and belongs in the maximum.
        # Their targeted ratio is concentration of absolute score load, not arm/mechanism
        # effective sample size, so it is deliberately excluded from the pooled minimum.
        for values in getattr(report, "group_leverage", {}).values():
            if "clipped_fraction" in values:
                clipped = float(values["clipped_fraction"])
                if np.isfinite(clipped):
                    truncated.append(clipped)
    if isinstance(report, LongitudinalDiagnostics):
        truncated.extend(float(row.share_truncated) for row in report.rows)
        ess.extend(float(row.effective_n / row.n_followed) for row in report.rows if row.n_followed)
    if isinstance(report, Mapping):
        # Only the shift and incremental rows carry a cap. A regime row has no
        # `capped_fraction`, and defaulting one to zero for it would turn "this report
        # truncates nothing" into a reported maximum of 0.0%.
        truncated.extend(float(getattr(row, "capped_fraction", 0.0)) for row in report.values())
    ess.extend(float(getattr(row, "ess_ratio", np.nan)) for _, row in _intervention_rows(report))
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


def _group_load_fact(report: Any) -> str | None:
    """The most concentrated score equation, separate from mechanism ESS.

    A target-group row has no draw count of its own, so this reads the report's. Reading it
    from the row instead printed "draw 01 of 01" for every repeated fit, which contradicted
    the draw count :meth:`~cleverly.sensitivity.PositivityReport.summary` printed from the
    same report. Copying the count onto each row would have made the two agree by making
    the same number storable in two places, which is what let them disagree.
    """
    group_draw = (REPORTED_DRAW, int(getattr(report, "n_repeats", 1)))
    rows: list[tuple[str, str, Any, tuple[int, int] | None]] = [
        ("group", group, values, group_draw)
        for group, values in getattr(report, "group_leverage", {}).items()
        if np.isfinite(float(values.get("targeted_ratio", np.nan)))
    ]
    # An intervention row records its own draw, so it is read from the row.
    rows.extend(
        ("intervention", str(name), values, None)
        for name, item in _intervention_rows(report)
        if (values := getattr(item, "score_load", None)) is not None
        and np.isfinite(float(values.get("targeted_ratio", np.nan)))
    )
    if not rows:
        return None
    kind, name, values, draw = min(rows, key=lambda item: float(item[2]["targeted_ratio"]))
    rendered = format_score_load(values, style="detail", draw=draw)
    return f"{kind} load: {name}:{values['equation']} {rendered}; not estimator ESS"


def _support_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    warning = _support_warning(report)
    facts = _support_facts(*_support_metrics(report))
    group_fact = _group_load_fact(report)
    if group_fact is not None:
        facts.append(group_fact)
    if warning:
        facts.append(warning)
    detail = "; ".join(facts) if facts else "stored support report completed"
    return AssessmentItem(
        "support",
        # `COMPLETED` rather than `PASSED`, because support has no pass criterion once the
        # effective-sample-size ratio is reported rather than graded.  What this row can
        # still assert is a breach: truncation above the fitted bound's tolerance, or an
        # intervention with estimated zero support.  Absent one, the honest statement is
        # that the check ran and produced numbers, and `PASSED` would read as a positivity
        # clearance that no threshold here is entitled to give.  `_nuisance_item` returns
        # `COMPLETED` for the same reason.
        AssessmentStatus.WARNING if warning else AssessmentStatus.COMPLETED,
        detail,
        () if warning is None else ("inspect result.diagnostics.support()",),
    )


def _nuisance_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    findings = tuple(getattr(report, "findings", ()))
    if isinstance(report, LongitudinalNuisanceDiagnostics):
        finite = [row.reported_loss for row in report.rows if np.isfinite(row.reported_loss)]
        if not finite:
            return AssessmentItem(
                "nuisance_models",
                AssessmentStatus.WARNING,
                "no finite longitudinal nuisance loss is available",
                ("inspect result.diagnostics.nuisance_models()",),
            )
        detail = f"{len(finite)} longitudinal nuisance loss value(s) are available"
        if report.omissions:
            detail += f"; {len(report.omissions)} role omission(s) are recorded"
    else:
        facts = [
            "; ".join(findings)
            if findings
            else f"{len(getattr(report, 'models', ()))} nuisance model report(s) are available"
        ]
        n_repeats = int(getattr(report, "n_repeats", 1))
        selection = getattr(report, "selection", None)
        if selection is not None:
            # The artifact says what it is. Discriminating a selector from an
            # outcome-adaptive fit here as well is what let this line and the one in
            # `NuisanceDiagnostics.summary` drift into two spellings of one fact.
            reported = int(getattr(report, "reported_repeat", REPORTED_DRAW))
            draw = f" on {format_draw(reported, n_repeats)}" if n_repeats > 1 else ""
            facts.append(f"{selection.describe()}{draw}")
        elif getattr(report, "selection_omission", None) is not None:
            facts.append(f"C-TMLE selection unavailable: {report.selection_omission}")
        spread = tuple(getattr(report, "repeat_spread", ()))
        if spread:
            finite_rows: list[Any] = [
                row for row in spread if np.isfinite(row.ratio_to_standard_error)
            ]
            if finite_rows:
                largest_row = max(finite_rows, key=lambda row: row.ratio_to_standard_error)
                facts.append(
                    f"split spread for {len(spread)} parameter(s) across "
                    f"{n_repeats} draws; largest sd/se "
                    f"{largest_row.ratio_to_standard_error:.3g} for {largest_row.estimand}"
                )
            else:
                facts.append(
                    f"split spread for {len(spread)} parameter(s) across "
                    f"{n_repeats} draws; sd/se unavailable"
                )
        spread_omission = getattr(report, "repeat_spread_omission", None)
        if n_repeats > 1 and spread_omission is not None:
            # Guarded on the draw count for the reason `NuisanceDiagnostics.summary` is:
            # the one-draw reason is the ordinary state and states nothing a reader of an
            # ordinary fit needs.
            facts.append(f"split spread unavailable: {spread_omission}")
        detail = "; ".join(facts)
    return AssessmentItem(
        "nuisance_models",
        AssessmentStatus.WARNING if findings else AssessmentStatus.COMPLETED,
        detail,
        ("inspect result.diagnostics.nuisance_models()",) if findings else (),
    )


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


def _finite(value: Any) -> bool:
    """True when a cell carries a number a range or a baseline can use."""
    return value is not None and bool(np.isfinite(value))


def _range(values: Sequence[Any]) -> tuple[float, float] | None:
    finite = [float(value) for value in values if _finite(value)]
    return (min(finite), max(finite)) if finite else None


def _format_range(values: tuple[float, float] | None) -> str:
    return "no finite values" if values is None else f"[{values[0]:.4g}, {values[1]:.4g}]"


def _alias_rows(aliases: Sequence[Any]) -> dict[str, list[int]]:
    """Row positions of each parameter, keyed by alias in first-seen order."""
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(aliases):
        groups.setdefault(str(value), []).append(index)
    return groups


def _format_alias_ranges(ranges: Mapping[str, tuple[float, float] | None]) -> str:
    """One comma-separated ``alias [low, high]`` list, in the order the mapping holds."""
    return ", ".join(f"{alias} {_format_range(value)}" for alias, value in ranges.items())


def _truncation_item(
    report: Any, _result: Any, _arguments: Mapping[str, Any] = _NO_ARGUMENTS
) -> AssessmentItem:
    payload = _frame_payload(report)
    bounds = _range(payload.get("bound", payload.get("g_bound", ())))
    psi = payload.get("psi", payload.get("estimate", ()))
    aliases = payload.get("estimand")
    if not aliases:
        # ``bound`` holds lower endpoints alone, because ``upper_bound`` is its own
        # column.  Both branches name the same column the same way.
        detail = (
            f"evaluated lower bounds {_format_range(bounds)}; "
            f"estimate range {_format_range(_range(psi))}"
        )
    else:
        groups = _alias_rows(aliases)
        deltas = payload.get("delta_from_fitted")
        markers = payload.get("is_fitted_bound")
        # One signed range for each parameter, against that parameter's own fitted
        # estimate.  The maximum absolute movement is the larger endpoint's magnitude, so
        # reporting the signed range keeps the direction and loses nothing.  The fitted
        # estimate and the fitted pair stay columns of the retained frame, and
        # ``result.psi(name)`` returns the first: this row presents the movement on one
        # scale rather than reprinting the curve.
        scale = f"{len(groups)} parameter(s) over evaluated lower bounds {_format_range(bounds)}"
        if deltas is None:
            movement = _format_alias_ranges(
                {alias: _range([psi[index] for index in rows]) for alias, rows in groups.items()}
            )
            detail = f"{scale}; estimate range: {movement}"
        else:
            movement = _format_alias_ranges(
                {alias: _range([deltas[index] for index in rows]) for alias, rows in groups.items()}
            )
            detail = f"{scale}; signed movement from the fitted estimate: {movement}"
        if markers is not None:
            # An explicit ``bounds=`` grid keeps its cardinality, so it can omit a
            # parameter's fitted pair.  A count says how many curves start away from the
            # estimate the fit reported.  A missing column says nothing either way, so the
            # clause is written only when the markers are there to read.  The deltas are a
            # separate column, so the markers decide this clause on their own.
            omitted = sum(
                not any(bool(markers[index]) for index in rows) for rows in groups.values()
            )
            if omitted:
                detail += f"; fitted pair not evaluated for {omitted} of {len(groups)}"
    return AssessmentItem(
        "truncation_curve",
        AssessmentStatus.COMPLETED,
        detail,
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
    psi = payload["psi"]
    facts = [f"gamma range {_format_range(_range(payload['gamma']))}"]
    aliases = payload.get("estimand")
    if not aliases:
        facts.append(f"estimate range {_format_range(_range(psi))}")
    else:
        # The frame holds one row for each ``(gamma, estimand)`` pair, so a pooled minimum
        # and maximum can belong to two different parameters and describe neither.  The
        # ``gamma == 0`` row is the estimate the fit itself reported, which makes it the
        # baseline the tilted rows move away from.
        groups = _alias_rows(aliases)
        markers = payload.get("is_mar")
        # ``missingness_tilt`` drops an untiltable estimand from the default sweep, so the
        # count is what tells a reader that the curve covers fewer parameters than the fit.
        facts.append(f"{len(groups)} parameter(s)")
        if markers is None:
            # A missing column says nothing about which row the fit itself reported, so
            # the row states the levels it can read and claims no baseline.  Inferring an
            # absent baseline here would deny a retained ``gamma == 0`` row.
            levels = {
                alias: _range([psi[index] for index in rows]) for alias, rows in groups.items()
            }
            facts.append(f"estimate range: {_format_alias_ranges(levels)}")
        else:
            movement: dict[str, tuple[float, float] | None] = {}
            unavailable: dict[str, tuple[float, float] | None] = {}
            for alias, rows in groups.items():
                # A marked row whose estimate is not finite is no more usable as a
                # baseline than an absent one, and every delta taken from it is not
                # finite.  Both cases report the level instead.
                baseline = next(
                    (index for index in rows if bool(markers[index]) and _finite(psi[index])),
                    None,
                )
                if baseline is None:
                    # The default grid contains zero, but an explicit ``gamma=`` grid need
                    # not.  The row then reports the level it has and names what is
                    # missing, rather than treating one tilted estimate as the estimate the
                    # fit made.
                    unavailable[alias] = _range([psi[index] for index in rows])
                else:
                    mar = float(psi[baseline])
                    movement[alias] = _range([psi[index] - mar for index in rows])
            if movement:
                ranges = _format_alias_ranges(movement)
                facts.append(f"signed movement from the MAR estimate: {ranges}")
            if unavailable:
                ranges = _format_alias_ranges(unavailable)
                facts.append(f"no MAR estimate retained; estimate range: {ranges}")
    return AssessmentItem("missingness", AssessmentStatus.COMPLETED, "; ".join(facts))


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
    # No report path reaches this. ``stagewise`` is ``include_in_combined=False`` on the
    # longitudinal family and ``available=False`` on the point one, so the combined loop
    # skips it and the point row renders as a refusal. It stays because ``INTERPRETERS`` and
    # ``ASSESSMENT_CAPABILITIES`` are checked against each other in both directions, and a
    # missing entry would read as a capability nobody can interpret rather than as an alias.
    #
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


#: What the combined row says when the point support report is not in its ``adequate``
#: tier.  The tier itself is decided by
#: :attr:`~cleverly.sensitivity.positivity.PositivityReport.severity`; these are only its
#: words on this surface, and the numbers behind them are already in the detail that
#: ``_support_facts`` builds.
#:
#: Both tiers describe truncation, because truncation is what that report grades.  The
#: effective-sample-size ratio reaches this row as a *fact* through ``_support_facts`` and
#: never as a status, which is why an unbreached row is ``COMPLETED`` rather than ``PASSED``.
_POSITIVITY_SEVERITY_REASON: dict[Literal["serious", "strain"], str] = {
    "serious": "the support report reports that truncation is carrying the estimate",
    "strain": "the support report reports material truncation",
}


def _support_warning(report: Any) -> str | None:
    from .sensitivity.positivity import PositivityReport

    if isinstance(report, PositivityReport):
        # Delegated rather than re-derived.  This branch used to threshold the truncated
        # fraction and the *fitted mechanism* effective sample sizes, and `mechanisms` is
        # empty unless the fit declared `delta=` or `intermediate=`.  So on an ordinary ATE
        # fit it applied one check, and reached a status the retained report contradicted.
        severity = report.severity
        return None if severity == "adequate" else _POSITIVITY_SEVERITY_REASON[severity]
    # Every branch below grades truncation and estimated zero support, and none grades an
    # effective-sample-size ratio.  A Kish ratio has no derived cutoff, so a status keyed
    # on one would present a house convention as a finding and let a reader take `passed`
    # for a positivity clearance.  `_support_facts` reports the minimum ratio on every row
    # instead, which is the number an analyst has to judge.
    if isinstance(report, LongitudinalDiagnostics) and any(
        row.share_truncated > 0.05 for row in report.rows
    ):
        return "cumulative mechanism truncation exceeds 5% at one or more nodes"
    if isinstance(report, Mapping):
        # A shift or IPSI fit reports one support record per declared intervention rather
        # than a single object, so none of the attribute probes above sees it.  These
        # dataclasses carry the same quantities the other branches grade; read them by
        # field so the two classes need no separate branches -- ``IncrementalSupport`` has
        # an ``ess_ratio`` but no ``unsupported`` or ``capped_fraction``, and a missing
        # field must not read as a breach.  The ``ess_ratio`` is reported, never graded.
        for name, item in report.items():
            if int(getattr(item, "unsupported", 0)) > 0:
                return f"intervention {name!r} has units with estimated zero support"
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
    #: This facade fills in the sole eligible estimand, through
    #: :meth:`_with_default_parameter`, so one candidate settles the choice and only two
    #: defer the row.
    _substitutes_estimand = True

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
        # ``LongitudinalData`` declares neither treatment-family flag, so the family guards
        # both reads.  Neither one needs a ``getattr`` default: a default would answer for a
        # point result whose data lack the flag, and no such result exists, because both are
        # properties of ``CausalData``.
        continuous = not longitudinal and bool(self._result.data.is_continuous_treatment)
        from .sensitivity._simulated_confounding_request import (
            _eligible_binary_parameter_names,
            _fit_wide_refusal,
        )

        simulated_refusal = _fit_wide_refusal(self._result)
        if longitudinal or continuous:
            needs_estimand = False
        elif not self._result.data.is_binary_treatment:
            # A multi-arm fit reports no bare ``ate`` that this surface could assess, so the
            # truthful argument list names an estimand.  The row is unavailable, and its
            # declared arguments still describe the call a caller would have to write.
            needs_estimand = True
        else:
            binary_parameters = _eligible_binary_parameter_names(self._result)
            needs_estimand = "ate" not in binary_parameters and len(binary_parameters) > 1
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
                available=simulated_refusal is None,
                status=(
                    AssessmentStatus.PASSED
                    if simulated_refusal is None
                    else AssessmentStatus.UNAVAILABLE
                ),
                reason=simulated_refusal,
                requires_arguments=("grid", "estimand")
                if continuous or needs_estimand
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

        status: AssessmentStatus
        reason: str | None
        execution: Literal["summarize", "retarget"]
        requires_arguments: tuple[str, ...]
        try:
            selected = self._evalue_selection(estimand)
        except _EValueRefusal as error:
            # A refused row is refused, whichever status it carries.  Reporting
            # ``available=True`` beside a deferral published ``available: True | status:
            # passed | reason: an E-value needs one contrast`` on a multi-arm fit, and the
            # bare call the row invited then raised.  ``_dispatch`` skips ``_require`` for
            # an explicit ``estimand``, so a caller who supplies one still runs.
            available = False
            status = error.status
            reason = str(error)
            execution = "summarize"
            # The argument that lifts the deferral. Only a deferral has one: an
            # unsupported contrast is not a choice the caller can make.
            requires_arguments = ("estimand",) if status is AssessmentStatus.DEFERRED else ()
        else:
            available, status, reason = True, AssessmentStatus.PASSED, None
            execution = "retarget" if selected.branch == _DERIVED_RR else "summarize"
            requires_arguments = ()
        return _capability(
            "evalue",
            _family(self._result),
            artifacts=("structured arm contrast", "ratio or derivation artifacts"),
            interpretation="minimum risk-ratio association needed to explain away an effect",
            available=available,
            status=status,
            reason=reason,
            execution=execution,
            cost="cheap",
            requires_arguments=requires_arguments,
        )

    def _capability_for_arguments(
        self, operation: str, arguments: Mapping[str, Any]
    ) -> AssessmentCapability:
        if operation == "evalue":
            # The E-value selects for itself from a ``None`` sentinel rather than through
            # ``SENSITIVITY_ROUTES``, so its row is rebuilt for the requested estimand.
            return self._evalue_row(arguments.get("estimand"))
        return super()._capability_for_arguments(operation, arguments)

    def _estimand_candidates(self, operation: str) -> tuple[str, ...]:
        """The reported parameters a routed sensitivity analysis may be asked to choose.

        The gate is the target's signature: a route whose ``estimand`` defaults to the
        bare ``"ate"`` faces the choice, and ``evalue``, which defaults to ``None``, and
        ``missingness``, which takes no estimand, do not.  It is not
        ``needs_estimand``, which answers where the estimand goes rather than whether
        there is one.  ``benchmark`` takes ``covariates`` positionally and the same
        ambiguous default by keyword, and reading the positional flag left its row alone.

        ``simulated_confounding`` answers for ratio and population attributable contrasts
        too, so it consults its own eligible set first and falls back to the linear set
        when that set names no single parameter. The rest are functionals of one linear
        contrast, and their eligible set is the arm-indexed one.

        Parameters
        ----------
        operation : str
            Declared sensitivity operation.

        Returns
        -------
        tuple of str
            Eligible reported names, empty when the route settles its own estimand.
        """
        function, _ = self._routed_callable(operation)
        if not _defaults_to_ambiguous_estimand(function):
            return ()
        if operation == "simulated_confounding":
            from .sensitivity._simulated_confounding_request import (
                _eligible_binary_parameter_names,
            )

            binary = _default_estimand_candidates(
                self._result, _eligible_binary_parameter_names(self._result)
            )
            if len(binary) == 1:
                return binary
        from .sensitivity._parameters import arm_parameters

        return _default_estimand_candidates(self._result, arm_parameters(self._result))

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
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Supply the estimand only when the fit leaves no choice about which one.

        These analyses default to ``"ate"``, which a multi-arm fit never reports under that
        bare name -- it reports ``"ate[medium vs low]"``.  Filling the gap is worth doing
        when exactly one reported parameter is one the analysis applies to, and is a
        scientific choice made on the caller's behalf as soon as there are two: picking the
        first would answer about ``ey1`` on an ``ey1``/``ey0`` fit, silently returning a
        statement about a counterfactual mean to someone who asked about an effect.

        Which parameters are eligible is :meth:`_estimand_candidates`, and a combined
        report reads the same method to decide that the row is deferred rather than
        unavailable.  Both answers have to come from one predicate: while they did not,
        this method declined to guess between two contrasts and the row beside it still
        said the analysis was runnable.

        A direct call still hears the analysis refuse for itself when the choice stays
        ambiguous, and that refusal names every parameter it could have answered for.
        :func:`~cleverly.sensitivity.omitted_variable.resolve_parameter` and
        :func:`~cleverly.sensitivity.missingness.missingness_tilt` write those lists.
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
            The positional and keyword arguments, one of them naming the estimand when
            this fit leaves exactly one eligible parameter.
        """
        positional = SENSITIVITY_ROUTES[operation].needs_estimand
        # Where the name goes is ``needs_estimand``; whether there is one to supply is the
        # signature, which :meth:`_estimand_candidates` reads. ``benchmark`` takes its
        # estimand by keyword, so a positional substitution would pass a parameter name
        # as ``covariates``, and a positional argument the caller wrote there is not an
        # estimand and does not settle the choice.
        if "estimand" in kwargs or (positional and args):
            return args, kwargs
        candidates = self._estimand_candidates(operation)
        if len(candidates) != 1:
            return args, kwargs
        if positional:
            return (candidates[0], *args), kwargs
        return args, {**kwargs, "estimand": candidates[0]}

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
