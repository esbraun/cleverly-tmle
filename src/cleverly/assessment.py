"""Capability-aware post-fit diagnostics, validation, and replay metadata.

Assessment lives on a fitted result because its questions depend on the artifacts the
method actually produced.  The facades in this module do not infer support from a result
class and hope for the best: every public operation has a declaration for every public
scalar result family, including deliberate refusals.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from functools import cached_property
from typing import Any, Literal

import numpy as np

from ._typing import FloatArray, IntArray
from .data.weighting import effective_sample_size
from .exceptions import CapabilityError
from .inference.cluster import cluster_sums
from .utils.frames import emit_frame
from .utils.text import format_table
from .validation.drtmle import IDENTITY_TOLERANCE
from .validation.score import DEFAULT_TOLERANCE

__all__ = [
    "ASSESSMENT_CAPABILITIES",
    "INTERPRETERS",
    "SENSITIVITY_ROUTES",
    "STITCHED_SCORE_Z_TOLERANCE",
    "AssessmentCapability",
    "AssessmentItem",
    "AssessmentReport",
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


#: How far a cross-fitted longitudinal fit's *stitched* score may sit from zero, in
#: standard errors of its own residual, before :func:`_longitudinal_scores` calls it a
#: defect rather than sampling.
#:
#: The stitched score is not a solved equation.  Each outer fold fits its ``epsilon`` on
#: the rows it does not report, so what the pooled residual has to be is a mean-zero draw,
#: and the scale to judge a mean-zero draw on is its own standard error.  Measured over 300
#: replications of ``make_longitudinal`` at ``n=500`` and five folds, the mean ``|z|`` per
#: parameter ran from 0.006 to 0.08.  Four standard errors is therefore a long way outside
#: anything the construction produces, while a fold-mapping or stitching defect -- which
#: multiplies the residual by a constant rather than perturbing it -- moves ``z`` by orders
#: of magnitude and cannot hide under it.
#:
#: Not a caller argument.  ``tolerance`` on
#: :meth:`~cleverly.assessment.DiagnosticsFacade.score_equations` is a *relative-score*
#: tolerance, and one number cannot mean both "close enough to solved" and "consistent with
#: noise"; passing it to both gates would silently apply ``1e-3`` standard errors here and
#: fail every cross-fitted fit.
STITCHED_SCORE_Z_TOLERANCE = 4.0


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
    declared = getattr(result, "assessment_method", None)
    if isinstance(declared, str) and declared:
        return declared
    raise TypeError("assessment requires the fitted artifact to declare its fitted method")


def assessment_capabilities(result: Any) -> tuple[AssessmentCapability, ...]:
    """All operation declarations for the result's family."""

    family = _family(result)
    rows = tuple(item for item in ASSESSMENT_CAPABILITIES if item.result_family == family)
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
        item = self[name]
        if item._report is _AbsentReport.TOKEN:
            raise KeyError(f"diagnostic {name!r} did not run")
        return item.report

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
        return tuple(dict.fromkeys(step for item in self.items for step in item.next_steps))

    def summary(self) -> str:
        """Return a printable table of operation statuses.

        Returns
        -------
        str
            A printable table, one line per requested operation.
        """
        return format_table(
            ["diagnostic", "status", "detail", "next step"],
            [
                [item.name, item.status.value, item.detail, "; ".join(item.next_steps)]
                for item in self.items
            ],
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
        return tuple(
            dict.fromkeys(step for _, item in self._presented() for step in item.next_steps)
        )

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
        matches = [
            (owner, item)
            for owner, item in candidates
            if item.name == name and (surface is None or owner == surface)
        ]
        if not matches:
            raise KeyError(f"no presented assessment report named {name!r}")
        if len(matches) > 1:
            owners = [owner for owner, _ in matches]
            raise KeyError(
                f"assessment report {name!r} is ambiguous across {owners}; pass surface="
            )
        item = matches[0][1]
        if item._report is _AbsentReport.TOKEN:
            raise KeyError(f"assessment operation {name!r} did not run")
        return item.report

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
        backend = self.validation.backend
        return emit_frame(
            {
                "surface": [surface for surface, _ in rows],
                "check": [item.name for _, item in rows],
                "status": [item.status.value for _, item in rows],
                "detail": [item.detail for _, item in rows],
                "next_steps": ["; ".join(item.next_steps) for _, item in rows],
            },
            data,
            backend=backend,
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
            rows = [(owner, item) for owner, item in self._presented() if owner == surface]
            sections.extend(
                [
                    surface.capitalize(),
                    "-" * len(surface),
                    format_table(
                        ["operation", "status", "detail", "next step"],
                        [
                            [item.name, item.status.value, item.detail, "; ".join(item.next_steps)]
                            for _, item in rows
                        ],
                    ),
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


@dataclass(frozen=True)
class _CachedFrame:
    columns: tuple[str, ...]
    values: tuple[tuple[Any, ...], ...]
    backend: str | None

    @classmethod
    def from_frame(cls, frame: Any, backend: str | None) -> _CachedFrame:
        payload = _frame_payload(frame)
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


_RETAIN_PACKED: ContextVar[bool] = ContextVar("assessment_retain_packed", default=False)


def _cached(
    result: Any,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    compute: Callable[[], Any],
) -> Any:
    cache = result.assessment_cache
    key = _cache_key(operation, args, kwargs)
    if key not in cache:
        value = compute()
        cache[key] = _pack_cached(value, getattr(result.data, "backend", None))
    return cache[key] if _RETAIN_PACKED.get() else _unpack_cached(cache[key])


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
    """One verdict about one node's targeting.

    ``converged`` is the fit's own flag: whether that node's Newton step settled against
    the targeting tolerance it was configured with.  ``passed`` additionally holds the row
    to the tolerance the *caller* asked for.  They are kept apart because they can
    disagree, and because only their conjunction is safe -- a caller tolerance may tighten
    the verdict and may never license a fluctuation whose step failed.

    ``kind`` says which question the row answers, because a cross-fitted fit poses two and
    they have different right answers.

    ``component`` names an MSM score component. Such a row pools all live regimen cells,
    so ``regimen`` and ``horizon`` are ``None``. Ordinary regimen rows have no component.

    ``"solver"``
        Did the fluctuation reach the root of the equation it was *given*?  On an ordinary
        fit that equation is the node's own score and ``relative_score`` is it.  On a
        cross-fitted fit it is each outer fold's score on its training complement, and
        ``relative_score`` is the largest across the folds.  Either way the answer should
        be at solver tolerance, and a failure here is a solver failure.

    ``"stitching"``
        Is the score of the *stitched* fit where sampling alone would leave it?  Emitted
        only on a cross-fitted fit, where the answer is not zero and is not meant to be:
        every fold fits its ``epsilon`` on rows it does not report, so the pooled residual
        is noise about zero rather than a solved equation.  ``z`` is that residual over its
        own standard error and ``relative_score`` is the raw magnitude, reported so the
        reader can see what the ``z`` is a ratio of.  A stitching, indexing or fold-mapping
        defect moves ``z`` by orders of magnitude, which is what this row is for.
    """

    regimen: str | None
    cause: str | None
    horizon: int | None
    time: int
    component: str | None
    kind: str
    score: float
    relative_score: float
    #: The score over its own standard error.  ``nan`` on a ``"solver"`` row, whose claim
    #: is that the score is zero rather than that it is small relative to anything.
    z: float
    converged: bool
    passed: bool
    n_iter: int
    failure: str | None


@dataclass(frozen=True)
class LongitudinalScoreDiagnostics:
    """Stagewise targeting verdicts, gated at the tolerances they were asked for.

    ``tolerance`` bounds a ``"solver"`` row's *relative* score -- the largest score
    component as a fraction of its maximum possible magnitude, which is the quantity the
    sequential targeting loop itself gates on.  The point-treatment report answers the same
    question on a different scale, comparing the score in the outcome's own units against
    ``tolerance * se / sqrt(n)``; see :data:`~cleverly.validation.score.DEFAULT_TOLERANCE`.
    The number is carried here so a report says which gate produced its verdict.

    ``z_tolerance`` bounds a ``"stitching"`` row instead, in standard errors, because that
    row's score is not a solved equation and holding it to a relative tolerance would fail
    every cross-fitted fit for doing exactly what it is supposed to do.  A fit with no
    cross-fitting emits no such row and ``z_tolerance`` never binds.
    """

    rows: tuple[LongitudinalScoreRow, ...]
    tolerance: float
    backend: str | None = None
    z_tolerance: float = STITCHED_SCORE_Z_TOLERANCE

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
                "component": [row.component for row in self.rows],
                "kind": [row.kind for row in self.rows],
                "score": [row.score for row in self.rows],
                "relative_score": [row.relative_score for row in self.rows],
                "z": [row.z for row in self.rows],
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
    """One row per node: how heavy the weights got and how much the bounds moved.

    ``max_weight`` and ``effective_n`` read ``step.clever``, and ``share_truncated`` reads
    ``fit.cumulative``.  On a cross-fitted fit those are not two views of one array: the
    covariate is stitched from each fold's own mechanism slab while ``cumulative`` is the
    out-of-fold mechanism, so ``1 / cumulative`` does not reproduce the weight.  Each column
    is read from the array that answers its own question -- what a row was weighted by, and
    how far the bounds moved the mechanism -- rather than both from whichever one is nearer.
    """
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


def _standardized_score(contribution: FloatArray, cluster: IntArray | None = None) -> FloatArray:
    """Standardize mean score components by their independent-unit standard errors."""
    values = np.asarray(contribution, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    if n < 2:
        return np.full(values.shape[1], np.nan)
    if cluster is None:
        standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(n)
    else:
        sums = cluster_sums(values, np.asarray(cluster))
        n_clusters = sums.shape[0]
        if n_clusters < 2:
            return np.full(values.shape[1], np.nan)
        standard_error = np.sqrt(n_clusters * np.var(sums, axis=0, ddof=1) / n**2)
    mean = np.mean(values, axis=0)
    return np.divide(
        mean,
        standard_error,
        out=np.full(values.shape[1], np.nan),
        where=standard_error > 0.0,
    )


def _stitched_score_z(step: Any, weights: FloatArray, cluster: IntArray | None = None) -> float:
    r"""The stitched score over its own standard error.

    The score is :math:`P_n[w H (Z - \bar Q^*)]`. Independent rows use the row-level
    standard error. Clustered rows first sum their contributions within cluster and use
    the same finite-sample scaling as the inference layer.

    Returns ``nan`` when the residual has no spread, which is a degenerate node rather than
    a perfect one and is not something to report a ``z`` of zero for.
    """
    contribution = weights * step.clever * (step.pseudo_outcome - step.targeted)
    return float(_standardized_score(contribution, cluster)[0])


def _msm_node_contributions(
    result: Any, msm_fit: Any, time: int
) -> tuple[FloatArray, FloatArray, Any]:
    """Per-unit pooled score and scale contributions for one longitudinal MSM node."""
    model = msm_fit.model
    weights = np.asarray(result.data.weights, dtype=float)
    contribution = np.zeros((result.data.n, model.n_terms), dtype=float)
    maximum = np.zeros_like(contribution)
    fluctuation = None
    for cell_index, (cell, cell_fit) in enumerate(zip(model.cells, msm_fit.fits, strict=True)):
        if cell.horizon < time:
            continue
        step = next(item for item in cell_fit.steps if item.time == time)
        fluctuation = step.fluctuation
        multiplier = weights * model.weights[:, cell_index] * step.clever
        design = msm_fit.fluctuation_design[:, cell_index, :]
        residual = step.pseudo_outcome - step.targeted
        contribution += multiplier[:, None] * design * residual[:, None]
        maximum += np.abs(multiplier[:, None] * design)
    if fluctuation is None:  # pragma: no cover - a model cell always reaches its own nodes
        raise RuntimeError(f"the longitudinal MSM has no live cell at time {time}")
    return contribution, maximum, fluctuation


def _longitudinal_msm_scores(result: Any, *, tolerance: float) -> LongitudinalScoreDiagnostics:
    """Pooled component-wise node diagnostics for a longitudinal working model."""
    rows = []
    for msm_fit in result.msm_fits:
        times = sorted({step.time for fit in msm_fit.fits for step in fit.steps})
        for time in times:
            contribution, maximum, fluctuation = _msm_node_contributions(result, msm_fit, time)
            pooled_score = np.mean(contribution, axis=0)
            pooled_scale = np.mean(maximum, axis=0)
            z = _standardized_score(contribution, result.data.cluster)
            for component, term in enumerate(msm_fit.model.terms):
                converged = bool(fluctuation.converged)
                if fluctuation.folds:
                    relatives = []
                    scores = []
                    for record in fluctuation.folds:
                        scale = (
                            np.asarray(record.score_scale, dtype=float)
                            if record.score_scale is not None
                            else np.asarray(fluctuation.score_scale, dtype=float)
                        )
                        score = float(np.asarray(record.score, dtype=float)[component])
                        scores.append(abs(score))
                        relatives.append(abs(score) / max(float(scale[component]), 1e-300))
                    solver_score = max(scores)
                    solver_relative = max(relatives)
                else:
                    solver_score = abs(float(pooled_score[component]))
                    solver_relative = solver_score / max(float(pooled_scale[component]), 1e-300)
                rows.append(
                    LongitudinalScoreRow(
                        None,
                        msm_fit.cause,
                        None,
                        time,
                        term,
                        "solver",
                        float(result.scaler.range * solver_score),
                        solver_relative,
                        float("nan"),
                        converged,
                        converged and solver_relative <= tolerance,
                        int(fluctuation.n_iter),
                        fluctuation.failure,
                    )
                )
                if not fluctuation.folds:
                    continue
                component_z = float(z[component])
                relative = abs(float(pooled_score[component])) / max(
                    float(pooled_scale[component]), 1e-300
                )
                rows.append(
                    LongitudinalScoreRow(
                        None,
                        msm_fit.cause,
                        None,
                        time,
                        term,
                        "stitching",
                        float(result.scaler.range * pooled_score[component]),
                        relative,
                        component_z,
                        converged,
                        bool(
                            np.isfinite(component_z)
                            and abs(component_z) <= STITCHED_SCORE_Z_TOLERANCE
                        ),
                        int(fluctuation.n_iter),
                        fluctuation.failure,
                    )
                )
    return LongitudinalScoreDiagnostics(tuple(rows), tolerance, result.data.backend)


def _longitudinal_scores(result: Any, *, tolerance: float) -> LongitudinalScoreDiagnostics:
    """Every node's targeting verdicts: one row per question the node's fit poses.

    The solver gate is a *conjunction*, and deliberately so.  Sequential targeting settles
    against its own ``tol`` -- ``1e-10``, far tighter than the default asked for here -- so
    requiring ``converged`` as well leaves the default verdict exactly what it was while
    letting a caller tighten it.  Gating on the relative score alone would do the opposite: a
    node whose Newton step failed but whose residual score happens to sit under a loose
    tolerance would be reported as passing, which is the one answer this diagnostic must
    never give.

    A cross-fitted node earns a second row, because the first one stops being able to see
    the thing that can go wrong.  Its ``K`` solves each reach their own root on their own
    training complement, so the solver row is at machine precision whatever the stitched fit
    looks like -- including when the folds were stitched back in the wrong order, or a slab
    was read for the wrong fold.  The stitching row is where that shows.
    """
    if result.msm is not None:
        return _longitudinal_msm_scores(result, tolerance=tolerance)

    rows = []
    for fit in result.fits.values():
        weights = np.asarray(fit.obs_weights, dtype=float)
        for step in fit.steps:
            fluctuation = step.fluctuation
            horizon = fit.horizon if result.data.is_survival else None
            converged = bool(fluctuation.converged)
            # On a cross-fitted node the solved equations are the folds' own, and the
            # aggregate `score` is the stitched fit's -- a different quantity, reported on
            # the row below.  The worst fold is the honest summary of `K` solves: an
            # average would let nine good folds hide one that did not move.
            solver_relative = (
                max(
                    float(
                        np.max(
                            np.abs(record.score)
                            / np.maximum(
                                record.score_scale
                                if record.score_scale is not None
                                else fluctuation.score_scale,
                                1e-300,
                            )
                        )
                    )
                    for record in fluctuation.folds
                )
                if fluctuation.folds
                else float(fluctuation.relative_score_norm)
            )
            solver_score = (
                max(float(np.max(np.abs(record.score))) for record in fluctuation.folds)
                if fluctuation.folds
                else float(fluctuation.score_norm)
            )
            rows.append(
                LongitudinalScoreRow(
                    fit.regimen.label,
                    fit.cause,
                    horizon,
                    step.time,
                    None,
                    "solver",
                    float(result.scaler.range * solver_score),
                    solver_relative,
                    float("nan"),
                    converged,
                    converged and solver_relative <= tolerance,
                    int(fluctuation.n_iter),
                    fluctuation.failure,
                )
            )
            if not fluctuation.folds:
                continue
            z = _stitched_score_z(step, weights, result.data.cluster)
            rows.append(
                LongitudinalScoreRow(
                    fit.regimen.label,
                    fit.cause,
                    horizon,
                    step.time,
                    None,
                    "stitching",
                    float(result.scaler.range * fluctuation.score_norm),
                    float(fluctuation.relative_score_norm),
                    z,
                    converged,
                    bool(np.isfinite(z) and abs(z) <= STITCHED_SCORE_Z_TOLERANCE),
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

    @cached_property
    def _capability_map(self) -> dict[str, AssessmentCapability]:
        method = _method(self._result)
        return {
            item.operation: item
            if not item.available
            or method in item.methods
            or (method == "unknown" and item.execution == "summarize")
            else replace(
                item,
                available=False,
                status=AssessmentStatus.NOT_APPLICABLE,
                reason=f"the fitted method {method!r} does not support this operation",
            )
            for item in self.capabilities
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

    def _run_all(
        self,
        *,
        include_refits: bool,
        include_retargets: bool,
        arguments: Mapping[str, Mapping[str, Any]] | None,
        random_state: int | None,
    ) -> DiagnosticReport:
        supplied = self._validated_arguments(arguments, random_state)

        def compute() -> DiagnosticReport:
            items = []
            for capability in self.capabilities:
                operation_arguments = dict(supplied.get(capability.operation, {}))
                capability = self._capability_for_arguments(
                    capability.operation, operation_arguments
                )
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
                missing_arguments = tuple(
                    name
                    for name in capability.requires_arguments
                    if name not in operation_arguments
                )
                if missing_arguments:
                    # A combined report runs every operation argument-free, so one with a
                    # required argument and no default cannot appear in it.  Choosing a
                    # value here -- which covariates to benchmark against -- would be a
                    # scientific choice made silently on the caller's behalf.
                    # A row may declare more than one argument, so the sentence has to
                    # agree in number.  ``", ".join`` alone rendered "an explicit grid,
                    # estimand argument", which reads as one argument named "grid,
                    # estimand".
                    names = missing_arguments
                    needed = (
                        names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
                    )
                    phrase = (
                        f"an explicit {needed} argument"
                        if len(names) == 1
                        else f"explicit {needed} arguments"
                    )
                    items.append(
                        AssessmentItem(
                            capability.operation,
                            AssessmentStatus.UNAVAILABLE,
                            f"needs {phrase}, which a combined report has no basis to choose",
                            (
                                f"call result.{self._attribute}.{capability.operation}() "
                                f"directly with {needed}",
                            ),
                        )
                    )
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
                    interpreted = INTERPRETERS[capability.operation](report, self._result)
                    if capability.operation == "omitted_confounding":
                        defaults = [
                            name for name in ("cf_y", "cf_d") if name not in operation_arguments
                        ]
                        if defaults:
                            provenance = (
                                "at the default strengths"
                                if len(defaults) == 2
                                else f"at the default {defaults[0]} strength"
                            )
                            interpreted = replace(
                                interpreted, detail=f"{provenance}; {interpreted.detail}"
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
        if arguments is None:
            return {}
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping from operation names to mappings")
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
                selection = self._evalue_selection(None)
                if (
                    selection.branch != "derived_rr"
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
    def capabilities(self) -> tuple[AssessmentCapability, ...]:
        """Return declared operations and their availability."""
        rows = assessment_capabilities(self._result)
        if replayability(self._result).retarget_cached_nuisances:
            return rows
        return tuple(
            replace(
                row,
                available=False,
                status=AssessmentStatus.UNAVAILABLE,
                reason="retargeting requires the fitted estimator that produced the result",
            )
            if row.operation == "truncation_curve" and row.available
            else row
            for row in rows
        )

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
            arguments=arguments,
            random_state=random_state,
        )


def _item_from_capability(capability: AssessmentCapability) -> AssessmentItem:
    return AssessmentItem(
        capability.operation,
        capability.status,
        capability.reason or capability.interpretation,
    )


def _score_item(report: Any, result: Any) -> AssessmentItem:
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


def _support_item(report: Any, _result: Any) -> AssessmentItem:
    warning = _support_warning(report)
    truncated, ess = _support_metrics(report)
    facts = []
    if truncated is not None:
        facts.append(f"truncated fraction {truncated:.1%}")
    if ess is not None:
        facts.append(f"minimum effective-sample-size ratio {ess:.1%}")
    if warning:
        facts.append(warning)
    detail = "; ".join(facts) if facts else "stored support report completed"
    return AssessmentItem(
        "support",
        AssessmentStatus.WARNING if warning else AssessmentStatus.PASSED,
        detail,
        () if warning is None else ("inspect result.diagnostics.support()",),
    )


def _nuisance_item(report: Any, _result: Any) -> AssessmentItem:
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


def _correction_item(report: Any, _result: Any) -> AssessmentItem:
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


def _frame_payload(frame: Any) -> dict[str, Any]:
    if isinstance(frame, _CachedFrame):
        return dict(zip(frame.columns, frame.values, strict=True))
    if type(frame).__module__.startswith("polars"):
        return frame.to_dict(as_series=False)
    return frame.to_dict(orient="list")


def _range(values: Sequence[Any]) -> tuple[float, float] | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return (min(finite), max(finite)) if finite else None


def _format_range(values: tuple[float, float] | None) -> str:
    return "no finite values" if values is None else f"[{values[0]:.4g}, {values[1]:.4g}]"


def _truncation_item(report: Any, _result: Any) -> AssessmentItem:
    payload = _frame_payload(report)
    bounds = _range(payload.get("bound", payload.get("g_bound", ())))
    estimates = _range(payload.get("psi", payload.get("estimate", ())))
    return AssessmentItem(
        "truncation_curve",
        AssessmentStatus.COMPLETED,
        f"evaluated bound range {_format_range(bounds)}; estimate range {_format_range(estimates)}",
    )


def _refute_item(report: Any, _result: Any) -> AssessmentItem:
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


def _omitted_item(report: Any, _result: Any) -> AssessmentItem:
    spans = report.lower <= report.null_hypothesis <= report.upper
    return AssessmentItem(
        "omitted_confounding",
        AssessmentStatus.WARNING if spans else AssessmentStatus.COMPLETED,
        f"cf_y={report.cf_y:.3g}, cf_d={report.cf_d:.3g}, rho={report.rho:.3g}; "
        f"bias-adjusted interval [{report.lower:.4g}, {report.upper:.4g}]",
        () if not spans else ("inspect the retained omitted-confounding bounds",),
    )


def _robustness_item(report: Any, _result: Any) -> AssessmentItem:
    return AssessmentItem(
        "robustness_value",
        AssessmentStatus.COMPLETED,
        f"point robustness value {report['rv']:.4g}; confidence-limit value {report['rva']:.4g}",
    )


def _elements_item(report: Any, _result: Any) -> AssessmentItem:
    return AssessmentItem(
        "elements",
        AssessmentStatus.COMPLETED,
        f"sigma2={report.sigma2:.4g}, nu2={report.nu2:.4g}, max_bias={report.max_bias:.4g}",
    )


def _contour_item(report: Any, _result: Any) -> AssessmentItem:
    payload = _frame_payload(report)
    return AssessmentItem(
        "contour",
        AssessmentStatus.COMPLETED,
        f"grid {len(set(payload['cf_d']))} x {len(set(payload['cf_y']))}; "
        f"cf_d range {_format_range(_range(payload['cf_d']))}; "
        f"cf_y range {_format_range(_range(payload['cf_y']))}; "
        f"value range {_format_range(_range(payload['value']))}; inspect the retained frame",
    )


def _benchmark_item(report: Any, _result: Any) -> AssessmentItem:
    return AssessmentItem(
        "benchmark",
        AssessmentStatus.COMPLETED,
        f"covariates={report.covariates}; cf_y={report.cf_y:.3g}, cf_d={report.cf_d:.3g}, "
        f"rho={report.rho:.3g}, delta_psi={report.delta_psi:.4g}",
    )


def _simulated_item(report: Any, _result: Any) -> AssessmentItem:
    movements = [
        abs(float(cell.displacement))
        for cell in report.successful_cells
        if cell.displacement is not None
    ]
    corner = report.cells[-1].induced_treatment_association if report.cells else None
    detail = (
        f"maximum successful displacement {max(movements, default=float('nan')):.4g}; "
        f"failed cells {len(report.failures)}; corner association {corner}"
    )
    return AssessmentItem(
        "simulated_confounding",
        AssessmentStatus.WARNING if report.failures else AssessmentStatus.COMPLETED,
        detail,
        () if not report.failures else ("inspect the retained cell failures",),
    )


def _evalue_item(report: Any, _result: Any) -> AssessmentItem:
    detail = (
        f"point={report.point:.4g}, limit={report.limit:.4g}, source scale={report.scale}; "
        + ("approximate conversion" if report.approximate else "exact risk-ratio branch")
    )
    if report.limit == 1.0:
        detail += "; the interval already includes the null"
    return AssessmentItem("evalue", AssessmentStatus.COMPLETED, detail)


def _missingness_item(report: Any, _result: Any) -> AssessmentItem:
    payload = _frame_payload(report)
    return AssessmentItem(
        "missingness",
        AssessmentStatus.COMPLETED,
        f"gamma range {_format_range(_range(payload['gamma']))}; "
        f"estimate range {_format_range(_range(payload['psi']))}",
    )


def _tipping_item(report: Any, _result: Any) -> AssessmentItem:
    detail = (
        "no tipping value occurred in the searched interval"
        if report is None
        else f"tipping gamma {float(report):.4g}"
    )
    return AssessmentItem("tipping_gamma", AssessmentStatus.COMPLETED, detail)


def _stagewise_item(report: Any, _result: Any) -> AssessmentItem:
    truncated, ess = _support_metrics(report)
    return AssessmentItem(
        "stagewise",
        AssessmentStatus.COMPLETED,
        f"{len(report.rows)} stage row(s); maximum truncated fraction {truncated}; "
        f"minimum effective-sample-size ratio {ess}",
    )


INTERPRETERS: dict[str, Callable[[Any, Any], AssessmentItem]] = {
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
        reports = {
            "score_equations": facade.score_equations(),
            "support": facade.support(),
            "nuisance_models": facade.nuisance_models(),
        }
        items = [
            replace(
                INTERPRETERS[name](report, result),
                _report=_pack_cached(report, result.data.backend),
            )
            for name, report in reports.items()
        ]
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
    if arguments is not None and not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping from operation names to mappings")
    supplied = {} if arguments is None else dict(arguments)
    diagnostics, sensitivity = result.diagnostics, result.sensitivity
    diagnostic_names = set(diagnostics._capability_map)
    sensitivity_names = set(sensitivity._capability_map)
    unknown = sorted(set(supplied) - diagnostic_names - sensitivity_names)
    if unknown:
        raise KeyError(f"unknown assessment operation(s): {unknown}")
    diagnostic_arguments = {k: v for k, v in supplied.items() if k in diagnostic_names}
    sensitivity_arguments = {k: v for k, v in supplied.items() if k in sensitivity_names}
    diagnostics._validated_arguments(diagnostic_arguments, random_state)
    sensitivity._validated_arguments(sensitivity_arguments, random_state)
    return AssessmentReport(
        validation=validate_result(result, diagnostics),
        diagnostics=diagnostics.run_all(
            include_refits=include_refits,
            include_retargets=include_retargets,
            arguments=diagnostic_arguments,
            random_state=random_state,
        ),
        sensitivity=sensitivity.run_all(
            include_refits=include_refits,
            include_retargets=include_retargets,
            arguments=sensitivity_arguments,
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
        # ``simulated_confounding`` refuses the bare ``ate`` default on a continuous fit.
        # A binary means fit can use the facade's sole-parameter substitution, but a fit
        # that reports several means needs the caller to choose one.
        continuous = not longitudinal and bool(
            getattr(self._result.data, "is_continuous_treatment", False)
        )
        if longitudinal or continuous:
            binary_needs_estimand = False
        else:
            from .sensitivity.simulated_confounding import _eligible_binary_parameter_names

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
                    else "benchmarking requires a replayable point-treatment estimator"
                ),
                requires_arguments=("covariates",),
                accepts_random_state=True,
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
                    if longitudinal
                    else "simulation requires a replayable point-treatment estimator"
                ),
                requires_arguments=("grid", "estimand")
                if continuous or binary_needs_estimand
                else ("grid",),
                accepts_random_state=True,
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
        from .sensitivity.evalue import _EValueRefusal

        status: str | None
        reason: str | None
        execution: Literal["summarize", "retarget"]
        try:
            selected = self._evalue_selection(estimand)
        except _EValueRefusal as error:
            available, status, reason, execution = False, error.status, str(error), "summarize"
        else:
            available, status, reason = True, None, None
            execution = "retarget" if selected.branch == "derived_rr" else "summarize"
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

        ``simulated_confounding`` also answers for the two ratio contrasts, which are not
        linear functionals and so are absent from the linear set every other operation
        reads.  It consults its own eligible set first, and falls back to the linear set
        when that set does not name exactly one parameter.  The fallback is what keeps an
        ``att``-only fit on the path that supplies ``"att"``, so the caller reads the
        accurate source-boundary refusal rather than one about a missing ``"ate"``.

        When the choice stays ambiguous this returns the arguments untouched and the
        analysis refuses for itself.
        :func:`~cleverly.sensitivity.omitted_variable.resolve_parameter` and
        :func:`~cleverly.sensitivity.missingness.missingness_tilt` both name every estimand
        they could have answered for.
        :func:`~cleverly.sensitivity.simulated_confounding.simulated_confounding` refuses on
        its own ``"ate"`` default instead, and lists every reported estimand, so its message
        can name a parameter that a second explicit call would then decline.

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
            from .sensitivity.simulated_confounding import _eligible_binary_parameter_names

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
            arguments=arguments,
            random_state=random_state,
        )
