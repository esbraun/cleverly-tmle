"""The inference statuses, and the one table of what each non-inferential status says.

An estimate declares one :data:`InferenceStatus`. ``"influence_curve"`` is the ordinary
case: the package supplies ``std_error``, ``ci`` and ``pvalue``. Every other status names a
reason the package supplies none, and :data:`NON_INFERENTIAL` holds that reason with every
text a report prints for it. A consumer reads the table and never branches on a status
name, so a new status is one new entry here and one new member of the Literal.

A leaf module. It imports the standard library only, as :mod:`cleverly._typing` does, so
:mod:`cleverly.exceptions`, which the rest of the package imports, can read the table at
module level without an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

__all__ = [
    "NON_INFERENTIAL",
    "InferenceStatus",
    "StatusRecord",
    "assessment_note",
    "inference_reason",
    "precedent_status",
    "status_record",
    "supplies_inference",
]

#: Whether the package supplies inference for an estimate, or only a point estimate and a
#: named diagnostic. ``"influence_curve"`` supplies inference. Each other member is a key
#: of :data:`NON_INFERENTIAL`, and ``tests/unit/test_inference_status_registry.py`` pins
#: that the two lists agree.
InferenceStatus = Literal["influence_curve", "working_mechanism_plugin"]


@dataclass(frozen=True)
class StatusRecord:
    """Every text the package prints for one non-inferential status.

    Parameters
    ----------
    reason : str
        The clause that says what the fit reports, what result is missing, and which
        roadmap item reopens it. Every refusal ends with it, and ``summary()`` prints it
        as a sentence under the table.
    assessment_note : str
        The clause the nuisance report and the assessment's nuisance-model item add.
    summary_label : str
        The header of the spread column in ``TMLEResult.summary()`` and the label in
        ``ParameterEstimate.__repr__``.
    bootstrap_note : str
        The parenthesis ``summary()`` prints beside a bootstrap percentile range.
    diagnostic_noun : str
        The noun a coverage-study report uses for the spread it measured.
    reopened_by : str
        The roadmap id whose anchor in ``docs/roadmap.md`` holds the reopen route.
    """

    reason: str
    assessment_note: str
    summary_label: str
    bootstrap_note: str
    diagnostic_noun: str
    reopened_by: str


#: One record per non-inferential status. The insertion order is the precedence: when a
#: fit meets more than one status, it takes the first one here, as the ordered refusals
#: in ``TMLE._resolve_estimands_for_data`` give "a fit that breaks several rules" the
#: first. :func:`precedent_status` applies that order. The text of each record is
#: asserted by importing it, so a report and a raise cannot drift apart.
NON_INFERENTIAL: Mapping[str, StatusRecord] = MappingProxyType(
    {
        "working_mechanism_plugin": StatusRecord(
            reason=(
                "the greedy, ordered and discrete collaborative paths report no confidence "
                "interval, no p-value and no standard error. The reported curve is the "
                "ordinary efficient influence curve at the candidate the search stopped at, "
                "and no result shows it is this estimator's influence curve when that "
                "working mechanism is not consistent for the treatment law. The point "
                "estimate and the selection path stand. The plug-in standard error of that "
                "curve remains as a diagnostic under plugin_std_error and plugin_interval. "
                "F18 in docs/roadmap.md reopens this when it supplies the estimator's "
                "influence curve."
            ),
            assessment_note=(
                "the reported curve is a working-mechanism diagnostic: no confidence "
                "interval or p-value is available for this path, and F18 in the roadmap is "
                "the condition that reopens it"
            ),
            summary_label="working-mechanism se",
            bootstrap_note=(
                "a diagnostic; the refit bootstrap reruns the selection, and no result "
                "validates its coverage for this path"
            ),
            diagnostic_noun="working-mechanism plug-in diagnostic",
            reopened_by="F18",
        ),
    }
)


def supplies_inference(status: str) -> bool:
    """Whether the package supplies inference at an inference status.

    The one comparison against ``"influence_curve"``. An estimate, a fit, a frame and a
    report each ask it of the status they hold. Any other string is non-inferential, so
    a status with no record fails closed at :func:`status_record` rather than reporting
    an interval.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus`.

    Returns
    -------
    bool
        ``True`` for ``"influence_curve"``, which is when ``std_error``, ``ci`` and
        ``pvalue`` answer. ``False`` for every diagnostic status.
    """
    return status == "influence_curve"


def status_record(status: str) -> StatusRecord:
    """The record of a non-inferential status.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus` other than ``"influence_curve"``.

    Returns
    -------
    StatusRecord
        The texts the package prints for ``status``.

    Raises
    ------
    KeyError
        When ``status`` has no record, which includes ``"influence_curve"``.
    """
    return NON_INFERENTIAL[status]


def inference_reason(status: str) -> str:
    """The reason a non-inferential status gives, as a clause.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus` other than ``"influence_curve"``.

    Returns
    -------
    str
        :attr:`StatusRecord.reason` of ``status``.
    """
    return status_record(status).reason


def assessment_note(status: str) -> str:
    """The note the nuisance report adds for a non-inferential status, as a clause.

    Parameters
    ----------
    status : str
        A declared :data:`InferenceStatus` other than ``"influence_curve"``.

    Returns
    -------
    str
        :attr:`StatusRecord.assessment_note` of ``status``.
    """
    return status_record(status).assessment_note


def precedent_status(statuses: Iterable[str]) -> InferenceStatus:
    """The one status a fit takes when several apply.

    The first of ``statuses`` in the order of :data:`NON_INFERENTIAL`, or
    ``"influence_curve"`` when none of them is non-inferential. A hook that finds more
    than one reason to withhold inference passes them all here, so the precedence lives
    in the table and not in the order a subclass happens to call its parent.

    Parameters
    ----------
    statuses : iterable of str
        The statuses that apply to one fit. ``"influence_curve"`` entries are ignored.

    Returns
    -------
    str
        One of :data:`InferenceStatus`.

    Raises
    ------
    KeyError
        When an entry is neither ``"influence_curve"`` nor a key of
        :data:`NON_INFERENTIAL`.
    """
    applied = {status for status in statuses if not supplies_inference(status)}
    for status in applied:
        status_record(status)
    for status in NON_INFERENTIAL:
        if status in applied:
            return cast(InferenceStatus, status)
    return "influence_curve"
