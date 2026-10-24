"""The estimates table, the one place a summary applies the inference-status rule.

``docs/architecture-invariants.md`` (inference status) requires that a reader of a spread
branch on each estimate's ``supplies_inference``. This builder does that for every
summary: a fit whose estimates supply no inference prints the estimate and the diagnostic
spread under its diagnostic label, a row that supplies none in an otherwise inferential
table is starred with its inference cells empty, and the ``no_inference`` note gives the
status's plain reason from ``STATUS_RECORDS``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .._inference_status import InferenceStatus

from .._inference_status import (
    FEW_CLUSTER_THRESHOLD,
    is_labelled,
    status_record,
)
from ..inference.influence import spread_name
from . import Cell, Note, SummaryTable, note, vocabulary
from ._render import decimals_for, number_text, pvalue_text


def _names(names: list[str]) -> str:
    return ", ".join(names)


def _spread_label(status: str) -> str:
    """The label of the diagnostic spread an estimate with this status prints."""
    return vocabulary.SPREAD_LABELS[spread_name("std_err", cast("InferenceStatus", status))]


def reason_kind(reason: str) -> str:
    """Which kind of gap a support reason names: a key of ``EXPERIMENTAL_REASONS``.

    Parameters
    ----------
    reason : str
        An estimate's ``support.reason``.

    Returns
    -------
    str
        ``"no_row"`` when no validation study covers the configuration, ``"family"`` when
        the study covers other outcome types or interval kinds, ``"condition"`` otherwise.
    """
    if reason.startswith("No registered study"):
        return "no_row"
    if " licenses the " in reason or " licenses no " in reason:
        return "family"
    return "condition"


#: A validation-row id (``T01.continuous``), a roadmap id (``F18``, ``RM42b``) or a support
#: id (``G12``): library bookkeeping a reader of the analysis has no use for.
_ID = re.compile(r"\b(?:[A-Z]{1,2}\d+[a-z]?(?:\.[A-Za-z_]+)*)\b")


#: A sentence that points at a roadmap item, and a roadmap id in parentheses.
_ROADMAP_SENTENCE = re.compile(r"\s*\b[A-Z]{1,2}\d+[a-z]? in docs/roadmap\.md\b[^.]*\.")
_ROADMAP_PAREN = re.compile(r"\s*\(\s*(?:roadmap\s+)?[A-Z]{1,2}\d+[a-z]?(?:\s+part\s+[a-z])?\s*\)")


def without_roadmap(text: str) -> str:
    """Library prose with its roadmap pointers taken out.

    Only a sentence of the form ``"X58 in docs/roadmap.md ..."`` and an id in parentheses
    go; a bare name such as a column called ``X1`` stays as written.

    Parameters
    ----------
    text : str
        A warning message or a method's description.

    Returns
    -------
    str
        The text without the pointers.
    """
    return _ROADMAP_PAREN.sub("", _ROADMAP_SENTENCE.sub("", text)).strip()


def plain_support_reason(reason: str) -> str:
    """A support reason with the library's row and roadmap ids taken out.

    Parameters
    ----------
    reason : str
        An estimate's ``support.reason``.

    Returns
    -------
    str
        The same facts in words a reader of the analysis can use.
    """

    def nearest(match: re.Match[str]) -> str:
        axes = [vocabulary.SUPPORT_AXES.get(axis, axis) for axis in match[1].split(", ")]
        return f"The nearest validated kind of fit differs in {', '.join(axes)}."

    text = re.sub(
        r"Nearest row: \S+(?: \([^)]*\))?, differing on ([^.]*)\.", nearest, without_roadmap(reason)
    )
    text = text.replace("covers this configuration", "covers this kind of fit")
    text = re.sub(r"\bRow \S+ licenses", "The validation study licenses", text)
    text = text.replace("registered study", "validation study")
    text = _ID.sub("", text)
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    return re.sub(r"\s{2,}", " ", text).replace(" .", ".").strip()


def estimates_table(
    estimates: Mapping[str, Any],
    *,
    level: float,
) -> tuple[SummaryTable, list[Note]]:
    """The estimates table of a result and the notes its rows need.

    Parameters
    ----------
    estimates : mapping of str to ParameterEstimate
        The rows, in report order.
    level : float
        The confidence level of the intervals, such as ``0.95``.

    Returns
    -------
    tuple
        The table (``kind="estimates"``) and the notes: ``no_inference``,
        ``experimental``, ``spread_scale`` and ``t_reference``, each only when it applies.
    """
    items = list(estimates.items())
    inferential = {name: bool(estimate.supplies_inference) for name, estimate in items}
    whole_diagnostic = not any(inferential.values())
    t_reference = any(estimate.reference_df is not None for _, estimate in items)
    experimental = [
        name for name, estimate in items if estimate.support.label == vocabulary.EXPERIMENTAL
    ]
    spreads = [
        estimate.std_error if inferential[name] else estimate.plugin_std_error
        for name, estimate in items
    ]
    decimals = decimals_for(spreads)

    def num(value: float) -> Cell:
        return Cell(float(value), number_text(value, decimals))

    rows: list[tuple[Cell, ...]] = []
    if whole_diagnostic:
        status = items[0][1].inference
        spread_label = vocabulary.SPREAD_LABELS[
            spread_name("std_err", cast("InferenceStatus", status))
        ]
        columns: list[str] = ["", vocabulary.ESTIMATE, spread_label]
        for name, estimate in items:
            rows.append((Cell(name, name), num(estimate.psi), num(estimate.plugin_std_error)))
    else:
        columns = [
            "",
            vocabulary.ESTIMATE,
            vocabulary.SPREAD_LABELS["std_err"],
            vocabulary.interval_label(level),
            vocabulary.P_VALUE,
            *([vocabulary.DF] if t_reference else []),
        ]
        for name, estimate in items:
            if not inferential[name]:
                missing = Cell(None, vocabulary.MISSING)
                rows.append(
                    (
                        Cell(name, name + vocabulary.NO_INFERENCE_MARK),
                        num(estimate.psi),
                        missing,
                        missing,
                        missing,
                        *([missing] if t_reference else []),
                    )
                )
                continue
            low, high = estimate.ci
            rows.append(
                (
                    Cell(name, name),
                    num(estimate.psi),
                    num(estimate.std_error),
                    Cell(
                        (float(low), float(high)),
                        f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
                    ),
                    # A transformed estimate whose default null has no value on its
                    # scale has no default test, and prints none.
                    Cell(float(pvalue := estimate._reported_pvalue()), pvalue_text(pvalue)),
                    *(
                        [
                            Cell(
                                estimate.reference_df,
                                "normal"
                                if estimate.reference_df is None
                                else str(estimate.reference_df),
                            )
                        ]
                        if t_reference
                        else []
                    ),
                )
            )
    if experimental:
        columns.append(vocabulary.SUPPORT)
        rows = [
            (
                *row,
                Cell(
                    estimate.support.label,
                    vocabulary.EXPERIMENTAL
                    if estimate.support.label == vocabulary.EXPERIMENTAL
                    else "",
                ),
            )
            for row, (_, estimate) in zip(rows, items, strict=True)
        ]
    table = SummaryTable(tuple(columns), tuple(rows), kind="estimates")

    notes: list[Note] = []
    withheld: dict[str, list[str]] = {}
    for name, estimate in items:
        if not inferential[name]:
            withheld.setdefault(estimate.inference, []).append(name)
    if withheld:
        sentences = []
        for status, names in withheld.items():
            record = status_record(status)
            label = vocabulary.SPREAD_LABELS[
                spread_name("std_err", cast("InferenceStatus", status))
            ]
            if whole_diagnostic:
                sentences.append(
                    vocabulary.NOTE_TEXTS["no_inference"].format(
                        names=_names(names), reason=record.plain_reason, label=label
                    )
                )
            else:
                listed = ", ".join(
                    f"{name} {number_text(estimates[name].plugin_std_error, decimals)}"
                    for name in names
                )
                sentences.append(
                    vocabulary.NOTE_TEXTS["no_inference_row"].format(
                        names=_names(names),
                        reason=record.plain_reason,
                        label=label,
                        spreads=listed,
                    )
                )
        notes.append(note("no_inference", " ".join(sentences)))
    if experimental:
        sentences = []
        shown: dict[str, list[str]] = {}
        generic: list[str] = []
        for name in experimental:
            status = estimates[name].inference
            if is_labelled(status) and inferential[name]:
                shown.setdefault(status, []).append(name)
            else:
                generic.append(name)
        for status, names in shown.items():
            sentences.append(f"{_names(names)}: {status_record(status).plain_reason}")
        by_kind: dict[str, list[str]] = {}
        for name in generic:
            by_kind.setdefault(reason_kind(estimates[name].support.reason), []).append(name)
        for kind, names in by_kind.items():
            sentences.append(vocabulary.EXPERIMENTAL_REASONS[kind].format(names=_names(names)))
        with_interval = any(inferential[name] for name in experimental)
        sentences.append(
            vocabulary.NOTE_TEXTS["experimental"].format(
                what="the interval" if with_interval else "them"
            )
        )
        notes.append(note("experimental", " ".join(sentences)))
    ratio = [
        name for name, estimate in items if estimate.scale == "ratio" and estimate.transform is None
    ]
    transformed = [name for name, estimate in items if estimate.transform is not None]
    if (ratio or transformed) and not whole_diagnostic:
        sentences = []
        if ratio:
            sentences.append(vocabulary.NOTE_TEXTS["spread_scale"].format(names=_names(ratio)))
        if transformed:
            sentences.append(
                vocabulary.NOTE_TEXTS["spread_scale_transform"].format(names=_names(transformed))
            )
        notes.append(note("spread_scale", " ".join(sentences)))
    if t_reference:
        notes.append(
            note(
                "t_reference",
                vocabulary.NOTE_TEXTS["t_reference"].format(threshold=FEW_CLUSTER_THRESHOLD),
            )
        )
    return table, notes
