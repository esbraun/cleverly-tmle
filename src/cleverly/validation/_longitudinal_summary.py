"""The printed summaries of the three longitudinal checks (X89 T3).

Each prints like every other check: a header of facts, one table with a row per regimen and
time point, one reading sentence, and notes for what the table has no column for. Columns a
fit does not have (a cause, a horizon, a trimmed share) are left out, as ``to_frame()``
leaves them out. The SDR and IPSI columns print at full detail. The numbers are read from
the check; nothing here decides a verdict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    note,
    short_repr,
    vocabulary,
)
from .._summary._estimates import without_roadmap
from .._summary._render import count_text, share_text
from ..utils.text import format_negligible

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .longitudinal import (
        LongitudinalDiagnostics,
        LongitudinalNuisanceDiagnostics,
        LongitudinalNuisanceRow,
        LongitudinalScoreDiagnostics,
        LongitudinalScoreRow,
        LongitudinalStageRow,
    )

__all__ = [
    "models_repr",
    "models_summary",
    "node_words",
    "score_repr",
    "score_summary",
    "stage_repr",
    "stage_summary",
]

#: A cumulative probability the limit replaced on every row reads as this share.
_EVERY_ROW = 1.0


def _joined(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _number(value: Any, decimals: int) -> Cell:
    number = _finite(value)
    if number is None:
        return Cell(None, vocabulary.MISSING)
    return Cell(number, f"{number:.{decimals}f}")


def _share(value: Any) -> Cell:
    number = _finite(value)
    if number is None:
        return Cell(None, vocabulary.MISSING)
    return Cell(number, share_text(number))


def _text(value: Any) -> Cell:
    if value is None:
        return Cell(None, vocabulary.MISSING)
    return Cell(value, str(value))


def node_words(
    regimen: str | None,
    time: int,
    *,
    cause: str | None = None,
    horizon: int | None = None,
    component: str | None = None,
    draw: int | None = None,
) -> str:
    """One row of a longitudinal check in words.

    Parameters
    ----------
    regimen : str or None
        The regimen; ``None`` on a row pooled over every regimen.
    time : int
        The one-based time point.
    cause : str or None
        The cause of a competing-risk row.
    horizon : int or None
        The horizon of a horizon-indexed row.
    component : str or None
        The working-model term of a pooled row.
    draw : int or None
        The one-based split of a row on a fit with several splits.

    Returns
    -------
    str
        Such as ``"always (cause relapse, horizon 2) at time 1"``.
    """
    name = vocabulary.EVERY_REGIMEN if regimen is None else str(regimen)
    qualifiers = []
    if component is not None:
        qualifiers.append(vocabulary.NODE_COMPONENT.format(component=component))
    if cause is not None:
        qualifiers.append(vocabulary.NODE_CAUSE.format(cause=cause))
    if horizon is not None:
        qualifiers.append(vocabulary.NODE_HORIZON.format(horizon=horizon))
    if draw is not None:
        qualifiers.append(vocabulary.NODE_SPLIT.format(split=draw))
    if qualifiers:
        name = f"{name} ({', '.join(qualifiers)})"
    return vocabulary.NODE_AT.format(name=name, time=time)


def _identity(
    rows: Sequence[Any], *, regimens: bool = True, draws: bool = False
) -> tuple[tuple[str, ...], list[tuple[Cell, ...]]]:
    """The identity columns of a table: split, regimen, cause, horizon and time.

    A column is printed only when some row has a value for it, as ``to_frame()`` does.
    """
    columns: list[str] = []
    keys: list[str] = []
    if draws and any(getattr(row, "draw", None) is not None for row in rows):
        columns.append("split")
        keys.append("draw")
    if regimens:
        columns.append("regimen")
        keys.append("regimen")
    for key in ("cause", "horizon"):
        if any(getattr(row, key) is not None for row in rows):
            columns.append(key)
            keys.append(key)
    columns.append("time")
    keys.append("time")
    cells = []
    for row in rows:
        values = []
        for key in keys:
            value = getattr(row, key)
            if key == "regimen" and value is None:
                values.append(Cell(vocabulary.ALL_REGIMENS, vocabulary.ALL_REGIMENS))
            elif key in ("regimen", "cause"):
                # Words read left-aligned, so a missing cause is a word too.
                text = vocabulary.MISSING if value is None else str(value)
                values.append(Cell(text, text))
            else:
                values.append(_text(value))
        cells.append(tuple(values))
    return tuple(columns), cells


# ---------------------------------------------------------------------- overlap over time


def _stage_node(row: LongitudinalStageRow) -> str:
    return node_words(row.regimen, row.time, cause=row.cause, horizon=row.horizon)


def _capped_rows(check: LongitudinalDiagnostics) -> list[LongitudinalStageRow]:
    return [row for row in check.rows if row.share_truncated >= _EVERY_ROW]


def _unconverged(check: LongitudinalDiagnostics) -> list[LongitudinalStageRow]:
    return [row for row in check.rows if row.converged is False]


def _other_splits(check: LongitudinalDiagnostics, flags: Sequence[bool]) -> list[str]:
    return [
        vocabulary.NODE_SPLIT.format(split=index)
        for index, flag in enumerate(flags, 1)
        if flag and index != check.reported_repeat
    ]


def _capped_splits(check: LongitudinalDiagnostics) -> list[str]:
    return _other_splits(
        check,
        [math.isfinite(share) and share >= _EVERY_ROW for share in check.draw_max_share_truncated],
    )


def _unconverged_splits(check: LongitudinalDiagnostics) -> list[str]:
    return _other_splits(check, [not converged for converged in check.draw_converged])


def _stage_attention(check: LongitudinalDiagnostics) -> bool:
    return bool(
        _capped_rows(check)
        or _unconverged(check)
        or _capped_splits(check)
        or _unconverged_splits(check)
    )


def _stage_result(check: LongitudinalDiagnostics) -> str:
    return vocabulary.CHECK_RESULTS["warning" if _stage_attention(check) else "descriptive"]


def _assignment_cell(value: float | str) -> Cell:
    if isinstance(value, str):
        return Cell(value, value)
    return _share(value)


def _stage_table(check: LongitudinalDiagnostics) -> SummaryTable:
    columns, identity = _identity(check.rows)
    trimmed = any(row.share_trimmed is not None for row in check.rows)
    converged = bool(_unconverged(check))
    labels = [*columns, *vocabulary.STAGE_COLUMNS]
    if trimmed:
        labels.append(vocabulary.STAGE_TRIMMED)
    if converged:
        labels.append(vocabulary.STAGE_CONVERGED)
    labels.append(vocabulary.STAGE_ASSIGNED)
    rows = []
    for row, cells in zip(check.rows, identity, strict=True):
        values = [
            *cells,
            Cell(row.n_followed, count_text(row.n_followed)),
            _number(row.max_weight, 2),
            _number(row.effective_n, 1),
            _share(row.share_truncated),
        ]
        if trimmed:
            values.append(_share(row.share_trimmed))
        if converged:
            values.append(
                Cell(
                    row.converged,
                    vocabulary.MISSING
                    if row.converged is None
                    else vocabulary.CONVERGED_WORDS[row.converged],
                )
            )
        values.append(_assignment_cell(row.assignment))
        rows.append(tuple(values))
    return SummaryTable(tuple(labels), tuple(rows))


def _thinnest(check: LongitudinalDiagnostics) -> LongitudinalStageRow | None:
    rows = [row for row in check.rows if math.isfinite(row.effective_n)]
    if not rows:
        return None
    return min(rows, key=lambda row: row.effective_n)


def _stage_reading(check: LongitudinalDiagnostics) -> str:
    row = _thinnest(check)
    if row is None:
        return vocabulary.STAGE_READING_NONE
    return vocabulary.STAGE_READING.format(
        node=_stage_node(row),
        effective=f"{row.effective_n:.1f}",
        rows=count_text(row.n_followed),
    )


def _stage_notes(check: LongitudinalDiagnostics) -> list[Note]:
    texts = vocabulary.NOTE_TEXTS
    notes = [note("cumulative_weights", texts["cumulative_weights"])]
    sentence = (
        vocabulary.ASSIGNED_SHARES
        if check.categorical
        else vocabulary.ASSIGNED_SHARE.format(zero=0, one=1)
    )
    notes.append(note("assigned", texts["assigned"].format(sentence=sentence)))
    policy = [_stage_node(row) for row in check.rows if row.node_kind == "policy"]
    if policy:
        notes.append(note("policy_nodes", texts["policy_nodes"].format(nodes=_joined(policy))))
    if any(row.node_kind == "tilt" for row in check.rows):
        notes.append(note("tilt_nodes", texts["tilt_nodes"]))
    if any(row.share_trimmed is not None for row in check.rows):
        notes.append(note("trimmed", texts["trimmed"]))
    capped = [_stage_node(row) for row in _capped_rows(check)]
    capped_splits = _capped_splits(check)
    if capped or capped_splits:
        sentences = []
        if capped:
            sentences.append(vocabulary.CAPPED_NODES.format(nodes=_joined(capped)))
        if capped_splits:
            sentences.append(vocabulary.CAPPED_OTHER_SPLITS.format(splits=_joined(capped_splits)))
        notes.append(
            note("capped_nodes", texts["capped_nodes"].format(sentences=" ".join(sentences)))
        )
    unconverged = [_stage_node(row) for row in _unconverged(check)]
    unconverged_splits = _unconverged_splits(check)
    if unconverged or unconverged_splits:
        sentences = []
        if unconverged:
            sentences.append(vocabulary.NOT_CONVERGED.format(nodes=_joined(unconverged)))
        if unconverged_splits:
            sentences.append(
                vocabulary.NOT_CONVERGED_OTHER_SPLITS.format(splits=_joined(unconverged_splits))
            )
        notes.append(
            note("not_converged", texts["not_converged"].format(sentences=" ".join(sentences)))
        )
    if check.n_repeats > 1:
        notes.append(
            note(
                "first_split",
                texts["first_split"].format(split=check.reported_repeat, total=check.n_repeats),
            )
        )
    return notes


def _stage_details(check: LongitudinalDiagnostics) -> list[SummarySection]:
    lines = []
    if check.rows and all(row.sdr is not None for row in check.rows):
        for row in check.rows:
            sdr = row.sdr
            assert sdr is not None
            lines.append(
                vocabulary.SDR_LINE.format(
                    node=_stage_node(row),
                    outside=_share(sdr.target_outside).text,
                    zero=0,
                    one=1,
                    low=_number(sdr.target_min, 3).text,
                    high=_number(sdr.target_max, 3).text,
                    predictions=_share(sdr.prediction_outside).text,
                    bound=_share(sdr.bound_binding).text,
                )
            )
        if all(row.ipsi is not None for row in check.rows):
            for row in check.rows:
                ipsi = row.ipsi
                assert ipsi is not None
                lines.append(
                    vocabulary.IPSI_LINE.format(
                        node=_stage_node(row),
                        low=_number(ipsi.tilt_factor_min, 3).text,
                        high=_number(ipsi.tilt_factor_max, 3).text,
                        share=_share(ipsi.mechanism_share).text,
                        degenerate=_share(ipsi.propensity_degenerate).text,
                        zero=0,
                        one=1,
                    )
                )
    if not lines:
        lines.append(vocabulary.OVERLAP_NOTHING_MORE)
    return [SummarySection(vocabulary.SECTION_DETAILS, tuple(lines))]


def _stage_header(check: LongitudinalDiagnostics) -> list[tuple[str, str]]:
    regimens = {row.regimen for row in check.rows}
    times = {row.time for row in check.rows}
    header = [
        (vocabulary.REGIMENS, count_text(len(regimens))),
        (vocabulary.TIME_POINTS, count_text(len(times))),
    ]
    if check.n_repeats > 1:
        header.append(
            (
                vocabulary.SPLIT_SHOWN,
                vocabulary.SPLIT_OF.format(split=check.reported_repeat, total=check.n_repeats),
            )
        )
    header.append((vocabulary.RESULT, _stage_result(check)))
    return header


def stage_summary(check: LongitudinalDiagnostics, detail: Detail) -> Summary:
    """The printed summary of the overlap-over-time check.

    Parameters
    ----------
    check : LongitudinalDiagnostics
        The check.
    detail : {"short", "full"}
        ``"full"`` adds the SDR and IPSI measures of each row.

    Returns
    -------
    Summary
        The summary.
    """
    return Summary.build(
        title=vocabulary.STAGE_TITLE,
        header=_stage_header(check),
        tables=[_stage_table(check)],
        notes=_stage_notes(check),
        reading=_stage_reading(check),
        sections=_stage_details(check) if detail == "full" else (),
    )


def stage_repr(check: LongitudinalDiagnostics) -> str:
    """A one-line repr of the overlap-over-time check.

    Parameters
    ----------
    check : LongitudinalDiagnostics
        The check.

    Returns
    -------
    str
        Such as ``LongitudinalDiagnostics(4 rows, no pass or fail)``.
    """
    return short_repr(
        "LongitudinalDiagnostics",
        [("", f"{len(check.rows)} rows"), ("", _stage_result(check))],
    )


# ------------------------------------------------------------ equations solved over time


def _score_node(row: LongitudinalScoreRow) -> str:
    return node_words(
        row.regimen,
        row.time,
        cause=row.cause,
        horizon=row.horizon,
        component=row.component,
        draw=row.draw,
    )


def _score_result(check: LongitudinalScoreDiagnostics) -> str:
    if not check.rows:
        return vocabulary.CHECK_RESULTS["descriptive"]
    return vocabulary.CHECK_RESULTS["passed" if check.passed else "failed"]


def _score_identity(
    check: LongitudinalScoreDiagnostics,
) -> tuple[tuple[str, ...], list[tuple[Cell, ...]]]:
    pooled = any(row.component is not None for row in check.rows)
    regimens = any(row.regimen is not None for row in check.rows)
    columns, identity = _identity(check.rows, regimens=regimens or not pooled, draws=True)
    if pooled:
        columns = (*columns, "component")
        identity = [
            (*cells, _text(row.component)) for row, cells in zip(check.rows, identity, strict=True)
        ]
    return columns, identity


def _score_tables(check: LongitudinalScoreDiagnostics) -> list[SummaryTable]:
    if not check.rows:
        return [SummaryTable(("equations",), ((Cell(None, "none reported"),),))]
    columns, identity = _score_identity(check)
    rows = []
    for row, cells in zip(check.rows, identity, strict=True):
        error = abs(row.relative_score)
        result = vocabulary.CHECK_RESULTS["passed" if row.passed else "failed"]
        rows.append(
            (
                *cells,
                Cell(_finite(error), format_negligible(error, check.tolerance, digits=3)),
                Cell(result, result),
            )
        )
    return [SummaryTable((*columns, *vocabulary.EQUATION_COLUMNS), tuple(rows))]


def _score_details(check: LongitudinalScoreDiagnostics) -> list[SummarySection]:
    lines = [vocabulary.EQUATION_SCALE]
    lines.extend(
        vocabulary.STEPS_LINE.format(
            node=_score_node(row),
            steps=count_text(row.n_iter),
            word="step" if row.n_iter == 1 else "steps",
            score=f"{row.score:.3e}",
        )
        for row in check.rows
    )
    return [SummarySection(vocabulary.SECTION_DETAILS, tuple(lines))]


def _score_reading(check: LongitudinalScoreDiagnostics) -> str:
    if not check.rows:
        return vocabulary.EQUATION_READINGS["none"]
    failed = [row for row in check.rows if not row.passed]
    if not failed:
        return vocabulary.EQUATION_READINGS["passed"].format(
            spread=vocabulary.LONGITUDINAL_SPREAD_LABEL
        )
    return vocabulary.LONGITUDINAL_EQUATION_FAILED.format(
        count=f"{len(failed):,}",
        total=f"{len(check.rows):,}",
        verb="was" if len(failed) == 1 else "were",
        names=_joined([_score_node(row) for row in failed]),
    )


def _score_notes(check: LongitudinalScoreDiagnostics) -> list[Note]:
    texts = vocabulary.NOTE_TEXTS
    notes = [note("relative_error", texts["relative_error"])]
    sentences = []
    for row in check.rows:
        if row.converged:
            continue
        name = _score_node(row)
        if row.failure:
            reason = vocabulary.STEP_FAILURES.get(row.failure, row.failure.replace("_", " "))
            sentences.append(vocabulary.ROW_STOPPED.format(name=name, reason=reason))
        else:
            sentences.append(vocabulary.NODE_NOT_SETTLED.format(name=name))
    if sentences:
        notes.append(
            note(
                "row_not_converged",
                texts["row_not_converged"].format(sentences=" ".join(sentences)),
            )
        )
    if any(row.component is not None for row in check.rows):
        notes.append(note("working_model_rows", texts["working_model_rows"]))
    if any(row.draw is not None for row in check.rows):
        notes.append(note("every_split", texts["every_split"]))
    return notes


def score_summary(check: LongitudinalScoreDiagnostics, detail: Detail) -> Summary:
    """The printed summary of the equations-solved-over-time check.

    Parameters
    ----------
    check : LongitudinalScoreDiagnostics
        The check.
    detail : {"short", "full"}
        ``"full"`` adds the steps each correction took and each equation's raw value.

    Returns
    -------
    Summary
        The summary.
    """
    header = [
        (vocabulary.EQUATIONS, count_text(len(check.rows))),
        (vocabulary.ALLOWED_ERROR, vocabulary.ALLOWED_SHARE.format(tolerance=check.tolerance)),
        (vocabulary.RESULT, _score_result(check)),
    ]
    return Summary.build(
        title=vocabulary.LONGITUDINAL_EQUATIONS_TITLE,
        header=header,
        tables=_score_tables(check),
        notes=_score_notes(check),
        reading=_score_reading(check),
        sections=_score_details(check) if detail == "full" else (),
    )


def score_repr(check: LongitudinalScoreDiagnostics) -> str:
    """A one-line repr of the equations-solved-over-time check.

    Parameters
    ----------
    check : LongitudinalScoreDiagnostics
        The check.

    Returns
    -------
    str
        Such as ``LongitudinalScoreDiagnostics(4 rows, passed)``.
    """
    return short_repr(
        "LongitudinalScoreDiagnostics",
        [("", f"{len(check.rows)} rows"), ("", _score_result(check))],
    )


# ------------------------------------------------------------------- models over time


def _role_words(role: str) -> str:
    return vocabulary.LONGITUDINAL_ROLES.get(role, role.replace("_", " "))


def _models_table(rows: Sequence[LongitudinalNuisanceRow]) -> SummaryTable:
    columns, identity = _identity(rows)
    printed = []
    for row, cells in zip(rows, identity, strict=True):
        measure = vocabulary.LOSS_SHORT.get(row.loss_name, row.loss_name.replace("_", " "))
        printed.append(
            (
                *cells,
                Cell(row.role, _role_words(row.role)),
                Cell(row.n, count_text(row.n)),
                _number(row.loss, 4),
                Cell(row.loss_name, measure),
            )
        )
    return SummaryTable((*columns, *vocabulary.MODEL_COLUMNS), tuple(printed))


def _metric_line(row: LongitudinalNuisanceRow) -> str:
    metrics = []
    if row.model is not None:
        for key, name in vocabulary.METRIC_NAMES.items():
            value = _finite(row.model.metrics.get(key))
            if value is not None:
                metrics.append(f"{name} {value:.3f}")
    return vocabulary.METRIC_LINE.format(
        node=node_words(row.regimen, row.time, cause=row.cause, horizon=row.horizon),
        model=_role_words(row.role),
        metrics=", ".join(metrics) if metrics else vocabulary.METRIC_NONE,
    )


def _spread_table(check: LongitudinalNuisanceDiagnostics) -> SummaryTable | None:
    if not check.repeat_spread:
        return None
    rows = tuple(
        (
            Cell(row.estimand, row.estimand),
            Cell(row.n_repeats, count_text(row.n_repeats)),
            _number(row.standard_deviation, 4),
            _number(row.reported_standard_error, 4),
            _number(row.ratio_to_standard_error, 2),
        )
        for row in check.repeat_spread
    )
    columns = (
        "estimate",
        "splits",
        "SD between splits",
        vocabulary.LONGITUDINAL_SPREAD_LABEL,
        "ratio",
    )
    return SummaryTable(columns, rows, title=vocabulary.SPREAD_TABLE)


def _omission_sentence(omission: Any) -> str:
    model = _role_words(omission.role)
    if omission.regimen is not None:
        model = f"{model} ({omission.regimen})"
    folds = (
        vocabulary.MODEL_OMITTED_FOLDS.format(folds=_joined([str(f) for f in omission.folds]))
        if omission.folds
        else ""
    )
    reason = without_roadmap(omission.reason)
    if omission.time is None:
        return vocabulary.MODEL_OMITTED_ALL.format(model=model, folds=folds, reason=reason)
    return vocabulary.MODEL_OMITTED_AT.format(
        model=model, time=omission.time, folds=folds, reason=reason
    )


def _models_notes(check: LongitudinalNuisanceDiagnostics) -> list[Note]:
    texts = vocabulary.NOTE_TEXTS
    notes = []
    if any(row.role == "pseudo_outcome" for row in check.rows):
        notes.append(note("iterated_outcome", texts["iterated_outcome"]))
    if check.omissions:
        sentences = " ".join(_omission_sentence(omission) for omission in check.omissions)
        notes.append(note("model_omitted", texts["model_omitted"].format(sentences=sentences)))
    if check.repeat_spread:
        notes.append(
            note(
                "split_ratio",
                texts["split_ratio"].format(label=vocabulary.LONGITUDINAL_SPREAD_LABEL),
            )
        )
    return notes


def _models_details(check: LongitudinalNuisanceDiagnostics) -> list[SummarySection]:
    lines = [_metric_line(row) for row in check.rows]
    if check.n_repeats > 1 and check.repeat_spread_omission is not None:
        lines.append(
            vocabulary.MODEL_OMISSIONS["spread"].format(
                reason=without_roadmap(check.repeat_spread_omission)
            )
        )
    if not lines:
        lines.append(vocabulary.OVERLAP_NOTHING_MORE)
    return [SummarySection(vocabulary.SECTION_DETAILS, tuple(lines))]


def models_summary(check: LongitudinalNuisanceDiagnostics, detail: Detail) -> Summary:
    """The printed summary of the models-over-time check.

    Parameters
    ----------
    check : LongitudinalNuisanceDiagnostics
        The check.
    detail : {"short", "full"}
        ``"full"`` adds each model's AUC, calibration slope or R2.

    Returns
    -------
    Summary
        The summary.
    """
    evaluations = {row.evaluation for row in check.rows}
    header = [(vocabulary.MODELS, count_text(len(check.rows)))]
    if len(evaluations) == 1:
        header.append(
            (vocabulary.PREDICTIONS, vocabulary.EVALUATION_WORDS[next(iter(evaluations))])
        )
    if check.n_repeats > 1:
        header.append(
            (
                vocabulary.SPLIT_SHOWN,
                vocabulary.SPLIT_OF.format(split=check.reported_repeat, total=check.n_repeats),
            )
        )
    header.append((vocabulary.RESULT, vocabulary.CHECK_RESULTS["descriptive"]))
    tables: list[SummaryTable] = []
    if check.rows:
        tables.append(_models_table(check.rows))
    else:
        tables.append(SummaryTable(("models",), ((Cell(None, "none reported"),),)))
    spread = _spread_table(check)
    if spread is not None:
        tables.append(spread)
    return Summary.build(
        title=vocabulary.LONGITUDINAL_MODELS_TITLE,
        header=header,
        tables=tables,
        notes=_models_notes(check),
        reading=vocabulary.LONGITUDINAL_MODELS_READING,
        sections=_models_details(check) if detail == "full" else (),
    )


def models_repr(check: LongitudinalNuisanceDiagnostics) -> str:
    """A one-line repr of the models-over-time check.

    Parameters
    ----------
    check : LongitudinalNuisanceDiagnostics
        The check.

    Returns
    -------
    str
        Such as ``LongitudinalNuisanceDiagnostics(8 rows, no pass or fail)``.
    """
    return short_repr(
        "LongitudinalNuisanceDiagnostics",
        [("", f"{len(check.rows)} rows"), ("", vocabulary.CHECK_RESULTS["descriptive"])],
    )
