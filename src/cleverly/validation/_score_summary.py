"""The printed summaries of the equation check and the DR-TMLE correction check (X89 T3).

Each prints like every other check: a header of facts with its ``Result``, a table with
one row per equation or correction, one reading sentence, and notes for what the table has
no column for. The numbers are read from the check; nothing here decides a verdict. The
residues and thresholds keep the ``format_negligible`` and ``format_within`` rules, so a
value far below its bar prints as zero on every machine and the frames keep the raw value.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from .._inference_status import is_labelled, status_record
from .._summary import (
    Cell,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    note,
    short_repr,
    vocabulary,
)
from ..inference.influence import spread_name
from ..utils.text import format_negligible, format_within

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .drtmle import CorrectionCheck, CorrectionRow
    from .score import ScoreCheck, ScoreCheckRow

__all__ = [
    "correction_repr",
    "correction_summary",
    "equation_words",
    "score_repr",
    "score_summary",
]

#: The Hessian condition number above which a row's step size is barely determined.
CONDITION_LIMIT = 1e8

#: The smallest stopping-rule remainder a note prints as a number (the estimate summary's).
_SMALLEST_REMAINDER = 0.001

_ROW_NAME = re.compile(
    r"^(?P<group>[^\s\[(]+)"
    r"(?: \((?P<part>[^)]+)\))?"
    r"(?:\[(?!draw )(?P<arm>[^\]]+)\])?"
    r"(?P<identity> identity)?"
    r"(?:\[draw (?P<draw>\d+)\])?$"
)


def _joined(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def equation_words(name: str) -> str:
    """A row name of the equation check in plain words.

    Parameters
    ----------
    name : str
        The row's ``name``, for example ``"mean (D*_g)[1][draw 0]"``.

    Returns
    -------
    str
        For example ``"outcome means (treatment correction, arm 1, split 1)"``. A name
        that is not a targeted group's, such as an estimand alias, prints as written.
    """
    match = _ROW_NAME.match(name)
    if match is None:
        return name
    group, part, arm, draw = match.group("group", "part", "arm", "draw")
    words = vocabulary.EQUATION_GROUPS.get(group, group)
    qualifiers = []
    if part is not None:
        qualifiers.append(vocabulary.EQUATION_PARTS.get(part, part))
    if arm is not None:
        qualifiers.append(f"arm {arm}")
    if draw is not None:
        qualifiers.append(f"split {int(draw) + 1}")
    return f"{words} ({', '.join(qualifiers)})" if qualifiers else words


def _row_words(row: ScoreCheckRow) -> str:
    """Targeted groups and recomputed corrections are renamed; estimand aliases are not."""
    if row.kind in ("fluctuation", "correction", "identity") or row.method == "recomputed":
        return equation_words(row.name)
    return row.name


def _spread(check: ScoreCheck) -> str:
    """What the fit reports as its spread, in the words of its estimate summary."""
    return spread_name("reported standard error", check.inference).removeprefix("reported ")


def _row_result(row: ScoreCheckRow) -> str:
    if not math.isfinite(row.threshold):
        return vocabulary.CHECK_RESULTS["descriptive"]
    return vocabulary.CHECK_RESULTS["passed" if row.passed else "failed"]


def _number_cells(row: ScoreCheckRow) -> tuple[Cell, ...]:
    """The error, allowed, ratio and result cells of a row."""
    error = abs(row.score)
    allowed = f"{row.threshold:.3e}" if math.isfinite(row.threshold) else vocabulary.MISSING
    return (
        Cell(_finite(error), format_negligible(error, row.threshold, digits=3)),
        Cell(_finite(row.threshold), allowed),
        Cell(_finite(row.ratio), format_negligible(row.ratio, 1.0, digits=2)),
        Cell(_row_result(row), _row_result(row)),
    )


_NUMBER_COLUMNS = ("error", "allowed", "ratio", "result")


def _is_correction(row: ScoreCheckRow) -> bool:
    return row.kind in ("correction", "identity") or row.method == "recomputed"


def _correction_cells(row: ScoreCheckRow, splits: bool) -> tuple[Cell, ...]:
    match = _ROW_NAME.match(row.name)
    part, arm, draw = match.group("part", "arm", "draw") if match else (None, None, None)
    # A row built by hand may name no correction; it prints as its plain name.
    words = (
        vocabulary.EQUATION_PARTS.get(part, part).removesuffix(" correction")
        if part is not None
        else equation_words(row.name)
    )
    cells = [Cell(row.name, words), Cell(arm, arm or vocabulary.MISSING)]
    if splits:
        cells.append(
            Cell(
                None if draw is None else int(draw),
                vocabulary.MISSING if draw is None else f"{int(draw) + 1}",
            )
        )
    cells.append(Cell(row.kind, vocabulary.CORRECTION_ROW_KINDS[row.kind]))
    return (*cells, *_number_cells(row))


def _score_tables(check: ScoreCheck) -> list[SummaryTable]:
    steps = [row for row in check.rows if row.kind == "fluctuation"]
    corrections = [row for row in check.rows if _is_correction(row)]
    estimates = [row for row in check.rows if row.kind != "fluctuation" and not _is_correction(row)]
    tables = []
    if steps:
        tables.append(
            SummaryTable(
                columns=("equation", *_NUMBER_COLUMNS),
                rows=tuple(
                    (Cell(row.name, equation_words(row.name)), *_number_cells(row)) for row in steps
                ),
                title=vocabulary.STEP_EQUATIONS_TITLE,
            )
        )
    if corrections:
        splits = any("[draw " in row.name for row in corrections)
        tables.append(
            SummaryTable(
                columns=(
                    "correction",
                    "arm",
                    *(("split",) if splits else ()),
                    "check",
                    *_NUMBER_COLUMNS,
                ),
                rows=tuple(_correction_cells(row, splits) for row in corrections),
                title=vocabulary.CORRECTIONS_TITLE,
            )
        )
    if estimates:
        tables.append(
            SummaryTable(
                columns=("estimate", *_NUMBER_COLUMNS),
                rows=tuple((Cell(row.name, row.name), *_number_cells(row)) for row in estimates),
                title=vocabulary.ESTIMATE_EQUATIONS_TITLE,
            )
        )
    return tables


def _score_reading(check: ScoreCheck) -> str:
    spread = _spread(check)
    failures = check.failures
    if not failures:
        key = "passed_declared" if check.declared else "passed"
        return vocabulary.EQUATION_READINGS[key].format(spread=spread)
    sentences = []
    identity = check.identity_failures
    if identity:
        sentences.append(
            vocabulary.EQUATION_READINGS["failed_identity"].format(
                names=_joined([_row_words(row) for row in identity]), spread=spread
            )
        )
    ruled = check.rule_failures
    if ruled:
        sentences.append(
            vocabulary.EQUATION_READINGS["failed_rule"].format(
                names=_joined([_row_words(row) for row in ruled]), spread=spread
            )
        )
    others = [row for row in failures if row not in ruled and row not in identity]
    if others:
        sentences.append(
            vocabulary.EQUATION_READINGS["failed"].format(
                count=f"{len(others):,}",
                total=f"{len(check.rows):,}",
                verb="was" if len(others) == 1 else "were",
                its="its" if len(others) == 1 else "their",
                names=_joined([_row_words(row) for row in others]),
                spread=spread,
            )
        )
    return " ".join(sentences)


def _stopping_note(check: ScoreCheck) -> Note | None:
    if not check.declared:
        return None
    rows = [row for row in check.rows if math.isfinite(row.residual_se)]
    if not rows:
        return note("stopping_rule", vocabulary.NOTE_TEXTS["stopping_rule_plain"])
    worst = max(rows, key=lambda row: row.residual_se)
    # A lower bound, so it is rounded down, as the estimate summary rounds it.
    bound = math.floor(worst.coverage_bound * 10000) / 100
    b = worst.residual_se
    return note(
        "stopping_rule",
        vocabulary.NOTE_TEXTS["stopping_rule"].format(
            b=f"less than {_SMALLEST_REMAINDER:g}" if b < _SMALLEST_REMAINDER else f"{b:.3g}",
            name=worst.name,
            level=f"{(1.0 - check.alpha_sig) * 100:g}%",
            bound=f"{bound:.2f}%",
        ),
    )


def _stopped(row: ScoreCheckRow) -> list[str]:
    """The sentences of the ``row_not_converged`` note for one row."""
    name = _row_words(row)
    sentences = []
    if row.failure and row.kind == "fluctuation":
        reason = vocabulary.STEP_FAILURES.get(row.failure, row.failure.replace("_", " "))
        sentences.append(vocabulary.ROW_STOPPED.format(name=name, reason=reason))
    if row.folds_converged is not None:
        good, total = row.folds_converged
        if good < total:
            sentences.append(
                vocabulary.ROW_FOLDS.format(name=name, failed=f"{total - good:,}", total=total)
            )
    if math.isfinite(row.hessian_condition) and row.hessian_condition > CONDITION_LIMIT:
        sentences.append(
            vocabulary.ROW_CONDITION.format(name=name, condition=f"{row.hessian_condition:.2e}")
        )
    return sentences


def _held_note(check: ScoreCheck) -> Note | None:
    """Why each recomputed diagnostic row is held to nothing, grouped by reason."""
    unsolved: list[str] = []
    ruled: list[str] = []
    for row in check.rows:
        if row.kind != "diagnostic" or row.method != "recomputed":
            continue
        (ruled if "stopping rule" in row.failure else unsolved).append(_row_words(row))
    reasons = []
    if unsolved:
        reasons.append(
            vocabulary.HELD_UNSOLVED.format(
                names=_joined(unsolved), verb="is" if len(unsolved) == 1 else "are"
            )
        )
    if ruled:
        reasons.append(
            vocabulary.HELD_BY_RULE.format(names=_joined(ruled), s="s" if len(ruled) == 1 else "")
        )
    if not reasons:
        return None
    text = "; ".join(reasons)
    return note(
        "held_to_nothing",
        vocabulary.NOTE_TEXTS["held_to_nothing"].format(reasons=text[0].lower() + text[1:] + "."),
    )


def _score_notes(check: ScoreCheck) -> list[Note]:
    notes: list[Note] = []
    stopping = _stopping_note(check)
    if stopping is not None:
        notes.append(stopping)
    identity = check.identity_failures
    if identity:
        notes.append(
            note(
                "identity_failure",
                vocabulary.NOTE_TEXTS["identity_failure"].format(
                    count=f"{len(identity):,}",
                    verb="compares" if len(identity) == 1 else "compare",
                ),
            )
        )
    sentences = [sentence for row in check.rows for sentence in _stopped(row)]
    if sentences:
        notes.append(
            note(
                "row_not_converged",
                vocabulary.NOTE_TEXTS["row_not_converged"].format(sentences=" ".join(sentences)),
            )
        )
    if is_labelled(check.inference):
        reason = status_record(check.inference).plain_reason
        notes.append(
            note("equations_only", vocabulary.NOTE_TEXTS["equations_only"].format(reason=reason))
        )
    elif check.corrected:
        notes.append(note("dr_curve", vocabulary.NOTE_TEXTS["dr_curve"]))
    if any(_is_correction(row) for row in check.rows):
        notes.append(note("correction_rows", vocabulary.NOTE_TEXTS["correction_rows"]))
    held = _held_note(check)
    if held is not None:
        notes.append(held)
    return notes


def _score_details(check: ScoreCheck) -> SummarySection:
    lines = []
    for row in check.rows:
        name = _row_words(row)
        if math.isfinite(row.score_initial):
            steps = f"{row.n_iter:,} step{'s' * (row.n_iter != 1)}"
            before = format_negligible(abs(row.score_initial), row.threshold, digits=3)
            after = format_negligible(abs(row.score), row.threshold, digits=3)
            lines.append(
                f"{name}: error {before} before the correction step and {after} after it, "
                f"in {steps}."
            )
        if math.isfinite(row.residual_se):
            bound = (
                f"; the declared rule bounds it at {row.rule_bound_se:.3g}"
                if math.isfinite(row.rule_bound_se)
                else ""
            )
            lines.append(
                f"{name}: the remaining error is worth {row.residual_se:.3g} standard errors, "
                f"and a Wald interval keeps at least {row.coverage_bound:.4f} coverage{bound}."
            )
    if not lines:
        lines.append(
            "No equation has an error recorded before the correction step; each row is "
            "computed from the fit's final state."
        )
    return SummarySection(vocabulary.SECTION_DETAILS, tuple(lines))


def _allowed(check: ScoreCheck) -> str:
    if check.declared:
        return vocabulary.ALLOWED_DECLARED.format(rule=check.stopping)
    return vocabulary.ALLOWED_RELATIVE.format(tolerance=check.tolerance)


def _check_result(passed: bool) -> str:
    return vocabulary.CHECK_RESULTS["passed" if passed else "failed"]


def score_summary(check: ScoreCheck, detail: str) -> Summary:
    """The summary of an equation check.

    Parameters
    ----------
    check : ScoreCheck
        The check.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    header = [
        (vocabulary.RESULT, _check_result(check.passed)),
        (vocabulary.EQUATIONS, f"{len(check.rows):,}"),
        (vocabulary.ROWS, f"{check.n:,}"),
        (vocabulary.ALLOWED_ERROR, _allowed(check)),
    ]
    return Summary.build(
        title=vocabulary.EQUATION_TITLE,
        header=header,
        header_split=3,
        tables=_score_tables(check),
        reading=_score_reading(check),
        notes=_score_notes(check),
        sections=[_score_details(check)] if detail == "full" else (),
    )


def score_repr(check: ScoreCheck) -> str:
    """The one-line repr of an equation check.

    Parameters
    ----------
    check : ScoreCheck
        The check.

    Returns
    -------
    str
        For example ``ScoreCheck(passed, 8 equations, worst ratio 0.00)``.
    """
    ratios = [row.ratio for row in check.rows if math.isfinite(row.ratio)]
    worst = max(ratios) if ratios else float("nan")
    return short_repr(
        "ScoreCheck",
        [
            ("", _check_result(check.passed)),
            ("", f"{len(check.rows)} equation{'s' * (len(check.rows) != 1)}"),
            ("worst ratio", format_negligible(worst, 1.0, digits=2)),
        ],
    )


# ------------------------------------------------------------------ DR-TMLE corrections


def _correction_name(row: CorrectionRow) -> str:
    words = vocabulary.EQUATION_PARTS.get(row.equation, row.equation)
    return words.removesuffix(" correction")


def _correction_table(check: CorrectionCheck) -> SummaryTable:
    if not check.rows:
        return SummaryTable(
            columns=("corrections",), rows=((Cell("none reported", "none reported"),),)
        )
    splits = len({row.draw for row in check.rows}) > 1
    rows = []
    for row in check.rows:
        bar = check.threshold if row.solved else 0.0
        lead = [Cell(row.equation, _correction_name(row)), Cell(row.label, row.label)]
        if splits:
            lead.append(Cell(row.draw, f"{row.draw + 1}"))
        cells = (
            *lead,
            Cell(_finite(row.stored), format_negligible(row.stored, bar, digits=3)),
            Cell(_finite(row.reported), format_negligible(row.reported, bar, digits=3)),
            Cell(
                _finite(row.residual),
                format_within(row.residual, check.identity_threshold, digits=3),
            ),
            Cell(
                _finite(row.clip_bias),
                format_negligible(row.clip_bias, check.identity_threshold, digits=3),
            ),
        )
        rows.append(cells)
    columns = (
        "correction",
        "arm",
        *(("split",) if splits else ()),
        "solved",
        "in curve",
        "difference",
        "cap bias",
    )
    return SummaryTable(columns=columns, rows=tuple(rows))


def _correction_reading(check: CorrectionCheck) -> str:
    if not check.rows:
        return vocabulary.CORRECTION_READINGS["none"]
    identity, correction = check.identity_failures(), check.correction_failures()
    if not identity and not correction:
        return vocabulary.CORRECTION_READINGS["passed"]
    sentences = []
    if identity:
        sentences.append(
            vocabulary.CORRECTION_READINGS["identity"].format(
                count=f"{len(identity):,}", checks="check" if len(identity) == 1 else "checks"
            )
        )
    if correction:
        sentences.append(
            vocabulary.CORRECTION_READINGS["correction"].format(
                count=f"{len(correction):,}",
                corrections="correction is" if len(correction) == 1 else "corrections are",
            )
        )
    return " ".join(sentences)


def _correction_notes(check: CorrectionCheck) -> list[Note]:
    if not check.rows:
        return []
    notes = [
        note(
            "correction_scale",
            vocabulary.NOTE_TEXTS["correction_scale"].format(
                scale=""
                if check.scale == 1.0
                else vocabulary.RESCALED.format(scale=f"{check.scale:.4g}"),
                threshold=f"{check.threshold:.2e}",
                identity=f"{check.identity_threshold:.2e}",
            ),
        )
    ]
    active = check.truncations_active
    if active:
        which = _joined([vocabulary.CAP_NAMES.get(name, name) for name in active])
        notes.append(note("cap_active", vocabulary.NOTE_TEXTS["cap_active"].format(which=which)))
    else:
        code = "theorem_cross_fitted" if check.cross_fitted else "theorem_in_sample"
        notes.append(note(code, vocabulary.NOTE_TEXTS[code]))
    unsolved = sorted(
        {_correction_name(row) for row in check.rows if not row.solved},
    )
    if unsolved:
        one = len(unsolved) == 1
        notes.append(
            note(
                "unsolved_guard",
                vocabulary.NOTE_TEXTS["unsolved_guard"].format(
                    equations=_joined(unsolved) + (" correction" if one else " corrections"),
                    verb="is" if one else "are",
                    pronoun="it" if one else "them",
                    rows="its rows are" if one else "their rows are",
                ),
            )
        )
    return notes


def _correction_details(check: CorrectionCheck) -> SummarySection:
    if not check.rows:
        return SummarySection(vocabulary.SECTION_DETAILS, ("This fit reports no corrections.",))
    lines = []
    draws = sorted({row.draw for row in check.rows})
    observation = check.has_observation_mechanism
    for draw in draws:
        rows = [row for row in check.rows if row.draw == draw]
        margin = min(row.margin for row in rows)
        reduced = min(row.gr1_margin for row in rows)
        recording = min(row.observation_margin for row in rows)
        line = vocabulary.CORRECTION_MARGINS.format(
            split=f"Split {draw + 1}: " if len(draws) > 1 else "",
            clipped=f"{max(row.clipped for row in rows):,}",
            n=f"{check.n:,}",
            margin=f"{margin:.2g}",
            observation=(f", the recording model within {recording:.2g}" if observation else ""),
            regression="regressions" if observation else "regression",
            reduced=f"{reduced:.2g}",
        )
        lines.append(line[0].upper() + line[1:])
    return SummarySection(vocabulary.SECTION_DETAILS, tuple(lines))


def correction_summary(check: CorrectionCheck, detail: str) -> Summary:
    """The summary of a DR-TMLE correction check.

    Parameters
    ----------
    check : CorrectionCheck
        The check.
    detail : {"short", "full"}
        How much to print.

    Returns
    -------
    Summary
        The summary.
    """
    result = (
        vocabulary.CHECK_RESULTS["descriptive"] if not check.rows else _check_result(check.passed)
    )
    judged = sum(1 for row in check.rows if row.solved)
    header = [
        (vocabulary.RESULT, result),
        (vocabulary.EQUATIONS, f"{judged:,}"),
        (vocabulary.ALLOWED_ERROR, vocabulary.ALLOWED_RELATIVE.format(tolerance=check.tolerance)),
        (vocabulary.ROWS, f"{check.n:,}"),
        (vocabulary.PREDICTIONS, vocabulary.PREDICTION_WORDS[bool(check.cross_fitted)]),
    ]
    return Summary.build(
        title=vocabulary.CORRECTION_TITLE,
        header=header,
        header_split=3,
        tables=[_correction_table(check)],
        reading=_correction_reading(check),
        notes=_correction_notes(check),
        sections=[_correction_details(check)] if detail == "full" else (),
    )


def correction_repr(check: CorrectionCheck) -> str:
    """The one-line repr of a correction check.

    Parameters
    ----------
    check : CorrectionCheck
        The check.

    Returns
    -------
    str
        For example ``CorrectionCheck(passed, 4 corrections, in-sample)``.
    """
    result = (
        vocabulary.CHECK_RESULTS["descriptive"] if not check.rows else _check_result(check.passed)
    )
    count = len(check.rows)
    return short_repr(
        "CorrectionCheck",
        [
            ("", result),
            ("", f"{count} correction{'s' * (count != 1)}"),
            ("", vocabulary.PREDICTION_WORDS[bool(check.cross_fitted)]),
        ],
    )
