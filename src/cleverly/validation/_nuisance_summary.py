"""The printed summary of the outcome and treatment model check (X89 T3).

One table per kind of model: the models that predict a probability (treatment, recording)
with their discrimination and calibration, and the models that predict a mean with their
fit. Each finding the check makes is a warning; the reading counts them. Super learner
weights and the spread between splits get their own tables when the fit has them. The
numbers are read from the check; nothing here decides a finding.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from .._inference_status import is_labelled, status_record, supplies_inference
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
from .._summary._render import count_text
from ..inference.influence import spread_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .nuisance import ModelFinding, NuisanceDiagnostics, NuisanceModelReport

__all__ = ["model_words", "nuisance_repr", "nuisance_summary", "report_repr"]

#: The smallest learner weight the weights table prints, as the old summary did.
_SMALLEST_WEIGHT = 1e-3

_NAME = re.compile(r"^(?P<stem>[^\[]+)(?:\[(?P<arm>.*)\])?$")


def model_words(name: str) -> str:
    """A model's report name in plain words.

    Parameters
    ----------
    name : str
        The report name, such as ``"propensity"`` or ``"propensity[placebo]"``.

    Returns
    -------
    str
        Such as ``"treatment model"`` or ``"treatment model (arm placebo)"``. An unknown
        name prints as written.
    """
    match = _NAME.match(name)
    if match is None:
        return name
    words = vocabulary.MODEL_NAMES.get(match["stem"], match["stem"])
    return words if match["arm"] is None else f"{words} (arm {match['arm']})"


def _number(value: Any, decimals: int) -> Cell:
    if value is None or not math.isfinite(float(value)):
        return Cell(None, vocabulary.MISSING)
    return Cell(float(value), f"{float(value):.{decimals}f}")


def _metric_table(
    models: list[NuisanceModelReport], columns: Any, title: str
) -> SummaryTable | None:
    if not models:
        return None
    rows = tuple(
        (
            Cell(model.name, model_words(model.name)),
            *(_number(model.metrics.get(key), 4) for key in columns),
        )
        for model in models
    )
    return SummaryTable(("model", *columns.values()), rows, title=title)


def _learner_table(check: NuisanceDiagnostics) -> SummaryTable | None:
    rows = tuple(
        (
            Cell(model.name, model_words(model.name)),
            Cell(learner, learner),
            Cell(float(weight), f"{weight:.3f}"),
        )
        for model in check.models
        for learner, weight in model.learner_weights.items()
        if weight > _SMALLEST_WEIGHT
    )
    if not rows:
        return None
    return SummaryTable(("model", "learner", "weight"), rows, title=vocabulary.LEARNER_TABLE)


def _spread_label(check: NuisanceDiagnostics) -> str:
    return vocabulary.SPREAD_LABELS[spread_name("std_err", check.inference)]


def _spread_table(check: NuisanceDiagnostics) -> SummaryTable | None:
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
    columns = ("estimate", "splits", "SD between splits", _spread_label(check), "ratio")
    return SummaryTable(columns, rows, title=vocabulary.SPREAD_TABLE)


def _tables(check: NuisanceDiagnostics) -> list[SummaryTable]:
    probability = [model for model in check.models if model.kind == "probability"]
    regression = [model for model in check.models if model.kind != "probability"]
    found = [
        _metric_table(probability, vocabulary.PROBABILITY_COLUMNS, vocabulary.PROBABILITY_TABLE),
        _metric_table(regression, vocabulary.REGRESSION_COLUMNS, vocabulary.REGRESSION_TABLE),
        _learner_table(check),
        _spread_table(check),
    ]
    tables = [table for table in found if table is not None]
    if not tables:
        tables = [SummaryTable(("models",), ((Cell(None, "none reported"),),))]
    return tables


def _finding_note(finding: ModelFinding) -> Note:
    texts = vocabulary.NOTE_TEXTS
    numbers = finding.numbers
    if finding.code in ("separation", "recording_separation"):
        text = texts[finding.code].format(auc=f"{numbers['auc']:.3f}")
    elif finding.code == "mean_only":
        text = texts["mean_only"].format(
            model=model_words(finding.model.name), share=f"{numbers['share']:.0%}"
        )
    elif finding.code == "weak_outcome_model":
        text = texts["weak_outcome_model"].format(r2=f"{numbers['r2']:.3f}")
    else:
        extra = ""
        if math.isfinite(numbers["auc"]):
            extra += f", AUC {numbers['auc']:.3f}"
        if math.isfinite(numbers["weight"]):
            extra += f", largest uncapped weight {numbers['weight']:.3g}"
        text = texts["calibration_finding"].format(
            model=model_words(finding.model.name),
            direction=vocabulary.CALIBRATION_DIRECTIONS[numbers["extreme"]],
            slope=f"{numbers['slope']:.2f}",
            level=numbers["level"],
            low=f"{numbers['low']:.2f}",
            high=f"{numbers['high']:.2f}",
            extra=extra,
            noise=vocabulary.CALIBRATION_NOISE if numbers["extreme"] else "",
        )
    return note(finding.code, text)


def _calibration_note(check: NuisanceDiagnostics) -> Note | None:
    from .nuisance import _level_label

    probability = [model for model in check.models if model.kind == "probability"]
    if not probability:
        return None
    if check.evaluation == "in_sample":
        sentences = [vocabulary.CALIBRATION_IN_SAMPLE.format(one=1)]
    else:
        sentences = [vocabulary.CALIBRATION_OUT_OF_FOLD.format(one=1)]
        tested = len(check._calibration_tested)
        if tested:
            sentences.append(
                vocabulary.CALIBRATION_RULE.format(
                    level=_level_label(tested),
                    zero=0,
                    one=1,
                    tested=tested,
                    models="model" if tested == 1 else "models",
                )
            )
    sentences.extend(
        vocabulary.CALIBRATION_MISSING.format(
            model=model_words(model.name), reason=without_roadmap(model.calibration_omission)
        )
        for model in probability
        if model.calibration_omission is not None
    )
    return note(
        "calibration", vocabulary.NOTE_TEXTS["calibration"].format(sentences=" ".join(sentences))
    )


def _selection_note(check: NuisanceDiagnostics) -> Note | None:
    from ..estimators.ctmle import CTMLESelection

    selection = check.selection
    if selection is None:
        return None
    if isinstance(selection, CTMLESelection):
        more = ""
        if selection.stopped_at is not None:
            more += vocabulary.SELECTION_STOPPED.format(stopped=selection.stopped_at + 1)
        if selection.final is not None:
            more += vocabulary.SELECTION_FINAL.format(final=selection.final + 1)
        sentence = vocabulary.SELECTION_SENTENCE.format(
            selected=selection.selected + 1,
            count=len(selection.path),
            estimand=selection.estimand,
            more=more,
        )
    else:
        sentence = without_roadmap(selection.describe()).rstrip(".") + "."
    return note("selection", vocabulary.NOTE_TEXTS["selection"].format(sentence=sentence))


def _notes(check: NuisanceDiagnostics) -> list[Note]:
    texts = vocabulary.NOTE_TEXTS
    notes = [_finding_note(finding) for finding in check._findings()]
    for model in check.weak_treatment_models:
        notes.append(
            note(
                "weak_treatment_model",
                texts["weak_treatment_model"].format(auc=f"{model.metrics['auc']:.3f}"),
            )
        )
    calibration = _calibration_note(check)
    if calibration is not None:
        notes.append(calibration)
    if check._working_model:
        notes.append(note("collaborative_model", texts["collaborative_model"]))
    if check.treatment_role == "declared_treatment_law":
        notes.append(note("declared_treatment", texts["declared_treatment"]))
    if is_labelled(check.inference):
        sentences = status_record(check.inference).plain_reason
        if not supplies_inference(check.inference):
            sentences = f"{vocabulary.NO_INTERVAL} {sentences}"
        notes.append(note("fit_status", texts["fit_status"].format(sentences=sentences)))
    selection = _selection_note(check)
    if selection is not None:
        notes.append(selection)
    if check.repeat_spread:
        notes.append(note("split_ratio", texts["split_ratio"].format(label=_spread_label(check))))
    return notes


def _reading(check: NuisanceDiagnostics) -> str:
    count = len(check.findings)
    if not count:
        return vocabulary.MODEL_READINGS["clear"]
    return vocabulary.MODEL_READINGS["findings"].format(
        count=count, findings="finding" if count == 1 else "findings"
    )


def _details(check: NuisanceDiagnostics) -> list[SummarySection]:
    omissions = vocabulary.MODEL_OMISSIONS
    lines = []
    if check.selection is None and check.selection_omission is not None:
        lines.append(
            omissions["selection"].format(reason=without_roadmap(check.selection_omission))
        )
    if check.outcome_omission is not None:
        lines.append(omissions["outcome"].format(reason=without_roadmap(check.outcome_omission)))
    if check.n_repeats > 1 and check.repeat_spread_omission is not None:
        lines.append(
            omissions["spread"].format(reason=without_roadmap(check.repeat_spread_omission))
        )
    if not lines:
        lines.append(vocabulary.MODEL_NOTHING_OMITTED)
    return [SummarySection(vocabulary.SECTION_DETAILS, tuple(lines))]


def nuisance_summary(check: NuisanceDiagnostics, detail: Detail) -> Summary:
    """The printed summary of the outcome and treatment model check.

    Parameters
    ----------
    check : NuisanceDiagnostics
        The check.
    detail : {"short", "full"}
        ``"full"`` adds what the fit could not report and why.

    Returns
    -------
    Summary
        The summary.
    """
    result = "warning" if check.findings else "descriptive"
    header = [
        (vocabulary.MODELS, count_text(len(check.models))),
        (vocabulary.PREDICTIONS, vocabulary.EVALUATION_WORDS[check.evaluation]),
    ]
    if check.n_repeats > 1:
        header.append(
            (
                vocabulary.SPLIT_SHOWN,
                vocabulary.SPLIT_OF.format(split=check.reported_repeat, total=check.n_repeats),
            )
        )
    header.append((vocabulary.RESULT, vocabulary.CHECK_RESULTS[result]))
    return Summary.build(
        title=vocabulary.MODEL_TITLE,
        header=header,
        tables=_tables(check),
        notes=_notes(check),
        reading=_reading(check),
        sections=_details(check) if detail == "full" else (),
    )


def nuisance_repr(check: NuisanceDiagnostics) -> str:
    """A one-line repr of the model check.

    Parameters
    ----------
    check : NuisanceDiagnostics
        The check.

    Returns
    -------
    str
        Such as ``NuisanceDiagnostics(2 models, out of fold, no pass or fail)``.
    """
    result = "warning" if check.findings else "descriptive"
    return short_repr(
        "NuisanceDiagnostics",
        [
            ("", f"{len(check.models)} models"),
            ("", vocabulary.EVALUATION_WORDS[check.evaluation]),
            ("", vocabulary.CHECK_RESULTS[result]),
        ],
    )


def report_repr(model: NuisanceModelReport) -> str:
    """A one-line repr of one model's report.

    Parameters
    ----------
    model : NuisanceModelReport
        The report.

    Returns
    -------
    str
        Such as ``NuisanceModelReport(treatment model, AUC=0.597, Brier score=0.242)``.
    """
    columns = (
        vocabulary.PROBABILITY_COLUMNS
        if model.kind == "probability"
        else vocabulary.REGRESSION_COLUMNS
    )
    facts: list[tuple[str, Any]] = [("", model_words(model.name))]
    for key, label in list(columns.items())[:2]:
        value = model.metrics.get(key)
        if value is not None and math.isfinite(value):
            facts.append((label, f"{value:.3f}"))
    return short_repr("NuisanceModelReport", facts)
