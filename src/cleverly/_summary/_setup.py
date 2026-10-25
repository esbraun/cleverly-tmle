"""The printed summaries of the objects a user meets before a fit, and of a simulation study.

A :class:`~cleverly.CausalStudy`, an :class:`~cleverly.IdentifiedEffect`, the methods that can
estimate it, a :class:`~cleverly.StudyProtocol`, a
:class:`~cleverly.validation.simulation.StudyResult` and a
:class:`~cleverly.utils.phases.PhaseProfile` print through these builders, on the layer every
result summary shares. Each builder reads only the object's own fields.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    check_detail,
    count_text,
    note,
    number_text,
    share_text,
    vocabulary,
)
from ._data import (
    clusters_fact,
    covariate_sources,
    listed,
    outcome_fact,
    rows_fact,
    share_fact,
    treatment_fact,
    weights_facts,
)
from ._estimates import without_roadmap

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..methods import MethodAvailability
    from ..protocol import StudyProtocol
    from ..study import CausalStudy, IdentifiedEffect
    from ..utils.phases import PhaseProfile
    from ..validation.simulation import StudyResult

#: The most distinct messages one warning note quotes before it counts the rest.
_MOST_MESSAGES = 3
#: The most failures the full simulation summary lists one per line.
_MOST_FAILURES = 10


# ----------------------------------------------------------------- shared pieces


def warning_notes(records: Sequence[Any], template: str = "data_warning") -> list[Note]:
    """One note per recorded warning category, each distinct message once, in first-seen
    order, quoted as the user saw it without roadmap pointers.

    Parameters
    ----------
    records : sequence of FitWarning
        The recorded warnings.
    template : str
        The key of the note text that quotes the messages.

    Returns
    -------
    list of Note
        The notes.
    """
    grouped: dict[str, list[Any]] = {}
    for record in records:
        code = vocabulary.WARNING_NOTE_CODES.get(record.category, "fit_warning_other")
        grouped.setdefault(code, []).append(record)
    notes = []
    for code, held in grouped.items():
        quoted = [
            without_roadmap(record.message)
            + ("" if record.count == 1 else vocabulary.REPEATS.format(count=record.count))
            for record in held[:_MOST_MESSAGES]
        ]
        text = vocabulary.NOTE_TEXTS[template].format(messages=" ".join(quoted))
        if len(held) > _MOST_MESSAGES:
            text += " " + vocabulary.NOTE_TEXTS["fit_warning_more"].format(
                count=len(held) - _MOST_MESSAGES
            )
        notes.append(note(code, text))
    return notes


def _design_word(design: Any, data: Any) -> str:
    template = vocabulary.DESIGN_WORDS.get(type(design).__name__, type(design).__name__)
    return template.format(T=getattr(data, "n_times", 1))


def _is_longitudinal(data: Any) -> bool:
    return hasattr(data, "n_times") and not hasattr(data, "covariate_names")


def _longitudinal_outcome(data: Any) -> str:
    events = tuple(getattr(data, "event_names", ()) or ())
    if events:
        return f"{listed(events)} (event by each time)"
    kind = "binary" if data.family == "binomial" else "continuous"
    return f"{data.outcome_name} ({kind})"


def _censoring_fact(data: Any) -> str | None:
    names = tuple(getattr(data, "censoring_names", ()) or ())
    if not names:
        return None
    return listed(names)


def _followed_fact(data: Any) -> tuple[str, str] | None:
    """The share of units still observed at the last time when the design censors, or, for
    an event time, the share censored before the end: a unit after its event is fully
    observed, so it does not count as lost."""
    if getattr(data, "event", None) is not None and getattr(data, "repeated", None) is None:
        times, codes = data.to_time_to_event()
        end = data.time_grid[-1] if data.time_grid else float(data.n_times)
        censored = (np.asarray(codes) == 0) & (np.asarray(times, dtype=float) < end)
        return vocabulary.CENSORED_BEFORE_END, share_text(float(np.mean(censored)))
    if not tuple(getattr(data, "censoring_names", ()) or ()):
        return None
    return vocabulary.FOLLOWED, share_text(float(data.uncensored_through(data.n_times).mean()))


def _declared(design: Any, field: str) -> tuple[str, ...]:
    """The columns a design field names, flattened, in the order the caller gave them."""
    value = getattr(design, field, None)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(str(item) for item in value.values())
    names: list[str] = []
    for item in value:
        names.extend((item,) if isinstance(item, str) else (str(part) for part in item))
    return tuple(dict.fromkeys(names))


def protocol_lines(protocol: StudyProtocol) -> tuple[str, ...]:
    """Every protocol field on one line, a list of items joined by semicolons."""
    fields = (
        ("Population", protocol.target_population),
        ("Eligibility", protocol.eligibility),
        (vocabulary.TIME_ZERO, protocol.time_zero),
        ("Strategies", protocol.treatment_strategies),
        ("Versions", protocol.treatment_versions),
        ("Outcome", protocol.outcome),
        ("Horizon", protocol.horizon),
        ("Intercurrent events", protocol.intercurrent_event_handling),
        ("Interference unit", protocol.interference_unit),
        ("Rationale", protocol.assumption_rationale),
    )
    return tuple(
        f"{label}: {value if isinstance(value, str) else '; '.join(value)}"
        for label, value in fields
    )


def _protocol_section(protocol: StudyProtocol | None) -> SummarySection:
    if protocol is None:
        return SummarySection(vocabulary.SECTION_STUDY, (vocabulary.NO_PROTOCOL,))
    return SummarySection(vocabulary.SECTION_STUDY, protocol.summary_lines())


# ----------------------------------------------------------------- the study


def _point_header(design: Any, data: Any) -> tuple[list[tuple[str, str]], int]:
    left: list[tuple[str, str]] = [
        (vocabulary.DESIGN, _design_word(design, data)),
        (vocabulary.OUTCOME, outcome_fact(data)),
        (vocabulary.TREATMENT, treatment_fact(data)),
    ]
    sources = covariate_sources(data)
    if sources:
        left.append((vocabulary.ADJUSTED_FOR, listed(sources)))
    elif getattr(design, "randomized", False):
        left.append((vocabulary.ADJUSTED_FOR, vocabulary.RANDOMIZED_NO_ADJUSTMENT))
    intermediate = getattr(design, "intermediate", None)
    if intermediate is not None:
        left.append((vocabulary.HELD_FIXED, str(intermediate)))
    share = share_fact(data)
    right: list[tuple[str, str]] = [(vocabulary.ROWS, rows_fact(data))]
    # A dose range is too long for the right column, and arm shares print as a table.
    if share[0].startswith(vocabulary.TREATED_SHARE):
        right.append(share)
    elif arm_table(data) is None:
        left.append(share)
    if getattr(data, "known_treatment", None) is not None:
        right.append((vocabulary.TREATMENT_MODEL, vocabulary.KNOWN_MECHANISM))
    clusters = clusters_fact(data)
    if clusters is not None:
        right.append((vocabulary.CLUSTERS, clusters))
    strata = tuple(getattr(data, "strata_names", ()) or ())
    if strata:
        right.append((vocabulary.STRATA, listed(strata)))
    right.extend(weights_facts(data))
    return [*left, *right], len(left)


def _longitudinal_header(design: Any, data: Any) -> tuple[list[tuple[str, str]], int]:
    # A long panel and an event time are reshaped into derived node columns, so the
    # header names the columns the caller declared.
    reshaped = type(design).__name__ in ("LongPanel", "TimeToEvent")
    if type(design).__name__ == "TimeToEvent":
        outcome = vocabulary.EVENT_OUTCOME.format(event=design.event, time=design.time)
    else:
        outcome = _longitudinal_outcome(data)
    treatment = _declared(design, "treatment") if reshaped else tuple(data.treatment_names)
    left: list[tuple[str, str]] = [
        (vocabulary.DESIGN, _design_word(design, data)),
        (vocabulary.OUTCOME, outcome),
        (vocabulary.TREATMENT, listed(treatment)),
    ]
    baseline = tuple(data.baseline_names)
    if baseline:
        left.append((vocabulary.BASELINE, listed(baseline)))
    censoring = (
        listed(_declared(design, "censoring")) or None if reshaped else _censoring_fact(data)
    )
    if censoring is not None:
        left.append((vocabulary.CENSORING, censoring))
    right: list[tuple[str, str]] = [
        (vocabulary.ROWS, count_text(data.n)),
        (vocabulary.TIME_POINTS, str(data.n_times)),
    ]
    followed = _followed_fact(data)
    if followed is not None:
        right.append(followed)
    if getattr(data, "known_mechanisms", None) is not None:
        right.append((vocabulary.TREATMENT_MODEL, vocabulary.KNOWN_MECHANISM))
    clusters = clusters_fact(data)
    if clusters is not None:
        right.append((vocabulary.CLUSTERS, clusters))
    right.extend(weights_facts(data))
    return [*left, *right], len(left)


def _named(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def study_roles(study: CausalStudy) -> list[tuple[str, tuple[str, ...]]]:
    """Each role the design declares, with the columns that fill it, in reading order.

    Parameters
    ----------
    study : CausalStudy
        The study.

    Returns
    -------
    list of tuple
        ``(role, columns)`` pairs; a role with no column is left out.
    """
    data: Any = study.data
    design = study.design
    kind = type(design).__name__
    if kind in ("LongPanel", "TimeToEvent"):
        declared = [
            ("unit", _declared(design, "unit")),
            ("time", _declared(design, "time")),
            ("outcome", _declared(design, "outcome") if kind == "LongPanel" else ()),
            ("event", _declared(design, "event")),
            ("treatment", _declared(design, "treatment")),
            ("baseline", _declared(design, "baseline")),
            ("measured over time", _declared(design, "time_varying")),
            ("censoring", _declared(design, "censoring")),
            ("clusters", _declared(design, "cluster")),
            ("weights", _declared(design, "weights")),
        ]
        return [(role, names) for role, names in declared if names]
    if _is_longitudinal(data):
        roles: list[tuple[str, tuple[str, ...]]] = [
            ("outcome", tuple(getattr(data, "event_names", ()) or ()) or (data.outcome_name,)),
            ("treatment", tuple(data.treatment_names)),
            ("baseline", tuple(data.baseline_names)),
        ]
        for node, names in zip(data.treatment_names, data.time_varying_names, strict=False):
            roles.append((f"measured before {node}", tuple(names)))
        roles.append(("censoring", tuple(getattr(data, "censoring_names", ()) or ())))
        roles.append(("clusters", _named(getattr(data, "cluster_name", None))))
        roles.append(("weights", _named(getattr(data, "weights_name", None))))
        return [(role, names) for role, names in roles if names]
    roles = [
        ("outcome", (data.outcome_name,)),
        ("treatment", tuple(data.component_names) or (data.treatment_name,)),
        ("adjusted for", covariate_sources(data)),
        ("treatment model reads", _named(getattr(design, "treatment_adjustment", None))),
        ("outcome recorded", _named(getattr(design, "missingness", None))),
        ("treatment recorded", _named(getattr(design, "treatment_missingness", None))),
        ("held fixed", _named(getattr(design, "intermediate", None))),
        ("weights", _named(getattr(data, "weights_name", None))),
        ("clusters", _named(getattr(data, "cluster_name", None))),
        ("strata", tuple(getattr(data, "strata_names", ()) or ())),
    ]
    return [(role, names) for role, names in roles if names]


def study_header(study: CausalStudy) -> tuple[list[tuple[str, str]], int]:
    """The header facts of a study summary and how many go in the left column."""
    if _is_longitudinal(study.data):
        header, split = _longitudinal_header(study.design, study.data)
    else:
        header, split = _point_header(study.design, study.data)
    header.append((vocabulary.PROTOCOL, "recorded" if study.protocol is not None else "none"))
    return header, split


def study_summary(study: CausalStudy, detail: Detail) -> Summary:
    """The summary of a study design and the data it validated.

    Parameters
    ----------
    study : CausalStudy
        The study.
    detail : {"short", "full"}
        ``"full"`` adds the study description.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    header, split = study_header(study)
    roles = SummaryTable(
        ("role", "columns"),
        tuple(
            (Cell(role, role), Cell(", ".join(names), ", ".join(names)))
            for role, names in study_roles(study)
        ),
        title=vocabulary.ROLES_TITLE,
    )
    notes = warning_notes(getattr(study.data, "validation_warnings", ()))
    sections = [_protocol_section(study.protocol)] if detail == "full" else []
    arms = None if _is_longitudinal(study.data) else arm_table(study.data)
    return Summary.build(
        title=vocabulary.STUDY_TITLE,
        header=header,
        header_split=split,
        tables=[roles] if arms is None else [roles, arms],
        notes=notes,
        sections=sections,
    )


def arm_table(data: Any) -> SummaryTable | None:
    """The share of rows at each treatment level, for an arm-coded treatment other than a
    0/1 one, whose treated share the header prints. ``None`` otherwise."""
    if data.treatment_kind == "vector" or data.is_continuous_treatment:
        return None
    label, _ = share_fact(data)
    if label.startswith(vocabulary.TREATED_SHARE):
        return None
    column = vocabulary.SHARE_COLUMN + (vocabulary.WEIGHTED if data.is_weighted else "")
    rows = tuple(
        (Cell(data.arm_label(arm), str(data.arm_label(arm))), Cell(share, share_text(share)))
        for arm, share in zip(data.arm_codes, data.arm_fractions, strict=True)
    )
    return SummaryTable(("level", column), rows, title=vocabulary.ARMS_TITLE)


def study_repr(study: CausalStudy) -> str:
    """The one-line repr of a study."""
    from . import short_repr

    data: Any = study.data
    design = study.design
    facts: list[tuple[str, Any]] = [("", _design_word(design, data)), ("n", data.n)]
    if type(design).__name__ in ("LongPanel", "TimeToEvent"):
        outcome = _declared(design, "outcome") or _declared(design, "event")
        facts.append(("outcome", ", ".join(outcome)))
        facts.append(("treatment", ", ".join(_declared(design, "treatment"))))
    elif _is_longitudinal(data):
        facts.append(("outcome", data.outcome_name))
        facts.append(("treatments", len(data.treatment_names)))
    else:
        facts.append(("outcome", data.outcome_name))
        facts.append(("treatment", data.treatment_name))
        facts.append(("covariates", len(covariate_sources(data))))
    return short_repr("CausalStudy", facts)


# ----------------------------------------------------------------- the question


def condition_words(
    statement: str, features: frozenset[str] = frozenset()
) -> tuple[str, str, str] | None:
    """The plain name, sentence and checkable mark of an identification statement.

    Parameters
    ----------
    statement : str
        One statement of an identification record.
    features : frozenset of str
        The design's features that change the words, keys of
        :data:`~cleverly._summary.vocabulary.CONDITION_VARIANTS`.

    Returns
    -------
    tuple of str or None
        ``(name, sentence, checkable)`` with the ``{A}``, ``{Y}``, ``{Z}`` and ``{held}``
        placeholders unfilled, or ``None`` when no entry of
        :data:`~cleverly._summary.vocabulary.CONDITION_WORDS` matches.
    """
    for feature in sorted(features):
        for start, words in vocabulary.CONDITION_VARIANTS.get(feature, ()):
            if statement.startswith(start):
                return words
    for start, words in vocabulary.CONDITION_WORDS:
        if statement.startswith(start):
            return words
    return None


def design_features(effect: IdentifiedEffect) -> frozenset[str]:
    """The design features that change a question's condition words.

    Parameters
    ----------
    effect : IdentifiedEffect
        The identified question.

    Returns
    -------
    frozenset of str
        Keys of :data:`~cleverly._summary.vocabulary.CONDITION_VARIANTS`.
    """
    functional = effect.functional
    study = effect._study
    data = getattr(study, "data", None)
    features = set()
    if functional.intermediate is not None:
        features.add("held")
    if _randomized_without_adjustment(effect):
        features.add("randomized")
    if (
        functional.longitudinal
        and data is not None
        and not tuple(getattr(data, "censoring_names", ()) or ())
    ):
        features.add("uncensored")
    return frozenset(features)


def _randomized_without_adjustment(effect: IdentifiedEffect) -> bool:
    study = effect._study
    if study is None or not getattr(study.design, "randomized", False):
        return False
    data = study.data
    return not _is_longitudinal(data) and not covariate_sources(data)


def method_note(statement: str, effect: IdentifiedEffect) -> str | None:
    """The note code that states ``statement`` in plain words, when it describes how the
    method reads the data rather than a condition on it.

    Parameters
    ----------
    statement : str
        An identification statement.
    effect : IdentifiedEffect
        The identified question, whose design says whether the outcome can be missing.

    Returns
    -------
    str or None
        The note code, or ``None`` for a statement that is a condition.
    """
    for prefix, (complete, incomplete) in vocabulary.METHOD_STATEMENTS:
        if statement.startswith(prefix):
            return incomplete if _outcome_missing(effect) else complete
    return None


def _outcome_missing(effect: IdentifiedEffect) -> bool:
    """Whether any outcome is unrecorded: read from the data, as the estimator reads it, and
    from the declared response column only for a question bound to no data."""
    data = getattr(effect._study, "data", None)
    if data is None or not hasattr(data, "has_missing_outcome"):
        return effect.functional.missingness is not None
    return bool(data.has_missing_outcome)


def _as_treated_text(code: str, effect: IdentifiedEffect) -> str:
    """The as-treated note, naming the weights of a weighted design."""
    data = getattr(effect._study, "data", None)
    weights = getattr(data, "weights_name", None)
    average = vocabulary.AS_TREATED_AVERAGE.format(Y=effect.functional.outcome)
    if weights is not None:
        average += vocabulary.AS_TREATED_WEIGHTED.format(weights=weights)
    return _fill(vocabulary.NOTE_TEXTS[code].replace("{average}", average), effect)


def _fill(text: str, effect: IdentifiedEffect) -> str:
    functional = effect.functional
    held = functional.intermediate
    return text.format(
        A=_treatment_name(effect),
        Y=functional.outcome,
        Z=functional.intermediate_name or "the held variable",
        held=(
            ""
            if held is None
            else vocabulary.HELD_CLAUSE.format(Z=functional.intermediate_name or "it")
        ),
    )


def _treatment_name(effect: IdentifiedEffect) -> str:
    treatment = effect.functional.treatment
    return treatment if isinstance(treatment, str) else ", ".join(map(str, treatment))


def conditions(effect: IdentifiedEffect) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """The statements with plain words, and the statements that print only in full.

    Parameters
    ----------
    effect : IdentifiedEffect
        The identified question.

    Returns
    -------
    tuple
        The rows ``(statement, name, checkable, sentence)`` for the statements an entry
        matches, filled with the question's names, and the other statements.
    """
    rows: list[tuple[str, str, str, str]] = []
    others: list[str] = []
    features = design_features(effect)
    for statement in _statements(effect):
        words = condition_words(statement, features)
        if words is None:
            others.append(statement)
            continue
        name, sentence, checkable = words
        rows.append((statement, _fill(name, effect), checkable, _fill(sentence, effect)))
    return rows, others


def _statements(effect: IdentifiedEffect) -> list[str]:
    """The identification statements this question rests on, without the caveats of a
    feature the design does not declare."""
    from ..targets.builtin import INTERMEDIATE_CAVEAT_PREFIX, MISSINGNESS_CAVEAT_PREFIX

    functional = effect.functional
    kept = []
    for statement in effect.identification.assumptions:
        if statement.startswith(MISSINGNESS_CAVEAT_PREFIX) and functional.missingness is None:
            continue
        if statement.startswith(INTERMEDIATE_CAVEAT_PREFIX) and functional.intermediate is None:
            continue
        kept.append(statement)
    return kept


def question_text(effect: IdentifiedEffect) -> str:
    """The Question fact: the question in plain words where the vocabulary has them.

    Parameters
    ----------
    effect : IdentifiedEffect
        The identified question.

    Returns
    -------
    str
        For example ``"average effect of A on Y (1 compared with 0)"``.
    """
    functional = effect.functional
    treatment = _treatment_name(effect)
    if functional.intermediate is not None:
        return vocabulary.DIRECT_EFFECT_QUESTION.format(
            A=treatment,
            Y=functional.outcome,
            Z=functional.intermediate_name or "the held variable",
            z=f"{functional.intermediate:g}",
        )
    phrase = vocabulary.QUESTION_PHRASES.get(functional.target)
    if functional.longitudinal or functional.axis != "arm" or phrase is None:
        return effect.estimand.definition
    reference = functional.reference_arm
    levels = tuple(functional.treatment_levels)
    value = functional.treatment_value
    if value is None:
        others = [level for level in levels if level != reference]
        if len(others) != 1 and "{v}" in phrase.split("(")[0]:
            return effect.estimand.definition
        value = others[0] if len(others) == 1 else "each other level"
    return phrase.format(A=treatment, Y=functional.outcome, v=value, r=reference)


def methods_fact(records: Sequence[MethodAvailability]) -> str:
    """The Methods that fit fact: how many of the methods fit, such as ``"4 of 11"``."""
    fitting = sum(bool(record.available) for record in records)
    return f"{count_text(fitting)} of {count_text(len(records))}"


def methods_table(records: Sequence[MethodAvailability], *, fits: bool = True) -> SummaryTable:
    """The Methods table: each method's name, whether it fits, and its plain name.

    ``fits=False`` leaves out the fits column, for a table of only the methods that fit.
    """
    rows = []
    for record in records:
        plain = vocabulary.METHOD_NAMES.get(record.name, record.name)
        cells = [Cell(record.name, record.name)]
        if fits:
            cells.append(Cell(record.available, "yes" if record.available else "no"))
        cells.append(Cell(plain, plain))
        rows.append(tuple(cells))
    columns = ("method", "fits", "name") if fits else ("method", "name")
    return SummaryTable(columns, tuple(rows), title=vocabulary.METHODS_TABLE_TITLE)


def unavailable_section(records: Sequence[MethodAvailability]) -> SummarySection:
    """Why each method that does not fit does not, one line each, without roadmap ids."""
    lines = tuple(
        f"{record.name}: {without_roadmap(record.reason or '')}"
        for record in records
        if not record.available
    )
    return SummarySection(vocabulary.SECTION_UNAVAILABLE, lines or ("Every method fits.",))


def assumption_lines(effect: IdentifiedEffect, fitted_method: str | None = None) -> list[str]:
    """The Assumptions section of a question or a fit: every statement, verbatim.

    Parameters
    ----------
    effect : IdentifiedEffect
        The identified question.
    fitted_method : str or None
        The method that fitted a result. An ``"adaptive_iptw"`` fit fits no outcome
        regression, so its model restriction is left to the fit's own lines.

    Returns
    -------
    list of str
        The lead sentence, then one line per statement, reading and restriction.
    """
    identification = effect.identification
    lines = [vocabulary.CONDITIONS_LEAD.format(functional=effect.functional.expression)]
    lines.extend(f"- {item}" for item in _statements(effect))
    lines.extend(
        f"- {reading.label}: {reading.meaning} (needs {'; '.join(reading.assumptions)})"
        for reading in identification.interpretations
    )
    if fitted_method != "adaptive_iptw":
        lines.extend(f"- model restriction: {item}" for item in identification.model_restrictions)
    return lines


def _question_header(effect: IdentifiedEffect) -> tuple[list[tuple[str, str]], int]:
    functional = effect.functional
    study = effect._study
    data = getattr(study, "data", None)
    left: list[tuple[str, str]] = [(vocabulary.QUESTION, question_text(effect))]
    if data is not None and not _is_longitudinal(data):
        left.append((vocabulary.OUTCOME, outcome_fact(data)))
        left.append((vocabulary.TREATMENT, treatment_fact(data)))
    elif data is not None:
        left.append((vocabulary.OUTCOME, _longitudinal_outcome(data)))
        left.append((vocabulary.TREATMENT, listed(data.treatment_names)))
    else:
        left.append((vocabulary.OUTCOME, str(functional.outcome)))
        left.append((vocabulary.TREATMENT, _treatment_name(effect)))
    adjustment = tuple(functional.adjustment)
    # The columns the data kept: a constant column the study dropped is not adjusted for.
    if data is not None and not _is_longitudinal(data):
        adjustment = covariate_sources(data)
    elif data is not None:
        adjustment = tuple(data.baseline_names)
    if functional.longitudinal:
        if adjustment:
            left.append((vocabulary.BASELINE, listed(adjustment)))
    elif adjustment:
        left.append((vocabulary.ADJUSTED_FOR, listed(adjustment)))
    elif getattr(getattr(study, "design", None), "randomized", False):
        left.append((vocabulary.ADJUSTED_FOR, vocabulary.RANDOMIZED_NO_ADJUSTMENT))
    right: list[tuple[str, str]] = []
    if data is not None:
        right.append((vocabulary.ROWS, count_text(data.n)))
    answered = (
        vocabulary.ANSWERED_BY_RANDOMIZED
        if _randomized_without_adjustment(effect)
        else vocabulary.ANSWERED_BY_WORDS[bool(functional.longitudinal)]
    )
    right.append((vocabulary.ANSWERED_BY, answered))
    return [*left, *right], len(left)


def question_summary(effect: IdentifiedEffect, detail: Detail) -> Summary:
    """The summary of an identified question.

    Parameters
    ----------
    effect : IdentifiedEffect
        The identified question.
    detail : {"short", "full"}
        ``"full"`` adds every statement verbatim, why each method does not fit, and the
        study description.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    records = tuple(effect.available_methods())
    header, split = _question_header(effect)
    rows, others = conditions(effect)
    table = SummaryTable(
        ("condition", "checkable", "what it says"),
        tuple(
            (Cell(statement, name), Cell(checkable, checkable), Cell(sentence, sentence))
            for statement, name, checkable, sentence in rows
        ),
        title=vocabulary.CONDITIONS_TITLE,
    )
    texts = vocabulary.NOTE_TEXTS
    notes: list[Note] = []
    if effect._study is None:
        notes.append(note("unbound_effect", texts["unbound_effect"]))
    if not any(record.available for record in records):
        notes.append(note("no_method", texts["no_method"]))
    codes = [method_note(item, effect) for item in others]
    method_codes = [code for code in codes if code is not None]
    further = [item for item, code in zip(others, codes, strict=True) if code is None]
    if further:
        notes.append(
            note(
                "further_conditions",
                texts["further_conditions"].format(
                    count=len(further),
                    conditions="condition" if len(further) == 1 else "conditions",
                ),
            )
        )
    for code in dict.fromkeys(method_codes):
        notes.append(note(code, _as_treated_text(code, effect)))
    if any(checkable == "partly" for _, _, checkable, _ in rows):
        notes.append(note("checkable", texts["checkable"]))
    identification = effect.identification
    if identification.model_restrictions:
        notes.append(note("model_restriction", texts["model_restriction"]))
    if identification.interpretations:
        notes.append(
            note(
                "several_readings",
                texts["several_readings"].format(count=len(identification.interpretations)),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(vocabulary.SECTION_ASSUMPTIONS, tuple(assumption_lines(effect)))
        )
        sections.append(unavailable_section(records))
        sections.append(_protocol_section(effect.protocol))
    tables = [table] if table.rows else []
    history = tuple(effect.functional.history or ())
    if history:
        # Each time's treatment model reads its own history, so one list of the baseline
        # columns would understate every later time.
        tables.insert(
            0,
            SummaryTable(
                ("treatment", "adjusted for"),
                tuple(
                    (
                        Cell(str(node), str(node)),
                        Cell(listed(names) or "nothing", listed(names) or "nothing"),
                    )
                    for node, names in history
                ),
                title=vocabulary.HISTORY_TITLE,
            ),
        )
    fitting = tuple(record for record in records if record.available)
    if fitting and len(fitting) < len(records):
        unfit = len(records) - len(fitting)
        notes.append(
            note(
                "other_methods",
                texts["other_methods"].format(
                    count=count_text(unfit), methods="method" if unfit == 1 else "methods"
                ),
            )
        )
    tables.append(methods_table(fitting or records, fits=not fitting))
    return Summary.build(
        title=vocabulary.QUESTION_TITLE.format(name=effect.estimand.name),
        header=header,
        header_split=split,
        tables=tables,
        notes=notes,
        sections=sections,
    )


def question_repr(effect: IdentifiedEffect) -> str:
    """The one-line repr of an identified question."""
    from . import short_repr

    records = effect.available_methods()
    fitting = sum(record.available for record in records)
    functional = effect.functional
    return short_repr(
        "IdentifiedEffect",
        (
            ("", f"{effect.estimand.name} of {_treatment_name(effect)} on {functional.outcome}"),
            ("methods", f"{fitting} fit"),
        ),
    )


def methods_summary(
    records: Sequence[MethodAvailability], question: str | None, detail: Detail
) -> Summary:
    """The summary of the methods that can and cannot estimate one question.

    Parameters
    ----------
    records : sequence of MethodAvailability
        One record per method.
    question : str or None
        The question in plain words, when known.
    detail : {"short", "full"}
        ``"full"`` adds why each method does not fit.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    header = []
    if question is not None:
        header.append((vocabulary.QUESTION, question))
    split = len(header)
    header.append((vocabulary.METHODS_FIT, methods_fact(records)))
    notes = []
    if not any(record.available for record in records):
        notes.append(note("no_method", vocabulary.NOTE_TEXTS["no_method"]))
    return Summary.build(
        title=vocabulary.METHODS_TITLE,
        header=header,
        header_split=split,
        tables=[methods_table(records)],
        notes=notes,
        sections=[unavailable_section(records)] if detail == "full" else [],
    )


# ----------------------------------------------------------------- the protocol


def protocol_summary(protocol: StudyProtocol, detail: Detail) -> Summary:
    """The summary of a study protocol.

    Parameters
    ----------
    protocol : StudyProtocol
        The record.
    detail : {"short", "full"}
        ``"full"`` adds the intercurrent-event handling and the rationale.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    header = [
        (vocabulary.POPULATION, protocol.target_population),
        (vocabulary.TIME_ZERO, protocol.time_zero),
        (vocabulary.OUTCOME, protocol.outcome),
        (vocabulary.HORIZON, protocol.horizon),
        (vocabulary.INTERFERENCE_UNIT, protocol.interference_unit),
    ]
    strategies = SummaryTable(
        ("strategy", "version"),
        tuple(
            (Cell(strategy, strategy), Cell(version, version))
            for strategy, version in zip(
                protocol.treatment_strategies, protocol.treatment_versions, strict=True
            )
        ),
        title=vocabulary.STRATEGIES_TITLE,
    )
    eligibility = SummaryTable(
        ("", "criterion"),
        tuple(
            (Cell(str(index), str(index)), Cell(item, item))
            for index, item in enumerate(protocol.eligibility, 1)
        ),
        title=vocabulary.ELIGIBILITY_TITLE,
    )
    sections = (
        [
            SummarySection(
                vocabulary.SECTION_INTERCURRENT,
                tuple(f"- {item}" for item in protocol.intercurrent_event_handling),
            ),
            SummarySection(
                vocabulary.SECTION_RATIONALE,
                tuple(f"- {item}" for item in protocol.assumption_rationale),
            ),
        ]
        if detail == "full"
        else []
    )
    return Summary.build(
        title=vocabulary.PROTOCOL_TITLE,
        header=header,
        header_split=3,
        tables=[strategies, eligibility],
        sections=sections,
    )


# ----------------------------------------------------------------- the simulation study


def _signed(value: float, decimals: int) -> str:
    text = number_text(value, decimals)
    return text if text.startswith("-") or float(text) == 0.0 else f"+{text}"


#: A coverage shortfall at most this large, though beyond Monte Carlo error, reads as small.
COVERAGE_FLOOR = 0.02
#: A bias at most this share of max(1, |truth|), though beyond Monte Carlo error, reads as
#: small.
BIAS_FLOOR = 0.02


def simulation_reading(result: StudyResult) -> str:
    """The reading of a simulation study: whether coverage and bias look right.

    A coverage shortfall or a bias beyond three Monte Carlo standard errors is named for
    its estimand. One within the floors, :data:`COVERAGE_FLOOR` points of coverage or
    :data:`BIAS_FLOOR` of ``max(1, |truth|)``, is named as small rather than left out, so
    the reading says the study passed only when nothing exceeds Monte Carlo error.

    Parameters
    ----------
    result : StudyResult
        The study.

    Returns
    -------
    str
        One or more sentences.
    """
    from ..inference.influence import spread_name

    level = 1.0 - result.alpha
    level_text = f"{level:.0%}"
    floor = f"{100 * COVERAGE_FLOOR:.0f}"
    sentences: list[str] = []
    for summary in result.summaries.values():
        shortfall = level - summary.coverage
        # A shortfall beyond Monte Carlo error is reported however small it is; one under
        # the floor is named as small, so the reading never calls it zero.
        if shortfall > 3.0 * summary.coverage_se:
            coverage = f"{summary.coverage:.3f}"
            if shortfall <= COVERAGE_FLOOR:
                sentences.append(
                    vocabulary.SIMULATION_SMALL_UNDERCOVERAGE.format(
                        name=summary.estimand, coverage=coverage, level=level_text, floor=floor
                    )
                    + "."
                )
            else:
                sentence = vocabulary.SIMULATION_UNDERCOVERAGE.format(
                    name=summary.estimand, coverage=coverage, level=level_text
                )
                if summary.mean_std_error <= 0.0 or summary.se_ratio <= 0.0:
                    sentence += vocabulary.SIMULATION_SE_ZERO
                elif summary.se_ratio < 0.95:
                    sentence += vocabulary.SIMULATION_SE_SMALL.format(
                        spread=spread_name("reported standard error", summary.inference),
                        factor=f"{1 / summary.se_ratio:.2f}",
                    )
                else:
                    sentence += vocabulary.SIMULATION_SE_RIGHT
                sentences.append(sentence + ".")
        if summary.bias_se <= 0.0:
            # Every replicate returned the same estimate, so any bias at all is detectable.
            if summary.bias != 0.0:
                sentences.append(
                    vocabulary.SIMULATION_CONSTANT_BIAS.format(
                        name=summary.estimand, bias=f"{summary.bias:+.4f}"
                    )
                    + "."
                )
        elif abs(summary.bias) > 3.0 * summary.bias_se:
            template = (
                vocabulary.SIMULATION_BIAS
                if abs(summary.bias) > BIAS_FLOOR * max(1.0, abs(summary.truth))
                else vocabulary.SIMULATION_SMALL_BIAS
            )
            sentences.append(
                template.format(
                    name=summary.estimand,
                    bias=f"{summary.bias:+.4f}",
                    count=f"{abs(summary.bias) / summary.bias_se:.1f}",
                    floor=f"{BIAS_FLOOR:.0%}",
                )
                + "."
            )
    if sentences:
        return " ".join(sentences)
    if result._diagnostic:
        return vocabulary.SIMULATION_DIAGNOSTIC_PASS.format(noun=result._diagnostic_noun)
    return vocabulary.SIMULATION_PASS.format(level=level_text)


#: The columns of the simulation table, each with the ``to_frame`` column it prints and its
#: decimals.
SIMULATION_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("truth", "truth", 4),
    ("bias", "bias", 4),
    ("MC SD", "monte_carlo_se", 4),
    ("mean SE", "mean_std_error", 4),
    ("SE ratio", "se_ratio", 3),
    ("coverage", "coverage", 3),
    ("rejected", "rejection_rate", 3),
)


def _simulation_notes(result: StudyResult) -> list[Note]:
    from .._inference_status import supplies_inference
    from ..validation.simulation import _diagnostic_noun

    texts = vocabulary.NOTE_TEXTS
    summaries = list(result.summaries.values())
    notes: list[Note] = []
    if result.n_failed:
        notes.append(
            note(
                "failed_replications",
                texts["failed_replications"].format(
                    count=result.n_failed, total=result.n_replicates
                ),
            )
        )
    if result._diagnostic:
        some = any(
            any(status == "influence_curve" for status, _ in summary.status_counts)
            for summary in summaries
        )
        notes.append(
            note(
                "measures_diagnostic",
                texts["measures_diagnostic"].format(
                    some=" on some replicates" if some else "", noun=result._diagnostic_noun
                ),
            )
        )
    mixed = [summary for summary in summaries if summary.status_counts]
    if mixed:
        clauses = []
        for summary in mixed:
            # Each estimand reads its own statuses: a diagnostic one reads the plug-in
            # diagnostic, and one whose statuses all supply an interval reads that interval.
            statuses = [status for status, _ in summary.status_counts]
            what = (
                "the interval each replicate reported"
                if all(supplies_inference(status) for status in statuses)
                else "the " + _diagnostic_noun(statuses)
            )
            clauses.append(
                vocabulary.MIXED_STATUS_CLAUSE.format(
                    name=summary.estimand, statuses=summary.mixed_statuses, what=what
                )
            )
        notes.append(
            note("mixed_statuses", texts["mixed_statuses"].format(clauses="; ".join(clauses)))
        )
    ratios = [summary.estimand for summary in summaries if summary.inference_estimates is not None]
    if ratios:
        notes.append(
            note(
                "spread_scale",
                texts["spread_scale_simulation"].format(names=", ".join(ratios)),
            )
        )
    experimental = [summary.estimand for summary in summaries if summary.support == "experimental"]
    if experimental:
        notes.append(
            note(
                "experimental",
                vocabulary.SIMULATION_EXPERIMENTAL.format(names=", ".join(experimental)),
            )
        )
    return notes


def simulation_summary(result: StudyResult, detail: Detail) -> Summary:
    """The summary of a simulation study.

    Parameters
    ----------
    result : StudyResult
        The study.
    detail : {"short", "full"}
        ``"full"`` adds the per-estimand figures and the failed replications.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    replications = count_text(result.n_replicates)
    if result.n_failed:
        replications += f" ({count_text(result.n_failed)} failed)"
    header = [
        (vocabulary.ROWS_PER_REPLICATION, count_text(result.n)),
        (vocabulary.REPLICATIONS, replications),
        (vocabulary.INTERVAL_LEVEL, f"{1.0 - result.alpha:.0%}"),
    ]
    rows = []
    for summary in result.summaries.values():
        values = {
            "truth": summary.truth,
            "bias": summary.bias,
            "monte_carlo_se": summary.monte_carlo_se,
            "mean_std_error": summary.mean_std_error,
            "se_ratio": summary.se_ratio,
            "coverage": summary.coverage,
            "rejection_rate": summary.rejection_rate,
        }
        cells = [Cell(summary.estimand, summary.estimand)]
        for _, column, decimals in SIMULATION_COLUMNS:
            value = values[column]
            text = _signed(value, decimals) if column == "bias" else number_text(value, decimals)
            # A missing number is stored as None, since a nan cell never compares equal.
            cells.append(Cell(value if np.isfinite(value) else None, text))
        rows.append(tuple(cells))
    table = SummaryTable(("estimand", *(label for label, _, _ in SIMULATION_COLUMNS)), tuple(rows))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_PER_ESTIMAND,
                tuple(
                    f"{summary.estimand}: mean {summary.mean_estimate:.4f}, RMSE "
                    f"{summary.rmse:.4f}, root-n bias {summary.root_n_bias:+.3f}, bias SE "
                    f"{summary.bias_se:.4f}, coverage SE {summary.coverage_se:.3f}"
                    for summary in result.summaries.values()
                ),
            )
        )
        sections.append(_failures_section(result))
    return Summary.build(
        title=vocabulary.SIMULATION_STUDY_TITLE.format(label=result.label),
        header=header,
        header_split=2,
        tables=[table],
        notes=_simulation_notes(result),
        reading=simulation_reading(result),
        sections=sections,
    )


def _failures_section(result: StudyResult) -> SummarySection:
    if not result.failures:
        return SummarySection(vocabulary.SECTION_FAILURES, ("No replication failed.",))
    lines = [
        f"replicate {failure.replicate}: {failure.error_type}: {without_roadmap(failure.message)}"
        for failure in result.failures[:_MOST_FAILURES]
    ]
    if len(result.failures) > _MOST_FAILURES:
        lines.append(f"and {len(result.failures) - _MOST_FAILURES} more")
    return SummarySection(vocabulary.SECTION_FAILURES, tuple(lines))


# ----------------------------------------------------------------- phase timings


def _seconds(value: float) -> str:
    return number_text(value, 4)


def _share(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.1%}"


def _phase_rows(profile: PhaseProfile, names: Iterable[str]) -> tuple[tuple[Cell, ...], ...]:
    rows = []
    for name in names:
        share = profile.share(name)
        rows.append(
            (
                Cell(name, name),
                Cell(profile.counts[name], count_text(profile.counts[name])),
                Cell(profile.exclusive[name], _seconds(profile.exclusive[name])),
                Cell(share if np.isfinite(share) else None, _share(share)),
                Cell(profile.inclusive[name], _seconds(profile.inclusive[name])),
            )
        )
    return tuple(rows)


def phases_summary(profile: PhaseProfile, detail: Detail) -> Summary:
    """The summary of a phase timing profile, ordered by each phase's own time.

    Parameters
    ----------
    profile : PhaseProfile
        The profile.
    detail : {"short", "full"}
        ``"full"`` adds how to read the columns.

    Returns
    -------
    Summary
        The summary.
    """
    check_detail(detail)
    accounted = sum(profile.exclusive.values())
    share = accounted / profile.total_seconds if profile.total_seconds else float("nan")
    header = [
        (vocabulary.TOTAL_TIME, f"{_seconds(profile.total_seconds)} s"),
        (vocabulary.ACCOUNTED, f"{_seconds(accounted)} s ({_share(share)})"),
    ]
    columns = ("phase", "calls", "own s", "share", "with nested s")
    order = sorted(profile.exclusive, key=lambda key: -profile.exclusive[key])
    tables = [SummaryTable(columns, _phase_rows(profile, order))]
    notes: list[Note] = []
    workers = profile.workers
    if workers is not None:
        worker_order = sorted(workers.exclusive, key=lambda key: -workers.exclusive[key])
        worker_rows = tuple(
            (row[0], row[1], row[2], Cell(None, vocabulary.MISSING), row[4])
            for row in _phase_rows(workers, worker_order)
        )
        tables.append(SummaryTable(columns, worker_rows, title=vocabulary.WORKERS_TITLE))
        notes.append(note("worker_time", vocabulary.NOTE_TEXTS["worker_time"]))
    sections = (
        [SummarySection(vocabulary.SECTION_COLUMNS, vocabulary.PHASE_COLUMNS)]
        if detail == "full"
        else []
    )
    return Summary.build(
        title=vocabulary.PHASES_TITLE,
        header=header,
        tables=tables,
        notes=notes,
        sections=sections,
    )
