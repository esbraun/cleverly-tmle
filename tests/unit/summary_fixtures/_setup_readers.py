"""Readers that parse the printed study-setup summaries back (X89 T5).

Each reader takes the object and the printed short summary, reads every number the text
prints, and compares it with the value computed here from the object's own fields. A
number the reader does not account for fails, so a summary that prints a new number needs
a reader line for it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np

from cleverly._summary import vocabulary

__all__ = [
    "header_value",
    "printed_numbers",
    "question_statements",
    "read_methods",
    "read_phases",
    "read_protocol",
    "read_question",
    "read_simulation",
    "read_study",
    "table_rows",
]

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")


def header_value(text: str, label: str) -> str | None:
    """The first line of the value printed after ``label:`` in the header, or ``None``."""
    for line in text.splitlines():
        if set(line) == {"="} and line:
            continue
        match = re.search(rf"(?:^|\s){re.escape(label)}:\s+(.*)$", line)
        if match:
            return re.split(r"\s{2,}", match.group(1).strip())[0]
    return None


def table_rows(text: str, title: str | None) -> list[list[str]]:
    """The cells of each row of the table whose title is ``title``, or of the first table
    when ``title`` is ``None``, with a wrapped last cell joined."""
    lines = text.splitlines()
    start = 0 if title is None else lines.index(title)
    rule = next(i for i in range(start, len(lines)) if lines[i] and set(lines[i]) == {"-"})
    labels = lines[rule - 1]
    # A cell starts where its column label starts, or a right-aligned label ends.
    rows: list[list[str]] = []
    for line in lines[rule + 1 :]:
        if line and set(line) == {"="}:
            break
        cells = re.split(r"\s{2,}", line.strip())
        if line.startswith(" ") and rows:
            rows[-1][-1] += " " + line.strip()
        else:
            rows.append(cells)
    assert rows, (title, labels)
    return rows


def _without_names(text: str, names: Iterable[str]) -> str:
    for name in sorted({str(item) for item in names if item}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", "NAME", text)
    return text


def printed_numbers(text: str, names: Iterable[str] = ()) -> list[str]:
    """Every number in ``text`` after the user's column names are taken out."""
    return _NUMBER.findall(_without_names(text, names))


def _close(shown: str, value: float) -> bool:
    shown = shown.replace(",", "").lstrip("+")
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return abs(float(shown) - value) <= 0.5 * 10**-decimals + 1e-12


def _check_accounted(text: str, names: Iterable[str], expected: Iterable[float]) -> None:
    """Every printed number equals one of the ``expected`` values to its printed places."""
    values = list(expected)
    for shown in printed_numbers(text, names):
        assert any(_close(shown, value) for value in values), (shown, values)


def _notes_text(text: str) -> str:
    lines = text.splitlines()
    marks = [i for i, line in enumerate(lines) if line in ("Warnings:", "Notes:")]
    return "\n".join(lines[marks[0] :]) if marks else ""


# ----------------------------------------------------------------- the study


def _data_names(data: Any) -> set[str]:
    names = {
        getattr(data, "outcome_name", None),
        getattr(data, "treatment_name", None),
        getattr(data, "cluster_name", None),
        *getattr(data, "covariate_names", ()),
        *getattr(data, "indicator_sources", {}).values(),
        *getattr(data, "treatment_names", ()),
        *getattr(data, "baseline_names", ()),
        *getattr(data, "censoring_names", ()),
        *(name for names in getattr(data, "time_varying_names", ()) for name in names),
    }
    for record in getattr(data, "validation_warnings", ()) or ():
        names.add(str(getattr(record, "message", record)))
    return {str(name) for name in names if name}


def _roles(text: str) -> dict[str, list[str]]:
    """The Columns by role table: each role with the columns it lists."""
    return {
        row[0]: [name.strip() for name in row[1].split(",")]
        for row in table_rows(text, vocabulary.ROLES_TITLE)
    }


def _declared(design: Any, field: str) -> list[str]:
    value = getattr(design, field, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    return [str(item) for item in value]


def _check_roles(study: Any, text: str) -> None:
    """Each role lists the columns the design or the validated data gives it."""
    from cleverly.data.validate import RANDOMIZED_INTERCEPT

    roles = _roles(text)
    design, data = study.design, study.data
    kind = type(design).__name__
    if kind in ("LongPanel", "TimeToEvent"):
        expected = {
            "unit": _declared(design, "unit"),
            "time": _declared(design, "time"),
            "outcome": _declared(design, "outcome") if kind == "LongPanel" else [],
            "event": _declared(design, "event"),
            "treatment": _declared(design, "treatment"),
            "baseline": _declared(design, "baseline"),
            "measured over time": _declared(design, "time_varying"),
            "censoring": _declared(design, "censoring"),
        }
    elif hasattr(data, "n_times") and not hasattr(data, "covariate_names"):
        expected = {
            "outcome": [data.outcome_name],
            "treatment": list(data.treatment_names),
            "baseline": list(data.baseline_names),
            "censoring": list(data.censoring_names or ()),
        }
        for node, names in zip(data.treatment_names, data.time_varying_names, strict=False):
            expected[f"measured before {node}"] = list(names)
    else:
        sources = data.indicator_sources or {}
        kept = [sources.get(name, name) for name in data.covariate_names]
        expected = {
            "outcome": [data.outcome_name],
            "treatment": [data.treatment_name],
            "adjusted for": list(dict.fromkeys(n for n in kept if n != RANDOMIZED_INTERCEPT)),
            "weights": [data.weights_name] if data.is_weighted else [],
            "clusters": [data.cluster_name] if getattr(data, "cluster", None) is not None else [],
        }
    for role, names in expected.items():
        assert roles.get(role, []) == names, (role, roles.get(role), names)


def read_study(study: Any, text: str) -> None:
    """The header counts and the role table of a study equal its design and data, and no
    other number prints."""
    data = study.data
    expected: list[float] = [data.n]
    rows = header_value(text, vocabulary.ROWS)
    assert rows is not None and rows.split()[0] == f"{data.n:,}", rows
    _check_roles(study, text)
    if data.is_weighted:
        report = data.weight_report()
        shown = header_value(text, vocabulary.EFFECTIVE_N)
        assert shown is not None and _close(shown, report.effective_n), shown
        expected.append(report.effective_n)
    if hasattr(data, "n_times") and not hasattr(data, "covariate_names"):
        assert header_value(text, vocabulary.TIME_POINTS) == str(data.n_times)
        expected.append(data.n_times)
        expected.extend(range(1, data.n_times + 1))
        if data.event is not None and data.repeated is None:
            times, codes = data.to_time_to_event()
            end = data.time_grid[-1] if data.time_grid else data.n_times
            share = float(np.mean((codes == 0) & (times < end)))
            shown = header_value(text, vocabulary.CENSORED_BEFORE_END)
            assert shown is not None and _close(shown, share), (shown, share)
            expected.append(share)
        elif data.censoring_names:
            share = float(np.mean(data.uncensored_through(data.n_times)))
            followed = header_value(text, vocabulary.FOLLOWED)
            assert followed is not None and _close(followed, share), (followed, share)
            expected.append(share)
    else:
        treatment = np.asarray(data.treatment, dtype=float)
        recorded = np.isfinite(treatment)
        weights = np.asarray(data.weights, dtype=float)
        share = header_value(
            text, vocabulary.TREATED_SHARE + (vocabulary.WEIGHTED if data.is_weighted else "")
        )
        if data.is_continuous_treatment:
            assert share is None
            dose = header_value(
                text, vocabulary.DOSE + (vocabulary.WEIGHTED if data.is_weighted else "")
            )
            mean = float(np.average(treatment[recorded], weights=weights[recorded]))
            assert dose is not None and _close(dose.split()[1].rstrip(","), mean), dose
            expected.extend([mean, treatment[recorded].min(), treatment[recorded].max()])
        elif share is not None:
            treated = float(np.average(treatment[recorded] == 1, weights=weights[recorded]))
            assert _close(share, treated), (share, treated)
            expected.append(treated)
        else:
            arms = {row[0]: row[1] for row in table_rows(text, vocabulary.ARMS_TITLE)}
            for code in data.arm_codes:
                level = str(data.arm_label(code))
                kept = treatment[recorded] == code
                value = float(np.average(kept, weights=weights[recorded]))
                assert _close(arms[level], value), (level, arms, value)
                expected.append(value)
        expected.extend(float(level) for level in data.treatment_levels if _numeric(level))
        cluster = getattr(data, "cluster", None)
        if cluster is not None:
            count = int(np.unique(cluster).size)
            assert header_value(text, vocabulary.CLUSTERS).startswith(f"{count} "), text
            expected.append(count)
        if data.has_missing_outcome:
            expected.append(int(np.sum(data.observed)))
    _check_accounted(text, _data_names(data), [*expected, *_note_numbers(text)])


def _numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _note_numbers(text: str) -> list[float]:
    """The bracketed note markers, such as ``[1]``, which number the notes in order."""
    return [float(mark) for mark in re.findall(r"^\[(\d+)\]", text, flags=re.MULTILINE)]


# ----------------------------------------------------------------- the question


def _methods_check(text: str, records: Any, *, fitting_only: bool) -> None:
    rows = table_rows(text, vocabulary.METHODS_TABLE_TITLE)
    shown = [record for record in records if record.available or not fitting_only]
    with_fits = not (fitting_only and shown)
    if fitting_only and not shown:
        shown = list(records)
    assert [row[0] for row in rows] == [record.name for record in shown]
    for row, record in zip(rows, shown, strict=True):
        assert len(row) == (3 if with_fits else 2), row
        if with_fits:
            assert row[1] == ("yes" if record.available else "no"), (row, record)


def question_statements(effect: Any) -> list[str]:
    """The statements a question rests on, read off its identification record: every
    assumption, less the missing-outcome and held-intermediate caveats of a design that
    declares neither."""
    from cleverly.targets.builtin import INTERMEDIATE_CAVEAT_PREFIX, MISSINGNESS_CAVEAT_PREFIX

    functional = effect.functional
    dropped = []
    if functional.missingness is None:
        dropped.append(MISSINGNESS_CAVEAT_PREFIX)
    if functional.intermediate is None:
        dropped.append(INTERMEDIATE_CAVEAT_PREFIX)
    return [
        item
        for item in effect.identification.assumptions
        if not any(item.startswith(prefix) for prefix in dropped)
    ]


def read_question(effect: Any, text: str) -> None:
    """The header, the conditions count, the methods table and the counted notes equal the
    question's identification record, its methods and its data."""
    from cleverly.data.validate import RANDOMIZED_INTERCEPT

    study = effect._study
    functional = effect.functional
    expected: list[float] = []
    names: set[str] = {str(functional.outcome), *(str(name) for name in functional.adjustment)}
    answered = header_value(text, vocabulary.ANSWERED_BY)
    if study is not None:
        data = study.data
        assert header_value(text, vocabulary.ROWS) == f"{data.n:,}"
        expected.append(data.n)
        names |= _data_names(data)
        if getattr(data, "has_missing_outcome", False):
            expected.append(int(np.sum(data.observed)))
        for level in getattr(data, "treatment_levels", None) or ():
            expected.extend(float(item) for item in np.ravel(level) if _numeric(item))
        if hasattr(data, "covariate_names"):
            sources = data.indicator_sources or {}
            kept = [sources.get(name, name) for name in data.covariate_names]
            kept = list(dict.fromkeys(name for name in kept if name != RANDOMIZED_INTERCEPT))
            randomized = getattr(study.design, "randomized", False) and not kept
            shown = header_value(text, vocabulary.ADJUSTED_FOR)
            if kept:
                assert shown is not None and shown.split(",")[0] == kept[0], (shown, kept)
            assert answered == (
                vocabulary.ANSWERED_BY_RANDOMIZED
                if randomized
                else vocabulary.ANSWERED_BY_WORDS[False]
            ), answered
        else:
            assert answered == vocabulary.ANSWERED_BY_WORDS[True], answered
    else:
        assert header_value(text, vocabulary.ROWS) is None
    records = tuple(effect.available_methods())
    _methods_check(text, records, fitting_only=True)
    unfit = sum(not record.available for record in records)
    notes = " ".join(_notes_text(text).split())
    if 0 < unfit < len(records):
        assert f" {unfit} other method" in notes, (unfit, notes)
        expected.append(unfit)
    # Every statement is either a row of the short table or counted as a further one.
    statements = question_statements(effect)
    printed = (
        table_rows(text, vocabulary.CONDITIONS_TITLE)
        if vocabulary.CONDITIONS_TITLE in text.splitlines()
        else []
    )
    for row in printed:
        assert row[1] in ("no", "partly"), row
    match = re.search(r"states (\d+) further condition", notes)
    further = int(match.group(1)) if match else 0
    assert len(printed) + further == len(statements), (len(printed), further, statements)
    expected.append(further)
    readings = len(effect.identification.interpretations)
    if readings:
        assert f"has {readings} causal readings" in notes
        expected.append(readings)
    # The question's own words: a contrast level, a held value or a dose it declares.
    expected.extend(
        float(value.rstrip(",").replace(",", ""))
        for value in _NUMBER.findall(repr(effect.estimand))
    )
    _check_accounted(
        _without_table(text, vocabulary.CONDITIONS_TITLE),
        names,
        [*expected, *_note_numbers(text)],
    )


def _without_table(text: str, title: str) -> str:
    lines = text.splitlines()
    if title not in lines:
        return text
    start = lines.index(title)
    end = next(i for i in range(start, len(lines)) if lines[i] and set(lines[i]) == {"="})
    return "\n".join(lines[:start] + lines[end:])


# ----------------------------------------------------------------- the methods


def read_methods(methods: Any, text: str) -> None:
    """The fitting count and every row's yes or no equal the records."""
    records = tuple(methods)
    fitting = sum(bool(record.available) for record in records)
    assert header_value(text, vocabulary.METHODS_FIT) == f"{fitting} of {len(records)}"
    _methods_check(text, records, fitting_only=False)
    question = methods.question or ""
    _check_accounted(
        text,
        (),
        [fitting, len(records), *(float(value) for value in _NUMBER.findall(question))],
    )


# ----------------------------------------------------------------- the protocol


def read_protocol(protocol: Any, text: str) -> None:
    """The strategies and the numbered eligibility criteria equal the protocol's fields."""
    rows = table_rows(text, vocabulary.ELIGIBILITY_TITLE)
    assert [row[0] for row in rows] == [str(i) for i in range(1, len(protocol.eligibility) + 1)]
    assert [row[1] for row in rows] == list(protocol.eligibility)
    strategies = table_rows(text, vocabulary.STRATEGIES_TITLE)
    assert [row[0] for row in strategies] == list(protocol.treatment_strategies)
    assert [row[1] for row in strategies] == list(protocol.treatment_versions)
    authored = [
        protocol.target_population,
        protocol.time_zero,
        protocol.outcome,
        protocol.horizon,
        protocol.interference_unit,
        *protocol.treatment_strategies,
        *protocol.treatment_versions,
        *protocol.eligibility,
    ]
    _check_accounted(text, authored, range(1, len(protocol.eligibility) + 1))


# ----------------------------------------------------------------- the simulation


def read_simulation(result: Any, text: str) -> None:
    """Every cell of the table, the header counts and the reading equal the records."""
    assert header_value(text, vocabulary.ROWS_PER_REPLICATION) == f"{result.n:,}"
    replications = header_value(text, vocabulary.REPLICATIONS)
    failed = len(result.failures)
    expected_replications = f"{result.n_replicates:,}"
    if failed:
        expected_replications += f" ({failed} failed)"
    assert replications == expected_replications, replications
    level = round(100 * (1 - result.alpha))
    assert header_value(text, vocabulary.INTERVAL_LEVEL) == f"{level}%"
    rows = {row[0]: row[1:] for row in table_rows(text, None)}
    assert sorted(rows) == sorted(result.summaries)
    from cleverly._summary._setup import BIAS_FLOOR, COVERAGE_FLOOR

    expected: list[float] = [result.n, result.n_replicates, failed, level]
    expected += [100 * COVERAGE_FLOOR, 100 * BIAS_FLOOR]
    for name, summary in result.summaries.items():
        values = [
            summary.truth,
            summary.bias,
            summary.monte_carlo_se,
            summary.mean_std_error,
            summary.se_ratio,
            summary.coverage,
            summary.rejection_rate,
        ]
        for shown, value in zip(rows[name], values, strict=True):
            assert _close(shown, float(value)), (name, shown, value)
        expected.extend(values)
        if summary.se_ratio > 0:
            expected.append(1 / summary.se_ratio)
        if summary.bias_se > 0:
            expected.append(abs(summary.bias) / summary.bias_se)
        expected.extend(count for _, count in summary.status_counts)
    _check_accounted(text, result.summaries, [*expected, *_note_numbers(text)])


# ----------------------------------------------------------------- phase timings


def _phase_check(rows: list[list[str]], profile: Any, *, shares: bool) -> list[float]:
    order = sorted(profile.exclusive, key=lambda key: -profile.exclusive[key])
    assert [row[0] for row in rows] == order
    values: list[float] = []
    for row, name in zip(rows, order, strict=True):
        calls, own, share, nested = row[1:]
        assert calls == f"{profile.counts[name]:,}", (name, calls)
        assert _close(own, profile.exclusive[name]), (name, own)
        assert _close(nested, profile.inclusive[name]), (name, nested)
        if shares:
            assert _close(share.rstrip("%"), 100 * profile.share(name)), (name, share)
            values.append(100 * profile.share(name))
        else:
            assert share == vocabulary.MISSING, (name, share)
        values.extend([profile.counts[name], profile.exclusive[name], profile.inclusive[name]])
    return values


def read_phases(profile: Any, text: str) -> None:
    """Each phase's calls and times, and the header totals, equal the profile's own."""
    total = header_value(text, vocabulary.TOTAL_TIME)
    assert total is not None and _close(total.removesuffix(" s"), profile.total_seconds)
    accounted = sum(profile.exclusive.values())
    shown = header_value(text, vocabulary.ACCOUNTED)
    assert shown is not None and _close(shown.split()[0], accounted), shown
    share = 100 * accounted / profile.total_seconds
    assert _close(shown.split("(")[1].rstrip("%)"), share), shown
    expected = [profile.total_seconds, accounted, share]
    expected += _phase_check(table_rows(text, None), profile, shares=True)
    if profile.workers is not None:
        rows = table_rows(text, vocabulary.WORKERS_TITLE)
        expected += _phase_check(rows, profile.workers, shares=False)
    names = [*profile.exclusive, *(profile.workers.exclusive if profile.workers else ())]
    _check_accounted(text, names, [*expected, *_note_numbers(text)])
