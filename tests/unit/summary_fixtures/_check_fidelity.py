"""Readers that check every number a check or refutation summary prints (X89 T3).

A reader parses the printed text, not the ``Summary`` parts, and compares each number in the
header and the tables with the value read independently from the object: its ``to_frame()``
or its attributes, never a ``Cell.value``. A table the reader does not know fails, so a new
table cannot slip through unread. Numbers inside notes and full-detail sections are checked
by ``tests/unit/test_check_summaries.py``.
"""

from __future__ import annotations

import math
import re
from typing import Any

from cleverly._summary import vocabulary

__all__ = [
    "header_values",
    "read_corrections",
    "read_equations",
    "read_influence",
    "read_models",
    "read_overlap",
    "read_refutation",
    "read_weights",
    "sci_close",
    "text_tables",
]

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])")


def _rule(line: str, char: str) -> bool:
    return bool(line) and set(line) == {char}


def header_values(text: str) -> dict[str, str]:
    """The header facts of a printed summary, label to printed value.

    Parameters
    ----------
    text : str
        The printed summary.

    Returns
    -------
    dict of str to str
        One entry per header fact.
    """
    lines = text.splitlines()
    first = next(index for index, line in enumerate(lines) if _rule(line, "="))
    facts: dict[str, str] = {}
    for line in lines[first + 1 :]:
        if _rule(line, "="):
            break
        for part in re.split(r"\s{3,}(?=[A-Z][^:]*:)", line.strip()):
            label, _, value = part.partition(":")
            facts[label.strip()] = value.strip()
    return facts


def text_tables(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Every table of a printed summary: its title, its column labels and its rows.

    Parameters
    ----------
    text : str
        The printed summary.

    Returns
    -------
    list of tuple
        ``(title, labels, rows)``, the title empty when the table has none. A row is the
        list of its cells, split on runs of two or more spaces.
    """
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        if not _rule(line, "-") or index == 0:
            continue
        labels = re.split(r"\s{2,}", lines[index - 1].strip())
        title = lines[index - 2].strip() if index >= 2 and not _rule(lines[index - 2], "=") else ""
        rows: list[list[str]] = []
        for row in lines[index + 1 :]:
            if _rule(row, "=") or not row.strip():
                break
            cells = re.split(r"\s{2,}", row.strip())
            if rows and len(cells) == 1 and len(labels) > 1:
                # A wrapped row: a last column of words continues indented, a row name
                # continues at the margin.
                at = -1 if row.startswith(" ") else 0
                rows[-1][at] = f"{rows[-1][at]} {cells[0]}"
                continue
            rows.append(cells)
        found.append((title, labels, rows))
    return found


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def close(shown: str, value: float | None) -> bool:
    """Whether a printed number agrees with ``value`` to half a unit of its last place.

    Parameters
    ----------
    shown : str
        The printed number; ``"-"`` matches a missing or non-finite value.
    value : float or None
        The value read from the object.

    Returns
    -------
    bool
        Whether the two agree.
    """
    missing = value is None or (isinstance(value, float) and not math.isfinite(value))
    if shown == vocabulary.MISSING:
        return missing
    if missing:
        return False
    if shown.startswith("<"):
        return float(value) < float(shown[1:])
    return (
        abs(float(shown.replace(",", "")) - float(value)) <= 0.5 * 10 ** -_decimals(shown) + 1e-12
    )


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def _test_names(report: Any) -> dict[str, Any]:
    return {
        vocabulary.REFUTATION_NAMES.get(test.name, test.name.replace("_", " ")): test
        for test in report.tests
    }


_RESULTS = {
    "passed": vocabulary.CHECK_RESULTS["passed"],
    "failed": vocabulary.CHECK_RESULTS["failed"],
    "descriptive": vocabulary.CHECK_RESULTS["descriptive"],
}


def read_refutation(report: Any, text: str) -> None:
    """Assert that every number a refutation summary prints equals the report's.

    Parameters
    ----------
    report : RefutationResult
        The report.
    text : str
        Its short summary.
    """
    frame = report.to_frame().set_index("test")
    names = _test_names(report)
    facts = header_values(text)
    estimand, _, shown = facts[vocabulary.ESTIMATE_TESTED].partition(" = ")
    assert estimand == report.estimand, facts
    for value in frame["original"]:
        assert close(shown, float(value)), (shown, value)
    assert int(facts[vocabulary.TESTS_RUN].replace(",", "")) == len(frame)
    repeats = [int(item.replace(",", "")) for item in _numbers(facts[vocabulary.REPEATS_PER_TEST])]
    assert repeats == sorted({min(repeats), max(repeats)})
    counts = {test.n_replicates for test in report.tests}
    assert (min(repeats), max(repeats)) == (min(counts), max(counts)), (repeats, counts)
    verdict = report.passed
    expected = "descriptive" if verdict is None else ("passed" if verdict else "failed")
    assert facts[vocabulary.RESULT] == _RESULTS[expected]

    tables = text_tables(text)
    assert tables, "the summary printed no table"
    _, labels, rows = tables[0]
    assert labels == ["test", "mean", "SD", "expected", "result"], labels
    assert [row[0] for row in rows] == list(names), rows
    for row in rows:
        test = names[row[0]]
        record = frame.loc[test.name]
        assert close(row[1], record["refuted_mean"]), (row, record["refuted_mean"])
        assert close(row[2], record["refuted_sd"]), (row, record["refuted_sd"])
        promised = _numbers(str(record["expectation"]))
        for number in _numbers(row[3]):
            assert any(close(number, float(item.replace(",", ""))) for item in promised), (
                row,
                record["expectation"],
            )
        assert row[4] == _RESULTS[str(record["verdict"])], (row, record["verdict"])

    for title, labels, rows in tables[1:]:
        if title == vocabulary.STATISTICS_TITLE:
            _read_statistics(report, frame, names, labels, rows)
        elif title == vocabulary.READINGS_TITLE:
            _read_readings(names, labels, rows)
        else:
            raise AssertionError(f"no reader for the table {title!r}")


def _read_statistics(
    report: Any, frame: Any, names: dict[str, Any], labels: list[str], rows: list[list[str]]
) -> None:
    assert labels == ["test", "statistic", "value", "p-value"], labels
    expected: list[tuple[str, str, float, float | None]] = []
    for test in report.tests:
        record = frame.loc[test.name]
        statistic = record["statistic"]
        if statistic is not None and not (isinstance(statistic, float) and math.isnan(statistic)):
            label = _label(str(record["statistic_name"]))
            expected.append((test.name, label, float(statistic), record["statistic_pvalue"]))
        expected.extend(
            (test.name, _label(name), float(value), None) for name, value in test.extras
        )
    assert len(rows) == len(expected), (rows, expected)
    for row, (name, label, value, pvalue) in zip(rows, expected, strict=True):
        assert names[row[0]].name == name, (row, name)
        assert row[1].split(" against ")[0] == label.split(" against ")[0], (row, label)
        assert close(row[2], value), (row, value)
        assert close(row[3], pvalue), (row, pvalue)


def _label(name: str) -> str:
    return vocabulary.STATISTIC_NAMES.get(name, name)


def _read_readings(names: dict[str, Any], labels: list[str], rows: list[list[str]]) -> None:
    assert labels == ["test", "reading", "estimate", "95% CI", "allowed band", "inside"], labels
    expected = [(test, reading) for test in names.values() for reading in test.readings]
    assert len(rows) == len(expected), (rows, expected)
    for row, (test, reading) in zip(rows, expected, strict=True):
        assert names[row[0]] is test, row
        assert row[1] == _label(reading.name), (row, reading.name)
        assert close(row[2], reading.estimate), (row, reading)
        low, high = _numbers(row[3])
        assert close(low, reading.lower) and close(high, reading.upper), (row, reading)
        band_low, band_high = _numbers(row[4])
        assert close(band_low, reading.band[0]) and close(band_high, reading.band[1]), row
        assert row[5] == ("yes" if reading.inside else "no"), (row, reading)


# ----------------------------------------------------------------- equations and corrections


def sci_close(shown: str, value: float | None, reference: float = float("nan")) -> bool:
    """Whether a number printed in scientific notation agrees with ``value``.

    A printed zero also matches a residue far below ``reference`` (five orders), which the
    summary prints as zero so that it reads the same on every machine; ``"< x"`` matches a
    value at or below ``x``.

    Parameters
    ----------
    shown : str
        The printed number, ``"-"`` for a missing one.
    value : float or None
        The value read from the object.
    reference : float
        The bar the value is read against.

    Returns
    -------
    bool
        Whether the two agree.
    """
    missing = value is None or not math.isfinite(float(value))
    if shown == vocabulary.MISSING:
        return missing
    if missing:
        return False
    number = float(value)  # type: ignore[arg-type]
    if shown.startswith("<"):
        return abs(number) <= float(shown[1:])
    mantissa, _, exponent = shown.partition("e")
    printed = float(shown)
    if printed == 0.0:
        # A printed zero is a residue far below the bar, or an exact zero; nothing larger.
        if math.isfinite(reference) and reference > 0:
            return abs(number) < reference * 1e-5
        return number == 0.0
    unit = 10 ** (int(exponent or 0) - _decimals(mantissa))
    return abs(printed - number) <= 0.5 * unit * (1 + 1e-9)


def _result_word(passed: bool, threshold: float) -> str:
    if not math.isfinite(threshold):
        return vocabulary.CHECK_RESULTS["descriptive"]
    return vocabulary.CHECK_RESULTS["passed" if passed else "failed"]


def _read_number_cells(cells: list[str], record: dict[str, Any]) -> None:
    error, allowed, ratio, result = cells
    threshold = float(record["threshold"])
    assert sci_close(error, abs(float(record["score"])), threshold), (cells, record)
    assert sci_close(allowed, threshold), (cells, record)
    assert sci_close(ratio, float(record["ratio"]), 1.0), (cells, record)
    assert result == _result_word(bool(record["passed"]), threshold), (cells, record)


#: What each correction and each row kind is called in print, written out here rather than
#: read from the vocabulary, so a wrong entry there fails the reader.
_CORRECTION_WORDS = {
    "D*_Q": "outcome",
    "D*_Y": "outcome",
    "D*_g": "treatment",
    "D*_A": "treatment",
    "D*_M": "recording",
}
_CHECK_WORDS = {"correction": "size", "identity": "identity", "diagnostic": "described"}
_GROUP_WORDS = {"mean": "outcome means"}
_PART_WORDS = {
    "mechanism": "treatment model",
    "reduced": "reduced regression",
    "strata": "strata",
}


def _read_step_label(shown: str, name: str) -> None:
    """A step row's label names its group and its part in plain words."""
    stem = re.sub(r"\[draw \d+\]$", "", name)
    match = re.fullmatch(r"(\S+)(?: \((\w+)\))?", stem)
    assert match is not None, name
    group, part = match[1], match[2]
    expected = _GROUP_WORDS.get(group, group)
    if part is not None:
        expected += f" ({_PART_WORDS[part]})"
    assert shown == expected, (shown, name)


def read_equations(check: Any, text: str) -> None:
    """Assert that every number an equation-check summary prints equals the check's.

    Parameters
    ----------
    check : ScoreCheck
        The check.
    text : str
        Its short summary.
    """
    frame = check.to_frame()
    records = [
        dict(zip(frame.columns, values, strict=True)) for values in frame.itertuples(index=False)
    ]
    facts = header_values(text)
    result = "descriptive" if not records else ("passed" if check.passed else "failed")
    assert facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS[result]
    assert int(facts[vocabulary.EQUATIONS].replace(",", "")) == len(records)
    assert int(facts[vocabulary.ROWS].replace(",", "")) == check.n
    allowed = facts[vocabulary.ALLOWED_ERROR]
    if check.declared:
        assert check.stopping in allowed, allowed
    else:
        assert close(_numbers(allowed)[0], check.tolerance), allowed

    def recomputed(record: dict[str, Any]) -> bool:
        return record["kind"] in ("correction", "identity") or record["method"] == "recomputed"

    expected = {
        vocabulary.STEP_EQUATIONS_TITLE: [r for r in records if r["kind"] == "fluctuation"],
        vocabulary.CORRECTIONS_TITLE: [r for r in records if recomputed(r)],
        vocabulary.ESTIMATE_EQUATIONS_TITLE: [
            r for r in records if r["kind"] != "fluctuation" and not recomputed(r)
        ],
    }
    tables = text_tables(text)
    if not records:
        assert [(labels, rows) for _, labels, rows in tables] == [
            (["equations"], [["none reported"]])
        ], tables
        return
    assert [title for title, _, _ in tables] == [
        title for title, rows in expected.items() if rows
    ], tables
    for title, labels, rows in tables:
        wanted = expected[title]
        assert len(rows) == len(wanted), (title, rows)
        splits = ["split"] if "split" in labels else []
        lead = {
            vocabulary.STEP_EQUATIONS_TITLE: ["equation", *splits],
            vocabulary.ESTIMATE_EQUATIONS_TITLE: ["estimate"],
        }.get(title, ["correction", "arm", *splits, "check"])
        assert labels == [*lead, "error", "allowed", "ratio", "result"], labels
        for row, record in zip(rows, wanted, strict=True):
            name = str(record["name"])
            draw = re.search(r"\[draw (\d+)\]$", name)
            if "split" in labels:
                assert draw is not None and int(row[lead.index("split")]) == int(draw[1]) + 1
            if title == vocabulary.ESTIMATE_EQUATIONS_TITLE:
                assert row[0] == name, (row, record)
            elif title == vocabulary.STEP_EQUATIONS_TITLE:
                _read_step_label(row[0], name)
            else:
                part = re.search(r"\((D\*_\w)\)\[(.+?)\]( identity)?(\[draw \d+\])?$", name)
                assert part is not None, name
                assert row[0] == _CORRECTION_WORDS[part[1]], (row, record)
                assert row[1] == part[2], (row, record)
                assert row[lead.index("check")] == _CHECK_WORDS[str(record["kind"])], row
            _read_number_cells(row[len(lead) :], record)


def read_corrections(check: Any, text: str) -> None:
    """Assert that every number a correction-check summary prints equals the check's.

    Parameters
    ----------
    check : CorrectionCheck
        The check.
    text : str
        Its short summary.
    """
    frame = check.to_frame()
    records = [
        dict(zip(frame.columns, values, strict=True)) for values in frame.itertuples(index=False)
    ]
    facts = header_values(text)
    result = "descriptive" if not records else ("passed" if check.passed else "failed")
    assert facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS[result]
    assert int(facts[vocabulary.EQUATIONS].replace(",", "")) == sum(
        bool(record["solved"]) for record in records
    )
    assert int(facts[vocabulary.ROWS].replace(",", "")) == check.n
    assert close(_numbers(facts[vocabulary.ALLOWED_ERROR])[0], check.tolerance)
    assert facts[vocabulary.PREDICTIONS] == ("cross-fitted" if check.cross_fitted else "in-sample")
    tables = text_tables(text)
    assert len(tables) == 1, tables
    _, labels, rows = tables[0]
    if not records:
        assert labels == ["corrections"] and rows == [["none reported"]], (labels, rows)
        return
    splits = "split" in labels
    assert splits == (len({record["draw"] for record in records}) > 1)
    lead = ["correction", "arm", *(["split"] if splits else [])]
    assert labels == [*lead, "solved", "in SE", "difference", "cap bias"], labels
    assert len(rows) == len(records), rows
    identity = check.identity_threshold
    for row, record in zip(rows, records, strict=True):
        assert row[0] == _CORRECTION_WORDS[str(record["equation"])], (row, record)
        assert row[1] == str(record["arm"]), (row, record)
        if splits:
            assert int(row[2]) == int(record["draw"]) + 1, (row, record)
        solved, curve, difference, cap = row[len(lead) :]
        bar = check.threshold if record["solved"] else 0.0
        assert sci_close(solved, float(record["stored"]), bar), (row, record)
        assert sci_close(curve, float(record["reported"]), bar), (row, record)
        assert sci_close(difference, float(record["residual"]), identity), (row, record)
        assert sci_close(cap, float(record["clip_bias"]), identity), (row, record)


def read_influence(report: Any, text: str) -> None:
    """Assert that every number an influence-report summary prints equals the report's.

    Parameters
    ----------
    report : InfluenceReport
        The report.
    text : str
        Its short summary.
    """
    frame = report.to_frame()
    records = [
        dict(zip(frame.columns, values, strict=True)) for values in frame.itertuples(index=False)
    ]
    facts = header_values(text)
    label = vocabulary.CLUSTERS if report.clustered else vocabulary.ROWS
    if records:
        assert int(facts[label].replace(",", "")) == int(records[0]["n"])
    assert int(facts[vocabulary.ESTIMATES_MEASURED]) == len(records)
    tables = text_tables(text)
    if not records:
        assert [(labels, rows) for _, labels, rows in tables] == [
            (["estimates"], [["none reported"]])
        ], tables
        return
    assert len(tables) == 1, tables
    _, labels, rows = tables[0]
    units = "clusters" if report.clustered else "rows"
    assert labels == [
        "estimate",
        f"effective {units}",
        "skewness",
        "SE uncertainty",
        "coverage loss",
        "flag",
    ], labels
    assert len(rows) == len(records), rows
    for row, record in zip(rows, records, strict=True):
        # A long name wraps at "[" or a space, and the reader joins its pieces with a space.
        assert row[0].replace(" ", "") == str(record["name"]).replace(" ", ""), (row, record)
        for shown, key in zip(
            row[1:5], ("kish", "skewness", "variance_rel_sd", "edgeworth_loss"), strict=True
        ):
            assert close(shown, float(record[key])), (key, shown, record)
        sparse = float(record["variance_rel_sd"]) > 0.125
        assert row[5] == ("noisy" if sparse else vocabulary.MISSING), (row, record)


#: The weight report's table rows, label to ``to_dict`` key, written out here so a label
#: paired with the wrong number fails.
_WEIGHT_ROWS = {
    "design effect": "design_effect",
    "smallest weight / mean": "min_ratio",
    "largest weight / mean": "max_ratio",
    "variation (CV)": "cv",
    "share held by the top 1%": "top_1pct",
    "share held by the top 5%": "top_5pct",
    "rows with zero weight": "n_zero",
}


def read_weights(report: Any, text: str) -> None:
    """Assert that every number a weight-report summary prints equals the report's.

    Parameters
    ----------
    report : WeightReport
        The report.
    text : str
        Its short summary.
    """
    values = report.to_dict()
    facts = header_values(text)
    assert facts[vocabulary.WEIGHT_COLUMN] == (values["name"] or vocabulary.MISSING)
    assert facts[vocabulary.WEIGHT_KIND] == values["kind"]
    assert int(facts[vocabulary.ROWS].replace(",", "")) == values["n"]
    assert facts[vocabulary.WEIGHTS_TREATED] == ("estimated" if values["estimated"] else "fixed")
    assert close(facts[vocabulary.EFFECTIVE_ROWS], values["effective_n"])
    tables = text_tables(text)
    assert len(tables) == 1, tables
    _, labels, rows = tables[0]
    assert labels == ["measure", "value"], labels
    assert [row[0] for row in rows] == list(_WEIGHT_ROWS), rows
    for label, shown in rows:
        key = _WEIGHT_ROWS[label]
        value = values[key]
        if key in ("min_ratio", "max_ratio") and shown != vocabulary.MISSING:
            # A ratio prints three significant figures.
            assert math.isclose(float(shown), value, rel_tol=5e-3), (label, shown, value)
        else:
            assert close(shown, value), (label, shown, value)


#: Plain model names, report name stem to printed words, written out here so a row paired
#: with the wrong model fails.
_MODEL_WORDS = {
    "propensity": "treatment model",
    "outcome": "outcome model",
    "missingness": "recording model",
    "intermediate": "intermediate model",
    "treatment_observation": "treatment-recording model",
}
_PROBABILITY_KEYS = {
    "AUC": "auc",
    "Brier score": "brier",
    "log loss": "log_loss",
    "calib. slope": "calibration_slope",
    "slope SE": "calibration_slope_se",
}
_REGRESSION_KEYS = {"R2": "r2", "MSE": "mse", "regression slope": "regression_slope"}


def _model_words(name: str) -> str:
    stem, _, arm = name.partition("[")
    words = _MODEL_WORDS.get(stem, stem)
    return words if not arm else f"{words} (arm {arm.rstrip(']')})"


def read_models(check: Any, text: str) -> None:
    """Assert that every number a model-check summary prints equals the check's.

    Parameters
    ----------
    check : NuisanceDiagnostics
        The check.
    text : str
        Its short summary.
    """
    frame = check.to_frame()
    records = [
        dict(zip(frame.columns, values, strict=True)) for values in frame.itertuples(index=False)
    ]
    facts = header_values(text)
    assert int(facts[vocabulary.MODELS]) == len(records)
    assert facts[vocabulary.PREDICTIONS] == check.evaluation.replace("_", " ")
    result = "warning" if check.findings else "descriptive"
    assert facts[vocabulary.RESULT] == vocabulary.CHECK_RESULTS[result]
    if check.n_repeats > 1:
        assert (
            facts[vocabulary.SPLIT_SHOWN] == f"split {check.reported_repeat} of {check.n_repeats}"
        )
    tables = {title: (labels, rows) for title, labels, rows in text_tables(text)}
    if not records and not check.repeat_spread:
        assert list(tables.values()) == [(["models"], [["none reported"]])], tables
        return
    probability = [r for r in records if r["kind"] == "probability"]
    regression = [r for r in records if r["kind"] != "probability"]
    expected_titles = []
    for title, wanted, keys in (
        (vocabulary.PROBABILITY_TABLE, probability, _PROBABILITY_KEYS),
        (vocabulary.REGRESSION_TABLE, regression, _REGRESSION_KEYS),
    ):
        if not wanted:
            continue
        expected_titles.append(title)
        labels, rows = tables[title]
        assert labels == ["model", *keys], labels
        assert len(rows) == len(wanted), rows
        for row, record in zip(rows, wanted, strict=True):
            assert row[0] == _model_words(str(record["model"])), (row, record)
            for shown, key in zip(row[1:], keys.values(), strict=True):
                assert close(shown, float(record.get(key, math.nan))), (key, shown, record)
    weights = [
        (model.name, learner, weight)
        for model in check.models
        for learner, weight in model.learner_weights.items()
        if weight > 1e-3
    ]
    if weights:
        expected_titles.append(vocabulary.LEARNER_TABLE)
        labels, rows = tables[vocabulary.LEARNER_TABLE]
        assert labels == ["model", "learner", "weight"], labels
        assert len(rows) == len(weights), rows
        for row, (name, learner, weight) in zip(rows, weights, strict=True):
            assert row[0] == _model_words(name) and row[1] == learner, row
            assert close(row[2], weight), (row, weight)
    if check.repeat_spread:
        expected_titles.append(vocabulary.SPREAD_TABLE)
        spread = check.repeat_spread_frame()
        spread_records = [
            dict(zip(spread.columns, values, strict=True))
            for values in spread.itertuples(index=False)
        ]
        labels, rows = tables[vocabulary.SPREAD_TABLE]
        assert labels[:3] == ["estimate", "splits", "SD between splits"], labels
        assert len(rows) == len(spread_records), rows
        for row, record in zip(rows, spread_records, strict=True):
            values = list(record.values())
            assert row[0] == record["estimand"], (row, record)
            assert int(row[1]) == record["n_repeats"], (row, record)
            for shown, value in zip(row[2:], values[2:], strict=True):
                assert close(shown, float(value)), (shown, record)
    assert list(tables) == expected_titles, (list(tables), expected_titles)


#: The ``Result`` of the overlap check on the treatment branch, written out here so a tier
#: paired with the wrong words fails.
_OVERLAP_RESULTS = {
    "adequate": "at most 1% of units capped",
    "strain": "1% to 5% of units capped",
    "serious": "over 5% of units capped",
}


def _limits_close(shown: str, lower: float, upper: float) -> bool:
    low, _, high = shown.strip("[]").partition(", ")
    return close(low, lower) and close(high, upper)


def _capped_close(shown: str, count: float, fraction: float) -> bool:
    if shown == vocabulary.MISSING:
        return not math.isfinite(fraction)
    number, _, percent = shown.partition(" (")
    return close(number, count) and close(percent.rstrip("%)"), 100 * fraction)


def read_overlap(report: Any, text: str) -> None:
    """Assert that every number an overlap-check summary prints equals the report's.

    Parameters
    ----------
    report : PositivityReport
        The report.
    text : str
        Its short summary.
    """
    facts = header_values(text)
    assert int(facts[vocabulary.ROWS].replace(",", "")) == report.n
    assert _limits_close(facts[vocabulary.PROBABILITY_LIMITS], *report.bounds), facts
    assert _capped_close(
        facts[vocabulary.UNITS_CAPPED], report.truncated["count"], report.truncated["fraction"]
    ), facts
    if report.grade().mechanism is None:
        assert facts[vocabulary.RESULT] == _OVERLAP_RESULTS[report.severity], facts
    else:
        stats = report.mechanisms[report.grade().mechanism]
        capped = stats["clipped_fraction"] > 0.01
        wanted = "another probability capped" if capped else "weights concentrated"
        assert facts[vocabulary.RESULT] == wanted, facts
    if report.treatment_mechanism == "known":
        assert facts[vocabulary.TREATMENT_MODEL] == "known (declared)"
    else:
        assert vocabulary.TREATMENT_MODEL not in facts
    if report.n_repeats > 1:
        assert facts[vocabulary.SPLIT_SHOWN] == f"split 1 of {report.n_repeats}"
    tables = text_tables(text)
    titles = [title for title, _, _ in tables]
    expected = [vocabulary.QUANTILE_TABLE, vocabulary.ARM_TABLE, vocabulary.TAIL_TABLE]
    if report.group_score_load:
        expected.append(vocabulary.EQUATION_TABLE)
    if report.mechanisms:
        expected.append(vocabulary.MECHANISM_TABLE)
    assert titles == expected, titles
    found = {title: (labels, rows) for title, labels, rows in tables}

    frame = report.to_frame()
    quantiles: dict[str, list[tuple[float, float]]] = {}
    for group, quantile, value in frame.itertuples(index=False):
        quantiles.setdefault(str(group), []).append((float(quantile), float(value)))
    labels, rows = found[vocabulary.QUANTILE_TABLE]
    assert [row[0] for row in rows] == list(quantiles), rows
    for row in rows:
        values = sorted(quantiles[row[0]])
        assert labels[1:] == [f"{quantile:.0%}" for quantile, _ in values], labels
        for shown, (_, value) in zip(row[1:], values, strict=True):
            assert close(shown, value), (row, values)

    labels, rows = found[vocabulary.ARM_TABLE]
    assert labels == ["arm", "rows", "effective rows", "eff. share", "top 1%", "top 5%"]
    assert [row[0] for row in rows] == [str(arm) for arm in report.effective_sample_size]
    for row, (arm, ess) in zip(rows, report.effective_sample_size.items(), strict=True):
        share = report.weight_share[arm]
        wanted = (ess["n"], ess["effective"], ess["ratio"], share["top_1pct"], share["top_5pct"])
        for shown, value in zip(row[1:], wanted, strict=True):
            assert close(shown, value), (row, wanted)

    labels, rows = found[vocabulary.TAIL_TABLE]
    assert labels == ["threshold t", "share below t", "share above 1 - t"], labels
    tails = sorted(report.tail_mass.items())
    assert len(rows) == len(tails), rows
    for row, (threshold, mass) in zip(rows, tails, strict=True):
        assert math.isclose(float(row[0]), threshold, rel_tol=1e-3), (row, threshold)
        assert close(row[1], mass["below"]) and close(row[2], mass["above"]), (row, mass)

    if report.group_score_load:
        labels, rows = found[vocabulary.EQUATION_TABLE]
        assert labels == [
            "group",
            "rows",
            "effective rows",
            "eff. share",
            "top 1%",
            "top 5%",
            "largest",
            "capped",
        ], labels
        assert [row[0] for row in rows] == list(report.group_score_load), rows
        for row, load in zip(rows, report.group_score_load.values(), strict=True):
            wanted = (
                load["n_targeted"],
                load["effective"],
                load["targeted_ratio"],
                load["top_1pct"],
                load["top_5pct"],
                load["max_load"],
            )
            for shown, value in zip(row[1:7], wanted, strict=True):
                assert close(shown, value), (row, wanted)
            assert _capped_close(row[7], load["clipped_count"], load["clipped_fraction"]), row

    if report.mechanisms:
        labels, rows = found[vocabulary.MECHANISM_TABLE]
        assert labels == [
            "probability of",
            "min",
            "5%",
            "median",
            "eff. share",
            "top 1%",
            "top 5%",
            "capped",
        ], labels
        assert [row[0] for row in rows] == [
            vocabulary.MECHANISM_LABELS[name] for name in report.mechanisms
        ], rows
        for row, stats in zip(rows, report.mechanisms.values(), strict=True):
            keys = ("min", "q05", "median", "ess_ratio", "top_1pct", "top_5pct")
            for shown, key in zip(row[1:7], keys, strict=True):
                assert close(shown, stats[key]), (row, key, stats)
            assert _capped_close(row[7], stats["clipped"], stats["clipped_fraction"]), row
