"""Fidelity readers for the sensitivity summaries.

Each reader parses the printed short summary, independently of the summary's own cells,
and asserts that every number equals the field it reports, read off the object. A reader
that misses a table fails, so a summary cannot drop a table without the check noticing.
"""

from __future__ import annotations

import math
import re
from typing import Any

#: A printed number: an optional sign, digits, an optional decimal part.
_NUMBER = re.compile(r"^[+-]?\d+(\.\d+)?$")


def blocks(text: str) -> list[tuple[str | None, list[str], list[list[str]]]]:
    """Every printed table as ``(title, column labels, rows of cells)``.

    A table is a label line over a rule of dashes, with its rows below it down to the next
    rule of ``=``. The line above the labels is its title unless it is a ``=`` rule.
    """
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        if not line or set(line) != {"-"}:
            continue
        labels = re.split(r"\s{2,}", lines[index - 1].strip())
        above = lines[index - 2] if index >= 2 else ""
        title = None if not above or set(above) == {"="} else above.strip()
        rows = []
        for row in lines[index + 1 :]:
            if row and set(row) == {"="}:
                break
            rows.append(re.split(r"\s{2,}", row.strip()))
        found.append((title, labels, rows))
    return found


def table(text: str, title: str | None) -> tuple[list[str], list[list[str]]]:
    """The labels and rows of the table titled ``title``; ``None`` is the first table."""
    tables = blocks(text)
    if title is None:
        return tables[0][1], tables[0][2]
    matches = [(labels, rows) for name, labels, rows in tables if name == title]
    assert matches, f"no table titled {title!r}"
    return matches[0]


def close(shown: str, value: Any) -> bool:
    """Whether a printed number equals ``value`` to half a unit of its last place."""
    if value is None or not math.isfinite(float(value)):
        return shown in {"-", "nan", "inf", "-inf"}
    assert _NUMBER.match(shown), shown
    places = len(shown.split(".")[1]) if "." in shown else 0
    return abs(float(shown) - float(value)) <= 0.5 * 10**-places + 1e-12


def pair(shown: str, low: Any, high: Any) -> None:
    """Assert a printed ``[low, high]`` equals the two values."""
    left, right = (part.strip() for part in shown.strip("[]").split(","))
    assert close(left, low), (shown, low)
    assert close(right, high), (shown, high)


def header(summary: Any, label: str) -> str:
    (value,) = [value for name, value in summary.header if name == label]
    return str(value)


def _estimate_fact(obj: Any, text: str, estimand: str, psi: float) -> None:
    shown = header(obj.summary(), "Estimate").removeprefix(f"{estimand} = ")
    assert close(shown, psi), (shown, psi)
    assert shown in text


def _robustness_rows(rows: list[list[str]], rv: Any, limit: Any, has_limit: bool) -> None:
    def value(shown: str, expected: Any) -> None:
        if expected is None:
            assert shown == "none below 1", shown
        else:
            assert close(shown, expected), (shown, expected)

    assert rows[0][0] == "estimate"
    value(rows[0][1], rv)
    if not has_limit:
        assert len(rows) == 1, rows
    else:
        assert len(rows) == 2, rows
        value(rows[1][1], limit)


def bounds(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.SensitivityBounds` summary."""
    _estimate_fact(obj, text, obj.estimand, obj.psi)
    labels, rows = table(text, None)
    ((name, psi, span, *limits),) = rows
    assert name == obj.estimand
    assert close(psi, obj.psi)
    pair(span, obj.lower, obj.upper)
    if obj._limits_derived:
        pair(limits[0], obj._ci_lower, obj._ci_upper)
    else:
        assert not limits and len(labels) == 2, labels
    _, rv_rows = table(text, "Robustness value")
    limit = obj._robustness_value_ci if obj._limits_derived else None
    _robustness_rows(rv_rows, obj.robustness_value, limit, obj._limits_derived)
    summary = obj.summary()
    assert close(header(summary, "Largest shift"), obj.bias)
    assert header(summary, "Outcome-side strength") == f"{obj.cf_y:.3f} (cf_y)"
    assert header(summary, "Treatment-side strength") == f"{obj.cf_d:.3f} (cf_d)"


def robustness(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.RobustnessValue` summary."""
    _estimate_fact(obj, text, obj.estimand, obj.psi)
    _, rows = table(text, None)
    limit_keys = [key for key in obj if key.startswith("rv") and key != "rv"]
    limit = obj[limit_keys[0]] if limit_keys else None
    _robustness_rows(rows, obj["rv"], limit, bool(limit_keys))


def elements(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.SensitivityElements` summary."""
    _, rows = table(text, None)
    expected = [obj.sigma2, obj.nu2, obj.max_bias]
    if obj.nu2_correction is not None:
        expected.append(obj.nu2_correction)
    assert len(rows) == len(expected), rows
    for row, value in zip(rows, expected, strict=True):
        assert close(row[1], value), (row, value)


def benchmark(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.BenchmarkResult` summary."""
    _estimate_fact(obj, text, obj.estimand, obj.psi_long)
    _, rows = table(text, "With and without the benchmark covariates")
    expected = [
        (obj.psi_long, obj.psi_short),
        (obj.sigma2_long, obj.sigma2_short),
        (obj.nu2_long, obj.nu2_short),
    ]
    for row, (long, short) in zip(rows, expected, strict=True):
        assert close(row[1], long) and close(row[2], short), (row, long, short)
    summary = obj.summary()
    assert header(summary, "Outcome-side strength") == f"{obj.cf_y:.3f} (cf_y)"
    assert header(summary, "Treatment-side strength") == f"{obj.cf_d:.3f} (cf_d)"
    assert header(summary, "Implied alignment") == f"{obj.rho:.3f} (rho)"


def contour(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.SensitivityContour` summary."""
    _estimate_fact(obj, text, obj.estimand, obj.psi)
    frame = obj.to_frame()
    cells = {
        (round(float(d), 12), round(float(y), 12)): float(v)
        for d, y, v in zip(frame["cf_d"], frame["cf_y"], frame["value"], strict=True)
    }
    d_values = sorted({d for d, _ in cells})
    y_values = sorted({y for _, y in cells})
    labels, rows = table(text, f"{obj.bound.capitalize()} end of the range")
    assert rows, "no grid rows"
    for row in rows:
        (y,) = [value for value in y_values if close(row[0], value)]
        for label, shown in zip(labels[1:], row[1:], strict=True):
            (d,) = [value for value in d_values if close(label, value)]
            assert close(shown, cells[(d, y)]), (row[0], label, shown)


def evalue(obj: Any, text: str) -> None:
    """An :class:`~cleverly.sensitivity.EValue` summary."""
    _, rows = table(text, None)
    low, high = obj.risk_ratio_ci
    nearest = low if obj.risk_ratio >= 1.0 else high
    assert rows[0][0] == "estimate"
    assert close(rows[0][1], obj.risk_ratio) and close(rows[0][2], obj.point), rows[0]
    assert close(rows[1][1], nearest) and close(rows[1][2], obj.limit), rows[1]


def marginal(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.MarginalSensitivityBounds` summary."""
    _, rows = table(text, None)
    assert len(rows) == len(obj.rows)
    for shown, row in zip(rows, obj.rows, strict=True):
        assert close(shown[0], row.gamma)
        assert close(shown[1], row.lower) and close(shown[2], row.upper), shown
        pair(shown[3], row.lower_limit, row.upper_limit)
    if obj.null is None:
        return
    _, breakdown = table(text, f"Where the range reaches {obj.null:g}")
    grid = sorted(row.gamma for row in obj.rows)
    for shown, first in zip(breakdown, (obj.breakdown, obj.limit_breakdown), strict=True):
        if first is None:
            assert shown[1] == f"none up to {max(grid):g}", shown
            assert close(shown[2], max(grid)), shown
        else:
            assert close(shown[1], first), shown
            earlier = [value for value in grid if value < first]
            assert shown[2] == "-" if not earlier else close(shown[2], max(earlier)), shown


def tilt(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.MissingnessTilt` summary."""
    frame = obj.to_frame()
    names = list(dict.fromkeys(frame["estimand"]))
    for name in names:
        labels, rows = table(text, name)
        subset = frame[frame["estimand"] == name].reset_index(drop=True)
        assert len(rows) == len(subset)
        for shown, (_, record) in zip(rows, subset.iterrows(), strict=True):
            assert close(shown[0].split()[0].rstrip("*"), record["gamma"]), shown
            assert close(shown[1], record["psi"]), (shown, record["psi"])
            spread, low, high = _tilt_spread(record, labels[2])
            assert close(shown[2], spread), (shown, spread)
            pair(shown[3], low, high)
    summary = obj.summary()
    assert header(summary, "Missing outcomes") == f"{obj.n_missing:,} of {obj.n:,}"


def _tilt_spread(record: Any, label: str) -> tuple[float, float, float]:
    """The spread a tilt row prints: the inferential columns, or the diagnostic ones."""
    keys = ("std_err", "ci_lower", "ci_upper")
    if "diagnostic" in label or not math.isfinite(float(record.get("std_err", math.nan))):
        keys = ("plugin_std_err", "plugin_interval_lower", "plugin_interval_upper")
    return tuple(float(record[key]) for key in keys)  # type: ignore[return-value]


def surface(obj: Any, text: str) -> None:
    """A :class:`~cleverly.sensitivity.SimulatedConfoundingResult` summary."""
    _estimate_fact(obj, text, obj.estimand, obj.original_estimate)
    _, movement = table(text, "Movement")
    _, associations = table(text, "Associations")
    assert len(movement) == len(associations) == len(obj.cells)
    for number, (row, linked, cell) in enumerate(
        zip(movement, associations, obj.cells, strict=True), start=1
    ):
        assert row[0] == linked[0] == str(number)
        strengths = cell.treatment_strength
        strengths = strengths if isinstance(strengths, tuple) else (strengths,)
        for shown, value in zip(row[1:], strengths, strict=False):
            assert close(shown, value), (row, value)
        if cell.failure is None:
            assert close(row[-2], cell.estimate) and close(row[-1], cell.displacement), row
        else:
            assert row[-2:] == ["failed", cell.failure.error_type], row
        association = cell.induced_treatment_association
        values = association if isinstance(association, tuple) else (association,)
        for shown, value in zip(linked[1:], values, strict=False):
            assert close(shown, value), (linked, value)
        if obj.population == "perturbed_treatment_group":
            assert close(linked[-1], cell.target_population_fraction), linked
    if obj.calibrations:
        _, rows = table(text, "Observed covariates on the same scale")
        for shown, calibration in zip(rows, obj.calibrations, strict=True):
            assert shown[0] == calibration.covariate
            assert close(shown[-1], calibration.strength), shown
    summary = obj.summary()
    assert header(summary, "Cells") == f"{len(obj.cells)}, {len(obj.failures)} failed"
