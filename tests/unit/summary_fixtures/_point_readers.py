"""Number readers for the point-estimate objects of ``point_methods`` (X89, thread T1).

Each reader parses the printed text of a short summary, table by table, and compares every
printed cell with a value it reads from the object itself, never from the summary's parts.
So a renderer that swaps two rows or two columns, or a builder that prints the wrong field,
fails here. The label-to-value map lives in this file, not in the code under test.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cleverly._summary import vocabulary

#: The value of one cell: a number, an interval, a count, a word, or ``None`` for a cell
#: that prints the missing mark or nothing.
Expected = Mapping[str, Mapping[str, Any]]


def read_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every table of the printed text: its column labels and the cells of its rows.

    The labels exclude the row-label column, whether its heading is printed or blank.
    """
    lines = text.splitlines()
    tables = []
    for index, line in enumerate(lines):
        if not line or set(line) != {"-"} or index == 0:
            continue
        heading = lines[index - 1]
        labels = re.split(r"\s{2,}", heading.strip())
        if not heading.startswith(" "):
            labels = labels[1:]
        rows = []
        for row in lines[index + 1 :]:
            if row and set(row) == {"="}:
                break
            rows.append(re.split(r"\s{2,}", row.strip()))
        tables.append((labels, rows))
    return tables


def _places(shown: str) -> int:
    return len(shown.split(".")[1]) if "." in shown else 0


def _close(shown: str, value: float) -> bool:
    if not math.isfinite(value):
        return shown in (vocabulary.MISSING, "", "nan")
    number = float(shown.replace(",", ""))
    return abs(number - value) <= 0.5 * 10 ** -_places(shown) + 1e-12


def _check_cell(shown: str, value: Any, where: tuple[Any, ...]) -> None:
    if value is None:
        assert shown in (vocabulary.MISSING, ""), where
    elif isinstance(value, tuple):
        low, high = (part.strip() for part in shown.strip("[]").split(","))
        assert _close(low, float(value[0])) and _close(high, float(value[1])), where
    elif isinstance(value, PValue):
        if shown.startswith("<"):
            assert value.value < float(shown[1:]), where
        else:
            assert _close(shown, value.value), where
    elif isinstance(value, bool | str):
        assert shown == str(value), where
    elif isinstance(value, int | np.integer):
        assert int(shown.replace(",", "")) == int(value), where
    else:
        assert _close(shown, float(value)), where


class PValue:
    """A p-value cell, which prints ``<0.0001`` below its floor."""

    def __init__(self, value: float) -> None:
        self.value = float(value)


def check_tables(text: str, expected: Sequence[Expected], *, every_row: bool = True) -> None:
    """Assert each printed table holds the expected values, matched by row label.

    Parameters
    ----------
    text : str
        The printed summary.
    expected : sequence of mapping
        Per table, in print order: the row label as printed, mapped to the value of each
        column label. A printed column whose label is not expected fails.
    every_row : bool
        Whether every expected row must print. A table that shows only some rows of a long
        path passes ``False``, and each printed row must still be expected.
    """
    tables = read_tables(text)
    assert len(tables) == len(expected), (len(tables), len(expected))
    for number, ((labels, rows), want) in enumerate(zip(tables, expected, strict=True)):
        assert rows, f"table {number} printed no rows"
        printed = set()
        for cells in rows:
            columns = labels
            key, values = cells[0], cells[1:]
            assert key in want, (number, key, list(want))
            printed.add(key)
            for label, shown in zip(columns, values, strict=False):
                assert label in want[key], (number, key, label)
                _check_cell(shown, want[key][label], (number, key, label, shown))
            # A trailing cell left empty prints nothing; it must be expected empty.
            for label in columns[len(values) :]:
                assert want[key][label] in (None, "", False), (number, key, label)
        if every_row:
            assert printed == set(want), (number, set(want) - printed)


# ----------------------------------------------------------------- per object


def estimate_values(estimate: Any, level: str = "95%") -> dict[str, Any]:
    """The estimates-table cells of one estimate, read off the estimate."""
    values: dict[str, Any] = {vocabulary.ESTIMATE: estimate.psi}
    if estimate.supplies_inference:
        values[vocabulary.SPREAD_LABELS["std_err"]] = estimate.std_error
        values[f"{level} CI"] = tuple(estimate.ci)
        values[vocabulary.P_VALUE] = PValue(estimate._reported_pvalue())
    else:
        values[vocabulary.SPREAD_LABELS["plugin_std_err"]] = estimate.plugin_std_error
    values[vocabulary.SUPPORT] = estimate.support.label
    values[vocabulary.DF] = estimate.reference_df if estimate.reference_df else "normal"
    return values


def read_estimate(estimate: Any, text: str) -> None:
    check_tables(text, [{estimate.name: estimate_values(estimate)}])


def read_wald(test: Any, text: str) -> None:
    check_tables(
        text,
        [
            {
                test.name: {
                    vocabulary.STATISTIC: test.statistic,
                    vocabulary.P_VALUE: PValue(test.pvalue),
                }
            }
        ],
    )


def _replicates(draws: Any) -> dict[str, Any]:
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    words = vocabulary.REPLICATE_COLUMNS
    return {
        words["usable"]: int(values.size),
        words["mean"]: float(values.mean()),
        words["sd"]: float(values.std(ddof=1)),
        words["smallest"]: float(values.min()),
        words["largest"]: float(values.max()),
    }


def read_bootstrap(bootstrap: Any, text: str) -> None:
    check_tables(text, [{name: _replicates(draws) for name, draws in bootstrap.draws.items()}])


def read_bootstrap_interval(summary: Any, text: str) -> None:
    kind = vocabulary.BOOTSTRAP_INTERVAL_WORDS[summary.interval]
    diagnostic = summary.diagnostic or summary.below_floor
    limits = f"{kind} range" if diagnostic else kind
    suffix = " (diagnostic)" if diagnostic else ""
    low, high = summary.ci
    check_tables(
        text,
        [
            {
                vocabulary.REPLICATES_ROW: {
                    vocabulary.REPLICATE_COLUMNS["sd"]: summary.std_error,
                    f"{limits} lower{suffix}": low,
                    f"{limits} upper{suffix}": high,
                }
            }
        ],
    )


def read_bands(bands: Any, text: str) -> None:
    check_tables(
        text,
        [
            {
                name: {vocabulary.LOWER: low, vocabulary.UPPER: high}
                for name, (low, high) in bands.bands.items()
            }
        ],
    )


def read_spread(spread: Any, text: str) -> None:
    withheld = not any(spread.inferential.values())
    se_label = vocabulary.SPREAD_PLUGIN_SE if withheld else vocabulary.SPREAD_LABELS["std_err"]
    ratio_label = vocabulary.SPREAD_RATIO_PLUGIN if withheld else vocabulary.SPREAD_RATIO
    want = {}
    for name, value in spread.items():
        against = spread.std_errors[name] if withheld or spread.inferential[name] else float("nan")
        ratio = value / against if math.isfinite(against) and against > 0 else float("nan")
        want[name] = {
            vocabulary.SPREAD_DIAGNOSTIC: value,
            se_label: against,
            ratio_label: ratio,
            vocabulary.SCALE.lower(): spread.scales[name],
        }
    check_tables(text, [want])


def read_selection(selection: Any, text: str) -> None:
    frame = selection.to_frame()
    indexed = selection.strategy in ("lasso", "learners")
    want = {}
    for k in frame["candidate"]:
        model = (
            frame["label"][k]
            if indexed
            else (frame["covariates"][k] if frame["n_covariates"][k] else "intercept only")
        )
        mark = (
            vocabulary.FINAL_MARK
            if frame["final"][k]
            else (vocabulary.CHOSEN_MARK if frame["selected"][k] else "")
        )
        want[str(k + 1)] = {
            vocabulary.TREATMENT_MODEL_COLUMN: model,
            vocabulary.IN_SAMPLE_RISK: frame["train_risk"][k],
            vocabulary.CV_RISK: frame["cv_risk"][k],
            vocabulary.CHOSEN.lower(): mark,
        }
    check_tables(text, [want], every_row=len(want) <= 20)
    printed = {cells[0] for cells in read_tables(text)[0][1]}
    assert str(selection.selected + 1) in printed
    assert {"1", str(len(selection.path))} <= printed


def read_oat(fit: Any, text: str) -> None:
    column = "treatment-model risk"
    arms = (
        {str(arm): {column: value} for arm, value in fit.treatment_risk_by_arm.items()}
        if fit.treatment_risk_by_arm is not None
        else {"every arm": {column: fit.treatment_risk}}
    )
    tables: list[Expected] = [arms]
    if len(fit.factors) > 1:
        words = {
            "P(A=a": "treatment",
            "P(Delta=1": "outcome recorded",
            "P(Delta_A=1": "treatment recorded",
        }
        tables.append(
            {
                next(w for start, w in words.items() if factor.startswith(start)): {
                    column: fit.risk_by_factor[factor]
                }
                for factor in fit.factors
            }
        )
    check_tables(text, tables)


def read_derivative(fit: Any, text: str) -> None:
    tables: list[Expected] = [
        {
            f"arm {k}": {"score before": before, "score after": after}
            for k, (before, after) in enumerate(
                zip(fit.derivative_score_before, fit.derivative_score_after, strict=True)
            )
        }
    ]
    if fit.source == "lasso":
        tables.append(
            {
                str(k + 1): {"penalty": mechanism, "partner penalty": partner}
                for k, (mechanism, partner) in enumerate(fit.penalties)
            }
        )
    check_tables(text, tables)


def read_reduced(fit: Any, text: str) -> None:
    want = {}
    for key, entries in fit.diagnostics.items():
        if entries:
            first = entries[0]
            risk = float(first.cv_risk[list(first.names).index(first.best)])
            best: Any = first.best
        else:
            best, risk = "not a super learner", float("nan")
        want[vocabulary.REDUCED_REGRESSIONS[key]] = {
            vocabulary.BEST.lower(): best,
            vocabulary.CV_RISK: None if not math.isfinite(risk) else risk,
        }
    if not want:
        want["none"] = {vocabulary.BEST.lower(): None, vocabulary.CV_RISK: None}
    check_tables(text, [want])


def read_iptw(fit: Any, text: str) -> None:
    adjusted = fit.targeted if fit.form == "targeted" else fit.one_step
    tables: list[Expected] = [
        {
            f"arm {name}": {
                "initial weighted mean": value,
                "adjusted estimate": adjusted[name][0],
            }
            for name, value in fit.untargeted.items()
        }
    ]
    if any(fit.bound_active.values()):
        tables.append(
            {key: {"rows at a probability limit": count} for key, count in fit.bound_active.items()}
        )
    check_tables(text, tables)


def read_cv(report: Any, text: str) -> None:
    frame = report.to_frame()
    columns = [c for c in frame.columns if c.endswith("std_err")]
    cv_column = next(c for c in columns if c.startswith("cv_"))
    whole_column = next(c for c in columns if not c.startswith("cv_"))
    labels = vocabulary.CV_SPREAD_LABELS
    main, folds = {}, {}
    for row in frame.to_dict("records"):
        name = row["estimand"]
        main[name] = {
            vocabulary.FOLD_AVERAGED: row["fold_evaluated_psi"],
            labels[cv_column]: row[cv_column],
            vocabulary.WHOLE_SAMPLE: row["pooled_psi"],
            labels[whole_column]: row[whole_column],
        }
        draws = report.fold_estimates[name]
        folds[name] = {
            vocabulary.FOLD_SD: row["fold_sd"],
            vocabulary.FOLD_RANGE: (min(draws), max(draws)),
        }
    check_tables(text, [main, folds])


def read_result_set(result_set: Any, text: str) -> None:
    z = result_set.intermediate_name
    want = {
        f"{name} at {z} = {level:g}": estimate_values(estimate)
        for level in result_set.levels
        for name, estimate in result_set[level].estimates.items()
    }
    check_tables(text, [want])


def read_importance(result: Any, text: str) -> None:
    counts: dict[str, int] = {}
    for entry in result.entries:
        counts[entry.candidate] = counts.get(entry.candidate, 0) + 1
    want = {}
    for entry in result.entries:
        label = (
            entry.candidate
            if counts[entry.candidate] == 1
            else f"{entry.candidate}: {entry.parameter}"
        )
        values = estimate_values(entry.estimate)
        values[vocabulary.ADJUSTED_P_VALUE] = PValue(entry.adjusted_pvalue)
        want[label] = values
    check_tables(text, [want])


def read_ensemble(diagnostics: Any, text: str) -> None:
    check_tables(
        text,
        [
            {
                name: {vocabulary.CV_RISK: risk, vocabulary.WEIGHT: weight}
                for name, risk, weight in zip(
                    diagnostics.names, diagnostics.cv_risk, diagnostics.weights, strict=True
                )
            }
        ],
    )
