"""Number readers for the longitudinal and support records of ``longitudinal_records``.

Each reader parses the printed short summary table by table and compares every printed cell
with a value read from the object's own fields, never from the summary's parts. A row name
that wraps onto a second line is joined back before it is matched.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from cleverly._summary import vocabulary

from ._point_readers import PValue, _check_cell

#: One expected row: the printed row label, then the value of each further column.
Row = tuple[str, Sequence[Any]]


def _tables(text: str) -> list[list[list[str]]]:
    """The cells of every row of every table, with a wrapped row name joined back.

    The first cell of a row is its label, or the first value when the table has no
    label column.
    """
    lines = text.splitlines()
    tables = []
    for index, line in enumerate(lines):
        if not line or set(line) != {"-"} or index == 0:
            continue
        rows: list[list[str]] = []
        for row in lines[index + 1 :]:
            if row and set(row) == {"="}:
                break
            cells = re.split(r"\s{2,}", row.strip())
            if rows and len(cells) == 1 and not row.startswith(" "):
                rows[-1][0] = f"{rows[-1][0]} {cells[0]}"
            else:
                rows.append(cells)
        tables.append(rows)
    return tables


def check_rows(text: str, expected: Sequence[Sequence[Row]]) -> None:
    """Assert each printed table holds the expected rows, in order.

    Parameters
    ----------
    text : str
        The printed summary.
    expected : sequence of sequence of Row
        Per table: each row's printed label (or first value, as text), then the expected
        value of every other cell.
    """
    tables = _tables(text)
    assert len(tables) == len(expected), (len(tables), len(expected))
    for number, (rows, want) in enumerate(zip(tables, expected, strict=True)):
        assert len(rows) == len(want), (number, rows, want)
        for cells, (label, values) in zip(rows, want, strict=True):
            assert cells[0] == label, (number, cells[0], label)
            assert len(cells) - 1 == len(values), (number, cells, values)
            for shown, value in zip(cells[1:], values, strict=True):
                _check_cell(shown, value, (number, label, shown))


def _fact(text: str, label: str) -> str:
    """The value printed after ``label:`` in the header, joined across wrapped lines.

    The value ends at the next run of two spaces, where the other header column starts.
    """
    lines = text.splitlines()
    start, line = next((i, line) for i, line in enumerate(lines) if f"{label}:" in line)
    column = line.index(f"{label}:")
    rest = line[column + len(label) + 1 :]
    value_at = column + len(label) + 1 + len(rest) - len(rest.lstrip())
    parts = [re.split(r"\s{2,}", rest.strip())[0]]
    for line in lines[start + 1 :]:
        piece = line[value_at:]
        if not line[column:value_at].isspace() or not piece.strip() or piece.startswith(" "):
            break
        parts.append(re.split(r"\s{2,}", piece.strip())[0])
    return " ".join(parts)


def _load(load: Any) -> str:
    """The score-load cell, ``effective of rows``, written from the stored mapping."""
    if load is None:
        return vocabulary.MISSING
    return f"{float(load['effective']):,.1f} of {int(load['n_targeted']):,}"


def _g(value: float) -> float:
    return float(value)


# ----------------------------------------------------------------- support


def read_support_report(report: Any, text: str) -> None:
    """The overlap of each regime: one row per regime, in the report's order."""
    rows = [
        (
            name,
            (
                _g(item.min_support_propensity),
                _g(item.max_ratio),
                _g(item.effective_sample_size),
                _load(item.score_load),
                int(item.unsupported),
            ),
        )
        for name, item in report.regimes.items()
    ]
    check_rows(text, [rows])
    assert _fact(text, "Rows") == f"{report.n:,}"
    assert _fact(text, "Regimes") == str(len(report.regimes))


def read_regime_support(item: Any, text: str) -> None:
    """One regime's overlap, as the single row of the report's table."""
    row = (
        item.name,
        (
            _g(item.min_support_propensity),
            _g(item.max_ratio),
            _g(item.effective_sample_size),
            _load(item.score_load),
            int(item.unsupported),
        ),
    )
    check_rows(text, [[row]])


def read_policy_support(row: Any, text: str) -> None:
    """One policy's overlap: the ratio quantiles, the largest, the sizes and the load."""
    quantiles = [(f"{q:.0%} quantile", (_g(v),)) for q, v in sorted(row.ratio_quantiles.items())]
    folds = (
        [(f"mean ratio, fold {i}", (_g(v),)) for i, v in enumerate(row.fold_mean_ratio, 1)]
        if len(row.fold_mean_ratio) > 1
        else []
    )
    ess = f"{row.effective_sample_size:,.0f} ({row.ess_ratio:.1%} of the rows)"
    rows = [
        *quantiles,
        ("largest", (_g(row.max_ratio),)),
        ("effective size", (ess,)),
        ("mean ratio", (_g(row.mean_ratio),)),
        *folds,
        ("score load", (_load(row.score_load),)),
    ]
    check_rows(text, [rows])
    assert _fact(text, "Rows moved") == f"{row.moved_fraction:.1%}"
    if row.min_density is not None:
        assert np.isclose(
            float(_fact(text, "Smallest density at an observed dose")), row.min_density, rtol=5e-3
        )


def read_tilt_support(row: Any, text: str) -> None:
    """One tilt's overlap: the bound in the header and the single row of the table."""
    low, high = row.guaranteed
    assert _fact(text, "Tilt weight bound") == f"[{low:.3g}, {high:.3g}]"
    factors = [f"level {level} x{factor:g}" for level, factor in row.delta if factor != 1]
    assert _fact(text, "Odds factors") == (", ".join(factors) or "none (the natural course)")
    check_rows(
        text,
        [
            [
                (
                    f"{row.max_ratio:.3g}",
                    (
                        round(row.effective_sample_size),
                        f"{row.ess_ratio:.1%}",
                        _load(row.score_load),
                        _g(row.min_propensity),
                    ),
                )
            ]
        ],
    )


# ----------------------------------------------------------------- longitudinal records


def read_check(check: Any, text: str) -> None:
    """A structural model check: the contrasts against LTMLE, then the joint test."""
    structural = np.asarray(check.structural, dtype=float)
    ltmle = np.asarray(check.nonparametric, dtype=float)
    spread = np.sqrt(np.diag(np.asarray(check.covariance, dtype=float)))
    contrasts = [
        (name, (structural[i], ltmle[i], structural[i] - ltmle[i], spread[i]))
        for i, name in enumerate(check.names)
    ]
    tables: list[list[Row]] = [contrasts]
    if check.tested:
        reference = float(check.reference_df)
        joint: list[Row] = [
            (
                f"{check.statistic:.3f}",
                (
                    int(check.df),
                    reference if np.isfinite(reference) else "infinite",
                    PValue(check.pvalue),
                ),
            )
        ]
        tables.append(joint)
    check_rows(text, tables)
    assert _fact(text, "Fold splits") == str(check.n_splits)
    assert _fact(text, "Contrasts") == str(len(check.names))


def read_gformula_report(report: Any, text: str) -> None:
    """A simulation report: the models, the simulation shares and the set-dose support."""
    from cleverly._summary import vocabulary as words

    models = [
        (
            None,
            (
                words.GFORMULA_ROLES.get(model.role, model.role),
                words.GFORMULA_RESPONSES.get(model.response, model.response),
                words.GFORMULA_CONSTANT
                if model.learner == "constant"
                else words.GFORMULA_LEARNERS.get(model.learner, model.learner),
                int(model.n_rows),
                "yes" if model.stacked else "no",
            ),
        )
        for model in report.models
    ]
    tables = _tables(text)
    # The model names are written from the key; compare everything after the name.
    for cells, (_, values) in zip(tables[0], models, strict=True):
        assert len(cells) - 1 == len(values), cells
        for shown, value in zip(cells[1:], values, strict=True):
            _check_cell(shown, value, ("models", cells[0], shown))
    rest: list[list[Row]] = []
    shares = [
        (name, (share,)) for name, share in report.monte_carlo_share.items() if np.isfinite(share)
    ]
    if shares:
        rest.append(shares)
    if report.support:
        rest.append(
            [
                (
                    row.plan,
                    (
                        row.name,
                        vocabulary.GFORMULA_RULE_DOSE if row.dose is None else f"{row.dose:g}",
                        f"{row.off_support:.1%}",
                        f"{row.window:.1%}",
                    ),
                )
                for row in report.support
            ]
        )
    for number, (rows, want) in enumerate(zip(tables[1:], rest, strict=True), 1):
        assert len(rows) == len(want), (number, rows)
        for cells, (label, values) in zip(rows, want, strict=True):
            assert cells[0] == label, (cells[0], label)
            for shown, value in zip(cells[1:], values, strict=True):
                if isinstance(value, float):
                    assert shown == f"{value:.1%}", (label, shown, value)
                else:
                    _check_cell(shown, value, (number, label, shown))
    assert len(tables) == 1 + len(rest)
    assert _fact(text, "Models") == str(len(report.models))


def read_msm_fit(fit: Any, text: str) -> None:
    """A working-model record: one coefficient per term."""
    beta = np.asarray(fit.beta, dtype=float)
    check_rows(text, [[(term, (beta[i],)) for i, term in enumerate(fit.model.terms)]])
    assert _fact(text, "Plan and time cells") == f"{len(fit.model.cells):,}"
    assert _fact(text, "Link") == fit.model.link


def read_structural_diagnostics(diagnostics: Any, text: str) -> None:
    """The positivity record: the eigenvalue and the extreme share at each time."""
    rows = []
    for index, eigenvalue in enumerate(diagnostics.smallest_eigenvalue):
        share = diagnostics.extreme_share[index]
        rows.append(
            (
                f"time {index + 1}",
                (eigenvalue, None if np.isnan(share) else f"{share:.1%}"),
            )
        )
    check_rows(text, [rows])
    assert _fact(text, "Fits with a one-class fold") == f"{diagnostics.n_constant_fits:,}"
