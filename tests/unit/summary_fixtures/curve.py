"""Truncation curves whose summaries the contract test checks (X89 T3).

``result.diagnostics.truncation_curve()`` on a compact point fit with one estimand, on a fit
with five estimands, on a fit whose limits cap units, on a collaborative fit whose curve
carries a diagnostic spread, and on a longitudinal fit, whose curve counts capped score cells.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

from cleverly._summary import vocabulary
from cleverly.utils.frames import as_frame

from . import Case, point_methods, point_tmle
from . import longitudinal as longitudinal_fits
from ._check_fidelity import close, header_values, text_tables
from .battery import _fit

__all__ = ["FIXTURES", "read_curve"]


def _records(curve: Any) -> list[dict[str, Any]]:
    columns = as_frame(curve.to_frame()).to_dict(as_series=False)
    return [
        dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)
    ]


def _read_limits(shown: str, lower: float, upper: float, fitted: bool) -> None:
    assert shown.endswith("*") == bool(fitted), (shown, fitted)
    low, high = re.findall(r"-?\d+\.\d+", shown)
    assert close(low, lower) and close(high, upper), (shown, lower, upper)


def read_curve(curve: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches ``curve``.

    Parameters
    ----------
    curve : TruncationCurve
        The curve.
    text : str
        Its short printed summary.
    """
    records = _records(curve)
    names = list(dict.fromkeys(str(record["estimand"]) for record in records))
    facts = header_values(text)
    pairs = {
        (record.get("lower_bound", record.get("bound")), record["upper_bound"])
        for record in records
    }
    assert facts[vocabulary.CURVE_LIMITS_TRIED] == str(len(pairs))
    if len(names) == 1:
        first = records[0]
        assert facts[vocabulary.CURVE_ESTIMATE] == names[0]
        assert close(facts[vocabulary.CURVE_FITTED_ESTIMATE], first["fitted_psi"])
        low, high = re.findall(r"\d+\.\d+", facts[vocabulary.CURVE_FITTED_LIMITS])
        assert close(low, first["fitted_lower_bound"]) and close(high, first["fitted_upper_bound"])
    else:
        assert facts[vocabulary.CURVE_ESTIMATES] == str(len(names))
    [(_, labels, rows)] = text_tables(text)
    assert len(rows) == len(records)
    offset = 1 if len(names) > 1 else 0
    for cells, record in zip(rows, records, strict=True):
        if offset:
            assert cells[0] == str(record["estimand"]), (cells, record["estimand"])
        lower = record.get("lower_bound", record.get("bound"))
        _read_limits(cells[offset], lower, record["upper_bound"], record["is_fitted_bound"])
        assert close(cells[offset + 1], record["psi"]), (cells, record["psi"])
        assert close(cells[offset + 2], record["delta_from_fitted"]), cells
        capped = cells[offset + 3]
        if "truncated_score_cells" in record:
            assert capped == (
                f"{record['truncated_score_cells']:,} of {record['evaluated_score_cells']:,}"
            )
        else:
            shown = float(capped.rstrip("%")) / 100
            assert math.isclose(shown, record["truncated_fraction"], abs_tol=0.0005), capped
    assert labels[-1] == "capped"


def _curve(build: Any, *args: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build().diagnostics.truncation_curve(*args)


FIXTURES: dict[str, Case] = {
    "curve_point": Case(lambda: _curve(_fit), fidelity=read_curve),
    "curve_estimands": Case(lambda: _curve(point_tmle.canonical), fidelity=read_curve),
    "curve_capped": Case(lambda: _curve(point_tmle.capped), fidelity=read_curve),
    "curve_collaborative": Case(lambda: _curve(point_methods.greedy), fidelity=read_curve),
    "curve_longitudinal": Case(
        lambda: _curve(longitudinal_fits.means, [0.01, 0.05, (0.1, 0.9)]),
        fidelity=read_curve,
    ),
}
