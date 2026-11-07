"""Targeting curves whose summaries the contract test checks (X92 PR-5).

An external score compared with a second one, a curve refit on a declared split, and a small
frame whose first grid point holds too few rows for an interval, so each targeting note code
fires.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Linear, Modifiers

from . import Case
from ._check_fidelity import close, header_values, text_tables

__all__ = ["FIXTURES", "read_targeting_curve"]

_LEARNERS: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(),
    "outcome_by_arm": True,
    "q_bounds": (-5.0, 6.0),
}


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    w1, w2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = w2 + a * (1 + w1) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "risk": w1 + 0.3 * w2, "age": w2})


@functools.cache
def _effect(n: int, linear: bool = False) -> Any:
    study = CausalStudy(
        _frame(n), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    return study.identify(
        ConditionalContrast(Modifiers(["W1"]), model=Linear(["W1"]) if linear else None)
    )


def compared() -> Any:
    """An external score compared with a second one on the same draws."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _effect(600).targeting_curve(
            "risk",
            computed_from=["W1", "W2"],
            score_kind="external",
            compare="age",
            compare_computed_from=["W2"],
            n_half_samples=50,
            random_state=0,
            **_LEARNERS,
        )


def split() -> Any:
    """The projection curve refit on half the rows and evaluated on the other half."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = _effect(1000, linear=True).estimate(**_LEARNERS, random_state=0).effect_curve()
        return curve.diagnostics.targeting_curve(n_half_samples=50, random_state=0)


def withheld() -> Any:
    """Three hundred rows: the first grid point holds 30 rows and reports no interval."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _effect(300).targeting_curve(
            "risk",
            computed_from=["W1", "W2"],
            score_kind="external",
            n_half_samples=50,
            random_state=0,
            **_LEARNERS,
        )


def _table(text: str, title: str) -> list[list[str]]:
    for found, _, rows in text_tables(text):
        if found.startswith(title):
            return rows
    raise AssertionError(f"no table {title!r} in the summary")


def read_targeting_curve(curve: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches ``curve``.

    Parameters
    ----------
    curve : TargetingCurve
        The curve.
    text : str
        Its short printed summary.
    """
    facts = header_values(text)
    assert facts["Rows"] == f"{curve.n_eval:,}"
    assert close(facts["Average effect"], curve.ate)
    keys = ("estimate", "std_error", "lower", "upper")
    areas = _table(text, "Areas")
    for cells, quantity in zip(areas, ("autoc", "qini"), strict=True):
        row = curve.area(quantity)
        for shown, key in zip(cells[1:5], keys, strict=True):
            assert close(shown, float(row[key])), (cells, key)
    tocs = _table(text, "TOC")
    rows = curve.rows("toc")
    assert len(tocs) == len(rows)
    for cells, row in zip(tocs, rows, strict=True):
        assert cells[0].rstrip("*") == f"{row['q']:g}"
        for shown, key in zip(cells[1:7], (*keys, "band_lower", "band_upper"), strict=True):
            assert close(shown, float(row[key])), (cells, key)
        assert cells[7] == f"{int(row['rows_in_top']):,}"


FIXTURES: dict[str, Case] = {
    "targeting_compared": Case(compared, names=("risk", "age"), fidelity=read_targeting_curve),
    "targeting_split": Case(split, names=("effect_curve:cate",), fidelity=read_targeting_curve),
    "targeting_withheld": Case(withheld, names=("risk",), fidelity=read_targeting_curve),
}
