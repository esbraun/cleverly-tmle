"""Effect curves and external-score checks whose summaries the contract test checks (P1 (a)).

A projection curve of a conditional contrast on a compact trial-like frame, with and without a
band over a declared grid, on the log-ratio scale, and with its in-sample predictions moved
outside the range of a difference of two means, so each curve note code fires. One check of an
external score on each scale, a comparison of two scores, and a comparison on two clusters, whose
rows carry no interval.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Cosine, Linear, Modifiers, Sieve

from . import Case
from ._check_fidelity import close, header_values, text_tables

__all__ = [
    "FIXTURES",
    "curve_checks",
    "read_effect_curve",
    "read_ep_curve",
    "read_score_assessment",
]

_GRID = {"age": [30.0, 50.0, 70.0]}


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 300
    age = rng.uniform(18, 90, n)
    w2 = rng.normal(size=n)
    a = rng.binomial(1, expit(0.3 * w2))
    y = rng.binomial(1, expit(-0.5 + 0.4 * w2 + a * (age - 50) / 40))
    return pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "age": age,
            "W2": w2,
            "score": 0.01 * (age - 50) + 0.05 * w2,
            "other": 0.005 * (age - 50),
            "site": np.arange(n) % 2,
        }
    )


@functools.cache
def _effect(scale: str) -> Any:
    study = CausalStudy(
        _frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("age", "W2"))
    )
    return study.identify(
        ConditionalContrast(
            Modifiers(["age"], domains={"age": (18, 90)}), scale=scale, model=Linear(["age"])
        )
    )


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "random_state": 0,
    }


@functools.cache
def _fit(scale: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _effect(scale).estimate(**_learners())


def curve_plain() -> Any:
    """The difference-scale curve with no grid, so it carries no band."""
    return _fit("difference").effect_curve()


def curve_band() -> Any:
    """The difference-scale curve with a band over three declared ages."""
    return _fit("difference").effect_curve(grid=_GRID, random_state=0)


def curve_log_ratio() -> Any:
    """The log-ratio curve with a band."""
    return _fit("log_ratio").effect_curve(grid=_GRID, random_state=0)


def curve_outside() -> Any:
    """A curve whose in-sample predictions are stretched past the outcome's range.

    The projection is not clipped, so a steep basis can leave ``[-width, width]``. The
    stretch is synthetic: it moves the printed quantiles and the outside share together.
    """
    curve = curve_plain()
    return replace(curve, in_sample=np.asarray(curve.in_sample) * 4.0)


@functools.cache
def _ep_fit() -> Any:
    study = CausalStudy(
        _frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("age", "W2"))
    )
    effect = study.identify(ConditionalContrast(Modifiers(["age"], domains={"age": (18, 90)})))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate("ep", sieve=Sieve(Cosine(["age"], 3), size=2), **_learners())


def curve_ep() -> Any:
    """The EP learner's difference-scale curve at a fixed sieve size."""
    return _ep_fit()


def curve_ep_clipped() -> Any:
    """The EP curve with a synthetic clipped share, so its clip note fires."""
    return replace(curve_ep(), clipped_share=0.25)


def score_difference() -> Any:
    """The check of an external score on the difference scale."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _effect("difference").assess_score(
            "score", computed_from=["age", "W2"], score_kind="external", **_learners()
        )


def score_log_ratio() -> Any:
    """The check of an external score on the log-ratio scale."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _effect("log_ratio").assess_score(
            "score", computed_from=["age", "W2"], score_kind="external", **_learners()
        )


def score_paired() -> Any:
    """The comparison of two external scores on the difference scale."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _effect("difference").assess_score(
            "score",
            computed_from=["age", "W2"],
            score_kind="external",
            compare="other",
            compare_computed_from=["age"],
            **_learners(),
        )


def score_two_clusters() -> Any:
    """A comparison on two clusters: every row reports its point with no interval."""
    study = CausalStudy(
        _frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("age", "W2"), cluster="site"),
    )
    effect = study.identify(
        ConditionalContrast(Modifiers(["age"], domains={"age": (18, 90)}), model=Linear(["age"]))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.assess_score(
            "score",
            computed_from=["age", "W2"],
            score_kind="external",
            compare="other",
            compare_computed_from=["age"],
            **_learners(),
        )


@functools.cache
def curve_checks() -> dict[str, Any]:
    """The held-out checks of a curve over two modifiers, by class name.

    They print a one-line repr and convert with ``to_frame``; they have no summary, so the
    coverage test builds them here from the curve that holds them.
    """
    study = CausalStudy(
        _frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("age", "W2")),
    )
    two = study.identify(
        ConditionalContrast(
            Modifiers(["age", "W2"], domains={"age": (18, 90)}), model=Linear(["age", "W2"])
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checks = two.estimate(**_learners()).effect_curve().diagnostics
        return {
            "CurveCalibration": checks.calibration(n_folds=2, random_state=0),
            "CurveRiskComparison": checks.risk_comparison(n_folds=2, random_state=0),
            "ModifierStability": checks.modifier_stability(n_folds=2, random_state=0),
            "DebiasingScores": curve_ep().diagnostics.score_equations(),
        }


def _table(text: str, title: str) -> tuple[list[str], list[list[str]]]:
    for found, labels, rows in text_tables(text):
        if found.startswith(title):
            return labels, rows
    raise AssertionError(f"no table {title!r} in the summary")


def read_effect_curve(curve: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches ``curve``.

    Parameters
    ----------
    curve : EffectCurveResult
        The curve.
    text : str
        Its short printed summary.
    """
    facts = header_values(text)
    assert facts["Rows"] == f"{int(curve.in_sample.size):,}"
    _, rows = _table(text, "Coefficients")
    se = np.sqrt(np.diag(curve.model.covariance))
    assert len(rows) == len(curve.model.terms)
    for cells, term, beta, error in zip(rows, curve.model.terms, curve.model.beta, se, strict=True):
        assert cells[0] == term
        assert close(cells[1], float(beta)), (cells, beta)
        assert close(cells[2], float(error)), (cells, error)
    _, rows = _table(text, "In-sample predictions")
    quantiles = [float(cells[0]) for cells in rows]
    spread = np.quantile(curve.in_sample, quantiles)
    for cells, value in zip(rows, spread, strict=True):
        assert close(cells[1], float(value)), (cells, value)


def read_ep_curve(curve: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches the EP ``curve``.

    Parameters
    ----------
    curve : EffectCurveResult
        The EP curve, which prints no coefficients.
    text : str
        Its short printed summary.
    """
    facts = header_values(text)
    assert facts["Rows"] == f"{int(curve.in_sample.size):,}"
    _, rows = _table(text, "In-sample predictions")
    quantiles = [float(cells[0]) for cells in rows]
    spread = np.quantile(curve.in_sample, quantiles)
    for cells, value in zip(rows, spread, strict=True):
        assert close(cells[1], float(value)), (cells, value)


def read_score_assessment(check: Any, text: str) -> None:
    """Assert that every number printed in ``text`` matches ``check``.

    Parameters
    ----------
    check : ScoreAssessment
        The check.
    text : str
        Its short printed summary.
    """
    facts = header_values(text)
    assert facts["Status"] == check.status
    labels, rows = _table(text, "Estimates")
    assert labels == ["score", "parameter", "estimate", "std err", "lower", "upper", "status"]
    expected = check.rows()
    shown = [(s, "risk_difference" if n == check.paired_name else n) for s, n, _ in expected]
    assert [(cells[0], cells[1]) for cells in rows] == shown
    for cells, (_, _, estimate) in zip(rows, expected, strict=True):
        assert close(cells[2], float(estimate.psi)), (cells, estimate.psi)
        if estimate.supplies_inference:
            low, high = estimate.ci
            for shown, value in zip(cells[3:6], (estimate.std_error, low, high), strict=True):
                assert close(shown, float(value)), (cells, value)
        else:
            assert cells[3:6] == ["*", "*", "*"], cells
        assert cells[6] == estimate.support.label


FIXTURES: dict[str, Case] = {
    "curve_plain": Case(curve_plain, names=("age",), fidelity=read_effect_curve, sectioned=False),
    "curve_band": Case(curve_band, names=("age",), fidelity=read_effect_curve),
    "curve_log_ratio": Case(curve_log_ratio, names=("age",), fidelity=read_effect_curve),
    "curve_outside": Case(
        curve_outside, names=("age",), fidelity=read_effect_curve, sectioned=False
    ),
    "curve_ep": Case(curve_ep, names=("age",), fidelity=read_ep_curve, sectioned=False),
    "curve_ep_clipped": Case(
        curve_ep_clipped, names=("age",), fidelity=read_ep_curve, sectioned=False
    ),
    "score_difference": Case(score_difference, names=("score",), fidelity=read_score_assessment),
    "score_log_ratio": Case(score_log_ratio, names=("score",), fidelity=read_score_assessment),
    "score_paired": Case(score_paired, names=("score", "other"), fidelity=read_score_assessment),
    "score_two_clusters": Case(
        score_two_clusters, names=("score", "other"), fidelity=read_score_assessment
    ),
}
