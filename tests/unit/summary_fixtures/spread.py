"""Influence reports and weight reports whose summaries the contract test checks (X89 T3).

One fixture per table, reading and note the two reports can print, so each of their note
codes fires on at least one fixture. Most are built from synthetic curves and weight
vectors, which cost nothing; one influence report comes from a compact longitudinal fit,
whose regimens few rows follow, so the fit warns and the report points at the warning.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.data.weighting import WeightReport, WeightSpec, describe_weights
from cleverly.datasets import make_longitudinal
from cleverly.inference.sparse import InfluenceReport, influence_row
from cleverly.longitudinal import LTMLE

from . import Case
from ._check_fidelity import read_influence, read_weights

__all__ = ["FIXTURES"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def influence_clean() -> InfluenceReport:
    """Two estimates whose curves are close to normal, so neither is flagged."""
    rows = (
        influence_row("ate", _rng(1).normal(size=2000)),
        influence_row("att", _rng(2).normal(size=2000) + 0.3 * _rng(3).exponential(size=2000)),
    )
    return InfluenceReport(rows, 0.05)


def influence_flagged() -> InfluenceReport:
    """A curve that a few rows carry, beside a clean one, from a fit that warned."""
    sparse = np.zeros(400)
    sparse[:12] = _rng(4).normal(loc=3.0, size=12)
    rows = (
        influence_row("ey_regimen[always]", sparse),
        influence_row("ey_regimen[never]", _rng(5).normal(size=400)),
    )
    return InfluenceReport(rows, 0.05, warned=True)


def influence_clustered() -> InfluenceReport:
    """A clustered fit: every measure reads cluster sums, so the units are clusters."""
    cluster = np.repeat(np.arange(60), 5)
    curve = _rng(6).normal(size=300) + np.repeat(_rng(7).normal(size=60), 5)
    return InfluenceReport((influence_row("ate", curve, cluster=cluster),), 0.05, clustered=True)


def influence_bootstrap() -> InfluenceReport:
    """An estimate with an attached bootstrap, whose shape the full detail prints."""
    curve = _rng(8).gamma(2.0, size=500) - 2.0
    draws = 0.5 + _rng(9).normal(scale=0.05, size=199)
    return InfluenceReport((influence_row("ate", curve, estimate=0.5, draws=draws),), 0.1)


def influence_empty() -> InfluenceReport:
    """A report with no estimate that carries a curve."""
    return InfluenceReport((), 0.05)


@functools.cache
def _longitudinal() -> Any:
    frame, _ = make_longitudinal(n=200, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            {"always": 1, "never": 0},
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )


def influence_fit() -> InfluenceReport:
    """A longitudinal fit's report, through the result's own call."""
    return _longitudinal().diagnostics.influence()


def _weights(weights: np.ndarray, **spec: Any) -> WeightReport:
    w = np.asarray(weights, dtype=float)
    return describe_weights(w / w.mean(), WeightSpec(name="w", scale=float(w.mean()), **spec))


def weights_fixed() -> WeightReport:
    """Fixed probability weights, a few of them zero."""
    w = 0.5 + _rng(10).uniform(size=500)
    w[:4] = 0.0
    return _weights(w)


def weights_estimated() -> WeightReport:
    """Weights declared estimated, such as fitted non-response weights."""
    return _weights(1.0 / _rng(11).uniform(0.3, 0.9, size=500), estimated=True)


def weights_dominant() -> WeightReport:
    """Weights so uneven that the effective rows fall under a quarter of the rows."""
    w = np.ones(500)
    w[:10] = 60.0
    return _weights(w)


def weights_constant() -> WeightReport:
    """A constant weight column, which leaves the fit unweighted."""
    return replace(_weights(np.ones(300)), name=None)


def _influence(build: Any) -> Case:
    return Case(build, fidelity=read_influence)


def _weight(build: Any) -> Case:
    return Case(build, names=("w",), fidelity=read_weights)


FIXTURES = {
    "influence_clean": _influence(influence_clean),
    "influence_flagged": _influence(influence_flagged),
    "influence_clustered": _influence(influence_clustered),
    "influence_bootstrap": _influence(influence_bootstrap),
    "influence_empty": _influence(influence_empty),
    "influence_fit": _influence(influence_fit),
    "weights_fixed": _weight(weights_fixed),
    "weights_estimated": _weight(weights_estimated),
    "weights_dominant": _weight(weights_dominant),
    "weights_constant": Case(weights_constant, fidelity=read_weights),
}
