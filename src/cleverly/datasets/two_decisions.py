r"""Two care decisions after discharge, with the true value of a learned pair of rules.

At discharge a program decides whether to offer a patient navigation.  At day 14 it decides
whether to start home telemonitoring, after measuring a self-care score.  Some patients leave
follow-up after each decision.  The outcome is whether the patient stays home, without an
unplanned readmission, through day 30.  Navigation helps patients who are more ready to manage at
home and does worse than usual care for the least ready, and telemonitoring helps patients whose
self-care score is high, less so after navigation, so the best plan treats some patients only.

====================  =========================================================================
column                law
====================  =========================================================================
``readiness``         Uniform(-1, 1), readiness to manage at home, from lowest (-1) to highest
``lives_alone``       Bernoulli(0.5)
``navigation``        Bernoulli, :math:`\operatorname{logit} g_1 = 0.8\,\text{readiness}
                      - 0.3\,\text{lives\_alone}`
``tracked_14``        1 while the patient is still followed after the first decision,
                      :math:`\operatorname{logit} = 2.5 + 0.3\,\text{readiness}`
``self_care_14``      Normal(:math:`0.5\,\text{readiness} + 0.2\,\text{lives\_alone}
                      - 0.8\,\text{navigation}`, 1)
``telemonitoring``    Bernoulli, :math:`\operatorname{logit} g_2 = -0.2
                      + 0.3\,\text{self\_care\_14} + 0.4\,\text{navigation}`
``tracked_30``        1 while still followed after the second decision,
                      :math:`\operatorname{logit} = 2.5 - 0.2\,\text{self\_care\_14}`
``home_30``           Bernoulli, :math:`\operatorname{logit} \bar Q = -0.2 + 0.6\,\text{readiness}
                      - 0.2\,\text{lives\_alone} + 0.3\,\text{self\_care\_14}
                      + a_2 (-0.1 + 1.2\,\text{self\_care\_14} - 0.6\,a_1)
                      + a_1 (0.3 + 1.2\,\text{readiness})`
====================  =========================================================================

A patient who leaves follow-up after the first decision has no ``self_care_14``,
``telemonitoring``, ``tracked_30`` or ``home_30``; one who leaves after the second has no
``home_30``.

This is the law of the registered study ``learned-regimen-ltmle`` under program names:
:func:`make_two_decision_rules` draws the same random stream in the same order as that study's
generator (``tests/studies/_learned_regimen_law.py``), so a draw equals the study's draw with its
columns renamed, and ``tests/unit/test_datasets_two_decisions.py`` checks it.

**The best rules.**  Telemonitoring improves the outcome logit by
:math:`-0.1 + 1.2\,\text{self\_care\_14} - 0.6\,a_1`, so the best second rule starts it where
``self_care_14`` exceeds :math:`(0.1 + 0.6\,a_1) / 1.2`.  The best first rule offers navigation
where the expected outcome, with telemonitoring then started by the best second rule, is higher
with navigation: a cutoff in ``readiness`` for each value of ``lives_alone``.  Both boundaries are
crossed with a nonzero slope, so the best rules are unique.

**Truth.**  Under a plan the treatment and dropout mechanisms drop out, so the value of a pair of
rules is an integral of :math:`\bar Q` over ``readiness``, ``lives_alone`` and ``self_care_14``.
The trapezoid rule on 1,001 points of ``readiness`` handles the first axis, and for
``self_care_14`` given the first decision a Gauss-Legendre rule on the probability scale, split
at the rule's cutoff so that no node straddles the rule's jump.  The values are accurate to about
``1e-5``: the study's design test compares the grid with 16,001 points, and 200 nodes in place of
24 move the best pair's value by under ``1e-5``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .._typing import Backend, FloatArray
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict
from .navigation import longitudinal_navigation_protocol

__all__ = [
    "TWO_DECISION_COLUMNS",
    "learned_regimen_truth",
    "make_two_decision_rules",
    "two_decision_rules_protocol",
]

#: The ``LTMLE.fit`` roles of the columns of :func:`make_two_decision_rules`.
TWO_DECISION_COLUMNS: dict[str, Any] = {
    "outcome": "home_30",
    "treatment": ["navigation", "telemonitoring"],
    "baseline": ["readiness", "lives_alone"],
    "time_varying": [[], ["self_care_14"]],
    "censoring": ["tracked_14", "tracked_30"],
}

# The coefficients, on the logit scale except the mean of ``self_care_14``.
_G1 = (0.8, -0.3)  # navigation: readiness, lives_alone
_C1 = (2.5, 0.3)  # tracked after the first decision: intercept, readiness
_L2 = (0.5, 0.2, -0.8)  # mean of self_care_14: readiness, lives_alone, navigation
_G2 = (-0.2, 0.3, 0.4)  # telemonitoring: intercept, self_care_14, navigation
_C2 = (2.5, -0.2)  # tracked after the second decision: intercept, self_care_14
_Y_MAIN = (-0.2, 0.6, -0.2, 0.3)  # home_30: intercept, readiness, lives_alone, self_care_14
_Y_SECOND = (-0.1, 1.2, -0.6)  # telemonitoring's effect: intercept, self_care_14, navigation
_Y_FIRST = (0.3, 1.2)  # navigation's effect: intercept, readiness

_ALONE = (0.0, 1.0)
_GRID = np.linspace(-1.0, 1.0, 1001)
# The trapezoid mass of each grid point, times the density 1/2 of readiness and the
# probability 1/2 of each value of lives_alone.
_MASS = np.full(_GRID.size, _GRID[1] - _GRID[0])
_MASS[[0, -1]] /= 2.0
_MASS *= 0.25
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)


def _g1(readiness: Any, alone: Any) -> FloatArray:
    return np.asarray(expit(_G1[0] * np.asarray(readiness) + _G1[1] * np.asarray(alone)))


def _c1(readiness: Any) -> FloatArray:
    return np.asarray(expit(_C1[0] + _C1[1] * np.asarray(readiness)))


def _l2_mean(navigation: Any, readiness: Any, alone: Any) -> FloatArray:
    return np.asarray(
        _L2[0] * np.asarray(readiness)
        + _L2[1] * np.asarray(alone)
        + _L2[2] * np.asarray(navigation),
        dtype=float,
    )


def _g2(self_care: Any, navigation: Any) -> FloatArray:
    return np.asarray(
        expit(_G2[0] + _G2[1] * np.asarray(self_care) + _G2[2] * np.asarray(navigation))
    )


def _c2(self_care: Any) -> FloatArray:
    return np.asarray(expit(_C2[0] + _C2[1] * np.asarray(self_care)))


def _home(navigation: Any, telemonitoring: Any, self_care: Any, readiness: Any, alone: Any) -> Any:
    """``P(home_30 = 1 | navigation, telemonitoring, self_care_14, readiness, lives_alone)``."""
    a1 = np.asarray(navigation, dtype=float)
    l2 = np.asarray(self_care, dtype=float)
    w1 = np.asarray(readiness, dtype=float)
    second = _Y_SECOND[0] + _Y_SECOND[1] * l2 + _Y_SECOND[2] * a1
    logit = (
        _Y_MAIN[0]
        + _Y_MAIN[1] * w1
        + _Y_MAIN[2] * np.asarray(alone)
        + _Y_MAIN[3] * l2
        + np.asarray(telemonitoring) * second
        + a1 * (_Y_FIRST[0] + _Y_FIRST[1] * w1)
    )
    return np.asarray(expit(logit), dtype=float)


@dataclass(frozen=True)
class _Pair:
    """A pair of rules on the grid of ``readiness``.

    ``first[alone]`` is the first decision at each grid point.  The second rule treats where
    ``offset[alone][a1] + slope[alone][a1] * self_care_14 > 0``, with ``offset`` at each grid point.
    """

    first: dict[float, FloatArray]
    slope: dict[float, dict[float, float]]
    offset: dict[float, dict[float, FloatArray]]


def _pieces(cut: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Legendre nodes and weights on ``[0, cut]`` and ``[cut, 1]`` of the probability scale."""
    cut = np.asarray(cut, dtype=float)[:, None]
    half = (_NODES[None, :] + 1.0) / 2.0
    low = cut * half
    high = cut + (1.0 - cut) * half
    probability = np.concatenate([low, high], axis=1)
    weight = np.concatenate(
        [cut * _WEIGHTS[None, :] / 2.0, (1.0 - cut) * _WEIGHTS[None, :] / 2.0], axis=1
    )
    above = np.concatenate([np.zeros_like(low, dtype=bool), np.ones_like(high, dtype=bool)], axis=1)
    return probability, weight, above


def _conditional_value(pair: _Pair, alone: float, grid: FloatArray = _GRID) -> FloatArray:
    """``E[Qbar(d1, d2, L2, W) | A1 = d1, W]`` at each point of ``grid``, a grid of
    ``readiness`` whose size matches the arrays of ``pair``."""
    a1 = np.asarray(pair.first[alone], dtype=float)
    slope = np.where(a1 > 0.5, pair.slope[alone][1.0], pair.slope[alone][0.0])
    offset = np.where(a1 > 0.5, pair.offset[alone][1.0], pair.offset[alone][0.0])
    mean = _l2_mean(a1, grid, alone)
    with np.errstate(divide="ignore", invalid="ignore"):
        cutoff = np.where(slope != 0.0, -offset / slope, 0.0)
    cut = np.where(slope != 0.0, norm.cdf(cutoff - mean), 0.5)
    probability, weight, above = _pieces(cut)
    # A cutoff far in a tail leaves one piece of zero width; its nodes carry no weight, and the
    # clip keeps their self_care_14 finite.
    l2 = mean[:, None] + norm.ppf(np.clip(probability, 1e-15, 1.0 - 1e-15))
    treat_above = np.where(slope != 0.0, slope > 0.0, offset > 0.0)[:, None]
    treat_below = np.where(slope != 0.0, slope < 0.0, offset > 0.0)[:, None]
    a2 = np.where(above, treat_above, treat_below).astype(float)
    q = _home(a1[:, None], a2, l2, grid[:, None], alone)
    return np.asarray(np.sum(weight * q, axis=1), dtype=float)


def _value(pair: _Pair) -> float:
    """The value of a pair of rules: the share who would stay home had everyone followed it."""
    return float(sum(np.sum(_MASS * _conditional_value(pair, alone)) for alone in _ALONE))


def _static(a1: float, a2: float) -> _Pair:
    return _Pair(
        first=dict.fromkeys(_ALONE, np.full(_GRID.size, a1)),
        slope={alone: {0.0: 0.0, 1.0: 0.0} for alone in _ALONE},
        offset={
            alone: {0.0: np.full(_GRID.size, a2 - 0.5), 1.0: np.full(_GRID.size, a2 - 0.5)}
            for alone in _ALONE
        },
    )


@cache
def _best_pair() -> _Pair:
    """The best pair: the law's second-decision effect, then the better first decision."""
    first: dict[float, FloatArray] = {}
    slope: dict[float, dict[float, float]] = {}
    offset: dict[float, dict[float, FloatArray]] = {}
    for alone in _ALONE:
        slope[alone] = dict.fromkeys((0.0, 1.0), _Y_SECOND[1])
        offset[alone] = {
            a1: np.full(_GRID.size, _Y_SECOND[0] + _Y_SECOND[2] * a1) for a1 in (0.0, 1.0)
        }
        arms = {
            a1: _conditional_value(
                _Pair(
                    first={alone: np.full(_GRID.size, a1)},
                    slope={alone: slope[alone]},
                    offset={alone: offset[alone]},
                ),
                alone,
            )
            for a1 in (0.0, 1.0)
        }
        first[alone] = (arms[1.0] > arms[0.0]).astype(float)
    return _Pair(first=first, slope=slope, offset=offset)


def _navigation_gap(readiness: float, alone: float, telemonitoring: str) -> float:
    """How much navigation raises the chance of staying home at one ``readiness``, when
    telemonitoring then follows the best second rule (``"best"``) or is given to nobody
    (``"none"``)."""
    grid = np.array([readiness])
    values = []
    for a1 in (0.0, 1.0):
        if telemonitoring == "best":
            slope, offset = _Y_SECOND[1], _Y_SECOND[0] + _Y_SECOND[2] * a1
        else:
            slope, offset = 0.0, -0.5
        pair = _Pair(
            first={alone: np.array([a1])},
            slope={alone: dict.fromkeys((0.0, 1.0), slope)},
            offset={alone: dict.fromkeys((0.0, 1.0), np.array([offset]))},
        )
        values.append(float(_conditional_value(pair, alone, grid)[0]))
    return values[1] - values[0]


def _navigation_cutoff(alone: float, telemonitoring: str) -> float:
    """The ``readiness`` above which navigation helps, by root-finding on ``[-1, 1]``."""
    return float(brentq(_navigation_gap, -1.0, 1.0, args=(alone, telemonitoring), xtol=1e-12))


@cache
def _truth() -> tuple[tuple[str, float], ...]:
    best = _best_pair()
    truth = {
        "ey_best_rules": _value(best),
        "ey_neither": _value(_static(0.0, 0.0)),
        "ey_navigation_only": _value(_static(1.0, 0.0)),
        "ey_telemonitoring_only": _value(_static(0.0, 1.0)),
        "ey_both": _value(_static(1.0, 1.0)),
    }
    for a1, name in ((0.0, "without_navigation"), (1.0, "after_navigation")):
        truth[f"telemonitoring_cutoff_{name}"] = (-_Y_SECOND[0] - _Y_SECOND[2] * a1) / _Y_SECOND[1]
    for alone, name in ((0.0, "with_others"), (1.0, "alone")):
        truth[f"navigation_cutoff_{name}"] = _navigation_cutoff(alone, "best")
    for alone, name in ((0.0, "with_others"), (1.0, "alone")):
        truth[f"navigation_cutoff_without_telemonitoring_{name}"] = _navigation_cutoff(
            alone, "none"
        )
    return tuple(truth.items())


def make_two_decision_rules(
    n: int = 2_000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw patients through two care decisions after discharge, with the exact best plan.

    Parameters
    ----------
    n : int, default=2000
        The number of patients.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per patient, with the columns ``readiness``, ``lives_alone``, ``navigation``,
        ``tracked_14``, ``self_care_14``, ``telemonitoring``, ``tracked_30`` and ``home_30`` in
        time order.  :data:`TWO_DECISION_COLUMNS` gives their ``LTMLE.fit`` roles.
    truth : dict of str to float
        The share who would stay home through day 30 under the best pair of rules
        (``ey_best_rules``) and under each fixed plan (``ey_neither``, ``ey_navigation_only``,
        ``ey_telemonitoring_only``, ``ey_both``), each by quadrature; the ``self_care_14`` above
        which the best second rule starts telemonitoring, without and after navigation
        (``telemonitoring_cutoff_without_navigation``,
        ``telemonitoring_cutoff_after_navigation``); and the ``readiness`` above which the best
        first rule offers navigation, for patients living with others and alone
        (``navigation_cutoff_with_others``, ``navigation_cutoff_alone``), and the cutoff above
        which navigation helps when nobody gets telemonitoring
        (``navigation_cutoff_without_telemonitoring_with_others``,
        ``navigation_cutoff_without_telemonitoring_alone``), each by
        root-finding.

    Raises
    ------
    ValueError
        If ``n`` is not positive.

    See Also
    --------
    cleverly.datasets.learned_regimen_truth : The true value of the rules a learned fit used.
    cleverly.datasets.two_decision_rules_protocol : The study description of these data.

    Examples
    --------
    >>> from cleverly.datasets import make_two_decision_rules
    >>> frame, truth = make_two_decision_rules(n=5, seed=0)
    >>> list(frame.columns)
    ['readiness', 'lives_alone', 'navigation', 'tracked_14', 'self_care_14', 'telemonitoring', 'tracked_30', 'home_30']
    >>> round(truth["ey_best_rules"], 4)
    0.5579
    """  # noqa: E501
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    readiness = rng.uniform(-1.0, 1.0, n)
    alone = rng.binomial(1, 0.5, n).astype(float)
    navigation = rng.binomial(1, _g1(readiness, alone)).astype(float)
    tracked_14 = rng.binomial(1, _c1(readiness)).astype(float)
    self_care = _l2_mean(navigation, readiness, alone) + rng.standard_normal(n)
    telemonitoring = rng.binomial(1, _g2(self_care, navigation)).astype(float)
    tracked_30 = rng.binomial(1, _c2(self_care)).astype(float)
    home = rng.binomial(1, _home(navigation, telemonitoring, self_care, readiness, alone))
    home = home.astype(float)
    gone = tracked_14 == 0
    self_care[gone] = np.nan
    telemonitoring[gone] = np.nan
    tracked_30[gone] = np.nan
    home[gone | (tracked_30 == 0)] = np.nan
    payload = {
        "readiness": readiness,
        "lives_alone": alone,
        "navigation": navigation,
        "tracked_14": tracked_14,
        "self_care_14": self_care,
        "telemonitoring": telemonitoring,
        "tracked_30": tracked_30,
        "home_30": home,
    }
    return frame_from_dict(payload, backend=backend), dict(_truth())


def two_decision_rules_protocol() -> StudyProtocol:
    """Return the study description of the two care decisions after discharge.

    Returns
    -------
    StudyProtocol
        :func:`~cleverly.datasets.longitudinal_navigation_protocol` with telemonitoring at day
        14 as the second decision, both decisions made by rules learned from the records, and
        staying home through day 30 as the outcome.

    See Also
    --------
    cleverly.datasets.make_two_decision_rules : The simulated program this description fits.

    Examples
    --------
    >>> from cleverly.datasets import two_decision_rules_protocol
    >>> two_decision_rules_protocol().outcome
    'Stayed home through day 30 after discharge, with no unplanned readmission'
    """
    program = longitudinal_navigation_protocol()
    return replace(
        program,
        treatment_strategies=(
            "Offer navigation at discharge, and start home telemonitoring at day 14, each by a "
            "rule learned from the records available at that decision",
        ),
        treatment_versions=(
            "The declared discharge navigation contacts for the patients the first rule selects, "
            "and the declared telemonitoring kit and nurse review for those the second rule "
            "selects; usual support for the others",
        ),
        outcome="Stayed home through day 30 after discharge, with no unplanned readmission",
        intercurrent_event_handling=(
            "Count an unplanned readmission before day 30 as not staying home",
            "Treat loss to follow-up as censoring, so the target is the share who would stay "
            "home had every patient stayed in follow-up (hypothetical strategy)",
            "Count death before day 30 as not staying home (composite strategy)",
            "Analyze each assigned offer regardless of how much of it the patient used",
        ),
        assumption_rationale=(
            program.assumption_rationale[0],
            "Version records support that the navigation offer and the telemonitoring start are "
            "each one well-defined action, so a patient's recorded outcome is the outcome they "
            "would have had under the offers they got",
            program.assumption_rationale[2],
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def learned_regimen_truth(result: Any, frame: Any) -> float:
    """Return the true value of the rule pairs that a learned-regimen fit of these data used.

    ``LTMLE(regimens=LearnedRegimen())`` learns a pair of rules in each training fold and reports
    the value of those pairs, each weighted by its fold's share of the rows.  This function
    refits each fold's backward Q-learning on the fold's training rows with clones of the fit's
    own learners (``result.estimator``), checks that the refitted pair equals the fit's stored
    pair (``result.learned_regimen.fold_rules``) on every row, and returns the same weighted
    average of the values of the refitted pairs.

    The refit follows the fit's conventions: each prediction is clipped to ``[0, 1]``, a tie
    goes to the lower arm at both decisions, and a patient who left follow-up after the first
    decision carries the second-decision code 0.  Each value is integrated as
    :func:`make_two_decision_rules` integrates the best pair, which is accurate to about
    ``1e-5``.

    Parameters
    ----------
    result : LongitudinalResult
        An unweighted ``LTMLE(regimens=LearnedRegimen(), repeats=1)`` fit of a
        :func:`make_two_decision_rules` draw, with the roles of :data:`TWO_DECISION_COLUMNS`.
        Its ``outcome_learner``, which fits the second decision, must be a main-terms
        ``LogisticRegression``, because the integral needs a second rule that is a cutoff in
        ``self_care_14``.  Its ``pseudo_learner``, which fits the first decision, must be given
        and deterministic, so that a refit reproduces the fitted model.
    frame : pandas.DataFrame
        The frame the fit was given.

    Returns
    -------
    float
        :math:`\\sum_v (n_v / n) \\Psi_{d_v}`, where :math:`d_v` is fold :math:`v`'s pair.

    Raises
    ------
    ValueError
        In this order: if the fit holds no learned regimen; if it used ``repeats`` above 1 or
        observation weights; if its roles are not those of :data:`TWO_DECISION_COLUMNS`; if
        ``frame`` is not a pandas frame or any of its model columns differs from the fit's
        data; if its
        ``outcome_learner`` is not a ``LogisticRegression`` or its ``pseudo_learner`` is
        ``None``; or if a refitted pair differs from the stored one on a row.

    See Also
    --------
    cleverly.datasets.make_two_decision_rules : The data, and the value of the best rules.
    cleverly.longitudinal.LearnedRegimen : The regimen whose value the fit estimates.
    """
    import pandas as pd
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression

    record: Any = getattr(result, "learned_regimen", None)
    _require(record is not None, "the fit must be an LTMLE(regimens=LearnedRegimen()) fit")
    estimator = result.estimator
    _require(int(estimator.repeats) == 1, "the fit must use repeats=1")
    data = result.data
    _require(not data.is_weighted, "the fit must be unweighted")
    roles = TWO_DECISION_COLUMNS
    _require(
        data.outcome_name == roles["outcome"]
        and tuple(data.treatment_names) == tuple(roles["treatment"])
        and tuple(data.baseline_names) == tuple(roles["baseline"])
        and tuple(map(tuple, data.time_varying_names)) == tuple(map(tuple, roles["time_varying"]))
        and tuple(data.censoring_names) == tuple(roles["censoring"]),
        "the fit must declare the roles of TWO_DECISION_COLUMNS, with the same columns in the "
        "same order",
    )
    _require(
        isinstance(frame, pd.DataFrame),
        "frame must be a pandas DataFrame; convert a polars or pyarrow draw with to_pandas()",
    )
    _require(len(frame) == data.n, "frame must be the frame the fit was given")
    stored_columns = {
        "baseline": (["readiness", "lives_alone"], np.asarray(data.baseline, dtype=float)),
        "treatment": (["navigation", "telemonitoring"], np.asarray(data.treatment, dtype=float)),
        "time_varying": (["self_care_14"], np.asarray(data.time_varying[1], dtype=float)),
        "outcome": (["home_30"], np.asarray(data.outcome, dtype=float)[:, None]),
    }
    for names, fitted in stored_columns.values():
        _require(
            np.array_equal(frame[names].to_numpy(dtype=float), fitted, equal_nan=True),
            f"frame must be the frame the fit was given: its {', '.join(names)} differ",
        )
    tracked = np.nan_to_num(frame[list(roles["censoring"])].to_numpy(dtype=float))
    _require(
        np.array_equal(tracked, np.asarray(data.uncensored, dtype=float)),
        "frame must be the frame the fit was given: its tracked_14, tracked_30 differ",
    )
    x1 = frame[["readiness", "lives_alone"]].to_numpy(dtype=float)
    outcome_learner, pseudo_learner = estimator.outcome_learner, estimator.pseudo_learner
    _require(
        isinstance(outcome_learner, LogisticRegression),
        "the fit's outcome_learner must be a LogisticRegression: the integral needs a second "
        "rule that is a cutoff in self_care_14",
    )
    _require(pseudo_learner is not None, "the fit must declare its pseudo_learner")
    n = data.n
    a1 = frame["navigation"].to_numpy(dtype=float)
    a2 = np.nan_to_num(frame["telemonitoring"].to_numpy(dtype=float))
    followed = (frame["tracked_14"] == 1).to_numpy()
    complete = followed & (frame["tracked_30"] == 1).to_numpy()
    x2 = np.nan_to_num(frame[["readiness", "lives_alone", "self_care_14"]].to_numpy(dtype=float))
    y = np.nan_to_num(frame["home_30"].to_numpy(dtype=float))
    stored = np.asarray(record.fold_rules).astype(np.int64)
    values, sizes = [], []
    for fold, (train, test) in enumerate(result.folds):
        rows = np.zeros(n, dtype=bool)
        rows[np.asarray(train)] = True
        second: dict[tuple[float, float], Any] = {}
        first: dict[float, Any] = {}
        blips = np.full((n, 2), np.nan)
        branch = np.zeros((n, 2))
        for arm in (0.0, 1.0):
            predicted = []
            for later in (0.0, 1.0):
                fit_rows = rows & complete & (a1 == arm) & (a2 == later)
                model = clone(outcome_learner).fit(x2[fit_rows], y[fit_rows])
                second[arm, later] = model
                predicted.append(np.clip(model.predict_proba(x2)[:, 1], 0.0, 1.0))
            blip = np.where(followed, predicted[1] - predicted[0], np.nan)
            blips[:, int(arm)] = blip
            target = np.where(np.nan_to_num(blip) > 0.0, predicted[1], predicted[0])
            fit_rows = rows & followed & (a1 == arm)
            model = clone(pseudo_learner).fit(x1[fit_rows], target[fit_rows])
            first[arm] = model
            branch[:, int(arm)] = np.clip(model.predict(x1), 0.0, 1.0)
        first_rule = (branch[:, 1] - branch[:, 0] > 0.0).astype(int)
        chosen = blips[np.arange(n), first_rule]
        rules = np.column_stack([first_rule, (np.nan_to_num(chosen) > 0.0).astype(int)])
        differ = int(np.sum(np.any(rules != stored[fold], axis=1)))
        _require(
            differ == 0,
            f"fold {fold}: the refitted pair of rules differs from the fit's on {differ} rows",
        )
        values.append(_value(_grid_pair(first, second)))
        sizes.append(len(test))
    return float(np.dot(np.asarray(sizes, dtype=float) / n, values))


def _grid_pair(first: dict[float, Any], second: dict[tuple[float, float], Any]) -> _Pair:
    """A fold's pair on the grid.  The second rule is the sign of the difference of the two
    stratum logits, which is affine in ``self_care_14`` for main-terms logistic models."""
    rule: dict[float, FloatArray] = {}
    slope: dict[float, dict[float, float]] = {}
    offset: dict[float, dict[float, FloatArray]] = {}
    for alone in _ALONE:
        x1 = np.column_stack([_GRID, np.full(_GRID.size, alone)])
        arms = {arm: np.clip(first[arm].predict(x1), 0.0, 1.0) for arm in (0.0, 1.0)}
        rule[alone] = (arms[1.0] - arms[0.0] > 0.0).astype(float)
        slope[alone], offset[alone] = {}, {}
        for arm in (0.0, 1.0):
            treated, control = second[arm, 1.0], second[arm, 0.0]
            coefficient = np.ravel(treated.coef_) - np.ravel(control.coef_)
            intercept = float(np.ravel(treated.intercept_)[0] - np.ravel(control.intercept_)[0])
            slope[alone][arm] = float(coefficient[2])
            offset[alone][arm] = intercept + coefficient[0] * _GRID + coefficient[1] * alone
    return _Pair(first=rule, slope=slope, offset=offset)
