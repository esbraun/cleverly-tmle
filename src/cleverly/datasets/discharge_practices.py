r"""Five discharge practices of the care-transition program, with exact truths.

Hospital teams chose, patient by patient, which of five discharge practices each patient going home
got.  The plan asks which practices change whether a patient stays home, with no readmission, for
30 days.  Navigators book follow-up visits, so the navigation offer changes the chance that a
follow-up visit is booked before discharge.  Two practices have no effect at all.

==========================  =============================================================
node                        law
==========================  =============================================================
``discharge_risk``          standard normal
``age``                     standard normal, independent of the others
``prior_utilization``       standard normal, independent of the others
``transition_navigation``   Bernoulli, ``expit(-0.3 + 0.7 risk - 0.3 age + 0.2 use)``
``followup_booked``         Bernoulli, ``expit(-0.8 + 1.5 navigation + 0.3 risk - 0.2 age)``
``medication_review``       Bernoulli, ``expit(-0.2 + 0.5 use + 0.3 age)``
``teach_back``              Bernoulli, ``expit(0.1 - 0.2 risk + 0.3 age)``
``home_health_referral``    Bernoulli, ``expit(-1.0 + 1.0 risk + 0.4 age)``
``stayed_home``             Bernoulli, ``expit(0.8 + 0.1 navigation + 0.6 followup``
                            ``+ 0.3 review - 0.8 risk - 0.3 age - 0.3 use)``
==========================  =============================================================

Over the cube of covariates between their 0.1% and 99.9% quantiles, the true chance of each
practice stays within 0.0178 to 0.968 for navigation, 0.0875 to 0.9042 for the booking, 0.0646 to
0.9065 for the review, 0.1908 to 0.8382 for teach-back and 0.0048 to 0.9653 for the home-health
referral, whose overlap is the thinnest.

``teach_back`` and ``home_health_referral`` do not enter the outcome.  Every practice but
``followup_booked`` depends on the baseline covariates only, so given the covariates and the
navigation offer the practices are independent of one another.

Two kinds of truth are returned for each practice ``c``, both risk differences of
``stayed_home``:

``adjusted[c]``
    :math:`E[Q(1, A_{-c}, W) - Q(0, A_{-c}, W)]` over the observed joint law of the covariates and
    the other practices, where :math:`Q` is the chance of staying home given every practice and
    the covariates.  This is what a fit with ``c`` as the treatment estimates when it adjusts for
    the covariates and the other four practices.  For ``transition_navigation`` it holds each
    patient's follow-up booking at its observed law, so it leaves out the part of the offer's
    effect that runs through the booked visit.
``total[c]``
    The difference between everyone and no one getting ``c``, with ``followup_booked`` following
    the navigation offer it then has and the other practices at their law given the covariates.
    It differs from ``adjusted[c]`` only for ``transition_navigation``.  For
    ``transition_navigation`` it is what a fit estimates that adjusts for the covariates alone.
    For each other practice it is what a fit estimates that adjusts for the covariates and the
    navigation offer; for ``followup_booked`` that offer is a confounder, so a fit adjusting for
    the covariates alone estimates another number.

The truth is computed by **quadrature and enumeration, not simulation**: a sum over the 32
combinations of the practices inside a product Gauss--Hermite rule over the three covariates.
``tests/unit/test_datasets_discharge_practices.py`` checks the rule's convergence, compares each
truth with a Monte Carlo draw of potential outcomes, and rejects deliberately wrong truths.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["PRACTICES", "discharge_practices_truth", "make_discharge_practices"]

#: The baseline covariates, in the order of the latent columns.
COVARIATES = ("discharge_risk", "age", "prior_utilization")

#: The five candidate practices, in the order they are drawn.
PRACTICES = (
    "transition_navigation",
    "followup_booked",
    "medication_review",
    "teach_back",
    "home_health_referral",
)

#: Gauss--Hermite nodes per covariate axis for the truth.
_NODES = 40

_Chance = Callable[[FloatArray, Mapping[str, float]], FloatArray]


def _navigation(w: FloatArray, _: Mapping[str, float]) -> FloatArray:
    return np.asarray(expit(-0.3 + 0.7 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2]))


def _followup(w: FloatArray, practices: Mapping[str, float]) -> FloatArray:
    navigation = practices["transition_navigation"]
    return np.asarray(expit(-0.8 + 1.5 * navigation + 0.3 * w[:, 0] - 0.2 * w[:, 1]))


def _review(w: FloatArray, _: Mapping[str, float]) -> FloatArray:
    return np.asarray(expit(-0.2 + 0.5 * w[:, 2] + 0.3 * w[:, 1]))


def _teach_back(w: FloatArray, _: Mapping[str, float]) -> FloatArray:
    return np.asarray(expit(0.1 - 0.2 * w[:, 0] + 0.3 * w[:, 1]))


def _home_health(w: FloatArray, _: Mapping[str, float]) -> FloatArray:
    return np.asarray(expit(-1.0 + 1.0 * w[:, 0] + 0.4 * w[:, 1]))


#: ``P(practice = 1 | earlier practices, W)`` for each practice, in drawing order.
CHANCES: Mapping[str, _Chance] = {
    "transition_navigation": _navigation,
    "followup_booked": _followup,
    "medication_review": _review,
    "teach_back": _teach_back,
    "home_health_referral": _home_health,
}


def outcome_probability(w: FloatArray, practices: Mapping[str, float]) -> FloatArray:
    """``P(stayed_home = 1 | practices, W)``."""
    logit = (
        0.8
        + 0.1 * practices["transition_navigation"]
        + 0.6 * practices["followup_booked"]
        + 0.3 * practices["medication_review"]
        - 0.8 * w[:, 0]
        - 0.3 * w[:, 1]
        - 0.3 * w[:, 2]
    )
    return np.asarray(expit(logit))


def _grid(nodes: int) -> tuple[FloatArray, FloatArray]:
    """The product Gauss--Hermite points and weights for three standard normal covariates."""
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    mesh = np.meshgrid(points, points, points, indexing="ij")
    mass = np.einsum("i,j,k->ijk", weights, weights, weights)
    return np.column_stack([axis.reshape(-1) for axis in mesh]), mass.reshape(-1)


def _joint(w: FloatArray, combination: Mapping[str, float], fixed: str | None) -> FloatArray:
    """The chance of ``combination`` given ``W``, with ``fixed`` set rather than drawn."""
    chance = np.ones(len(w))
    for name, model in CHANCES.items():
        if name == fixed:
            continue
        p = model(w, combination)
        chance = chance * (p if combination[name] == 1.0 else 1.0 - p)
    return chance


def _truth(nodes: int) -> dict[str, float]:
    """Every population truth of the law, on a product rule with ``nodes`` points per axis."""
    w, mass = _grid(nodes)
    combinations = [
        dict(zip(PRACTICES, map(float, levels), strict=True))
        for levels in itertools.product((0.0, 1.0), repeat=len(PRACTICES))
    ]
    observed = [_joint(w, combination, None) for combination in combinations]
    outcome = [outcome_probability(w, combination) for combination in combinations]
    truth = {
        "stayed_home_rate": float(mass @ sum(p * q for p, q in zip(observed, outcome, strict=True)))
    }
    for name in PRACTICES:
        truth[f"share[{name}]"] = float(
            mass @ sum(p for p, c in zip(observed, combinations, strict=True) if c[name] == 1.0)
        )
    index = {tuple(c.values()): k for k, c in enumerate(combinations)}
    for position, name in enumerate(PRACTICES):
        adjusted = np.zeros(len(w))
        means = {0.0: np.zeros(len(w)), 1.0: np.zeros(len(w))}
        for k, combination in enumerate(combinations):
            means[combination[name]] += _joint(w, combination, name) * outcome[k]
            if combination[name] == 1.0:
                levels = list(combination.values())
                levels[position] = 0.0
                flipped = index[tuple(levels)]
                adjusted += (observed[k] + observed[flipped]) * (outcome[k] - outcome[flipped])
        truth[f"adjusted[{name}]"] = float(mass @ adjusted)
        truth[f"total[{name}]"] = float(mass @ (means[1.0] - means[0.0]))
    return truth


@cache
def _cached_truth() -> tuple[tuple[str, float], ...]:
    return tuple(_truth(_NODES).items())


def discharge_practices_truth() -> dict[str, float]:
    """Return the exact population truths of the discharge-practices law.

    Returns
    -------
    dict of str to float
        For each practice ``c`` of :data:`PRACTICES`: ``adjusted[c]``, the effect on the chance
        of staying home of every patient against no patient getting ``c`` while the other
        practices stay as they were; ``total[c]``, the same difference while the follow-up booking
        follows the navigation offer; and ``share[c]``, the share of patients who get ``c`` today.
        ``stayed_home_rate`` is the share who stay home under current practice.  The two effects
        differ only for ``transition_navigation``, whose ``adjusted`` value leaves out the part
        of its effect that runs through the booked follow-up visit.

    See Also
    --------
    cleverly.datasets.make_discharge_practices : The draw these values describe.

    Examples
    --------
    >>> from cleverly.datasets import discharge_practices_truth
    >>> truth = discharge_practices_truth()
    >>> round(truth["adjusted[transition_navigation]"], 4)
    0.0165
    >>> round(truth["total[transition_navigation]"], 4)
    0.0506
    >>> truth["total[teach_back]"]
    0.0
    """
    return dict(_cached_truth())


def make_discharge_practices(
    n: int = 4000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw the discharge-practices program data and return the exact truths.

    Parameters
    ----------
    n : int
        Number of discharges.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per discharge: ``stayed_home`` (1 for no readmission by day 30), the five
        practices of :data:`PRACTICES` (1 where the patient got it), and the three standardized
        baseline covariates.
    truth : dict of str to float
        The population truths of :func:`discharge_practices_truth`.

    See Also
    --------
    cleverly.datasets.discharge_practices_truth : The truths, without a draw.
    cleverly.variable_importance : One fit for each candidate practice.

    Examples
    --------
    >>> from cleverly.datasets import make_discharge_practices
    >>> frame, truth = make_discharge_practices(n=500, seed=3)
    >>> print(*frame.columns, sep="\\n")
    stayed_home
    transition_navigation
    followup_booked
    medication_review
    teach_back
    home_health_referral
    discharge_risk
    age
    prior_utilization
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    w = rng.normal(size=(n, len(COVARIATES)))
    practices: dict[str, FloatArray] = {}
    for name, model in CHANCES.items():
        practices[name] = rng.binomial(1, model(w, practices)).astype(float)
    stayed = rng.binomial(1, outcome_probability(w, practices)).astype(float)
    payload: dict[str, Any] = {"stayed_home": stayed, **practices}
    for index, name in enumerate(COVARIATES):
        payload[name] = w[:, index]
    return frame_from_dict(payload, backend=backend), discharge_practices_truth()
