r"""The population-impact law of the care-transition program, with an exact truth.

Discharge teams offer transition navigation to some patients going home.  Within 7 days a patient
may see a primary-care doctor, and the offer makes that visit more likely.  On day 30 a survey
asks whether the transition was top box, and some patients never return it.  The law answers three
kinds of question about one population: what current practice achieves, what offering everyone or
no one would achieve, and what the offer does when the 7-day visit is held at one level.

=========================  ==============================================================
node                       law
=========================  ==============================================================
``discharge_risk``         standard normal
``age``                    standard normal, independent of the others
``prior_utilization``      standard normal, independent of the others
``transition_navigation``  Bernoulli, ``expit(-0.3 + 0.7 risk - 0.3 age + 0.2 use)``
``pcp_visit_7d``           Bernoulli, ``expit(-0.5 + 1.2 A - 0.4 risk + 0.3 age)``
``top_box``                Bernoulli, ``expit(-0.4 + 0.5 A + 0.8 Z + 0.4 A risk - 0.6 risk``
                           ``- 0.2 age - 0.2 use)``
``responded``              Bernoulli, ``expit(1.0 + 0.4 A - 0.6 risk + 0.2 age)``; ``top_box`` is
                           missing where it is 0
=========================  ==============================================================

Every patient, offered or not, can have the visit or not, so the controlled direct effect at
either visit level has positivity.  Survey response reads the offer and the baseline covariates
only, so the outcomes are missing at random given ``(A, W)``, and also given ``(A, Z, W)``.  The
offer's effect on the log-odds scale grows with discharge risk, and current practice offers
higher-risk patients more often, so the patients offered today gain more on average than those who
are not.

The truth is computed by **quadrature and enumeration, not simulation**: a sum over the two visit
levels inside a product Gauss--Hermite rule over the three covariates.  Each integrand is a smooth
function of the covariates, and 40 nodes per axis agree with 60 to below ``1e-12``.
``tests/unit/test_datasets_population_impact.py`` compares each truth with a large Monte Carlo
draw and rejects deliberately wrong truths.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["make_population_impact", "population_impact_truth"]

#: The baseline covariates, in the order of the latent columns.
COVARIATES = ("discharge_risk", "age", "prior_utilization")

#: Gauss--Hermite nodes per covariate axis for the truth.
_NODES = 40


def propensity(w: FloatArray) -> FloatArray:
    """``P(A = 1 | W)``, the chance of the offer under current practice."""
    return np.asarray(expit(-0.3 + 0.7 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2]))


def visit_probability(w: FloatArray, a: float) -> FloatArray:
    """``P(Z = 1 | A = a, W)``, the chance of a primary-care visit within 7 days."""
    return np.asarray(expit(-0.5 + 1.2 * a - 0.4 * w[:, 0] + 0.3 * w[:, 1]))


def outcome_probability(w: FloatArray, a: float, z: float) -> FloatArray:
    """``P(Y = 1 | A = a, Z = z, W)``, the chance of a top-box transition."""
    risk, age, use = w[:, 0], w[:, 1], w[:, 2]
    logit = -0.4 + 0.5 * a + 0.8 * z + 0.4 * a * risk - 0.6 * risk - 0.2 * age - 0.2 * use
    return np.asarray(expit(logit))


def response_probability(w: FloatArray, a: float) -> FloatArray:
    """``P(Delta = 1 | A = a, W)``, the chance the day-30 survey comes back."""
    return np.asarray(expit(1.0 + 0.4 * a - 0.6 * w[:, 0] + 0.2 * w[:, 1]))


def arm_mean(w: FloatArray, a: float) -> FloatArray:
    """``E(Y | A = a, W)``, the outcome mean with the visit left to follow the offer."""
    visit = visit_probability(w, a)
    return visit * outcome_probability(w, a, 1.0) + (1.0 - visit) * outcome_probability(w, a, 0.0)


def _grid(nodes: int) -> tuple[FloatArray, FloatArray]:
    """The product Gauss--Hermite points and weights for three standard normal covariates."""
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    mesh = np.meshgrid(points, points, points, indexing="ij")
    mass = np.einsum("i,j,k->ijk", weights, weights, weights)
    return np.column_stack([axis.reshape(-1) for axis in mesh]), mass.reshape(-1)


def _truth(nodes: int) -> dict[str, float]:
    """Every population truth of the law, on a product rule with ``nodes`` points per axis."""
    w, mass = _grid(nodes)
    g = propensity(w)
    m1, m0 = arm_mean(w, 1.0), arm_mean(w, 0.0)
    ey1, ey0 = float(mass @ m1), float(mass @ m0)
    ey_obs = float(mass @ (g * m1 + (1.0 - g) * m0))
    effect = m1 - m0
    truth = {
        "ey_obs": ey_obs,
        "ey1": ey1,
        "ey0": ey0,
        "ate": ey1 - ey0,
        "att": float(mass @ (g * effect) / (mass @ g)),
        "atc": float(mass @ ((1.0 - g) * effect) / (mass @ (1.0 - g))),
        "rr": ey1 / ey0,
        "or": (ey1 / (1.0 - ey1)) / (ey0 / (1.0 - ey0)),
        "offered_share": float(mass @ g),
        "response_rate": float(
            mass @ (g * response_probability(w, 1.0) + (1.0 - g) * response_probability(w, 0.0))
        ),
    }
    for reference, mean in ((0, ey0), (1, ey1)):
        truth[f"par[reference={reference}]"] = ey_obs - mean
        truth[f"paf[reference={reference}]"] = 1.0 - mean / ey_obs
    for z in (0, 1):
        held1 = float(mass @ outcome_probability(w, 1.0, float(z)))
        held0 = float(mass @ outcome_probability(w, 0.0, float(z)))
        truth[f"cde_ey1[visit={z}]"] = held1
        truth[f"cde_ey0[visit={z}]"] = held0
        truth[f"cde_ate[visit={z}]"] = held1 - held0
        truth[f"cde_rr[visit={z}]"] = held1 / held0
    return truth


@cache
def _cached_truth() -> tuple[tuple[str, float], ...]:
    return tuple(_truth(_NODES).items())


def population_impact_truth() -> dict[str, float]:
    """Return the exact population truths of the population-impact law.

    Returns
    -------
    dict of str to float
        ``ey_obs`` is the top-box rate under current practice, ``ey1`` and ``ey0`` the rates if
        every patient or no patient got the offer, and ``ate``, ``att``, ``atc``, ``rr`` and
        ``or`` the contrasts of those two.  ``par[reference=a]`` is ``ey_obs`` minus the rate
        with every patient at level ``a``, and ``paf[reference=a]`` is one minus their ratio.
        ``cde_ate[visit=z]`` and ``cde_rr[visit=z]`` compare the offer with no offer while every
        patient's 7-day visit is held at ``z``, and ``cde_ey1[visit=z]`` and ``cde_ey0[visit=z]``
        are their two rates.  ``offered_share`` and ``response_rate`` are the population shares
        offered and responding.

    See Also
    --------
    cleverly.datasets.make_population_impact : The draw these values describe.

    Examples
    --------
    >>> from cleverly.datasets import population_impact_truth
    >>> truth = population_impact_truth()
    >>> round(truth["ey_obs"], 4), round(truth["ey1"], 4), round(truth["ey0"], 4)
    (0.5641, 0.644, 0.4791)
    >>> round(truth["par[reference=1]"], 4)
    -0.0799
    """
    return dict(_cached_truth())


def make_population_impact(
    n: int = 4000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw the population-impact program data and return the exact truths.

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
        One row per discharge: ``top_box`` (1 for a top-box transition, missing where the survey
        did not come back), ``transition_navigation`` (1 for the offer), the three standardized
        baseline covariates, ``pcp_visit_7d`` (1 for a primary-care visit within 7 days) and
        ``responded`` (1 where the survey came back).
    truth : dict of str to float
        The population truths of :func:`population_impact_truth`.

    See Also
    --------
    cleverly.datasets.population_impact_truth : The truths, without a draw.
    cleverly.datasets.navigation_protocol : The program protocol a tutorial derives its own from.

    Examples
    --------
    >>> from cleverly.datasets import make_population_impact
    >>> frame, truth = make_population_impact(n=500, seed=3)
    >>> print(*frame.columns, sep="\\n")
    top_box
    transition_navigation
    discharge_risk
    age
    prior_utilization
    pcp_visit_7d
    responded
    >>> bool(frame["top_box"].isna().equals(frame["responded"] == 0))
    True
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    w = rng.normal(size=(n, len(COVARIATES)))
    a = rng.binomial(1, propensity(w)).astype(float)
    z = rng.binomial(1, np.where(a == 1.0, visit_probability(w, 1.0), visit_probability(w, 0.0)))
    z = z.astype(float)
    chance = np.where(
        a == 1.0,
        np.where(z == 1.0, outcome_probability(w, 1.0, 1.0), outcome_probability(w, 1.0, 0.0)),
        np.where(z == 1.0, outcome_probability(w, 0.0, 1.0), outcome_probability(w, 0.0, 0.0)),
    )
    y = rng.binomial(1, chance).astype(float)
    observed = rng.binomial(
        1, np.where(a == 1.0, response_probability(w, 1.0), response_probability(w, 0.0))
    ).astype(float)
    payload: dict[str, Any] = {
        "top_box": np.where(observed == 1.0, y, np.nan),
        "transition_navigation": a,
    }
    for index, name in enumerate(COVARIATES):
        payload[name] = w[:, index]
    payload["pcp_visit_7d"] = z
    payload["responded"] = observed
    return frame_from_dict(payload, backend=backend), population_impact_truth()
