r"""A navigation offer whose effect differs by living situation, with exact group truths.

Discharge teams offer transition navigation to some patients with a discharge-home order.  The
outcome is whether the patient stays home, with no unplanned readmission, for 30 days.  Patients
live with family, alone, or with a paid carer, and the offer helps those who live alone most.
Within 7 days a patient may see a primary-care doctor, and the offer makes that visit more likely.

==================  ===========================================================================
column              law
==================  ===========================================================================
``age``             Uniform(50, 90)
``discharge_risk``  Uniform(0, 1)
``home_support``    text, ``"family"``, ``"alone"`` or ``"paid carer"`` with chances 0.5, 0.3
                    and 0.2, independent of age and risk
``navigation``      Bernoulli, :math:`\operatorname{logit} g = -1.2 + 2.5\,\text{risk}
                    + 0.6\,[\text{alone}] - 0.4\,[\text{carer}] + 0.03\,(\text{age} - 70)`
``pcp_visit_7d``    Bernoulli, :math:`0.35 + 0.30\,A - 0.15\,\text{risk}`
``home_30``         Bernoulli, :math:`0.50 - 0.08\,[\text{alone}] - 0.30\,\text{risk}
                    - 0.002\,(\text{age} - 70) + 0.10\,Z + A\,\{\tau(V) + 0.04\,Z\}`
==================  ===========================================================================

Here :math:`A` is ``navigation``, :math:`Z` is ``pcp_visit_7d``, and
:math:`\tau(V) = \tau_g + 0.003\,(\text{age} - 70)` with :math:`\tau_g = 0.02, 0.14, 0.06` for
family, alone and paid carer.  The chance of an offer stays in 0.10 to 0.92, the chance of a
visit in 0.20 to 0.65, and the chance of staying home in 0.08 to 0.72, over every corner of the
covariate box.

**Truth.**  The outcome law is linear in the chances, so every truth is closed form.  Risk is
uniform and independent of the group and age, so :math:`E\{Z(1)\} = 0.575` and
:math:`E\{Z(1) - Z(0)\} = 0.30`.

- The effect with the visit held at :math:`z` (the controlled direct effect) is
  :math:`\tau(V) + 0.04\,z` for each group and age.
- The total effect of the offer, with the visit left to follow it, is
  :math:`\tau(V) + 0.10 \times 0.30 + 0.04 \times 0.575 = \tau(V) + 0.053`.  The difference
  between the total effect and the direct effect is therefore the same in every group: 0.053
  with the visit held at no, and 0.013 with it held at yes.
- The group effects average over age (mean 70): 0.073, 0.193 and 0.113.  A projection on the
  group indicators is saturated in the group, so its coefficients are family's effect and the
  differences 0.120 and 0.040 from it.  The projection on the indicators plus a straight line in
  age contains the truth exactly: the line's slope is 0.003 per year.
- Without the offer, the chance of staying home is 0.3775 for family and paid carer and 0.2975
  for those living alone, so the risk ratios are 1.193, 1.649 and 1.299.

The ``0.04 A Z`` term is a nonzero witness for the direct effect: it makes the direct effect at
visit 1 differ from visit 0 by 0.04 in every group, and the ``0.10 Z`` term makes the total effect
differ from both.  ``tests/unit/test_datasets_home_support.py`` compares each truth with a Monte
Carlo draw and rejects deliberately wrong truths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["HOME_SUPPORT_LEVELS", "home_support_effect", "home_support_truth", "make_home_support"]

#: The living situations, in the order a fit takes as reference first.
HOME_SUPPORT_LEVELS = ("family", "alone", "paid carer")

#: The chance of each living situation.
_SHARES = (0.5, 0.3, 0.2)

#: The direct effect of the offer at visit 0 and age 70, by living situation.
_TAU = {"family": 0.02, "alone": 0.14, "paid carer": 0.06}

#: The change in the offer's effect per year of age.
AGE_SLOPE = 0.003

#: The extra effect of the offer when the 7-day visit happens (the ``A Z`` term).
_OFFER_BY_VISIT = 0.04

#: The effect of the 7-day visit itself (the ``Z`` term).
_VISIT = 0.10

#: Age range and its mean.
_AGE_LOW, _AGE_HIGH, _AGE_CENTER = 50.0, 90.0, 70.0

#: The name of a projection's intercept term, as a fit prints it.
_INTERCEPT = "(intercept)"


def _levels(level: Any) -> np.ndarray:
    values = np.asarray(level, dtype=object)
    known = np.isin(values, np.asarray(HOME_SUPPORT_LEVELS, dtype=object))
    if not np.all(known):
        unknown = sorted({str(value) for value in np.atleast_1d(values)[~np.atleast_1d(known)]})
        raise ValueError(f"home_support levels must be in {HOME_SUPPORT_LEVELS}; got {unknown}")
    return values


def offer_chance(level: Any, age: Any, risk: Any) -> FloatArray:
    """``P(navigation = 1 | W)``, the chance of the offer under current practice."""
    values = _levels(level)
    logit = (
        -1.2
        + 2.5 * np.asarray(risk, dtype=float)
        + 0.6 * (values == "alone")
        - 0.4 * (values == "paid carer")
        + 0.03 * (np.asarray(age, dtype=float) - _AGE_CENTER)
    )
    return np.asarray(expit(logit), dtype=float)


def visit_chance(offer: Any, risk: Any) -> FloatArray:
    """``P(pcp_visit_7d = 1 | A, W)``: it reads the offer and risk, not the group."""
    return np.asarray(
        0.35 + 0.30 * np.asarray(offer, dtype=float) - 0.15 * np.asarray(risk, dtype=float),
        dtype=float,
    )


def _direct(level: Any, age: Any) -> FloatArray:
    """``tau(V)``, the offer's effect with the visit held at 0."""
    values = _levels(level)
    base = sum((values == name) * effect for name, effect in _TAU.items())
    return np.asarray(base + AGE_SLOPE * (np.asarray(age, dtype=float) - _AGE_CENTER), dtype=float)


def home_chance(offer: Any, visit: Any, level: Any, age: Any, risk: Any) -> FloatArray:
    """``P(home_30 = 1 | A, Z, W)``, the chance of staying home for 30 days."""
    values = _levels(level)
    a = np.asarray(offer, dtype=float)
    z = np.asarray(visit, dtype=float)
    return np.asarray(
        0.50
        - 0.08 * (values == "alone")
        - 0.30 * np.asarray(risk, dtype=float)
        - 0.002 * (np.asarray(age, dtype=float) - _AGE_CENTER)
        + _VISIT * z
        + a * (_direct(values, age) + _OFFER_BY_VISIT * z),
        dtype=float,
    )


#: ``E{Z(a)}`` over risk, which is uniform on (0, 1) and independent of the group and age.
_MEAN_VISIT = {a: float(0.35 + 0.30 * a - 0.15 * 0.5) for a in (0, 1)}

#: The part of the total effect that the visit adds to the direct effect at visit 0.
_THROUGH_VISIT = _VISIT * (_MEAN_VISIT[1] - _MEAN_VISIT[0]) + _OFFER_BY_VISIT * _MEAN_VISIT[1]


def home_support_effect(level: Any, age: Any, visit: float | None = None) -> FloatArray:
    """Return the true effect of the offer for patients of a living situation and age.

    Parameters
    ----------
    level : str or array-like of str
        ``"family"``, ``"alone"`` or ``"paid carer"``.
    age : float or array-like of float
        Age in years.
    visit : {0, 1} or None
        ``None`` gives the total effect, with the 7-day visit left to follow the offer.  ``0`` or
        ``1`` gives the effect with every patient's visit held at that level (the controlled
        direct effect).

    Returns
    -------
    ndarray
        The difference in the chance of staying home for 30 days, offer minus no offer, averaged
        over discharge risk.

    Raises
    ------
    ValueError
        If a level is not one of the three, or ``visit`` is not ``None``, 0 or 1.

    See Also
    --------
    cleverly.datasets.home_support_truth : The group averages and projection coefficients.

    Examples
    --------
    >>> from cleverly.datasets import home_support_effect
    >>> home_support_effect(["family", "alone"], [70, 80]).round(3).tolist()
    [0.073, 0.223]
    >>> home_support_effect("alone", 70, visit=0).round(3).tolist()
    0.14
    """
    if visit is not None and visit not in (0, 1):
        raise ValueError(f"visit must be None, 0 or 1; got {visit!r}")
    direct = _direct(level, age)
    if visit is None:
        return np.asarray(direct + _THROUGH_VISIT, dtype=float)
    return np.asarray(direct + _OFFER_BY_VISIT * float(visit), dtype=float)


def home_support_truth() -> dict[str, float]:
    """Return the exact group truths of the home-support law.

    Returns
    -------
    dict of str to float
        ``group_effect[<level>]`` is the total effect of the offer in each living situation,
        ``direct_effect[visit=<z>][<level>]`` the effect with the visit held at ``z``,
        ``untreated_rate[<level>]`` the chance of staying home without the offer, and
        ``risk_ratio[<level>]`` the ratio of the chances with and without it.  The keys
        ``indicators:<term>``, ``additive:<term>``, ``direct[visit=<z>]:<term>`` and
        ``log_ratio:<term>`` hold the coefficients of four projections by the estimand names a
        fit prints, such as ``indicators:cate_projection[home_support[alone]]``: the projection
        on the group indicators, on the indicators plus a straight line in age, the direct
        effect on the indicators, and the log risk ratio on the indicators.  ``age_slope`` is
        the change in the effect per year of age.

    See Also
    --------
    cleverly.datasets.make_home_support : The draw these values describe.
    cleverly.datasets.home_support_effect : The true effect at any group and age.

    Examples
    --------
    >>> from cleverly.datasets import home_support_truth
    >>> truth = home_support_truth()
    >>> [round(truth[f"group_effect[{level}]"], 3) for level in ("family", "alone", "paid carer")]
    [0.073, 0.193, 0.113]
    >>> round(truth["indicators:cate_projection[home_support[alone]]"], 3)
    0.12
    """
    truth: dict[str, float] = {"age_slope": AGE_SLOPE}
    reference = HOME_SUPPORT_LEVELS[0]
    log_ratio: dict[str, float] = {}
    for level in HOME_SUPPORT_LEVELS:
        effect = float(home_support_effect(level, _AGE_CENTER))
        untreated = 0.50 - 0.08 * (level == "alone") - 0.30 * 0.5 + _VISIT * _MEAN_VISIT[0]
        truth[f"group_effect[{level}]"] = effect
        truth[f"untreated_rate[{level}]"] = untreated
        truth[f"risk_ratio[{level}]"] = (untreated + effect) / untreated
        log_ratio[level] = float(np.log((untreated + effect) / untreated))
        for z in (0, 1):
            truth[f"direct_effect[visit={z}][{level}]"] = float(
                home_support_effect(level, _AGE_CENTER, visit=z)
            )
    terms = {level: f"home_support[{level}]" for level in HOME_SUPPORT_LEVELS[1:]}
    fits: dict[str, tuple[str, Callable[[str], float]]] = {
        "indicators": ("cate_projection", lambda level: truth[f"group_effect[{level}]"]),
        "log_ratio": ("log_crr_projection", lambda level: log_ratio[level]),
    }

    def direct(z: int) -> Callable[[str], float]:
        return lambda level: truth[f"direct_effect[visit={z}][{level}]"]

    for z in (0, 1):
        fits[f"direct[visit={z}]"] = ("cate_projection", direct(z))
    for prefix, (target, value) in fits.items():
        truth[f"{prefix}:{target}[{_INTERCEPT}]"] = value(reference)
        for level, term in terms.items():
            truth[f"{prefix}:{target}[{term}]"] = value(level) - value(reference)
    truth[f"additive:cate_projection[{_INTERCEPT}]"] = (
        truth[f"group_effect[{reference}]"] - AGE_SLOPE * _AGE_CENTER
    )
    for term in terms.values():
        truth[f"additive:cate_projection[{term}]"] = truth[f"indicators:cate_projection[{term}]"]
    truth["additive:cate_projection[age]"] = AGE_SLOPE
    return truth


def make_home_support(
    n: int = 8000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw navigation data whose effect differs by living situation, with the exact truths.

    Parameters
    ----------
    n : int, default=8000
        The number of patients.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per patient: ``age`` in years, ``discharge_risk`` from 0 to 1,
        ``home_support`` (``"family"``, ``"alone"`` or ``"paid carer"``), ``navigation`` (1 for
        the offer), ``pcp_visit_7d`` (1 for a primary-care visit within 7 days) and ``home_30``
        (1 for staying home with no unplanned readmission for 30 days).
    truth : dict of str to float
        The truths of :func:`home_support_truth`.

    Raises
    ------
    ValueError
        If ``n`` is not positive.

    See Also
    --------
    cleverly.datasets.home_support_effect : The true effect at any group and age.
    cleverly.datasets.navigation_protocol : The program protocol a tutorial derives its own from.

    Examples
    --------
    >>> from cleverly.datasets import make_home_support
    >>> frame, truth = make_home_support(n=500, seed=3)
    >>> list(frame.columns)
    ['age', 'discharge_risk', 'home_support', 'navigation', 'pcp_visit_7d', 'home_30']
    >>> sorted(set(frame["home_support"]))
    ['alone', 'family', 'paid carer']
    >>> round(truth["group_effect[alone]"] - truth["group_effect[family]"], 3)
    0.12
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    age = rng.uniform(_AGE_LOW, _AGE_HIGH, n)
    risk = rng.uniform(0.0, 1.0, n)
    level = rng.choice(np.asarray(HOME_SUPPORT_LEVELS, dtype=object), size=n, p=_SHARES)
    offer = rng.binomial(1, offer_chance(level, age, risk))
    visit = rng.binomial(1, visit_chance(offer, risk))
    home = rng.binomial(1, home_chance(offer, visit, level, age, risk))
    payload = {
        "age": age,
        "discharge_risk": risk,
        "home_support": level,
        "navigation": offer,
        "pcp_visit_7d": visit,
        "home_30": home,
    }
    return frame_from_dict(payload, backend=backend), home_support_truth()
