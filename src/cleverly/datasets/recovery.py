r"""Navigation offered at three points after discharge, a 30-day recovery score, and its truths.

A hospital discharges adults home.  A navigator decides whether to offer navigation three times:
at discharge, on day 10 and on day 20.  Before each later decision a check-in flags whether the
patient has symptoms, and the navigator reads that flag.  The outcome is a recovery score on day
30, on a scale from 0 to 100.  Every patient stays in follow-up, and nobody dies in the simulated
program.

==========================  =======================================================================
column                      law
==========================  =======================================================================
``lives_alone``             Bernoulli(0.35)
``discharge_risk``          uniform on [0, 1]
``offer_discharge``         Bernoulli, logit linear in ``discharge_risk`` and ``lives_alone``
``symptoms_day10``          Bernoulli, probability linear in ``lives_alone``, ``offer_discharge``
                            and their product: the offer lowers the chance of symptoms only for
                            patients who live alone
``offer_day10``             Bernoulli, logit linear in the history
``symptoms_day20``          Bernoulli, probability linear in ``lives_alone``, ``offer_day10``
                            and their product: the offer lowers the chance of symptoms a little
                            for every patient and much more for patients who live alone
``offer_day20``             Bernoulli, logit linear in the history
``recovery_day30``          linear in every earlier column, plus uniform noise on [-8, 8]; every
                            value lies between 10 and 84, inside the scale of 0 to 100
==========================  =======================================================================

**Why this law.**  The tutorial ``docs/examples/model-based-over-time.ipynb`` fits two methods
that need different models to be right, and both are right here.

- The parametric g-formula models each symptom flag and the score.  Each flag has two binary
  parents only (``discharge_risk`` does not enter it), so a logistic model with every product of
  two history columns (``GLM(interactions=2)``) is saturated in them and holds the true law.
  The score is linear in the history, so a main-terms Gaussian model holds its mean.  A
  main-terms logistic model of the flags misses the product of each offer with ``lives_alone``,
  which is the wrong model the tutorial shows.  The offer at discharge goes mostly to patients who
  live alone, so that model learns the offer's effect from them and carries it to the others.
- The structural nested mean model needs the effect of each offer, with the later offers withheld,
  to be linear in declared baseline features.  The flags are linear-probability laws, so the
  effect of the discharge offer is :math:`\gamma_1 = 2 + 12.1 \cdot` ``lives_alone``, that of
  the day-10 offer :math:`\gamma_2 = 2 + 2.8 \cdot` ``lives_alone``, and that of the day-20 offer
  :math:`\gamma_3 = 2`.  The treatment logits are main terms in the history, and the mean of each
  blipped-down outcome given the history is linear in it, so main-terms ``GLM`` nuisances are
  exact.

**Truths.**  ``discharge_risk`` enters only the offers and the score, linearly, so it averages to
its mean 0.5.  Every other variable is binary, so a plan mean is a finite sum over
``lives_alone`` and the two flags (:func:`recovery_score_truth`), and each blip is a difference of
two such sums within a ``lives_alone`` group.  ``tests/unit/test_datasets_recovery.py`` compares
each truth with a Monte Carlo draw of the intervened law.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .._typing import Backend
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict
from .navigation import longitudinal_navigation_protocol

__all__ = [
    "RECOVERY_BOUNDS",
    "RECOVERY_PLANS",
    "make_recovery_score",
    "recovery_score_protocol",
    "recovery_score_truth",
]

#: The known range of the 30-day recovery score.  Every draw of the law falls inside it.
RECOVERY_BOUNDS = (0.0, 100.0)

#: The plans of the tutorial, as (offer at discharge, offer on day 10, offer on day 20).
RECOVERY_PLANS: dict[str, tuple[float, float, float]] = {
    "always": (1.0, 1.0, 1.0),
    "never": (0.0, 0.0, 0.0),
}

#: The share of patients who live alone.
_ALONE = 0.35

#: The coefficients of the law.  The offers are on the logit scale, the flags on the
#: probability scale, and the score on its own scale.
_LAW: dict[str, float] = {
    # offer at discharge
    "o1_intercept": -2.4,
    "o1_risk": 1.0,
    "o1_alone": 2.4,
    # symptoms flagged on day 10
    "s1_intercept": 0.30,
    "s1_alone": 0.45,
    "s1_offer_alone": -0.55,
    # offer on day 10
    "o2_intercept": -1.2,
    "o2_risk": 1.5,
    "o2_alone": 0.4,
    "o2_symptoms": 1.2,
    "o2_offer": 0.6,
    # symptoms flagged on day 20
    "s2_intercept": 0.20,
    "s2_alone": 0.10,
    "s2_offer": -0.05,
    "s2_offer_alone": -0.20,
    # offer on day 20
    "o3_intercept": -1.3,
    "o3_risk": 1.5,
    "o3_alone": 0.3,
    "o3_symptoms": 1.2,
    "o3_offer": 0.7,
    # recovery score on day 30
    "y_intercept": 70.0,
    "y_risk": -12.0,
    "y_alone": -4.0,
    "y_offer_discharge": 2.0,
    "y_offer_day10": 1.3,
    "y_offer_day20": 2.0,
    "y_symptoms_day10": -22.0,
    "y_symptoms_day20": -14.0,
}

#: The half-width of the uniform noise of the score.
_NOISE = 8.0


def _symptoms_day10(alone: Any, offer: Any) -> Any:
    """The chance of symptoms on day 10."""
    return _LAW["s1_intercept"] + _LAW["s1_alone"] * alone + _LAW["s1_offer_alone"] * offer * alone


def _symptoms_day20(alone: Any, offer: Any) -> Any:
    """The chance of symptoms on day 20."""
    return (
        _LAW["s2_intercept"]
        + _LAW["s2_alone"] * alone
        + _LAW["s2_offer"] * offer
        + (_LAW["s2_offer_alone"] * offer * alone)
    )


def _score_mean(risk: Any, alone: Any, o1: Any, s1: Any, o2: Any, s2: Any, o3: Any) -> Any:
    """The mean recovery score given the history."""
    return (
        _LAW["y_intercept"]
        + _LAW["y_risk"] * risk
        + _LAW["y_alone"] * alone
        + _LAW["y_offer_discharge"] * o1
        + _LAW["y_offer_day10"] * o2
        + _LAW["y_offer_day20"] * o3
        + _LAW["y_symptoms_day10"] * s1
        + _LAW["y_symptoms_day20"] * s2
    )


def _group_mean(alone: float, o1: float, o2: float, o3: float) -> float:
    """The mean score under the plan ``(o1, o2, o3)`` given ``lives_alone = alone``."""
    total = 0.0
    p1 = float(_symptoms_day10(alone, o1))
    for s1, w1 in ((1.0, p1), (0.0, 1.0 - p1)):
        p2 = float(_symptoms_day20(alone, o2))
        for s2, w2 in ((1.0, p2), (0.0, 1.0 - p2)):
            total += w1 * w2 * float(_score_mean(0.5, alone, o1, s1, o2, s2, o3))
    return total


def recovery_score_truth(o1: float, o2: float, o3: float) -> float:
    """Return the exact mean 30-day recovery score under one static plan.

    Parameters
    ----------
    o1 : float
        The offer at discharge, 0 or 1.
    o2 : float
        The offer on day 10, 0 or 1.
    o3 : float
        The offer on day 20, 0 or 1.

    Returns
    -------
    float
        The mean score had every patient followed the plan, a finite sum over ``lives_alone``
        and the two symptom flags.

    See Also
    --------
    cleverly.datasets.make_recovery_score : The simulated program whose truths these are.

    Examples
    --------
    >>> from cleverly.datasets import recovery_score_truth
    >>> round(recovery_score_truth(1, 1, 1) - recovery_score_truth(0, 0, 0), 4)
    11.215
    """
    return (1.0 - _ALONE) * _group_mean(0.0, o1, o2, o3) + _ALONE * _group_mean(1.0, o1, o2, o3)


def _blips() -> dict[str, float]:
    """The blip coefficients, as the structural nested mean model names them."""
    zero = (0.0, 0.0, 0.0)
    columns = ("offer_discharge", "offer_day10", "offer_day20")
    out: dict[str, float] = {}
    for node, column in enumerate(columns):
        plan = tuple(1.0 if t == node else 0.0 for t in range(3))
        effect = {alone: _group_mean(alone, *plan) - _group_mean(alone, *zero) for alone in (0, 1)}
        out[f"blip[{column}:1.0:intercept]"] = effect[0]
        if node < 2:
            out[f"blip[{column}:1.0:lives_alone]"] = effect[1] - effect[0]
    return out


def make_recovery_score(
    n: int = 4000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw the three-decision navigation program with a 30-day recovery score.

    Parameters
    ----------
    n : int, default=4000
        The number of patients.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per patient, with the columns ``lives_alone``, ``discharge_risk``,
        ``offer_discharge``, ``symptoms_day10``, ``offer_day10``, ``symptoms_day20``,
        ``offer_day20`` and ``recovery_day30`` in time order.
    truth : dict of str to float
        ``ey_regimen[always]``, ``ey_regimen[never]`` and ``ate_regimen[always vs never]``,
        and each blip coefficient of the declared features ``lives_alone`` at the first two
        offers and the intercept at the third, under the names a fit reports them by.

    See Also
    --------
    cleverly.datasets.recovery_score_truth : The exact mean score under one plan.
    cleverly.datasets.recovery_score_protocol : The study description of the program.

    Examples
    --------
    >>> from cleverly.datasets import make_recovery_score
    >>> frame, truth = make_recovery_score(10, seed=0)
    >>> frame.shape
    (10, 8)
    >>> round(truth["ate_regimen[always vs never]"], 4)
    11.215
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    alone = rng.binomial(1, _ALONE, n).astype(float)
    risk = rng.uniform(0.0, 1.0, n)
    o1 = rng.binomial(
        1, expit(_LAW["o1_intercept"] + _LAW["o1_risk"] * risk + _LAW["o1_alone"] * alone)
    ).astype(float)
    s1 = rng.binomial(1, _symptoms_day10(alone, o1)).astype(float)
    o2 = rng.binomial(
        1,
        expit(
            _LAW["o2_intercept"]
            + _LAW["o2_risk"] * risk
            + _LAW["o2_alone"] * alone
            + _LAW["o2_symptoms"] * s1
            + _LAW["o2_offer"] * o1
        ),
    ).astype(float)
    s2 = rng.binomial(1, _symptoms_day20(alone, o2)).astype(float)
    o3 = rng.binomial(
        1,
        expit(
            _LAW["o3_intercept"]
            + _LAW["o3_risk"] * risk
            + _LAW["o3_alone"] * alone
            + _LAW["o3_symptoms"] * s2
            + _LAW["o3_offer"] * o2
        ),
    ).astype(float)
    score = _score_mean(risk, alone, o1, s1, o2, s2, o3) + rng.uniform(-_NOISE, _NOISE, n)
    payload = {
        "lives_alone": alone,
        "discharge_risk": risk,
        "offer_discharge": o1,
        "symptoms_day10": s1,
        "offer_day10": o2,
        "symptoms_day20": s2,
        "offer_day20": o3,
        "recovery_day30": score,
    }
    truth = {
        f"ey_regimen[{name}]": recovery_score_truth(*plan) for name, plan in RECOVERY_PLANS.items()
    }
    truth["ate_regimen[always vs never]"] = truth["ey_regimen[always]"] - truth["ey_regimen[never]"]
    truth.update(_blips())
    return frame_from_dict(payload, backend=backend), truth


def recovery_score_protocol() -> StudyProtocol:
    """Return the study description of the three-decision navigation program.

    Returns
    -------
    StudyProtocol
        :func:`~cleverly.datasets.longitudinal_navigation_protocol` with offers at discharge, day
        10 and day 20, and the day-30 recovery score as the outcome.

    See Also
    --------
    cleverly.datasets.make_recovery_score : The simulated program this description fits.

    Examples
    --------
    >>> from cleverly.datasets import recovery_score_protocol
    >>> recovery_score_protocol().outcome
    'Patient-reported recovery score on day 30, on a scale from 0 to 100'
    """
    program = longitudinal_navigation_protocol()
    return replace(
        program,
        treatment_strategies=(
            "Offer navigation at discharge, on day 10 and on day 20",
            "Offer no navigation at any of the three decisions",
        ),
        treatment_versions=(
            "The declared navigation contact at each of the three decisions",
            "Usual discharge support without navigation contacts",
        ),
        outcome="Patient-reported recovery score on day 30, on a scale from 0 to 100",
        intercurrent_event_handling=(
            "Every patient completes the day-30 score, so no patient leaves follow-up",
            "The protocol scores death before day 30 as 0 (composite strategy)",
            program.intercurrent_event_handling[2],
        ),
        assumption_rationale=(
            "The navigator reads only the recorded discharge risk, living situation, symptom "
            "flags and earlier offers, which supports no hidden confounding at each decision",
            "Version records support consistency at all three navigation decisions",
            program.assumption_rationale[2],
        ),
    )
