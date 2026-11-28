r"""Exact times to readmission or death after discharge, with an exact truth.

The care-transition program offers navigation at discharge, more often to patients with a higher
readmission-risk score.  Follow-up records, in months since discharge, the exact time of the
first unplanned readmission, of death before any readmission, or of leaving follow-up.  Death is
a competing event: a patient who dies can no longer be readmitted.

======================  =====================================================================
column                  law
======================  =====================================================================
``risk_score``          :math:`X \sim N(0, 1)`
``offered``             :math:`A \sim \operatorname{Bern}\{\operatorname{expit}(X)\}`
readmission             constant hazard :math:`h_1(a, x) = \exp(-1.5 + 0.5 x - a)`
death                   constant hazard :math:`h_2(a, x) = \exp(-2 - 0.5 x - 0.5 a)`
leaving follow-up       exponential with rate 0.15, independent of :math:`(A, X)` and of the
                        event times
======================  =====================================================================

``time`` is the earlier of the first event and leaving follow-up, and ``event`` is 1 for a
readmission, 2 for a death and 0 for leaving follow-up first.  The law has no unit of its own:
the plots tutorial reads the same law as years to relapse.

**Truth.**  With constant hazards and :math:`h = h_1 + h_2`, the risk of cause :math:`j` by
:math:`t` given :math:`(a, x)` is :math:`F_j(t \mid a, x) = h_j / h \cdot (1 - e^{-h t})`, and
the time lost to cause :math:`j` up to :math:`\tau` is

.. math::

    \int_0^\tau F_j(s \mid a, x)\,ds = \frac{h_j}{h}\Bigl(\tau - \frac{1 - e^{-h\tau}}{h}\Bigr).

The restricted mean time free of both events is :math:`\tau` minus the sum of the two.  In a
world where nobody dies and each patient keeps the same readmission hazard, the risk of
readmission by :math:`t` is :math:`1 - e^{-h_1 t}`.  Each truth averages its conditional quantity
over :math:`X` by adaptive quadrature against the normal density on :math:`[-12, 12]`
(:func:`exact_readmission_truth`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import integrate, stats

from .._typing import Backend
from ..protocol import StudyProtocol
from ..utils.frames import frame_from_dict
from .navigation import longitudinal_navigation_protocol

__all__ = [
    "EXACT_ARMS",
    "EXACT_CAUSES",
    "exact_readmission_protocol",
    "exact_readmission_truth",
    "make_exact_readmission",
]

#: The two plans, by the value of ``offered`` each one sets.
EXACT_ARMS: dict[str, int] = {"offered": 1, "not offered": 0}
#: The two events, by their code in ``event``.
EXACT_CAUSES: dict[int, str] = {1: "readmission", 2: "death"}
#: The rate of leaving follow-up, per month.
_DROPOUT_RATE = 0.15
_REFERENCE = "not offered"
_CONTRAST = "offered vs not offered"


def _rates(arm: Any, x: Any) -> tuple[Any, Any]:
    """The readmission and death hazards given the offer and the risk score."""
    return np.exp(-1.5 + 0.5 * x - arm), np.exp(-2.0 - 0.5 * x - 0.5 * arm)


def _average(conditional: Callable[[float], float]) -> float:
    """``E conditional(X)`` for ``X ~ N(0, 1)``, by adaptive quadrature."""
    value, _ = integrate.quad(
        lambda x: conditional(x) * stats.norm.pdf(x), -12.0, 12.0, epsabs=1e-13, epsrel=1e-12
    )
    return float(value)


def _risk(arm: float, cause: int, t: float) -> float:
    def conditional(x: float) -> float:
        h1, h2 = _rates(arm, x)
        h = h1 + h2
        return float((h1 if cause == 1 else h2) / h * (1.0 - np.exp(-h * t)))

    return _average(conditional)


def _time_lost(arm: float, cause: int, tau: float) -> float:
    def conditional(x: float) -> float:
        h1, h2 = _rates(arm, x)
        h = h1 + h2
        return float((h1 if cause == 1 else h2) / h * (tau - (1.0 - np.exp(-h * tau)) / h))

    return _average(conditional)


def _readmission_without_death(arm: float, t: float) -> float:
    return _average(lambda x: float(1.0 - np.exp(-_rates(arm, x)[0] * t)))


def _check_times(
    horizons: Sequence[float], restricted_horizons: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    times = tuple(float(t) for t in horizons)
    if (
        not times
        or not all(np.isfinite(t) and t > 0 for t in times)
        or list(times) != sorted(set(times))
    ):
        raise ValueError("horizons must be finite, positive, increasing and distinct")
    restricted = tuple(float(t) for t in restricted_horizons)
    if any(t not in times for t in restricted):
        raise ValueError("each restricted horizon must be one of the horizons")
    return times, restricted


def exact_readmission_truth(
    horizons: Sequence[float] = (1.5, 3.0),
    *,
    restricted_horizons: Sequence[float] = (),
    death_prevented: bool = False,
) -> dict[str, float]:
    """The true risks, contrasts and restricted means of :func:`make_exact_readmission`.

    Parameters
    ----------
    horizons : sequence of float
        The target times, in months, positive and increasing.
    restricted_horizons : sequence of float
        The cut-offs of the restricted means.  This generator asks each to be one of
        ``horizons``; the continuous-time fit itself also takes other cut-offs.
    death_prevented : bool
        Return instead the risk of readmission in a world where nobody dies and each patient
        keeps the same readmission hazard, keyed as a fit with readmission as its only event
        names it.

    Returns
    -------
    dict of str to float
        Each truth keyed by the name a continuous-time fit with the plans :data:`EXACT_ARMS`
        and the events :data:`EXACT_CAUSES` gives its estimate: each plan's risk of each event
        (``cif_regimen[...]``), the difference and ratio of the two plans' risks
        (``ate_regimen[...]``, ``rr_regimen[...]``), and at each restricted horizon the time
        lost to each event (``rmtl_regimen[...]``) and the time free of both
        (``rmst_regimen[...]``) for each plan and their difference.  With ``death_prevented``
        the keys are ``risk_regimen[...]``, ``ate_regimen[...]`` and ``rr_regimen[...]`` with
        no event name, and ``restricted_horizons`` must be empty.

    Examples
    --------
    >>> from cleverly.datasets import exact_readmission_truth
    >>> truth = exact_readmission_truth((1.5, 3.0), restricted_horizons=(3.0,))
    >>> round(truth["ate_regimen[offered vs not offered, readmission @ t=3]"], 4)
    -0.2125
    >>> round(truth["rmst_regimen[offered vs not offered @ t=3]"], 3)
    0.556
    """
    times, restricted = _check_times(horizons, restricted_horizons)
    out: dict[str, float] = {}
    if death_prevented:
        if restricted:
            raise ValueError("death_prevented has no restricted means")
        for t in times:
            for arm, code in EXACT_ARMS.items():
                out[f"risk_regimen[{arm} @ t={t:g}]"] = _readmission_without_death(code, t)
            treated = out[f"risk_regimen[offered @ t={t:g}]"]
            control = out[f"risk_regimen[{_REFERENCE} @ t={t:g}]"]
            out[f"ate_regimen[{_CONTRAST} @ t={t:g}]"] = treated - control
            out[f"rr_regimen[{_CONTRAST} @ t={t:g}]"] = treated / control
        return out
    for arm, code in EXACT_ARMS.items():
        for cause_code, cause in EXACT_CAUSES.items():
            for t in times:
                out[f"cif_regimen[{arm}, {cause} @ t={t:g}]"] = _risk(code, cause_code, t)
    for cause in EXACT_CAUSES.values():
        for t in times:
            treated = out[f"cif_regimen[offered, {cause} @ t={t:g}]"]
            control = out[f"cif_regimen[{_REFERENCE}, {cause} @ t={t:g}]"]
            out[f"ate_regimen[{_CONTRAST}, {cause} @ t={t:g}]"] = treated - control
            out[f"rr_regimen[{_CONTRAST}, {cause} @ t={t:g}]"] = treated / control
    for tau in restricted:
        for arm, code in EXACT_ARMS.items():
            lost = 0.0
            for cause_code, cause in EXACT_CAUSES.items():
                value = _time_lost(code, cause_code, tau)
                out[f"rmtl_regimen[{arm}, {cause} @ t={tau:g}]"] = value
                lost += value
            out[f"rmst_regimen[{arm} @ t={tau:g}]"] = tau - lost
        for cause in EXACT_CAUSES.values():
            out[f"rmtl_regimen[{_CONTRAST}, {cause} @ t={tau:g}]"] = (
                out[f"rmtl_regimen[offered, {cause} @ t={tau:g}]"]
                - out[f"rmtl_regimen[{_REFERENCE}, {cause} @ t={tau:g}]"]
            )
        out[f"rmst_regimen[{_CONTRAST} @ t={tau:g}]"] = (
            out[f"rmst_regimen[offered @ t={tau:g}]"]
            - out[f"rmst_regimen[{_REFERENCE} @ t={tau:g}]"]
        )
    return out


def make_exact_readmission(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    horizons: Sequence[float] = (1.5, 3.0),
    restricted_horizons: Sequence[float] = (3.0,),
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Patients offered navigation at discharge, with exact times to readmission or death.

    One row per patient, with the columns ``time`` (months since discharge), ``event`` (0 left
    follow-up first, 1 readmission, 2 death before any readmission), ``offered`` and
    ``risk_score``.

    Parameters
    ----------
    n : int
        Number of patients.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    horizons : sequence of float
        The target times of the returned truth, in months.
    restricted_horizons : sequence of float
        The cut-offs of the returned restricted means, each one of ``horizons``.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        The observations.
    truth : dict of str to float
        :func:`exact_readmission_truth` at ``horizons`` and ``restricted_horizons``.

    See Also
    --------
    cleverly.datasets.exact_readmission_protocol : The study description of this question.

    Examples
    --------
    >>> from cleverly.datasets import make_exact_readmission
    >>> frame, truth = make_exact_readmission(n=200, seed=0)
    >>> list(frame.columns)
    ['time', 'event', 'offered', 'risk_score']
    >>> round(truth["cif_regimen[not offered, readmission @ t=3]"], 4)
    0.4251
    """
    if n < 1:
        raise ValueError("n must be positive")
    times, restricted = _check_times(horizons, restricted_horizons)
    rng = np.random.default_rng(seed)
    risk_score = rng.normal(size=n)
    offered = rng.binomial(1, 1 / (1 + np.exp(-risk_score)))
    readmission_rate, death_rate = _rates(offered, risk_score)
    event_time = rng.exponential(1 / (readmission_rate + death_rate))
    cause = np.where(rng.uniform(size=n) < readmission_rate / (readmission_rate + death_rate), 1, 2)
    dropout = rng.exponential(1 / _DROPOUT_RATE, size=n)
    payload = {
        "time": np.minimum(event_time, dropout),
        "event": np.where(event_time <= dropout, cause, 0),
        "offered": offered,
        "risk_score": risk_score,
    }
    truth = exact_readmission_truth(times, restricted_horizons=restricted)
    return frame_from_dict(payload, backend=backend), truth


def exact_readmission_protocol() -> StudyProtocol:
    """Return the study description of the navigation question with exact event times.

    It is :func:`cleverly.datasets.longitudinal_navigation_protocol` with one decision at
    discharge, the two competing events and a three-month horizon.

    Returns
    -------
    StudyProtocol
        The study description of :func:`make_exact_readmission`.

    See Also
    --------
    cleverly.datasets.make_exact_readmission : The simulated patients this description fits.

    Examples
    --------
    >>> from cleverly.datasets import exact_readmission_protocol
    >>> exact_readmission_protocol().horizon
    '3 months after discharge, with risks read at 1.5 and 3 months'
    """
    program = longitudinal_navigation_protocol()
    return replace(
        program,
        time_zero="Hospital discharge, when the risk score is recorded and the offer is made",
        treatment_strategies=(
            "Offer navigation to every patient at discharge",
            "Offer navigation to no patient",
        ),
        treatment_versions=(
            "The declared discharge navigation contacts",
            "Usual discharge support without navigation contacts",
        ),
        outcome="First unplanned readmission, and death before any readmission",
        horizon="3 months after discharge, with risks read at 1.5 and 3 months",
        intercurrent_event_handling=(
            "Treat death before readmission as a competing event: report the risk of each event, "
            "with neither prevented",
            "Treat leaving follow-up as censoring, so the target is the risk had every patient "
            "stayed in follow-up (hypothetical strategy)",
            program.intercurrent_event_handling[2],
        ),
        assumption_rationale=(
            "The risk score recorded at discharge holds every common cause of the offer, of "
            "the two events and of leaving follow-up",
            "Version records support that the discharge offer is one well-defined action, so a "
            "patient's recorded events are those they would have had under the offer they got",
            program.assumption_rationale[2],
        ),
    )
