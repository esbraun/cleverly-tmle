r"""A heart-failure medicine with two parts and two competing events, with an exact truth.

A clinic sends patients home after a heart-failure admission with no water pill, a standard dose
or a high dose.  The simulated medicine has two parts: one clears fluid and lowers the chance of a
readmission, and one lowers blood potassium and raises the chance of death.  Each patient is
followed for 90 days, with checks on days 30 and 60 after which a patient can leave follow-up.
The first event is a readmission or a death.

======================  =====================================================================
column                  law
======================  =====================================================================
``older``               Bernoulli(0.4): aged 75 or older
``prior_admissions``    uniform on :math:`\{0, 1, 2, 3\}`: admissions in the past year
``dose``                0 (none), 1 (standard) or 2 (high); multinomial logit in the two
                        covariates (:func:`dose_probabilities`)
leaving follow-up       after the day-30 and day-60 checks, logit linear in the covariates
                        and the dose
death in month k        hazard :math:`\lambda_D(k \mid a, W)`, logit linear in the month, the
                        covariates and the potassium part of the dose
readmission in month k  among those who did not die that month: hazard
                        :math:`h_Y(k \mid a, W)`, logit linear in the month, the covariates
                        and the fluid part of the dose
======================  =====================================================================

``day`` is the day of the first event, uniform on the days of its month, or the day of the last
check (30, 60 or 90) for a patient with no event.  ``event`` is 1 for a readmission, 2 for a
death and 0 for none.  Binning ``day`` onto the grid ``(30, 60, 90)`` gives the discrete law
above, with death counted before readmission within a month.

**The two parts.**  The structural law reads the potassium part of the dose only in the death
hazard and the fluid part only in the readmission hazard.  So the law of a world where the fluid
part is set to level :math:`a_Y` and the potassium part to level :math:`a_D` is

.. math::

    \tilde\lambda_D(k) = \lambda_D(k \mid a_D), \qquad
    \tilde\lambda_Y(k) = h_Y(k \mid a_Y)\,\bigl(1 - \lambda_D(k \mid a_D)\bigr),

which is the separable law of Stensrud, Young, Didelez, Robins and Hernán (2022),
identification (8), with death counted first.  :func:`heart_failure_truth` sums it exactly over
the eight covariate cells.  At :math:`a_Y = a_D` it is the risk of that static dose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._typing import Backend, FloatArray
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = [
    "HEART_FAILURE_DAYS",
    "HEART_FAILURE_EVENTS",
    "dose_probabilities",
    "heart_failure_protocol",
    "heart_failure_truth",
    "make_heart_failure",
]

#: The days of the follow-up checks, which are the grid of the fit.
HEART_FAILURE_DAYS: tuple[int, ...] = (30, 60, 90)

#: The event codes of the ``event`` column.
HEART_FAILURE_EVENTS: dict[int, str] = {1: "readmission", 2: "death"}

_P_OLDER = 0.4
_PRIOR = (0, 1, 2, 3)
_DOSES = (0, 1, 2)

# The dose terms of the two hazards, in log-odds, by dose level.
_POTASSIUM = (0.0, 0.6, 1.2)
_FLUID = (0.0, -0.3, -0.6)


def _death(month: int, dose: Any, older: Any, prior: Any) -> Any:
    """``lambda_D(k | a, W)``: the chance of death in month ``k`` among those at risk."""
    return expit(
        -3.8
        + 0.1 * (month - 1)
        + 0.8 * np.asarray(older, dtype=float)
        + 0.3 * np.asarray(prior, dtype=float)
        + np.asarray(_POTASSIUM)[np.asarray(dose, dtype=int)]
    )


def _readmission(month: int, dose: Any, older: Any, prior: Any) -> Any:
    """``h_Y(k | a, W)``: the chance of readmission in month ``k`` among those who did not die."""
    return expit(
        -2.2
        - 0.1 * (month - 1)
        + 0.2 * np.asarray(older, dtype=float)
        + 0.35 * np.asarray(prior, dtype=float)
        + np.asarray(_FLUID)[np.asarray(dose, dtype=int)]
    )


def _leaving(dose: Any, older: Any, prior: Any) -> Any:
    """The chance of leaving follow-up after a check, among those still followed."""
    return expit(
        -3.0
        + 0.3 * np.asarray(older, dtype=float)
        + 0.15 * np.asarray(prior, dtype=float)
        + 0.2 * np.asarray(dose, dtype=float)
    )


def dose_probabilities(older: Any, prior_admissions: Any) -> FloatArray:
    """``P(dose = a | older, prior_admissions)``, one column per dose.

    Parameters
    ----------
    older : array-like
        1 for a patient aged 75 or older, else 0.
    prior_admissions : array-like
        Admissions in the past year, 0 to 3.

    Returns
    -------
    ndarray
        ``(n, 3)``, the chances of no water pill, a standard dose and a high dose; each row
        sums to one.

    Examples
    --------
    >>> from cleverly.datasets.heart_failure import dose_probabilities
    >>> dose_probabilities([0], [0]).round(3)
    array([[0.351, 0.474, 0.174]])
    """
    x1 = np.atleast_1d(np.asarray(older, dtype=float))
    x2 = np.atleast_1d(np.asarray(prior_admissions, dtype=float))
    scores = np.column_stack(
        [np.zeros_like(x1), 0.3 + 0.35 * x2 - 0.4 * x1, -0.7 + 0.7 * x2 - 0.9 * x1]
    )
    weights = np.exp(scores)
    return weights / weights.sum(axis=1, keepdims=True)


def _cell_risks(
    readmission_dose: int, death_dose: int, older: int, prior: int, first: str
) -> FloatArray:
    """``(3, 2)`` risks of readmission and death by each check, in one covariate cell."""
    out = np.zeros((len(HEART_FAILURE_DAYS), 2))
    survival = 1.0
    running = np.zeros(2)
    for month in range(1, len(HEART_FAILURE_DAYS) + 1):
        if first == "death":
            # The observed law is already in this order: death, then readmission among the rest.
            lam_d = float(_death(month, death_dose, older, prior))
            lam_y = float(_readmission(month, readmission_dose, older, prior)) * (1.0 - lam_d)
        else:
            # Readmission first: its cause-specific hazard at its own dose, then death among
            # those not readmitted, at the death dose.
            at_y = float(_death(month, readmission_dose, older, prior))
            lam_y = float(_readmission(month, readmission_dose, older, prior)) * (1.0 - at_y)
            d_level = float(_death(month, death_dose, older, prior))
            y_at_d = float(_readmission(month, death_dose, older, prior)) * (1.0 - d_level)
            lam_d = d_level / (1.0 - y_at_d) * (1.0 - lam_y)
        running = running + survival * np.array([lam_y, lam_d])
        survival = survival * (1.0 - lam_y - lam_d)
        out[month - 1] = running
    return out


def heart_failure_truth(
    readmission_dose: int, death_dose: int, *, first: str = "death"
) -> pd.DataFrame:
    """The exact risks of :func:`make_heart_failure` when each part of the dose is set apart.

    Parameters
    ----------
    readmission_dose : {0, 1, 2}
        The dose whose fluid part, which acts on readmission, every patient gets.
    death_dose : {0, 1, 2}
        The dose whose potassium part, which acts on death, every patient gets.
    first : {"death", "readmission"}
        Which event counts first within a month.  ``"death"`` is the order of the simulated
        law, so it gives the risks of the world where the two parts are set apart.
        ``"readmission"`` gives the other order's functional of the same data, which differs
        when the two doses differ.

    Returns
    -------
    pandas.DataFrame
        One row per check day, with the columns ``day``, ``readmission`` and ``death``: the
        share of patients with that event first by that day, had every patient stayed in
        follow-up.  At equal doses it is the risk of that dose.

    Examples
    --------
    >>> from cleverly.datasets import heart_failure_truth
    >>> heart_failure_truth(2, 0)["readmission"].round(4).tolist()
    [0.0998, 0.1752, 0.2327]
    """
    for name, value in (("readmission_dose", readmission_dose), ("death_dose", death_dose)):
        if value not in _DOSES:
            raise ValueError(f"{name} must be 0, 1 or 2; got {value!r}")
    if first not in ("death", "readmission"):
        raise ValueError(f"first must be 'death' or 'readmission'; got {first!r}")
    total = np.zeros((len(HEART_FAILURE_DAYS), 2))
    for older in (0, 1):
        for prior in _PRIOR:
            mass = (_P_OLDER if older else 1.0 - _P_OLDER) / len(_PRIOR)
            total += mass * _cell_risks(readmission_dose, death_dose, older, prior, first)
    return pd.DataFrame(
        {"day": list(HEART_FAILURE_DAYS), "readmission": total[:, 0], "death": total[:, 1]}
    )


def make_heart_failure(
    n: int = 4000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Patients sent home on a water pill with two parts, followed for readmission and death.

    One row per patient, with the columns ``older``, ``prior_admissions``, ``dose``, ``day``
    and ``event``.  ``event`` is 1 for a readmission, 2 for a death and 0 for no event by
    ``day``.  A patient who left follow-up after the day-30 or day-60 check has ``event`` 0 and
    that day.

    Parameters
    ----------
    n : int
        Number of patients.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend.

    Returns
    -------
    dataframe
        The observations.
    truth : pandas.DataFrame
        The exact risks for every pair of the dose whose fluid part a patient gets
        (``readmission_dose``) and the dose whose potassium part they get (``death_dose``), at
        each check day, with death counted first within a month: the columns
        ``readmission_dose``, ``death_dose``, ``day``, ``readmission`` and ``death``.  A row
        with equal doses is the risk of that dose.

    Examples
    --------
    >>> from cleverly.datasets import make_heart_failure
    >>> frame, truth = make_heart_failure(n=200, seed=0)
    >>> list(frame.columns)
    ['older', 'prior_admissions', 'dose', 'day', 'event']
    >>> high = truth.query("readmission_dose == 2 and death_dose == 2 and day == 90")
    >>> round(float(high["death"].iloc[0]), 4)
    0.361
    """
    rng = np.random.default_rng(seed)
    older = rng.binomial(1, _P_OLDER, n)
    prior = rng.integers(0, len(_PRIOR), n)
    cumulative = np.cumsum(dose_probabilities(older, prior), axis=1)
    dose = np.minimum((rng.uniform(size=n)[:, None] > cumulative).sum(axis=1), len(_DOSES) - 1)

    day = np.full(n, float(HEART_FAILURE_DAYS[-1]))
    event = np.zeros(n, dtype=np.int64)
    followed = np.ones(n, dtype=bool)
    start = 0
    for month, end in enumerate(HEART_FAILURE_DAYS, start=1):
        if month > 1:
            leave = followed & (rng.uniform(size=n) < _leaving(dose, older, prior))
            day[leave] = float(start)
            followed &= ~leave
        died = followed & (rng.uniform(size=n) < _death(month, dose, older, prior))
        readmitted = (
            followed & ~died & (rng.uniform(size=n) < _readmission(month, dose, older, prior))
        )
        fired = died | readmitted
        day[fired] = rng.integers(start + 1, end + 1, n)[fired].astype(float)
        event[readmitted] = 1
        event[died] = 2
        followed &= ~fired
        start = end

    rows = []
    for readmission_dose in _DOSES:
        for death_dose in _DOSES:
            risks = heart_failure_truth(readmission_dose, death_dose)
            risks.insert(0, "death_dose", death_dose)
            risks.insert(0, "readmission_dose", readmission_dose)
            rows.append(risks)
    truth = pd.concat(rows, ignore_index=True)
    payload = {
        "older": older.astype(float),
        "prior_admissions": prior.astype(float),
        "dose": dose.astype(float),
        "day": day,
        "event": event,
    }
    return frame_from_dict(payload, backend=backend), truth


def heart_failure_protocol() -> StudyProtocol:
    """Return the study description of the heart-failure water-pill question.

    Returns
    -------
    StudyProtocol
        The population, the three dose strategies and the separable strategy, the two events
        and the 90-day horizon of :func:`make_heart_failure`.

    See Also
    --------
    cleverly.datasets.make_heart_failure : The simulated clinic this description fits.

    Examples
    --------
    >>> from cleverly.datasets import heart_failure_protocol
    >>> heart_failure_protocol().horizon
    '90 days after discharge, with checks on days 30, 60 and 90'
    """
    return StudyProtocol(
        target_population=(
            "Adults sent home from the clinic's hospital after an admission for heart failure"
        ),
        eligibility=(
            "Age 18 years or older",
            "Discharged home alive after a heart-failure admission",
        ),
        time_zero="Discharge, after baseline measurement and the discharge prescription",
        treatment_strategies=(
            "Prescribe the water pill at the high dose",
            "Prescribe no water pill",
            "Prescribe the high dose with a potassium supplement that offsets its potassium loss",
        ),
        treatment_versions=(
            "The high dose, taken as prescribed or not",
            "Usual heart-failure care without the water pill",
            "The high dose's fluid-clearing action, with blood potassium kept at its level "
            "without the pill",
        ),
        outcome="First unplanned readmission, and death before any readmission",
        horizon="90 days after discharge, with checks on days 30, 60 and 90",
        intercurrent_event_handling=(
            "Treat death before readmission as a competing event: report the risk of each event "
            "first, with neither prevented",
            "Treat leaving follow-up after a check as censoring, so the target is the risk had "
            "every patient stayed in follow-up (hypothetical strategy)",
            "Analyze the prescribed dose regardless of how the pills were taken",
        ),
        interference_unit="Patient",
        assumption_rationale=(
            "Age and prior admissions recorded at discharge hold every common cause of the dose, "
            "of leaving follow-up, and of the two events",
            "The fluid-clearing part changes death only through readmission, and the potassium "
            "part changes readmission only through death",
            "No unrecorded cause of both readmission and death exists beyond the recorded "
            "characteristics",
        ),
    )
