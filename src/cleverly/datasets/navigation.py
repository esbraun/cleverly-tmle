"""The study protocols and the program data of the care-transition navigation tutorials.

The tutorials under ``docs/examples/`` analyse one program with several methods.  Each
point-treatment tutorial asks a variant of one question, so its protocol differs from the
program's in a few fields.  :func:`navigation_protocol` holds the program's values once, and a
tutorial changes only the fields its question changes with :func:`dataclasses.replace`.
:func:`longitudinal_navigation_protocol` does the same for the tutorials whose time zero is the
discharge and whose navigation is offered at more than one decision.

Several tutorials draw :func:`~cleverly.datasets.make_nonlinear_bounded` and read its columns
under the program's names.  :func:`navigation_data` holds that one mapping.  Other generators give
their ``W`` columns other meanings, so each of those tutorials keeps its own mapping.

The program records the transition score as a share of the maximum score, so the outcome has the
known support ``(0, 1)``.  A cross-fitted fit of a continuous outcome must declare that support as
``q_bounds``, because an undeclared scale is read from the held-out rows as well.  The tutorials
that cross-fit these data therefore pass ``Targeting(q_bounds=(0.0, 1.0))``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import Backend
from ..protocol import StudyProtocol
from ..utils.frames import as_frame
from .synthetic import make_nonlinear_bounded

__all__ = ["longitudinal_navigation_protocol", "navigation_data", "navigation_protocol"]

#: The program name of each :func:`make_nonlinear_bounded` column, in the generator's order.
_PROGRAM_COLUMNS = {
    "Y": "transition_score",
    "A": "transition_navigation",
    "W1": "discharge_risk",
    "W2": "prior_utilization",
    "W3": "medication_burden",
    "W4": "age",
}


def navigation_data(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw the point-treatment program data, with the columns under the program's names.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        The draw of :func:`make_nonlinear_bounded` with each column renamed. The values and the
        column order are unchanged.
    truth : dict of str to float
        Exact causal parameters for the data-generating process.

    See Also
    --------
    cleverly.datasets.make_nonlinear_bounded : The generator this function renames.
    cleverly.datasets.navigation_protocol : The protocol of the program these data describe.

    Notes
    -----
    The same seed gives the same rows as :func:`make_nonlinear_bounded`.  Only the names change.

    The transition score is a share of the maximum score, so every value lies in ``(0, 1)``.  A
    cross-fitted fit of it declares ``q_bounds=(0.0, 1.0)``.

    ========= =======================
    generator program
    ========= =======================
    ``Y``     ``transition_score``
    ``A``     ``transition_navigation``
    ``W1``    ``discharge_risk``
    ``W2``    ``prior_utilization``
    ``W3``    ``medication_burden``
    ``W4``    ``age``
    ========= =======================

    Examples
    --------
    >>> from cleverly.datasets import navigation_data
    >>> frame, truth = navigation_data(n=500, seed=21)
    >>> print(*frame.columns, sep="\n")
    transition_score
    transition_navigation
    discharge_risk
    prior_utilization
    medication_burden
    age
    >>> frame.shape
    (500, 6)
    >>> round(truth["ate"], 3)
    0.163
    """
    frame, truth = make_nonlinear_bounded(n, seed=seed, backend=backend)
    return as_frame(frame).rename(_PROGRAM_COLUMNS).to_native(), truth


def navigation_protocol() -> StudyProtocol:
    """Return the protocol for the standard navigation offer in the tutorial program.

    Returns
    -------
    StudyProtocol
        The point-treatment protocol that compares the standard navigation offer with usual
        discharge support on the 30-day transition score.

    See Also
    --------
    cleverly.StudyProtocol : The record this function fills.
    cleverly.datasets.navigation_data : The program data the point-treatment tutorials draw.
    cleverly.datasets.longitudinal_navigation_protocol : The protocol of the two-decision program.

    Notes
    -----
    The values restate the shared study design in ``docs/examples/index.md``.  A tutorial whose
    question changes a field, such as the assumption rationale or the strategies, passes the
    result to :func:`dataclasses.replace`.  The synthetic generators make the identification
    assumptions true by construction.  The rationale here describes the program, not the draw.

    The outcome field states the score's support.  A cross-fitted fit of a continuous outcome
    declares that support as ``q_bounds``, so a tutorial whose synthetic stand-in has another
    support replaces this field and says which fit that support allows.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from cleverly.datasets import navigation_protocol
    >>> protocol = navigation_protocol()
    >>> protocol.horizon
    '30 days after discharge'
    >>> protocol.fingerprint
    '623fc2c240615d26'

    A tutorial with another question replaces only the fields that question changes:

    >>> screened = replace(protocol, interference_unit="Navigator team")
    >>> screened.outcome == protocol.outcome
    True
    >>> screened.fingerprint == protocol.fingerprint
    False
    """
    return StudyProtocol(
        target_population=(
            "Adults with a discharge-home order at a participating hospital "
            "during the enrollment period"
        ),
        eligibility=(
            "Age 18 years or older",
            "Discharge home ordered at a participating hospital",
        ),
        time_zero=(
            "Discharge-home order, after baseline measurement and before the navigation offer"
        ),
        treatment_strategies=(
            "Offer standard transition navigation",
            "Provide usual discharge support",
        ),
        treatment_versions=(
            "Bedside transition plan and two scheduled navigator contacts within 30 days",
            "No access to the transition-navigation offer",
        ),
        outcome="Patient-reported transition score, as a share of the maximum score",
        horizon="30 days after discharge",
        intercurrent_event_handling=(
            "Use the transition score regardless of readmission",
            "Analyze the offer regardless of completed contacts",
            "The protocol scores death before day 30 as the worst transition score "
            "(composite strategy)",
        ),
        interference_unit="Individual patient",
        assumption_rationale=(
            "The recorded baseline variables cover the measured common causes",
            "The standardized offer and version records support consistency",
            "Reserved navigator capacity and access controls support no interference",
        ),
    )


def longitudinal_navigation_protocol() -> StudyProtocol:
    """Return the protocol for navigation offered at two decisions after discharge.

    Returns
    -------
    StudyProtocol
        The protocol that compares navigation at discharge and day seven with no navigation at
        either decision, on the top-box transition score at day 30.

    See Also
    --------
    cleverly.datasets.navigation_protocol : The protocol of the one-decision program.
    cleverly.datasets.make_longitudinal : The synthetic law the two-decision tutorial draws.

    Notes
    -----
    Time zero is the discharge itself, so eligibility requires a live discharge.  The strategies
    name the two static plans.  A tutorial that reports another plan or a dynamic rule appends its
    strategy and its version with :func:`dataclasses.replace`, and a tutorial with another outcome
    replaces the outcome, the horizon, and the intercurrent-event handling.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from cleverly.datasets import longitudinal_navigation_protocol
    >>> program = longitudinal_navigation_protocol()
    >>> program.treatment_strategies
    ('Offer navigation at discharge and day seven', 'Offer no navigation at either decision')
    >>> program.fingerprint
    '0fe91ade3fc3f249'

    A tutorial that also reports a dynamic rule names it as a third strategy:

    >>> with_rule = replace(
    ...     program,
    ...     treatment_strategies=(
    ...         *program.treatment_strategies,
    ...         "Offer navigation at discharge, and continue on day seven if engaged",
    ...     ),
    ...     treatment_versions=(
    ...         *program.treatment_versions,
    ...         "The discharge contact, and the day-seven contact for engaged patients",
    ...     ),
    ... )
    >>> len(with_rule.treatment_strategies)
    3
    """
    program = navigation_protocol()
    return StudyProtocol(
        target_population=(
            "Adults discharged home from a participating hospital during the enrollment period"
        ),
        eligibility=(
            "Age 18 years or older",
            "Discharged alive",
            "Discharged home from a participating hospital",
        ),
        time_zero="Hospital discharge, after baseline measurement and before first assignment",
        treatment_strategies=(
            "Offer navigation at discharge and day seven",
            "Offer no navigation at either decision",
        ),
        treatment_versions=(
            "The declared discharge and day-seven navigation contacts",
            "Usual discharge support without navigation contacts",
        ),
        outcome="Top-box patient-reported transition score",
        horizon=program.horizon,
        intercurrent_event_handling=(
            program.intercurrent_event_handling[0],
            "The protocol scores death before day 30 as not top box (composite strategy)",
            "Analyze each navigation offer regardless of completed contacts",
        ),
        interference_unit=program.interference_unit,
        assumption_rationale=(
            "Recorded history covers the measured common causes at each decision",
            "Version records support consistency at both navigation decisions",
            "Reserved navigator capacity supports no interference between patients",
        ),
    )
