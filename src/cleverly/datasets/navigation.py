"""The study protocol of the care-transition navigation program the tutorials share.

The tutorials under ``docs/examples/`` analyse one program with several methods.  Each
point-treatment tutorial asks a variant of one question, so its protocol differs from the
program's in a few fields.  :func:`navigation_protocol` holds the program's values once, and a
tutorial changes only the fields its question changes with :func:`dataclasses.replace`.
"""

from __future__ import annotations

from ..protocol import StudyProtocol

__all__ = ["navigation_protocol"]


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
    cleverly.datasets.make_nonlinear_ate : The synthetic law the reference tutorial draws.

    Notes
    -----
    The values restate the shared study design in ``docs/examples/index.md``.  A tutorial whose
    question changes a field, such as the assumption rationale or the strategies, passes the
    result to :func:`dataclasses.replace`.  The synthetic generators make the identification
    assumptions true by construction.  The rationale here describes the program, not the draw.

    Examples
    --------
    >>> from dataclasses import replace
    >>> from cleverly.datasets import navigation_protocol
    >>> protocol = navigation_protocol()
    >>> protocol.horizon
    '30 days after discharge'
    >>> protocol.fingerprint
    '2dd268e1f5ab29ae'

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
        outcome="Patient-reported transition score",
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
