"""The attributable population-intervention estimands and their shared refusal.

``par`` and ``paf`` contain both the natural-course mean and a reference-intervention
mean.  Their joint missing-outcome construction remains outside the implemented surface
and is tracked by ``docs/roadmap.md`` RM8.  The scalar natural-course mean itself has its
own missing-outcome score equation and therefore no longer belongs to this refusal.

Three modules that do not import one another reach this refusal: the target context that
would compute the mean, the estimator that resolves an estimand list, and the study that
identifies a typed question.  Written three times it was three sentences, three copies of
the estimand set, three copies of the roadmap mapping, and three exception types, only one
of which derived from :class:`~cleverly.CleverlyError`.  This module holds the one set,
the one sentence and the one type.  It imports only :mod:`cleverly.exceptions`, so any of
the three callers can reach it.

Each caller keeps its own condition, because the three conditions are genuinely
different: an observation mask, a requested estimand list, and one identified target.
What they share is the message, so this module builds the error and the caller raises it.

The natural-course mean's own name, its fit predicate and the two refusals that name it
live here for the same reason.  Five modules across four subpackages ask "is this the
missing-outcome natural-course fit?" -- the estimator, the assessment facade, the
positivity report, the missingness tilt and the nuisance diagnostics.  This module
imports only :mod:`cleverly.exceptions`, so every one of them can reach it at module
scope, which the previous home under :mod:`cleverly.sensitivity` could not offer: that
package's ``__init__`` imports :mod:`cleverly.assessment`, so an assessment-side import
of it had to be written three times inside functions to break the cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..exceptions import CapabilityError

__all__ = [
    "NATURAL_COURSE_SUPPORT_REFUSAL",
    "NATURAL_COURSE_TARGET",
    "NATURAL_COURSE_TILT_REFUSAL",
    "POPULATION_INTERVENTION_TARGETS",
    "is_natural_course_fit",
    "population_intervention_refusal",
]

#: The natural-course mean's estimand name, written once so the string is not respelled
#: at the eight sites that test for it.
NATURAL_COURSE_TARGET = "ey_obs"

NATURAL_COURSE_TILT_REFUSAL = (
    "the implemented missingness tilt is arm-specific; NaturalCourseMean needs a "
    "natural-course sensitivity parameter, which is not implemented"
)

NATURAL_COURSE_SUPPORT_REFUSAL = (
    "NaturalCourseMean with missing outcomes fits no treatment propensity, so the "
    "arm-propensity support report is not applicable; inspect the missingness row from "
    "diagnostics.nuisance_models() and the targeting score instead"
)


def is_natural_course_fit(result: Any) -> bool:
    """Whether ``result`` is the missing-outcome natural-course fit.

    Both halves are load-bearing.  A fit reports the natural-course mean *and* has
    missing outcomes, or it is an ordinary complete-data ``ey_obs`` fit that shares the
    estimand name and nothing else: the complete-data mean fits no mechanism at all,
    while this one fits a response mechanism and no treatment law.  Testing the estimand
    alone made a complete-outcome fit refuse the missingness tilt for the natural-course
    reason rather than for the true one, which is that it has no observation mechanism.

    Parameters
    ----------
    result : object
        A fitted result. Read through :func:`getattr` so this module keeps importing
        only :mod:`cleverly.exceptions` and stays reachable from every caller.

    Returns
    -------
    bool
        True when every structured parameter is the natural-course mean and the fit
        declared an observation mask.
    """
    data = getattr(result, "data", None)
    if not getattr(data, "has_missing_outcome", False):
        return False
    keys = getattr(result, "parameter_keys", {})
    if keys:
        return all(getattr(key, "estimand", None) == NATURAL_COURSE_TARGET for key in keys.values())
    estimates = getattr(result, "estimates", {})
    return bool(estimates) and set(estimates) == {NATURAL_COURSE_TARGET}


#: The attributable estimands whose missing-outcome construction remains open, in report order.
_ORDERED = ("par", "paf")

#: The estimands whose functional reads the natural-course mean ``E[Y]``.
POPULATION_INTERVENTION_TARGETS = frozenset(_ORDERED)

#: The roadmap row that tracks each one.
_ROADMAP_ROW = {"par": "RM8", "paf": "RM8"}

_ROADMAP_SENTENCE: dict[tuple[str, ...], str] = {
    ("RM8",): "docs/roadmap.md RM8 tracks this identification boundary."
}


def _name_phrase(refused: tuple[str, ...]) -> str:
    """Join the refused estimand names the way a sentence reads them."""
    if len(refused) == 1:
        return refused[0]
    return f"{', '.join(refused[:-1])} and {refused[-1]}"


def population_intervention_refusal(
    targets: Iterable[str],
    *,
    declaration: str,
    subject: str | None = None,
) -> CapabilityError:
    """Build the refusal a population-intervention estimand earns on a missing outcome.

    Parameters
    ----------
    targets : iterable of str
        The estimand names the caller is refusing. Names outside
        :data:`POPULATION_INTERVENTION_TARGETS` are ignored, and an iterable naming none
        of them is read as both, which is the shared-context case.
    declaration : str
        How the caller's own API spells the missingness declaration, so the message
        names the keyword the reader wrote.
    subject : str or None
        The noun phrase the sentence opens with, for a caller that knows the typed
        estimand the user asked for. Defaults to the refused estimand names.

    Returns
    -------
    CapabilityError
        The error to raise. It states what the derivation is missing and cites the
        roadmap row that tracks each refused estimand.

    Examples
    --------
    >>> from cleverly.targets.population_intervention import population_intervention_refusal
    >>> error = population_intervention_refusal(("par",), declaration="delta=")
    >>> print(str(error).split(":")[0])
    par does not yet support delta=
    """
    named_targets = set(targets)
    refused = tuple(name for name in _ORDERED if name in named_targets)
    if not refused:
        refused = _ORDERED
    rows = tuple(sorted({_ROADMAP_ROW[name] for name in refused}))
    verb = "does" if len(refused) == 1 else "do"
    named = _name_phrase(refused) if subject is None else subject
    return CapabilityError(
        f"{named} {verb} not yet support {declaration}: under missingness at random "
        "the natural-course mean E[Y] needs an additional outcome/missingness score "
        "equation, and using complete cases would identify a different parameter. "
        f"{_ROADMAP_SENTENCE[rows]}"
    )
