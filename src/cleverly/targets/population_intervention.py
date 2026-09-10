"""The population-intervention estimands, and the one refusal the three of them share.

``ey_obs``, ``par`` and ``paf`` are the estimands whose functional contains the
natural-course mean :math:`E[Y]`.  Under missingness at random that mean is not the
empirical mean of the observed rows, so all three are refused until the extra
outcome/missingness score equation is derived.

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
"""

from __future__ import annotations

from collections.abc import Iterable

from ..exceptions import CapabilityError

__all__ = [
    "POPULATION_INTERVENTION_TARGETS",
    "population_intervention_refusal",
]

#: The estimands whose functional reads the natural-course mean ``E[Y]``, in report order.
_ORDERED = ("ey_obs", "par", "paf")

#: The estimands whose functional reads the natural-course mean ``E[Y]``.
POPULATION_INTERVENTION_TARGETS = frozenset(_ORDERED)

#: The roadmap row that tracks each one.  RM8 depends on RM7, and both are open.
_ROADMAP_ROW = {"ey_obs": "RM7", "par": "RM8", "paf": "RM8"}

_ROADMAP_SENTENCE = {
    ("RM7",): "docs/roadmap.md RM7 tracks this identification boundary.",
    ("RM8",): "docs/roadmap.md RM8 tracks this identification boundary.",
    ("RM7", "RM8"): (
        "docs/roadmap.md RM7 tracks this stop for the natural-course mean, and "
        "docs/roadmap.md RM8 tracks it for par and paf."
    ),
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
        of them is read as all three, which is the shared-context case.
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
