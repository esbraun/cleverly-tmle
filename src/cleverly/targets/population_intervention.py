"""The natural-course mean, the attributable estimands, and the predicates that read them.

``par`` and ``paf`` contain both the natural-course mean and a reference-intervention
mean.  With missing outcomes the estimator stacks the shipped natural-course fluctuation
and the shipped ``mean`` fluctuation on the same rows.  ``ey_obs`` beside any arm target
is the same stack.  The estimator spells that composition once, before any nuisance
exists, as ``_is_joint_natural_course``.

The names, the post-fit predicate and the support refusal live here because modules in
several subpackages import them.  This module imports nothing from the package but the
refusal record, so each one reaches it at module scope.

:func:`is_natural_course_fit` reads a finished result.  It is the scalar missing-outcome
natural-course fit, which fits no treatment mechanism.  The estimator asks
``_is_natural_course`` before any nuisance exists, and the nuisance diagnostics ask
:attr:`~cleverly.estimators._nuisance.NuisanceEstimates.fits_treatment` after the fit.
"""

from __future__ import annotations

from typing import Any

from .._refusal import Refusal, RefusalReason

__all__ = [
    "NATURAL_COURSE_SUPPORT_REFUSAL",
    "NATURAL_COURSE_TARGET",
    "POPULATION_INTERVENTION_TARGETS",
    "is_natural_course_fit",
]

#: The natural-course mean's estimand name, for the modules that test for it rather
#: than merely mention it.  Other spellings of the literal remain, in registries and in
#: messages; this constant exists so that a *predicate* never respells it.
NATURAL_COURSE_TARGET = "ey_obs"

#: Why the arm-propensity support report is unavailable, not merely inapplicable: a
#: response-only overlap report could be written and has not been.
#: ``docs/architecture-invariants.md`` separates the two words, and the capability row
#: this sentence fills carries ``UNAVAILABLE``.
NATURAL_COURSE_SUPPORT_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "NaturalCourseMean with missing outcomes fits no treatment propensity, and a "
    "response-only support report is not implemented.",
    alternative=(
        "Inspect the missingness row from diagnostics.nuisance_models() and the targeting "
        "score instead."
    ),
    roadmap="X63",
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


#: The attributable estimands, whose functional reads the natural-course mean ``E[Y]``.
POPULATION_INTERVENTION_TARGETS = frozenset({"par", "paf"})
