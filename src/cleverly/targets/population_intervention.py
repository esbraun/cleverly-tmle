"""The natural-course mean, the attributable estimands, and the predicates that read them.

``par`` and ``paf`` contain both the natural-course mean and a reference-intervention
mean.  With missing outcomes the estimator stacks the shipped natural-course fluctuation
and the shipped ``mean`` fluctuation on the same rows.  ``ey_obs`` beside any arm target
is the same stack.  The estimator spells that composition once, before any nuisance
exists, as ``_is_joint_natural_course``.

The names, the post-fit predicates and the two refusals that name the natural-course mean
live here because three modules import them -- :mod:`cleverly.assessment`,
:mod:`cleverly.sensitivity.positivity` and :mod:`cleverly.sensitivity.missingness` --
and they sit in three different subpackages.  This module imports nothing from the
package, so each one reaches it at module scope.

Two predicates read a finished result.  :func:`is_natural_course_fit` is the scalar
missing-outcome natural-course fit, which fits no treatment mechanism.
:func:`reads_natural_course_mean` is any missing-outcome fit that reports a target read
from the natural-course mean, scalar or joint.  The estimator asks ``_is_natural_course``
before any nuisance exists, and the nuisance diagnostics ask
:attr:`~cleverly.estimators._nuisance.NuisanceEstimates.fits_treatment` after the fit.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NATURAL_COURSE_SUPPORT_REFUSAL",
    "NATURAL_COURSE_TARGET",
    "NATURAL_COURSE_TILT_REFUSAL",
    "POPULATION_INTERVENTION_TARGETS",
    "is_natural_course_fit",
    "natural_course_names",
    "natural_course_tilt_refusal",
    "reads_natural_course_mean",
]

#: The natural-course mean's estimand name, for the modules that test for it rather
#: than merely mention it.  Other spellings of the literal remain, in registries and in
#: messages; this constant exists so that a *predicate* never respells it.
NATURAL_COURSE_TARGET = "ey_obs"

NATURAL_COURSE_TILT_REFUSAL = (
    "the implemented missingness tilt is arm-specific; NaturalCourseMean needs a "
    "natural-course sensitivity parameter, which is not implemented"
)

#: Why the arm-propensity support report is unavailable, not merely inapplicable: a
#: response-only overlap report could be written and has not been.
#: ``docs/architecture-invariants.md`` separates the two words, and the capability row
#: this sentence fills carries ``UNAVAILABLE``.
NATURAL_COURSE_SUPPORT_REFUSAL = (
    "NaturalCourseMean with missing outcomes fits no treatment propensity, and a "
    "response-only support report is not implemented; inspect the missingness row from "
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


#: The attributable estimands, whose functional reads the natural-course mean ``E[Y]``.
POPULATION_INTERVENTION_TARGETS = frozenset({"par", "paf"})


#: The estimand stems whose functional reads the natural-course mean.
NATURAL_COURSE_READERS = frozenset({NATURAL_COURSE_TARGET, *POPULATION_INTERVENTION_TARGETS})


def natural_course_names(names: Any) -> list[str]:
    """The names in ``names`` whose functional reads the natural-course mean, sorted.

    Parameters
    ----------
    names : iterable of str
        Reported parameter names, such as ``"par[low]"``.

    Returns
    -------
    list of str
        Each name whose stem, the text before ``[``, is ``ey_obs``, ``par`` or ``paf``.
    """
    return sorted(name for name in names if name.split("[", 1)[0] in NATURAL_COURSE_READERS)


def reads_natural_course_mean(result: Any) -> bool:
    """Whether a missing-outcome ``result`` reports a target read from the natural course.

    True for the scalar natural-course fit and for a joint fit that reports ``ey_obs``,
    ``par`` or ``paf`` beside arm targets.  The post-fit rules that a natural-course
    mean defeats read this one predicate.

    Parameters
    ----------
    result : object
        A fitted result. Read through :func:`getattr` so this module keeps importing
        nothing from the package.

    Returns
    -------
    bool
        True when the fit declared an observation mask with missing outcomes and some
        reported parameter is ``ey_obs``, ``par`` or ``paf``.
    """
    data = getattr(result, "data", None)
    if not getattr(data, "has_missing_outcome", False):
        return False
    keys = getattr(result, "parameter_keys", {})
    if keys:
        return any(
            getattr(key, "estimand", None) in NATURAL_COURSE_READERS for key in keys.values()
        )
    return bool(natural_course_names(getattr(result, "estimates", {})))


def natural_course_tilt_refusal(names: Any) -> str:
    """The sentence a tilt request for a natural-course target raises on a joint fit.

    Parameters
    ----------
    names : iterable of str
        The requested names that read the natural-course mean.

    Returns
    -------
    str
        The refusal, which names the targets and the arm-mean remedy.

    Examples
    --------
    >>> from cleverly.targets.population_intervention import natural_course_tilt_refusal
    >>> print(natural_course_tilt_refusal(["par"]).split(";")[0])
    the implemented missingness tilt is arm-specific
    """
    return (
        "the implemented missingness tilt is arm-specific; "
        f"{sorted(names)} read the natural-course mean, which needs a natural-course "
        "sensitivity parameter that is not implemented. Request the arm means by name "
        "with estimands="
    )
