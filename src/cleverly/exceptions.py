"""Exception and warning types raised by cleverly, and the refusals several modules share.

A refusal helper lives here rather than beside one of its callers because the callers sit
in three subpackages that do not import each other.  This module imports nothing, so any
of them can reach it.
"""

from __future__ import annotations

__all__ = [
    "CapabilityError",
    "CleverlyError",
    "ConvergenceWarning",
    "DataError",
    "DataWarning",
    "LongitudinalError",
    "MethodConfigurationError",
    "NotFittedError",
    "PositivityWarning",
    "WeightingWarning",
    "refuse_after_repeats",
]


class CleverlyError(Exception):
    """Base class for every error raised by cleverly."""


class CapabilityError(CleverlyError, ValueError):
    """A well-posed estimand/design/method composition is not implemented."""


class LongitudinalError(CleverlyError):
    """A longitudinal fit cannot proceed on the data or regimen it was given.

    Lives here rather than beside the recursion that raises it, because it is the one
    error a caller has to catch by name -- a regimen no unit in the sample followed --
    and every other error type in the library is looked up in this module.
    """


class DataError(CleverlyError, ValueError):
    """The supplied data violates an assumption the estimator relies on."""


class MethodConfigurationError(CleverlyError, ValueError):
    """An estimation-method declaration is invalid or cannot take effect."""


class NotFittedError(CleverlyError, RuntimeError):
    """A result was requested from an estimator that has not been fitted."""


class DataWarning(UserWarning):
    """The supplied data is usable, but is probably not what the caller meant.

    Distinguished from :class:`DataError` on purpose: an error says the estimator cannot
    proceed, and a warning says it can but the reading of the data may be the wrong one.
    Declaring a six-level dose continuous is the motivating case -- estimable, and usually
    a mistake.
    """


class ConvergenceWarning(UserWarning):
    """The targeting step stopped before reaching the requested tolerance."""


class WeightingWarning(UserWarning):
    """The supplied observation weights may not mean what the estimator assumes.

    Weights are read as *probability* weights: a tilt of the population, under which
    the sample size is still the number of rows.  Counts of identical units are a
    different experiment.  See :mod:`cleverly.data.weighting`.
    """


class PositivityWarning(UserWarning):
    """Estimated treatment (or missingness) probabilities are near 0 or 1.

    Practical positivity violations inflate the influence curve and can make
    the influence-curve based confidence intervals anti-conservative. See
    :mod:`cleverly.sensitivity.positivity` for diagnostics.
    """


def refuse_after_repeats(n_repeats: int, *, operation: str, reason: str) -> None:
    """Refuse an operation that median aggregation over fold draws leaves undefined.

    A repeated cross-fitted report is a coordinatewise median, so it is not the report of
    any one draw.  Several operations need a single draw's joint object, and each one
    refuses here rather than returning a convenient approximation to a different quantity.

    Parameters
    ----------
    n_repeats : int
        How many cross-fitting draws the report combines. One draw refuses nothing.
    operation : str
        The refused operation, named as the caller writes it.
    reason : str
        Why the median report cannot supply it, and what the caller can do instead.

    Raises
    ------
    CapabilityError
        When the report combines more than one draw.
    """
    if n_repeats > 1:
        raise CapabilityError(f"{operation} is not defined for median-combined repeats. {reason}")
