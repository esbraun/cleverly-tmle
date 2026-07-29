"""Exception and warning types raised by cleverly."""

from __future__ import annotations

__all__ = [
    "CleverlyError",
    "ConvergenceWarning",
    "DataError",
    "NotFittedError",
    "PositivityWarning",
    "WeightingWarning",
]


class CleverlyError(Exception):
    """Base class for every error raised by cleverly."""


class DataError(CleverlyError, ValueError):
    """The supplied data violates an assumption the estimator relies on."""


class NotFittedError(CleverlyError, RuntimeError):
    """A result was requested from an estimator that has not been fitted."""


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
