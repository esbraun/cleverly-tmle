"""Exception and warning types raised by cleverly, and the refusals several modules share.

A refusal helper lives here rather than beside one of its callers because the callers sit
in three subpackages that do not import each other.  This module imports nothing, so any
of them can reach it.
"""

from __future__ import annotations

__all__ = [
    "WORKING_MECHANISM_ASSESSMENT_NOTE",
    "WORKING_MECHANISM_NOT_INFERENTIAL",
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
    "refuse_working_mechanism_inference",
    "repeats_refusal",
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


#: Why a selector-path collaborative estimate reports no interval and no p-value.  One text
#: for every raise :func:`refuse_working_mechanism_inference` makes (the estimate
#: accessors, ``simultaneous_bands``, ``variable_importance`` and
#: ``tipping_gamma(use_ci=True)``), the refusal paragraph of ``TMLEResult.summary``, the
#: E-value and ``tipping_gamma(use_ci=True)`` capability rows, and the tests that assert
#: the refusal states its cause, on the model of
#: :data:`cleverly.validation.nuisance.NUISANCE_SELECTION_MISSING`.
WORKING_MECHANISM_NOT_INFERENTIAL = (
    "the greedy, ordered and discrete collaborative paths report no confidence interval, "
    "no p-value and no standard error. The reported curve is the ordinary efficient "
    "influence curve at the candidate the search stopped at, and no result shows it is "
    "this estimator's influence curve when that working mechanism is not consistent for "
    "the treatment law. The point estimate and the selection path stand. The plug-in "
    "standard error of that curve remains as a diagnostic under plugin_std_error and "
    "plugin_interval. F18 in docs/roadmap.md reopens this when it supplies the "
    "estimator's influence curve."
)

#: The fact the nuisance report adds for the same fit.  ``NuisanceDiagnostics.inference_note``
#: returns it, and its summary and the assessment's nuisance-model item both print that.
#: Held beside the refusal so the report and the raise cannot drift apart, and asserted by
#: importing it.
WORKING_MECHANISM_ASSESSMENT_NOTE = (
    "the reported curve is a working-mechanism diagnostic: no confidence interval or "
    "p-value is available for this path, and F18 in the roadmap is the condition that "
    "reopens it"
)


def capitalize_first(clause: str) -> str:
    """Raise the first letter of a clause, and leave every other letter as written.

    The two constants above are clauses, so a report that prints one as a sentence has to
    raise its first letter. ``str.capitalize`` also lowers every later letter, and it
    printed "F18" as "f18".

    Parameters
    ----------
    clause : str
        The clause to open a sentence with.

    Returns
    -------
    str
        ``clause`` with its first character upper-cased.
    """
    return clause[:1].upper() + clause[1:]


def working_mechanism_refusal(operation: str) -> str:
    """The sentence that refuses an inferential operation on a selector-path estimate.

    One text for the raise in :func:`refuse_working_mechanism_inference` and for a
    capability row that declares the same refusal before the call, so the row and the
    raise cannot disagree.

    Parameters
    ----------
    operation : str
        The refused operation, named as the caller writes it.

    Returns
    -------
    str
        The operation, followed by :data:`WORKING_MECHANISM_NOT_INFERENTIAL`.
    """
    return f"{operation} is not defined here. {WORKING_MECHANISM_NOT_INFERENTIAL}"


def refuse_working_mechanism_inference(status: str, *, operation: str) -> None:
    """Refuse an inferential accessor on an estimate the package supplies no inference for.

    Parameters
    ----------
    status : str
        The estimate's declared
        :data:`~cleverly.inference.influence.InferenceStatus`. Only
        ``"influence_curve"`` passes.
    operation : str
        The refused accessor, named as the caller writes it.

    Raises
    ------
    CapabilityError
        When the estimate declares any status other than ``"influence_curve"``.
    """
    # Function-local, because :mod:`cleverly.inference.influence` imports this module.
    from .inference.influence import supplies_inference

    if not supplies_inference(status):
        raise CapabilityError(working_mechanism_refusal(operation))


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
    refusal = repeats_refusal(n_repeats, operation=operation, reason=reason)
    if refusal is not None:
        raise CapabilityError(refusal)


def repeats_refusal(n_repeats: int, *, operation: str, reason: str) -> str | None:
    """Return the refusal :func:`refuse_after_repeats` raises, or ``None``.

    A capability row declares its reason before any call, so a surface that reports rows
    needs the sentence without the raise.  One formatter serves both, so the declared
    reason and the raised one cannot drift apart.

    Parameters
    ----------
    n_repeats : int
        How many cross-fitting draws the report combines. One draw refuses nothing.
    operation : str
        The refused operation, named as the caller writes it.
    reason : str
        Why the median report cannot supply it, and what the caller can do instead.

    Returns
    -------
    str or None
        The refusal, or ``None`` when the report is one draw.
    """
    if n_repeats > 1:
        return f"{operation} is not defined for median-combined repeats. {reason}"
    return None
