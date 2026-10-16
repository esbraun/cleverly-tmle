"""Exception and warning types raised by cleverly, and the refusals several modules share.

A refusal helper lives here rather than beside one of its callers because the callers sit
in three subpackages that do not import each other.  This module imports only the leaves
:mod:`cleverly._inference_status` and :mod:`cleverly._refusal`, which import the standard
library alone, so any of them can reach it.

Every error carries a :class:`~cleverly._refusal.Refusal` record under ``error.refusal``. Its
``reason`` is a :class:`~cleverly._refusal.RefusalReason` code, and ``str(error)`` is always
``error.refusal.message()``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

from ._inference_status import status_record, supplies_inference
from ._refusal import NO_ALTERNATIVE, Refusal, RefusalReason

__all__ = [
    "NO_ALTERNATIVE",
    "CapabilityError",
    "CleverlyError",
    "ConvergenceWarning",
    "DataError",
    "DataWarning",
    "LongitudinalError",
    "MethodConfigurationError",
    "NotFittedError",
    "ParameterSpaceWarning",
    "PositivityWarning",
    "Refusal",
    "RefusalReason",
    "SparseInfluenceWarning",
    "VersionMismatchWarning",
    "WeightingWarning",
    "inference_refusal",
    "refuse_after_repeats",
    "refuse_inference",
    "repeats_refusal",
]


def _restore(cls: type[CleverlyError], refusal: Refusal, state: dict[str, Any]) -> CleverlyError:
    """Rebuild a pickled error without calling its subclass's constructor.

    Parameters
    ----------
    cls : type
        The error class.
    refusal : Refusal
        The record the error carried.
    state : dict
        The instance attributes the error carried.

    Returns
    -------
    CleverlyError
        An error of ``cls`` with the same record and attributes.
    """
    error = cls.__new__(cls)
    Exception.__init__(error, refusal.message())
    error.__dict__.update(state)
    error.refusal = refusal
    return error


class CleverlyError(Exception):
    """Base class for every error raised by cleverly.

    Every error carries a :class:`~cleverly._refusal.Refusal` under ``refusal``. A subclass
    declares the code its errors take by default in ``default_reason``. A subclass with no
    default requires ``reason=`` at every raise.

    Parameters
    ----------
    condition : str or None
        The sentences that state the failed condition. Leave it out when ``refusal`` is given.
    reason : RefusalReason or None
        The reason code. ``None`` takes the class's ``default_reason``.
    alternative : str or None
        The sentence that names a working call.
    roadmap : str or None
        The ``docs/roadmap.md`` id that tracks a missing capability.
    refusal : Refusal or None
        A complete record, as a capability row stores it.

    Raises
    ------
    TypeError
        When ``refusal`` is given together with any other argument, or when no reason code is
        given and the class declares no default.
    """

    #: The code an error of this class takes when a raise names none.
    default_reason: ClassVar[RefusalReason | None] = RefusalReason.INVALID_DECLARATION

    refusal: Refusal

    def __init__(
        self,
        condition: str | None = None,
        /,
        *,
        reason: RefusalReason | None = None,
        alternative: str | None = None,
        roadmap: str | None = None,
        refusal: Refusal | None = None,
    ) -> None:
        if refusal is not None:
            if (condition, reason, alternative, roadmap) != (None, None, None, None):
                raise TypeError("pass either a complete refusal or its parts, not both")
        else:
            code = reason if reason is not None else type(self).default_reason
            if code is None:
                raise TypeError(f"{type(self).__name__} needs reason=, a RefusalReason")
            refusal = Refusal(code, condition or "", alternative=alternative, roadmap=roadmap)
        self.refusal = refusal
        super().__init__(refusal.message())

    @classmethod
    def from_refusal(cls, refusal: Refusal) -> Self:
        """Build the error a capability row's record declares.

        Parameters
        ----------
        refusal : Refusal
            The record.

        Returns
        -------
        CleverlyError
            An error of this class that carries ``refusal``.
        """
        return cls(refusal=refusal)

    def __reduce__(self) -> tuple[Any, ...]:
        state = {key: value for key, value in self.__dict__.items() if key != "refusal"}
        return (_restore, (type(self), self.refusal, state))


class CapabilityError(CleverlyError, ValueError):  # numpydoc ignore=PR01
    """A request the package refuses for a reason other than the data alone.

    It has no default code: every raise names the
    :class:`~cleverly._refusal.RefusalReason` that applies.
    """

    default_reason: ClassVar[RefusalReason | None] = None


class LongitudinalError(CleverlyError):  # numpydoc ignore=PR01
    """A longitudinal fit cannot proceed on the data or regimen it was given.

    Lives here rather than beside the recursion that raises it, because it is the one
    error a caller has to catch by name -- a regimen no unit in the sample followed --
    and every other error type in the library is looked up in this module.
    """


class DataError(CleverlyError, ValueError):  # numpydoc ignore=PR01
    """The supplied data violates an assumption the estimator relies on."""


class MethodConfigurationError(CleverlyError, ValueError):  # numpydoc ignore=PR01
    """An estimation-method declaration is invalid or cannot take effect."""


class NotFittedError(CleverlyError, RuntimeError):  # numpydoc ignore=PR01
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


class VersionMismatchWarning(UserWarning):
    """A saved result was written by a different cleverly version than the one loading it.

    Alpha releases keep no compatibility for saved objects. :func:`cleverly.load` and
    :func:`cleverly.estimators.serialize.loads` emit this warning and then load the object
    as saved, with no migration. Fit the analysis again with the loading version.
    """


class WeightingWarning(UserWarning):
    """The supplied observation weights may not mean what the estimator assumes.

    Weights are read as *probability* weights: a tilt of the population, under which
    the sample size is still the number of rows.  Counts of identical units are a
    different experiment.  See :mod:`cleverly.data.weighting`.
    """


class ParameterSpaceWarning(UserWarning):
    """An estimate left the range of the parameter it estimates.

    A sequentially doubly robust estimate is an average of transformed outcomes, and a
    transform can leave the outcome range (Díaz, Williams, Hoffman and Schenck 2023, Section
    5.3).  The estimate is reported as computed, with no truncation.
    """


class PositivityWarning(UserWarning):
    """Estimated treatment (or missingness) probabilities are near 0 or 1.

    Practical positivity violations inflate the influence curve and can make
    the influence-curve based confidence intervals anti-conservative. See
    :mod:`cleverly.sensitivity.positivity` for diagnostics.
    """


class SparseInfluenceWarning(UserWarning):
    """A few rows carry the influence curve, so the Wald interval may under-cover.

    The warning fires when :math:`\\sqrt{(\\hat\\kappa - 1)/n} > 0.125`, with
    :math:`\\hat\\kappa` the raw kurtosis of the influence curve. That is close to a Kish
    effective sample size :math:`n / \\hat\\kappa` below 64. The threshold was set after
    the X39 pilots were read. ``result.diagnostics.influence()`` reports the measures per
    estimate.
    """


def capitalize_first(clause: str) -> str:
    """Raise the first letter of a clause, and leave every other letter as written.

    A status record's assessment note is a clause, so a report that prints one as a
    sentence has to raise its first letter. ``str.capitalize`` also lowers every later
    letter, and it printed "F18" as "f18".

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


def inference_refusal(operation: str, status: str) -> Refusal:
    """The refusal of an inferential operation at a non-inferential status.

    One record for the raise in :func:`refuse_inference` and for a capability row that
    declares the same refusal before the call, so the row and the raise cannot disagree.

    Parameters
    ----------
    operation : str
        The refused operation, named as the caller writes it.
    status : str
        The declared :data:`~cleverly.inference.influence.InferenceStatus`, other than
        ``"influence_curve"``.

    Returns
    -------
    Refusal
        A record with the code the status declares (most statuses declare
        :attr:`~cleverly._refusal.RefusalReason.INFERENCE_GAP`): the operation and the
        reason :data:`~cleverly._inference_status.NON_INFERENTIAL` records for ``status``,
        then that record's alternative.
    """
    record = status_record(status)
    return Refusal(
        record.code,
        f"{operation} is not defined here. {record.reason}",
        alternative=record.alternative,
        roadmap=record.reopened_by,
    )


def refuse_inference(status: str, *, operation: str) -> None:
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
        When the estimate declares any status other than ``"influence_curve"``. The
        error carries :func:`inference_refusal`.
    """
    if not supplies_inference(status):
        raise CapabilityError.from_refusal(inference_refusal(operation, status))


def refuse_after_repeats(
    n_repeats: int,
    *,
    operation: str,
    text: str,
    reason: RefusalReason,
    alternative: str,
) -> None:
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
    text : str
        Why the median report cannot supply it.
    reason : RefusalReason
        The code of the refusal.
    alternative : str
        The sentence that names what the caller can do instead.

    Raises
    ------
    CapabilityError
        When the report combines more than one draw.
    """
    refusal = repeats_refusal(
        n_repeats, operation=operation, text=text, reason=reason, alternative=alternative
    )
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)


def repeats_refusal(
    n_repeats: int,
    *,
    operation: str,
    text: str,
    reason: RefusalReason,
    alternative: str,
    roadmap: str | None = None,
) -> Refusal | None:
    """Return the refusal :func:`refuse_after_repeats` raises, or ``None``.

    A capability row declares its refusal before any call, so a surface that reports rows
    needs the record without the raise.  One builder serves both, so the declared refusal
    and the raised one cannot drift apart.

    Parameters
    ----------
    n_repeats : int
        How many cross-fitting draws the report combines. One draw refuses nothing.
    operation : str
        The refused operation, named as the caller writes it.
    text : str
        Why the median report cannot supply it.
    reason : RefusalReason
        The code of the refusal.
    alternative : str
        The sentence that names what the caller can do instead.
    roadmap : str or None
        The id of the ``docs/roadmap.md`` item that tracks the gap.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when the report is one draw.
    """
    if n_repeats > 1:
        return Refusal(
            reason,
            f"{operation} is not defined for median-combined repeats. {text}",
            alternative=alternative,
            roadmap=roadmap,
        )
    return None
