"""One declaration of what a user-supplied function is, and the refusals it drives.

Some functions that a user passes are treated as fixed by the reported influence curve. A
callable can close over any estimate, and no code can inspect a closure. So the status of such
a function is a declaration with three states:

- ``"known"`` is a fixed function, chosen without reading the data. It is the one state that a
  fit accepts.
- ``"estimated"`` is a function computed from the analysis sample. If the intended target
  indexes that function by the population law, its influence curve can need an additional
  pathwise derivative. Inference for the realized learned-function target instead needs
  conditions that this API does not establish. The package refuses either use here.
- ``None`` is the default, and it means undeclared. The package refuses it for a supplied
  function, so an object restored from before its declaration field existed refuses every
  recomputation. A user can accept ``None`` when no function is supplied: an MSM with
  ``weights=None`` has uniform weights, which are known.

The declaration is a three-state ``Literal`` and not a bool, because an undeclared function
has to refuse and a bool default would declare it silently. :class:`FunctionDeclaration`
holds the check and the texts of the refusals for one field, so each user refuses the same way.
The MSM projection weight (roadmap row RM13) is the first user. The stochastic regime
density (roadmap row RM25) is the second, and the MSM design (roadmap row RM27) is the third.
RM28 adds two users: the rule declaration that :class:`~cleverly.interventions.Rule` and
:class:`~cleverly.longitudinal.DynamicRegimen` share, and the density declaration of a
user-written :class:`~cleverly.interventions.Intervention`.

A restored result can hold a function whose declaration this version refuses.
:func:`declaration_status` turns the refusal of such a function into the status
``"undeclared_function_plugin"``, and the point and the longitudinal estimators both call it.

A leaf module. It imports the standard library, :mod:`cleverly.exceptions` and the leaf
:mod:`cleverly._inference_status` only, so :mod:`cleverly.msm` and
:mod:`cleverly.interventions` can both import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ._inference_status import InferenceStatus
from .exceptions import CapabilityError, DataError

__all__ = ["FunctionDeclaration", "FunctionKind", "declaration_status"]

#: What a declaration says about a user-supplied function. ``"known"`` is a fixed function,
#: chosen independently of the analysis sample. ``"estimated"`` is one computed from that
#: sample, which is refused. A declaration field holds one of these or ``None``, which means
#: undeclared.
FunctionKind = Literal["known", "estimated"]


@dataclass(frozen=True)
class FunctionDeclaration:
    """One declaration field, and the texts of the refusals it drives.

    Parameters
    ----------
    field : str
        The name of the declaration field, such as ``"weights_kind"``. The refusal of an
        unknown value names it.
    meaning : str
        The sentence that the refusal of an unknown value appends. It says what the field
        declares.
    undeclared : str
        The text of the :class:`~cleverly.exceptions.CapabilityError` for ``None``.
    estimated : str
        The text of the :class:`~cleverly.exceptions.CapabilityError` for ``"estimated"``.
    """

    field: str
    meaning: str
    undeclared: str
    estimated: str

    def check(self, kind: object) -> None:
        """Raise unless ``kind`` is ``"known"``, ``"estimated"``, or ``None``.

        A user of the declaration calls this apart from :meth:`refuse` when a check of its
        own comes between the two, as the MSM weight check does for a weight that is not
        callable.

        Parameters
        ----------
        kind : object
            The value of the declaration field at run time. A restored or modified object
            can hold any value, so the type is not narrowed.

        Raises
        ------
        DataError
            If ``kind`` is not one of the three states. A value that is not ``None`` and
            not a ``str`` raises, so an array or ``pandas.NA`` gets this error and not the
            error of its truth value.
        """
        # The ``isinstance`` test comes first, so no comparison can return an array or
        # ``pandas.NA``. A ``str`` subclass, such as ``numpy.str_``, passes it.
        if kind is not None and not (isinstance(kind, str) and kind in ("known", "estimated")):
            raise DataError(
                f"{self.field} must be 'known', 'estimated' or None; got {kind!r}. {self.meaning}"
            )

    def refuse(self, kind: object) -> None:
        """Raise unless ``kind`` declares a known function.

        Parameters
        ----------
        kind : object
            The value of the declaration field at run time.

        Raises
        ------
        DataError
            If ``kind`` is not one of the three states, from :meth:`check`.
        CapabilityError
            If ``kind`` is ``None``, with the ``undeclared`` text, or ``"estimated"``, with
            the ``estimated`` text.
        """
        self.check(kind)
        if kind is None:
            raise CapabilityError(self.undeclared)
        if kind == "estimated":
            raise CapabilityError(self.estimated)


def declaration_status(refuse: Callable[[], None]) -> InferenceStatus:
    """The inference status of a configuration, from the refusal of its declarations.

    The status predicate of roadmap row RM28. ``refuse`` is the declaration check that
    every fit runs before any learner, so a live fit never reaches the refusal here. Only
    a restored or modified configuration does, and its saved estimates then take
    ``"undeclared_function_plugin"``. ``TMLE._declared_function_status`` passes the point
    check, and the longitudinal ``_declared_regimen_status`` passes the regimen check.

    Parameters
    ----------
    refuse : callable
        Called with no argument. It raises if a function of the configuration is not
        declared known.

    Returns
    -------
    str
        ``"undeclared_function_plugin"`` when ``refuse`` raises
        :class:`~cleverly.exceptions.CapabilityError` or
        :class:`~cleverly.exceptions.DataError`, and ``"influence_curve"`` otherwise.
    """
    try:
        refuse()
    except (CapabilityError, DataError):
        return "undeclared_function_plugin"
    return "influence_curve"
