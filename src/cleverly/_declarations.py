"""One declaration of what a user-supplied function is, and the refusals it drives.

Some functions that a user passes are treated as fixed by the reported influence curve. A
callable can close over any estimate, and no code can inspect a closure. So the status of such
a function is a declaration with three states:

- ``"known"`` is a fixed function, chosen without reading the data. It is the one state that a
  fit accepts.
- ``"estimated"`` is a function computed from the sample. The target is then a functional of
  :math:`P` through the function, and the reported influence curve omits that pathwise
  derivative, so the package refuses it.
- ``None`` is the default, and it means undeclared. The package refuses it, so an object
  restored from before its declaration field existed refuses every recomputation.

The declaration is a three-state ``Literal`` and not a bool, because an undeclared function
has to refuse and a bool default would declare it silently. :class:`FunctionDeclaration`
holds the check and the texts of the refusals for one field, so each user refuses the same way.
The MSM projection weight (roadmap row RM13) is the first user. The stochastic regime
density (roadmap row RM25) is the second, and RM27 can add the MSM design as a third.

A leaf module. It imports the standard library and :mod:`cleverly.exceptions` only, so
:mod:`cleverly.msm` and :mod:`cleverly.interventions` can both import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .exceptions import CapabilityError, DataError

__all__ = ["FunctionDeclaration", "FunctionKind"]

#: What a declaration says about a user-supplied function. ``"known"`` is a fixed function,
#: chosen without reading the data. ``"estimated"`` is one computed from the sample, which is
#: refused. A declaration field holds one of these or ``None``, which means undeclared.
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
            If ``kind`` is not one of the three states.
        """
        if kind not in (None, "known", "estimated"):
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
