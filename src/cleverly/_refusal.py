"""The reason codes of a refusal, and the one record a refusal is built from.

Every :class:`~cleverly.exceptions.CleverlyError` carries a :class:`Refusal`. The record holds
the reason code, the condition that failed, the alternative a caller can use, and the roadmap
item that tracks a missing capability. A capability row stores the same record before any call,
so the row and the raise state one reason in one text.

A leaf module. It imports the standard library only, as :mod:`cleverly._inference_status` does,
so :mod:`cleverly.exceptions` can import it at module level without an import cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "NO_ALTERNATIVE",
    "Refusal",
    "RefusalReason",
]


class RefusalReason(StrEnum):  # numpydoc ignore=PR01,PR02
    """Why the package refuses a request.

    The code says where the problem is, and so what a caller can do about it.

    Attributes
    ----------
    INVALID_DECLARATION
        The requested analysis cannot run on the declared sample or design. The message states
        the failed condition.
    DIFFERENT_ESTIMAND
        The requested option changes the causal question. The message names both questions.
    IMPLEMENTATION_GAP
        A valid method has no implementation here yet. The message names the available
        alternatives and the roadmap item that tracks the gap.
    INFERENCE_GAP
        The estimate exists, but the requested uncertainty calculation is incomplete. The
        message offers the supported inference.
    API_REDIRECTION
        The feature exists through another argument or object. The message points to a working
        call. This code does not mark a missing capability.
    """

    INVALID_DECLARATION = "invalid_declaration"
    DIFFERENT_ESTIMAND = "different_estimand"
    IMPLEMENTATION_GAP = "implementation_gap"
    INFERENCE_GAP = "inference_gap"
    API_REDIRECTION = "api_redirection"


#: The alternative of a refusal for which no call in this package answers the request. A
#: refusal with a code other than :attr:`RefusalReason.INVALID_DECLARATION` must name an
#: alternative, and this value states that none exists rather than leaving the field empty.
NO_ALTERNATIVE: Final[str] = "No call in this package answers this request."


@dataclass(frozen=True)
class Refusal:
    """One refusal: its reason code, the failed condition, and what to do instead.

    Parameters
    ----------
    reason : RefusalReason
        Where the problem is.
    condition : str
        The sentences that state the failed condition. They end with a full stop. A condition
        that a raise splices into a longer message can open with a lowercase word.
    alternative : str or None
        The sentence that names a working call, or :data:`NO_ALTERNATIVE`. Required for every
        code except :attr:`RefusalReason.INVALID_DECLARATION`.
    roadmap : str or None
        The id of the ``docs/roadmap.md`` item that tracks a missing capability, such as
        ``"F21"``. An :attr:`RefusalReason.IMPLEMENTATION_GAP` refusal always names one.

    Raises
    ------
    TypeError
        When ``reason`` is not a :class:`RefusalReason`, when ``condition`` is empty, or when a
        code that needs an alternative has none.
    """

    reason: RefusalReason
    condition: str
    alternative: str | None = None
    roadmap: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, RefusalReason):
            raise TypeError(f"a refusal needs a RefusalReason; got {self.reason!r}")
        if not self.condition.strip():
            raise TypeError("a refusal needs the condition that failed")
        if self.reason is not RefusalReason.INVALID_DECLARATION and not self.alternative:
            raise TypeError(
                f"a {self.reason.value} refusal names an alternative, or NO_ALTERNATIVE"
            )

    def message(self) -> str:
        """The text a raise prints and a capability row quotes.

        Returns
        -------
        str
            :attr:`condition`, then a sentence that names :attr:`roadmap` when the condition
            does not already name it, then :attr:`alternative`.
        """
        parts = [self.condition.strip()]
        if self.roadmap is not None and not re.search(
            rf"\b{re.escape(self.roadmap)}\b", self.condition
        ):
            parts.append(f"{self.roadmap} in docs/roadmap.md tracks this.")
        if self.alternative:
            parts.append(self.alternative.strip())
        return " ".join(parts)
