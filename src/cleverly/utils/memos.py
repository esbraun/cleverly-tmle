"""What a persisted object may not carry across a pickle: a value it derived.

A :func:`~functools.cached_property` stores its answer in the instance dictionary, and
the default pickle protocol writes that dictionary out and reads it back verbatim.  So
every memo a caller warms before :func:`cleverly.save` would become part of the artifact,
and the loaded object would then answer with a stored conclusion rather than with the
conclusion it reaches from the stored records.

Two classes of object hit this.  A result carries its assessment facades and its own
score verdict in ``__dict__``, and ``save`` pickles the result whole.  A facade carries
its capability map and its E-value selections the same way.  Both restate what the
package concludes from artifacts the file already holds, so recomputing costs one pass
and persisting pins an answer nothing can check afterwards.

:func:`without_memos` is the one filter both use, applied in ``__getstate__`` before the
artifact is written.

The filter reads the owning class rather than a list of names, because a list of names
goes stale the day a fourth ``cached_property`` is added and the artifact written that
day is already wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any

__all__ = ["without_memos"]


def without_memos(owner: type, state: Mapping[str, Any]) -> dict[str, Any]:
    """Instance state without the entries a ``cached_property`` on ``owner`` owns.

    Parameters
    ----------
    owner : type
        The class whose ``cached_property`` descriptors name the derived entries.  Pass
        ``type(self)``, so that a subclass's own memos are dropped too.
    state : mapping of str to Any
        Instance state, on the way into or out of a pickle.

    Returns
    -------
    dict of str to Any
        The same state, without any entry a ``cached_property`` computed.
    """
    return {
        name: value
        for name, value in state.items()
        if not isinstance(getattr(owner, name, None), cached_property)
    }
