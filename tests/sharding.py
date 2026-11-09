"""Split one pytest run across several CI runners.

``CLEVERLY_TEST_SHARD=i/N`` keeps the ``i``-th of ``N`` shards and deselects the rest. A shard
is a set of *scheduling units*: an ``xdist_group`` stays whole, because the group exists to put
its tests on one worker, and a test without a group is a unit of its own. A unit goes to the
shard its CRC-32 names, so the assignment depends on the unit alone: it does not move when an
unrelated test is added, and the ``N`` shards between them run every test exactly once.
"""

from __future__ import annotations

import os
import zlib

__all__ = ["SHARD_ENV", "parse", "shard_of"]

SHARD_ENV = "CLEVERLY_TEST_SHARD"


def shard_of(unit: str, count: int) -> int:
    """The 1-based shard that runs ``unit``, out of ``count`` shards."""
    return zlib.crc32(unit.encode()) % count + 1


def parse(value: str | None = None) -> tuple[int, int] | None:
    """Read ``i/N`` from ``value`` (default: the environment), or None when unset."""
    text = (os.environ.get(SHARD_ENV, "") if value is None else value).strip()
    if not text:
        return None
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError as error:
        raise ValueError(f"{SHARD_ENV}={text!r} is not of the form i/N") from error
    if not 1 <= index <= count:
        raise ValueError(f"{SHARD_ENV}={text!r} names no shard of {count}")
    return index, count
