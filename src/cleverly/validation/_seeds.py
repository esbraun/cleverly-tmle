"""Distinct sample seeds with stable prefixes and intentional stream sharing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

_SEED_SPACE = 2**32
_COLLISION_TAG = 0x434F5645


def sample_seed_streams(budgets: Mapping[int, int]) -> dict[int, tuple[int, ...]]:
    """Plan distinct uint32 seeds for each root, retaining every unoccupied candidate.

    Equal roots name one intentionally shared stream.  Assignment proceeds by replication
    index, then root.  Extending a stream preserves assignments before its former budget.
    After that index, a new draw can occupy a longer stream's candidate and change it.
    A collision receives a deterministic domain-separated retry, rather than another draw
    from the same stream.  Distinct inputs to a PRNG do not prove independence.
    """
    if any(count < 0 for count in budgets.values()):
        raise ValueError("sample seed budgets must be nonnegative")
    if sum(budgets.values()) > _SEED_SPACE:
        raise ValueError("sample seed budgets exceed the uint32 seed space")
    roots = sorted(budgets)
    candidates = {
        root: np.random.SeedSequence(root).generate_state(budgets[root]) for root in roots
    }
    streams: dict[int, list[int]] = {root: [] for root in roots}
    occupied: set[int] = set()
    for replicate in range(max(budgets.values(), default=0)):
        for root in roots:
            if replicate >= budgets[root]:
                continue
            candidate = int(candidates[root][replicate])
            retry = 0
            while candidate in occupied:
                candidate = int(
                    np.random.SeedSequence([root, replicate, _COLLISION_TAG, retry]).generate_state(
                        1
                    )[0]
                )
                retry += 1
            occupied.add(candidate)
            streams[root].append(candidate)
    return {root: tuple(stream) for root, stream in streams.items()}


def validate_sample_seeds(seeds: Sequence[int], count: int) -> tuple[int, ...]:
    """Freeze explicit sample seeds and refuse invalid or repeated inputs before a fit."""
    if len(seeds) != count:
        raise ValueError(f"sample_seeds has {len(seeds)} entries, not n_replicates={count}")
    if any(
        isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < _SEED_SPACE
        for seed in seeds
    ):
        raise ValueError("sample_seeds must contain uint32 integers, without booleans")
    frozen = tuple(int(seed) for seed in seeds)
    if len(set(frozen)) != count:
        raise ValueError("sample_seeds must be distinct within a study")
    return frozen
