"""Per-replication seeds that do not move when the replication count changes.

Deriving seeds from ``SeedSequence(seed).generate_state(2 * replicates)`` -- one flat draw
sized by the total -- re-seeds every replication as soon as the total changes, so a short
run shares no sample with the published one and cannot be used to re-execute it.  Spawning
on ``(scenario, replicate)`` instead makes replication *k* of a scenario a fixed sample:
a two-replication probe redraws exactly the first two samples of the full study, which is
what lets a fast test refit committed replications and compare.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tests.studies.evidence.registry import StudyRecord


def replicate_seed(record: StudyRecord, scenario: str, replicate: int) -> int:
    """The seed replication ``replicate`` of ``scenario`` is drawn with."""
    scenarios = tuple(record.scenarios)
    if scenario not in scenarios:
        raise KeyError(f"{record.slug} has no scenario {scenario!r}")
    if replicate < 0:
        raise ValueError(f"replicate must be non-negative; got {replicate}")
    sequence = np.random.SeedSequence(
        entropy=record.seed, spawn_key=(scenarios.index(scenario), replicate)
    )
    return int(sequence.generate_state(1)[0])


def stream_seed(record: StudyRecord, *labels: str | int) -> int:
    """The seed a named resampling analysis of ``record`` draws its bootstrap indices with.

    Every bootstrap stream in the framework used to be ``record.seed + <offset> + <index>``.
    Two things go wrong with that.  Within a study the offsets are close enough to collide:
    :func:`~tests.studies.evidence.properties.rate` took a base seed and added the size index
    to it, while its two callers were themselves one apart, so ``empirical_sd`` and
    ``reported_se`` -- published side by side as separate evidence -- drew *bit-identical*
    resample index matrices for two of their three sizes.  Across studies it is worse: the
    three registered seeds are consecutive integers, so cell ``k`` of one study shares a
    stream with cell ``k-1`` of the next, and the published Monte Carlo errors of rows a
    reader compares are correlated by construction.

    Hashing a *label* instead fixes both.  ``SeedSequence`` mixes the study's entropy with the
    spawn key, so streams are disjoint whatever the study seeds happen to be, and naming the
    analysis rather than numbering it means adding a cell no longer moves every other cell's
    stream.  ``blake2b`` rather than :func:`hash`, which is salted per process.
    """
    joined = "\x1f".join(str(label) for label in labels)
    digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=8).digest()
    sequence = np.random.SeedSequence(
        entropy=record.seed, spawn_key=(int.from_bytes(digest, "big"),)
    )
    return int(sequence.generate_state(1)[0])
