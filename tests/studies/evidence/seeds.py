"""Per-replication seeds that do not move when the replication count changes.

Deriving seeds from ``SeedSequence(seed).generate_state(2 * replicates)`` -- one flat draw
sized by the total -- re-seeds every replication as soon as the total changes, so a short
run shares no sample with the published one and cannot be used to re-execute it.  Spawning
on ``(scenario owner, replicate)`` instead makes replication *k* of a scenario a fixed sample:
a two-replication probe redraws exactly the first two samples of the full study, which is
what lets a fast test refit committed replications and compare.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from tests.studies.evidence.registry import StudyRecord

#: What a runner's ``draw_from_seed`` returns.  A type variable rather than a concrete
#: annotation, so this module needs no dataframe import and every runner keeps its own.
Sample = TypeVar("Sample")


def replicate_seed(record: StudyRecord, scenario: str, replicate: int) -> int:
    """The seed replication ``replicate`` of ``scenario`` is drawn with."""
    scenarios = tuple(record.scenarios)
    if scenario not in scenarios:
        raise KeyError(f"{record.slug} has no scenario {scenario!r}")
    if replicate < 0:
        raise ValueError(f"replicate must be non-negative; got {replicate}")
    owner = record.scenario_seed_owners.get(scenario, scenario)
    sequence = np.random.SeedSequence(
        entropy=record.seed, spawn_key=(scenarios.index(owner), replicate)
    )
    return int(sequence.generate_state(1)[0])


def draw_replicate(
    record: StudyRecord,
    sampler: Callable[[str, int, int], Sample],
    scenario: str,
    n: int,
    replicate: int,
) -> Sample:
    """Replication ``replicate`` of ``scenario``, from ``record``'s own seed stream.

    Both ``record`` and ``sampler`` are required arguments, and neither has a default.  A
    helper that closed over a module-level ``STUDY`` would hand every adopting study the
    seed of whichever module defined the helper, while each published its own in
    ``manifest.json``, which is the failure ``canonical_tmle.draw_for`` already describes
    and ``test_each_study_draws_from_the_seed_it_publishes`` already catches.
    """
    return sampler(scenario, n, replicate_seed(record, scenario, replicate))


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
    entropy = record.seed if record.resampling_seed is None else record.resampling_seed
    sequence = np.random.SeedSequence(entropy=entropy, spawn_key=(int.from_bytes(digest, "big"),))
    return int(sequence.generate_state(1)[0])
