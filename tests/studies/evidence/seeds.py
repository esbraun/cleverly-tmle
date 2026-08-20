"""Per-replication seeds that do not move when the replication count changes.

Deriving seeds from ``SeedSequence(seed).generate_state(2 * replicates)`` -- one flat draw
sized by the total -- re-seeds every replication as soon as the total changes, so a short
run shares no sample with the published one and cannot be used to re-execute it.  Spawning
on ``(scenario, replicate)`` instead makes replication *k* of a scenario a fixed sample:
a two-replication probe redraws exactly the first two samples of the full study, which is
what lets a fast test refit committed replications and compare.
"""

from __future__ import annotations

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
