"""Twelve R runners restate one published contract, and it has to stay one contract.

Every canonical study hands its samples to a reference implementation through an R script,
and each of those scripts ends the same way: split the rows by replication, fit them in
parallel, refuse the run if a worker failed or a replication went missing, and write the
fourteen-column table :mod:`tests.studies.evidence.schema` declares.

That ending was written out twelve times, and it had already drifted.
``tests/canonical/tmle3/run_tmle3.R`` grew a memory cap and a malformed-worker check after a
run lost 86 of 3,200 replications while reporting success; ``tests/canonical/drtmle/run_drtmle.R``
has neither.  Nine of the twelve now reach it through ``tests/canonical/study_harness.R``
instead.  The remaining three keep their own copies, because
``test_method_evidence.py::test_the_manifest_hashes_every_published_result`` asserts the exact
sha256 of every reference source: collapsing one is a regeneration rather than a refactor, and
the three left out are the ones whose fits consume randomness, where harmonising the core
count could move a published number.

What this module does is make that safe.  It does not compare the scripts as text, which would
only assert that nobody had improved one of them, and it reads each runner *as executed* --
the file plus everything it sources -- so a check passes or fails on what the run does rather
than on which file a line sits in.  It asserts the three things a copy cannot quietly lose:

* the published row is the schema, in the schema's order, so twelve R restatements of
  :data:`~tests.studies.evidence.schema.REPLICATE_COLUMNS` cannot disagree with the one Python
  declaration that names it;
* the run stops rather than publishes when a replication goes missing, which is the failure
  the ``tmle3`` incident is a record of and the one a row count alone does not catch;
* the run checks what a worker actually returned, because ``mclapply`` hands back a value for
  a forked child that was killed and ``rbind`` drops it without a word.

A runner that loses one of those publishes a short or mangled study that reads exactly like a
complete one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.canonical import runner_source
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.schema import REPLICATE_COLUMNS

#: Every reference runner, by the study directory that owns it.
RUNNERS = tuple(sorted((ROOT / "tests" / "canonical").glob("*/run_*.R")))

#: The ``name = `` keys inside a ``data.frame(`` call.
_KEY = re.compile(r"^\s{2,}(?P<name>[a-z_]+) = ", re.M)


def published_keys(source: str) -> tuple[str, ...]:
    """The column names one runner's published row is built from, in written order.

    The block is found by the column only a published row carries rather than by position, so
    a runner that builds a second frame for its own use is still read at the right one.
    """
    for match in re.finditer(r"^\s*data\.frame\($", source, re.M):
        tail = source[match.end() :]
        end = tail.find("stringsAsFactors")
        if end < 0:
            continue
        keys = tuple(dict.fromkeys(_KEY.findall(tail[:end])))
        if "inference_scale" in keys:
            return keys
    raise AssertionError("no published row builder found")


def test_every_registered_runner_is_covered() -> None:
    """A new runner joins these checks by existing, not by being added to a list."""
    assert len(RUNNERS) >= 12, f"only {len(RUNNERS)} reference runners found under tests/canonical"


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda path: path.parent.name)
class TestTheSharedPublishedContract:
    def test_the_published_row_is_the_declared_schema(self, runner: Path) -> None:
        keys = published_keys(runner_source(runner))
        assert keys[: len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS, (
            f"{runner.parent.name} builds its published row as {keys}, which is not "
            f"{REPLICATE_COLUMNS} in order. The driver reindexes to the declared columns, so a "
            f"reordering here does not raise: it silently files one quantity under another's "
            f"name"
        )
        # Anything after the shared fourteen is a declared extra artefact, which the study's
        # own ``extra_artifacts`` hook consumes before the driver reindexes.  The drtmle row
        # carries four fit-health columns that way.
        assert len(set(keys)) == len(keys), f"{runner.parent.name} names a column twice"

    def test_the_runner_takes_the_three_paths_the_driver_passes(self, runner: Path) -> None:
        assert "if (length(args) != 3) stop(" in runner_source(runner), (
            f"{runner.parent.name} does not refuse a wrong argument count. "
            f"tests/canonical/regenerate.py::Reference.run passes exactly samples, truths and "
            f"output, and a runner that reads them positionally without checking will write "
            f"its results over one of its inputs when the contract changes"
        )

    def test_a_missing_replication_stops_the_run(self, runner: Path) -> None:
        source = runner_source(runner)
        assert any(
            "replication" in source[match.end() : match.end() + 200]
            for match in re.finditer(r"stop\(", source)
        ), (
            f"{runner.parent.name} has no refusal that names a replication. A study that "
            f"publishes fewer replications than it drew is not a shorter study, it is a study "
            f"whose Monte Carlo error every cell understates"
        )

    def test_the_runner_checks_what_a_worker_returned(self, runner: Path) -> None:
        assert re.search(r"\binherits\b|\bis\.data\.frame\b", runner_source(runner)), (
            f"{runner.parent.name} collects its workers without checking what they returned. "
            f"mclapply returns a value rather than raising for a child that errored or was "
            f"killed, and rbind drops a non-frame silently"
        )
