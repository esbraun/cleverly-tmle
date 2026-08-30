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

import json
import re
from pathlib import Path

import pytest

from tests.canonical import runner_source
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.schema import REPLICATE_COLUMNS

#: Every reference runner, by the study directory that owns it.
RUNNERS = tuple(sorted((ROOT / "tests" / "canonical").glob("*/run_*.R")))

_KEY = re.compile(r"^\s*(?P<name>[a-z_]+)\s*=")


def _top_level_arguments(source: str) -> tuple[str, ...]:
    """Split R call arguments without mistaking commas inside an expression for columns."""
    arguments: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    comment = False
    for index, character in enumerate(source):
        if comment:
            if character == "\n":
                comment = False
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "#":
            comment = True
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(source[start:index])
            start = index + 1
    arguments.append(source[start:])
    return tuple(arguments)


def _argument_key(argument: str) -> str | None:
    """Read a named argument after discarding any explanatory R comment lines."""
    uncommented = re.sub(r"(?m)^\s*#.*(?:\n|$)", "", argument)
    match = _KEY.match(uncommented)
    return None if match is None else match.group("name")


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
        keys = tuple(
            dict.fromkeys(
                name
                for argument in _top_level_arguments(tail[:end])
                if (name := _argument_key(argument)) is not None
            )
        )
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


#: A ``source()``d file's path, as the runner writes it, relative to ``tests/canonical``.
_SOURCED_PATH = re.compile(r"""source\(["']/fixture/(?P<path>[^"']+)["']\)""")


def sourced_files(runner: Path) -> tuple[Path, ...]:
    """Every file one runner reads into its environment, in written order."""
    root = ROOT / "tests" / "canonical"
    return tuple(root / match.group("path") for match in _SOURCED_PATH.finditer(runner.read_text()))


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda path: path.parent.name)
def test_every_file_a_runner_sources_is_hashed_by_its_own_manifest(runner: Path) -> None:
    """A shared adapter is half of a comparator, so the manifest has to record it.

    ``test_method_evidence.py::test_the_manifest_hashes_every_published_result`` reads the
    ledger and asserts every name in it still exists with the recorded bytes.  It cannot see
    a file the ledger omits.  Sharing R between two studies is therefore a way to lose
    provenance rather than to gain it: move a guard into a new common file, list it in
    ``regenerate.py`` so the container mounts it, and the comparator now runs code no manifest
    records.  ``regenerate.py`` is not hashed, so nothing else notices.

    This closes that direction.  A file the runner sources must be named in
    ``reference_sha256``, which makes adding one a deliberate act with a regeneration attached
    rather than an edit that quietly drops a source out of the record.
    """
    manifest = json.loads((runner.parent / "manifest.json").read_text())
    hashed = set(manifest["reference_sha256"])
    for source in sourced_files(runner):
        name = source.relative_to(ROOT).as_posix()
        assert name in hashed, (
            f"{runner.parent.name} sources {name}, which its manifest does not hash. The "
            f"container runs that file, so the study's recorded provenance does not cover the "
            f"code that produced its rows. Add it to Reference.extra_files and regenerate"
        )


#: The two point-treatment adapters, which restate one transcription.
POINT_ADAPTERS = (
    ROOT / "tests" / "canonical" / "tmle_point_adapter.R",
    ROOT / "tests" / "canonical" / "tmle_continuous_point_adapter.R",
)

#: A refusal message, taken from the first string literal of each ``stop()``.
_REFUSAL = re.compile(r"""stop\(\s*(?:sprintf\(\s*)?"(?P<message>[^"]*)\"""", re.S)

#: What each point adapter refuses that the other must not be expected to, and why.
#:
#: Declared rather than derived, so the pair's *differences* are a stated fact and everything
#: else is required to match.  Both entries are about the outcome type the adapter transcribes:
#: only the binary one has probabilities to bound and ratio estimands to check a log scale on.
DECLARED_ADAPTER_DIVERGENCES = {
    "tmle_point_adapter.R": frozenset(
        {
            "binary-outcome nuisance predictions must be finite and strictly between zero and one",
            "tmle returned inconsistent native log inference for %s",
        }
    ),
    "tmle_continuous_point_adapter.R": frozenset(
        {"continuous-outcome nuisance predictions must be finite"}
    ),
}

#: What both adapters must refuse, whatever the outcome type.
#:
#: Named rather than counted.  Losing one of these leaves the pair still *agreeing*, so the
#: symmetric-difference check below stays silent while both files drop a guard together.  That
#: is the one drift the difference cannot see, and it is the one a shared refactor causes.
SHARED_POINT_REFUSALS = (
    "tmle fit omitted estimates: %s",
    "qn must be an n by 2 matrix aligned with weights",
    "observation weights must be finite and strictly positive",
    "truth join found %d rows for %s/%s/%s",
    "tmle returned an invalid %s result for %s/%s",
    "tmle returned a reversed %s interval for %s/%s",
)

#: What both adapters must transcribe the same way, whatever the outcome type.
#:
#: Each is a quantity a published row carries, and each has a plausible wrong answer that no
#: other check would see.  ``covered`` read off anything but the reported interval would
#: publish a coverage the implementation did not report; ``initial_estimate`` taken as an
#: unweighted mean would publish an untargeted plug-in against a different population than the
#: estimand it sits beside; ``n`` taken from the frame rather than the aligned nuisance matrix
#: would survive a misaligned join that the row count alone cannot show.
SHARED_TRANSCRIPTION = (
    "ey0 = stats::weighted.mean(qn[, 1], weights)",
    "ey1 = stats::weighted.mean(qn[, 2], weights)",
    "covered = as.integer(interval[[1]] <= truth && truth <= interval[[2]])",
    "std_error = sqrt(variance)",
    "n = nrow(qn)",
)


def refusals(adapter: Path) -> frozenset[str]:
    """Every message one adapter refuses on, by the literal it names it with."""
    return frozenset(_REFUSAL.findall(adapter.read_text(encoding="utf-8")))


class TestTheTwoPointAdaptersStayOneTranscription:
    """``tmle_point_adapter.R`` and ``tmle_continuous_point_adapter.R`` restate one contract.

    About fifty-five of the continuous adapter's sixty-nine lines are the binary adapter's:
    the nuisance and weight validation, the weighted initial means, the truth join and its
    one-row refusal, the invalid and reversed-interval guards, and the published frame.  They
    are two files because each is hashed by a different study's manifest, so collapsing them
    is a regeneration rather than a refactor.

    That leaves a fix applied to one and not the other, which nothing detected.  These checks
    make the pair's agreement the default and its differences declared, so a guard added to
    one fails until it is either added to the other or written down as a difference with a
    reason.
    """

    def test_both_adapters_are_reached_by_a_registered_runner(self) -> None:
        """Neither adapter is dead code, so the checks below are about a live comparator."""
        sourced = {path for runner in RUNNERS for path in sourced_files(runner)}
        for adapter in POINT_ADAPTERS:
            assert adapter in sourced, f"{adapter.name} is sourced by no registered runner"

    @pytest.mark.parametrize("message", SHARED_POINT_REFUSALS)
    def test_both_adapters_keep_every_shared_refusal(self, message: str) -> None:
        for adapter in POINT_ADAPTERS:
            assert message in refusals(adapter), (
                f"{adapter.name} no longer refuses on {message!r}. Both point adapters answer "
                f"to this guard, so dropping it from one publishes a row the other would have "
                f"stopped. Remove it from SHARED_POINT_REFUSALS only when neither needs it"
            )

    def test_a_guard_added_to_one_adapter_is_added_to_the_other(self) -> None:
        """The only differences are the declared ones, so a new guard cannot land in one file."""
        for adapter in POINT_ADAPTERS:
            other = next(path for path in POINT_ADAPTERS if path != adapter)
            only_here = refusals(adapter) - refusals(other)
            declared = DECLARED_ADAPTER_DIVERGENCES[adapter.name]
            assert only_here == declared, (
                f"{adapter.name} refuses {sorted(only_here)} and {other.name} does not. The two "
                f"adapters transcribe one contract, so either add the guard to {other.name}, or "
                f"declare the difference in DECLARED_ADAPTER_DIVERGENCES with the reason it is "
                f"one. Expected {sorted(declared)}"
            )

    @pytest.mark.parametrize("construct", SHARED_TRANSCRIPTION)
    def test_both_adapters_build_the_shared_quantities_the_same_way(self, construct: str) -> None:
        for adapter in POINT_ADAPTERS:
            assert construct in adapter.read_text(encoding="utf-8"), (
                f"{adapter.name} does not build its published row with `{construct}`. The two "
                f"adapters feed the same schema, so a quantity computed two ways files two "
                f"different numbers under one column name"
            )
