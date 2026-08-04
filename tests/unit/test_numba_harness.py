"""The benchmark harness itself: what it records, and in what order it measures.

``tests/unit/test_numba_benchmark.py`` is about the *kernels* -- that the compiled
implementations compute what the numpy reference computes.  Nothing there touches
``benchmarks.numba.runner`` or ``benchmarks.numba.timing``, so "the correctness tier stays
green" is a nearly vacuous gate for a change to either.  This module is the gate for those.

Two properties, and each was a defect before it was a test:

* **the effective thread count is read while the plan is in force.**  It was read after,
  where ``applied`` has already put numba's count back, so every measurement on a
  four-core sweep was filed at the count the *process* booted with rather than the one it
  ran at.
* **arms are timed in a rotation, not in blocks.**  Shuffling the order of whole
  implementations and then running each one's repetitions back to back is randomised block
  order: every sample of an arm comes from one contiguous window, so a neighbour waking up
  during that window is charged to that arm alone.

Neither needs numba to state, but the module under test imports it through the kernel
registry, so the whole file skips without it exactly as the correctness tier does.
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

numba = pytest.importorskip("numba", reason="numba lives in the `bench` extra")

from benchmarks.numba import runner  # noqa: E402
from benchmarks.numba.resources import ThreadPlan  # noqa: E402
from benchmarks.numba.timing import (  # noqa: E402
    Arm,
    measure,
    measure_interleaved,
    speedup_interval,
)


class TestTheEffectiveThreadCount:
    """``num_cores_effective`` is audit metadata, and it has to be audit-worthy.

    Nothing reads the column -- no table, threshold or flag -- so this was never a wrong
    *ratio*.  It was a wrong record of the conditions a right ratio was taken under, which
    is the thing a provenance column exists to be.
    """

    def test_it_is_read_inside_the_applied_plan(self, monkeypatch):
        """The mutation: move the read out of the block and it returns the ceiling.

        The stand-in ``applied`` flips a flag for the duration, and the stand-in
        ``effective_threads`` answers differently depending on it -- so the assertion is
        literally "this was read while the plan was in force", not a proxy for it.
        """
        from benchmarks.numba.implementations import numba_parallel

        inside = []

        @staticmethod
        def _applied(plan):
            class _Block:
                def __enter__(self_inner):
                    inside.append(True)
                    return None

                def __exit__(self_inner, *exc):
                    inside.pop()
                    return False

            return _Block()

        monkeypatch.setattr(runner, "applied", _applied)
        monkeypatch.setattr(numba_parallel, "effective_threads", lambda: 2 if inside else 99)

        with runner.applied(ThreadPlan(numba_threads=2)):
            assert runner._effective_cores(ThreadPlan(numba_threads=2)) == 2
        assert runner._effective_cores(ThreadPlan(numba_threads=2)) == 99

    def test_a_serial_plan_reports_the_cores_it_asked_for(self):
        """No numba pool is involved, so there is nothing to read and nothing to cap."""
        assert runner._effective_cores(ThreadPlan(numba_threads=1, workers=1)) == 1

    def test_a_skipped_row_records_no_effective_count(self):
        """It entered no plan, so there is no count -- and ``None`` says that.

        Filling in the requested value instead would put a number in the column that no
        measurement stands behind, which is the same defect as reading it too late wearing
        a tidier face.
        """
        row = runner._skipped(
            _spec(),
            "numba_parallel",
            1_000,
            4,
            ThreadPlan(numba_threads=4),
            {"n": 1_000},
            _environment(),
            "asked for 4 cores; the box has 2",
        )

        assert row.num_cores_effective is None
        assert row.skipped_reason.startswith("asked for 4 cores")


class TestTheRotation:
    """One step of every arm before a second step of any.

    What this replaced is randomised *block* order: the whole implementations shuffled, then
    each one's warmups and repetitions run back to back.  Every sample of an arm came from
    one contiguous window, so machine drift slower than a block was confounded with arm
    rather than spread over both -- and the module docstring said "interleaving" anyway.
    """

    def test_every_arm_advances_one_step_per_round(self):
        """The claim in one assertion, and the one that fails against block order.

        The log splits into ``repeats`` slices of ``len(arms)``, and each slice must be a
        permutation of the arms.  Under the old loop the log was ``AAAABBBBCCCC`` and the
        first slice alone fails it.
        """
        log: list[str] = []
        arms = [_logging_arm(key, log) for key in "abc"]

        measure_interleaved(arms, warmups=0, repeats=4, measure_memory=False)

        timed = log[len(arms) :]  # the probe pass runs one call per arm first
        rounds = [timed[i : i + len(arms)] for i in range(0, len(timed), len(arms))]
        assert len(rounds) == 4
        for taken in rounds:
            assert sorted(taken) == ["a", "b", "c"], log

    def test_no_arms_samples_are_contiguous(self):
        """Stated directly, because it is the property a ratio's honesty rests on."""
        log: list[str] = []
        arms = [_logging_arm(key, log) for key in "ab"]

        measure_interleaved(arms, warmups=0, repeats=5, measure_memory=False)

        timed = "".join(log[len(arms) :])
        assert "aaa" not in timed and "bbb" not in timed, timed

    def test_the_warm_ups_all_precede_the_first_timed_call(self):
        """Warmups are per arm and outside the rotation.

        A numba kernel's compilation happens on its first call, and if that landed inside
        round zero it would be charged to whichever arm the round started with -- and to
        that round, which every other arm's sample zero is paired against.
        """
        log: list[str] = []
        arms = [_logging_arm(key, log) for key in "ab"]

        measure_interleaved(arms, warmups=3, repeats=2, measure_memory=False)

        # Four calls per arm before any rotation: three warmups and one probe.
        assert log[:8] == list("aaaabbbb"), log

    def test_every_arm_gets_the_same_number_of_samples(self):
        """Lockstep is what pairs sample ``r`` of one arm with sample ``r`` of another."""
        arms = [_sleeping_arm("fast", 0.0), _sleeping_arm("slow", 0.002)]

        out = measure_interleaved(
            arms, warmups=0, repeats=6, min_total_seconds=0.05, measure_memory=False
        )

        assert len(out["fast"].samples) == len(out["slow"].samples) == 6

    def test_a_fast_arm_is_batched_and_a_slow_one_is_not(self):
        """``min_total_seconds`` became a batch size, and this is what that buys.

        A microsecond kernel needs hundreds of calls before the clock stops dominating; a
        millisecond one needs none.  Taking that out of the *sample count* is what lockstep
        costs, so it goes into the batch instead.
        """
        arms = [_sleeping_arm("fast", 0.0), _sleeping_arm("slow", 0.03)]

        out = measure_interleaved(
            arms, warmups=0, repeats=4, min_total_seconds=0.08, measure_memory=False
        )

        # The slow arm already covers 0.08s in four single calls; the fast one needs
        # thousands, and gets them without taking a fifth sample.
        assert out["fast"].calls_per_sample > 1
        assert out["slow"].calls_per_sample == 1
        assert sum(out["slow"].samples) >= 0.08

    def test_the_collector_is_re_enabled_when_an_arm_raises(self):
        """One disable for the whole rotation means one ``finally`` to get right."""
        import gc

        def explode():
            raise RuntimeError("deliberate")

        was_enabled = gc.isenabled()
        with pytest.raises(RuntimeError, match="deliberate"):
            measure_interleaved(
                [Arm(key="boom", call=explode)], warmups=0, repeats=1, measure_memory=False
            )

        assert gc.isenabled() == was_enabled

    def test_the_plan_is_entered_once_per_group_per_round(self):
        """The property that makes the rotation affordable at all.

        Entering a thread plan builds a ``ThreadpoolController``, which this repository
        measured at ~0.7 ms -- an order of magnitude more than a fast kernel's call.  So a
        round rotates over *plan groups* and takes every arm sharing one inside a single
        entry.  Rotating arm by arm would spend the run switching plans.
        """
        entries: list[str] = []

        def context(tag):
            @contextlib.contextmanager
            def enter():
                entries.append(tag)
                yield

            return enter

        arms = [
            Arm(key="a", call=lambda: None, context=context("x"), group="x"),
            Arm(key="b", call=lambda: None, context=context("x"), group="x"),
            Arm(key="c", call=lambda: None, context=context("y"), group="y"),
        ]

        measure_interleaved(arms, warmups=0, repeats=3, measure_memory=False)

        # One entry per group for the warm-up/probe pass, then one per group per round.
        assert entries.count("x") == 1 + 3
        assert entries.count("y") == 1 + 3

    def test_measure_is_the_one_arm_case(self):
        """Kept as an entry point, implemented through the rotation, so there is one loop."""
        calls = []
        out = measure(lambda: calls.append(1), warmups=2, repeats=3, measure_memory=False)

        assert len(out.samples) == 3
        assert out.cold_seconds is None
        assert out.calls_per_sample == 1

    def test_measure_still_times_a_cold_call_separately(self):
        arms_run = []
        out = measure(
            lambda: None,
            warmups=0,
            repeats=2,
            cold=lambda: arms_run.append("cold"),
            measure_memory=False,
        )

        assert arms_run == ["cold"]
        assert out.cold_seconds is not None and out.cold_seconds >= 0.0


class TestThePairedInterval:
    """Under a rotation the samples are paired, so the bootstrap should be.

    ``speedup_interval`` used to justify independent resampling with "the two
    implementations were timed on interleaved calls, so there is no pairing to preserve".
    That was true of block order and is exactly what the rotation retired: keeping it would
    discard the correlation the rotation exists to create and report an interval wider than
    the design earns.
    """

    def test_a_common_drift_cancels_out_of_a_paired_interval(self):
        """Same ratio in every round, and a machine that wanders by a factor of three."""
        drift = [1.0, 1.4, 3.0, 1.1, 2.2, 1.0, 2.8, 1.3, 1.9, 1.05]
        baseline = [2.0 * d for d in drift]
        candidate = [1.0 * d for d in drift]

        _, low, high = speedup_interval(baseline, candidate, seed=0, paired=True)
        _, wide_low, wide_high = speedup_interval(baseline, candidate, seed=0, paired=False)

        # Every round says exactly 2.0, so a paired interval has nothing to be wide about.
        assert low == high == 2.0
        assert wide_high - wide_low > 0.0

    def test_unequal_lengths_fall_back_to_independent_resampling(self):
        """No pairing to preserve, and the honest construction says so rather than crashing."""
        point, low, high = speedup_interval([2.0] * 8, [1.0] * 5)

        assert point == 2.0
        assert low <= point <= high


def _logging_arm(key: str, log: list[str]) -> Arm:
    return Arm(key=key, call=lambda k=key: log.append(k))


def _sleeping_arm(key: str, seconds: float) -> Arm:
    return Arm(key=key, call=lambda s=seconds: time.sleep(s) if s else None)


class TestTheCoreListOverride:
    """``--num-cores`` replaces the config's list, and must not learn to clamp.

    The workflow's ``full`` job used to pass ``--num-cores 1 2`` alongside
    ``--config full.yaml``, which discarded that file's ``[1, 2, 4, 8]`` -- so no job in
    this repository had ever run above two cores while the reports read as though a full
    sweep had.  Dropping the flag is the fix; these are the tests that keep the *semantics*
    the fix relies on from being "helpfully" softened afterwards.
    """

    def test_an_explicit_list_replaces_the_configs(self):
        """A replace, not an intersection: ``--num-cores 1 2`` means 1 and 2."""
        from benchmarks.numba.cli import _apply_overrides
        from benchmarks.numba.config import Config

        config = _apply_overrides(Config(num_cores=(1, 2, 4, 8)), _args(num_cores=[1, 2]))

        assert config.num_cores == (1, 2)

    def test_omitting_the_flag_keeps_the_configs_list(self):
        """The regression itself.  A flag that is not passed must change nothing."""
        from benchmarks.numba.cli import _apply_overrides
        from benchmarks.numba.config import Config

        config = _apply_overrides(Config(num_cores=(1, 2, 4, 8)), _args(num_cores=None))

        assert config.num_cores == (1, 2, 4, 8)

    def test_a_core_count_above_the_box_is_skipped_rather_than_capped(self):
        """What honouring the config's list costs on a small runner, and why it is right.

        `resources.py` refuses a count the machine cannot serve instead of capping it,
        because an efficiency column computed against a silently capped count is a
        fabrication.  So the 4- and 8-core rows of a full sweep on a two-core box come back
        *named as skipped*, which is a reader-visible gap rather than a plausible number.
        """
        row = runner._skipped(
            _spec(),
            "numba_parallel",
            1_000,
            8,
            ThreadPlan(numba_threads=8),
            {"n": 1_000},
            _environment(),
            "asked for 8 cores; the box has 2",
        )

        assert "8" in row.skipped_reason and "2" in row.skipped_reason
        assert row.repeat_count == 0
        assert row.warm_seconds != row.warm_seconds  # nan: nothing was timed


def _args(**overrides):
    """An ``argparse.Namespace`` with every overridable flag unset but the named ones."""
    import argparse

    from benchmarks.numba.cli import _OVERRIDABLE

    return argparse.Namespace(**{name: overrides.get(name) for name in _OVERRIDABLE})


def _spec():
    """Any registered kernel: ``_common`` reads a name and two flags off it, nothing more.

    Through ``resolve`` rather than ``REGISTRY`` directly, since that is what populates it.
    """
    from benchmarks.numba.kernels import resolve

    return resolve(None)[0]


def _environment():
    from benchmarks.numba.resources import environment_record

    return environment_record()
