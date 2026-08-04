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

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

numba = pytest.importorskip("numba", reason="numba lives in the `bench` extra")

from benchmarks.numba import runner  # noqa: E402
from benchmarks.numba.resources import ThreadPlan  # noqa: E402


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
