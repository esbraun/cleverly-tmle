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


def _spec():
    """Any registered kernel: ``_common`` reads a name and two flags off it, nothing more.

    Through ``resolve`` rather than ``REGISTRY`` directly, since that is what populates it.
    """
    from benchmarks.numba.kernels import resolve

    return resolve(None)[0]


def _environment():
    from benchmarks.numba.resources import environment_record

    return environment_record()
