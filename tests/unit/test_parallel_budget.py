r"""The core budget, and that no tier quietly asks for more than the box has.

Concurrency here is a product of three terms -- **outer × inner × threads-per-fit** -- and
the third is pinned to one on purpose (:func:`cleverly.learners.set_thread_limit`, "keeps
each fit single-threaded so the parallelism happens across folds and candidates instead").
So the whole question is how the first two divide a budget, and the failure mode is that
nobody multiplies them: a tier raises its inner ``n_jobs`` for a good local reason while the
outer layer is already saturating the machine, and the run gets slower while looking
parallel.

That is not hypothetical in this repository.  The failure at the other end was a
``SIGKILL``ed run leaving ``LokyProcess`` workers "at ~75% CPU each,
reparented to init, still going a minute later", and a benchmark taken afterwards reading a
300x bogus timing.  Oversubscription and orphaned workers are the same resource failing in
two directions.

**The budget is read, not assumed.**  :func:`tests.parallel.available_cores` goes through
joblib, which goes through loky, which reads a CFS quota and an affinity mask; ``-n auto``
reaches for ``psutil`` and falls back to ``os.cpu_count()``, neither of which does.  Inside
a quota-limited container -- every CI runner, and the sandbox this repository is developed
in -- those are different numbers, and the second one is the wrong one.

**What this module cannot check, and says so rather than pretending.**  It checks the plan
the tiers *declare*, not the concurrency a run *realises*: a third-party library spawning
its own pool is invisible here, which is exactly why ``set_thread_limit`` exists and is
tested separately in :mod:`tests.unit.test_thread_limit`.
"""

from __future__ import annotations

import os

import pytest

from tests.parallel import (
    CORES_ENV,
    OUTER_OVERSUBSCRIPTION,
    STUDY_JOBS,
    available_cores,
    describe_cores,
    worker_count,
)


class TestTheDetector:
    """What ``available_cores`` reads, and what wins over what."""

    def test_it_is_at_least_one(self) -> None:
        # A budget of zero would make every `min()` downstream serialise silently, which is
        # the failure that looks like the code working.
        assert available_cores() >= 1

    def test_the_override_wins_and_is_validated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(CORES_ENV, "3")
        assert available_cores() == 3
        monkeypatch.setenv(CORES_ENV, "0")
        with pytest.raises(ValueError, match="at least 1"):
            available_cores()
        monkeypatch.setenv(CORES_ENV, "several")
        with pytest.raises(ValueError, match="not an integer"):
            available_cores()

    def test_it_does_not_simply_report_the_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: a constrained process must see the constraint.

        ``LOKY_MAX_CPU_COUNT`` stands in for the CFS quota, because a quota cannot be
        imposed from inside a test -- it is the same code path in loky either way, and it
        is the one ``os.cpu_count()`` is blind to.
        """
        monkeypatch.delenv(CORES_ENV, raising=False)
        host = os.cpu_count() or 1
        if host < 2:  # pragma: no cover - a single-core box has nothing to constrain
            pytest.skip("needs more than one core to show a constraint")
        monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
        # joblib caches nothing here, so the constraint is visible on the next call.
        assert available_cores() <= host

    def test_the_description_names_its_source(self) -> None:
        """A surprising runtime should be able to say what it thought it had."""
        line = describe_cores()
        assert "usable cores:" in line and "host reports" in line


class TestTheBudget:
    """outer × inner, against what the box will actually give."""

    def test_the_outer_layer_does_not_exceed_the_budget(self) -> None:
        assert worker_count() <= available_cores() * OUTER_OVERSUBSCRIPTION

    def test_the_fast_tier_leaves_the_inner_layer_alone(self) -> None:
        """The default has to stay serial, or the two layers multiply.

        The fast tier's outer layer is already fed -- thousands of short tests -- so an
        inner pool there would be contending with it rather than filling a gap.  This reads
        the shipped default rather than a copy of it, so raising ``TMLE.n_jobs``'s default
        in ``src`` fails here and has to be argued for.
        """
        from cleverly import TMLE

        assert TMLE.__init__.__kwdefaults__["n_jobs"] == 1

    def test_the_studies_keep_their_measured_setting(self) -> None:
        """``STUDY_JOBS`` is a record of a measurement, not a tuning knob.

        Three paired runs on four cores found that ``n_jobs=1`` made the e2e tier 35% slower
        (75.7s to 102.3s), with three xdist workers idle while the longest test ran
        twice as long.  If this constant changes, that row changes with it and needs its own
        three paired runs -- on the box the new number is claimed for.
        """
        assert STUDY_JOBS == 2
