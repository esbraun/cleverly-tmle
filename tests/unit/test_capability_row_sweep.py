"""Every kind of fit answers every row that a request resolves as available.

RM23 found capability rows that read ``available`` while their call refused. A combined
report then ran the row and published "the operation declined this request", or raised
and returned no report. Each defect was a kind of fit that no test had asked the row
about. This module asks every row about every kind in
:data:`tests.unit._capability_sweep_support.KINDS`, with every cost paid and every
argument a row waits on supplied.

A sweep that refuses nothing would pass on a row refused by mistake, so each kind names
the rows that must answer. A sweep that sees nothing would pass on every defect, so each
mutation in :data:`~tests.unit._capability_sweep_support.MUTATIONS` restores one pre-RM23
answer and must make the sweep fail on exactly the kinds it names, with a problem that
matches its signature.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from cleverly.assessment import replayability
from tests.unit._capability_sweep_support import (
    KINDS,
    MUTATIONS,
    SEAMS,
    SPLIT_PLAN_DEFAULT,
    problems,
    ran,
    sweep,
    sweep_arguments,
)


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_row_a_request_resolves_available_answers_it(kind: str) -> None:
    result = KINDS[kind].build()
    arguments = sweep_arguments(result)
    assert problems(result, arguments) == []
    # The nonzero witness: the rows this kind must run did run, and no row it must
    # not run answered.
    answered = ran(result, arguments)
    assert KINDS[kind].must_run <= answered
    assert not KINDS[kind].must_not_run & answered
    if result.assessment_family == "point":
        # A fit passed the refit preflight, and a restored kind is one it refuses.
        assert replayability(result).refit_nuisances is KINDS[kind].refits


def test_each_build_is_a_fresh_copy_of_one_fit() -> None:
    """The tests share each kind's fit, and never a report or a facade another one filled."""
    first, second = KINDS["ordinary"].build(), KINDS["ordinary"].build()
    assert first is not second
    assert first.estimator is second.estimator and first.data is second.data
    first.diagnostics.run_all()
    assert first.assessment_cache
    assert second.assessment_cache == {}
    assert "diagnostics" not in vars(second)


class _Recorder:
    """A stand-in for ``pytest.MonkeyPatch`` that records each target and patches nothing."""

    def __init__(self) -> None:
        self.targets: set[tuple[Any, str]] = set()

    def setattr(self, owner: Any, name: str, value: Any) -> None:
        self.targets.add((owner, name))

    def delattr(self, owner: Any, name: str) -> None:
        self.targets.add((owner, name))


class TestEachMutationRestoresAMismatch:
    """Each mutation makes the sweep fail on exactly the kinds it names."""

    @pytest.mark.parametrize("name", list(MUTATIONS))
    def test_each_mutation_names_its_kinds_and_patches_a_seam(self, name: str) -> None:
        """A mutation outside the seams would test code the fixes did not add."""
        mutation = MUTATIONS[name]
        assert mutation.fails_on <= set(KINDS), mutation.describe
        if name != "M0":
            assert mutation.fails_on, mutation.describe
            assert mutation.signature, mutation.describe
        recorder = _Recorder()
        mutation.apply(recorder)  # type: ignore[arg-type]
        assert recorder.targets, mutation.describe
        assert recorder.targets <= {*SEAMS, SPLIT_PLAN_DEFAULT}, mutation.describe

    @pytest.mark.parametrize("kind", list(KINDS))
    @pytest.mark.parametrize("name", list(MUTATIONS))
    def test_the_sweep_fails_on_exactly_the_named_kinds(
        self, name: str, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutation = MUTATIONS[name]
        # Built before the mutation, because M7 removes what a restored build reads. The
        # copy has an empty report cache and no memoized facade, so no answer computed
        # before the mutation can hide it.
        result = KINDS[kind].build()
        mutation.apply(monkeypatch)
        found = sweep(result)
        if kind in mutation.fails_on:
            # The failure is the mutated component's, and not a stale seam's.
            signed = [problem for problem in found if re.search(mutation.signature, problem)]
            assert signed, (mutation.describe, found)
        else:
            assert found == [], mutation.describe
