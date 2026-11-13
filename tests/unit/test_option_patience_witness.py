"""Fast-tier witnesses of ``patience=`` on the C-TMLE path (Fix-8 plan section 4, row C).

Each witness reads a fixed sample of a declared ``option-patience`` law, fitted with the
study's own settings (:mod:`tests.studies._option_patience_laws`), and each has a mutation
control that must fail it:

* **The counter stops the path.**  On an ``LC1`` sample the greedy fit with patience ``p*``
  stops before the whole path's end, its path is a prefix of the whole path, and its estimate is
  the whole path's because the smallest risk lies in the prefix.  A counter that never stops
  fails it.
* **Greedy with patience keeps the greedy selection.**  On ``LC3`` samples where the counter
  stops the greedy path, the patient fit returns the whole path's estimate.  A selection that
  takes the last scored candidate in place of the smallest risk fails it.
* **A pre-ordered path stops before the confounder.**  On an ``LC3`` sample the ordered fit with
  patience 2 (instruments first) selects a model without ``W1`` and misses the truth by more than
  three whole-path SDs, while the whole path selects ``W1``.  A counter that never stops fails
  it.
* **The outcome regression omits ``W1``.**  A spy reads the columns the LC3 outcome learner is
  handed: ``[A, W1, Z1, Z2, Z3]``, of which it keeps ``A, Z1, Z2, Z3``.
* **The support label.**  A pre-ordered fit with patience carries the permanent condition with
  no roadmap item; a greedy fit with patience lands on ``C12.greedy`` and reads supported, and
  without that row the support key alone keeps it off ``C01.greedy``.  A permanent condition
  that fires on the greedy path fails the greedy test.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any

import numpy as np
import pytest

from cleverly import _support as support_module
from cleverly.estimators import _option_support
from cleverly.estimators.ctmle import CTMLE, _PatienceCounter
from tests.studies import _option_patience_laws as law
from tests.studies import option_patience_pilot as pilot
from tests.unit._source_mutation import mutate

ctmle_module = importlib.import_module("cleverly.estimators.ctmle")

#: Pilot draws with the behaviour each witness reads (``tests/studies/option_patience_pilot.py``
#: per-draw rows).  The draws are fixed samples; the pilot stream is no study's.
LC1_STOPPED_DRAW = 1
LC3_GREEDY_STOPPED_DRAWS = (1, 3)
LC3_ORDERED_STOPPED_DRAW = 10
#: The whole-path SD of the LC3 ordered fit at n = 2,000, rounded down from the pilot (0.032).
LC3_WHOLE_SD = 0.03
#: The pre-ordered condition's text, which is permanent.
PRE_ORDERED_TEXT = "stops a pre-ordered candidate path"


@functools.cache
def sample(stream: str, draw: int) -> Any:
    return law.draw(stream, law.N, pilot.pilot_seed(stream, draw))


@functools.cache
def fitted(stream: str, draw: int, name: str) -> Any:
    return law.fit(sample(stream, draw), name)


def selection(result: Any) -> Any:
    return result.extra["ctmle"]


def assert_counter_stops_the_lc1_path() -> None:
    patient = law.fit(sample("lc1", LC1_STOPPED_DRAW), "lc1_greedy_patience")
    whole = fitted("lc1", LC1_STOPPED_DRAW, "lc1_greedy_whole")
    assert selection(patient).patience == law.LC1_PATIENCE
    assert selection(patient).stopped_at is not None
    assert law.stopped(patient, whole)
    assert selection(whole).path[: len(selection(patient).path)] == selection(patient).path
    assert law.ate(patient) == law.ate(whole)


def assert_greedy_patience_keeps_the_selection() -> None:
    for draw in LC3_GREEDY_STOPPED_DRAWS:
        patient = law.fit(sample("lc3", draw), "lc3_greedy_patience")
        whole = fitted("lc3", draw, "lc3_greedy_whole")
        assert law.stopped(patient, whole)
        assert law.ate(patient) == law.ate(whole)


def assert_ordered_patience_stops_before_the_confounder() -> None:
    patient = law.fit(sample("lc3", LC3_ORDERED_STOPPED_DRAW), "lc3_ordered_patience")
    whole = fitted("lc3", LC3_ORDERED_STOPPED_DRAW, "lc3_ordered_whole")
    truth = law.truth("lc3")["ate"]
    assert law.excludes_confounder(patient)
    assert not law.excludes_confounder(whole)
    assert law.ate(patient) - truth > 3 * LC3_WHOLE_SD
    assert abs(law.ate(whole) - truth) < LC3_WHOLE_SD


class TestTheCounterStopsThePath:
    def test_the_lc1_greedy_path_stops_early(self) -> None:
        assert_counter_stops_the_lc1_path()

    def test_mutation_a_counter_that_never_stops_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ctmle_module,
            _PatienceCounter,
            "offer",
            [("return self.count >= self.patience", "return False")],
        )
        with pytest.raises(AssertionError):
            assert_counter_stops_the_lc1_path()


class TestGreedyPatienceKeepsTheSelection:
    def test_the_stopped_greedy_path_returns_the_whole_path_estimate(self) -> None:
        assert_greedy_patience_keeps_the_selection()

    def test_mutation_the_last_scored_candidate_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ctmle_module,
            CTMLE,
            "_nuisances",
            [
                (
                    "selected = int(np.argmin(cv_risk))",
                    "selected = len(cv_risk) - 1 if stopped_at is not None else "
                    "int(np.argmin(cv_risk))",
                )
            ],
        )
        with pytest.raises(AssertionError):
            assert_greedy_patience_keeps_the_selection()


class TestAPreOrderedPathStopsBeforeTheConfounder:
    def test_the_ordered_path_leaves_w1_out_and_misses_the_truth(self) -> None:
        assert_ordered_patience_stops_before_the_confounder()

    def test_mutation_a_counter_that_never_stops_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ctmle_module,
            _PatienceCounter,
            "offer",
            [("return self.count >= self.patience", "return False")],
        )
        with pytest.raises(AssertionError):
            assert_ordered_patience_stops_before_the_confounder()


def test_the_lc3_outcome_learner_omits_the_confounder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[np.ndarray] = []
    pipeline_type = type(law.lc3_outcome_learner())
    original = pipeline_type.fit

    def spy(self: Any, x: Any, y: Any, **kwargs: Any) -> Any:
        if "omit_confounder" in self.named_steps:
            seen.append(np.asarray(x))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(pipeline_type, "fit", spy)
    frame = sample("lc3", LC3_ORDERED_STOPPED_DRAW)
    result = law.fit(frame, "lc3_ordered_whole")
    full = [x for x in seen if len(x) == len(frame)]
    assert full, "the outcome learner was never fitted on the full sample"
    columns = ["A", *law.LC3_COVARIATES]
    assert np.allclose(full[0], frame[columns].to_numpy(dtype=float))
    assert result.extra["ctmle"].strategy == "ordered"
    kept = law.lc3_outcome_learner().fit(full[0], frame["Y"].to_numpy())
    assert kept.named_steps["omit_confounder"].transform(full[0]).shape[1] == 4
    assert np.allclose(
        kept.named_steps["omit_confounder"].transform(full[0]),
        frame[["A", "Z1", "Z2", "Z3"]].to_numpy(dtype=float),
    )


class TestTheSupportLabel:
    def test_a_pre_ordered_fit_carries_the_permanent_condition(self) -> None:
        support = fitted("lc3", LC3_ORDERED_STOPPED_DRAW, "lc3_ordered_patience")["ate"].support
        assert support.label == "experimental"
        assert support.row is None
        assert PRE_ORDERED_TEXT in support.reason
        assert "patience=2" in support.reason
        assert "X76" not in support.evidence

    @pytest.mark.parametrize("name", ["lc1_ordered_patience", "lc1_discrete_patience"])
    def test_every_pre_ordered_strategy_carries_it(self, name: str) -> None:
        support = law.fit(sample("lc1", LC1_STOPPED_DRAW), name)["ate"].support
        assert PRE_ORDERED_TEXT in support.reason
        assert "X76" not in support.evidence

    def test_the_conditions_cover_lasso_and_learners(self) -> None:
        for estimator in (
            CTMLE(strategy="lasso", patience=3),
            CTMLE(strategy="learners", treatment_candidates=[], patience=3),
        ):
            (condition,) = _option_support.patience_conditions(estimator)
            assert PRE_ORDERED_TEXT in condition.condition
            assert condition.roadmap is None

    def assert_greedy_row(self) -> None:
        support = fitted("lc1", LC1_STOPPED_DRAW, "lc1_greedy_whole")["ate"].support
        assert support.row == "C01.greedy"
        patient = law.fit(sample("lc1", LC1_STOPPED_DRAW), "lc1_greedy_patience")["ate"].support
        assert patient.label == "supported"
        assert patient.row == "C12.greedy"
        assert PRE_ORDERED_TEXT not in patient.reason

    def test_a_greedy_fit_lands_on_its_c12_row(self) -> None:
        self.assert_greedy_row()

    def test_mutation_the_permanent_condition_on_greedy_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            _option_support,
            _option_support,
            "patience_conditions",
            [('== "greedy":', '== "never":')],
        )
        with pytest.raises(AssertionError):
            self.assert_greedy_row()

    def test_the_key_alone_keeps_a_greedy_patience_fit_off_its_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``C12`` the greedy patience fit matches no row, not ``C01.greedy``."""
        kept = tuple(row for row in support_module.registered_rows() if row.id != "C12.greedy")
        monkeypatch.setattr(support_module, "registered_rows", lambda: kept)
        support = law.fit(sample("lc1", LC1_STOPPED_DRAW), "lc1_greedy_patience")["ate"].support
        assert support.label == "experimental"
        assert support.row is None
