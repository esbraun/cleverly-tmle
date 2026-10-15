"""Early stop on a C-TMLE path (part (g), ``patience=``).

``patience=p`` stops the cross-validated scoring of a candidate path, as R ``ctmle``'s
``ctmleDiscrete(patience=)`` does.  The counter keeps the smallest risk seen (the pointer).  A
risk above the pointer adds one to the count, and the search stops when the count reaches ``p``.
A risk at or below the pointer moves the pointer and resets the count.  The search reads the
full-sample path and every selection fold's path lazily and in step, so no position after the
stop is fitted, and the selection is the smallest risk among the scored positions.

* The counter equals a literal transcription of ``functions_discrete.R:600-650`` on stub risk
  sequences, ties included.  Two mutation controls each fail it: ``>=`` for ``>``, and no reset
  on a tie.
* ``_cross_validate_patient`` on a stub path stops at the transcribed index, reads no later
  position in any branch, and returns the risks of the scored prefix.
* On real fits the lazy path equals the eager path at every position when ``p`` exceeds the
  path, for every strategy that selects a path position by cross-validated risk.
* A stopped real fit is the prefix of the whole-path fit, and a law whose global minimum lies
  past the stop changes ``selected`` and the estimate.
* R-g1 and R-g2 refuse a malformed ``patience`` and a strategy that scores no path.
"""

from __future__ import annotations

import importlib
import re
import types
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly import CollaborativeTMLEMethod
from cleverly.estimators import CTMLE
from cleverly.estimators.ctmle import CTMLESelection, _PatienceCounter, _Selector
from tests.unit._source_mutation import mutate
from tests.unit.test_treatment_covariates_collaborative import COVARIATES, lasso_frame

ctmle_module = importlib.import_module("cleverly.estimators.ctmle")

R_G1 = "patience must be a positive integer; got {value!r}"
R_G2 = (
    "patience= stops the cross-validated scoring of a candidate path, and strategy={strategy!r} "
    "scores no path. It applies to 'greedy', 'ordered', 'discrete', 'lasso' and 'learners'"
)


# ----------------------------------------------------------------- the transcribed counter


def r_stop(scores: Sequence[float], patience: int) -> int | None:
    """``functions_discrete.R:600-650``, the ``if(patience)`` block, one score at a time.

    The R loop runs over the finite scores of a pass.  Here one pass holds the whole sequence,
    so ``pointer`` starts at ``score[1]`` and ``patience.count`` at 0, and the loop over ``j``
    is the body below.  Returns the 0-based index of the score at which ``DONE`` first becomes
    true through the count, or ``None``.  ``ctmle`` finishes the rest of its batch after that
    score and takes the last tied minimum (declared differences D7b and D7).
    """
    pointer = scores[0]
    count = 0
    done = False
    for j, score in enumerate(scores):
        if score > pointer:
            count += 1
        if count >= patience:
            done = True
        if score <= pointer and not done:
            pointer = score
            count = 0
        if done:
            return j
    return None


def counter_stop(scores: Sequence[float], patience: int) -> int | None:
    counter = _PatienceCounter(patience)
    for index, score in enumerate(scores):
        if counter.offer(score):
            return index
    return None


#: Sequences with ties, a dip after the count has started, and a reset in the middle.
SEQUENCES: list[list[float]] = [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 1.0, 2.0, 2.0],
    [3.0, 2.0, 1.0, 2.0, 3.0, 0.5],
    [2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
    [2.0, 3.0, 1.0, 3.0, 3.0, 0.0],
    [1.0, 2.0, 2.0, 0.5, 9.0],
    [5.0],
    [1.0, float("inf"), float("nan"), 0.5, float("inf")],
]


@pytest.mark.parametrize("patience", [1, 2, 3, 4])
@pytest.mark.parametrize("scores", SEQUENCES, ids=lambda s: "-".join(f"{v:g}" for v in s))
def test_the_counter_equals_the_transcribed_loop(scores: list[float], patience: int) -> None:
    clean = [v if np.isfinite(v) else np.inf for v in scores]
    assert counter_stop(scores, patience) == r_stop(clean, patience)


def test_the_counter_equals_the_transcribed_loop_on_random_tied_sequences() -> None:
    rng = np.random.default_rng(3)
    stopped = 0
    for _ in range(400):
        scores = rng.integers(0, 5, size=int(rng.integers(1, 12))).astype(float).tolist()
        patience = int(rng.integers(1, 5))
        expected = r_stop(scores, patience)
        assert counter_stop(scores, patience) == expected
        stopped += expected is not None
    assert 100 < stopped < 400


def test_the_count_is_not_a_run_length() -> None:
    """A better-than-pointer risk resets, and a tie does too; a worse one after it adds one."""
    assert counter_stop([1.0, 2.0, 1.0, 2.0], 2) is None
    assert counter_stop([1.0, 2.0, 3.0], 2) == 2
    assert counter_stop([2.0, 3.0, 2.0, 3.0, 3.0], 2) == 4
    assert counter_stop([1.0, 1.0, 1.0, 1.0], 1) is None


def test_the_first_position_sets_the_pointer_and_never_counts() -> None:
    for first in (0.0, 5.0, np.inf, np.nan):
        counter = _PatienceCounter(1)
        assert counter.offer(first) is False
        assert counter.count == 0


def test_a_risk_that_is_not_finite_counts_as_worse() -> None:
    assert counter_stop([1.0, float("nan")], 1) == 1
    assert counter_stop([1.0, float("inf"), 0.5], 2) is None


#: A sequence where the two mutants stop at an index the transcription does not.
GE_WITNESS = ([1.0, 1.0, 1.0, 5.0], 2)
TIE_WITNESS = ([2.0, 3.0, 2.0, 3.0], 2)


def witness() -> None:
    for scores, patience in (GE_WITNESS, TIE_WITNESS, *((s, 2) for s in SEQUENCES)):
        clean = [v if np.isfinite(v) else np.inf for v in scores]
        assert counter_stop(scores, patience) == r_stop(clean, patience)


def test_mutation_ge_for_gt_fails_the_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        _PatienceCounter,
        "offer",
        [("if value > self.pointer:", "if value >= self.pointer:")],
    )
    with pytest.raises(AssertionError):
        witness()


def test_mutation_no_reset_on_a_tie_fails_the_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    witness()
    mutate(
        monkeypatch,
        ctmle_module,
        _PatienceCounter,
        "offer",
        [
            (
                "        self.pointer = value\n        self.count = 0\n",
                "        if value < self.pointer:\n            self.count = 0\n"
                "        self.pointer = value\n",
            )
        ],
    )
    with pytest.raises(AssertionError):
        witness()


# ------------------------------------------------------------------ a stub path, no fits


def stub_selector() -> _Selector:
    """A real selector on a small sample, two selection folds, a mean outcome regression."""
    estimator = CTMLE(
        strategy="discrete",
        candidates=[["c1"]],
        outcome_learner=DummyRegressor(),
        treatment_learner=LogisticRegression(max_iter=500),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
        estimands=("ate",),
        q_bounds=(-20.0, 20.0),
        selection_folds=2,
        penalty=False,
    )
    data = estimator._prepare(
        lasso_frame(n=120),
        outcome="Y",
        treatment="A",
        covariates=COVARIATES,
        delta=None,
        weights=None,
        id=None,
        intermediate=None,
    )
    scaler = estimator._scaler(data)
    folds = estimator._folds(data)
    config = estimator._config(data, ("ate",), scaler, folds)
    base = estimator._fit_nuisances(data, folds, scaler, None)
    return _Selector(estimator, data, base, config.g_bounds, None)


class Reads:
    """How many positions each branch of a stub path was asked for."""

    def __init__(self) -> None:
        self.by_branch: dict[str, int] = {}


def install_stub(
    monkeypatch: pytest.MonkeyPatch,
    table: list[float],
    *,
    fold_lengths: dict[str, int] | None = None,
) -> Reads:
    """Replace the path, the loss and the influence of every selector by a table of risks.

    A branch's position ``k`` holds ``k`` in ``targeted``.  The loss of a fold is half the
    table entry, so the two folds sum to it exactly, and the influence is zero.
    """
    reads = Reads()

    def iter_path(self: Any, train: Any, tag: str) -> Any:
        limit = (fold_lengths or {}).get(tag, len(table))
        for index in range(limit):
            reads.by_branch[tag] = index + 1
            yield types.SimpleNamespace(targeted=index, submodel=None, tag=tag)

    def loss(self: Any, targeted: int, rows: Any) -> float:
        return table[targeted] / 2.0

    def influence(self: Any, targeted: int, submodel: Any, rows: Any) -> Any:
        return np.zeros((len(rows), len(self.target_names)))

    monkeypatch.setattr(_Selector, "iter_path", iter_path)
    monkeypatch.setattr(_Selector, "loss", loss)
    monkeypatch.setattr(_Selector, "influence", influence)
    return reads


#: Dyadic risks, so the half-loss of each fold sums exactly.  The global minimum is last.
TABLE = [4.0, 4.5, 5.0, 4.0, 6.0, 1.0]


@pytest.mark.parametrize("patience", [1, 2, 3, 4, 5, 10])
def test_the_stub_search_stops_where_the_transcribed_counter_stops(
    patience: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    selector = stub_selector()
    reads = install_stub(monkeypatch, TABLE)
    path, risk, folds, stopped_at = selector._cross_validate_patient(patience)
    expected = r_stop(TABLE, patience)
    assert stopped_at == expected
    scored = len(TABLE) if expected is None else expected + 1
    assert len(path) == scored
    np.testing.assert_array_equal(risk, np.array(TABLE[:scored]))
    # The search read exactly the scored positions in the full-sample path and in each fold.
    assert reads.by_branch == {"full": scored, "cv0": scored, "cv1": scored}
    assert folds.n_folds == 2


def test_a_global_minimum_past_the_stop_is_not_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    selector = stub_selector()
    install_stub(monkeypatch, TABLE)
    _, whole, _, never = selector._cross_validate_patient(100)
    _, stopped, _, at = selector._cross_validate_patient(2)
    assert never is None and at == 2
    assert int(np.argmin(whole)) == 5
    assert int(np.argmin(stopped)) == 0
    assert TABLE[5] < TABLE[0] - 1.0


def test_a_selection_fold_path_that_ends_early_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    selector = stub_selector()
    install_stub(monkeypatch, TABLE, fold_lengths={"cv1": 2})
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "selection fold 1 produced 2 candidates but the full-sample search produced at "
            "least 3; the candidate sequence must have the same length in every fold"
        ),
    ):
        selector._cross_validate_patient(100)


def test_a_fold_path_longer_than_the_full_path_is_not_read_past_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = stub_selector()
    reads = install_stub(monkeypatch, TABLE, fold_lengths={"cv0": 9, "cv1": 9, "full": 4})
    path, _, _, stopped_at = selector._cross_validate_patient(100)
    assert stopped_at is None and len(path) == 4
    assert reads.by_branch["cv0"] == 4


# ------------------------------------------------------------------------- real fits


def make(**settings: Any) -> CTMLE:
    options: dict[str, Any] = {
        "outcome_learner": DummyRegressor(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ate",),
        "q_bounds": (-20.0, 20.0),
        "selection_folds": 3,
    }
    options.update(settings)
    return CTMLE(**options)


def fit(**settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            make(**settings)
            .fit(lasso_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
            .single()
        )


STRATEGIES: dict[str, dict[str, Any]] = {
    "greedy": {"strategy": "greedy"},
    "ordered": {"strategy": "ordered", "ordering": ["c4", "c3", "c2", "c1"]},
    "discrete": {
        "strategy": "discrete",
        "candidates": [["c4"], ["c3"], ["c3", "c4"], ["c1"], ["c1", "c2"]],
    },
    "lasso": {"strategy": "lasso", "lambdas": (0.2, 0.05, 0.02, 0.005)},
    "learners": {
        "strategy": "learners",
        "treatment_candidates": [LogisticRegression(C=0.001), LogisticRegression(C=0.05)],
    },
}

PER_POSITION = ("n_steps", "train_risk", "train_loss", "penalty", "treatment_risk", "cv_risk")


def same_position(left: CTMLESelection, right: CTMLESelection, count: int) -> None:
    """Every per-position field of the first ``count`` positions is bit-equal."""
    assert left.path[:count] == right.path[:count]
    assert left.labels[:count] == right.labels[:count]
    for name in PER_POSITION:
        np.testing.assert_array_equal(
            np.asarray(getattr(left, name))[:count], np.asarray(getattr(right, name))[:count]
        )


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_a_patience_beyond_the_path_equals_the_whole_path_fit(name: str) -> None:
    """The lazy full-sample and fold paths equal the eager ones at every position."""
    whole = fit(**STRATEGIES[name])
    lazy = fit(patience=10**6, **STRATEGIES[name])
    selection, reference = lazy.extra["ctmle"], whole.extra["ctmle"]
    assert selection.patience == 10**6 and selection.stopped_at is None
    assert reference.patience is None and reference.stopped_at is None
    assert len(selection.path) == len(reference.path) >= 3
    same_position(selection, reference, len(reference.path))
    assert selection.selected == reference.selected and selection.final == reference.final
    assert selection.folds.assignment.tolist() == reference.folds.assignment.tolist()
    assert lazy.estimates["ate"].psi == whole.estimates["ate"].psi
    np.testing.assert_array_equal(lazy.nuisance.propensity.values, whole.nuisance.propensity.values)


@pytest.mark.parametrize("name", list(STRATEGIES))
@pytest.mark.parametrize("patience", [1, 2])
def test_a_stopped_fit_is_the_prefix_of_the_whole_path_fit(name: str, patience: int) -> None:
    whole = fit(**STRATEGIES[name]).extra["ctmle"]
    expected = r_stop(whole.cv_risk.tolist(), patience)
    result = fit(patience=patience, **STRATEGIES[name])
    selection = result.extra["ctmle"]
    assert selection.stopped_at == expected
    count = len(whole.path) if expected is None else expected + 1
    assert len(selection.path) == count == len(selection.cv_risk)
    same_position(selection, whole, count)
    assert selection.selected == int(np.argmin(whole.cv_risk[:count]))
    assert selection.selected < count
    assert result.inference_status == "working_mechanism_plugin"


def test_at_least_one_strategy_stops_early_at_patience_one() -> None:
    stops = {
        name: fit(patience=1, **settings).extra["ctmle"].stopped_at
        for name, settings in STRATEGIES.items()
    }
    assert any(stop is not None for stop in stops.values()), stops


#: The law of the witness: ``c4`` is noise, ``c1`` drives the treatment so strongly that every
#: candidate holding it has a larger risk, and the last candidate has the smallest risk of all.
WITNESS = {
    "strategy": "discrete",
    "candidates": [["c4"], ["c1"], ["c1", "c2"], ["c2", "c3", "c4"]],
}


def test_a_law_whose_global_minimum_lies_past_the_stop_changes_the_selection() -> None:
    whole = fit(**WITNESS)
    stopped = fit(patience=2, **WITNESS)
    reference, selection = whole.extra["ctmle"], stopped.extra["ctmle"]
    assert reference.selected == 3
    assert selection.stopped_at == 2 and selection.selected == 0
    risk = reference.cv_risk
    assert risk[1] > risk[0] and risk[2] > risk[0]
    assert risk[3] < risk[0] - 1e-2
    assert abs(stopped.estimates["ate"].psi - whole.estimates["ate"].psi) > 1e-3
    assert selection.selected_covariates == ("c4",)


def test_the_stopped_summary_and_description_say_where_the_search_stopped() -> None:
    selection = fit(patience=2, **WITNESS).extra["ctmle"]
    lines = selection.summary().splitlines()
    assert (
        "patience = 2: the counter stopped the search after candidate 3, and later "
        "candidates were not fitted or scored"
    ) in lines
    assert selection.describe() == (
        "C-TMLE discrete selected candidate 1 of 3 for ate; patience=2 stopped the search "
        "after candidate 3"
    )
    whole = fit(patience=100, **WITNESS).extra["ctmle"]
    assert (
        "patience = 100: the counter never triggered, so the search read every candidate"
    ) in whole.summary().splitlines()
    assert "patience" not in fit(**WITNESS).extra["ctmle"].summary()


def test_the_lasso_final_step_reads_only_the_stopped_path() -> None:
    stopped = fit(patience=1, **STRATEGIES["lasso"]).extra["ctmle"]
    assert stopped.stopped_at is not None
    assert len(stopped.path) == stopped.stopped_at + 1
    assert stopped.final is None or stopped.final <= stopped.stopped_at


def test_the_default_records_no_patience() -> None:
    selection = fit(**STRATEGIES["greedy"]).extra["ctmle"]
    assert selection.patience is None and selection.stopped_at is None


# ------------------------------------------------------------------- the method route


def test_the_method_passes_patience_to_the_engine() -> None:
    method = CollaborativeTMLEMethod(strategy="greedy", patience=3)
    assert method.estimator_kwargs()["patience"] == 3
    assert CollaborativeTMLEMethod().estimator_kwargs()["patience"] is None
    estimator = CTMLE(**method.estimator_kwargs())
    assert estimator.patience == 3


def test_the_method_route_refuses_what_the_engine_refuses() -> None:
    with pytest.raises(ValueError, match=re.escape(R_G1.format(value=0))):
        CTMLE(**CollaborativeTMLEMethod(patience=0).estimator_kwargs())
    with pytest.raises(ValueError, match=re.escape(R_G2.format(strategy="oat"))):
        CTMLE(**CollaborativeTMLEMethod(strategy="oat", patience=2).estimator_kwargs())


# --------------------------------------------------------------------------- R-g1, R-g2


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, "2", 2.0, np.float64(2)])
def test_r_g1_a_patience_that_is_not_a_positive_integer_is_refused(value: Any) -> None:
    with pytest.raises(ValueError, match=re.escape(R_G1.format(value=value))):
        CTMLE(strategy="greedy", patience=value)


@pytest.mark.parametrize("value", [1, 2, np.int64(3)])
def test_a_positive_integer_is_accepted(value: Any) -> None:
    assert CTMLE(strategy="greedy", patience=value).patience == value


@pytest.mark.parametrize(
    "settings",
    [
        {"strategy": "oat"},
        {"strategy": "derivative", "lambdas": (0.05, 0.02)},
    ],
    ids=["oat", "derivative"],
)
def test_r_g2_a_strategy_that_scores_no_path_is_refused(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=re.escape(R_G2.format(strategy=settings["strategy"]))):
        CTMLE(patience=2, **settings)


def test_r_g2_does_not_fire_without_a_patience() -> None:
    CTMLE(strategy="oat")
    CTMLE(strategy="derivative", lambdas=(0.05, 0.02))


def test_r_g1_comes_before_r_g2() -> None:
    with pytest.raises(ValueError, match=re.escape(R_G1.format(value=0))):
        CTMLE(strategy="oat", patience=0)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_selector_strategy_accepts_a_patience(name: str) -> None:
    assert make(patience=2, **STRATEGIES[name]).patience == 2
