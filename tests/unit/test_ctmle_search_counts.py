"""The work counts a C-TMLE selection record reports (X106 K2b).

``CTMLESelection.candidate_fits``, ``mechanism_fits`` and ``target_solves`` count, per search
branch, the candidate fluctuations, the candidate treatment mechanisms computed and the
fluctuation solves. They are checked against a hand count of the analysis branch and against
independent wrappers on every branch, and they leave every other output of the fit unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import CTMLE
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators.ctmle import PropensityArrays
from cleverly.learners import GLM

COVARIATES = ["W1", "W2", "W3", "W4"]


def _frame(n: int = 1000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 4))
    a = rng.binomial(1, expit(0.6 * w[:, 0] - 0.5 * w[:, 1]))
    y = rng.binomial(1, expit(0.3 * w[:, 0] + 0.4 * a))
    frame = pd.DataFrame(w, columns=COVARIATES)
    frame["A"] = a
    frame["Y"] = y
    return frame


def _fit(frame: pd.DataFrame, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "outcome_learner": GLM(family="binomial"),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 5,
        "estimands": ("ate",),
        **overrides,
    }
    return CTMLE(**settings).fit(frame, outcome="Y", treatment="A").single()


class _Wrapped:
    """Independent counts of the events, per selector in creation order and in total.

    The analysis-sample selector is created first, then one per selection fold in fold
    order, which is the order the record's tuples use.
    """

    def __init__(self) -> None:
        self.order: list[int] = []
        # Holding each selector keeps its id from being reused by a later one.
        self.selectors: list[Any] = []
        self.solves: dict[int, int] = {}
        self.fits: dict[int, int] = {}
        self.lasso: dict[int, int] = {}
        self.solve_calls = 0

    def per_branch(self, counts: dict[int, int]) -> tuple[int, ...]:
        return tuple(counts.get(key, 0) for key in self.order)


@pytest.fixture
def wrapped(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Wrapped]:
    counts = _Wrapped()
    selector = ctmle_module._Selector
    init = selector.__init__
    solve = ctmle_module.solve_submodel
    target = selector.target
    fit_with = selector._fit_propensity_with
    lasso = selector._lasso_propensities

    def registering_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        counts.order.append(id(self))
        counts.selectors.append(self)

    def counting_solve(*args: Any, **kwargs: Any) -> Any:
        counts.solve_calls += 1
        return solve(*args, **kwargs)

    def counting_target(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts.solves[id(self)] = counts.solves.get(id(self), 0) + 1
        return target(self, *args, **kwargs)

    def counting_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts.fits[id(self)] = counts.fits.get(id(self), 0) + 1
        return fit_with(self, *args, **kwargs)

    def counting_lasso(self: Any, train: Any, tag: str) -> Any:
        keys = [(tag, ("lasso", k)) for k in range(self.lambdas.size)]
        if not all(key in self._cache for key in keys):
            counts.lasso[id(self)] = counts.lasso.get(id(self), 0) + self.lambdas.size
        return lasso(self, train, tag)

    monkeypatch.setattr(selector, "__init__", registering_init)
    # `solve_submodel` is patched where ctmle.py holds it, so the estimator's final
    # targeting in tmle.py is not counted, as the record does not count it.
    monkeypatch.setattr(ctmle_module, "solve_submodel", counting_solve)
    monkeypatch.setattr(selector, "target", counting_target)
    monkeypatch.setattr(selector, "_fit_propensity_with", counting_fit)
    monkeypatch.setattr(selector, "_lasso_propensities", counting_lasso)
    yield counts


def _check_branches(selection: Any, counts: _Wrapped) -> None:
    """Every branch's solves and mechanisms equal the wrappers', in the record's order."""
    assert len(counts.order) == 1 + selection.folds.n_folds
    assert selection.target_solves == counts.per_branch(counts.solves)
    assert sum(selection.target_solves) == counts.solve_calls
    fits = counts.per_branch(counts.fits)
    lasso = counts.per_branch(counts.lasso)
    assert selection.mechanism_fits == tuple(a + b for a, b in zip(fits, lasso, strict=True))


def _hand_count(selection: Any) -> tuple[int, int]:
    """Candidates and mechanisms of a greedy analysis branch, read off its path.

    Position 0 is the intercept. Each later position scored every covariate still in the
    pool; a position reached after a retry (its ``n_steps`` one above the previous one)
    scored that pool twice, once before the retry and once after it. Mechanisms are the
    distinct covariate sets, each fitted once.
    """
    p = len(selection.covariates)
    candidates = 1
    mechanisms = 1
    for k in range(1, len(selection.path)):
        pool = p - (k - 1)
        retried = selection.n_steps[k] > selection.n_steps[k - 1]
        candidates += pool * (2 if retried else 1)
        mechanisms += pool
    return candidates, mechanisms


def test_greedy_counts_match_the_hand_count_and_the_wrappers(wrapped: _Wrapped) -> None:
    selection = _fit(_frame()).extra["ctmle"]
    n_branches = 1 + selection.folds.n_folds
    assert len(selection.candidate_fits) == len(selection.mechanism_fits) == n_branches
    # The fixture retries at least once on the analysis branch, so cache hits occur and
    # mechanisms are fewer than candidates there.
    assert any(b > a for a, b in zip(selection.n_steps, selection.n_steps[1:], strict=False))
    candidates, mechanisms = _hand_count(selection)
    assert selection.candidate_fits[0] == candidates
    assert selection.mechanism_fits[0] == mechanisms
    assert selection.mechanism_fits[0] < selection.candidate_fits[0]
    assert selection.target_solves == selection.candidate_fits
    _check_branches(selection, wrapped)
    # The folds differ, so a reversed or shuffled tuple would not match the wrappers.
    assert len(set(selection.candidate_fits)) > 1


def test_counting_a_cache_hit_as_a_fit_would_be_seen(wrapped: _Wrapped) -> None:
    """Mutation control: a count taken at every ``propensity`` call, hits included, differs."""
    calls = 0
    propensity = ctmle_module._Selector.propensity

    def every_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return propensity(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ctmle_module._Selector, "propensity", every_call)
        selection = _fit(_frame()).extra["ctmle"]
    assert calls != sum(selection.mechanism_fits)
    _check_branches(selection, wrapped)


def test_patience_counts_the_work_done(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), patience=1).extra["ctmle"]
    assert selection.stopped_at is not None
    _check_branches(selection, wrapped)


def test_lasso_counts_every_penalty_even_under_patience(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), strategy="lasso", lambdas=[0.2, 0.05, 0.01, 0.0], patience=1).extra[
        "ctmle"
    ]
    branches = 1 + selection.folds.n_folds
    # Every penalty of every branch is computed before position 0 is read.
    assert selection.mechanism_fits == (4,) * branches
    assert selection.stopped_at is not None
    assert selection.candidate_fits[0] > selection.stopped_at + 1
    assert selection.target_solves == selection.candidate_fits
    _check_branches(selection, wrapped)


def test_the_logistic_preorder_adds_solves_and_fits(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), strategy="ordered", preorder="logistic").extra["ctmle"]
    p = len(selection.covariates)
    for candidates, solves in zip(selection.candidate_fits, selection.target_solves, strict=True):
        assert solves == candidates + p
    _check_branches(selection, wrapped)


def test_discrete_counts_one_candidate_and_one_fit_per_declared_set(wrapped: _Wrapped) -> None:
    sets = [[], ["W1"], COVARIATES]
    selection = _fit(_frame(), strategy="discrete", candidates=sets).extra["ctmle"]
    assert set(selection.candidate_fits) == {len(sets)}
    assert set(selection.mechanism_fits) == {len(sets)}
    _check_branches(selection, wrapped)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_learners_leave_supplied_arrays_out_of_the_fits(wrapped: _Wrapped) -> None:
    frame = _frame()
    n = len(frame)
    values = np.full(n, 0.5)
    entries = [LogisticRegression(max_iter=1000), PropensityArrays(full=values, cv=values)]
    split = np.arange(n) % 5
    selection = _fit(
        frame,
        strategy="learners",
        treatment_candidates=entries,
        selection_split=split,
        cross_fit=False,
    ).extra["ctmle"]
    # The declared learner and the terminal treatment_learner are fitted; the arrays are read.
    assert set(selection.mechanism_fits) == {2}
    _check_branches(selection, wrapped)


def test_counting_changes_no_other_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the counters made inert, every other field of the record is bitwise the same."""
    frame = _frame()
    counted = _fit(frame).extra["ctmle"]

    class Inert(ctmle_module._SearchWork):
        def __setattr__(self, name: str, value: Any) -> None:
            if name in self.__dict__:
                return
            object.__setattr__(self, name, value)

    monkeypatch.setattr(ctmle_module, "_SearchWork", Inert)
    inert = _fit(frame).extra["ctmle"]
    assert set(inert.candidate_fits) == {0}
    for field in dataclasses.fields(counted):
        if field.name in {"candidate_fits", "mechanism_fits", "target_solves"}:
            continue
        a, b = getattr(counted, field.name), getattr(inert, field.name)
        if field.name == "folds":
            np.testing.assert_array_equal(a.assignment, b.assignment)
        elif isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes()
        else:
            assert a == b


def test_repeats_report_the_first_draw(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), repeats=2).extra["ctmle"]
    branches = 1 + selection.folds.n_folds
    # Two draws ran; the record holds the first draw's branches only.
    assert len(wrapped.order) == 2 * branches
    first = wrapped.order[:branches]
    assert selection.target_solves == tuple(wrapped.solves.get(key, 0) for key in first)


def test_summary_reports_the_search_work() -> None:
    selection = _fit(_frame()).extra["ctmle"]
    header = dict(selection.summary().header)
    assert header["Search work"] == (
        f"{sum(selection.candidate_fits)} candidate fits, {sum(selection.mechanism_fits)} "
        f"mechanism fits, {sum(selection.target_solves)} solves; "
        f"{selection.candidate_fits[0]} candidate fits on the analysis sample"
    )
    assert "Search work:" in str(selection.summary())
