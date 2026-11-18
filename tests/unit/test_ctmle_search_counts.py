"""The work counts a C-TMLE selection record reports (X106 K2b).

``CTMLESelection.candidates``, ``mechanism_fits`` and ``target_solves`` count, per search
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
    """Independent counts of the events, taken at the functions the search calls."""

    def __init__(self) -> None:
        self.solves = 0
        self.fits = 0
        self.lasso = 0


@pytest.fixture
def wrapped(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Wrapped]:
    counts = _Wrapped()
    solve = ctmle_module.solve_submodel
    fit_with = ctmle_module._Selector._fit_propensity_with
    lasso = ctmle_module._Selector._lasso_propensities

    def counting_solve(*args: Any, **kwargs: Any) -> Any:
        counts.solves += 1
        return solve(*args, **kwargs)

    def counting_fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts.fits += 1
        return fit_with(self, *args, **kwargs)

    def counting_lasso(self: Any, train: Any, tag: str) -> Any:
        keys = [(tag, ("lasso", k)) for k in range(self.lambdas.size)]
        if not all(key in self._cache for key in keys):
            counts.lasso += self.lambdas.size
        return lasso(self, train, tag)

    # `solve_submodel` is patched where ctmle.py holds it, so the estimator's final
    # targeting in tmle.py is not counted, as the record does not count it.
    monkeypatch.setattr(ctmle_module, "solve_submodel", counting_solve)
    monkeypatch.setattr(ctmle_module._Selector, "_fit_propensity_with", counting_fit)
    monkeypatch.setattr(ctmle_module._Selector, "_lasso_propensities", counting_lasso)
    yield counts


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
    assert len(selection.candidates) == len(selection.mechanism_fits) == n_branches
    # The fixture retries at least once on the analysis branch, so cache hits occur and
    # mechanisms are fewer than candidates there.
    assert any(b > a for a, b in zip(selection.n_steps, selection.n_steps[1:], strict=False))
    candidates, mechanisms = _hand_count(selection)
    assert selection.candidates[0] == candidates
    assert selection.mechanism_fits[0] == mechanisms
    assert selection.mechanism_fits[0] < selection.candidates[0]
    assert selection.target_solves == selection.candidates
    assert sum(selection.target_solves) == wrapped.solves
    assert sum(selection.mechanism_fits) == wrapped.fits


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
    assert sum(selection.mechanism_fits) == wrapped.fits


def test_patience_counts_the_work_done(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), patience=1).extra["ctmle"]
    assert selection.stopped_at is not None
    assert sum(selection.target_solves) == wrapped.solves
    assert sum(selection.mechanism_fits) == wrapped.fits


def test_lasso_counts_every_penalty_even_under_patience(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), strategy="lasso", lambdas=[0.2, 0.05, 0.01, 0.0], patience=1).extra[
        "ctmle"
    ]
    branches = 1 + selection.folds.n_folds
    # Every penalty of every branch is computed before position 0 is read.
    assert selection.mechanism_fits == (4,) * branches
    assert wrapped.lasso == 4 * branches
    assert selection.stopped_at is not None
    assert selection.candidates[0] > selection.stopped_at + 1
    assert sum(selection.target_solves) == wrapped.solves


def test_the_logistic_preorder_adds_solves_and_fits(wrapped: _Wrapped) -> None:
    selection = _fit(_frame(), strategy="ordered", preorder="logistic").extra["ctmle"]
    p = len(selection.covariates)
    for candidates, solves in zip(selection.candidates, selection.target_solves, strict=True):
        assert solves == candidates + p
    assert sum(selection.target_solves) == wrapped.solves
    assert sum(selection.mechanism_fits) == wrapped.fits


def test_discrete_counts_one_candidate_and_one_fit_per_declared_set(wrapped: _Wrapped) -> None:
    sets = [[], ["W1"], COVARIATES]
    selection = _fit(_frame(), strategy="discrete", candidates=sets).extra["ctmle"]
    assert set(selection.candidates) == {len(sets)}
    assert set(selection.mechanism_fits) == {len(sets)}
    assert sum(selection.mechanism_fits) == wrapped.fits


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
    assert sum(selection.mechanism_fits) == wrapped.fits


def test_counts_change_no_other_output() -> None:
    """Every other field of the record is what it would be without the counts."""
    selection = _fit(_frame()).extra["ctmle"]
    stripped = dataclasses.replace(selection, candidates=(), mechanism_fits=(), target_solves=())
    for field in dataclasses.fields(selection):
        if field.name in {"candidates", "mechanism_fits", "target_solves"}:
            continue
        a, b = getattr(selection, field.name), getattr(stripped, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a is b or a == b


def test_summary_reports_the_search_work() -> None:
    selection = _fit(_frame()).extra["ctmle"]
    text = str(selection.summary())
    assert "Search work:" in text
    total = sum(selection.candidates)
    assert f"{total} candidates" in " ".join(text.split())
