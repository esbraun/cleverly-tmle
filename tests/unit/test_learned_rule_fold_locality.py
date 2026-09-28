"""No learned rule at a row comes from a fit that saw that row (roadmap row RM30).

The rule of fold ``v`` is the plug-in rule of the outcome regression fitted on the
training complement of ``v``.  A row spy records the training rows of every outcome fit,
and the rule at each validation row must come from a fit whose rows exclude it.  A
longhand refit on each complement must reproduce the rule row by row.  The tie witness
pins the direction of the threshold: a blip of exactly zero assigns control.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tests.unit import _learned_rule_support as support


@pytest.fixture(scope="module")
def spied() -> tuple[Any, list[np.ndarray]]:
    support.RowSpy.fits = []
    result = support.fit(outcome_learner=support.RowSpy())
    return result, list(support.RowSpy.fits)


def test_every_rule_comes_from_a_fit_that_never_saw_its_row(spied: Any) -> None:
    result, fits = spied
    w1 = np.asarray(result.data.covariates[:, 0], dtype=float)
    assert all(rows.size < result.data.n for rows in fits), "no outcome fit may read every row"
    for train, test in result.nuisance.folds:
        matching = [rows for rows in fits if np.array_equal(np.sort(rows), np.sort(w1[train]))]
        assert len(matching) == 1, "each fold's rule comes from one fit on its complement"
        assert not np.isin(w1[test], matching[0]).any()


def test_the_rule_is_the_longhand_rule_of_each_complement(spied: Any) -> None:
    result, _ = spied
    data = result.data
    design = np.column_stack([data.treatment, data.covariates])
    outcome = np.asarray(data.outcome, dtype=float)
    rule = result.nuisance.regimes.values[:, 1, 0]
    for train, test in result.nuisance.folds:
        model = support.outcome_learner().fit(design[train], outcome[train])
        blip = support.blip(model, data.covariates[test])
        np.testing.assert_array_equal(rule[test], (blip > 0.0).astype(float))
    assert 0.0 < float(np.mean(rule)) < 1.0


def test_control_a_rule_read_from_an_all_row_fit_differs(spied: Any) -> None:
    """The all-row rule disagrees with the fold rules at some row, so the checks above see
    a rule learned on the rows it is evaluated on."""
    result, _ = spied
    data = result.data
    design = np.column_stack([data.treatment, data.covariates])
    model = support.outcome_learner().fit(design, np.asarray(data.outcome, dtype=float))
    everywhere = (support.blip(model, data.covariates) > 0.0).astype(float)
    assert not np.array_equal(everywhere, result.nuisance.regimes.values[:, 1, 0])


def test_a_tie_assigns_control() -> None:
    """An outcome regression that ignores the treatment has a blip of exactly zero."""
    blind = Pipeline(
        [
            ("drop", ColumnTransformer([("keep", "passthrough", [1, 2])])),
            ("fit", LogisticRegression(max_iter=1000)),
        ]
    )
    result = support.fit(outcome_learner=blind)
    values = result.nuisance.regimes.values
    np.testing.assert_array_equal(values[:, 1, 0], np.zeros(result.data.n))
    assert result.extra["learned_rule"].treated_shares == (0.0,) * 5
