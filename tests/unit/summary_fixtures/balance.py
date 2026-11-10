"""Covariate-balance summaries the contract test checks (X92 V11).

A two-arm fit prints one table, a three-arm fit one per arm against the others. Fits are
compact and in sample with linear and logistic learners.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly._summary import vocabulary
from cleverly.datasets import make_linear_ate, make_multi_arm
from cleverly.sensitivity import CovariateBalance, covariate_balance

from . import Case
from ._check_fidelity import close, header_values, text_tables

__all__ = ["FIXTURES"]


def _balance(frame: Any, adjustment: tuple[str, ...]) -> CovariateBalance:
    # A treatment named in lower case, because the contract's header reader takes a capital
    # letter followed by a colon later in the line for the next label.
    frame = frame.rename(columns={"A": "arm"})
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="arm", adjustment=adjustment)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    return covariate_balance(result)


@functools.cache
def two_arms() -> CovariateBalance:
    """A two-arm fit with a confounded treatment."""
    return _balance(make_linear_ate(n=400, seed=71)[0], ("W1", "W2", "W3", "W4"))


@functools.cache
def three_arms() -> CovariateBalance:
    """Three arms: each against the others."""
    return _balance(make_multi_arm(n=400, seed=5)[0], ("W1", "W2", "W3"))


def read_balance(balance: CovariateBalance, text: str) -> None:
    """Assert that every number a balance summary prints equals the object's.

    Parameters
    ----------
    balance : CovariateBalance
        The balance.
    text : str
        Its short summary.
    """
    facts = header_values(text)
    assert facts[vocabulary.BALANCE_TREATMENT] == balance.treatment
    assert int(facts[vocabulary.ROWS].replace(",", "")) == balance.n
    covariates = {row.covariate for row in balance.rows}
    assert int(facts[vocabulary.BALANCE_COVARIATES]) == len(covariates)
    tables = text_tables(text)
    assert len(tables) == len(balance.contrasts)
    for contrast, (title, labels, rows) in zip(balance.contrasts, tables, strict=True):
        assert title == (contrast if len(balance.contrasts) > 1 else "")
        assert labels == list(vocabulary.BALANCE_COLUMNS)
        expected = sorted(balance.rows_of(contrast), key=lambda row: -abs(row.smd_unweighted))
        assert [cells[0] for cells in rows] == [row.covariate for row in expected]
        for cells, row in zip(rows, expected, strict=True):
            assert close(cells[1], row.smd_unweighted), (cells, row)
            assert close(cells[2], row.smd_weighted), (cells, row)


def _case(build: Any) -> Case:
    return Case(build, fidelity=read_balance)


FIXTURES = {
    "balance_two_arms": _case(two_arms),
    "balance_three_arms": _case(three_arms),
}
