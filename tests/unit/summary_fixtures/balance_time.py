"""Covariate-balance-over-time summaries the contract test checks (X103).

One longitudinal fit (with the feedback check), one survival-family fit (feedback omitted),
one continuous-time survival fit, and one collaborative survival fit, which carries the
note that its weights come from a selected propensity model.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

from cleverly._summary import vocabulary
from cleverly._summary._balance_time import summary_rows
from cleverly.sensitivity import BalanceOverTime, covariate_balance

from .. import _capability_sweep_support as sweep
from . import Case
from ._check_fidelity import close, header_values, text_tables

__all__ = ["FIXTURES"]


def _balance(fit: Any) -> BalanceOverTime:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balance = covariate_balance(fit())
    assert isinstance(balance, BalanceOverTime)
    return balance


@functools.cache
def longitudinal() -> BalanceOverTime:
    """A longitudinal fit: unweighted, weighted and feedback checks."""
    return _balance(sweep.fit_ltmle)


@functools.cache
def survival() -> BalanceOverTime:
    """A discrete-time survival fit: feedback is omitted with its reason."""
    return _balance(sweep.fit_hazard_multinode)


@functools.cache
def continuous() -> BalanceOverTime:
    """A continuous-time survival fit: confounding, joint and censoring rows."""
    return _balance(sweep.fit_continuous_survival)


@functools.cache
def collaborative() -> BalanceOverTime:
    """A collaborative survival fit, whose weights come from a selected model."""
    return _balance(sweep.fit_survival_ctmle)


def read_balance_time(balance: BalanceOverTime, text: str) -> None:
    """Assert that every number a balance-over-time summary prints equals the object's.

    Parameters
    ----------
    balance : BalanceOverTime
        The balance.
    text : str
        Its short summary.
    """
    facts = header_values(text)
    assert facts[vocabulary.BALANCE_TIME_TREATMENTS] == ", ".join(balance.treatments)
    assert facts[vocabulary.BALANCE_TIME_DIAGNOSTICS] == ", ".join(balance.diagnostics)
    assert int(facts[vocabulary.ROWS].replace(",", "")) == balance.n
    tables = text_tables(text)
    assert len(tables) == len(balance.diagnostics)
    for diagnostic, (title, labels, rows) in zip(balance.diagnostics, tables, strict=True):
        assert title == vocabulary.BALANCE_TIME_DIAGNOSTIC_TITLES[diagnostic]
        assert labels == list(vocabulary.BALANCE_TIME_COLUMNS)
        expected = summary_rows(balance, diagnostic)
        assert len(rows) == len(expected)
        for cells, group in zip(rows, expected, strict=True):
            assert cells[1] == group["covariate"], (cells, group)
            assert close(cells[2], group["smd_unweighted"]), (cells, group)
            assert close(cells[3], group["smd_weighted"]), (cells, group)


def _case(build: Any, names: tuple[str, ...]) -> Case:
    return Case(build, names=names, fidelity=read_balance_time)


FIXTURES = {
    "balance_time_longitudinal": _case(longitudinal, ("A1", "A2", "L2", "W1", "W2")),
    "balance_time_survival": _case(survival, ("A1", "A2", "W1", "W2")),
    "balance_time_continuous": _case(continuous, ("A", "W")),
    "balance_time_collaborative": _case(collaborative, ("A", "W1", "W2")),
}
