"""Independent finite-law checks for the controlled-direct-effect MAR contract."""

from __future__ import annotations

import itertools

import numpy as np

from cleverly import CausalStudy, ControlledDirectEffect, PointTreatment
from tests import discrete_law_cde


def _conditional_mean(cells: np.ndarray, outcome: int, given: dict[int, int]) -> float:
    """Return an exact empirical conditional mean from an equiprobable support table."""
    selected = np.ones(len(cells), dtype=bool)
    for column, value in given.items():
        selected &= cells[:, column] == value
    return float(np.mean(cells[selected, outcome]))


def _assert_pairwise_response_independence(cells: np.ndarray) -> None:
    """Assert Y and Z are each marginally independent of Delta."""
    for response in (0, 1):
        assert _conditional_mean(cells, 0, {2: response}) == 0.5
        assert _conditional_mean(cells, 1, {2: response}) == 0.5


def test_pairwise_missingness_claims_do_not_identify_the_cde_regression() -> None:
    """The XOR law satisfies both old clauses but violates MAR within Z."""
    # Columns are (Y, Z, Delta), with Delta = Y XOR Z. All four cells are equiprobable.
    xor = np.asarray(((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)), dtype=int)
    _assert_pairwise_response_independence(xor)
    for intermediate in (0, 1):
        assert _conditional_mean(xor, 0, {1: intermediate}) == 0.5
        observed = _conditional_mean(xor, 0, {1: intermediate, 2: 1})
        assert observed == float(1 - intermediate)
        assert observed != 0.5


def test_conditional_mar_control_identifies_the_observed_regression() -> None:
    """Three independent fair bits satisfy pairwise and conditional MAR."""
    independent = np.asarray(tuple(itertools.product((0, 1), repeat=3)), dtype=int)
    _assert_pairwise_response_independence(independent)
    for intermediate in (0, 1):
        assert _conditional_mean(independent, 0, {1: intermediate}) == 0.5
        for response in (0, 1):
            assert _conditional_mean(independent, 0, {1: intermediate, 2: response}) == 0.5


def test_identified_cde_states_both_response_restrictions() -> None:
    effect = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=0.0))

    assumptions = effect.identification.assumptions
    assert any("Y is independent of Delta given (A, Z, W)" in item for item in assumptions)
    assert any("Delta is independent of Z given (A, W)" in item for item in assumptions)
