"""A known rule fitted by the parametric g-formula is labelled by a row with a rule cell.

The g-formula evaluates a rule on simulated covariates, so each covariate model reaches the
estimate through the treatment the rule assigns, which no fixed plan exercises.  A rule is
therefore its own regimen kind on this engine (:func:`cleverly._support.regimen_kinds`):
``G29.g1`` and ``G29.g4`` list it because their study cells include a rule, and ``G29.g2_g5``,
whose cells are fixed plans, a stochastic policy and a tilt, does not.  A rule on the Gaussian
law then reads ``experimental``.  Estimators that follow a rule on observed histories keep the
``static`` key.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly._support import regimen_kinds
from cleverly.longitudinal import DynamicRegimen
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal.estimator import ParametricGFormula
from tests.studies import canonical_parametric_gformula as study

pytestmark = pytest.mark.xdist_group("gformula_rule_support")


def _b_rule(node: int) -> Any:
    column = f"B{node}"

    def rule(history: Any) -> Any:
        return np.asarray(history[column], dtype=float).astype(int)

    return rule


def _with_g2_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add a known rule that reads ``B_t`` to the plans of ``G2``."""
    declared = study.regimens

    def regimens(law: str, **kwargs: Any) -> dict[str, Any]:
        plans = declared(law, **kwargs)
        if law == "G2":
            plans["rule_b"] = DynamicRegimen(
                "rule_b", tuple(_b_rule(t) for t in (1, 2, 3)), rule_kind="known"
            )
        return plans

    monkeypatch.setattr(study, "regimens", regimens)


def _frame(law: str) -> pd.DataFrame:
    return study.law_of(law).sample(400, 0)


def _rows(result: Any) -> set[tuple[str, str | None]]:
    return {(estimate.support.label, estimate.support.row) for estimate in result.values()}


def test_a_rule_on_the_gaussian_law_reads_experimental(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_g2_rule(monkeypatch)
    result = study.fit("nice", "G2", _frame("G2"), n_draws=2)
    assert "ey_regimen[rule_b]" in result.estimates
    assert _rows(result) == {("experimental", None)}


def test_the_gaussian_law_without_a_rule_keeps_its_row() -> None:
    result = study.fit("nice", "G2", _frame("G2"), n_draws=2)
    assert _rows(result) == {("supported", "G29.g2_g5")}


@pytest.mark.parametrize(("law", "row"), [("G1", "G29.g1"), ("G4", "G29.g4")])
def test_a_law_with_a_rule_cell_covers_its_rule(law: str, row: str) -> None:
    result = study.fit("nice", law, _frame(law), n_draws=2)
    assert any("rule" in name for name in result.estimates)
    assert {support_row for _, support_row in _rows(result)} == {row}


def test_without_the_rule_kind_the_gaussian_rule_borrows_the_fixed_plan_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: classed as a fixed plan, the rule reads the row that never measured it."""
    _with_g2_rule(monkeypatch)
    monkeypatch.setattr(ParametricGFormula, "_simulated_rules", False)
    result = study.fit("nice", "G2", _frame("G2"), n_draws=2)
    assert _rows(result) == {("supported", "G29.g2_g5")}


def test_an_observed_history_rule_keeps_the_static_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Witness: LTMLE follows the G1 rule through its indicator, so it keys the rule as static."""
    seen: list[frozenset[str]] = []

    def spy(plans: Any, **kwargs: Any) -> frozenset[str]:
        kinds = regimen_kinds(plans, **kwargs)
        seen.append(kinds)
        return kinds

    monkeypatch.setattr(estimator_module, "regimen_kinds", spy)
    with_rule = study.fit("ltmle", "G1", study.law_of("G1").sample(2000, 0))
    assert "ey_regimen[rule]" in with_rule.estimates
    assert seen == [frozenset({"static"})]
    seen.clear()
    study.fit("nice", "G1", _frame("G1"), n_draws=2)
    assert seen and all("rule" in kinds and "static" in kinds for kinds in seen)
