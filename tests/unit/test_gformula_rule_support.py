"""A known rule fitted by the parametric g-formula is labelled by a row with a rule cell.

The g-formula evaluates a rule on simulated covariates, so each covariate model reaches the
estimate through the treatment the rule assigns, which no fixed plan exercises.  A rule is
therefore its own regimen kind on this engine (:func:`cleverly._support.regimen_kinds`), and a
row lists it only when its study cells include a rule: ``G29.g1``, ``G29.g4`` and ``G29.g2_g5``
(the rule cells of ``G2`` and ``G5``).  ``G29.g3`` measures fixed plans only, so a rule on the
competing-risks law reads ``experimental``.  Estimators that follow a rule on observed histories
keep the ``static`` key.
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


def _l_rule(node: int) -> Any:
    column = f"L{node}"

    def rule(history: Any) -> Any:
        return np.asarray(history[column], dtype=float).astype(int)

    return rule


def _with_g3_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Add a known rule that reads ``L_t`` to the plans of ``G3``."""
    declared = study.regimens

    def regimens(law: str, **kwargs: Any) -> dict[str, Any]:
        plans = declared(law, **kwargs)
        if law == "G3":
            plans["rule_l"] = DynamicRegimen(
                "rule_l", tuple(_l_rule(t) for t in (1, 2, 3, 4)), rule_kind="known"
            )
        return plans

    monkeypatch.setattr(study, "regimens", regimens)


def _frame(law: str) -> pd.DataFrame:
    return study.law_of(law).sample(400, 0)


def _rows(result: Any) -> set[tuple[str, str | None]]:
    return {(estimate.support.label, estimate.support.row) for estimate in result.values()}


@pytest.mark.parametrize("law", ["G2", "G5"])
def test_a_rule_on_the_continuous_laws_reads_their_row(law: str) -> None:
    result = study.fit("nice", law, _frame(law), n_draws=2, rule_fit=True)
    assert "ey_regimen[rule]" in result.estimates
    assert _rows(result) == {("supported", "G29.g2_g5")}


def test_a_rule_on_the_competing_risks_law_reads_experimental(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_g3_rule(monkeypatch)
    result = study.fit("nice", "G3", _frame("G3"), n_draws=2)
    assert any("rule_l" in name for name in result.estimates)
    assert _rows(result) == {("experimental", None)}


def test_the_competing_risks_law_without_a_rule_keeps_its_row() -> None:
    result = study.fit("nice", "G3", _frame("G3"), n_draws=2)
    assert _rows(result) == {("supported", "G29.g3")}


@pytest.mark.parametrize(("law", "row"), [("G1", "G29.g1"), ("G4", "G29.g4")])
def test_a_law_with_a_rule_cell_covers_its_rule(law: str, row: str) -> None:
    result = study.fit("nice", law, _frame(law), n_draws=2)
    assert any("rule" in name for name in result.estimates)
    assert {support_row for _, support_row in _rows(result)} == {row}


def test_without_the_rule_kind_the_competing_risks_rule_borrows_the_fixed_plan_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: classed as a fixed plan, the rule reads the row that never measured it."""
    _with_g3_rule(monkeypatch)
    monkeypatch.setattr(ParametricGFormula, "_simulated_rules", False)
    result = study.fit("nice", "G3", _frame("G3"), n_draws=2)
    assert _rows(result) == {("supported", "G29.g3")}


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
