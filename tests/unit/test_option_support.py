"""The support label of a fit that declares a targeting or fitting option.

``patience=`` runs, and its estimates carry the experimental label with roadmap X76 as the
evidence (``cleverly.estimators._option_support``).  A declared stopping rule,
``outcome_by_arm=True`` and ``treatment_adjustment=`` have registered rows (``T52`` and ``G55``,
``T53``, ``T54``); the key routes a fit that declares one, and a fit outside those rows matches no
row.  Under a declared rule the key records ``step_bound=`` and ``iterate=`` too, and no row
declares them.  ``iterate="best"`` and ``step_bound=`` under ``stopping="relative"`` keep the
label of the fit without them.  The witness for each row is the same fit with the row removed
from the table, and a mutation that removes the key's stopping options returns the default row.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import pytest

from cleverly import _support as support_module
from cleverly._summary._estimates import plain_support_reason
from cleverly.estimators._option_support import OPTION_STUDIES_ROADMAP
from tests.unit import test_patience as patience_module
from tests.unit import test_stopping_rule_engines as engines
from tests.unit import test_stopping_rule_longitudinal as longitudinal

REGISTERED_ROWS = support_module.registered_rows()


def point(**settings: Any) -> Any:
    frame, columns, base, _ = engines.KINDS["arm-means"]()
    stopping = settings.pop("stopping", "relative")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engines._fit(stopping, frame, columns={**columns}, **base, **settings)


def labels(result: Any) -> set[str]:
    return {estimate.support.label for estimate in result.estimates.values()}


def evidence(result: Any) -> set[str]:
    return {item for e in result.estimates.values() for item in e.support.evidence}


def test_the_default_fit_is_supported() -> None:
    # The witness: a registered row covers the fit without any option.
    assert labels(point()) == {"supported"}


#: The estimands of the ``option-stopping-point`` fits: the ``P1`` family of the ``T52`` rows.
RULE_ESTIMANDS = ("ey1", "ey0", "ate")


def rows(result: Any) -> set[str | None]:
    return {estimate.support.row for estimate in result.estimates.values()}


def _without_rows(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    kept = tuple(row for row in REGISTERED_ROWS if not row.id.startswith(prefix))
    monkeypatch.setattr(support_module, "registered_rows", lambda: kept)


@pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
@pytest.mark.parametrize(("targeting", "walk"), [("iterative", "newton"), ("one_step", "one_step")])
def test_a_declared_stopping_rule_lands_on_its_t52_row(
    rule: str, targeting: str, walk: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = point(stopping=rule, targeting=targeting, estimands=RULE_ESTIMANDS)
    assert rows(result) == {f"T52.{rule}.{walk}.continuous"}
    assert labels(result) == {"supported"}
    assert OPTION_STUDIES_ROADMAP not in evidence(result)
    # The witness: without the T52 rows the same fit matches no row.
    _without_rows(monkeypatch, "T52.")
    stripped = point(stopping=rule, targeting=targeting, estimands=RULE_ESTIMANDS)
    assert labels(stripped) == {"experimental"} and rows(stripped) == {None}
    reason = stripped.estimates["ate"].support.reason
    assert f"stopping={rule!r}" in reason
    assert "experimental" in stripped.summary().note_codes
    settings = stripped.summary(detail="full").section("Settings").lines
    assert any(plain_support_reason(reason) in line for line in settings)


@pytest.mark.parametrize("settings", [{"step_bound": 0.25, "max_iter": 500}, {"iterate": "best"}])
def test_a_step_bound_or_best_iterate_under_a_rule_matches_no_row(
    settings: dict[str, Any],
) -> None:
    result = point(stopping="scaled_var", estimands=RULE_ESTIMANDS, **settings)
    assert labels(result) == {"experimental"} and rows(result) == {None}
    assert OPTION_STUDIES_ROADMAP not in evidence(result)


@pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
def test_a_rule_fit_outside_the_t52_family_matches_no_row(rule: str) -> None:
    # The fixture asks for the ATT and ATC too (family P3), which T52 does not cover.
    result = point(stopping=rule)
    assert labels(result) == {"experimental"} and rows(result) == {None}
    assert OPTION_STUDIES_ROADMAP not in evidence(result)


def test_outcome_by_arm_carries_no_condition() -> None:
    # The fixture asks for the ATT and ATC too (family P3), which T53 does not cover, so the
    # fit matches no row; no X76 condition fires.  The study fits land on T53
    # (tests/unit/test_option_outcome_by_arm_witness.py).
    result = point(outcome_by_arm=True)
    assert labels(result) == {"experimental"}
    assert OPTION_STUDIES_ROADMAP not in evidence(result)
    assert {estimate.support.row for estimate in result.estimates.values()} == {None}


def test_a_treatment_restriction_carries_no_condition() -> None:
    frame, _, _, _ = engines.KINDS["arm-means"]()
    from cleverly.estimators import TMLE
    from tests.conftest import linear_in_sample

    covariates = [c for c in frame.columns if c.startswith("W")]
    result = (
        TMLE(**linear_in_sample())
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=covariates,
            treatment_covariates=covariates[:1],
        )
        .single()
    )
    # As above: the ATT and ATC leave T54, whose study fits land on it
    # (tests/unit/test_option_adjustment_witness.py).
    assert labels(result) == {"experimental"}
    assert OPTION_STUDIES_ROADMAP not in evidence(result)
    assert {estimate.support.row for estimate in result.estimates.values()} == {None}


def test_patience_is_experimental_and_the_whole_path_keeps_its_label() -> None:
    whole = patience_module.fit()
    patient = patience_module.fit(patience=2)
    assert "experimental" in labels(patient)
    assert any("patience=2" in e.support.reason for e in patient.estimates.values())
    assert all("patience" not in e.support.reason for e in whole.estimates.values())


@pytest.mark.parametrize(
    "settings",
    [
        {"iterate": "best"},
        {"step_bound": 0.25, "max_iter": 500},
        {"iterate": "best", "step_bound": 0.25, "max_iter": 500},
    ],
)
def test_best_iterate_and_step_bound_keep_the_label_under_relative(
    settings: dict[str, Any],
) -> None:
    result = point(**settings)
    assert labels(result) == {"supported"}


def test_the_key_alone_keeps_a_rule_fit_off_the_default_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No preflight condition remains for a rule (the T52 and G55 rows replaced it), so the
    # key is what keeps a rule fit off T01. Without the T52 rows the fit matches no row; the
    # mutation that drops the key's stopping options lands it on the default row.
    _without_rows(monkeypatch, "T52.")
    result = point(stopping="scaled_var")
    assert labels(result) == {"experimental"}
    assert "options (stopping='scaled_var')" in result.estimates["ate"].support.reason
    monkeypatch.setattr(support_module, "_stopping_options", lambda estimator: {})
    monkeypatch.setattr(
        support_module,
        "registered_rows",
        lambda: tuple(
            replace(
                row,
                key=replace(
                    row.key,
                    options=tuple(
                        item
                        for item in row.key.options
                        if item[0] not in {"stopping", "step_bound", "iterate"}
                    ),
                ),
            )
            for row in REGISTERED_ROWS
            if not row.id.startswith("T52.")
        ),
    )
    assert labels(point(stopping="scaled_var")) == {"supported"}


@pytest.mark.parametrize(
    ("rule", "label"), [("relative", "supported"), ("scaled_var", "experimental")]
)
def test_a_longitudinal_stopping_rule_sets_the_label(rule: str, label: str) -> None:
    # A competing-risks fit: G55 covers the end-of-study and survival keys only, so a rule
    # fit here matches no row.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = longitudinal.competing(rule)
    assert {e.support.label for e in result.estimates.values()} == {label}
    if label == "experimental":
        assert {e.support.row for e in result.estimates.values()} == {None}
        assert OPTION_STUDIES_ROADMAP not in {
            i for e in result.estimates.values() for i in e.support.evidence
        }
