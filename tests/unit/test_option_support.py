"""The support label of a fit that declares a targeting or fitting option.

A declared stopping rule, ``outcome_by_arm=True``, ``patience=`` and ``treatment_adjustment=``
run, and their estimates carry the experimental label with roadmap X76 as the evidence
(``cleverly.estimators._option_support``).  ``iterate="best"`` and ``step_bound=`` under
``stopping="relative"`` keep the label of the fit without them.  The witness for each
experimental case is the same fit without the option, which a registered row covers, and a
mutation that removes both the preflight hook and the key's stopping options returns the
supported label.
"""

from __future__ import annotations

import importlib
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

tmle_module = importlib.import_module("cleverly.estimators.tmle")
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


@pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
def test_a_declared_stopping_rule_is_experimental(rule: str) -> None:
    result = point(stopping=rule)
    assert labels(result) == {"experimental"}
    assert OPTION_STUDIES_ROADMAP in evidence(result)
    reason = result.estimates["ate"].support.reason
    assert f"stopping={rule!r}" in reason
    assert "experimental" in result.summary().note_codes
    settings = result.summary(detail="full").section("Settings").lines
    assert any(plain_support_reason(reason) in line for line in settings)


def test_outcome_by_arm_is_experimental() -> None:
    result = point(outcome_by_arm=True)
    assert labels(result) == {"experimental"}
    assert "outcome_by_arm=True" in result.estimates["ate"].support.reason


def test_a_treatment_restriction_is_experimental() -> None:
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
    assert labels(result) == {"experimental"}
    assert "treatment_adjustment=" in result.estimates["ate"].support.reason


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


def test_the_key_and_the_preflight_hook_each_keep_a_rule_fit_experimental(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The key records the rule (Fix-8 PR 0), so dropping the hook leaves the fit off every
    # row. The mutation that drops both the hook and the key's stopping options restores
    # the supported label.
    monkeypatch.setattr(tmle_module, "point_option_conditions", lambda estimator, data: [])
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
        ),
    )
    assert labels(point(stopping="scaled_var")) == {"supported"}


@pytest.mark.parametrize(
    ("rule", "label"), [("relative", "supported"), ("scaled_var", "experimental")]
)
def test_a_longitudinal_stopping_rule_sets_the_label(rule: str, label: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = longitudinal.competing(rule)
    assert {e.support.label for e in result.estimates.values()} == {label}
    if label == "experimental":
        assert OPTION_STUDIES_ROADMAP in {
            i for e in result.estimates.values() for i in e.support.evidence
        }
