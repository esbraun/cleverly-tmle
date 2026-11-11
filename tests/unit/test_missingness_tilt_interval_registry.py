"""The registry rows that the targeted missingness tilt study licenses (A2, CDR-001).

Each ``missingness_tilt_interval`` row must name only primary cells that passed in
``performance-tests.csv``, and every rule the study's fits read must have a row.  The mutation
controls show that another learner, a smaller fit, another fold count, a third arm, a
continuous outcome or an unreadable fit stays descriptive.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import cache

import pandas as pd
import pytest

import cleverly.validation._interpretation as interpretation
from cleverly.sensitivity.missingness import TILT_INTERVAL_INSTRUMENT, missingness_tilt
from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    VerdictEvidence,
    fit_configuration,
    resolve,
)
from tests.studies import missingness_tilt_inference as study


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.instrument == TILT_INTERVAL_INSTRUMENT]


@cache
def _performance() -> dict[tuple[str, str], bool]:
    table = pd.read_csv(study.STUDY.artifacts / "performance-tests.csv")
    return {(str(row.scenario), str(row.estimand)): bool(row.passed) for row in table.itertuples()}


@cache
def _fits() -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            scenario: study.fit(study.draw_scenario(scenario, 1000, 0)[0], scenario)
            for scenario in study.SCENARIOS
        }


def _rules_read(result: object, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    seen: set[str] = set()
    original = interpretation.resolve

    def recording(instrument: str, rule: str, fit: object) -> object:
        seen.add(rule)
        return original(instrument, rule, fit)

    monkeypatch.setattr(interpretation, "resolve", recording)
    missingness_tilt(result, [1.0])
    monkeypatch.undo()
    return seen


def test_every_row_names_only_passing_primary_cells() -> None:
    passed = _performance()
    assert _rows()
    for row in _rows():
        assert row.study[0] == study.STUDY.slug and row.rung == "L2" and not row.exact
        primary = [cell for cell in row.study[1:] if cell.startswith("primary/")]
        assert primary, row.rule
        for cell in primary:
            _, scenario, estimand = cell.split("/", 2)
            assert passed[(scenario, estimand)], cell
    properties = pd.read_csv(study.STUDY.artifacts / "properties.csv")
    calibration = properties[properties["cell"] == "ate_gamma2__correctly_specified"]
    assert bool(calibration["property_passed"].iloc[0])


def test_every_rule_the_study_reads_has_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    registered = {row.rule for row in _rows()}
    read: set[str] = set()
    for result in _fits().values():
        read |= _rules_read(result, monkeypatch)
    assert read == registered


@pytest.mark.parametrize("scenario", list(study.SCENARIOS))
def test_each_study_fit_reads_licensed(scenario: str) -> None:
    result = _fits()[scenario]
    assert missingness_tilt(result, [1.0]).descriptive == ()


@pytest.mark.parametrize(
    "change",
    [
        {"learners": ("RandomForestClassifier()",)},
        {"n": 999},
        {"cross_fitting": "folds=5"},
        {"arms": 3},
        {"outcome_family": "gaussian"},
        {"known_mechanism": True},
    ],
    ids=["learner", "n", "folds", "arms", "family", "known mechanism"],
)
def test_a_configuration_outside_the_study_stays_descriptive(change: dict[str, object]) -> None:
    fit = fit_configuration(_fits()["in_sample"], "ate")
    rule = "contrast"
    assert resolve(TILT_INTERVAL_INSTRUMENT, rule, fit).licensed
    assert not resolve(TILT_INTERVAL_INSTRUMENT, rule, replace(fit, **change)).licensed


def test_an_unreadable_fit_and_an_unmeasured_rule_stay_descriptive() -> None:
    fit = fit_configuration(_fits()["in_sample"], "ate")
    assert not resolve(TILT_INTERVAL_INSTRUMENT, "contrast", None).licensed
    for rule in ("contrast+repeats", "contrast+clustered", "ratio+cv_evaluation", "paf"):
        assert not resolve(TILT_INTERVAL_INSTRUMENT, rule, fit).licensed, rule
