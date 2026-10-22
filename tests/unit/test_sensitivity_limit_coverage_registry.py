"""The registry rows that the part G limit coverage studies license (X82 SG).

Each ``ovb_limit`` and ``evalue_limit`` row must cover the configuration that ``cells.csv``
recorded for each cell it names, and each named cell must have passed in ``properties.csv``.
The mutation controls show that a smaller fit or another learner stays descriptive.

The page keys that ``tests/studies/evidence/descriptions.py`` mirrors from the study modules
must resolve to declared cells.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    FitConfiguration,
    VerdictEvidence,
)
from tests.studies import evalue_limit_coverage as evalue_study
from tests.studies import ovb_limit_coverage as ovb_study
from tests.studies.evidence import descriptions

STUDIES = {study.STUDY.slug: study for study in (ovb_study, evalue_study)}


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.study[:1] and row.study[0] in STUDIES]


def _configurations(slug: str) -> dict[str, FitConfiguration]:
    """The configuration each primary cell recorded, keyed ``<scenario>_<parameter>``."""
    table = pd.read_csv(STUDIES[slug].STUDY.artifacts / "cells.csv", keep_default_na=False)
    configurations: dict[str, FitConfiguration] = {}
    for row in table.itertuples():
        configurations[f"{row.scenario}_{row.parameter}"] = FitConfiguration(
            method=str(row.method),
            estimand_kind=str(row.estimand_kind),
            outcome_family=str(row.outcome_family),
            treatment_kind=str(row.treatment_kind),
            learners=tuple(str(row.learners).split(" | ")),
            cross_fitting=str(row.cross_fitting),
            inference=str(row.inference),
            known_mechanism=str(row.known_mechanism) == "True",
            n=int(row.n),
            covariates_per_row=float(row.covariates_per_row),
            min_event_count=None if row.min_event_count == "" else int(float(row.min_event_count)),
            clusters=None,
            arms=int(row.arms),
            nodes=None if row.nodes == "" else int(float(row.nodes)),
        )
    return configurations


def _named(row: VerdictEvidence) -> list[tuple[str, FitConfiguration]]:
    """Each named property cell with the primary configuration of its estimand."""
    configurations = _configurations(row.study[0])
    named = []
    for cell in row.study[1:]:
        family, rest = cell.split("/", 1)
        assert family == "interval_calibration"
        estimand = rest.split("__", 1)[0]
        for end in ("_lower", "_upper"):
            estimand = estimand.removesuffix(end)
        named.append((cell, configurations[estimand]))
    return named


def test_the_studies_register_one_row_per_rule() -> None:
    found = {(row.instrument, row.rule) for row in _rows()}
    assert found == {
        ("ovb_limit", rule) for rule in ("response", "intermediate", "regime", "msm", "shift")
    } | {("evalue_limit", "stratum"), ("evalue_limit", "time_to_event")}
    assert all(row.rung == "L2" and not row.exact for row in _rows())


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.rule}")
def test_each_row_names_passing_cells(row: VerdictEvidence) -> None:
    properties = pd.read_csv(STUDIES[row.study[0]].STUDY.artifacts / "properties.csv")
    passed = {
        f"{record.property}/{record.cell}": bool(record.property_passed)
        for record in properties.itertuples()
    }
    assert len(row.study) > 1
    for cell in row.study[1:]:
        assert passed[cell], cell


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.rule}")
def test_each_row_covers_every_cell_it_names(row: VerdictEvidence) -> None:
    for cell, configuration in _named(row):
        assert row.scope.mismatch(configuration) is None, cell


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.rule}")
def test_a_smaller_fit_or_another_learner_stays_descriptive(row: VerdictEvidence) -> None:
    # Deliberate-mutation controls: the row reaches no fit that no cell ran.
    _, configuration = _named(row)[0]
    smaller = replace(configuration, n=configuration.n - 500)
    assert row.scope.mismatch(smaller) == f"n = {smaller.n} (below {row.scope.n_min})"
    other = replace(configuration, learners=("Lasso()", *configuration.learners[1:]))
    assert row.scope.mismatch(other) == "learner Lasso()"
    folded = replace(configuration, cross_fitting="folds=5")
    assert row.scope.mismatch(folded) == "cross-fitting 'folds=5'"


def test_the_page_keys_resolve_to_declared_cells() -> None:
    for study in STUDIES.values():
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        scenarios = {
            scenario
            for (slug, scenario) in descriptions.STUDY_SCENARIOS
            if slug == study.STUDY.slug
        }
        assert scenarios == set(study.SCENARIOS)
        for cell in declared:
            assert cell.split("_", 1)[0] in scenarios, cell
