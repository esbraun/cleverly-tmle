"""The registry rows that the marginal sensitivity bound limit study licenses (X82 SH).

Each ``msm_bound_limit`` row must cover the configuration that ``cells.csv`` recorded for each
cell it names, and each named cell must have passed in ``properties.csv``. The mutation controls
show that a smaller fit, another learner, a repeated fit or a custom learner stays descriptive.

The ``binary`` row's event floor is read against the cell's 2,000 primary draws, not against the
one configuration draw: it must lie between their minimum and their 5th percentile, so every
licensed fit has an event count the coverage rows pooled.

The page keys that ``tests/studies/evidence/descriptions.py`` mirrors from the study module must
resolve to declared cells.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
import pandas as pd
import pytest

from cleverly.sensitivity.marginal import MSM_BOUND_INSTRUMENT
from cleverly.validation._interpretation import (
    VERDICT_EVIDENCE,
    FitConfiguration,
    VerdictEvidence,
)
from tests.studies import marginal_bound_coverage as study
from tests.studies.evidence import descriptions


def _rows() -> list[VerdictEvidence]:
    return [row for row in VERDICT_EVIDENCE if row.instrument == MSM_BOUND_INSTRUMENT]


def _configurations() -> dict[str, FitConfiguration]:
    """The configuration each primary cell recorded, keyed by its scenario."""
    table = pd.read_csv(study.STUDY.artifacts / study.CELLS_FILE, keep_default_na=False)
    configurations: dict[str, FitConfiguration] = {}
    for row in table.itertuples():
        configurations[str(row.scenario)] = FitConfiguration(
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
        )
    return configurations


def _cells(row: VerdictEvidence) -> set[str]:
    return {cell.split("/", 1)[1].split("_", 1)[0] for cell in row.study[1:]}


def test_the_study_registers_one_row_per_rule() -> None:
    assert {row.rule: _cells(row) for row in _rows()} == {
        study.RULES[cell]: {cell} for cell in study.RULES
    }
    assert all(row.rung == "L2" and not row.exact for row in _rows())
    assert all(row.study[0] == study.STUDY.slug for row in _rows())


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.rule)
def test_each_row_names_both_passing_ends(row: VerdictEvidence) -> None:
    properties = pd.read_csv(study.STUDY.artifacts / "properties.csv")
    passed = {
        f"{record.property}/{record.cell}": bool(record.property_passed)
        for record in properties.itertuples()
    }
    (cell,) = _cells(row)
    assert set(row.study[1:]) == {
        f"interval_calibration/{cell}_ate_{end}__correctly_specified" for end in study.ENDS
    }
    for name in row.study[1:]:
        assert passed[name], name


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.rule)
def test_each_row_covers_its_cell(row: VerdictEvidence) -> None:
    (cell,) = _cells(row)
    configuration = _configurations()[cell]
    if row.scope.min_event_count_min is not None:
        # The floor is read against every primary draw below; one draw does not decide it.
        configuration = replace(configuration, min_event_count=row.scope.min_event_count_min)
    assert row.scope.mismatch(configuration) is None


@cache
def _primary_event_counts() -> np.ndarray:
    """The smallest arm event or non-event count of each SH2 primary draw."""
    counts = []
    for replicate in range(study.PRIMARY_REPLICATES):
        frame, _ = study.draw_scenario("SH2", study.PRIMARY_N, replicate)
        found = []
        for arm in (0.0, 1.0):
            outcome = frame.loc[frame["A"] == arm, "Y"]
            events = int((outcome > 0.5).sum())
            found.extend((events, len(outcome) - events))
        counts.append(min(found))
    return np.array(counts)


def test_the_binary_floor_lies_inside_the_primary_draws() -> None:
    (row,) = [row for row in _rows() if row.rule == "binary"]
    floor = row.scope.min_event_count_min
    assert floor is not None
    counts = _primary_event_counts()
    assert counts.min() <= floor <= np.percentile(counts, 5)


@pytest.mark.parametrize("row", _rows(), ids=lambda row: row.rule)
def test_a_smaller_fit_or_another_learner_stays_descriptive(row: VerdictEvidence) -> None:
    # Deliberate-mutation controls: the row reaches no fit that no cell ran.
    (cell,) = _cells(row)
    configuration = _configurations()[cell]
    if row.scope.min_event_count_min is not None:
        configuration = replace(configuration, min_event_count=row.scope.min_event_count_min)
    smaller = replace(configuration, n=configuration.n - 500)
    assert row.scope.mismatch(smaller) == f"n = {smaller.n} (below {row.scope.n_min})"
    other = replace(configuration, learners=("Lasso()", *configuration.learners[1:]))
    assert row.scope.mismatch(other) == "learner Lasso()"
    in_sample = replace(configuration, cross_fitting="off")
    assert row.scope.mismatch(in_sample) == "cross-fitting 'off'"
    arm_mean = replace(configuration, estimand_kind="ey1")
    assert row.scope.mismatch(arm_mean) is not None


def test_a_composition_rule_has_no_row() -> None:
    # Each feature the study did not measure names its own rule, which no row covers.
    rules = {row.rule for row in _rows()}
    for feature in ("custom_learners", "repeats", "clustered", "ratio", "missing_outcome"):
        assert f"continuous+{feature}" not in rules
        assert f"binary+{feature}" not in rules


def test_the_page_keys_resolve_to_declared_cells() -> None:
    declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
    scenarios = {
        scenario for (slug, scenario) in descriptions.STUDY_SCENARIOS if slug == study.STUDY.slug
    }
    assert scenarios == set(study.SCENARIOS)
    for cell in declared:
        family = "interval_calibration"
        tested, required = descriptions.cell(family, cell, slug=study.STUDY.slug)
        assert tested and required, cell
    for estimand in ("ate_lower", "ate_upper"):
        assert descriptions.estimand(estimand, slug=study.STUDY.slug)
