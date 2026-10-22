"""The registry rows that the part G limit coverage studies license (X82 SG).

Each ``ovb_limit`` and ``evalue_limit`` row must cover the configuration that ``cells.csv``
recorded for each cell it names, and each named cell must have passed in ``properties.csv``.
The mutation controls show that a smaller fit or another learner stays descriptive.

An ``ovb_limit`` row's event floor is read against the cell's 1,600 primary draws, not against
the one configuration draw: it must lie between their minimum and their 5th percentile, so
every licensed fit has an event count the coverage rows pooled.

The page keys that ``tests/studies/evidence/descriptions.py`` mirrors from the study modules
must resolve to declared cells.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import numpy as np
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


def _event_count(frame: pd.DataFrame) -> int:
    """The smallest count of events or non-events in any arm, over the observed rows.

    This mirrors ``cleverly.validation._interpretation.event_count`` on the frames of the
    O laws, which carry ``A``, ``Y`` and, for O1, the response indicator ``D``.
    """
    observed = frame["D"] == 1.0 if "D" in frame else pd.Series(True, index=frame.index)
    counts = []
    for arm in (0.0, 1.0):
        in_arm = observed & (frame["A"] == arm)
        events = int((frame.loc[in_arm, "Y"] > 0.5).sum())
        counts.extend((events, int(in_arm.sum()) - events))
    return min(counts)


@cache
def _primary_event_counts(cell: str) -> np.ndarray:
    """The event count of each primary draw of an O cell, from the declared seeds."""
    return np.array(
        [
            _event_count(ovb_study.draw_scenario(cell, ovb_study.PRIMARY_N, replicate)[0])
            for replicate in range(ovb_study.PRIMARY_REPLICATES)
        ]
    )


def _floored(row: VerdictEvidence) -> bool:
    return row.study[0] == ovb_study.STUDY.slug and row.scope.min_event_count_min is not None


@pytest.mark.parametrize("row", _rows(), ids=lambda row: f"{row.instrument}-{row.rule}")
def test_each_row_covers_every_cell_it_names(row: VerdictEvidence) -> None:
    for cell, configuration in _named(row):
        if _floored(row):
            # The floor is read against every primary draw below; one draw does not decide it.
            configuration = replace(configuration, min_event_count=row.scope.min_event_count_min)
        assert row.scope.mismatch(configuration) is None, cell


@pytest.mark.parametrize("row", [row for row in _rows() if _floored(row)], ids=lambda row: row.rule)
def test_each_event_floor_lies_inside_the_primary_draws(row: VerdictEvidence) -> None:
    floor = row.scope.min_event_count_min
    assert floor is not None
    cells = {cell.split("/", 1)[1].split("_", 1)[0] for cell in row.study[1:]}
    for cell in cells:
        counts = _primary_event_counts(cell)
        assert counts.min() <= floor <= np.percentile(counts, 5), (cell, floor)


def test_the_event_count_reads_the_recorded_configuration() -> None:
    # The mirror above must agree with the package's own count on the configuration draw.
    from tests.studies.evidence.seeds import stream_seed

    for cell, configuration in _configurations(ovb_study.STUDY.slug).items():
        scenario = cell.split("_", 1)[0]
        if configuration.min_event_count is None:
            continue
        frame, _ = ovb_study.draw_from_seed(
            scenario,
            ovb_study.PRIMARY_N,
            stream_seed(ovb_study.STUDY, "configuration", scenario),
        )
        assert _event_count(frame) == configuration.min_event_count, cell


def test_a_floor_above_the_fifth_percentile_fails() -> None:
    # Deliberate-mutation control: the old floor of 350 sat at the O4 median.
    counts = _primary_event_counts("O4")
    assert np.percentile(counts, 5) < 350


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
