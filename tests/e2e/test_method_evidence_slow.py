"""The parts of the method-evidence gate the fast tier cannot afford.

Three things live here, all of them the same checks the fast module makes, at a price only a
slow tier can pay: the resampling bounds recomputed at the published bootstrap budget, the
estimator re-executed on committed replications, and the property study re-run from scratch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.parallel import STUDY_JOBS, available_cores
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import StudyRecord, registered

pytestmark = pytest.mark.slow

STUDIES = registered()

#: Half the budget, floored at the measured inner setting.  These tests are a handful of
#: long fits rather than the thousands of short ones the ``STUDY_JOBS`` measurement was taken
#: on, and one of them is the critical path of the whole tier.
JOBS = max(STUDY_JOBS, available_cores() // 2)

#: How many replications per scenario the re-execution refits.  Not all of them: the
#: complete re-execution is ``regenerate.py``, which also re-runs the R reference, and
#: quoting the cap here is the point -- a sampled check that reads as an exhaustive one is
#: worse than a sampled check that says so.
REFIT_SAMPLE = 50


@pytest.fixture(params=STUDIES, ids=[study.slug for study in STUDIES])
def study(request: pytest.FixtureRequest) -> StudyRecord:
    return request.param


@pytest.fixture(scope="module")
def rows() -> dict[str, pd.DataFrame]:
    return {study.slug: pd.read_csv(study.artifact("replicates.csv.gz")) for study in STUDIES}


def test_the_resampling_bounds_are_recomputed_from_the_replication_rows(
    study: StudyRecord, rows: dict[str, pd.DataFrame]
) -> None:
    """The bootstrap columns the fast tier takes on trust, at the published budget."""
    replicates = rows[study.slug]
    summaries = summarize(replicates)
    performance = independent_performance_tests(replicates, record=study, n_jobs=JOBS)
    pd.testing.assert_frame_equal(
        pd.read_csv(study.artifact("performance-tests.csv")),
        performance,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if study.reference is not None:
        pd.testing.assert_frame_equal(
            pd.read_csv(study.artifact("equivalence.csv")),
            equivalence(replicates, summaries, performance, record=study, n_jobs=JOBS),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


def test_refitting_committed_replications_reproduces_their_rows(
    study: StudyRecord, rows: dict[str, pd.DataFrame]
) -> None:
    """The study is evidence about this code only while this code still produces it."""
    runner = study.runner()
    published = rows[study.slug]
    chosen = np.linspace(0, study.replicates - 1, REFIT_SAMPLE, dtype=int)
    for scenario in study.scenarios:
        for replicate in chosen:
            frame, truth = runner.draw_scenario(scenario, study.n, int(replicate))
            refitted = pd.DataFrame(runner.cleverly_rows(frame, truth, scenario, int(replicate)))
            expected = published.loc[
                (published["implementation"] == study.implementation)
                & (published["scenario"] == scenario)
                & (published["replicate"] == replicate)
            ]
            merged = expected.merge(refitted, on="estimand", suffixes=("_published", "_refitted"))
            assert len(merged) == len(study.scenarios[scenario])
            for column in ("estimate", "std_error", "ci_lower", "ci_upper"):
                assert merged[f"{column}_refitted"].to_numpy() == pytest.approx(
                    merged[f"{column}_published"].to_numpy(), rel=1e-6, abs=1e-9
                ), f"{scenario} replicate {replicate} no longer reproduces its {column}"


def test_the_property_study_reproduces_its_verdicts_when_it_is_re_run(
    study: StudyRecord,
) -> None:
    properties = study.properties()
    regenerated = properties.summarize_properties(properties.generate_property_rows(n_jobs=JOBS))
    published = pd.read_csv(study.artifact("properties.csv"))
    pd.testing.assert_frame_equal(
        published, regenerated, check_exact=False, check_dtype=False, rtol=1e-9, atol=1e-9
    )
    if study.publication_policy == "gated":
        assert regenerated["passed"].all(), regenerated.loc[~regenerated["passed"]].to_string()


def test_every_declared_property_cell_is_present_and_passing(study: StudyRecord) -> None:
    published = pd.read_csv(study.artifact("properties.csv"))
    by_property = published.groupby("property")["cell"].apply(set).to_dict()
    expected = {name: set(cells) for name, cells in study.property_cells.items()}
    assert by_property == expected
    if study.publication_policy == "gated":
        assert published["passed"].all(), published.loc[~published["passed"]].to_string()
