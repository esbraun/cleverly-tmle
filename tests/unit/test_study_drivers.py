"""What a study's ``draw_and_fit`` returns, checked without regenerating anything.

``draw_and_fit`` is the entry point ``tests/canonical/regenerate.py`` calls to produce a
study's published replication file, and it is the only caller. Nothing in either test tier
touched it, so a driver that returned the wrong columns, dropped a scenario, or reported an
interval its coverage flag contradicts would first show up the next time somebody
regenerated a study. A truth that is wrong for every row stays invisible here, because
``covered`` is rebuilt from the same joined truth. The
cross-fitted drivers are about to share one implementation, which is why they need a gate
before they share it and not after.

Each case draws one replication of every declared scenario at ``n = 200`` on a single job.
Measured at about half a second per study, so the module costs under two seconds and stays
in the fast tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_cvtmle, fold_evaluated_cvtmle, repeated_crossfit
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import INFERENCE_SCALES, REPLICATE_COLUMNS

#: The three cross-fitted point-treatment runners.  ``canonical_cvtmle`` returns a triple
#: rather than one frame, so it is checked separately below.
SCENARIO_FITTED = (fold_evaluated_cvtmle, repeated_crossfit)


def _check_rows(rows: pd.DataFrame, record: StudyRecord) -> None:
    """The per-row contract, asserted directly rather than through the file validator.

    ``schema.validate_replicates`` is the validator for a *published* file, so it also
    requires the declared replication count and the declared ``n``. A one-replication probe
    at ``n = 200`` satisfies neither, and raising the probe to the declared 800 or 1600
    replications is the regeneration this branch must not run.
    """
    assert tuple(rows.columns) == REPLICATE_COLUMNS
    assert set(rows["implementation"]) == {record.implementation}
    assert set(rows["scenario"]) == set(record.scenarios)
    assert set(rows["replicate"]) == {0}
    assert set(rows["n"]) == {200}
    for scenario, estimands in record.scenarios.items():
        subset = rows.loc[rows["scenario"] == scenario]
        assert set(subset["estimand"]) == set(estimands)
        assert len(subset) == len(estimands)
    assert np.isfinite(rows[["estimate", "inference_estimate", "std_error"]].to_numpy()).all()
    assert (rows["std_error"] > 0.0).all()
    assert (rows["ci_lower"] <= rows["ci_upper"]).all()
    # The coverage indicator has to agree with the interval it came from.  This catches a
    # truth misjoined for some rows and not others.  It cannot catch one that is wrong
    # everywhere, because ``covered`` comes from the same joined truth.
    recomputed = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
    assert (recomputed.astype(int) == rows["covered"]).all()
    assert set(rows["inference_scale"]) <= INFERENCE_SCALES


@pytest.mark.parametrize("runner", SCENARIO_FITTED, ids=lambda module: module.STUDY.slug)
def test_a_scenario_fitted_driver_returns_the_replication_contract(runner: object) -> None:
    rows = runner.draw_and_fit(replicates=1, n=200, n_jobs=1)  # type: ignore[attr-defined]
    _check_rows(rows, runner.STUDY)  # type: ignore[attr-defined]


def test_the_shared_driver_gives_one_answer_on_one_job_or_two() -> None:
    """``n_jobs=1`` takes the sequential branch of ``map_parallel`` and never pickles.

    Regeneration runs the drivers on ``STUDY_JOBS``, so the nested call the shared driver
    passes to the pool has to survive being sent to a worker.
    """
    one = repeated_crossfit.draw_and_fit(replicates=1, n=200, n_jobs=1)
    two = repeated_crossfit.draw_and_fit(replicates=1, n=200, n_jobs=2)
    pd.testing.assert_frame_equal(one, two)


def test_the_stacked_driver_returns_its_samples_truths_and_rows() -> None:
    """This one publishes the sample and the truth as well, for the R side to read."""
    samples, truths, rows = canonical_cvtmle.draw_and_fit(replicates=1, n=200, n_jobs=1)
    record = canonical_cvtmle.STUDY

    # ``validate_replicates`` wants both implementations, and a driver produces only the
    # subject's half of the file, so the column contract is asserted directly here.
    assert tuple(rows.columns) == REPLICATE_COLUMNS
    assert set(rows["implementation"]) == {record.implementation}
    assert len(rows) == sum(len(record.scenarios[scenario]) for scenario in record.scenarios)
    assert (rows["std_error"] > 0.0).all()

    assert len(samples) == 200 * len(record.scenarios)
    assert {"scenario", "replicate", "fold"} <= set(samples.columns)
    assert set(truths["scenario"]) == set(record.scenarios)


def test_a_driver_draws_the_scenario_the_seed_names() -> None:
    """The rows a driver returns come from the study's own published seed stream.

    Without this the parametrised cases above hold for a driver that fits some other
    study's sample: every column would still be well formed.
    """
    runner = repeated_crossfit
    rows = runner.draw_and_fit(replicates=1, n=200, n_jobs=1)
    for scenario in runner.STUDY.scenarios:
        frame, truth = runner.draw_scenario(scenario, 200, 0)
        expected = pd.DataFrame(runner.cleverly_rows(frame, truth, scenario, 0))
        published = rows.loc[rows["scenario"] == scenario]
        merged = published.merge(expected, on="estimand", suffixes=("_driver", "_direct"))
        assert len(merged) == len(published)
        assert merged["estimate_driver"].to_numpy() == pytest.approx(
            merged["estimate_direct"].to_numpy(), rel=1e-9
        )
