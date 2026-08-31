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

from cleverly.datasets import nonlinear_dgp
from cleverly.utils.bounds import expit, logit
from tests.studies import (
    canonical_cvtmle,
    canonical_properties,
    cvtmle_properties,
    fold_evaluated_cvtmle,
    fold_targeted_cvtmle,
    repeated_crossfit,
)
from tests.studies.evidence.registry import StudyRecord, registered
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


def test_the_fold_targeted_driver_returns_its_exact_samples_truths_and_rows() -> None:
    samples, truths, rows = fold_targeted_cvtmle.draw_and_fit(replicates=1, n=200, n_jobs=1)
    record = fold_targeted_cvtmle.STUDY

    assert tuple(rows.columns) == REPLICATE_COLUMNS
    assert set(rows["implementation"]) == {record.implementation}
    assert rows["estimand"].tolist() == ["ate"]
    assert rows["initial_estimate"].notna().all()
    assert set(samples["fold"]) == {0, 1}
    assert samples.groupby("fold").size().tolist() == [100, 100]
    assert {"scenario", "replicate", "row_id", "partition_random_state"} <= set(samples)
    assert set(truths["scenario"]) == {"binary"}


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


def test_the_double_robustness_driver_declares_the_bounded_law_and_budgets() -> None:
    cells = tuple(
        cell for cell in canonical_properties.cells() if cell.property == "double_robustness"
    )
    assert {cell.cell: (cell.n, cell.replicates, cell.seed) for cell in cells} == {
        "both_correct": (700, 1_200, 17_100),
        "outcome_correct": (700, 1_200, 17_101),
        "treatment_correct": (2_000, 1_200, 17_102),
        "both_wrong": (700, 1_200, 17_103),
    }
    assert len({id(cell.dgp) for cell in cells}) == 1
    assert canonical_properties.DOUBLE_ROBUST_LOGIT_RANGE == (-2.5, 2.5)
    assert (
        canonical_properties.DOUBLE_ROBUST_G_RANGE[0]
        > canonical_properties.DOUBLE_ROBUST_G_BOUNDS[0]
    )
    assert (
        canonical_properties.DOUBLE_ROBUST_G_RANGE[1]
        < canonical_properties.DOUBLE_ROBUST_G_BOUNDS[1]
    )
    assert cells[0].dgp.truth()["ate"] == pytest.approx(1.75, abs=1e-6)


def test_the_bounded_mechanism_is_the_shipped_linear_predictor_squashed() -> None:
    """``double_robustness_dgp`` claims it keeps ``nonlinear_dgp``'s linear predictor.

    Nothing tied the two together. ``replace()`` inherits the outcome law by construction,
    but the propensity is rewritten, and its linear predictor is typed a second time in
    ``canonical_properties``. A silent edit to either copy would leave the docstring, the
    ``DOUBLE_ROBUST_G_RANGE`` derivation, and the both-wrong control describing a law the
    study no longer runs.

    The identity is exact rather than approximate: ``expit`` and ``logit`` are inverses on
    the open interval, so composing them recovers the shipped predictor. The grid stays
    inside two standard deviations, where no ``expit`` value saturates and ``logit`` is
    finite.
    """
    scale = canonical_properties.DOUBLE_ROBUST_LOGIT_SCALE
    axis = np.linspace(-2.0, 2.0, 7)
    grid = np.stack(
        [values.ravel() for values in np.meshgrid(*([axis] * 4), indexing="ij")], axis=1
    )
    assert grid.shape == (7**4, 4)

    shipped = nonlinear_dgp().propensity(grid)
    assert np.all((shipped > 0.0) & (shipped < 1.0))
    expected = expit(scale * np.tanh(logit(shipped) / scale))

    measured = canonical_properties.double_robustness_dgp().propensity(grid)
    np.testing.assert_allclose(measured, expected, rtol=0.0, atol=1e-12)

    # A deliberate-mutation control. The identity is only evidence if a changed predictor
    # breaks it, and a squashing map is flat enough in the middle to hide a small shift.
    mutated = expit(scale * np.tanh((logit(shipped) + 0.05) / scale))
    assert np.max(np.abs(mutated - expected)) > 1e-3


def test_the_double_robustness_bound_envelope_covers_every_consumer_of_the_law() -> None:
    """``DOUBLE_ROBUST_G_BOUNDS`` has to be the narrowest bound any consumer configures.

    The design check reads the envelope rather than one study's ``G_BOUNDS``, because the
    same law runs under ordinary TMLE at ``(0.01, 0.99)`` and under four cross-fitted
    studies at ``(0.025, 0.975)``. Reading one of them would let a study narrow its own
    clipping and weaken the check for the rest.

    Two things this reaches, and one it does not. It takes the bounds off the estimator the
    property path builds, not the module-level constant, so a property estimator that clips
    harder than its study's ``G_BOUNDS`` fails here. It walks the registry, so a new
    consumer of the law joins the candidate set without being named a second time.

    **It cannot bite on the five consumers today.** The envelope is the ``max`` and ``min``
    of the only two bounds they pass, so each comparison below is an identity until a study
    introduces a third one. The membership assertion is what carries this test today.
    """
    law = canonical_properties.double_robustness_dgp().name
    # ``cells`` and ``estimator`` take a variant that names the overfitting cell alone. The
    # calls here exclude that cell, and no arm's bounds vary with it, so one label serves.
    variant = "envelope_probe"
    consumers: dict[str, tuple[float, float]] = {}
    for record in registered():
        if "tests/studies/canonical_properties.py" not in record.modules:
            continue
        if "tests/studies/cvtmle_properties.py" in record.modules:
            declared = cvtmle_properties.cells(variant, include_overfitting=False)
            build = cvtmle_properties.estimator(variant)
        elif record.properties_module == canonical_properties.__name__:
            declared = canonical_properties.cells()
            build = canonical_properties._estimator
        else:
            declared = record.properties().cells()
            build = canonical_properties._estimator
        arms = [
            cell
            for cell in declared
            if cell.property == "double_robustness" and cell.dgp.name == law
        ]
        if not arms:
            continue
        configured = {tuple(build(cell)().g_bounds) for cell in arms}
        assert len(configured) == 1, f"{record.slug} clips its four arms differently"
        consumers[record.slug] = configured.pop()

    assert set(consumers) == {
        "canonical-tmle",
        "canonical-cvtmle",
        "fold-evaluated-cvtmle",
        "fold-targeted-cvtmle",
        "repeated-crossfit-tmle",
    }
    for slug, (lower, upper) in consumers.items():
        assert canonical_properties.DOUBLE_ROBUST_G_BOUNDS[0] >= lower, slug
        assert canonical_properties.DOUBLE_ROBUST_G_BOUNDS[1] <= upper, slug


def test_the_fold_targeted_property_driver_runs_every_deterministic_control() -> None:
    cvtmle_properties.assert_double_robustness_preflight(
        "fold_targeted",
        canonical_properties.cells(),
        repeats=1,
        n_folds=fold_targeted_cvtmle.N_FOLDS,
        targeting_scheme="fold",
        cv_evaluation=True,
    )
