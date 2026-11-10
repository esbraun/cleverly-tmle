"""The declaration of the unregistered study ``survival-ctmle`` (X91 part B, design section 8).

The study is declared before its pilot: these checks hold the parts a pilot must not move,
its truth, its seeds and its sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

from tests.studies import survival_ctmle as study
from tests.studies import survival_ctmle_pilot as pilot
from tests.studies import survival_ctmle_properties as properties
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed


def test_the_truth_is_the_source_s_table_1_value() -> None:
    study.check_truth()
    survival = 1.0 - study.TRUTH["risk_regimen[a1 @ t=2]"]
    assert survival == pytest.approx(0.461490, abs=5e-7)
    _, mass = study.clipped_sum_law()
    assert mass.sum() == pytest.approx(1.0, abs=1e-8)
    # More nodes and a tighter orthant tolerance move it by less than 1e-8.
    finer = 1.0 - study.truth(nodes=40, eps=1e-12)["risk_regimen[a1 @ t=2]"]
    assert survival == pytest.approx(finer, abs=1e-8)
    # An independent rule, Gauss-Hermite on the clipped integrand, agrees to 1e-5.
    covariance = np.full((3, 3), 0.2)
    np.fill_diagonal(covariance, 1.0)
    points, weights = hermegauss(80)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(points, points, points, indexing="ij"), -1).reshape(-1, 3)
    cube = np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), -1).reshape(-1, 3)
    clipped = np.clip(grid @ np.linalg.cholesky(covariance).T, -2.0, 2.0).sum(axis=1)
    hermite = float(np.prod(cube, axis=1) @ (1.0 - expit(-0.75 - 0.6 * clipped)) ** 2)
    assert survival == pytest.approx(hermite, abs=1e-5)
    # And a Monte Carlo draw of the clipped law, within four standard errors.
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(np.zeros(3), covariance, 2_000_000)
    values = (1.0 - expit(-0.75 - 0.6 * np.clip(draws, -2, 2).sum(axis=1))) ** 2
    error = values.std() / np.sqrt(values.size)
    assert abs(values.mean() - survival) < 4.0 * error
    # The quadrature tells the clipped law from the unclipped one the source's 0.462 reads.
    unclipped = (grid @ np.linalg.cholesky(covariance).T).sum(axis=1)
    source = float(np.prod(cube, axis=1) @ (1.0 - expit(-0.75 - 0.6 * unclipped)) ** 2)
    assert source == pytest.approx(0.46180, abs=1e-5)
    assert abs(source - survival) > 2e-4


def test_the_gated_cells_are_declared() -> None:
    assert set(study.GATED_SCENARIOS) | set(study.REPORTED_SCENARIOS) == set(study.SCENARIOS)
    assert not set(study.GATED_SCENARIOS) & set(study.REPORTED_SCENARIOS)
    for name in study.GATED_SCENARIOS:
        _, _, model, _, settings = study.SCENARIOS[name]
        assert model == "correct"
        assert not settings.get("binarized") and "g_lower" not in settings
    assert "or15_binarized" in study.REPORTED_SCENARIOS
    assert study.RMSE_GATE_SCENARIO in study.GATED_SCENARIOS
    # The pilot read the risk difference inside the margin at OR 15, so every gated cell gates it.
    for estimands in study.GATED_ESTIMANDS.values():
        assert "ate_regimen[a1 vs a0 @ t=2]" in estimands


def test_no_coverage_or_se_ratio_gate_applies() -> None:
    """F18: the plug-in interval is a diagnostic, so no coverage or SE ratio can fail a cell."""
    margins = study.STUDY.margins
    assert margins.coverage_floor == 0.0
    low, high = margins.se_ratio_sanity
    assert low == 0.0 and high >= 1e6
    assert margins.standardized_bias == 0.25


def _primary_rows(subject_error: float, comparator_error: float) -> pd.DataFrame:
    """Gated rows of two cells, the comparators' errors a fixed multiple of the subject's."""
    rng = np.random.default_rng(3)
    rows = []
    for scenario in ("or15_qcorrect", "or1_qcorrect"):
        for replicate in range(40):
            noise = rng.normal()
            for estimand in study.GATED_ESTIMANDS[scenario]:
                truth = study.TRUTH[estimand]
                rows.append(
                    {
                        "implementation": study.CTMLE_ARM,
                        "scenario": scenario,
                        "replicate": replicate,
                        "estimand": estimand,
                        "truth": truth,
                        "estimate": truth + subject_error * noise,
                        "seconds": 1.0 + replicate % 2,
                        **{column: float(replicate % 2) for column in study.EXTRA_COLUMNS[1:]},
                        **dict.fromkeys(study.SECONDS_COLUMNS, 0.5),
                        **{
                            f"estimate[{arm}]": truth + comparator_error * noise
                            for arm in study.COMPARATORS
                        },
                        "a0_truth": study.TRUTH["risk_regimen[a0 @ t=2]"],
                        "a0_estimate": study.TRUTH["risk_regimen[a0 @ t=2]"] + 0.01 * noise,
                        "a0_std_error": 0.02,
                        "a0_covered": 1,
                    }
                )
    frame = pd.DataFrame(rows)
    assert set(study.PRIMARY_COLUMNS) <= set(frame.columns)
    return frame


def test_the_primary_rows_publish_the_reported_tables() -> None:
    tables = study.extra_artifacts(_primary_rows(0.01, 0.02))
    assert set(tables) == set(study.STUDY.extra_artifacts)
    readings = tables[study.SELECTION_FILE]
    assert readings["share_w4_in_g1"].tolist() == [0.5] * 4
    assert readings["fits"].tolist() == [40] * 4
    assert readings[f"mean_seconds[{study.CTMLE_ARM}]"].tolist() == [1.5] * 4
    arm0 = tables[study.ARM0_FILE]
    assert arm0["coverage"].tolist() == [1.0, 1.0]
    comparison = tables[study.COMPARISON_FILE]
    assert len(comparison) == 2 * 2 * len(study.COMPARATORS)
    assert comparison["rmse_ratio"].to_numpy() == pytest.approx(0.5)
    (gated,) = comparison.loc[comparison["gated"]].itertuples()
    assert (gated.scenario, gated.estimand, gated.comparator) == (
        study.RMSE_GATE_SCENARIO,
        study.RMSE_GATE_ESTIMAND,
        study.FULL_G_ARM,
    )


def test_gate_b_fails_when_the_subject_is_worse() -> None:
    """A witness: the same rows with the arms' errors swapped fail gate (b), and only it."""
    better = study.scientific_failures(study.extra_artifacts(_primary_rows(0.01, 0.02)))
    worse = study.scientific_failures(study.extra_artifacts(_primary_rows(0.02, 0.01)))
    assert all(frame.empty for frame in better.values())
    (failed,) = worse.values()
    assert len(failed) == 1
    assert failed["rmse_ratio"].iloc[0] == pytest.approx(2.0)


def test_the_record_holds_the_gated_cells_and_the_property_phase_the_rest() -> None:
    assert dict(study.STUDY.scenarios) == {
        name: study.GATED_ESTIMANDS[name] for name in study.GATED_SCENARIOS
    }
    for name in study.GATED_SCENARIOS:
        assert study.STUDY.n_of(name) == study.SCENARIOS[name][3]
    assert study.STUDY.properties_module == properties.__name__
    assert set(study.STUDY.property_artifacts) == {
        properties.CELLS_FILE,
        properties.REPLICATES_FILE,
    }
    assert set(study.REPORTED_SCENARIOS).isdisjoint(study.STUDY.scenarios)


def test_a_reported_fit_that_raises_is_counted_not_summarized() -> None:
    rows = pd.DataFrame(
        {
            "scenario": "or1_qwrong",
            "implementation": study.CTMLE_ARM,
            "estimand": study.ESTIMAND_NAMES[0],
            "replicate": range(4),
            "n": 1_000,
            "truth": 0.5,
            "estimate": [0.4, 0.6, 0.5, np.nan],
            "std_error": [0.1, 0.1, 0.1, np.nan],
            "covered": [1.0, 1.0, 1.0, np.nan],
            "failure": ["", "", "", "LinAlgError"],
            "seconds": 1.0,
        }
    )
    (cell,) = properties.summarize_cells(rows).itertuples()
    assert (cell.fits, cell.failures) == (3, 1)
    assert cell.bias == pytest.approx(0.0)
    assert cell.coverage == 1.0


def test_the_seeds_are_used_by_no_registered_study() -> None:
    seeds = {record.seed for record in registered()}
    resampling = {record.resampling_seed for record in registered()}
    assert study.SEED not in seeds | resampling
    assert study.RESAMPLING_SEED not in seeds | resampling


def test_every_gated_cell_draws_at_least_one_thousand_units() -> None:
    assert study.STUDY.slug not in {record.slug for record in registered()}
    for name in study.PRIMARY_SCENARIOS:
        assert study.STUDY.n_of(name) >= 1_000
    assert study.RMSE_GATE_SCENARIO in study.PRIMARY_SCENARIOS
    assert set(study.PRIMARY_SCENARIOS) == {
        f"or{odds}_q{model}" for odds in (1, 10, 15) for model in ("correct", "wrong")
    }


def test_the_sampler_follows_the_declared_law() -> None:
    frame = study.sample(20_000, 1, odds=15.0, source=1)
    assert set(frame["time"]) <= set(study.GRID)
    assert frame[list(study.COVARIATES)].abs().max().max() <= 2.0
    # The censoring hazard expit(-3 - 0.1 (W1 + W2 + W3 + A)) is about 4.7% per interval.
    censored = ((frame["event"] == 0) & (frame["time"] == 1.0)).sum()
    at_risk = (frame["time"] >= 1.0).sum() - (frame["event"] == 1)[frame["time"] == 1.0].sum()
    assert 0.03 < censored / at_risk < 0.07


def test_the_pilot_draws_no_declared_sample() -> None:
    """Every pilot draw takes its own stream, distinct from every declared replicate seed."""
    declared = {
        replicate_seed(study.STUDY, name, r) for name in study.STUDY.scenarios for r in range(50)
    }
    streams = [
        stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
    ] + [properties.reported_seed(name, r) for name in study.REPORTED_SCENARIOS for r in range(50)]
    assert len(set(streams)) == len(streams)
    assert not declared & set(streams)
    assert set(pilot.PILOTS) == set(study.SCENARIOS)
    for scenario, n, draws in pilot.PILOTS.values():
        assert n == study.SCENARIOS[scenario][3]
        assert draws == study.PILOT_REPLICATES


PILOT = pd.read_csv(pilot.OUTPUT, low_memory=False) if pilot.OUTPUT.exists() else pd.DataFrame()


@pytest.mark.skipif(PILOT.empty, reason="the pilot has not been committed")
class TestTheSizes:
    """The declared size reads the committed pilot by the shared rule."""

    def test_the_pilot_is_complete(self) -> None:
        for label in pilot.PILOTS:
            assert pilot._done(PILOT, label), label

    def test_the_subject_lost_at_most_the_failure_bound(self) -> None:
        subject = PILOT.loc[PILOT["implementation"] == study.CTMLE_ARM]
        lost = subject.groupby("label")["refusal"].apply(lambda r: r.fillna("").ne("").mean())
        assert (lost <= study.FAILURE_BOUND).all(), lost[lost > study.FAILURE_BOUND]

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        assert pilot.primary_replicates(PILOT) == study.PRIMARY_REPLICATES
