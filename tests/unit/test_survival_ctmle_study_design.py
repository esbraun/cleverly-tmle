"""The declaration of the unregistered study ``survival-ctmle`` (X91 part B, design section 8).

The study is declared before its pilot: these checks hold the parts a pilot must not move,
its truth, its seeds and its sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit

from tests.studies import survival_ctmle as study
from tests.studies.evidence.registry import registered


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
    # The risk difference at OR 15 waits on the pilot.
    for name, estimands in study.GATED_ESTIMANDS.items():
        rd = "ate_regimen[a1 vs a0 @ t=2]" in estimands
        assert rd == (study.SCENARIOS[name][0] != 15.0)


def test_no_coverage_or_se_ratio_gate_applies() -> None:
    """F18: the plug-in interval is a diagnostic, so no coverage or SE ratio can fail a cell."""
    margins = study.STUDY.margins
    assert margins.coverage_floor == 0.0
    low, high = margins.se_ratio_sanity
    assert low == 0.0 and high >= 1e6
    assert margins.standardized_bias == 0.25


def test_the_rows_carry_the_selection_readings() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        [
            {
                "implementation": implementation,
                "scenario": "or15_qcorrect",
                "replicate": replicate,
                "estimand": "risk_regimen[a1 @ t=2]",
                "seconds": 1.0 + replicate,
                **{
                    column: (float(replicate) if implementation == study.CTMLE_ARM else np.nan)
                    for column in study.EXTRA_COLUMNS[1:]
                },
            }
            for implementation in study.IMPLEMENTATIONS
            for replicate in (0, 1)
        ]
    )
    (readings,) = study.extra_artifacts(rows).values()
    assert list(readings["arm"]) == list(study.ARMS)
    assert readings["share_w4_in_g1"].tolist() == [0.5, 0.5]
    assert readings["fits"].tolist() == [2, 2]
    assert readings[f"mean_seconds[{study.CTMLE_ARM}]"].tolist() == [1.5, 1.5]


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
