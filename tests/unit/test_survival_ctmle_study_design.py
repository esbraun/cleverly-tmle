"""The declaration of the unregistered study ``survival-ctmle`` (X91 part B, design section 8).

The study is declared before its pilot: these checks hold the parts a pilot must not move,
its truth, its seeds and its sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from tests.studies import survival_ctmle as study
from tests.studies.evidence.registry import registered


def test_the_truth_is_the_source_s_table_1_value() -> None:
    study.check_truth()
    # The quadrature agrees with a large Monte Carlo draw of the clipped law.
    rng = np.random.default_rng(0)
    covariance = np.full((3, 3), 0.2)
    np.fill_diagonal(covariance, 1.0)
    total = np.clip(rng.multivariate_normal(np.zeros(3), covariance, 400_000), -2, 2).sum(axis=1)
    survival = np.mean((1.0 - expit(-0.75 - 0.6 * total)) ** 2)
    assert 1.0 - study.TRUTH["risk_regimen[a1 @ t=2]"] == pytest.approx(survival, abs=3e-3)


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
