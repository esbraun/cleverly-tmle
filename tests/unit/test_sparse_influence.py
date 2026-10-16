"""Influence-curve measures and ``SparseInfluenceWarning`` (X39, T4)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from cleverly import SparseInfluenceWarning
from cleverly.inference.sparse import (
    SPARSE_INFLUENCE_THRESHOLD,
    InfluenceReport,
    influence_report,
    influence_row,
    warn_sparse_influence,
)
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_


class TestTheMeasures:
    def test_a_fixture_curve_with_known_moments(self) -> None:
        # Centred: m2 = 3, m3 = 6, m4 = 21.
        row = influence_row("x", np.array([3.0, -1.0, -1.0, -1.0]))
        assert row.skewness == pytest.approx(6.0 / 3.0**1.5, abs=1e-12)
        assert row.kurtosis == pytest.approx(21.0 / 9.0, abs=1e-12)
        assert row.kish == pytest.approx(4.0 / (21.0 / 9.0), abs=1e-12)
        assert row.variance_rel_sd == pytest.approx(math.sqrt((21.0 / 9.0 - 1.0) / 4.0))

    def test_the_kish_count_is_the_squared_curve_formula(self) -> None:
        curve = np.random.default_rng(1).exponential(size=500)
        centred = curve - curve.mean()
        row = influence_row("x", curve)
        assert row.kish == pytest.approx(np.sum(centred**2) ** 2 / np.sum(centred**4), rel=1e-12)

    def test_the_edgeworth_loss_is_the_two_sided_term(self) -> None:
        curve = np.random.default_rng(2).exponential(size=300) ** 2
        row = influence_row("x", curve, alpha=0.05)
        z = norm.ppf(0.975)
        gamma, kappa = row.skewness, row.kurtosis - 3.0
        q2 = (
            kappa * (z**3 - 3 * z) / 6
            - gamma**2 * (z**5 + 2 * z**3 - 3 * z) / 9
            - (z**3 + 3 * z) / 2
        )
        assert row.edgeworth_loss == pytest.approx(-norm.pdf(z) * q2 / 300, abs=1e-15)
        # At z = 1.96 the research note's coefficients: 0.275 kappa - 4.23 gamma^2 - 6.70.
        assert q2 == pytest.approx(0.275 * kappa - 4.23 * gamma**2 - 6.70, rel=2e-3)

    def test_bootstrap_measures_need_draws(self) -> None:
        curve = np.random.default_rng(3).exponential(size=50)
        assert math.isnan(influence_row("x", curve).bca_z0)
        draws = np.array([0.1, 0.2, 0.3, 0.4])
        row = influence_row("x", curve, estimate=0.25, draws=draws)
        assert row.bca_z0 == pytest.approx(0.0, abs=1e-15)
        assert row.bca_acceleration > 0

    def test_a_flat_curve(self) -> None:
        row = influence_row("x", np.ones(10))
        assert row.kish == math.inf and not row.sparse


class TestTheWarning:
    @staticmethod
    def _report(nonzero: int, n: int = 2000) -> InfluenceReport:
        curve = np.zeros(n)
        curve[:nonzero] = np.random.default_rng(4).normal(size=nonzero)
        return InfluenceReport((influence_row("x", curve),), 0.05)

    def test_the_threshold_is_a_kish_count_of_64(self) -> None:
        assert SPARSE_INFLUENCE_THRESHOLD**2 == 1 / 64
        for nonzero in (8, 30, 60, 70, 200, 1500):
            row = self._report(nonzero).rows[0]
            # (kappa - 1) / n > 1/64 differs from n / kappa < 64 only by 1/n.
            assert row.sparse == ((row.kurtosis - 1) / row.n > 1 / 64)
            assert row.sparse == (row.kish < 64) or abs(row.kish - 64) < 64 * 64 / row.n

    def test_a_dense_curve_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseInfluenceWarning)
            warn_sparse_influence(self._report(1500), cross_fitted=False)

    def test_a_sparse_curve_warns_with_the_bootstrap_advice(self) -> None:
        with pytest.warns(SparseInfluenceWarning, match="bootstrap_interval='bca'"):
            warn_sparse_influence(self._report(20), cross_fitted=False)

    def test_a_cross_fitted_fit_is_not_told_to_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=True)
        text = str(record[0].message)
        assert "X20" in text and "n_bootstrap" not in text

    def test_an_unmeasured_fit_is_not_told_to_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False, bootstrap_measured=False)
        assert "n_bootstrap" not in str(record[0].message)

    def test_the_warning_is_a_user_warning(self) -> None:
        assert issubclass(SparseInfluenceWarning, UserWarning)


def _fit(**settings: Any) -> Any:
    model = LTMLE(
        plans_.regimens(("low", "high", "respond"), 3, 2),
        outcome_learner=GLM(),
        pseudo_learner=GLM(),
        g_bounds=(1e-8, 1.0),
        random_state=0,
        **settings,
    )
    return model.fit(
        law_.draw(3, 2, 600, 9, law_.L_R), **law_.roles(3, 2), **law_.probability_columns(3, 2)
    )


class TestTheFit:
    def test_a_sparse_wald_fit_warns(self) -> None:
        with pytest.warns(SparseInfluenceWarning, match="n_bootstrap=399"):
            result = _fit(n_folds=1)
        report = result.diagnostics.influence()
        assert report.sparse
        assert set(report.sparse) <= set(result.estimates)

    def test_a_cross_fitted_fit_warns_without_the_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            _fit(n_folds=2)
        sparse = [w for w in record if issubclass(w.category, SparseInfluenceWarning)]
        assert sparse and "n_bootstrap" not in str(sparse[0].message)

    def test_a_bootstrap_fit_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", SparseInfluenceWarning)
            result = _fit(n_folds=1, n_bootstrap=3)
        report = result.diagnostics.influence()
        assert all(np.isfinite(row.bca_acceleration) for row in report.rows)
        assert report.rows[0].name == next(iter(result.estimates))

    def test_the_report_reads_every_estimate(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _fit(n_folds=1)
        report = influence_report(result.estimates)
        assert [row.name for row in report.rows] == list(result.estimates)
        frame = report.to_frame()
        assert list(frame["name"]) == list(result.estimates)


def test_linear_regression_is_frequency_exact() -> None:
    from cleverly.longitudinal.frequency import frequency_exact

    assert frequency_exact(LinearRegression()) and frequency_exact(GLM())
