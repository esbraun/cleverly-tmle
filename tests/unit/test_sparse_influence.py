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
        # Two of five draws lie below the estimate, so z0 = Phi^-1(0.4), which is not zero and
        # changes sign with a sign slip.
        draws = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        row = influence_row("x", curve, estimate=0.25, draws=draws)
        assert row.bca_z0 == pytest.approx(norm.ppf(0.4), abs=1e-15)
        assert row.bca_z0 < -0.2
        assert row.bca_acceleration > 0

    def test_a_tied_draw_counts_for_half(self) -> None:
        curve = np.random.default_rng(3).exponential(size=50)
        row = influence_row("x", curve, estimate=0.2, draws=np.array([0.1, 0.2, 0.2, 0.3]))
        assert row.bca_z0 == 0.0

    def test_a_clustered_curve_is_read_at_the_cluster_level(self) -> None:
        curve = np.random.default_rng(5).normal(size=40) ** 3
        cluster = np.repeat(np.arange(8), 5)
        row = influence_row("x", curve, cluster=cluster)
        sums = curve.reshape(8, 5).sum(axis=1)
        reference = influence_row("x", sums)
        assert row.n == 8
        for field in ("n", "kish", "skewness", "kurtosis", "variance_rel_sd", "edgeworth_loss"):
            assert getattr(row, field) == getattr(reference, field), field
        assert influence_row("x", curve).n == 40
        assert influence_row("x", curve).kish != row.kish

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

    def test_the_advice_says_199_and_experimental(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False)
        text = str(record[0].message)
        assert "n_bootstrap=199" in text and "399" not in text
        assert "BCa interval is experimental" in text

    def test_a_percentile_bootstrap_is_told_to_use_bca(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False, percentile_bootstrap=True)
        text = str(record[0].message)
        assert "percentile bootstrap interval keeps the skew" in text
        assert "bootstrap_interval='bca'" in text and "n_bootstrap" not in text
        assert "experimental" in text

    def test_a_flagged_row_shows_why_it_is_flagged(self) -> None:
        report = self._report(20)
        (row,) = report.rows
        assert row.sparse
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(report, cross_fitted=False)
        text = str(record[0].message)
        assert f"relative SD {row.variance_rel_sd:.3f}" in text
        assert f"Edgeworth coverage loss {row.edgeworth_loss:.3f}" in text

    def test_a_flagged_row_can_carry_a_negative_edgeworth_loss(self) -> None:
        """The flag reads the relative SD only, so a negative loss does not clear it."""
        # Six symmetric spikes in 300 rows: kurtosis 50 lifts q2 above zero, so the Edgeworth
        # term predicts over-coverage while the variance estimate is noisy.
        curve = np.zeros(300)
        curve[:6] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        flagged_negative = influence_row("y", curve)
        assert flagged_negative.sparse and flagged_negative.edgeworth_loss < 0
        with pytest.warns(SparseInfluenceWarning, match="relative SD .* Edgeworth coverage loss -"):
            warn_sparse_influence(InfluenceReport((flagged_negative,), 0.05), cross_fitted=False)

    def test_the_warning_points_at_the_caller_outside_the_package(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            warn_sparse_influence(self._report(20), cross_fitted=False)
        assert record[0].filename == __file__

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
        with pytest.warns(SparseInfluenceWarning, match="n_bootstrap=199"):
            result = _fit(n_folds=1)
        report = result.diagnostics.influence()
        assert report.sparse
        assert set(report.sparse) <= set(result.estimates)

    @pytest.mark.filterwarnings("ignore:.*constant_node_plugin")
    @pytest.mark.parametrize(
        "settings",
        [{"isotonic": True}, {"survival_model": "hazard"}],
        ids=["isotonic", "hazard"],
    )
    def test_a_hazard_or_isotonic_fit_is_not_told_to_bootstrap(
        self, settings: dict[str, Any]
    ) -> None:
        from tests.unit.test_isotonic_projection import _fit as survival_fit

        with pytest.warns(SparseInfluenceWarning) as record:
            survival_fit(36, **settings)
        sparse = [w for w in record if issubclass(w.category, SparseInfluenceWarning)]
        text = str(sparse[0].message)
        assert "No registered study measures a bootstrap" in text
        assert "n_bootstrap" not in text and "bca" not in text

    def test_a_cross_fitted_fit_warns_without_the_bootstrap(self) -> None:
        with pytest.warns(SparseInfluenceWarning) as record:
            _fit(n_folds=2)
        sparse = [w for w in record if issubclass(w.category, SparseInfluenceWarning)]
        assert sparse and "n_bootstrap" not in str(sparse[0].message)

    def test_a_percentile_bootstrap_fit_warns_with_the_bca_advice(self) -> None:
        with pytest.warns(SparseInfluenceWarning, match="percentile bootstrap interval"):
            _fit(n_folds=1, n_bootstrap=3)

    def test_a_clustered_fit_does_not_warn(self) -> None:
        from tests.unit.test_bootstrap_intervals import _clustered

        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseInfluenceWarning)
            _clustered(n_bootstrap=0, bootstrap_interval="percentile")

    def test_a_bca_bootstrap_fit_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.simplefilter("error", SparseInfluenceWarning)
            result = _fit(n_folds=1, n_bootstrap=3, bootstrap_interval="bca")
        report = result.diagnostics.influence()
        assert all(np.isfinite(row.bca_acceleration) for row in report.rows)
        assert report.rows[0].name == next(iter(result.estimates))

    def test_influence_is_a_direct_call_and_not_in_run_all(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _fit(n_folds=1)
        assert result.diagnostics.influence().rows
        battery = result.diagnostics.run_all()
        assert "influence" not in {item.name for item in battery.items}
        assert "influence" not in {row.operation for row in result.diagnostics.capabilities}

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
