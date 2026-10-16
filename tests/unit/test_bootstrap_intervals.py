"""BCa interval, studentized sup-t band, refit contract and support (X39, T2 and T3).

Every interval below is checked against a hand computation written here, independently of
:mod:`cleverly.inference.bootstrap`.  Each mutation control applies one plausible slip to
that hand computation and asserts that the slip moves the answer, so the witness can see it.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from cleverly import _support
from cleverly.inference.bootstrap import (
    REFIT_CONTRACT,
    BootstrapResult,
    bca_acceleration,
    bca_interval,
    run_bootstrap,
    studentized_critical_value,
)
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, SDR
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_
from tests.studies import canonical_ltmle
from tests.unit.test_ltmle_bootstrap import _end_columns

ALPHA = 0.05


# ------------------------------------------------------------------ T2: the BCa interval


def _skewed() -> tuple[np.ndarray, float, np.ndarray]:
    """An exponential sample of 40, its mean and 199 fixed bootstrap means."""
    sample = np.random.default_rng(20).exponential(size=40)
    rng = np.random.default_rng(21)
    draws = np.array([sample[rng.integers(0, 40, 40)].mean() for _ in range(199)])
    return sample, float(sample.mean()), draws


def _hand_bca(
    draws: np.ndarray,
    estimate: float,
    acceleration: float,
    alpha: float,
    *,
    z0_sign: float = 1.0,
    lower_rank: Any = math.floor,
    upper_rank: Any = math.ceil,
) -> tuple[float, float]:
    ordered = np.sort(draws)
    size = ordered.size
    z0 = z0_sign * norm.ppf(np.mean(ordered < estimate))
    limits = []
    for q, rank in ((alpha / 2, lower_rank), (1 - alpha / 2, upper_rank)):
        shifted = z0 + norm.ppf(q)
        level = norm.cdf(z0 + shifted / (1 - acceleration * shifted))
        limits.append(float(ordered[min(max(rank(level * (size + 1)), 1), size) - 1]))
    return limits[0], limits[1]


class TestBCa:
    def test_the_witness_is_skewed(self) -> None:
        sample, estimate, draws = _skewed()
        acceleration = bca_acceleration(sample - estimate)
        _, _, z0 = bca_interval(draws, estimate, acceleration, ALPHA)
        assert acceleration > 0.01
        assert abs(z0) > 0.01

    def test_the_acceleration_is_the_curve_formula(self) -> None:
        sample, estimate, _ = _skewed()
        centred = sample - estimate
        expected = np.sum(centred**3) / (6 * np.sum(centred**2) ** 1.5)
        assert bca_acceleration(sample) == pytest.approx(expected, abs=1e-15)
        assert bca_acceleration(3.0 * sample) == pytest.approx(expected, abs=1e-15)

    def test_the_limits_equal_the_hand_computation(self) -> None:
        sample, estimate, draws = _skewed()
        acceleration = bca_acceleration(sample)
        low, high, _ = bca_interval(draws, estimate, acceleration, ALPHA)
        hand = _hand_bca(draws, estimate, acceleration, ALPHA)
        assert low == pytest.approx(hand[0], abs=1e-12)
        assert high == pytest.approx(hand[1], abs=1e-12)

    def test_the_interval_is_not_the_percentile_interval(self) -> None:
        """Mutation control: a = 0 and z0 = 0 reduce BCa to the percentile interval."""
        sample, estimate, draws = _skewed()
        low, high, _ = bca_interval(draws, estimate, bca_acceleration(sample), ALPHA)
        percentile = np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
        assert max(abs(low - percentile[0]), abs(high - percentile[1])) > 1e-3
        flat = _hand_bca(draws, float(np.median(draws)), 0.0, ALPHA)
        assert max(abs(low - flat[0]), abs(high - flat[1])) > 1e-3

    @pytest.mark.parametrize(
        "slip",
        [
            {"acceleration_sign": -1.0},
            {"z0_sign": -1.0},
            {"lower_rank": math.ceil, "upper_rank": math.floor},
        ],
        ids=["acceleration sign", "z0 sign", "floor and ceil swapped"],
    )
    def test_each_slip_moves_the_limits(self, slip: dict[str, Any]) -> None:
        sample, estimate, draws = _skewed()
        acceleration = bca_acceleration(sample)
        correct = bca_interval(draws, estimate, acceleration, ALPHA)[:2]
        sign = slip.pop("acceleration_sign", 1.0)
        slipped = _hand_bca(draws, estimate, sign * acceleration, ALPHA, **slip)
        assert slipped != pytest.approx(correct, abs=1e-12)

    def test_a_draw_set_on_one_side_has_no_limits(self) -> None:
        low, high, z0 = bca_interval(np.array([1.0, 2.0, 3.0]), 0.0, 0.1, ALPHA)
        assert math.isnan(low) and math.isnan(high) and z0 == -math.inf

    def test_the_summary_reports_the_bca_limits(self) -> None:
        sample, estimate, draws = _skewed()
        acceleration = bca_acceleration(sample)
        result = BootstrapResult({"m": draws}, 199, 0, "iid")
        summary = result.summary(
            "m", ALPHA, interval="bca", estimate=estimate, acceleration=acceleration
        )
        assert summary.interval == "bca"
        assert summary.ci == pytest.approx(bca_interval(draws, estimate, acceleration, ALPHA)[:2])
        assert summary.acceleration == acceleration
        percentile = result.summary("m", ALPHA)
        assert percentile.interval == "percentile"
        assert percentile.ci == pytest.approx(tuple(np.quantile(draws, [0.025, 0.975])))
        with pytest.raises(ValueError, match="estimate and acceleration"):
            result.summary("m", ALPHA, interval="bca")


# ------------------------------------------------------------------ T3: the sup-t band


class TestStudentizedBand:
    @staticmethod
    def _draws() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(30)
        estimates = np.array([0.2, 0.5, -0.1])
        draws = estimates + rng.normal(scale=[0.05, 0.1, 0.02], size=(99, 3))
        errors = np.abs(rng.normal(loc=[0.05, 0.1, 0.02], scale=0.01, size=(99, 3)))
        return estimates, draws, errors

    def test_the_critical_value_is_the_order_statistic(self) -> None:
        estimates, draws, errors = self._draws()
        critical, used = studentized_critical_value(estimates, draws, errors, ALPHA)
        maxima = np.sort(np.max(np.abs(draws - estimates) / errors, axis=1))
        rank = math.ceil(0.95 * (99 + 1))
        assert used == 99
        assert critical == pytest.approx(maxima[rank - 1], abs=1e-14)

    def test_the_replicate_standard_errors_are_read(self) -> None:
        """Mutation: the full-sample SE in place of each replicate's own."""
        estimates, draws, errors = self._draws()
        critical, _ = studentized_critical_value(estimates, draws, errors, ALPHA)
        full = np.broadcast_to(errors.mean(axis=0), errors.shape)
        frozen, _ = studentized_critical_value(estimates, draws, full, ALPHA)
        assert abs(frozen - critical) > 1e-3

    def test_too_few_replicates_give_no_critical_value(self) -> None:
        estimates, draws, errors = self._draws()
        critical, used = studentized_critical_value(estimates, draws[:10], errors[:10], 0.01)
        assert used == 10 and math.isnan(critical)


# ------------------------------------------------------------------ S1: the refit contract


class TestRefitContract:
    def test_a_bare_estimate_is_refused(self) -> None:
        class Rows:
            n = 5
            cluster = None

            def subset(self, index: Any) -> Rows:
                return self

        with pytest.raises(TypeError, match="standard error"):
            run_bootstrap(Rows(), lambda sample: {"x": 0.0}, n_replicates=2, random_state=0)  # type: ignore[arg-type,return-value]
        assert "influence-curve standard error" in REFIT_CONTRACT


# ------------------------------------------------------------------ S5: support


K = 3


def _fit(cls: Any = LTMLE, **settings: Any) -> Any:
    settings.setdefault("n_folds", 1)
    settings.setdefault("outcome_learner", GLM())
    settings.setdefault("pseudo_learner", GLM() if cls is LTMLE else LinearRegression())
    model = cls(
        plans_.regimens(("low", "high", "respond", "mix"), K, 2),
        reference="low",
        g_bounds=(1e-8, 1.0),
        random_state=3,
        **settings,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(
            law_.draw(K, 2, 900, 5, law_.L_R),
            **law_.roles(K, 2),
            **law_.probability_columns(K, 2),
        )


def _canonical(**settings: Any) -> Any:
    """The registered ``canonical-ltmle`` fit (row ``G01``) at 400 units, with a bootstrap."""
    frame, _ = canonical_ltmle.draw_scenario(canonical_ltmle.SCENARIO, 400, 0)
    settings.setdefault("simultaneous", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            canonical_ltmle.REGIMENS,
            reference=canonical_ltmle.REFERENCE,
            outcome_learner=canonical_ltmle.QuasiBinomialGLM(),
            pseudo_learner=canonical_ltmle.QuasiBinomialGLM(),
            treatment_learner=canonical_ltmle.KnownLongitudinalMechanism("treatment"),
            censoring_learner=canonical_ltmle.KnownLongitudinalMechanism("censoring"),
            n_folds=1,
            g_bounds=canonical_ltmle.G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            **settings,
        ).fit(frame, **_end_columns())


def _g01_lists(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    """The mutation: row ``G01`` also lists ``kind``."""
    from dataclasses import replace

    rows = tuple(
        replace(row, kinds=row.kinds | {kind}) if row.id == "G01" else row
        for row in _support.registered_rows()
    )
    monkeypatch.setattr(_support, "registered_rows", lambda: rows)


class TestSupport:
    def test_no_row_lists_the_new_kinds(self) -> None:
        for row in _support.registered_rows():
            assert not {"bootstrap_bca", "bootstrap_band"} & set(row.kinds), row.id

    def test_the_bca_interval_of_a_supported_fit_is_experimental(self) -> None:
        percentile = _canonical(n_bootstrap=3)
        bca = _canonical(n_bootstrap=3, bootstrap_interval="bca")
        for name, estimate in bca.estimates.items():
            # The same fit and row: only the interval's kind differs.
            assert percentile.estimates[name].bootstrap.support.label == "supported"
            assert estimate.bootstrap.interval == "bca"
            assert estimate.bootstrap.support.label == "experimental"
            assert estimate.bootstrap.support.row == "G01"
            assert "BCa bootstrap interval" in estimate.bootstrap.support.reason
        text = bca.summary()
        assert "BCa CI" in text and "[experimental]" in text

    def test_a_row_that_lists_bca_supports_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mutation control: the label follows the row, not a constant."""
        _g01_lists(monkeypatch, "bootstrap_bca")
        result = _canonical(n_bootstrap=3, bootstrap_interval="bca")
        assert all(e.bootstrap.support.label == "supported" for e in result.estimates.values())

    @pytest.mark.parametrize("cls", [LTMLE, SDR])
    def test_a_rule_and_policy_bca_fit_is_experimental(self, cls: Any) -> None:
        result = _fit(cls, n_bootstrap=4, bootstrap_interval="bca")
        for estimate in result.estimates.values():
            assert estimate.bootstrap is not None
            assert estimate.bootstrap.interval == "bca"
            assert estimate.bootstrap.support.label == "experimental"
        assert "BCa CI" in result.summary()

    def test_a_cross_fitted_bca_fit_is_experimental(self) -> None:
        result = _fit(n_folds=2, n_bootstrap=3, bootstrap_interval="bca")
        assert all(e.bootstrap.support.label == "experimental" for e in result.estimates.values())

    def test_bca_needs_replicates(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            LTMLE({"a": 1}, bootstrap_interval="bca")
        with pytest.raises(ValueError, match="'percentile' or 'bca'"):
            LTMLE({"a": 1}, n_bootstrap=2, bootstrap_interval="studentized")


class TestTheBootstrapBand:
    def test_the_band_reads_the_replicate_standard_errors(self) -> None:
        result = _fit(n_bootstrap=39, simultaneous=True)
        band = result.bootstrap_band
        assert band is not None and band.support.label == "experimental"
        assert np.isfinite(band.critical_value)
        names = list(band.bands)
        estimates = np.array([result.estimates[name].psi for name in names])
        draws = np.column_stack([result.bootstrap.draws[name] for name in names])
        errors = np.column_stack([result.bootstrap.std_errors[name] for name in names])
        expected, _ = studentized_critical_value(estimates, draws, errors, ALPHA)
        assert band.critical_value == pytest.approx(expected, abs=1e-14)
        assert "replicates) [experimental]:" in result.summary()

    def test_the_band_of_a_supported_fit_is_experimental(self) -> None:
        band = _canonical(n_bootstrap=3, simultaneous=True).bootstrap_band
        assert band is not None
        assert band.support.label == "experimental"
        assert "studentized bootstrap band" in band.support.reason

    def test_a_row_that_lists_the_band_supports_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mutation control: the band's label follows the ``bootstrap_band`` kind."""
        _g01_lists(monkeypatch, "bootstrap_band")
        band = _canonical(n_bootstrap=3, simultaneous=True).bootstrap_band
        assert band is not None and band.support.label == "supported"

    def test_no_band_without_simultaneous(self) -> None:
        assert _fit(n_bootstrap=3, simultaneous=False).bootstrap_band is None
