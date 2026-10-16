"""BCa interval, studentized sup-t band, refit contract and licensing (X39, T2 and T3).

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
from cleverly.longitudinal import estimator as longitudinal_estimator
from cleverly.longitudinal.estimator import (
    LICENSED_BOOTSTRAP_DESIGNS,
    LICENSED_BOOTSTRAP_INTERVALS,
    PENDING_BOOTSTRAP_LICENSES,
    bootstrap_design_kind,
    bootstrap_licensed,
)
from tests import discrete_law_longitudinal_horizon as law_
from tests import longitudinal_horizon_plans as plans_

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


# ------------------------------------------------------------------ S5: licensing


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


class TestLicensing:
    def test_the_new_intervals_are_pending_and_named(self) -> None:
        assert frozenset() == LICENSED_BOOTSTRAP_INTERVALS
        assert {"end_of_study/in_sample", "survival/in_sample"} == LICENSED_BOOTSTRAP_DESIGNS
        for kind, interval in PENDING_BOOTSTRAP_LICENSES:
            assert interval in ("percentile", "bca", "supt")
            assert not bootstrap_licensed(kind, interval)
        assert not any(
            kind.startswith("sdr/") and i == "supt" for kind, i in PENDING_BOOTSTRAP_LICENSES
        )
        assert bootstrap_licensed("end_of_study/in_sample", "percentile")
        assert not bootstrap_licensed("end_of_study/in_sample", "bca")
        assert not bootstrap_licensed(None, "percentile")

    def test_rule_policy_and_sdr_kinds(self) -> None:
        ltmle = _fit(n_bootstrap=2)
        regimens = ltmle.config.regimens
        kind = bootstrap_design_kind(ltmle.data, ltmle.folds, regimens, None)
        assert kind == "end_of_study/in_sample/policy"
        rule = [regimen for regimen in regimens if regimen.label != "mix"]
        assert bootstrap_design_kind(ltmle.data, ltmle.folds, rule, None) == (
            "end_of_study/in_sample/rule"
        )
        assert bootstrap_design_kind(ltmle.data, ltmle.folds, regimens, None, estimator="sdr") == (
            "sdr/end_of_study/in_sample/policy"
        )
        assert bootstrap_design_kind(ltmle.data, ltmle.folds, rule, None, finite=False) is None
        static = [regimen for regimen in regimens if regimen.label in ("low", "high")]
        assert bootstrap_design_kind(ltmle.data, ltmle.folds, static, None, finite=False) == (
            "end_of_study/in_sample"
        )

    @pytest.mark.parametrize("cls", [LTMLE, SDR])
    def test_a_bca_fit_prints_a_diagnostic_range(self, cls: Any) -> None:
        result = _fit(cls, n_bootstrap=4, bootstrap_interval="bca")
        for estimate in result.estimates.values():
            assert estimate.bootstrap is not None
            assert estimate.bootstrap.interval == "bca"
            assert not estimate.bootstrap.inferential
        assert "BCa range" in result.summary()
        assert "BCa CI" not in result.summary()

    def test_a_cross_fitted_bca_fit_claims_no_interval(self) -> None:
        result = _fit(n_folds=2, n_bootstrap=3, bootstrap_interval="bca")
        assert all(not e.bootstrap.inferential for e in result.estimates.values())
        assert "BCa range" in result.summary()

    def test_a_data_adaptive_learner_claims_no_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mutation: licensing the policy kind.  The finite-dimensional fit then claims
        # the interval, and the same fit with a penalized outcome learner still does not.
        monkeypatch.setattr(
            longitudinal_estimator,
            "LICENSED_BOOTSTRAP_INTERVALS",
            frozenset({("end_of_study/in_sample/policy", "bca")}),
        )
        finite = _fit(n_bootstrap=3, bootstrap_interval="bca")
        assert all(e.bootstrap.inferential for e in finite.estimates.values())
        from sklearn.linear_model import LogisticRegression

        adaptive = _fit(
            n_bootstrap=3, bootstrap_interval="bca", outcome_learner=LogisticRegression()
        )
        assert all(not e.bootstrap.inferential for e in adaptive.estimates.values())

    def test_bca_needs_replicates(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            LTMLE({"a": 1}, bootstrap_interval="bca")
        with pytest.raises(ValueError, match="'percentile' or 'bca'"):
            LTMLE({"a": 1}, n_bootstrap=2, bootstrap_interval="studentized")


class TestTheBootstrapBand:
    def test_the_band_reads_the_replicate_standard_errors(self) -> None:
        result = _fit(n_bootstrap=39, simultaneous=True)
        band = result.bootstrap_band
        assert band is not None and not band.inferential
        assert np.isfinite(band.critical_value)
        names = list(band.bands)
        estimates = np.array([result.estimates[name].psi for name in names])
        draws = np.column_stack([result.bootstrap.draws[name] for name in names])
        errors = np.column_stack([result.bootstrap.std_errors[name] for name in names])
        expected, _ = studentized_critical_value(estimates, draws, errors, ALPHA)
        assert band.critical_value == pytest.approx(expected, abs=1e-14)
        assert "sup-t diagnostic ranges, not inference" in result.summary()

    def test_no_band_without_simultaneous(self) -> None:
        assert _fit(n_bootstrap=3, simultaneous=False).bootstrap_band is None
