"""BCa interval, studentized sup-t band, refit contract and support (X39, T2 and T3).

Every interval below is checked against a hand computation written here, independently of
:mod:`cleverly.inference.bootstrap`.  Each mutation control applies one plausible slip to
that hand computation and asserts that the slip moves the answer, so the witness can see it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Any, ClassVar

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
from cleverly.inference.delta import Transform
from cleverly.inference.results import derived_bootstrap
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


def _floor(position: float) -> int:
    """The rank of a lower limit, from the position rounded to six places (exact ties)."""
    return math.floor(round(position, 6))


def _ceil(position: float) -> int:
    """The rank of an upper limit, from the position rounded to six places (exact ties)."""
    return math.ceil(round(position, 6))


def _hand_bca(
    draws: np.ndarray,
    estimate: float,
    acceleration: float,
    alpha: float,
    *,
    z0_sign: float = 1.0,
    lower_rank: Any = _floor,
    upper_rank: Any = _ceil,
) -> tuple[float, float]:
    ordered = np.sort(draws)
    size = ordered.size
    below = np.sum(ordered < estimate) + 0.5 * np.sum(ordered == estimate)
    z0 = z0_sign * norm.ppf(below / size)
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
            {"lower_rank": _ceil, "upper_rank": _floor},
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

    def test_the_ranks_at_the_boundary_are_the_5th_and_195th_of_199(self) -> None:
        """At z0 = 0 and a = 0 the level is exactly 0.025, whatever the float round trip.

        ``norm.cdf(norm.ppf(0.025)) * 200`` is 4.999999999999996, so an unrounded floor gives
        the 4th order statistic (plan rev 4.4, disclosure 2, names the 5th and the 195th).
        """
        draws = np.arange(199.0)
        # One draw equals the estimate and counts for half: 99.5 / 199 = 1/2, so z0 = 0.
        low, high, z0 = bca_interval(draws, 99.0, 0.0, ALPHA)
        assert z0 == 0.0
        ordered = np.sort(draws)
        assert (low, high) == (ordered[4], ordered[194]) == (4.0, 194.0)
        assert norm.cdf(norm.ppf(0.025)) * 200 < 5.0, "the float round trip no longer needs a slack"
        assert _hand_bca(draws, 99.0, 0.0, ALPHA) == (4.0, 194.0)
        # The sup-t critical value rounds the same way: 0.95 * 200 is the 190th statistic.
        maxima = np.arange(1.0, 200.0)
        errors = np.ones((199, 1))
        critical, _ = studentized_critical_value(np.zeros(1), maxima.reshape(-1, 1), errors, 0.05)
        assert critical == maxima[189]

    def test_a_tied_draw_counts_for_half(self) -> None:
        z0 = bca_interval(np.array([1.0, 2.0, 2.0, 3.0]), 2.0, 0.0, ALPHA)[2]
        assert z0 == 0.0

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
        rank = 95
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


# ------------------------------------------------------------------ the derived BCa path


def _bca_fit(**settings: Any) -> Any:
    settings.setdefault("simultaneous", False)
    return _fit(n_bootstrap=59, bootstrap_interval="bca", **settings)


def _finite_pair(result: Any, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [np.asarray(result.bootstrap.draws[name], dtype=float) for name in names]
    )


class TestTheDerivedBCa:
    NAMES: ClassVar[list[str]] = ["ey_regimen[high]", "ey_regimen[low]"]

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return _bca_fit()

    def _hand(self, draws: np.ndarray, estimate: float, curve: np.ndarray) -> Any:
        finite = draws[np.isfinite(draws)]
        acceleration = bca_acceleration(curve)
        return bca_interval(finite, estimate, acceleration, ALPHA), acceleration

    def test_a_difference_reads_its_own_curve(self, result: Any) -> None:
        high, low = (result.estimates[name] for name in self.NAMES)
        derived = result.contrast(lambda x: x[0] - x[1], self.NAMES, name="d")
        draws = _finite_pair(result, self.NAMES)
        values = draws[:, 0] - draws[:, 1]
        (lo, hi, z0), acceleration = self._hand(
            values, high.psi - low.psi, high.influence_curve - low.influence_curve
        )
        summary = derived.bootstrap
        assert summary.interval == "bca"
        assert summary.acceleration == pytest.approx(acceleration, rel=1e-6)
        assert summary.z0 == pytest.approx(z0, abs=1e-12)
        assert summary.ci == pytest.approx((lo, hi), abs=1e-9)

    def test_a_ratio_reads_the_reported_scale_curve(self, result: Any) -> None:
        high, low = (result.estimates[name] for name in self.NAMES)
        derived = result.ratio(*self.NAMES)
        draws = _finite_pair(result, self.NAMES)
        values = draws[:, 0] / draws[:, 1]
        # The stored curve is the log ratio's, a positive multiple of the ratio's.
        curve = high.influence_curve / high.psi - low.influence_curve / low.psi
        (lo, hi, _), acceleration = self._hand(values, high.psi / low.psi, curve)
        assert derived.bootstrap.acceleration == pytest.approx(acceleration, rel=1e-6)
        assert derived.bootstrap.ci == pytest.approx((lo, hi), abs=1e-9)

    def test_a_decreasing_transform_leaves_the_limits_unchanged(self, result: Any) -> None:
        name = self.NAMES[0]
        negate = Transform("neg", lambda v: -v, lambda v: -v, lambda v: -1.0)
        plain = result.contrast(lambda x: x[0], [name], name="plain")
        flipped = result.contrast(lambda x: x[0], [name], name="neg", transform=negate)
        # The stored curve of the transformed estimate is -1 times the plain one.
        np.testing.assert_allclose(flipped.influence_curve, -plain.influence_curve, atol=1e-12)
        assert flipped.bootstrap.acceleration == pytest.approx(
            plain.bootstrap.acceleration, rel=1e-9
        )
        assert flipped.bootstrap.ci == plain.bootstrap.ci
        # Mutation: the stored curve gives the opposite sign and other limits.
        wrong = bca_acceleration(flipped.influence_curve)
        assert wrong == pytest.approx(-plain.bootstrap.acceleration, rel=1e-9)
        draws = np.asarray(result.bootstrap.draws[name], dtype=float)
        slipped = bca_interval(draws[np.isfinite(draws)], plain.psi, wrong, ALPHA)[:2]
        assert slipped != plain.bootstrap.ci

    def test_an_increasing_transform_only_rescales_the_curve(self, result: Any) -> None:
        name = self.NAMES[0]
        plain = result.contrast(lambda x: x[0], [name], name="plain")
        scaled = result.contrast(lambda x: x[0], [name], name="log", transform=Transform.log())
        assert scaled.bootstrap.acceleration == pytest.approx(
            plain.bootstrap.acceleration, rel=1e-6
        )
        assert scaled.bootstrap.ci == plain.bootstrap.ci

    def test_mixed_intervals_are_refused(self, result: Any) -> None:
        percentile = _fit(n_bootstrap=59, simultaneous=False)
        mixed = {
            self.NAMES[0]: result.estimates[self.NAMES[0]],
            self.NAMES[1]: replace(
                result.estimates[self.NAMES[1]],
                bootstrap=percentile.estimates[self.NAMES[1]].bootstrap,
            ),
        }
        with pytest.raises(ValueError, match="mix intervals"):
            derived_bootstrap(
                mixed, result.bootstrap, self.NAMES, lambda x: x[0] - x[1], alpha=ALPHA
            )

    def test_a_bca_summary_needs_the_derived_estimate(self, result: Any) -> None:
        with pytest.raises(ValueError, match="needs the derived estimate"):
            derived_bootstrap(
                result.estimates, result.bootstrap, self.NAMES, lambda x: x[0] - x[1], alpha=ALPHA
            )


# ------------------------------------------------------------------ the attached BCa interval


class TestTheAttachedBCa:
    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return _bca_fit()

    def test_each_estimate_reads_its_own_estimate_and_curve(self, result: Any) -> None:
        moved = 0
        for name, estimate in result.estimates.items():
            draws = np.asarray(result.bootstrap.draws[name], dtype=float)
            finite = draws[np.isfinite(draws)]
            acceleration = bca_acceleration(estimate.influence_curve)
            expected = bca_interval(finite, estimate.psi, acceleration, ALPHA)[:2]
            assert estimate.bootstrap.ci == pytest.approx(expected, abs=1e-12), name
            assert estimate.bootstrap.acceleration == pytest.approx(acceleration, abs=1e-15)
            # Mutation: the curve of the opposite sign (a contrast's `fit - base`).
            slipped = bca_interval(finite, estimate.psi, -acceleration, ALPHA)[:2]
            moved += slipped != expected
        assert moved > 0, "no estimate's limits move with the sign of the acceleration"

    def test_the_export_names_the_interval(self, result: Any) -> None:
        name = "ey_regimen[high]"
        row = result.estimates[name].to_dict()
        assert row["bootstrap_interval"] == "bca"
        assert row["bootstrap_z0"] == result.estimates[name].bootstrap.z0
        assert row["bootstrap_acceleration"] == result.estimates[name].bootstrap.acceleration
        percentile = _fit(n_bootstrap=5, simultaneous=False).estimates[name].to_dict()
        assert percentile["bootstrap_interval"] == "percentile"
        assert "bootstrap_z0" not in percentile and "bootstrap_acceleration" not in percentile

    def test_the_band_is_the_estimate_plus_or_minus_the_critical_value_standard_error(
        self,
    ) -> None:
        result = _fit(n_bootstrap=39, simultaneous=True)
        band = result.bootstrap_band
        assert band is not None
        for name, (low, high) in band.bands.items():
            estimate = result.estimates[name]
            half = band.critical_value * estimate.plugin_std_error
            assert low == pytest.approx(estimate.psi - half, abs=1e-14)
            assert high == pytest.approx(estimate.psi + half, abs=1e-14)


# ------------------------------------------------------------------ cluster resampling


CLUSTERS = 60
PER_CLUSTER = 15


def _clustered(**settings: Any) -> Any:
    """900 rows in 60 clusters of 15, fitted with a BCa bootstrap that resamples clusters."""
    frame = law_.draw(K, 2, CLUSTERS * PER_CLUSTER, 6, law_.L_R)
    frame["cid"] = np.arange(len(frame)) // PER_CLUSTER
    settings.setdefault("n_bootstrap", 19)
    settings.setdefault("bootstrap_interval", "bca")
    settings.setdefault("simultaneous", False)
    model = LTMLE(
        plans_.regimens(("low", "high"), K, 2),
        reference="low",
        outcome_learner=GLM(),
        pseudo_learner=GLM(),
        g_bounds=(1e-8, 1.0),
        random_state=3,
        n_folds=1,
        **settings,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(frame, id="cid", **law_.roles(K, 2), **law_.probability_columns(K, 2))


def _cluster_sums(result: Any, curve: np.ndarray) -> np.ndarray:
    _, codes = np.unique(np.asarray(result.data.cluster), return_inverse=True)
    return np.bincount(codes, weights=curve)


class TestClusterResampling:
    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return _clustered()

    def test_the_fixture_is_60_clusters_of_15(self, result: Any) -> None:
        assert result.bootstrap.resampling == "cluster"
        assert np.unique(np.asarray(result.data.cluster)).size == CLUSTERS
        assert (
            np.bincount(np.unique(np.asarray(result.data.cluster), return_inverse=True)[1]).tolist()
            == [PER_CLUSTER] * CLUSTERS
        )

    def test_the_acceleration_reads_the_cluster_sums(self, result: Any) -> None:
        differing = 0
        for name, estimate in result.estimates.items():
            sums = _cluster_sums(result, estimate.influence_curve)
            centred = sums - sums.mean()
            expected = np.sum(centred**3) / (6 * np.sum(centred**2) ** 1.5)
            assert estimate.bootstrap.acceleration == pytest.approx(expected, abs=1e-14), name
            row = bca_acceleration(estimate.influence_curve)
            # Mutation: the row curve gives another value, off by well over rounding.
            differing += abs(row - expected) > 5e-3
        assert differing > 0, "the fixture does not separate the row and cluster accelerations"

    def test_the_limits_read_the_cluster_acceleration(self, result: Any) -> None:
        for name, estimate in result.estimates.items():
            draws = np.asarray(result.bootstrap.draws[name], dtype=float)
            finite = draws[np.isfinite(draws)]
            acceleration = bca_acceleration(estimate.influence_curve, result.data.cluster)
            expected = bca_interval(finite, estimate.psi, acceleration, ALPHA)[:2]
            assert estimate.bootstrap.ci == pytest.approx(expected, abs=1e-12), name

    def test_a_derived_estimate_reads_the_cluster_sums(self, result: Any) -> None:
        names = ["ey_regimen[low]", "ey_regimen[high]"]
        derived = result.contrast(lambda x: x[0] + 0.0 * x[1], names, name="d")
        curve = result.estimates[names[0]].influence_curve
        expected = bca_acceleration(curve, result.data.cluster)
        assert derived.bootstrap.acceleration == pytest.approx(expected, rel=1e-6)
        # Mutation: the row curve gives a different acceleration on this fixture.
        assert abs(bca_acceleration(curve) - expected) > 5e-3

    def test_a_row_bootstrap_of_a_clustered_fit_reads_the_rows(self) -> None:
        result = _clustered(bootstrap_resampling="iid")
        assert result.bootstrap.resampling == "iid"
        for estimate in result.estimates.values():
            assert estimate.bootstrap.acceleration == pytest.approx(
                bca_acceleration(estimate.influence_curve), abs=1e-14
            )

    def test_the_influence_report_is_at_the_cluster_level(self, result: Any) -> None:
        report = result.diagnostics.influence()
        for row in report.rows:
            curve = result.estimates[row.name].influence_curve
            sums = _cluster_sums(result, curve)
            centred = sums - sums.mean()
            kappa = np.mean(centred**4) / np.mean(centred**2) ** 2
            assert row.n == CLUSTERS
            assert row.kurtosis == pytest.approx(kappa, rel=1e-12)
            assert row.kish == pytest.approx(CLUSTERS / kappa, rel=1e-12)
            assert row.skewness == pytest.approx(
                np.mean(centred**3) / np.mean(centred**2) ** 1.5, rel=1e-12
            )
            assert row.bca_acceleration == pytest.approx(
                result.estimates[row.name].bootstrap.acceleration, abs=1e-14
            )
