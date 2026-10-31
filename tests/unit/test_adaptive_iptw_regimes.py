r"""AdaptiveIPTW regimes and the natural course, against test-side population fixed points.

The deterministic and stochastic regimes run on the eight-point law of
:mod:`tests.unit._adaptive_iptw_support`. The natural course runs on an eighteen-point law with
a missing outcome, built here. Each test computes the round-zero mechanism, the fixed point and
the first step in numpy from the law tables, and compares them with the shipped values.

* W14: a deterministic known rule, a three-arm rule and the static set.
* W15: the natural-course key, the joint stack and complete data.
* W16: a known stochastic regime, and the routes of a degenerate density.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.estimators import AdaptiveIPTW
from cleverly.interventions import Rule, Static, Stochastic
from cleverly.interventions.base import _covariate_frame
from tests.unit import _adaptive_iptw_support as s

#: The rule: four points to arm 1 and four to arm 0. Points 1 and 2 tie in ``g~_d`` at 0.40
#: and split across ``d = 1`` and ``d = 0``. Points 0, 4 and 5 tie at 0.25.
D = np.array([1, 1, 0, 1, 0, 1, 0, 0])

#: ``P(A = 1 | W)`` under the stochastic regime. It differs within each tie pair of ``G_TILDE``.
S_ONE = np.array([0.30, 0.50, 0.70, 0.40, 0.60, 0.80, 0.20, 0.55])


def _level(frame: pd.DataFrame) -> np.ndarray:
    return np.rint(frame["W"].to_numpy(dtype=float)).astype(int)


def _rule(name: str = "rule") -> Rule:
    return Rule(lambda frame: D[_level(frame)], name=name, rule_kind="known")


def _stochastic(table: np.ndarray, name: str = "tilt") -> Stochastic:
    def density(frame: pd.DataFrame) -> np.ndarray:
        one = table[_level(frame)]
        return np.column_stack([1.0 - one, one])

    return Stochastic(density, name=name, density_kind="known")


def _fit_regimes(frame: pd.DataFrame, interventions: list[Any], **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.fit_arm(frame, interventions=interventions, estimands=None, **settings)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return s.law_frame()


# -------------------------------------------------------------------------------------- W14


class TestW14DeterministicRule:
    @pytest.fixture(scope="class")
    def run(self, frame: pd.DataFrame) -> tuple[Any, s.SolveKeyLog]:
        log = s.SolveKeyLog()
        with log:
            result = _fit_regimes(frame, [_rule()])
        return result, log

    def _inputs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = _level(frame)
        g_d = np.where(D[w] == 1, s.G_TILDE[w], 1.0 - s.G_TILDE[w])
        indicator = (frame["A"].to_numpy() == D[w]).astype(float)
        return indicator, frame["Y"].to_numpy(), g_d

    def test_the_round_zero_mechanism_is_the_mechanism_at_the_rule_arm(
        self, frame: pd.DataFrame, run: tuple[Any, s.SolveKeyLog]
    ) -> None:
        _, log = run
        _, _, g_d = self._inputs(frame)
        assert np.max(np.abs(log.mechanisms["rule"] - g_d)) < 1e-15
        # The tie pair split across the rule: points 1 (d = 1) and 2 (d = 0) share one value.
        assert g_d[_level(frame) == 1][0] == pytest.approx(g_d[_level(frame) == 2][0], abs=1e-15)
        assert s.G[1] != 1.0 - s.G[2]

    def test_the_shipped_value_and_first_step_equal_the_test_side_loop(
        self, frame: pd.DataFrame, run: tuple[Any, s.SolveKeyLog]
    ) -> None:
        result, _ = run
        indicator, target, g_d = self._inputs(frame)
        ref = s.fixed_point(indicator, target, g_d, frame["wt"].to_numpy())
        assert result.estimates["ey_regime[rule]"].psi == pytest.approx(ref["psi"], abs=1e-10)
        shipped = result.extra["adaptive_iptw"].epsilon["rule"]
        assert shipped[0] == pytest.approx(ref["eps"][0], abs=1e-10)
        assert abs(ref["eps"][0]) > 1e-3

    def test_the_fixed_point_identity_holds_with_a_nonzero_remainder(
        self, frame: pd.DataFrame
    ) -> None:
        indicator, target, g_d = self._inputs(frame)
        weight = frame["wt"].to_numpy()
        ref = s.fixed_point(indicator, target, g_d, weight)
        w = _level(frame)
        g_star = np.array([ref["g"][np.flatnonzero(w == point)[0]] for point in range(8)])
        qr_star = np.array([ref["qr"][np.flatnonzero(w == point)[0]] for point in range(8)])
        g0_d = np.where(D == 1, s.G, 1.0 - s.G)
        q0_d = np.where(D == 1, s.Q[:, 1], s.Q[:, 0])
        psi_0 = float(np.sum(s.P_W * q0_d))
        score = float(np.sum(s.P_W * qr_star / g_star * (g0_d - g_star)))
        assert abs(score) < 1e-13
        remainder = float(np.sum(s.P_W * (q0_d - qr_star) * (g0_d - g_star) / g_star))
        assert ref["psi"] - psi_0 == pytest.approx(remainder, abs=1e-9)
        assert abs(remainder) > 1e-6

    def test_the_static_set_equals_the_arm_fit_bit_for_bit(self, frame: pd.DataFrame) -> None:
        arms = s.fit_arm(frame, estimands=("ey1", "ey0"))
        static = _fit_regimes(frame, [Static(0, name="never"), Static(1, name="always")])
        assert static.estimates["ey_regime[never]"].psi == arms.estimates["ey0"].psi
        assert static.estimates["ey_regime[always]"].psi == arms.estimates["ey1"].psi
        assert np.array_equal(
            np.asarray(static.influence_curves["ey_regime[always]"]),
            np.asarray(arms.influence_curves["ey1"]),
        )

    def test_the_weight_tilted_support_has_its_own_fixed_point(self, frame: pd.DataFrame) -> None:
        tilt = np.array([1.0, 2.0, 0.5, 3.0, 1.0, 0.25, 2.0, 1.0])[_level(frame)]
        tilted = frame.assign(wt=frame["wt"] * tilt)
        result = _fit_regimes(tilted, [_rule()])
        indicator, target, g_d = self._inputs(tilted)
        ref = s.fixed_point(indicator, target, g_d, tilted["wt"].to_numpy())
        assert result.estimates["ey_regime[rule]"].psi == pytest.approx(ref["psi"], abs=1e-10)
        unweighted = s.fixed_point(indicator, target, g_d, frame["wt"].to_numpy())
        assert abs(ref["psi"] - unweighted["psi"]) > 1e-3


class TestW14ThreeArms:
    """A rule that uses all three arms of a four-point law."""

    P_W = np.array([0.2, 0.3, 0.3, 0.2])
    G = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.4, 0.4, 0.2]])
    G_TILDE = np.array([[0.4, 0.4, 0.2], [0.3, 0.4, 0.3], [0.25, 0.25, 0.5], [0.4, 0.35, 0.25]])
    Q = np.array([[0.2, 0.5, 0.7], [0.6, 0.3, 0.4], [0.5, 0.7, 0.2], [0.3, 0.6, 0.5]])
    D = np.array([2, 0, 1, 2])

    class Table(BaseEstimator, ClassifierMixin):
        def __init__(self, table: Any = None) -> None:
            self.table = table

        def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
            self.classes_ = np.array([0.0, 1.0, 2.0])
            return self

        def predict_proba(self, x: Any) -> np.ndarray:
            index = np.rint(np.asarray(x, dtype=float)[:, 0]).astype(int)
            return np.asarray(self.table)[index]

    def _frame(self) -> pd.DataFrame:
        rows = []
        for w, a, y in itertools.product(range(4), range(3), range(2)):
            share = self.Q[w, a] if y == 1 else 1.0 - self.Q[w, a]
            rows.append(
                {
                    "W": float(w),
                    "A": float(a),
                    "Y": float(y),
                    "wt": self.P_W[w] * self.G[w, a] * share,
                }
            )
        return pd.DataFrame(rows)

    def test_the_shipped_value_equals_the_test_side_loop(self) -> None:
        frame = self._frame()
        rule = Rule(
            lambda data: self.D[np.rint(data["W"].to_numpy()).astype(int)],
            name="rule",
            rule_kind="known",
        )
        log = s.SolveKeyLog()
        with log, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                AdaptiveIPTW(
                    treatment_learner=self.Table(self.G_TILDE),
                    reduced_outcome_learner=s.SaturatedRegressor(),
                    interventions=[rule],
                    estimands=["ey_regime"],
                    cross_fit=False,
                    g_bounds=(1e-3, 1 - 1e-3),
                    tol=1e-12,
                    max_iter=200,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="wt")
                .single()
            )
        w = np.rint(frame["W"].to_numpy()).astype(int)
        g_d = self.G_TILDE[w, self.D[w]]
        assert np.max(np.abs(log.mechanisms["rule"] - g_d)) < 1e-15
        indicator = (frame["A"].to_numpy() == self.D[w]).astype(float)
        ref = s.fixed_point(indicator, frame["Y"].to_numpy(), g_d, frame["wt"].to_numpy())
        assert result.estimates["ey_regime[rule]"].psi == pytest.approx(ref["psi"], abs=1e-10)
        assert abs(ref["eps"][0]) > 1e-3
        assert set(self.D) == {0, 1, 2}


# -------------------------------------------------------------------------------------- W16


class TestW16KnownStochasticRegime:
    @pytest.fixture(scope="class")
    def run(self, frame: pd.DataFrame) -> tuple[Any, s.SolveKeyLog]:
        log = s.SolveKeyLog()
        with log:
            result = _fit_regimes(frame, [_stochastic(S_ONE)])
        return result, log

    def _arm_inputs(self, frame: pd.DataFrame, arm: int) -> tuple[np.ndarray, ...]:
        w = _level(frame)
        g = s.G_TILDE[w] if arm == 1 else 1.0 - s.G_TILDE[w]
        density = S_ONE[w] if arm == 1 else 1.0 - S_ONE[w]
        indicator = (frame["A"].to_numpy() == arm).astype(float)
        return indicator, density * frame["Y"].to_numpy(), g

    @pytest.mark.parametrize("arm", [0, 1])
    def test_each_arm_key_reads_its_own_mechanism(
        self, frame: pd.DataFrame, run: tuple[Any, s.SolveKeyLog], arm: int
    ) -> None:
        _, log = run
        _, _, g = self._arm_inputs(frame, arm)
        assert np.max(np.abs(log.mechanisms[f"tilt/{arm}"] - g)) < 1e-15

    def test_the_shipped_value_is_the_sum_of_the_test_side_arm_fixed_points(
        self, frame: pd.DataFrame, run: tuple[Any, s.SolveKeyLog]
    ) -> None:
        result, _ = run
        weight = frame["wt"].to_numpy()
        total = 0.0
        for arm in (0, 1):
            indicator, target, g = self._arm_inputs(frame, arm)
            ref = s.fixed_point(indicator, target, g, weight)
            shipped = result.extra["adaptive_iptw"].epsilon[f"tilt/{arm}"]
            assert len(shipped) > 0, "the arm key took no targeting step"
            assert shipped[0] == pytest.approx(ref["eps"][0], abs=1e-10)
            assert abs(ref["eps"][0]) > 1e-3
            total += ref["psi"]
        assert result.estimates["ey_regime[tilt]"].psi == pytest.approx(total, abs=1e-10)

    @pytest.mark.parametrize("arm", [0, 1])
    def test_the_transformed_target_changes_the_regression(
        self, frame: pd.DataFrame, arm: int
    ) -> None:
        """``s`` differs within each tie pair, so ``Qr'`` differs from ``Qr`` by more than 1e-3."""
        weight = frame["wt"].to_numpy()
        indicator, target, g = self._arm_inputs(frame, arm)
        plain = s.saturated_regression(g, frame["Y"].to_numpy(), indicator > 0, weight)
        transformed = s.saturated_regression(g, target, indicator > 0, weight)
        assert np.max(np.abs(plain - transformed)) > 1e-3

    def test_the_identity_holds_per_arm_with_the_transformed_outcome(
        self, frame: pd.DataFrame
    ) -> None:
        weight = frame["wt"].to_numpy()
        w = _level(frame)
        for arm in (0, 1):
            indicator, target, g = self._arm_inputs(frame, arm)
            ref = s.fixed_point(indicator, target, g, weight)
            g_star = np.array([ref["g"][np.flatnonzero(w == point)[0]] for point in range(8)])
            qr_star = np.array([ref["qr"][np.flatnonzero(w == point)[0]] for point in range(8)])
            g0 = s.G if arm == 1 else 1.0 - s.G
            density = S_ONE if arm == 1 else 1.0 - S_ONE
            q0 = density * s.Q[:, arm]
            psi_0 = float(np.sum(s.P_W * q0))
            remainder = float(np.sum(s.P_W * (q0 - qr_star) * (g0 - g_star) / g_star))
            assert ref["psi"] - psi_0 == pytest.approx(remainder, abs=1e-9)

    def test_a_degenerate_stochastic_regime_equals_the_rule_fit_bit_for_bit(
        self, frame: pd.DataFrame
    ) -> None:
        table = D.astype(float)
        degenerate = _fit_regimes(frame, [_stochastic(table, name="rule")])
        rule = _fit_regimes(frame, [_rule()])
        name = "ey_regime[rule]"
        assert degenerate.estimates[name].psi == rule.estimates[name].psi
        assert degenerate.extra["adaptive_iptw"].epsilon == rule.extra["adaptive_iptw"].epsilon

    def test_a_user_intervention_with_a_zero_one_density_takes_the_deterministic_route(
        self, frame: pd.DataFrame
    ) -> None:
        class Mine:
            name = "rule"
            density_kind = "known"

            def density(self, data: Any) -> np.ndarray:
                one = D[_level(_covariate_frame(data))].astype(float)
                return np.column_stack([1.0 - one, one])

        log = s.SolveKeyLog()
        with log:
            result = _fit_regimes(frame, [Mine()])
        assert log.names == ["rule"]
        rule = _fit_regimes(frame, [_rule()])
        assert result.estimates["ey_regime[rule]"].psi == rule.estimates["ey_regime[rule]"].psi


# -------------------------------------------------------------------------------------- W15

#: The eighteen-point law of the natural course: ``W`` in 0..2, binary ``A``, three outcomes.
P_W3 = np.array([0.30, 0.40, 0.30])
G3 = np.array([0.40, 0.50, 0.70])
#: ``P(Delta = 1 | A, W)``, indexed ``[w, a]``.
PI = np.array([[0.80, 0.60], [0.50, 0.90], [0.70, 0.40]])
#: ``P(Y = 1 | Delta = 1, A, W)``, indexed ``[w, a]``.
Q3 = np.array([[0.30, 0.60], [0.50, 0.20], [0.70, 0.40]])
#: A fixed ``pi~``. Cells (0, 0) and (1, 1) tie at 0.7 across arms, and (0, 1) and (2, 0) tie at 0.5.
PI_TILDE = np.array([[0.70, 0.50], [0.60, 0.70], [0.50, 0.45]])


class ResponseTable(BaseEstimator, ClassifierMixin):
    """``P(Delta = 1 | A, W)`` from a fixed table, read from the design ``[A, W]``."""

    def __init__(self, table: Any = None) -> None:
        self.table = table

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> ResponseTable:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        design = np.asarray(x, dtype=float)
        one = np.asarray(self.table)[
            np.rint(design[:, 1]).astype(int), np.rint(design[:, 0]).astype(int)
        ]
        return np.column_stack([1.0 - one, one])


def _nc_frame(*, complete: bool = False) -> pd.DataFrame:
    rows = []
    for w, a in itertools.product(range(3), range(2)):
        mass = P_W3[w] * (G3[w] if a == 1 else 1.0 - G3[w])
        response = 1.0 if complete else PI[w, a]
        if not complete:
            rows.append(
                {
                    "W": float(w),
                    "A": float(a),
                    "Delta": 0.0,
                    "Y": np.nan,
                    "wt": mass * (1 - response),
                }
            )
        for y in (0, 1):
            share = Q3[w, a] if y == 1 else 1.0 - Q3[w, a]
            rows.append(
                {
                    "W": float(w),
                    "A": float(a),
                    "Delta": 1.0,
                    "Y": float(y),
                    "wt": mass * response * share,
                }
            )
    return pd.DataFrame(rows)


def _nc_fit(frame: pd.DataFrame, estimands: list[str], log: s.SolveKeyLog | None = None) -> Any:
    s.SpyClassifier.calls = []
    estimator = AdaptiveIPTW(
        treatment_learner=s.SpyClassifier(s.G_TILDE[:3]),
        missingness_learner=ResponseTable(PI_TILDE),
        reduced_outcome_learner=s.SaturatedRegressor(),
        estimands=estimands,
        cross_fit=False,
        g_bounds=(1e-3, 1 - 1e-3),
        tol=1e-12,
        max_iter=200,
        simultaneous=False,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if log is not None:
            with log:
                result = estimator.fit(
                    frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta", weights="wt"
                ).single()
        else:
            result = estimator.fit(
                frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta", weights="wt"
            ).single()
    return result


class TestW15NaturalCourse:
    @pytest.fixture(scope="class")
    def run(self) -> tuple[Any, s.SolveKeyLog]:
        log = s.SolveKeyLog()
        return _nc_fit(_nc_frame(), ["ey_obs"], log), log

    def _inputs(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        frame = _nc_frame()
        w = frame["W"].to_numpy(dtype=int)
        a = frame["A"].to_numpy(dtype=int)
        indicator = frame["Delta"].to_numpy()
        target = np.where(indicator == 1.0, frame["Y"].to_numpy(), 0.0)
        return frame, indicator, target, PI_TILDE[w, a]

    def test_the_law_has_eighteen_points_and_the_stated_ties(self) -> None:
        assert len(_nc_frame()) == 18
        assert PI_TILDE[0, 0] == PI_TILDE[1, 1] and PI[0, 0] != PI[1, 1]
        assert PI_TILDE[0, 1] == PI_TILDE[2, 0] and PI[0, 1] != PI[2, 0]
        assert Q3[0, 0] != Q3[1, 1] and Q3[0, 1] != Q3[2, 0]

    def test_the_round_zero_mechanism_is_pi_tilde_at_the_realized_arm(
        self, run: tuple[Any, s.SolveKeyLog]
    ) -> None:
        _, log = run
        _, _, _, g = self._inputs()
        assert np.max(np.abs(log.mechanisms["ey_obs"] - g)) < 1e-15

    def test_the_shipped_value_and_first_step_equal_the_test_side_loop(
        self, run: tuple[Any, s.SolveKeyLog]
    ) -> None:
        result, _ = run
        frame, indicator, target, g = self._inputs()
        ref = s.fixed_point(indicator, target, g, frame["wt"].to_numpy())
        assert result.estimates["ey_obs"].psi == pytest.approx(ref["psi"], abs=1e-10)
        assert result.extra["adaptive_iptw"].epsilon["ey_obs"][0] == pytest.approx(
            ref["eps"][0], abs=1e-10
        )
        assert abs(ref["eps"][0]) > 1e-3

    def test_the_fixed_point_identity_has_a_nonzero_remainder(self) -> None:
        frame, indicator, target, g = self._inputs()
        ref = s.fixed_point(indicator, target, g, frame["wt"].to_numpy())
        mass = P_W3[:, None] * np.column_stack([1.0 - G3, G3])
        g_star = np.empty((3, 2))
        qr_star = np.empty((3, 2))
        w = frame["W"].to_numpy(dtype=int)
        a = frame["A"].to_numpy(dtype=int)
        for cell_w, cell_a in itertools.product(range(3), range(2)):
            row = np.flatnonzero((w == cell_w) & (a == cell_a))[0]
            g_star[cell_w, cell_a] = ref["g"][row]
            qr_star[cell_w, cell_a] = ref["qr"][row]
        psi_0 = float(np.sum(mass * Q3))
        assert psi_0 == pytest.approx(0.494, abs=0.2)
        score = float(np.sum(mass * qr_star / g_star * (PI - g_star)))
        assert abs(score) < 1e-13
        remainder = float(np.sum(mass * (Q3 - qr_star) * (PI - g_star) / g_star))
        assert ref["psi"] - psi_0 == pytest.approx(remainder, abs=1e-9)
        assert abs(remainder) > 1e-6

    def test_the_scalar_fit_fits_no_treatment_mechanism(self) -> None:
        _nc_fit(_nc_frame(), ["ey_obs"])
        assert s.SpyClassifier.calls == []
        # The joint stack reads the arm mechanism, so the spy sees fits there.
        _nc_fit(_nc_frame(), ["ey_obs", "ey0", "par"])
        assert len(s.SpyClassifier.calls) > 0

    def test_the_joint_stack_keeps_the_key_and_the_curve_relation(self) -> None:
        scalar = _nc_fit(_nc_frame(), ["ey_obs"])
        joint = _nc_fit(_nc_frame(), ["ey_obs", "ey0", "par"])
        assert joint.estimates["ey_obs"].psi == scalar.estimates["ey_obs"].psi
        curves = joint.influence_curves
        assert (
            np.max(
                np.abs(
                    np.asarray(curves["par"])
                    - (np.asarray(curves["ey_obs"]) - np.asarray(curves["ey0"]))
                )
            )
            < 1e-12
        )
        covariance = float(np.cov(np.asarray(curves["ey_obs"]), np.asarray(curves["ey0"]))[0, 1])
        assert abs(covariance) > 1e-6

    def test_complete_data_use_the_closed_form(self) -> None:
        frame = _nc_frame(complete=True)
        result = _nc_fit(frame.assign(Delta=1.0), ["ey_obs"])
        y = frame["Y"].to_numpy()
        weight = frame["wt"].to_numpy()
        mean = float(np.average(y, weights=weight))
        assert result.estimates["ey_obs"].psi == pytest.approx(mean, abs=1e-12)
        assert result.extra["adaptive_iptw"].epsilon == {}


# -------------------------------------------------------------------------------------- W10


class TestW10RegimeAndNaturalCourseKeys:
    """Clusters and strata on a regime key, and strata on the natural-course key (Q18)."""

    def test_regime_clusters_the_standard_error_is_the_cluster_sum_of_the_curve(
        self, frame: pd.DataFrame
    ) -> None:
        sizes = np.array([1, 2, 3, 2, 4, 5, 3, 6, 2, 4])
        cluster = np.repeat(np.arange(len(sizes)), sizes)[: len(frame)]
        cluster = np.concatenate([cluster, np.full(len(frame) - len(cluster), len(sizes))])
        result = _fit_regimes(frame.assign(id=cluster), [_rule()], fit={"id": "id"})
        indicator = (frame["A"].to_numpy() == D[_level(frame)]).astype(float)
        g_d = np.where(
            D[_level(frame)] == 1, s.G_TILDE[_level(frame)], 1.0 - s.G_TILDE[_level(frame)]
        )
        ref = s.fixed_point(indicator, frame["Y"].to_numpy(), g_d, frame["wt"].to_numpy())
        curve = s.mean_one(frame["wt"].to_numpy()) * ref["curve"]
        sums = np.bincount(cluster, weights=curve)
        n = len(frame)
        variance = len(sums) * np.var(sums, ddof=1) / n**2
        estimate = result.estimates["ey_regime[rule]"]
        assert estimate.plugin_std_error == pytest.approx(float(np.sqrt(variance)), abs=1e-10)
        independent = float(np.sqrt(np.var(curve, ddof=1) / n))
        assert abs(independent - float(np.sqrt(variance))) > 1e-4

    def test_regime_strata_each_stratum_equals_a_fit_on_its_subset(
        self, frame: pd.DataFrame
    ) -> None:
        stratified = frame.assign(S=(frame["W"] >= 4).astype(float))
        result = _fit_regimes(stratified, [_rule()], covariates=("W", "S"), fit={"strata": ["S"]})
        first = []
        for code in (0.0, 1.0):
            subset = stratified[stratified["S"] == code].reset_index(drop=True)
            alone = _fit_regimes(subset, [_rule()], covariates=("W",))
            assert result.estimates[f"ey_regime[rule][S={code}]"].psi == pytest.approx(
                alone.estimates["ey_regime[rule]"].psi, abs=1e-9
            )
            first.append(alone.extra["adaptive_iptw"].epsilon["rule"][0])
        assert abs(first[0] - first[1]) > 1e-3

    def test_natural_course_strata_each_stratum_equals_a_fit_on_its_subset(self) -> None:
        # Stratum 0 holds the rows of W in {0, 1} and stratum 1 those of W in {1, 2}. Each holds
        # twelve rows and two values of W.
        law = _nc_frame()
        frame = pd.concat(
            [law[law["W"] <= 1].assign(S=0.0), law[law["W"] >= 1].assign(S=1.0)],
            ignore_index=True,
        )
        learner: dict[str, Any] = {
            "treatment_learner": s.SpyClassifier(s.G_TILDE[:3]),
            "missingness_learner": ResponseTable(PI_TILDE),
            "reduced_outcome_learner": s.SaturatedRegressor(),
            "estimands": ["ey_obs"],
            "cross_fit": False,
            "g_bounds": (1e-3, 1 - 1e-3),
            "tol": 1e-12,
            "max_iter": 200,
            "simultaneous": False,
            "random_state": 0,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stratified = (
                AdaptiveIPTW(**learner)
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W", "S"],
                    delta="Delta",
                    weights="wt",
                    strata=["S"],
                )
                .single()
            )
            first = []
            for code in (0.0, 1.0):
                subset = frame[frame["S"] == code].reset_index(drop=True)
                alone = (
                    AdaptiveIPTW(**learner)
                    .fit(
                        subset,
                        outcome="Y",
                        treatment="A",
                        covariates=["W"],
                        delta="Delta",
                        weights="wt",
                    )
                    .single()
                )
                assert stratified.estimates[f"ey_obs[S={code}]"].psi == pytest.approx(
                    alone.estimates["ey_obs"].psi, abs=1e-9
                )
                first.append(alone.extra["adaptive_iptw"].epsilon["ey_obs"][0])
        assert abs(first[0] - first[1]) > 1e-3
