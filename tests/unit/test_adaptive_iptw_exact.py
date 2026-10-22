r"""Exact-law checks of the AdaptiveIPTW estimator on arm means (plan 3.1 to 3.3).

Every check compares the shipped result with a value the test computes itself, in numpy, from
the law tables and a fixed mechanism ``G_TILDE``. The test-side targeting loop in
:mod:`tests.unit._adaptive_iptw_support` shares no code with the package. An identity that holds
at any solved score counts as a test and not as a witness: it cannot see which regression or
which mechanism produced the score. The witnesses are the comparisons of the shipped ``psi``,
the first tilt step and the curve with the test-side loop, and the counts of spy calls.

* W1, W1b, W2: the population fixed point, a finite sample under a GLM regression, the score.
* W3: the sign of the score term of the curve, against a closed form.
* W4: the one-step identity and the one-step value.
* W5: each arm tilts its own mechanism.
* W7: a known mechanism.
* W8: the bound, W10: weights, clusters and strata, W11: a constant design.
* W12: the range rule. W13: ``treatment_covariates``.
* Pins: the per-key solver equals ``solve_armwise_bounded_mechanism`` bit for bit, and the
  scalar and joint natural-course keys agree bit for bit.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import AdaptiveIPTW
from cleverly.estimators import iptw as iptw_module
from cleverly.exceptions import DataError, DataWarning
from tests.unit import _adaptive_iptw_support as s

TOL = 1e-10


def _record(result: Any) -> Any:
    return result.extra["adaptive_iptw"]


def _population_regression(arm: int, g_star: np.ndarray) -> np.ndarray:
    """``E(Y | A = arm, g*(W))`` per support point, from the law tables."""
    mass = s.P_W * (s.G if arm == 1 else 1.0 - s.G)
    value = mass * s.Q[:, arm]
    out = np.empty(8)
    keys = np.round(g_star, 12)
    for level in np.unique(keys):
        group = keys == level
        out[group] = value[group].sum() / mass[group].sum()
    return out


def _per_point(rows: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """A per-row vector that depends on ``W`` only, as one value per support point."""
    index = frame["W"].to_numpy(dtype=int)
    return np.array([rows[np.flatnonzero(index == point)[0]] for point in range(8)])


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return s.law_frame()


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame) -> Any:
    return s.fit_arm(frame)


@pytest.fixture(scope="module")
def reference(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    weight = frame["wt"].to_numpy()
    inputs = s.arm_inputs(frame)
    return {
        arm: s.fixed_point(item["indicator"], item["target"], item["g"], weight)
        for arm, item in inputs.items()
    }


# --------------------------------------------------------------------------------------- W1


class TestW1PopulationFixedPoint:
    """The shipped loop lands on the test's own fixed point, and the identity holds there."""

    @pytest.mark.parametrize(("arm", "name"), [(1, "ey1"), (0, "ey0")])
    def test_the_shipped_estimate_and_first_step_equal_the_test_side_loop(
        self, fitted: Any, reference: dict, arm: int, name: str
    ) -> None:
        ref = reference[arm]
        assert fitted.estimates[name].psi == pytest.approx(ref["psi"], abs=TOL)
        steps = _record(fitted).epsilon[str(arm)]
        assert steps[0] == pytest.approx(ref["eps"][0], abs=TOL)
        # The nonzero witness: a first step of zero would hide the refit.
        assert abs(ref["eps"][0]) > 1e-3
        assert len(steps) >= 4

    @pytest.mark.parametrize(("arm", "name"), [(1, "ey1"), (0, "ey0")])
    def test_the_fixed_point_identity_has_a_nonzero_remainder(
        self, frame: pd.DataFrame, fitted: Any, reference: dict, arm: int, name: str
    ) -> None:
        """``psi(g*) - psi_0 = P_0[(Q_0 - Qr*)(g_0 - g*)/g*]`` at a zero population score."""
        ref = reference[arm]
        g_star = _per_point(ref["g"], frame)
        qr_star = _per_point(ref["qr"], frame)
        population = _population_regression(arm, g_star)
        assert np.max(np.abs(population - qr_star)) < 1e-12
        g0 = s.G if arm == 1 else 1.0 - s.G
        psi_0 = float(np.sum(s.P_W * s.Q[:, arm]))
        score = float(np.sum(s.P_W * qr_star / g_star * (g0 - g_star)))
        assert abs(score) < 1e-13
        remainder = float(np.sum(s.P_W * (s.Q[:, arm] - qr_star) * (g0 - g_star) / g_star))
        assert fitted.estimates[name].psi - psi_0 == pytest.approx(remainder, abs=1e-9)
        assert abs(remainder) > 1e-6
        # The tied pairs make the regression on g* differ from the one on W.
        assert np.max(np.abs(s.Q[:, arm] - qr_star)) > 1e-2


class TestW1bFiniteSampleUnderAGLM:
    """A GLM regression changes with the mechanism, so a refit is not a no-op (M5, M9)."""

    @staticmethod
    def _draw(n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(20261021)
        w1 = rng.uniform(-2.0, 2.0, n)
        w2 = rng.binomial(1, 0.5, n).astype(float)
        g = expit(-w1 + 2.0 * w1 * w2)
        a = rng.binomial(1, g).astype(float)
        y = rng.binomial(1, expit(0.2 * a - w1 + 2.0 * w1 * w2)).astype(float)
        return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "g1": g})

    @classmethod
    def _fit(cls, frame: pd.DataFrame, form: str = "targeted") -> Any:
        from cleverly.estimators import AdaptiveIPTW

        g1 = frame["g1"].to_numpy()
        return (
            AdaptiveIPTW(
                reduced_outcome_learner=s.LogitGLM(),
                estimands=["ey1"],
                cross_fit=False,
                g_bounds=(1e-3, 1 - 1e-3),
                tol=1e-12,
                max_iter=200,
                simultaneous=False,
                random_state=0,
                form=form,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                treatment_probabilities=np.column_stack([1 - g1, g1]),
            )
            .single()
        )

    def test_the_one_step_equals_the_test_side_value_at_the_initial_mechanism(self) -> None:
        """W4 (b) on a non-saturated regression, where the one-step differs from the targeted
        value: ``mean(I Y / g) - mean(Qr (I - g) / g)`` with ``Qr`` fitted at the initial ``g``.
        """
        frame = self._draw()
        g1 = frame["g1"].to_numpy()
        indicator = (frame["A"].to_numpy() == 1).astype(float)
        y = frame["Y"].to_numpy()
        ones = np.ones(len(frame))
        qr = s.design_logit_regression(g1, y, indicator > 0, ones)
        expected = float(np.mean(indicator * y / g1) - np.mean(qr * (indicator - g1) / g1))
        naive = float(np.mean(indicator * y / g1))
        one_step = self._fit(frame, form="one_step")
        targeted = self._fit(frame)
        assert _record(one_step).one_step["1"][0] == pytest.approx(expected, abs=1e-7)
        assert one_step.estimates["ey1"].psi == pytest.approx(expected, abs=1e-7)
        assert _record(targeted).one_step["1"][0] == pytest.approx(expected, abs=1e-7)
        # The nonzero witnesses: the test-side targeted value at g* and the naive value are
        # both far from the one-step value, so a one-step left at g* or at the naive mean fails.
        at_target = s.fixed_point(
            indicator, y, g1, ones, regress=s.design_logit_regression, tolerance=1e-13
        )["psi"]
        assert abs(expected - at_target) > 1e-4  # measured gap 1.4e-3
        assert abs(expected - naive) > 1e-3
        # The curve of the one-step form has mean zero on the sample.
        curve = np.asarray(one_step.influence_curves["ey1"], dtype=float)
        assert abs(np.mean(curve)) < 1e-12

    def test_the_shipped_fit_equals_the_test_side_irls_loop(self) -> None:
        frame = self._draw()
        g1 = frame["g1"].to_numpy()
        result = self._fit(frame)
        weight = np.ones(len(frame))
        indicator = (frame["A"].to_numpy() == 1).astype(float)
        target = frame["Y"].to_numpy()
        refit = s.fixed_point(
            indicator, target, g1, weight, regress=s.design_logit_regression, tolerance=1e-13
        )
        no_refit = s.fixed_point(
            indicator,
            target,
            g1,
            weight,
            regress=s.design_logit_regression,
            refit=False,
            tolerance=1e-13,
        )
        assert result.estimates["ey1"].psi == pytest.approx(refit["psi"], abs=1e-7)
        # The nonzero witness: dropping the refit moves the value.
        assert abs(refit["psi"] - no_refit["psi"]) > 1e-5
        # The one-step value is a different number from the targeted one under a GLM.
        record = _record(result)
        assert abs(record.one_step["1"][0] - result.estimates["ey1"].psi) > 1e-6


# --------------------------------------------------------------------------------------- W2


class TestW2ScoreSolved:
    @pytest.mark.parametrize("arm", [1, 0])
    def test_the_score_is_solved_and_was_not_at_round_zero(
        self, frame: pd.DataFrame, fitted: Any, arm: int
    ) -> None:
        weight = s.mean_one(frame["wt"].to_numpy())
        item = s.arm_inputs(frame)[arm]
        qr0 = s.saturated_regression(item["g"], item["target"], item["indicator"] > 0, weight)
        initial = float(np.mean(weight * qr0 / item["g"] * (item["indicator"] - item["g"])))
        assert abs(initial) > 1e-3
        record = _record(fitted)
        assert abs(record.score[str(arm)]) < 1e-9
        assert record.stop == "tolerance"


# --------------------------------------------------------------------------------------- W3


def _law_p_rows() -> list[tuple[np.ndarray, np.ndarray]]:
    """Quadrature nodes of law P: ``(mass, P(A = 1 | W))`` for each value of ``W2``."""
    nodes, weights = np.polynomial.legendre.leggauss(2000)
    w1 = 2.0 * nodes
    return [(weights / 2.0 * 0.5, expit(-w1 + 2.0 * w1 * w2)) for w2 in (0.0, 1.0)]


def _law_p_moments(arm: int, sign: float) -> tuple[float, float]:
    """The mean and SD of ``I Y / g + sign * Q (I - g) / g - psi`` under law P, by quadrature."""
    rows = _law_p_rows()

    def regression(g1: np.ndarray) -> np.ndarray:
        return expit(0.2 * arm + logit(g1))

    psi = sum(float(np.sum(mass * regression(g1))) for mass, g1 in rows)
    first = second = 0.0
    for mass, g1 in rows:
        g = g1 if arm == 1 else 1.0 - g1
        q = regression(g1)
        for a in (0, 1):
            p_a = g1 if a == 1 else 1.0 - g1
            q_a = expit(0.2 * a + logit(g1))
            for y in (0, 1):
                p_y = q_a if y == 1 else 1.0 - q_a
                indicator = 1.0 if a == arm else 0.0
                curve = indicator * y / g + sign * q * (indicator - g) / g - psi
                first += float(np.sum(mass * p_a * p_y * curve))
                second += float(np.sum(mass * p_a * p_y * curve**2))
    return first, float(np.sqrt(second))


def _law_p_draw(n: int = 200_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-2.0, 2.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    g = expit(-w1 + 2.0 * w1 * w2)
    a = rng.binomial(1, g).astype(float)
    y = rng.binomial(1, expit(0.2 * a + logit(g))).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y, "g1": g})


@pytest.fixture(scope="module")
def law_p() -> pd.DataFrame:
    return _law_p_draw()


def _fit_law_p(frame: pd.DataFrame, arm: int, **settings: Any) -> Any:
    """Oracle mechanism and the exact regression on the mechanism, one arm."""
    from cleverly.estimators import AdaptiveIPTW

    g1 = frame["g1"].to_numpy()
    fn = (lambda g: expit(0.2 + logit(g))) if arm == 1 else (lambda g: 1.0 - g)
    base: dict[str, Any] = {
        "reduced_outcome_learner": s.FunctionRegressor(fn),
        "estimands": [f"ey{arm}"],
        "cross_fit": False,
        "g_bounds": (1e-4, 1 - 1e-4),
        "tol": 1e-12,
        "simultaneous": False,
        "random_state": 0,
    }
    base.update(settings)
    return (
        AdaptiveIPTW(**base)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            treatment_probabilities=np.column_stack([1 - g1, g1]),
        )
        .single()
    )


class TestW3CurveSign:
    @pytest.mark.parametrize("arm", [1, 0])
    def test_the_curve_sd_is_the_closed_form_and_not_the_plus_sign_form(
        self, law_p: pd.DataFrame, arm: int
    ) -> None:
        result = _fit_law_p(law_p, arm)
        curve = np.asarray(result.influence_curves[f"ey{arm}"], dtype=float)
        sd = float(np.std(curve))
        _, correct = _law_p_moments(arm, -1.0)
        _, plus = _law_p_moments(arm, +1.0)
        assert sd == pytest.approx(correct, abs=1e-2)
        # The nonzero witness: the two curves differ by 2 D_g, whose SD is above 0.5.
        assert abs(plus - correct) > 0.4
        assert abs(sd - plus) > 0.4

    def test_the_closed_form_curves_have_mean_zero(self) -> None:
        for arm in (1, 0):
            for sign in (-1.0, 1.0):
                assert abs(_law_p_moments(arm, sign)[0]) < 1e-12


# --------------------------------------------------------------------------------------- W4


class TestW4OneStep:
    @pytest.fixture(scope="class")
    def one_step_fit(self, frame: pd.DataFrame) -> Any:
        return s.fit_arm(frame, form="one_step")

    @pytest.mark.parametrize(("arm", "name"), [(1, "ey1"), (0, "ey0")])
    def test_the_population_one_step_identity(self, arm: int, name: str) -> None:
        """``P_0 [I Y / g~ - D_g] - psi_0`` equals ``P_0[(Q_0 - Qr~)(g_0 - g~)/g~]``."""
        g_tilde = s.G_TILDE if arm == 1 else 1.0 - s.G_TILDE
        g0 = s.G if arm == 1 else 1.0 - s.G
        qr = _population_regression(arm, g_tilde)
        psi_0 = float(np.sum(s.P_W * s.Q[:, arm]))
        population = float(
            np.sum(s.P_W * (g0 * s.Q[:, arm] / g_tilde - qr * (g0 - g_tilde) / g_tilde))
        )
        remainder = float(np.sum(s.P_W * (s.Q[:, arm] - qr) * (g0 - g_tilde) / g_tilde))
        assert population - psi_0 == pytest.approx(remainder, abs=1e-12)
        assert abs(remainder) > 1e-4

    @pytest.mark.parametrize(("arm", "name"), [(1, "ey1"), (0, "ey0")])
    def test_the_shipped_one_step_equals_the_test_side_value_at_the_initial_nuisances(
        self, frame: pd.DataFrame, one_step_fit: Any, reference: dict, arm: int, name: str
    ) -> None:
        weight = frame["wt"].to_numpy()
        item = s.arm_inputs(frame)[arm]
        qr = s.saturated_regression(item["g"], item["target"], item["indicator"] > 0, weight)
        naive = np.sum(weight * item["indicator"] * item["target"] / item["g"]) / weight.sum()
        d_g = np.sum(weight * qr / item["g"] * (item["indicator"] - item["g"])) / weight.sum()
        expected = float(naive - d_g)
        record = _record(one_step_fit)
        assert record.one_step[str(arm)][0] == pytest.approx(expected, abs=TOL)
        assert one_step_fit.estimates[name].psi == pytest.approx(expected, abs=TOL)
        assert record.untargeted[str(arm)] == pytest.approx(float(naive), abs=TOL)
        # A saturated regression makes the one-step value equal the targeted value: both are
        # the stratified mean of Qr over the classes of g. The GLM fit of W1b separates them.
        assert expected == pytest.approx(reference[arm]["psi"], abs=1e-9)
        # The naive value is a different number.
        assert abs(expected - float(naive)) > 1e-3
        # The one-step curve has mean zero on the sample.
        curve = np.asarray(one_step_fit.influence_curves[name], dtype=float)
        assert abs(np.mean(curve)) < 1e-12

    def test_the_targeted_fit_records_the_same_one_step_value(
        self, fitted: Any, one_step_fit: Any
    ) -> None:
        for name in ("1", "0"):
            assert _record(fitted).one_step[name][0] == pytest.approx(
                _record(one_step_fit).one_step[name][0], abs=TOL
            )
            assert _record(one_step_fit).targeted[name][0] == pytest.approx(
                _record(fitted).targeted[name][0], abs=TOL
            )


# --------------------------------------------------------------------------------------- W5


class TestW5EachArmTiltsItsOwnMechanism:
    def test_the_two_targeted_mechanisms_do_not_sum_to_one(
        self, fitted: Any, reference: dict, frame: pd.DataFrame
    ) -> None:
        total = reference[1]["g"] + reference[0]["g"] - 1.0
        assert np.max(np.abs(total)) > 1e-6
        # The shipped arm means are the test-side means of the two separate tilts.
        assert fitted.estimates["ey1"].psi == pytest.approx(reference[1]["psi"], abs=TOL)
        assert fitted.estimates["ey0"].psi == pytest.approx(reference[0]["psi"], abs=TOL)
        # The shipped curve of arm 0 is the curve built on the arm-0 tilt, not on 1 - g1*.
        curve = np.asarray(fitted.influence_curves["ey0"], dtype=float)
        weight = s.mean_one(frame["wt"].to_numpy())
        assert np.max(np.abs(curve - weight * reference[0]["curve"])) < 1e-9


# --------------------------------------------------------------------------------------- W7


class TestW7KnownMechanism:
    def test_a_known_mechanism_moves_and_has_the_smaller_variance(
        self, law_p: pd.DataFrame
    ) -> None:
        result = _fit_law_p(law_p, 1)
        steps = _record(result).epsilon["1"]
        assert abs(steps[0]) > 1e-4
        n = len(law_p)
        targeted = result.estimates["ey1"].std_error * np.sqrt(n)
        _, closed = _law_p_moments(1, -1.0)
        assert targeted == pytest.approx(closed, abs=1e-2)
        # The known-g IPTW curve I Y / g - psi, by quadrature: the targeted SE is below it.
        rows = _law_p_rows()
        psi = sum(float(np.sum(m * expit(0.2 + logit(g1)))) for m, g1 in rows)
        iptw = np.sqrt(
            sum(float(np.sum(m * expit(0.2 + logit(g1)) / g1)) for m, g1 in rows) - psi**2
        )
        assert iptw == pytest.approx(0.9, abs=2e-2)
        assert targeted < iptw - 0.1
        assert result.inference_status == "influence_curve"


# --------------------------------------------------------------------------------------- W8


class TestW8Bound:
    BOUNDS = (0.3, 0.7)

    @pytest.fixture(scope="class")
    def bounded(self, frame: pd.DataFrame) -> Any:
        return s.fit_arm(frame, g_bounds=self.BOUNDS)

    @pytest.mark.parametrize(("arm", "name"), [(1, "ey1"), (0, "ey0")])
    def test_the_clip_applies_at_every_step(
        self, frame: pd.DataFrame, bounded: Any, arm: int, name: str
    ) -> None:
        weight = frame["wt"].to_numpy()
        item = s.arm_inputs(frame)[arm]
        clipped = s.fixed_point(
            item["indicator"], item["target"], item["g"], weight, bounds=self.BOUNDS
        )
        free = s.fixed_point(item["indicator"], item["target"], item["g"], weight)
        assert bounded.estimates[name].psi == pytest.approx(clipped["psi"], abs=TOL)
        # The nonzero witness: the clip moves the value.
        assert abs(clipped["psi"] - free["psi"]) > 1e-3
        lower, upper = self.BOUNDS
        active = int(np.count_nonzero((clipped["g"] <= lower) | (clipped["g"] >= upper)))
        if arm == 0:
            assert active > 0
        assert _record(bounded).bound_active[str(arm)] == active


# --------------------------------------------------------------------------------------- W10


class TestW10WeightsClustersStrata:
    def test_weights_the_fixed_point_is_on_the_weight_tilted_support(
        self, frame: pd.DataFrame
    ) -> None:
        tilt = np.array([1.0, 2.0, 0.5, 3.0, 1.0, 0.25, 2.0, 1.0])[frame["W"].to_numpy(dtype=int)]
        tilted = frame.assign(wt=frame["wt"] * tilt)
        result = s.fit_arm(tilted)
        weight = tilted["wt"].to_numpy()
        unweighted = s.fixed_point(
            s.arm_inputs(frame)[1]["indicator"],
            frame["Y"].to_numpy(),
            s.arm_inputs(frame)[1]["g"],
            frame["wt"].to_numpy(),
        )
        for arm, name in ((1, "ey1"), (0, "ey0")):
            item = s.arm_inputs(tilted)[arm]
            ref = s.fixed_point(item["indicator"], item["target"], item["g"], weight)
            assert result.estimates[name].psi == pytest.approx(ref["psi"], abs=TOL)
            assert _record(result).epsilon[str(arm)][0] == pytest.approx(ref["eps"][0], abs=TOL)
        assert abs(result.estimates["ey1"].psi - unweighted["psi"]) > 1e-3

    def test_clusters_the_standard_error_is_the_cluster_sum_of_the_test_side_curve(
        self, frame: pd.DataFrame
    ) -> None:
        sizes = np.array([1, 2, 3, 2, 4, 5, 3, 6, 2, 4])
        cluster = np.repeat(np.arange(len(sizes)), sizes)[: len(frame)]
        cluster = np.concatenate([cluster, np.full(len(frame) - len(cluster), len(sizes))])
        clustered = frame.assign(id=cluster)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = s.fit_arm(clustered, fit={"id": "id"})
        weight = s.mean_one(frame["wt"].to_numpy())
        for arm, name in ((1, "ey1"), (0, "ey0")):
            item = s.arm_inputs(frame)[arm]
            ref = s.fixed_point(
                item["indicator"], item["target"], item["g"], frame["wt"].to_numpy()
            )
            curve = weight * ref["curve"]
            sums = np.bincount(cluster, weights=curve)
            n = len(frame)
            variance = len(sums) * np.var(sums, ddof=1) / n**2
            assert result.estimates[name].plugin_std_error == pytest.approx(
                float(np.sqrt(variance)), abs=1e-10
            )
            independent = float(np.sqrt(np.var(curve, ddof=1) / n))
            assert abs(independent - float(np.sqrt(variance))) > 1e-4

    def test_strata_each_stratum_equals_a_fit_on_its_subset(self, frame: pd.DataFrame) -> None:
        stratified = frame.assign(S=(frame["W"] >= 4).astype(float))
        result = s.fit_arm(
            stratified, covariates=("W", "S"), fit={"strata": ["S"]}, estimands=("ey1", "ey0")
        )
        record = _record(result)
        eps_first = []
        for code in (0.0, 1.0):
            subset = stratified[stratified["S"] == code].reset_index(drop=True)
            alone = s.fit_arm(subset, covariates=("W",), estimands=("ey1", "ey0"))
            for arm, name in ((1, "ey1"), (0, "ey0")):
                label = f"{name}[S={code}]"
                assert result.estimates[label].psi == pytest.approx(
                    alone.estimates[name].psi, abs=1e-9
                )
                steps = record.epsilon[f"{arm} | S={code}"]
                assert steps[0] == pytest.approx(_record(alone).epsilon[str(arm)][0], abs=1e-9)
            eps_first.append(_record(alone).epsilon["1"][0])
        # The nonzero witness: the strata tilt by different amounts.
        assert abs(eps_first[0] - eps_first[1]) > 1e-3


# --------------------------------------------------------------------------------------- W11


class TestW11ConstantDesign:
    def test_a_constant_known_mechanism_fits_no_reduced_regression(self) -> None:
        from cleverly.estimators import AdaptiveIPTW

        rng = np.random.default_rng(3)
        n = 400
        frame = pd.DataFrame({"W": rng.normal(size=n), "A": rng.binomial(1, 0.5, n).astype(float)})
        frame["Y"] = rng.binomial(1, expit(0.3 * frame["A"] + 0.2 * frame["W"])).astype(float)
        constant = np.full(n, 0.5)
        s.SpyRegressor.calls = []
        result = (
            AdaptiveIPTW(
                reduced_outcome_learner=s.SpyRegressor(),
                estimands=["ey1", "ey0"],
                cross_fit=False,
                simultaneous=False,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_probabilities=np.column_stack([1 - constant, constant]),
            )
            .single()
        )
        assert s.SpyRegressor.calls == []
        record = _record(result)
        assert set(record.constant_design) == {"0", "1"}
        # With a constant mechanism the tilt moves g to the treated fraction. The value is the
        # arm mean, and it differs from the Horvitz-Thompson mean at the known g = 0.5.
        for arm, name in ((1, "ey1"), (0, "ey0")):
            indicator = (frame["A"].to_numpy() == arm).astype(float)
            level = float(frame["Y"][frame["A"] == arm].mean())
            ht = float(np.mean(indicator * frame["Y"].to_numpy() / 0.5))
            assert result.estimates[name].psi == pytest.approx(level, abs=1e-9)
            assert abs(level - ht) > 1e-3

    def test_a_varying_design_calls_the_learner(self, frame: pd.DataFrame) -> None:
        s.SpyRegressor.calls = []
        s.fit_arm(frame, regressor=s.SpyRegressor())
        assert len(s.SpyRegressor.calls) > 0


# --------------------------------------------------------------------------------------- W12


class TestW12RangeRule:
    """A ratio target is refused when a Horvitz-Thompson mean leaves its range.

    The plan builds a sample with ``psi_1 > 1``. A search over constructed samples found no
    converged fit with ``psi > 1`` beyond the tolerance: at the fixed point the Horvitz-Thompson
    mean of a binary outcome stays in ``[0, 1]`` up to rounding. It is not found at
    convergence, and the claim goes no further. The lower side uses an arm that has no event,
    so ``psi_0 = 0`` is not positive and not in ``(0, 1)``. The upper side sets the means by
    hand, so a rounding of ``1 - 1e-12`` cannot decide the test.
    """

    @staticmethod
    def _sample() -> pd.DataFrame:
        rng = np.random.default_rng(105)
        n = 120
        w = rng.normal(size=n)
        a = (rng.random(n) < expit(0.8 * w)).astype(float)
        y = np.where(a == 1.0, (rng.random(n) < 0.6).astype(float), 0.0)
        return pd.DataFrame({"W": w, "A": a, "Y": y})

    def _fit(self, estimands: tuple[str, ...]) -> Any:
        return s.fit_arm(
            self._sample(),
            weights=None,
            estimands=estimands,
            g_bounds=(0.001, 0.999),
            treatment_learner=LogisticRegression(C=100),
            regressor=LogisticRegression(),
        )

    def test_a_ratio_is_refused_after_targeting_and_ate_alone_fits(self) -> None:
        arms = self._fit(("ey1", "ey0"))
        # The nonzero witness: arm 1 has events, so the refusal comes from arm 0 alone.
        assert arms.estimates["ey1"].psi > 0.1
        assert arms.estimates["ey0"].psi == 0.0
        with pytest.raises(DataError, match="not positive"):
            self._fit(("rr",))
        with pytest.raises(DataError, match=r"outside \(0, 1\)"):
            self._fit(("or",))
        result = self._fit(("ate",))
        assert "ate" in result.estimates
        assert result.estimates["ate"].psi == pytest.approx(arms.estimates["ey1"].psi, abs=1e-12)

    def test_a_default_or_all_report_drops_the_undefined_targets_with_a_warning(self) -> None:
        """The plan raises for ``"all"``. The code drops, because neither list names a target.

        An explicit list that names ``rr``, ``or`` or ``paf`` raises (the test above).
        """
        for estimands, dropped in ((None, ["or", "rr"]), ("all", ["or", "paf", "rr"])):
            with pytest.warns(DataWarning, match="dropped") as caught:
                result = self._fit(estimands)
            message = " ".join(str(w.message) for w in caught if "dropped" in str(w.message))
            assert str(dropped) in message
            assert not set(dropped) & set(result.estimates)
            # The kept estimates are the arm means and ate, equal to the explicit fit.
            kept = self._fit(("ey1", "ey0", "ate"))
            for name in ("ey1", "ey0", "ate"):
                assert result.estimates[name].psi == pytest.approx(
                    kept.estimates[name].psi, abs=1e-12
                )

    def test_a_retarget_that_names_an_undefined_target_raises_on_a_default_estimator(self) -> None:
        """A name in a ``retarget`` call is a request, even when the constructor used the default."""
        sample = self._sample()
        estimator = AdaptiveIPTW(
            estimands=None,
            treatment_learner=LogisticRegression(C=100),
            reduced_outcome_learner=LogisticRegression(),
            cross_fit=False,
            g_bounds=(0.001, 0.999),
            tol=1e-12,
            max_iter=200,
            simultaneous=False,
            random_state=0,
        )
        with pytest.warns(DataWarning):
            fitted = estimator.fit(sample, outcome="Y", treatment="A", covariates=["W"]).single()
        with pytest.raises(DataError, match="not positive"):
            estimator.retarget(fitted.data, fitted.nuisance, estimands=["rr"])

    @pytest.mark.parametrize(
        ("name", "value", "text"), [("or", 1.2, "outside"), ("rr", -0.1, "not")]
    )
    def test_a_mean_above_one_or_below_zero_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: float, text: str
    ) -> None:
        """The upper side of the rule, with the mean set by the test (a converged fit stays in
        ``[0, 1]`` up to the tolerance, so a sample cannot reach it)."""
        fitted = self._fit(("ey1", "ey0"))
        estimator = AdaptiveIPTW(estimands=[name])
        means = {0.0: (value, np.zeros(3)), 1.0: (0.4, np.zeros(3))}
        solved = [
            SimpleNamespace(spec=SimpleNamespace(owner=0.0, owner_name="0")),
            SimpleNamespace(spec=SimpleNamespace(owner=1.0, owner_name="1")),
        ]
        monkeypatch.setattr(iptw_module, "_owner_values", lambda *args, **kwargs: means)
        with pytest.raises(DataError, match=text):
            estimator._apply_range_rule(fitted.data, (name,), solved, None)
        means[0.0] = (0.5, np.zeros(3))
        assert estimator._apply_range_rule(fitted.data, (name,), solved, None) == (name,)


# --------------------------------------------------------------------------------------- W13


class TestW13TreatmentAdjustment:
    def test_the_mechanism_reads_the_subset_and_the_summary_prints_the_pinned_line(self) -> None:
        from sklearn.linear_model import LogisticRegression

        from cleverly.data.treatment_covariates import adaptive_iptw_restriction_line
        from cleverly.estimators import AdaptiveIPTW

        rng = np.random.default_rng(8)
        n = 600
        frame = pd.DataFrame({"W1": rng.normal(size=n), "W2": rng.normal(size=n)})
        frame["A"] = rng.binomial(1, expit(0.8 * frame["W1"])).astype(float)
        frame["Y"] = frame["A"] + frame["W1"] + rng.normal(size=n)
        spy: list[int] = []

        class Spy(LogisticRegression):
            def fit(self, x: Any, y: Any, sample_weight: Any = None) -> Any:
                spy.append(np.asarray(x).shape[1])
                return super().fit(x, y, sample_weight=sample_weight)

        result = (
            AdaptiveIPTW(
                treatment_learner=Spy(),
                reduced_outcome_learner=__import__(
                    "sklearn.linear_model", fromlist=["x"]
                ).LinearRegression(),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                treatment_covariates=["W1"],
            )
            .single()
        )
        assert set(spy) == {1}
        summary = result.summary()
        line = adaptive_iptw_restriction_line(["W1"], ["W2"])
        assert line in summary
        assert "consistent when the outcome regression is consistent" not in summary
        assert line.startswith("treatment-mechanism covariates: the treatment mechanism")


# --------------------------------------------------------------------------------------- pins


class TestSolverPins:
    def test_the_per_key_solver_equals_the_armwise_solver_bit_for_bit(
        self, frame: pd.DataFrame
    ) -> None:
        """Complete-data arm keys: ``solve_bounded_mechanism`` per key is the armwise solver."""
        from cleverly.fluctuation.mechanism import (
            solve_armwise_bounded_mechanism,
            solve_bounded_mechanism,
        )

        weight = frame["wt"].to_numpy()
        inputs = s.arm_inputs(frame)
        bounds = (0.05, 0.95)
        indicators = np.column_stack([inputs[0]["indicator"], inputs[1]["indicator"]])
        mechanism = np.column_stack([inputs[0]["g"], inputs[1]["g"]])
        qr = np.column_stack(
            [
                s.saturated_regression(
                    inputs[arm]["g"], inputs[arm]["target"], inputs[arm]["indicator"] > 0, weight
                )
                for arm in (0, 1)
            ]
        )
        h = qr / mechanism
        per_key = [
            solve_bounded_mechanism(
                indicators[:, k], mechanism[:, k], h[:, [k]], weight, bounds=bounds
            )
            for k in range(2)
        ]
        joint = solve_armwise_bounded_mechanism(
            frame["A"].to_numpy(dtype=float), mechanism, h, weight, (0.0, 1.0), bounds=bounds
        )
        # The strided covariate view that the armwise wrapper passes, for the layout control.
        strided = [
            solve_bounded_mechanism(
                indicators[:, k], mechanism[:, k], h[:, k:2:2], weight, bounds=bounds
            )
            for k in range(2)
        ]
        for k in range(2):
            assert np.array_equal(
                np.asarray(per_key[k].propensity), np.asarray(joint.propensity)[:, k]
            )
            # The wrapper hands the solver a strided column. A contiguous column changes the
            # summation order of one dot product, so epsilon differs by at most a few ulp.
            assert np.array_equal(
                np.asarray(strided[k].epsilon).ravel(), np.asarray(joint.epsilon).ravel()[[k]]
            )
            assert float(per_key[k].epsilon.ravel()[0]) == pytest.approx(
                float(np.asarray(joint.epsilon).ravel()[k]), rel=1e-14
            )
            assert abs(float(per_key[k].epsilon.ravel()[0])) > 1e-2

    def test_the_natural_course_key_is_bit_identical_scalar_and_joint(self) -> None:
        from tests.unit import _capability_sweep_support as sweep

        frame = sweep._missing_frame()
        kwargs: dict[str, Any] = {
            "outcome_learner": None,
            "treatment_learner": LogisticRegression(max_iter=1000),
            "missingness_learner": LogisticRegression(max_iter=1000),
            "reduced_outcome_learner": LinearRegression(),
            "cross_fit": False,
            "simultaneous": False,
            "random_state": 0,
        }
        from cleverly.estimators import AdaptiveIPTW

        scalar = (
            AdaptiveIPTW(estimands=["ey_obs"], **kwargs)
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        joint = (
            AdaptiveIPTW(estimands=["ey_obs", "ey0", "par"], **kwargs)
            .fit(frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        assert scalar.estimates["ey_obs"].psi == joint.estimates["ey_obs"].psi
        assert np.array_equal(
            np.asarray(scalar.influence_curves["ey_obs"]),
            np.asarray(joint.influence_curves["ey_obs"]),
        )
        assert joint.estimates["ey_obs"].psi != joint.estimates["ey0"].psi
