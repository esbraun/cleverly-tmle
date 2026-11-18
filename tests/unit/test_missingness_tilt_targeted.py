"""The targeted missingness tilt and its own influence curves (CDR-001, A2).

Off ``gamma = 0`` the tilted parameter ``E_W[pi Q + (1 - pi) h(Q)]`` has its own influence
curve: the outcome residual is weighted by ``kappa`` and a missingness-mechanism term
appears.  These tests check the curves against an exact Gateaux derivative on a discrete
law, show that each of the two terms is live with a deliberate mutation, check that the
targeting solves the tilted estimating equation, that the targeted tilt reproduces the fit
at ``gamma = 0`` for every supported construction, and that the reviewer's law gets the
delta-method standard error.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.sensitivity._tilt_targeting as tilt_module
from cleverly import CapabilityError
from cleverly.datasets import make_missing_outcome_binary
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.inference.cluster import influence_variance
from cleverly.interventions import Rule, Static
from cleverly.sensitivity._tilt_targeting import (
    TiltSolve,
    arm_mean_curves,
    contrast_curves,
    natural_course_curve,
    residual_block,
    solve_tilt,
    targeted_tilt_estimates,
    tilt_interval_exclusion,
    tilted_regression,
)
from cleverly.sensitivity.missingness import _tilt_targets, missingness_tilt, tipping_gamma
from cleverly.utils.bounds import expit, logit

COVARIATES = ("W1", "W2", "W3")

# ----------------------------------------------------------------------- the discrete law

_W = (0, 1, 2)
_PW = np.array([0.3, 0.5, 0.2])
_G1 = np.array([0.3, 0.6, 0.45])
_PI = np.array([[0.55, 0.8, 0.45], [0.7, 0.5, 0.85]])
_Q = np.array([[0.2, 0.45, 0.7], [0.35, 0.6, 0.15]])
_SUPPORT = [
    (w, a, d, y) for w in _W for a in (0, 1) for d in (0, 1) for y in ((0, 1) if d else (0,))
]


def _law(pw: Any, g1: Any, pi: Any, q: Any) -> np.ndarray:
    """The probability of each support point, in :data:`_SUPPORT` order."""
    out = []
    for w, a, d, y in _SUPPORT:
        ga = g1[w] if a else 1.0 - g1[w]
        if not d:
            out.append(pw[w] * ga * (1.0 - pi[a, w]))
        else:
            out.append(pw[w] * ga * pi[a, w] * (q[a, w] if y else 1.0 - q[a, w]))
    return np.asarray(out)


def _margins(p: np.ndarray) -> tuple[np.ndarray, ...]:
    """The law's ``P(W)``, ``g(1 | W)``, ``pi_a(W)`` and ``Q_a(W)`` from point masses."""
    support = np.asarray(_SUPPORT)
    w, a, d, y = support.T
    pw = np.array([p[w == v].sum() for v in _W])
    g1 = np.array([p[(w == v) & (a == 1)].sum() / pw[v] for v in _W])
    pi = np.zeros((2, 3))
    q = np.zeros((2, 3))
    for arm in (0, 1):
        for v in _W:
            cell = (w == v) & (a == arm)
            pi[arm, v] = p[cell & (d == 1)].sum() / p[cell].sum()
            q[arm, v] = p[cell & (d == 1) & (y == 1)].sum() / p[cell & (d == 1)].sum()
    return pw, g1, pi, q


def _psi(p: np.ndarray, gamma: np.ndarray) -> dict[str, float]:
    """Each tilted parameter of the law at point masses ``p``."""
    pw, g1, pi, q = _margins(p)
    m = [pi[a] * q[a] + (1 - pi[a]) * expit(logit(q[a]) + gamma[a]) for a in (0, 1)]
    return {
        "ey1": float(pw @ m[1]),
        "ate": float(pw @ (m[1] - m[0])),
        "att": float((pw * g1) @ (m[1] - m[0]) / (pw @ g1)),
        "atc": float((pw * (1 - g1)) @ (m[1] - m[0]) / (pw @ (1 - g1))),
        "ey_obs": float(pw @ (g1 * m[1] + (1 - g1) * m[0])),
    }


def _curves(p: np.ndarray, gamma: np.ndarray) -> dict[str, np.ndarray]:
    """The library's tilted curves at each support point, one row per point.

    The rows are the support points weighted by their probability, so a weighted average is
    an expectation under the law and the weighted curve divided by the weight is ``D(o)``.
    """
    _, g1, pi, q = _margins(p)
    support = np.asarray(_SUPPORT)
    w, a, d, y = support.T
    weights = p * len(p)
    outcome = np.column_stack([q[0, w], q[1, w]])
    mechanism = np.column_stack([pi[0, w], pi[1, w]])
    propensity = np.column_stack([1.0 - g1[w], g1[w]])
    solve = TiltSolve(outcome, mechanism, True, 0, 0.0)
    observed = d.astype(bool)
    scaled = y.astype(float)
    _, means = arm_mean_curves(solve, scaled, observed, a, weights, propensity, gamma)
    _, conditional = contrast_curves(
        solve, scaled, observed, a, weights, propensity, gamma, [(1, 0, 1), (1, 0, 0)]
    )
    _, natural = natural_course_curve(solve, scaled, observed, a, weights, gamma)
    return {
        "ey1": means[:, 1] / weights,
        "ate": (means[:, 1] - means[:, 0]) / weights,
        "att": conditional[:, 0] / weights,
        "atc": conditional[:, 1] / weights,
        "ey_obs": natural / weights,
    }


def _gateaux_gap(gamma: np.ndarray) -> dict[str, float]:
    """The largest gap between each curve and the central difference of its parameter."""
    p = _law(_PW, _G1, _PI, _Q)
    curves = _curves(p, gamma)
    eps = 1e-6
    gaps = dict.fromkeys(curves, 0.0)
    for index in range(len(p)):
        point = np.zeros_like(p)
        point[index] = 1.0
        up = _psi((1 - eps) * p + eps * point, gamma)
        down = _psi((1 + eps) * p - eps * point, gamma)
        for name in curves:
            derivative = (up[name] - down[name]) / (2 * eps)
            gaps[name] = max(gaps[name], abs(derivative - curves[name][index]))
    return gaps


GAMMAS = [np.array([-1.5, -1.5]), np.array([0.7 * 0.5, 0.7]), np.array([2.0, -2.0])]


class TestTheCurvesAreGateauxDerivatives:
    @pytest.mark.parametrize("gamma", GAMMAS, ids=["-1.5", "0.7 by arm", "2 opposite"])
    def test_every_curve_matches_the_exact_derivative(self, gamma: np.ndarray) -> None:
        gaps = _gateaux_gap(gamma)
        assert max(gaps.values()) < 1e-6, gaps

    @pytest.mark.parametrize(
        "mutation",
        ["kappa_one", "no_mechanism_term"],
    )
    def test_each_term_is_live(self, mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deliberate mutations: kappa = 1, or no mechanism term, must fail the check."""
        original: Callable[..., np.ndarray] = residual_block

        def mutated(
            scaled: Any, observed: Any, outcome: Any, mechanism: Any, gamma: Any
        ) -> np.ndarray:
            h = expit(logit(outcome) + gamma)
            delta = np.asarray(observed, dtype=float)[:, None]
            y = np.where(observed, scaled, 0.0)[:, None]
            if mutation == "kappa_one":
                return np.asarray(
                    delta * (y - outcome) / mechanism + (outcome - h) * (delta - mechanism)
                )
            kept = original(scaled, observed, outcome, mechanism, gamma)
            return np.asarray(kept - (outcome - h) * (delta - mechanism))

        monkeypatch.setattr(tilt_module, "residual_block", mutated)
        gaps = _gateaux_gap(np.array([0.7, 2.0]))
        assert min(gaps.values()) > 1e-2, gaps

    def test_at_gamma_zero_the_block_is_the_mar_residual(self) -> None:
        p = _law(_PW, _G1, _PI, _Q)
        _, _, pi, q = _margins(p)
        support = np.asarray(_SUPPORT)
        w, _, d, y = support.T
        outcome = np.column_stack([q[0, w], q[1, w]])
        mechanism = np.column_stack([pi[0, w], pi[1, w]])
        block = residual_block(y.astype(float), d.astype(bool), outcome, mechanism, np.zeros(2))
        expected = d[:, None] * (y[:, None] - outcome) / mechanism
        assert block == pytest.approx(expected, abs=1e-12)
        assert tilted_regression(outcome, mechanism, np.zeros(2)) == pytest.approx(outcome)


# ----------------------------------------------------------------------- fitted results


def _fit(estimator: type = TMLE, **settings: Any) -> Any:
    frame, _ = make_missing_outcome_binary(n=600, seed=82)
    defaults: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "missingness_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ey1", "ey0", "ate", "att", "rr", "or"),
    }
    defaults.update(settings)
    columns = settings.pop("covariates", COVARIATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            estimator(**{k: v for k, v in defaults.items() if k != "covariates"})
            .fit(frame, outcome="Y", treatment="A", covariates=columns, delta="Delta")
            .single()
        )


CONSTRUCTIONS = {
    "in sample": {},
    "stacked CV-TMLE": {"cross_fit": True},
    "repeats": {"cross_fit": True, "repeats": 3},
    "fold-evaluated": {
        "cross_fit": True,
        "cv_evaluation": True,
        "estimands": ("ey1", "ey0", "ate", "att"),
    },
    "natural course": {"estimands": ("ey_obs", "ey0", "par", "paf")},
}


@pytest.fixture(scope="module", params=list(CONSTRUCTIONS))
def fitted(request: pytest.FixtureRequest) -> Any:
    return _fit(**CONSTRUCTIONS[request.param])


class TestGammaZeroIsTheFit:
    def test_the_targeted_tilt_at_zero_is_the_reported_estimate(self, fitted: Any) -> None:
        targets = list(_tilt_targets(fitted).values())
        direction = dict.fromkeys(fitted.data.arm_codes, 1.0)
        for repeat in fitted.repeats:
            estimates, converged = targeted_tilt_estimates(fitted, repeat, targets, 0.0, direction)
            assert converged
            for name, estimate in estimates.items():
                if len(fitted.repeats) > 1:
                    # Each repeat's row is that repeat's own point estimate.
                    assert estimate.psi == pytest.approx(repeat.psi[name], rel=1e-9, abs=1e-12)
                    continue
                assert estimate.psi == pytest.approx(fitted[name].psi, rel=1e-9, abs=1e-12)
                assert estimate.std_error == pytest.approx(fitted[name].std_error, rel=1e-6)

    def test_the_curve_is_continuous_at_zero(self, fitted: Any) -> None:
        frame = missingness_tilt(fitted, [-1e-7, 0.0, 1e-7]).to_frame()
        for name in frame["estimand"].unique():
            rows = frame[frame["estimand"] == name]
            assert np.ptp(rows["psi"]) < 1e-5 * max(1.0, abs(fitted[name].psi))
            assert np.ptp(rows["std_err"]) < 1e-4 * fitted[name].std_error

    def test_off_zero_every_target_carries_an_interval(self, fitted: Any) -> None:
        frame = missingness_tilt(fitted, [1.0]).to_frame()
        assert frame["std_err"].notna().all() and frame["ci_lower"].notna().all()
        assert "plugin_std_err" not in frame.columns


def _curves_for(
    kind: str,
    solve: TiltSolve,
    scaled: np.ndarray,
    observed: np.ndarray,
    received: np.ndarray,
    weights: np.ndarray,
    propensity: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """The weighted curves of one target family: arm means, att and atc, or ey_obs."""
    if kind == "arm means":
        return arm_mean_curves(solve, scaled, observed, received, weights, propensity, gamma)[1]
    if kind == "att and atc":
        pairs = [(1, 0, 1), (1, 0, 0)]
        return contrast_curves(
            solve, scaled, observed, received, weights, propensity, gamma, pairs
        )[1]
    return natural_course_curve(solve, scaled, observed, received, weights, gamma)[1][:, None]


def _columns_for(kind: str, received: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    if kind == "arm means":
        return tilt_module.arm_columns(received, propensity)
    if kind == "att and atc":
        return tilt_module.contrast_columns(propensity, [(1, 0, 1), (1, 0, 0)])
    return tilt_module.natural_course_columns(len(received), propensity.shape[1])


class TestTheTargetingSolvesTheTiltedEquation:
    @pytest.mark.parametrize("kind", ["arm means", "att and atc", "ey_obs"])
    @pytest.mark.parametrize("covariates", [COVARIATES, ("W1",)], ids=["flexible", "wrong Q"])
    def test_the_score_is_solved(self, covariates: tuple[str, ...], kind: str) -> None:
        result = _fit(covariates=covariates)
        repeat = result.repeats[0]
        data = result.data
        arms = tuple(repeat.nuisance.arms)
        received = np.array([arms.index(float(a)) for a in data.treatment])
        observed = np.asarray(data.observed, dtype=bool)
        scaled = np.where(observed, np.nan_to_num(np.asarray(data.outcome, dtype=float)), 0.0)
        weights = np.ones(data.n)
        mechanism = repeat.nuisance.bounded_missingness(result.config.missingness_bound)
        propensity = repeat.nuisance.bounded_propensity(result.config.g_bounds)
        targeted = repeat.fluctuations["mean"].targeted
        outcome = np.column_stack([targeted.arms[arm] for arm in arms])
        gamma = np.array([1.5, -0.5])
        start = _curves_for(
            kind,
            TiltSolve(outcome, mechanism, True, 0, 0.0),
            scaled,
            observed,
            received,
            weights,
            propensity,
            gamma,
        )
        solved = solve_tilt(
            scaled,
            observed,
            received,
            weights,
            outcome,
            mechanism,
            gamma,
            _columns_for(kind, received, propensity),
        )
        assert solved.converged
        curves = _curves_for(kind, solved, scaled, observed, received, weights, propensity, gamma)
        # The plug-in terms average to zero by construction, so the mean of the curve is the
        # mean of its residual block: the tilted estimating equation.
        sd = np.sqrt(np.var(curves, axis=0))
        assert np.all(np.abs(curves.mean(axis=0)) < 1e-8 * sd)
        # Nonzero witness: the untargeted start misses it by at least a hundred times the
        # solved tolerance.  The natural course's start is the closest, near 1.6e-5 SD.
        assert np.max(np.abs(start.mean(axis=0)) / sd) > 1e-6


class TestTheReviewersLaw:
    """One arm, no prognostic covariate, ``q = 0.1``, ``pi = 0.3``, 5,000 units per arm."""

    @pytest.fixture(scope="class")
    def fit(self) -> Any:
        n_arm = 5000
        frame = pd.DataFrame(
            {
                "A": np.repeat([0, 1], n_arm),
                "Delta": np.tile(np.r_[np.ones(1500), np.zeros(3500)], 2),
                "Y": np.tile(np.r_[np.ones(150), np.zeros(1350), np.full(3500, np.nan)], 2),
                "W": np.tile([0, 1], n_arm),
            }
        )

        def learner() -> LogisticRegression:
            return LogisticRegression(C=np.inf, tol=1e-12, max_iter=2000)

        return (
            TMLE(
                outcome_learner=learner(),
                treatment_learner=learner(),
                missingness_learner=learner(),
                cross_fit=False,
                estimands=("ey1",),
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")
            .single()
        )

    def test_the_standard_error_is_the_delta_method(self, fit: Any) -> None:
        row = missingness_tilt(fit, [2.0]).to_frame().iloc[0]
        h = float(expit(logit(np.asarray(0.1)) + 2.0))
        assert row["psi"] == pytest.approx(0.3 * 0.1 + 0.7 * h, rel=1e-9)
        assert row["std_err"] == pytest.approx(0.017389, rel=1e-4)
        assert row["std_err"] / fit["ey1"].std_error > 2.2

    def test_the_interval_tipping_search_follows_the_tilted_limit(self, fit: Any) -> None:
        tip = tipping_gamma(fit, "ey1", null_hypothesis=0.3, use_ci=True)
        assert tip is not None
        row = missingness_tilt(fit, [tip]).to_frame().iloc[0]
        assert row["ci_upper"] == pytest.approx(0.3, abs=1e-4)


class TestWhatKeepsTheDiagnostic:
    def test_a_regime_mean_keeps_the_diagnostic(self) -> None:
        rule = Rule(lambda w: np.where(w["W1"] >= 0, 1, 0), name="rule", rule_kind="known")
        result = _fit(interventions=[rule, Static(0, name="ref")], estimands=("ey_regime",))
        frame = missingness_tilt(result, [1.0]).to_frame()
        assert frame["std_err"].isna().all() if "std_err" in frame else True
        assert frame["plugin_std_err"].notna().all()
        name = frame["estimand"].iloc[0]
        with pytest.raises(CapabilityError, match="regime mean or an MSM coefficient"):
            tipping_gamma(result, name, use_ci=True)

    @pytest.mark.parametrize("estimator", [DRTMLE, CTMLE], ids=["DRTMLE", "CTMLE"])
    def test_an_excluded_fit_keeps_the_diagnostic(self, estimator: type) -> None:
        settings: dict[str, Any] = {"estimands": ("ey1", "ey0", "ate")}
        if estimator is CTMLE:
            settings["strategy"] = "oat"
        else:
            settings["reduced_outcome_learner"] = LinearRegression()
            settings["reduced_mechanism_residual_learner"] = LinearRegression()
        result = _fit(estimator, **settings)
        reason = tilt_interval_exclusion(result)
        assert reason is not None
        frame = missingness_tilt(result, [1.0]).to_frame()
        assert frame["plugin_std_err"].notna().all()
        with pytest.raises(CapabilityError, match="claims no interval at gamma != 0"):
            tipping_gamma(result, "ate", use_ci=True)
        # Control: the ordinary fit on the same data publishes the interval.
        ordinary = _fit(estimands=("ey1", "ey0", "ate"))
        assert tilt_interval_exclusion(ordinary) is None
        assert missingness_tilt(ordinary, [1.0]).to_frame()["std_err"].notna().all()


def test_a_clustered_fit_uses_the_cluster_variance() -> None:
    frame, _ = make_missing_outcome_binary(n=600, seed=82)
    frame = frame.assign(site=np.arange(len(frame)) // 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                missingness_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
                estimands=("ey1", "ey0", "ate"),
            )
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta", id="site")
            .single()
        )
    targets = [_tilt_targets(result)["ate"]]
    estimates, _ = targeted_tilt_estimates(
        result, result.repeats[0], targets, 1.0, dict.fromkeys(result.data.arm_codes, 1.0)
    )
    ate = estimates["ate"]
    clustered = influence_variance(ate.influence_curve, result.data.cluster)
    assert ate.variance == pytest.approx(clustered)
    # Nonzero witness: the clustered variance is not the independent one.
    assert abs(clustered - influence_variance(ate.influence_curve, None)) > 1e-3 * clustered


def test_a_weighted_fold_evaluated_fit_starts_from_its_own_update() -> None:
    # The fold-evaluated update normalises weight mass inside each validation fold.  Solving
    # the tilt with the raw weights would move the fluctuation at gamma = 0, and the tilted
    # estimate would jump away from the reported one.
    frame, _ = make_missing_outcome_binary(n=600, seed=83)
    rng = np.random.default_rng(83)
    frame = frame.assign(sw=rng.uniform(0.2, 3.0, size=len(frame)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                missingness_learner=LogisticRegression(max_iter=1000),
                cross_fit=True,
                cv_evaluation=True,
                simultaneous=False,
                random_state=0,
                estimands=("ey1", "ey0", "ate"),
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=COVARIATES,
                delta="Delta",
                weights="sw",
            )
            .single()
        )
    targets = list(_tilt_targets(result).values())
    direction = dict.fromkeys(result.data.arm_codes, 1.0)
    estimates, converged = targeted_tilt_estimates(
        result, result.repeats[0], targets, 0.0, direction
    )
    assert converged
    for name, estimate in estimates.items():
        assert estimate.psi == pytest.approx(result[name].psi, rel=1e-9, abs=1e-12)
        assert estimate.std_error == pytest.approx(result[name].std_error, rel=1e-6)


class TestTheReadingNamesEveryComposition:
    """The registry rule names each setting that changes the targeted tilt's estimator."""

    @staticmethod
    def _rules(result: Any, monkeypatch: pytest.MonkeyPatch) -> set[str]:
        import cleverly.validation._interpretation as interpretation

        seen: set[str] = set()
        original = interpretation.resolve

        def recording(instrument: str, rule: str, fit: Any) -> Any:
            seen.add(rule)
            return original(instrument, rule, fit)

        monkeypatch.setattr(interpretation, "resolve", recording)
        missingness_tilt(result, [1.0])
        return seen

    def test_the_default_fit_reads_its_bases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rules = self._rules(_fit(), monkeypatch)
        assert {"level", "contrast", "conditional", "ratio"} <= rules

    @pytest.mark.parametrize(
        ("settings", "feature"),
        [
            ({"stopping": "scaled_var"}, "targeting_variant"),
            ({"targeting": "one_step"}, "targeting_variant"),
        ],
        ids=["stopping", "one step"],
    )
    def test_a_variant_update_rule_is_named(
        self, settings: dict[str, Any], feature: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rules = self._rules(_fit(**settings), monkeypatch)
        assert rules and all(rule.endswith(f"+{feature}") for rule in rules), rules

    def test_baseline_strata_are_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, _ = make_missing_outcome_binary(n=600, seed=82)
        frame["S"] = (frame["W1"] > 0).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                TMLE(
                    outcome_learner=LogisticRegression(max_iter=1000),
                    treatment_learner=LogisticRegression(max_iter=1000),
                    missingness_learner=LogisticRegression(max_iter=1000),
                    cross_fit=False,
                    simultaneous=False,
                    random_state=0,
                    estimands=("ey1", "ey0", "ate"),
                )
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=(*COVARIATES, "S"),
                    delta="Delta",
                    strata=["S"],
                )
                .single()
            )
        rules = self._rules(result, monkeypatch)
        assert rules and all("+stratified" in rule for rule in rules), rules

    def test_off_zero_every_target_keeps_the_fits_interval_scale(self, fitted: Any) -> None:
        """The tilted estimate forms its interval on the fit's scale (range-logit, MOVER).

        The control is the interval at gamma near zero, which must reproduce the fit's.
        """
        from cleverly.sensitivity.missingness import _targeted_tilt

        targets = list(_tilt_targets(fitted).values())
        direction = dict.fromkeys(fitted.data.arm_codes, 1.0)
        # The draws combine by the fit's own rule, as the reported estimates do.
        near = _targeted_tilt(fitted, targets, 1e-9, direction)
        far = _targeted_tilt(fitted, targets, 1.0, direction)
        for name, estimate in far.items():
            assert estimate.interval_scale == fitted[name].interval_scale, name
            assert near[name].ci == pytest.approx(fitted[name].ci, rel=1e-5, abs=1e-8)
        # A nonzero witness: a binary arm mean off zero is not symmetric about its estimate.
        level = next(e for e in far.values() if e.interval_scale == "range_logit")
        low, high = level.ci
        assert abs((high - level.psi) - (level.psi - low)) > 1e-6


def test_an_interval_tipping_value_carries_the_reading() -> None:
    from cleverly.assessment import INTERPRETERS
    from cleverly.sensitivity.missingness import tipping_interval_reading

    result = _fit(estimands=("ey1", "ey0", "ate"))
    reading = tipping_interval_reading(result, "ate")
    assert reading is not None
    item = INTERPRETERS["tipping_gamma"](1.5, result, {"estimand": "ate", "use_ci": True})
    assert ("descriptive" in item.detail) is (not reading.licensed)
    # Control: the point search follows no interval, so it carries no reading.
    point = INTERPRETERS["tipping_gamma"](1.5, result, {"estimand": "ate"})
    assert "descriptive" not in point.detail


def test_an_update_below_the_response_bound_warns() -> None:
    from cleverly.exceptions import DataWarning

    tight = _fit(estimands=("ey1", "ey0", "ate"), nuisance_bound=0.45)
    with pytest.warns(DataWarning, match=r"below the fit's missingness_bound 0\.45"):
        missingness_tilt(tight, [2.0])
    # Control: at the default bound the update stays above it, and nothing warns.
    default = _fit(estimands=("ey1", "ey0", "ate"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataWarning)
        missingness_tilt(default, [2.0])


@pytest.mark.parametrize("cross_fit", [False, True], ids=["in sample", "stacked"])
def test_a_scaled_outcome_reads_every_target_on_the_fits_own_scale(cross_fit: bool) -> None:
    """On a continuous outcome the scaler is not the identity, so a scale slip would show.

    ``rr``, ``or`` and ``paf`` exist only on a binary outcome, whose scaler is the identity;
    ``par`` and the arm means are the targets a non-identity scaler reaches.
    """
    from cleverly.datasets import make_missing_outcome

    frame, _ = make_missing_outcome(n=800, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                missingness_learner=LogisticRegression(max_iter=1000),
                cross_fit=cross_fit,
                simultaneous=False,
                estimands=("ey_obs", "ey1", "ey0", "ate", "par"),
                q_bounds=(-6.0, 10.0),
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
            .single()
        )
    assert not result.repeats[0].nuisance.scaler.is_identity
    targets = list(_tilt_targets(result).values())
    direction = dict.fromkeys(result.data.arm_codes, 1.0)
    estimates, converged = targeted_tilt_estimates(
        result, result.repeats[0], targets, 0.0, direction
    )
    assert converged and set(estimates) == {"ey_obs", "ey1", "ey0", "ate", "par"}
    for name, estimate in estimates.items():
        assert estimate.psi == pytest.approx(result[name].psi, rel=1e-9, abs=1e-12), name
        assert estimate.std_error == pytest.approx(result[name].std_error, rel=1e-6), name


def test_ratios_and_the_attributable_fraction_need_a_binary_outcome() -> None:
    from cleverly.datasets import make_missing_outcome

    frame, _ = make_missing_outcome(n=200, seed=3)
    for estimands in (("ey1", "ey0", "rr"), ("ey1", "ey0", "or"), ("ey_obs", "ey0", "paf")):
        with pytest.raises(CapabilityError, match="binary outcome"):
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                missingness_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                estimands=estimands,
                q_bounds=(-6.0, 10.0),
            ).fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
