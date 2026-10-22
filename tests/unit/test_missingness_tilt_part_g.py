"""The missingness tilt of the targets X82 part G added: ratios, natural course, regimes, MSMs.

Each curve repeats the arithmetic of the reported target on the tilted regressions, so the
tests check three things.  At ``gamma = 0`` each curve equals the reported estimate to
``rtol=1e-12``.  At ``gamma != 0`` each curve equals a hand computation from the fit's own
targeted regression and missingness mechanism, which is a nonzero witness because the
mechanism is below one on some rows.  And a mutation that tilts the observed rows too, or
that tilts no row, moves the curve away from the hand computation.
"""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.sensitivity.missingness as missingness_module
from cleverly import CapabilityError
from cleverly.datasets import make_missing_outcome, make_missing_outcome_binary
from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import tilt_missingness
from cleverly.exceptions import RefusalReason
from cleverly.interventions import Rule, Static
from cleverly.msm import MSM
from cleverly.sensitivity.missingness import missingness_tilt, tipping_gamma
from cleverly.utils.bounds import expit, logit

COVARIATES = ("W1", "W2", "W3")
GAMMA = 1.0


def _fit(*, binary: bool = True, cross_fit: bool = False, **settings: Any) -> Any:
    frame, _ = (
        make_missing_outcome_binary(n=600, seed=82)
        if binary
        else make_missing_outcome(n=600, seed=82)
    )
    outcome = LogisticRegression(max_iter=1000) if binary else LinearRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=outcome,
                treatment_learner=LogisticRegression(max_iter=1000),
                missingness_learner=LogisticRegression(max_iter=1000),
                cross_fit=cross_fit,
                simultaneous=False,
                random_state=0,
                **settings,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
            .single()
        )


def _rule() -> Rule:
    return Rule(lambda w: np.where(w["W1"] >= 0, 1, 0), name="rule", rule_kind="known")


@pytest.fixture(scope="module")
def ratios() -> Any:
    return _fit(estimands=("ey1", "ey0", "ate", "rr", "or"))


@pytest.fixture(scope="module")
def joint() -> Any:
    return _fit(estimands=("ey_obs", "ey0", "par", "paf"))


@pytest.fixture(scope="module")
def natural_course() -> Any:
    return _fit(estimands=("ey_obs",))


@pytest.fixture(scope="module")
def regime() -> Any:
    return _fit(
        interventions=[Static(0, name="ref"), _rule()], estimands=("ey_regime", "ate_regime")
    )


@pytest.fixture(scope="module")
def msm() -> Any:
    return _fit(msm=MSM.linear(), estimands=("msm",))


@pytest.fixture(scope="module")
def msm_modified() -> Any:
    return _fit(msm=MSM.linear(modifiers=("W1",)), estimands=("msm",))


def _rows(result: Any, gamma: tuple[float, ...] = (0.0, GAMMA), **kwargs: Any) -> Any:
    return nw.from_native(missingness_tilt(result, list(gamma), **kwargs), eager_only=True)


def _psi(result: Any, name: str, gamma: float, **kwargs: Any) -> float:
    frame = _rows(result, (gamma,), estimands=[name], **kwargs)
    return float(frame["psi"][0])


def _tilted_arm_means(result: Any, group: str, gamma: float) -> dict[float, float]:
    """Each arm's tilted mean, by hand, from the fit's own regression and mechanism."""
    repeat = result.repeats[0]
    pi = tilt_missingness(repeat.nuisance, result.config.missingness_bound)
    targeted = repeat.fluctuations[group].targeted
    out = {}
    for column, arm in enumerate(repeat.nuisance.arms):
        q = targeted.arms[arm]
        full = pi[:, column] * q + (1.0 - pi[:, column]) * expit(logit(q) + gamma)
        out[arm] = float(np.average(full, weights=result.data.weights))
    return out


def _tilted_natural_course(result: Any, gamma: dict[float, float]) -> float:
    """The tilted natural-course mean, by hand, with one tilt per received arm."""
    repeat = result.repeats[0]
    pi = tilt_missingness(repeat.nuisance, result.config.missingness_bound)
    m = repeat.fluctuations["natural_course"].targeted.observed
    a = np.asarray(result.data.treatment, dtype=float)
    total = np.zeros_like(m)
    for column, arm in enumerate(repeat.nuisance.arms):
        rows = a == arm
        p = pi[rows, column]
        total[rows] = p * m[rows] + (1.0 - p) * expit(logit(m[rows]) + gamma[arm])
    return float(np.average(total, weights=result.data.weights))


class TestTheCurvePassesThroughTheReportedEstimate:
    """At ``gamma = 0`` every new curve is the fit's own estimate."""

    @pytest.mark.parametrize("fixture", ["ratios", "joint", "natural_course", "regime", "msm"])
    def test_in_sample(self, fixture: str, request: pytest.FixtureRequest) -> None:
        result = request.getfixturevalue(fixture)
        frame = _rows(result, (0.0,))
        assert set(frame["estimand"]) == set(result.estimates)
        for name, psi in zip(frame["estimand"], frame["psi"], strict=True):
            assert psi == pytest.approx(result[name].psi, rel=1e-12, abs=0.0), name

    @pytest.mark.parametrize(
        "estimands", [("ey1", "ey0", "rr", "or"), ("ey_obs", "ey0", "par", "paf")]
    )
    def test_cross_fitted_with_repeats(self, estimands: tuple[str, ...]) -> None:
        """Three draws, combined by the median on each estimate's inference scale."""
        result = _fit(cross_fit=True, repeats=3, estimands=estimands)
        frame = _rows(result, (0.0,))
        for name, psi in zip(frame["estimand"], frame["psi"], strict=True):
            assert psi == pytest.approx(result[name].psi, rel=1e-12, abs=0.0), name

    @pytest.mark.parametrize(
        "settings",
        [
            # The natural-course mean needs a declared outcome support on a continuous Y.
            {"estimands": ("ey_obs",), "q_bounds": (-5.0, 8.0)},
            {"interventions": [Static(0, name="ref"), _rule()], "estimands": ("ey_regime",)},
            {"msm": MSM.linear(), "estimands": ("msm",)},
        ],
        ids=["natural_course", "regime", "msm"],
    )
    def test_a_gaussian_outcome_tilts_on_its_bounded_scale(self, settings: dict[str, Any]) -> None:
        """The tilt runs on the scaled regression and maps back to the outcome's units."""
        result = _fit(binary=False, **settings)
        assert not result.repeats[0].nuisance.scaler.is_identity
        frame = _rows(result)
        for gamma, name, psi in zip(frame["gamma"], frame["estimand"], frame["psi"], strict=True):
            if gamma == 0.0:
                assert psi == pytest.approx(result[name].psi, rel=1e-12, abs=0.0), name
            else:
                assert abs(psi - result[name].psi) > 1e-3, name


class TestTheCurveMatchesAHandComputation:
    """The nonzero witness: ``pi < 1`` on some rows, so the tilt moves each target."""

    def test_the_mechanism_is_below_one_somewhere(self, ratios: Any) -> None:
        pi = tilt_missingness(ratios.repeats[0].nuisance, ratios.config.missingness_bound)
        assert float(pi.min()) < 0.9

    def test_a_ratio_is_the_ratio_of_the_tilted_arm_means(self, ratios: Any) -> None:
        means = _tilted_arm_means(ratios, "mean", GAMMA)
        one, zero = means[1.0], means[0.0]
        assert _psi(ratios, "rr", GAMMA) == pytest.approx(one / zero, rel=1e-12)
        odds = (one / (1.0 - one)) / (zero / (1.0 - zero))
        assert _psi(ratios, "or", GAMMA) == pytest.approx(odds, rel=1e-12)
        # The arm rows, computed by the shipped path, agree with the hand means.
        assert _psi(ratios, "ey1", GAMMA) == pytest.approx(one, rel=1e-12)
        assert abs(_psi(ratios, "rr", GAMMA) - ratios["rr"].psi) > 0.01

    def test_the_natural_course_mean_tilts_the_received_arm(self, natural_course: Any) -> None:
        expected = _tilted_natural_course(natural_course, {0.0: GAMMA, 1.0: GAMMA})
        assert _psi(natural_course, "ey_obs", GAMMA) == pytest.approx(expected, rel=1e-12)
        assert abs(expected - natural_course["ey_obs"].psi) > 0.01

    def test_an_arm_direction_tilts_only_the_rows_that_received_that_arm(
        self, natural_course: Any
    ) -> None:
        """The received-arm mask: tilting arm 0 alone moves only the rows with A = 0."""
        direction = {0.0: 1.0, 1.0: 0.0}
        expected = _tilted_natural_course(natural_course, {0.0: GAMMA, 1.0: 0.0})
        psi = _psi(natural_course, "ey_obs", GAMMA, arm_gamma=direction)
        assert psi == pytest.approx(expected, rel=1e-12)
        # A mask on the counterfactual arm would tilt the A = 1 rows instead.
        swapped = _tilted_natural_course(natural_course, {0.0: 0.0, 1.0: GAMMA})
        assert abs(psi - swapped) > 1e-3

    def test_par_and_paf_read_the_tilted_natural_course_and_reference(self, joint: Any) -> None:
        observed = _tilted_natural_course(joint, {0.0: GAMMA, 1.0: GAMMA})
        reference = _tilted_arm_means(joint, "mean", GAMMA)[0.0]
        assert _psi(joint, "ey_obs", GAMMA) == pytest.approx(observed, rel=1e-12)
        assert _psi(joint, "par", GAMMA) == pytest.approx(observed - reference, rel=1e-12)
        assert _psi(joint, "paf", GAMMA) == pytest.approx(1.0 - reference / observed, rel=1e-12)

    def test_a_regime_mean_mixes_the_tilted_arms_by_the_regime(self, regime: Any) -> None:
        means = _tilted_arm_means(regime, "regime", GAMMA)
        # The static regime is arm 0 everywhere.
        assert _psi(regime, "ey_regime[ref]", GAMMA) == pytest.approx(means[0.0], rel=1e-12)
        repeat = regime.repeats[0]
        pi = tilt_missingness(repeat.nuisance, regime.config.missingness_bound)
        targeted = repeat.fluctuations["regime"].targeted
        density = repeat.nuisance.regimes.values[:, :, 1]
        # The rule treats some rows and not others, so the mixture reads both arms.
        assert 0.0 < float(density[:, 1].mean()) < 1.0
        full = np.column_stack(
            [
                pi[:, column] * targeted.arms[arm]
                + (1.0 - pi[:, column]) * expit(logit(targeted.arms[arm]) + GAMMA)
                for column, arm in enumerate(repeat.nuisance.arms)
            ]
        )
        expected = float(np.average((density * full).sum(axis=1), weights=regime.data.weights))
        assert _psi(regime, "ey_regime[rule]", GAMMA) == pytest.approx(expected, rel=1e-12)
        contrast = _psi(regime, "ate_regime[rule vs ref]", GAMMA)
        assert contrast == pytest.approx(expected - means[0.0], rel=1e-10)

    def test_a_saturated_msm_is_the_arm_mean_contrast(self, msm: Any) -> None:
        """With two arms, intercept and slope are the arm-0 mean and the contrast."""
        means = _tilted_arm_means(msm, "msm", GAMMA)
        assert _psi(msm, "msm[(intercept)]", GAMMA) == pytest.approx(means[0.0], rel=1e-10)
        assert _psi(msm, "msm[a]", GAMMA) == pytest.approx(means[1.0] - means[0.0], rel=1e-10)

    def test_an_effect_modified_msm_is_the_weighted_least_squares_projection(
        self, msm_modified: Any
    ) -> None:
        """A ``V``-dependent working model, against an independent stacked least squares."""
        repeat = msm_modified.repeats[0]
        model = repeat.nuisance.msm
        pi = tilt_missingness(repeat.nuisance, msm_modified.config.missingness_bound)
        targeted = repeat.fluctuations["msm"].targeted
        columns = [
            pi[:, column] * targeted.arms[arm]
            + (1.0 - pi[:, column]) * expit(logit(targeted.arms[arm]) + GAMMA)
            for column, arm in enumerate(targeted.levels)
        ]
        q = np.column_stack(columns)
        design = np.asarray(model.design, dtype=float)
        weight = np.asarray(model.weights, dtype=float) * msm_modified.data.weights[:, None]
        n, k, p = design.shape
        root = np.sqrt(weight.reshape(n * k))
        beta = np.linalg.lstsq(
            design.reshape(n * k, p) * root[:, None], q.reshape(n * k) * root, rcond=None
        )[0]
        frame = _rows(msm_modified, (GAMMA,))
        assert len(frame) == p
        for index, psi in enumerate(frame["psi"]):
            assert psi == pytest.approx(beta[index], rel=1e-8, abs=1e-10)


class TestEachMutationMovesTheCurve:
    """A mechanism that tilts every row, or none, disagrees with the hand computation."""

    @pytest.mark.parametrize(("fixture", "name"), [("joint", "ey_obs"), ("ratios", "rr")])
    def test_tilting_the_observed_rows_too_disagrees(
        self, fixture: str, name: str, request: pytest.FixtureRequest, monkeypatch: Any
    ) -> None:
        result = request.getfixturevalue(fixture)
        honest = _psi(result, name, GAMMA)
        monkeypatch.setattr(
            missingness_module,
            "tilt_missingness",
            lambda nuisance, bound: np.zeros_like(nuisance.missingness, dtype=float),
        )
        assert abs(_psi(result, name, GAMMA) - honest) > 1e-3

    @pytest.mark.parametrize(
        ("fixture", "name"), [("regime", "ey_regime[rule]"), ("msm", "msm[a]")]
    )
    def test_tilting_no_row_returns_the_mar_estimate(
        self, fixture: str, name: str, request: pytest.FixtureRequest, monkeypatch: Any
    ) -> None:
        result = request.getfixturevalue(fixture)
        assert abs(_psi(result, name, GAMMA) - result[name].psi) > 1e-3
        monkeypatch.setattr(
            missingness_module,
            "tilt_missingness",
            lambda nuisance, bound: np.ones_like(nuisance.missingness, dtype=float),
        )
        assert _psi(result, name, GAMMA) == pytest.approx(result[name].psi, rel=1e-10)


class TestTheSpreadClaimsNoIntervalOffMar:
    """At ``gamma != 0`` the new targets publish the ``plugin_`` columns only."""

    def test_the_columns_per_row(self, ratios: Any) -> None:
        frame = _rows(ratios)
        for row in frame.iter_rows(named=True):
            estimate = ratios[row["estimand"]]
            arm = row["estimand"] in {"ey1", "ey0", "ate"}
            if arm or row["gamma"] == 0.0:
                assert row["std_err"] == pytest.approx(estimate.std_error)
                assert np.isnan(row["plugin_std_err"])
            else:
                assert np.isnan(row["std_err"]) and np.isnan(row["ci_lower"])
                assert row["plugin_std_err"] == pytest.approx(estimate.plugin_std_error)
            if not arm and row["gamma"] == 0.0:
                assert (row["ci_lower"], row["ci_upper"]) == pytest.approx(estimate.ci)

    def test_a_ratio_interval_is_centred_on_the_log_of_the_tilted_point(self, ratios: Any) -> None:
        frame = _rows(ratios, (GAMMA,), estimands=["rr"])
        psi = float(frame["psi"][0])
        se = ratios["rr"].plugin_std_error
        z = 1.959963984540054
        low, high = (
            float(frame["plugin_interval_lower"][0]),
            float(frame["plugin_interval_upper"][0]),
        )
        assert low == pytest.approx(psi * np.exp(-z * se), rel=1e-6)
        assert high == pytest.approx(psi * np.exp(z * se), rel=1e-6)

    def test_an_arm_frame_keeps_its_columns(self, ratios: Any) -> None:
        frame = missingness_tilt(ratios, [0.0, GAMMA], estimands=["ey1", "ate"])
        assert "plugin_std_err" not in nw.from_native(frame, eager_only=True).columns


class TestTippingGamma:
    @pytest.mark.parametrize(
        ("fixture", "name"),
        [
            ("ratios", "rr"),
            ("natural_course", "ey_obs"),
            ("joint", "par"),
            ("regime", "ate_regime[rule vs ref]"),
            ("msm", "msm[a]"),
        ],
    )
    def test_the_interval_search_refuses_and_the_point_search_answers(
        self, fixture: str, name: str, request: pytest.FixtureRequest
    ) -> None:
        result = request.getfixturevalue(fixture)
        with pytest.raises(CapabilityError) as caught:
            tipping_gamma(result, name, use_ci=True)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert caught.value.refusal.roadmap == "X57"
        null = _psi(result, name, GAMMA)
        crossing = tipping_gamma(result, name, null_hypothesis=null)
        assert crossing is not None
        assert 0.0 < abs(crossing) <= GAMMA + 1e-3
        assert _psi(result, name, crossing) == pytest.approx(null, abs=1e-4)

    def test_an_arm_contrast_keeps_its_interval_search(self, ratios: Any) -> None:
        tipping_gamma(ratios, "ate", use_ci=True)
