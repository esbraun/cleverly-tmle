"""The exact sampling SD of :mod:`tests.long_horizon_sampling_sd` against ``cleverly`` and itself.

The module re-implements the single-fold ``LTMLE``, its plug-in and ``SDR`` in numpy.  Its
estimate and reported curve must equal ``cleverly``'s on the weighted exact frame for every
learner rung and for a wrong mechanism.  Its influence function must be the derivative of the
estimate: moving a little mass to one support point moves the estimate by that point's value.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests import discrete_law_longitudinal_horizon as H
from tests import long_horizon_sampling_sd as S
from tests.studies import long_horizon_common as C

RUNGS = {"O1": "glm", "O2": "wblind", "O3": "int", "O4": "lblind"}
CASES = [
    (K, m, plan, rung)
    for K, m in ((3, 3), (4, 2))
    for plan in (("respond", "mix", "low") if m == 3 else ("respond", "coin", "never"))
    for rung in RUNGS
]


def _fit(frame, K, m, plan, rung, estimator):  # type: ignore[no-untyped-def]
    from sklearn.linear_model import LinearRegression

    outcome = C.outcome_rung(rung, plan, K, m)
    extra = {"pseudo_learner": LinearRegression()} if estimator == "SDR" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return C.fit(
            frame, K, m, (plan,), estimator=estimator, weights="w", outcome_learner=outcome, **extra
        )


@pytest.mark.parametrize(("K", "m", "plan", "rung"), CASES)
@pytest.mark.parametrize("mechanism", ["known", "constant"])
def test_ltmle_equals_cleverly_on_the_frame(
    K: int, m: int, plan: str, rung: str, mechanism: str
) -> None:
    frame = H.weighted_frame(K, m)
    if mechanism != "known":
        frame = C.wrong_mechanism(frame, K, m, mechanism)
    result = _fit(frame, K, m, plan, rung, "LTMLE")
    engine = S.Sequential(frame, K, m, plan, learners=RUNGS[rung]).fit()
    name = f"ey_regimen[{plan}]"
    assert engine.psi == pytest.approx(float(result[name].psi), abs=1e-11)
    curve = C.per_unit_curve(result, name, frame["w"].to_numpy())
    np.testing.assert_allclose(engine.curve, curve, rtol=0, atol=1e-9)


@pytest.mark.parametrize(("K", "m", "plan", "rung"), CASES)
def test_sdr_equals_cleverly_on_the_frame(K: int, m: int, plan: str, rung: str) -> None:
    """``SDR`` fits the rung at the horizon and least squares on the GLM's columns before it."""
    frame = C.wrong_mechanism(H.weighted_frame(K, m), K, m, "constant")
    result = _fit(frame, K, m, plan, rung, "SDR")
    learners = dict.fromkeys(range(1, K), "glm") | {K: RUNGS[rung]}
    engine = S.Sequential(frame, K, m, plan, kind="sdr", learners=learners).fit()
    name = f"ey_regimen[{plan}]"
    assert engine.psi == pytest.approx(float(result[name].psi), abs=1e-12)
    curve = C.per_unit_curve(result, name, frame["w"].to_numpy())
    np.testing.assert_allclose(engine.curve, curve, rtol=0, atol=1e-10)


@pytest.mark.parametrize("plan", ["respond", "mix"])
def test_the_plugin_equals_the_untargeted_regression(plan: str) -> None:
    frame = H.weighted_frame(3, 3)
    with warnings.catch_warnings():
        # The plan density reads the missing covariates of rows that never reached the node.
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = C.untargeted_plugin(
            frame, plan, 3, 3, C.InterceptOnly(), weights=frame["w"].to_numpy()
        )
    engine = S.Sequential(frame, 3, 3, plan, kind="plugin", learners="int").fit()
    assert engine.psi == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    ("kind", "learners", "mechanism"),
    [
        ("ltmle", "glm", "known"),
        ("ltmle", "glm", "constant"),
        ("plugin", "wblind", "known"),
        ("sdr", "int", "constant"),
    ],
)
def test_the_influence_function_is_the_derivative_of_the_estimate(
    kind: str, learners: str, mechanism: str
) -> None:
    """``(psi((1 - e) P + e delta_j) - psi(P)) / e`` equals ``IF_j`` at three support points.

    The wrong mechanism is a fixed function of the history, so its columns do not move with
    the mass.  A curve that dropped the outcome-fit terms would miss this by the factor the
    section 5.3 Monte Carlo table records.
    """
    K, m, plan = 3, 2, "respond"
    frame = H.weighted_frame(K, m)
    if mechanism != "known":
        frame = C.wrong_mechanism(frame, K, m, mechanism)
    base = S.Sequential(frame, K, m, plan, kind=kind, learners=learners).fit()
    influence = base.influence()
    assert base.mean_function < 1e-12
    p = frame["w"].to_numpy() / frame["w"].sum()
    eps = 1e-6
    rows = np.argsort(-np.abs(influence))[:3]
    for j in rows:
        moved = frame.copy()
        mass = (1 - eps) * p
        mass[j] += eps
        moved["w"] = mass
        shifted = S.Sequential(moved, K, m, plan, kind=kind, learners=learners).fit().psi
        assert (shifted - base.psi) / eps == pytest.approx(influence[j], rel=1e-4)


def test_the_sampling_sd_reads_the_monte_carlo_table() -> None:
    """The three configurations of the section 5.3 Monte Carlo table, per unit."""
    K, m = 5, 2
    frame = H.weighted_frame(K, m)
    wrong = C.wrong_mechanism(frame, K, m, "constant")
    c1 = S.reading(wrong, K, m, "respond", "never")
    assert c1.bias == pytest.approx(0.0517, abs=5e-5)
    assert c1.sigma / np.sqrt(10_000) == pytest.approx(0.0314, abs=5e-5)
    assert c1.sigma_curve / np.sqrt(10_000) == pytest.approx(0.0887, abs=5e-5)
    coin = S.reading(frame, K, m, "coin", kind="plugin", learners="int")
    assert coin.bias == pytest.approx(-0.0448, abs=5e-5)
    assert coin.sigma / np.sqrt(4_000) == pytest.approx(0.0096, abs=5e-5)
    small = S.reading(H.weighted_frame(3, 2), 3, 2, "respond", kind="plugin", learners="wblind")
    assert small.bias == pytest.approx(-0.0184, abs=5e-5)
    assert small.sigma / np.sqrt(8_000) == pytest.approx(0.0109, abs=5e-5)
    assert small.sigma_curve / np.sqrt(8_000) == pytest.approx(0.0133, abs=5e-5)


def test_a_learner_or_estimator_it_does_not_know_is_refused() -> None:
    frame = H.weighted_frame(3, 2)
    with pytest.raises(ValueError, match="kind must be one of"):
        S.Sequential(frame, 3, 2, "respond", kind="tmle")
    with pytest.raises(ValueError, match="unknown learners"):
        S.Sequential(frame, 3, 2, "respond", learners="forest")
